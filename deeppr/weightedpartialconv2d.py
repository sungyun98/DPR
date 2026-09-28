# Oiginal partial Convolution from https://github.com/NVIDIA/partialconv
# Guinier-Porod model from https://doi.org/10.1107/S0021889810015773

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedPartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False
            
        if 'object_size' in kwargs:
            self.object_size = kwargs['object_size']
            kwargs.pop('object_size')
        else:
            self.object_size = 64
        
        # Guinier-Porod model
        if 'weight_model' in kwargs:
            self.trg_weight_model = kwargs['weight_model']
            kwargs.pop('weight_model')
        else:
            self.trg_weight_model = True

        super(WeightedPartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        
        self.last_size = (None, None, None, None)
        self.weight_model = None
        self.slide_weight = None
        self.update_mask = None
        self.mask_ratio = None
        
    @staticmethod
    def get_weight_model(i_max, j_max, sigma):
        i = torch.linspace(0, i_max - 1, steps=i_max) - i_max // 2
        j = torch.linspace(0, j_max - 1, steps=j_max) - j_max // 2
        m = torch.meshgrid(i, j, indexing='ij')
        q = torch.sqrt(m[0] ** 2 + m[1] ** 2)
        
        # Guinier-Porod model for ideal sphere
        q1 = np.sqrt(10) / np.pi * sigma
        kernel = torch.where(q <= q1,
                             torch.exp(-(np.pi * q / sigma) ** 2 / 5), # Guinier
                             (q1 / q) ** 4 / np.e ** 2 # Porod
                             )
        kernel = torch.clamp(kernel, min=1e-8) # limit low values
        
        kernel = kernel[(None,) * 2]
        return kernel

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        
        size = tuple(input.shape)
        if mask_in is not None or self.last_size != size:
            self.last_size = size

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(size[0], size[1], size[2], size[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, size[2], size[3]).to(input)
                else:
                    mask = mask_in
                
                if self.trg_weight_model:
                    # generate weight kernel based on Guinier-Porod model
                    if self.weight_model is None:
                        self.weight_model = self.get_weight_model(size[2], size[3], sigma=min(size[2], size[3]) / self.object_size).to(input)
                    self.update_mask = F.conv2d(mask * self.weight_model, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
                    self.slide_weight = F.conv2d(self.weight_model, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
                else:
                    self.update_mask = F.conv2d(mask.float(), self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
                    if self.slide_weight is None:
                        self.slide_weight = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]
                
                self.mask_ratio = torch.div(self.slide_weight, torch.clamp(self.update_mask, min=1e-8))
                self.update_mask = torch.ge(self.update_mask, 1e-8)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(WeightedPartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output
        
class WeightedPartialConv2d_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, return_mask=True,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity, **kwargs):
        super(WeightedPartialConv2d_BN_ACT, self).__init__()
        
        self.pconv = WeightedPartialConv2d(in_channels, out_channels,
                                           kernel_size, stride, padding, dilation, groups, bias,
                                           return_mask=True, **kwargs)
        self.return_mask = return_mask
        
        self.bn = norm_layer(out_channels)
        self.act = activation_layer

    def forward(self, x):
        assert type(x) is tuple, "Input should be a tuple of two tensors: image and mask."
        input, mask_in = x
        
        output, mask_out = self.pconv(input, mask_in)
        output = self.act(self.bn(output))
        
        if self.return_mask:
            return output, mask_out
        else:
            return output