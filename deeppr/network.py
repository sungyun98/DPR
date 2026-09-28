import torch
import torch.nn as nn
import torch.nn.functional as F

from .weightedpartialconv2d import WeightedPartialConv2d_BN_ACT
from .ffc import FFC_BN_ACT, FFCResNetBlock, ConcatTupleLayer, SplitDataLayer


def _fft2(input, s=None):
    return torch.fft.fftshift(torch.fft.fft2(input, s), dim=(-2, -1))

def _ifft2(input, s=None):
    return torch.fft.ifft2(torch.fft.ifftshift(input, dim=(-2, -1)), s)

class Network(nn.Module):
    def __init__(self, ngf=64, max_features=1024, weight_model=True, downsample_FFC=False, refinement=True):
        super(Network, self).__init__()
        
        self.downsample_FFC = downsample_FFC
        self.trg_refine = refinement
        
        n_downsample = 3
        n_residual = (3 if self.trg_refine else 6)
        n_refine = 6
        
        blocks = []
        
        # Initial Downsampling layer
        if self.downsample_FFC:
            blocks += [FFC_BN_ACT(2, ngf, ratio_gin=0, ratio_gout=0.5, kernel_size=7, stride=1, padding=3, padding_mode='reflect',
                                  norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU(inplace=True))]
        else:
            blocks += [WeightedPartialConv2d_BN_ACT(1, ngf, kernel_size=7, stride=1, padding=3, padding_mode='reflect',
                                                    norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU(inplace=True),
                                                    return_mask=True, weight_model=weight_model)]
        
        # Downsampling Layers
        for i in range(n_downsample):
            mult = 2 ** i
            if self.downsample_FFC:
                blocks += [FFC_BN_ACT(min(max_features, ngf * mult), min(max_features, ngf * mult * 2),
                                      ratio_gin=0.5, ratio_gout=0.5, kernel_size=3, stride=2, padding=1, padding_mode='reflect',
                                      norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU(inplace=True))]
            else:
                blocks += [WeightedPartialConv2d_BN_ACT(min(max_features, ngf * mult), min(max_features, ngf * mult * 2),
                                                        kernel_size=3, stride=2, padding=1, padding_mode='reflect',
                                                        norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU(inplace=True),
                                                        return_mask=(True if i < n_downsample - 1 else False), weight_model=weight_model)]
        nbf = min(max_features, ngf * 2 ** n_downsample)
        
        # Residual Layers
        blocks += [(ConcatTupleLayer() if self.downsample_FFC else nn.Identity()),
                   nn.Conv2d(nbf, nbf, 3, 1, 1, groups=nbf, bias=False),
                   nn.Conv2d(nbf, nbf, 1, 1, 0, bias=False),
                   nn.BatchNorm2d(nbf), nn.ReLU(inplace=True), SplitDataLayer()]
        
        for i in range(n_residual):
            blocks += [FFCResNetBlock(nbf, padding_mode='reflect',
                                      norm_layer=nn.BatchNorm2d,
                                      activation_layer=nn.ReLU(inplace=True))]
        
        # Merging Layers
        for i in range(n_downsample):
            mult = 2 ** (n_downsample - i)
            blocks += [FFC_BN_ACT(min(max_features, ngf * mult), min(max_features, int(ngf * mult / 2)),
                                  ratio_gin=0.5, ratio_gout=0.5, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                                  norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU(inplace=True))]
        blocks += [ConcatTupleLayer(),
                   nn.Conv2d(ngf, ngf, 3, 1, 1, groups=ngf, bias=False),
                   nn.Conv2d(ngf,   1, 1, 1, 0, bias=False),
                   nn.Sigmoid()]
        
        self.model = nn.Sequential(*blocks)
        
        # Refinement Layers
        if self.trg_refine:
            blocks = []
            blocks += [nn.Conv2d(2,   2, 3, 1, 1, groups=2, bias=False),
                       nn.Conv2d(2, ngf, 1, 1, 0, bias=False),
                       nn.BatchNorm2d(ngf), nn.ReLU(inplace=True), SplitDataLayer()]
            
            for i in range(n_refine):
                blocks += [FFCResNetBlock(ngf, padding_mode='reflect',
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU(inplace=True))]
            blocks += [ConcatTupleLayer(),
                       nn.Conv2d(ngf, ngf, 3, 1, 1, groups=ngf, bias=False),
                       nn.Conv2d(ngf,   1, 1, 1, 0, bias=False),
                       nn.Sigmoid()]
            
            self.refine = nn.Sequential(*blocks)
    
    @staticmethod
    def project_modulus(x, input, mask, beta=1):
        x_f = _fft2(F.pad(x, (224, 224, 224, 224)))
        abs, angle = torch.abs(x_f), torch.angle(x_f)
        scale = (torch.sum(input, dim=(-2, -1), keepdim=True)
                 / torch.sum(abs ** 2 * mask, dim=(-2, -1), keepdim=True))
        
        abs_proj = (torch.sqrt(torch.clamp(input / scale, min=1e-08)) * beta
                    + abs * (1 - mask * beta))
        x_proj = torch.abs(_ifft2(torch.polar(abs_proj, angle))
                           [:, :, 256-32:256+32, 256-32:256+32])
        
        return x_proj
    
    def forward(self, input, mask, false_scale=False):
        # normalization
        input = torch.clamp(input * mask, min=0)
        x = input / torch.amax(input, dim=(-2, -1), keepdim=True)
        
        # model
        x = self.model(torch.cat((x, mask), dim=1) if self.downsample_FFC else (x, mask))
        
        # refinement
        if self.trg_refine:
            x_proj = self.project_modulus(x, input, mask)
            x = self.refine(torch.cat((x, x_proj), dim=1))
        
        # scaling
        x_inten = torch.abs(_fft2(F.pad(x, (224, 224, 224, 224)))) ** 2
        scale = (torch.sum(input, dim=(-2, -1), keepdim=True)
                 / torch.sum(x_inten * mask, dim=(-2, -1), keepdim=True))
        
        if false_scale:
            output = x * scale
        else:
            output = x * torch.sqrt(scale)
        
        return output