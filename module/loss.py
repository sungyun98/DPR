# Reference = https://github.com/NVIDIA/partialconv/blob/master/models/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from torchvision.transforms._presets import ImageClassification

from .network import _fft2, _ifft2


def gram_matrix(input_tensor):
    (b, ch, h, w) = input_tensor.size()
    features = input_tensor.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    
    gram = torch.bmm(features, features_t) / (ch * h * w)
    return gram


class VGG19Partial(nn.Module):
    def __init__(self, block_num=5):
        super(VGG19Partial, self).__init__()
        
        self.preprocess = ImageClassification(crop_size=224, resize_size=224, antialias=True)
        
        vgg_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        vgg_pretrained_features = vgg_model.features

        self.block_num = block_num

        self.slice1 = torch.nn.Sequential()
        for x in range(5):              # block 1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if self.block_num > 1:
            self.slice2 = torch.nn.Sequential()
            for x in range(5, 10):      # block 2
                self.slice2.add_module(str(x), vgg_pretrained_features[x])

        if self.block_num > 2:
            self.slice3 = torch.nn.Sequential()
            for x in range(10, 19):     # block 3
                self.slice3.add_module(str(x), vgg_pretrained_features[x])

        if self.block_num > 3:
            self.slice4 = torch.nn.Sequential()
            for x in range(19, 28):     # block 4
                self.slice4.add_module(str(x), vgg_pretrained_features[x])
        
        if self.block_num > 4:
            self.slice5 = torch.nn.Sequential()
            for x in range(28, 37):     # block 5
                self.slice5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = torch.cat((x, x, x), dim=1)
        x = self.preprocess(x)
        
        h = self.slice1(x)
        h1 = h
        if self.block_num == 1:
            return [h1]
        
        h = self.slice2(h)
        h2 = h
        if self.block_num == 2:
            return [h1, h2]
        
        h = self.slice3(h)
        h3 = h
        if self.block_num == 3:
            return [h1, h2, h3]
        
        h = self.slice4(h)
        h4 = h
        if self.block_num == 4:
            return [h1, h2, h3, h4]
        
        h = self.slice5(h)
        h5 = h
        return [h1, h2, h3, h4, h5]
    
class VGGLoss(nn.Module):
    def __init__(self, block_range = (3, 5), style=False, device='cpu'):
        super(VGGLoss, self).__init__()
        
        self.block_range = block_range
        self.style = style
        self.vgg19partial = VGG19Partial(block_num=self.block_range[1]).eval().to(device)
        self.loss_fn = nn.L1Loss()
        
    def forward(self, output, target):
        with torch.no_grad():
            groundtruth = self.vgg19partial(target)
        generated = self.vgg19partial(output)
        
        # perceptual
        perceptual_loss = 0
        for m in range(len(generated) - self.block_range[0], len(generated)):
            gt_data = Variable(groundtruth[m].data, requires_grad=False)
            perceptual_loss += self.loss_fn(generated[m], gt_data)
        
        # style
        if self.style:
            style_loss = 0
            for m in range(len(generated) - self.block_range[0], len(generated)):
                gen_style = gram_matrix(generated[m])
                gt_style = gram_matrix(Variable(groundtruth[m].data, requires_grad=False))
                style_loss += self.loss_fn(gen_style, gt_style)
            
            return perceptual_loss, style_loss
        
        return perceptual_loss


class CombinedLoss(nn.Module):
    def __init__(self, coeffs=(1, 10, 0.1, 0.01), device='cpu'):
        super().__init__()
        self.VGGLoss = VGGLoss(block_range=(4, 5), style=False, device=device)
        self.coeffs = coeffs
        
    @staticmethod
    def align_obj(output, target, limit=32):
        N = output.shape[0]

        xcorr = F.conv2d(output.reshape(1, N, 64, 64), target, padding=limit, groups=N).squeeze(0)
        xcorrT = F.conv2d(torch.rot90(output, 2, dims=(-2, -1)).reshape(1, N, 64, 64), target, padding=limit, groups=N).squeeze(0)
        vmax = torch.amax(xcorr, dim=(-2, -1))
        vTmax = torch.amax(xcorrT, dim=(-2, -1))

        for i in range(N):
            trg = vTmax[i] > vmax[i]
            if trg:
                output[i, :, :, :] = torch.rot90(output[i], 2, dims=(-2, -1))
                dpos = limit - torch.nonzero(xcorrT[i] == vTmax[i]).squeeze()
            else:
                dpos = limit - torch.nonzero(xcorr[i] == vmax[i]).squeeze()
                
            # for failure of equality operation
            if dpos.numel() == 0:
                pos = torch.argmax(xcorrT[i] if trg else xcorr[i])
                dpos = limit - torch.stack([pos // 65, pos % 65])

            # for multiple maximal positions
            if len(dpos.shape) > 1:
                dpos = dpos[torch.argmin(torch.sum(torch.abs(dpos), dim=-1))]
                
            output[i, :, :, :] = torch.roll(output[i], dpos.tolist(), dims=(-2, -1))
            
        return output
        
    @staticmethod
    def grad_loss(output, target):
        grad = torch.gradient(output, dim=(-2, -1))
        grad_gt = torch.gradient(target, dim=(-2, -1))
        grad_loss = (torch.mean(torch.abs(grad[0][target > 0] - grad_gt[0][target > 0])) 
                     + torch.mean(torch.abs(grad[1][target > 0] - grad_gt[1][target > 0])))
        return grad_loss
    
    @staticmethod
    def fourier_loss(output, target, log):        
        output_f = torch.abs(_fft2(F.pad(output, (224, 224, 224, 224))))
        target_f = torch.abs(_fft2(F.pad(target, (224, 224, 224, 224))))
        if log: # Log-scaled intensity difference (L1)
            output_f = torch.log10(torch.clamp(output_f, min=1e-8)) * 2
            target_f = torch.log10(torch.clamp(target_f, min=1e-8)) * 2
            loss = F.l1_loss(output_f, target_f)
        else: # Amplitude difference (Normalized L1; R-factor in PR)
            loss = torch.mean(torch.sum(torch.abs(output_f - target_f), dim=(-2, -1))
                              / torch.sum(target_f, dim=(-2, -1)))
        return loss
        
    def forward(self, output, target, align_limit=32):
        if align_limit >= 0:
            # object align (translation & 180 deg rotation)
            output = self.align_obj(output, target, limit=align_limit)
        
        l1_loss = F.l1_loss(output, target)
        grad_loss = self.grad_loss(output, target)
        perceptual_loss = self.VGGLoss(output, target)
        fourier_loss = self.fourier_loss(output, target, log=False)
        
        loss = self.coeffs[0] * l1_loss + self.coeffs[1] * grad_loss + self.coeffs[2] * perceptual_loss + self.coeffs[3] * fourier_loss
        return loss