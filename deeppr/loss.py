# VGG19Partial and gram_matrix adapted from https://github.com/NVIDIA/partialconv
#   (models/loss.py at 3f869a3a096d8a25de66e5b8b2bc0dc55db43f38; VGG16Partial rewritten for VGG19;
#   reformatted with ruff)
#   Copyright (c) 2019, NVIDIA CORPORATION, BSD 3-Clause License
#   (LICENSES/partialconv-BSD-3-Clause.txt)

"""Training loss of DPR: L1, gradient, VGG19 perceptual and Fourier terms."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from phaseretrieval import AlignObject
from torch import Tensor
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from .network import _fft2


def gram_matrix(input_tensor: Tensor) -> Tensor:
    """Return the Gram matrix of feature maps, normalized by their size.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Feature maps of shape ``(B, C, H, W)``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, C, C)`` divided by ``C * H * W``.
    """
    (b, ch, h, w) = input_tensor.size()
    features = input_tensor.view(b, ch, w * h)
    features_t = features.transpose(1, 2)

    gram = torch.bmm(features, features_t) / (ch * h * w)
    return gram


class VGG19Partial(nn.Module):
    """Frozen feature extractor from the first blocks of VGG19 pretrained on ImageNet.

    The ImageNet weights (``VGG19_Weights.IMAGENET1K_V1``) are downloaded by torchvision on
    first use.

    Parameters
    ----------
    block_num : int, default 5
        Number of VGG19 blocks (1 to 5) to evaluate.
    """

    def __init__(self, block_num: int = 5) -> None:
        super().__init__()

        # same operations as
        # torchvision.transforms._presets.ImageClassification(crop_size=224, resize_size=224)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.BILINEAR, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        vgg_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        vgg_pretrained_features = vgg_model.features

        self.block_num = block_num

        self.slice1 = torch.nn.Sequential()
        for x in range(5):  # block 1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if self.block_num > 1:
            self.slice2 = torch.nn.Sequential()
            for x in range(5, 10):  # block 2
                self.slice2.add_module(str(x), vgg_pretrained_features[x])

        if self.block_num > 2:
            self.slice3 = torch.nn.Sequential()
            for x in range(10, 19):  # block 3
                self.slice3.add_module(str(x), vgg_pretrained_features[x])

        if self.block_num > 3:
            self.slice4 = torch.nn.Sequential()
            for x in range(19, 28):  # block 4
                self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if self.block_num > 4:
            self.slice5 = torch.nn.Sequential()
            for x in range(28, 37):  # block 5
                self.slice5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> list[Tensor]:
        """Extract the features of each block.

        Parameters
        ----------
        x : torch.Tensor
            Real single-channel images of shape ``(N, 1, H, W)``; they are repeated to three
            channels, resized to 224 x 224 and normalized as ImageNet images.

        Returns
        -------
        list of torch.Tensor
            Outputs of blocks 1 to ``block_num``.
        """
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
    """Perceptual (and optionally style) L1 loss on VGG19 features.

    Parameters
    ----------
    block_range : tuple of int, default (3, 5)
        ``(k, n)``: evaluate the first ``n`` VGG19 blocks and compare the last ``k`` of them.
    style : bool, default False
        Also return the style loss (L1 distance of Gram matrices).
    device : str, int or torch.device, default 'cpu'
        Device of the VGG19 network.
    """

    def __init__(
        self,
        block_range: tuple[int, int] = (3, 5),
        style: bool = False,
        device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()

        self.block_range = block_range
        self.style = style
        self.vgg19partial = VGG19Partial(block_num=self.block_range[1]).eval().to(device)
        self.loss_fn = nn.L1Loss()

    def forward(self, output: Tensor, target: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Compute the loss.

        Parameters
        ----------
        output, target : torch.Tensor
            Real images of shape ``(N, 1, H, W)``; no gradient flows through ``target``.

        Returns
        -------
        perceptual_loss : torch.Tensor
            Scalar tensor.
        style_loss : torch.Tensor
            Scalar tensor, returned only if ``style`` is True.
        """
        with torch.no_grad():
            groundtruth = self.vgg19partial(target)
        generated = self.vgg19partial(output)

        # perceptual
        perceptual_loss = 0
        for m in range(len(generated) - self.block_range[0], len(generated)):
            gt_data = groundtruth[m].detach()
            perceptual_loss += self.loss_fn(generated[m], gt_data)

        # style
        if self.style:
            style_loss = 0
            for m in range(len(generated) - self.block_range[0], len(generated)):
                gen_style = gram_matrix(generated[m])
                gt_style = gram_matrix(groundtruth[m].detach())
                style_loss += self.loss_fn(gen_style, gt_style)

            return perceptual_loss, style_loss

        return perceptual_loss


class CombinedLoss(nn.Module):
    """Training loss of DPR.

    The weighted sum of the L1 loss, the gradient loss (`grad_loss`), the VGG19 perceptual
    loss of the last four blocks (`VGGLoss`) and the Fourier amplitude loss (`fourier_loss`),
    computed after aligning the output to the target (`phaseretrieval.AlignObject`).

    Parameters
    ----------
    coeffs : tuple of float, default (1, 10, 0.1, 0.01)
        Weights of the L1, gradient, perceptual and Fourier terms.
    device : str, int or torch.device, default 'cpu'
        Device of the VGG19 network.
    """

    def __init__(
        self,
        coeffs: tuple[float, float, float, float] = (1, 10, 0.1, 0.01),
        device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.VGGLoss = VGGLoss(block_range=(4, 5), style=False, device=device)
        self.coeffs = coeffs

    @staticmethod
    def grad_loss(output: Tensor, target: Tensor) -> Tensor:
        """Return the L1 difference of the image gradients on the object pixels.

        Parameters
        ----------
        output, target : torch.Tensor
            Real images of shape ``(N, 1, H, W)``; the loss uses the pixels where
            ``target > 0``.

        Returns
        -------
        torch.Tensor
            Scalar tensor: sum of the mean absolute differences along both axes.
        """
        grad = torch.gradient(output, dim=(-2, -1))
        grad_gt = torch.gradient(target, dim=(-2, -1))
        grad_loss = torch.mean(
            torch.abs(grad[0][target > 0] - grad_gt[0][target > 0])
        ) + torch.mean(torch.abs(grad[1][target > 0] - grad_gt[1][target > 0]))
        return grad_loss

    @staticmethod
    def fourier_loss(output: Tensor, target: Tensor, log: bool) -> Tensor:
        """Compare the diffraction amplitudes of output and target.

        Both are zero-padded to 512 x 512 (from 64 x 64) before the Fourier transform.

        Parameters
        ----------
        output, target : torch.Tensor
            Real objects of shape ``(N, 1, 64, 64)``.
        log : bool
            If True, L1 difference of the log10 intensities; otherwise the normalized L1
            difference of the amplitudes (the R-factor of phase retrieval), averaged over the
            batch.

        Returns
        -------
        torch.Tensor
            Scalar tensor.
        """
        output_f = torch.abs(_fft2(F.pad(output, (224, 224, 224, 224))))
        target_f = torch.abs(_fft2(F.pad(target, (224, 224, 224, 224))))
        if log:  # Log-scaled intensity difference (L1)
            output_f = torch.log10(torch.clamp(output_f, min=1e-8)) * 2
            target_f = torch.log10(torch.clamp(target_f, min=1e-8)) * 2
            loss = F.l1_loss(output_f, target_f)
        else:  # Amplitude difference (Normalized L1; R-factor in PR)
            loss = torch.mean(
                torch.sum(torch.abs(output_f - target_f), dim=(-2, -1))
                / torch.sum(target_f, dim=(-2, -1))
            )
        return loss

    def forward(self, output: Tensor, target: Tensor, align: bool = True) -> Tensor:
        """Compute the combined loss.

        Parameters
        ----------
        output : torch.Tensor
            Network output of shape ``(N, 1, 64, 64)``.
        target : torch.Tensor
            Target objects of shape ``(N, 1, 64, 64)``.
        align : bool, default True
            Align the output to the target first (translation up to 32 pixels and twin
            image, see `phaseretrieval.AlignObject`).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if align:
            output = AlignObject(output, target)

        l1_loss = F.l1_loss(output, target)
        grad_loss = self.grad_loss(output, target)
        perceptual_loss = self.VGGLoss(output, target)
        fourier_loss = self.fourier_loss(output, target, log=False)

        loss = (
            self.coeffs[0] * l1_loss
            + self.coeffs[1] * grad_loss
            + self.coeffs[2] * perceptual_loss
            + self.coeffs[3] * fourier_loss
        )
        return loss
