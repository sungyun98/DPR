"""DPR network: direct phase retrieval of a single diffraction pattern.

Diffraction patterns are 512 x 512 intensities, fftshifted (zero frequency at the centre);
objects are 64 x 64 real images that sit at the centre of the 512 x 512 real-space grid, so
the oversampling ratio is 8 along each axis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ffc import FFC_BN_ACT, ConcatTupleLayer, FFCResNetBlock, SplitDataLayer
from .weightedpartialconv2d import WeightedPartialConv2d_BN_ACT


def _fft2(input: Tensor, s: tuple[int, int] | None = None) -> Tensor:
    """Return the 2-D FFT over the last two dimensions, fftshifted.

    Parameters
    ----------
    input : torch.Tensor
        Real or complex tensor of shape ``(..., H, W)``.
    s : tuple of int, optional
        Output size, as in `torch.fft.fft2`.

    Returns
    -------
    torch.Tensor
        Complex tensor with the zero frequency at the centre.
    """
    return torch.fft.fftshift(torch.fft.fft2(input, s), dim=(-2, -1))


def _ifft2(input: Tensor, s: tuple[int, int] | None = None) -> Tensor:
    """Return the inverse 2-D FFT of fftshifted data, the inverse of `_fft2`.

    Parameters
    ----------
    input : torch.Tensor
        Complex tensor of shape ``(..., H, W)`` with the zero frequency at the centre.
    s : tuple of int, optional
        Output size, as in `torch.fft.ifft2`.

    Returns
    -------
    torch.Tensor
        Complex tensor.
    """
    return torch.fft.ifft2(torch.fft.ifftshift(input, dim=(-2, -1)), s)


class Network(nn.Module):
    """DPR network: diffraction intensity to real-space object.

    The encoder downsamples the 512 x 512 pattern to 64 x 64 with weighted partial
    convolutions (or FFC layers), followed by FFC residual blocks and merging layers. The
    optional refinement stage projects the first estimate on the measured modulus and refines
    both with further FFC residual blocks. The output is scaled so that its diffraction
    intensity matches the measured one on the valid pixels.

    Parameters
    ----------
    ngf : int, default 64
        Number of channels after the first layer.
    max_features : int, default 1024
        Maximum number of channels.
    weight_model : bool, default True
        Weight the partial convolutions by the Guinier-Porod model (see
        `WeightedPartialConv2d`).
    downsample_FFC : bool, default False
        Use FFC layers instead of weighted partial convolutions for downsampling.
    refinement : bool, default True
        Add the refinement stage (3 instead of 6 residual blocks in the main stage).
    """

    def __init__(
        self,
        ngf: int = 64,
        max_features: int = 1024,
        weight_model: bool = True,
        downsample_FFC: bool = False,
        refinement: bool = True,
    ) -> None:
        super().__init__()

        self.downsample_FFC = downsample_FFC
        self.trg_refine = refinement

        n_downsample = 3
        n_residual = 3 if self.trg_refine else 6
        n_refine = 6

        blocks = []

        # Initial Downsampling layer
        if self.downsample_FFC:
            blocks += [
                FFC_BN_ACT(
                    2,
                    ngf,
                    ratio_gin=0,
                    ratio_gout=0.5,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    padding_mode="reflect",
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=nn.ReLU(inplace=True),
                )
            ]
        else:
            blocks += [
                WeightedPartialConv2d_BN_ACT(
                    1,
                    ngf,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    padding_mode="reflect",
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=nn.ReLU(inplace=True),
                    return_mask=True,
                    weight_model=weight_model,
                )
            ]

        # Downsampling Layers
        for i in range(n_downsample):
            mult = 2**i
            if self.downsample_FFC:
                blocks += [
                    FFC_BN_ACT(
                        min(max_features, ngf * mult),
                        min(max_features, ngf * mult * 2),
                        ratio_gin=0.5,
                        ratio_gout=0.5,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        padding_mode="reflect",
                        norm_layer=nn.BatchNorm2d,
                        activation_layer=nn.ReLU(inplace=True),
                    )
                ]
            else:
                blocks += [
                    WeightedPartialConv2d_BN_ACT(
                        min(max_features, ngf * mult),
                        min(max_features, ngf * mult * 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        padding_mode="reflect",
                        norm_layer=nn.BatchNorm2d,
                        activation_layer=nn.ReLU(inplace=True),
                        return_mask=(True if i < n_downsample - 1 else False),
                        weight_model=weight_model,
                    )
                ]
        nbf = min(max_features, ngf * 2**n_downsample)

        # Residual Layers
        blocks += [
            (ConcatTupleLayer() if self.downsample_FFC else nn.Identity()),
            nn.Conv2d(nbf, nbf, 3, 1, 1, groups=nbf, bias=False),
            nn.Conv2d(nbf, nbf, 1, 1, 0, bias=False),
            nn.BatchNorm2d(nbf),
            nn.ReLU(inplace=True),
            SplitDataLayer(),
        ]

        for i in range(n_residual):
            blocks += [
                FFCResNetBlock(
                    nbf,
                    padding_mode="reflect",
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=nn.ReLU(inplace=True),
                )
            ]

        # Merging Layers
        for i in range(n_downsample):
            mult = 2 ** (n_downsample - i)
            blocks += [
                FFC_BN_ACT(
                    min(max_features, ngf * mult),
                    min(max_features, int(ngf * mult / 2)),
                    ratio_gin=0.5,
                    ratio_gout=0.5,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="reflect",
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=nn.ReLU(inplace=True),
                )
            ]
        blocks += [
            ConcatTupleLayer(),
            nn.Conv2d(ngf, ngf, 3, 1, 1, groups=ngf, bias=False),
            nn.Conv2d(ngf, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        ]

        self.model = nn.Sequential(*blocks)

        # Refinement Layers
        if self.trg_refine:
            blocks = []
            blocks += [
                nn.Conv2d(2, 2, 3, 1, 1, groups=2, bias=False),
                nn.Conv2d(2, ngf, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(inplace=True),
                SplitDataLayer(),
            ]

            for i in range(n_refine):
                blocks += [
                    FFCResNetBlock(
                        ngf,
                        padding_mode="reflect",
                        norm_layer=nn.BatchNorm2d,
                        activation_layer=nn.ReLU(inplace=True),
                    )
                ]
            blocks += [
                ConcatTupleLayer(),
                nn.Conv2d(ngf, ngf, 3, 1, 1, groups=ngf, bias=False),
                nn.Conv2d(ngf, 1, 1, 1, 0, bias=False),
                nn.Sigmoid(),
            ]

            self.refine = nn.Sequential(*blocks)

    @staticmethod
    def project_modulus(x: Tensor, input: Tensor, mask: Tensor, beta: float = 1) -> Tensor:
        """Project an object estimate on the measured Fourier modulus.

        The estimate is zero-padded to 512 x 512 and its Fourier amplitude is replaced by the
        measured one (rescaled to the total intensity of the estimate) on valid pixels,
        keeping the phase.

        Parameters
        ----------
        x : torch.Tensor
            Real object estimate of shape ``(N, 1, 64, 64)``.
        input : torch.Tensor
            Measured intensity of shape ``(N, 1, 512, 512)``, fftshifted.
        mask : torch.Tensor
            Real or bool tensor of shape ``(N, 1, 512, 512)``, 1 for valid pixels.
        beta : float, default 1
            Relaxation: 1 replaces the amplitude, 0 keeps the estimate.

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``(N, 1, 64, 64)``: modulus of the central 64 x 64 region of
            the projected object.
        """
        x_f = _fft2(F.pad(x, (224, 224, 224, 224)))
        abs, angle = torch.abs(x_f), torch.angle(x_f)
        scale = torch.sum(input, dim=(-2, -1), keepdim=True) / torch.sum(
            abs**2 * mask, dim=(-2, -1), keepdim=True
        )

        abs_proj = torch.sqrt(torch.clamp(input / scale, min=1e-08)) * beta + abs * (
            1 - mask * beta
        )
        x_proj = torch.abs(
            _ifft2(torch.polar(abs_proj, angle))[:, :, 256 - 32 : 256 + 32, 256 - 32 : 256 + 32]
        )

        return x_proj

    def forward(self, input: Tensor, mask: Tensor, false_scale: bool = False) -> Tensor:
        """Reconstruct the object from a diffraction pattern.

        Parameters
        ----------
        input : torch.Tensor
            Measured intensity of shape ``(N, 1, 512, 512)``, fftshifted, in photon counts.
            Negative values are clipped to zero.
        mask : torch.Tensor
            Real or bool tensor of shape ``(N, 1, 512, 512)``, 1 for valid pixels and 0 for
            missing pixels.
        false_scale : bool, default False
            If True, multiply by the intensity scale factor instead of its square root, as for
            the targets of `GenerateDiffraction` with ``false_scale=True`` (training).

        Returns
        -------
        torch.Tensor
            Real object of shape ``(N, 1, 64, 64)``, whose diffraction intensity matches the
            total measured intensity on the valid pixels.
        """
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
        scale = torch.sum(input, dim=(-2, -1), keepdim=True) / torch.sum(
            x_inten * mask, dim=(-2, -1), keepdim=True
        )

        if false_scale:
            output = x * scale
        else:
            output = x * torch.sqrt(scale)

        return output
