# Fast Fourier Convolution (FFC) from https://github.com/pkumivision/FFC
#   (model_zoo/ffc.py at ddf26ddd1d22b2062b231311a23c9111db2997ac)
#   Copyright 2022 Lu Chi, Apache License 2.0 (LICENSES/FFC-Apache-2.0.txt)
# FFC ResNet Block from https://github.com/advimman/lama
#   (saicinpainting/training/modules/ffc.py at 552cd55c94c9a143080aa855e7a8bdb2998d55bb)
#   Copyright 2021 Samsung Research, Apache License 2.0 (LICENSES/LaMa-Apache-2.0.txt)
#
# Modified by Sung Yun Lee:
# - FourierUnit: torch.rfft/torch.irfft replaced by torch.fft.rfft2/irfft2 (norm='ortho')
# - FFC, FFC_BN_ACT: extra keyword arguments passed to the convolutions; activation layers
#   are passed as module instances instead of classes (default None: identity); FFCSE_block
#   omitted
# - FFCResNetBlock: simplified from LaMa's FFCResnetBlock (fixed 0.5 global ratio, no spatial
#   transform or inline mode); ConcatTupleLayer copied from LaMa; SplitDataLayer added
# - docstrings and type hints added; code reformatted with ruff

"""Fast Fourier convolution (FFC) layers.

An FFC splits the channels into a local branch (ordinary convolutions) and a global branch
(convolutions in the Fourier domain, with an image-wide receptive field). Feature maps are
passed between layers as tuples ``(x_l, x_g)``; a missing branch is the integer 0.

References
----------
.. [1] L. Chi, B. Jiang, Y. Mu, Fast Fourier convolution, NeurIPS 2020,
   https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf
.. [2] R. Suvorov et al., Resolution-robust large mask inpainting with Fourier
   convolutions (LaMa), https://arxiv.org/abs/2109.07161
"""

import torch
import torch.nn as nn
from torch import Tensor

#: Local and global feature maps; a missing branch is the integer 0.
Pair = tuple[Tensor | int, Tensor | int]


class FourierUnit(nn.Module):
    """Pointwise convolution in the Fourier domain.

    The real and imaginary parts of the real FFT (orthonormal) are stacked as channels,
    mixed by a 1 x 1 convolution with batch normalization and ReLU, and transformed back.

    Parameters
    ----------
    in_channels, out_channels : int
        Numbers of input and output channels.
    groups : int, default 1
        Groups of the 1 x 1 convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 1) -> None:
        # bn_layer not used
        super().__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the unit.

        Parameters
        ----------
        x : torch.Tensor
            Real tensor of shape ``(N, in_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``(N, out_channels, H, W)``.
        """
        size = x.shape

        ffted = torch.fft.rfft2(x, norm="ortho")
        ffted = torch.view_as_real(ffted)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((size[0], -1) + ffted.shape[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = (
            ffted.view((size[0], -1, 2) + ffted.shape[2:]).permute(0, 1, 3, 4, 2).contiguous()
        )  # (batch, c, t, h, w/2+1, 2)

        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=size[2:], norm="ortho")

        return output


class SpectralTransform(nn.Module):
    """Global branch of an FFC: a Fourier unit with an optional local Fourier unit (LFU).

    Parameters
    ----------
    in_channels, out_channels : int
        Numbers of input and output channels.
    stride : int, default 1
        2 halves the size with average pooling first.
    groups : int, default 1
        Groups of the convolutions.
    enable_lfu : bool, default True
        Add the local Fourier unit, applied to a quarter of the channels split into 2 x 2
        spatial patches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        enable_lfu: bool = True,
    ) -> None:
        # bn_layer not used
        super().__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the transform.

        Parameters
        ----------
        x : torch.Tensor
            Real tensor of shape ``(N, in_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``(N, out_channels, H / stride, W / stride)``.
        """
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(x[:, : c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):
    """Fast Fourier convolution.

    The output branches sum the local-to-local and global-to-local paths (local
    convolutions) and the local-to-global (local convolution) and global-to-global
    (`SpectralTransform`) paths.

    Parameters
    ----------
    in_channels, out_channels : int
        Total numbers of input and output channels.
    kernel_size : int
        Kernel size of the local convolutions.
    ratio_gin, ratio_gout : float
        Fractions of the input and output channels in the global branch.
    stride : {1, 2}, default 1
        Stride.
    padding, dilation, groups : int
        Parameters of the local convolutions (defaults 0, 1, 1).
    bias : bool, default False
        Bias of the local convolutions.
    enable_lfu : bool, default True
        Local Fourier unit in the global-to-global path.
    **kwargs
        Further arguments of the local `torch.nn.Conv2d` (for example ``padding_mode``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        enable_lfu: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(
            in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias, **kwargs
        )
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(
            in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias, **kwargs
        )
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(
            in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias, **kwargs
        )
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

    def forward(self, x: Tensor | Pair) -> Pair:
        """Apply the convolution.

        Parameters
        ----------
        x : torch.Tensor or tuple
            Local and global feature maps ``(x_l, x_g)``, or a single tensor for the local
            branch only.

        Returns
        -------
        tuple
            Output feature maps ``(out_l, out_g)``.
        """
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    """FFC followed by normalization and activation on each branch.

    Parameters
    ----------
    in_channels, out_channels, kernel_size, ratio_gin, ratio_gout
        Parameters of `FFC`.
    stride, padding, dilation, groups, bias, enable_lfu
        Parameters of `FFC`, with the same defaults.
    norm_layer : type of torch.nn.Module, default torch.nn.BatchNorm2d
        Normalization class, constructed with the number of channels of each branch.
    activation_layer : torch.nn.Module, optional
        Activation module shared by both branches. By default, the identity.
    **kwargs
        Further arguments of the local convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation_layer: nn.Module | None = None,
        enable_lfu: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.ffc = FFC(
            in_channels,
            out_channels,
            kernel_size,
            ratio_gin,
            ratio_gout,
            stride,
            padding,
            dilation,
            groups,
            bias,
            enable_lfu,
            **kwargs,
        )
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))

        act = nn.Identity() if activation_layer is None else activation_layer
        lact = nn.Identity() if ratio_gout == 1 else act
        gact = nn.Identity() if ratio_gout == 0 else act
        self.act_l = lact
        self.act_g = gact

    def forward(self, x: Tensor | Pair) -> Pair:
        """Apply the layer.

        Parameters
        ----------
        x : torch.Tensor or tuple
            Input of `FFC.forward`.

        Returns
        -------
        tuple
            Output feature maps ``(out_l, out_g)``.
        """
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResNetBlock(nn.Module):
    """Residual block of two 3 x 3 `FFC_BN_ACT` layers with half of the channels global.

    Parameters
    ----------
    channels : int
        Total number of channels (local plus global).
    norm_layer : type of torch.nn.Module, default torch.nn.BatchNorm2d
        Normalization class.
    activation_layer : torch.nn.Module, optional
        Activation module. By default, the identity.
    **kwargs
        Further arguments of the local convolutions (for example ``padding_mode``).
    """

    def __init__(
        self,
        channels: int,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation_layer: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.conv1 = FFC_BN_ACT(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            ratio_gin=0.5,
            ratio_gout=0.5,
            **kwargs,
        )
        self.conv2 = FFC_BN_ACT(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            ratio_gin=0.5,
            ratio_gout=0.5,
            **kwargs,
        )

    def forward(self, x: Tensor | Pair) -> Pair:
        """Apply the block with the identity shortcut.

        Parameters
        ----------
        x : torch.Tensor or tuple
            Feature maps ``(x_l, x_g)`` with ``channels / 2`` channels each, or a single
            tensor for the local branch only.

        Returns
        -------
        tuple
            Output feature maps ``(out_l, out_g)``.
        """
        x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g

        return x_l, x_g


class ConcatTupleLayer(nn.Module):
    """Concatenate the local and global feature maps along the channels."""

    def forward(self, x: Pair) -> Tensor:
        """Concatenate ``(x_l, x_g)``.

        Parameters
        ----------
        x : tuple
            Feature maps ``(x_l, x_g)``; at least one must be a tensor.

        Returns
        -------
        torch.Tensor
            ``x_l`` if ``x_g`` is not a tensor, otherwise ``cat((x_l, x_g), dim=1)``.
        """
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class SplitDataLayer(nn.Module):
    """Split a tensor into local and global halves along the channels."""

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Split ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(N, C, H, W)``.

        Returns
        -------
        tuple of torch.Tensor
            ``(x_l, x_g)``, the first and second halves of the channels (``x_l`` has one
            more channel if ``C`` is odd).
        """
        assert torch.is_tensor(x)
        x_l, x_g = torch.tensor_split(x, 2, dim=1)

        return x_l, x_g
