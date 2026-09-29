# Derived from the partial convolution of https://github.com/NVIDIA/partialconv
#   (models/partialconv2d.py at a99cd7cb9f6469c02181d9aa34fe5abd95fb0154)
#   Copyright (c) 2018, NVIDIA CORPORATION, BSD 3-Clause License
#   (LICENSES/partialconv-BSD-3-Clause.txt)
#   Modified by Sung Yun Lee: mask weighting by the Guinier-Porod model; docstrings and type
#   hints; reformatted with ruff
# Guinier-Porod model from https://doi.org/10.1107/S0021889810015773

"""Weighted partial convolution for diffraction patterns with missing pixels."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class WeightedPartialConv2d(nn.Conv2d):
    """Partial convolution whose mask renormalization is weighted by a diffraction model.

    As in a partial convolution, only valid pixels contribute and the output is rescaled by
    the ratio of the total to the valid weight under the kernel. Here each pixel is weighted
    by the Guinier-Porod intensity profile of a sphere (see `get_weight_model`), so that
    missing pixels near the centre of the pattern, where the intensity is high, count more
    than missing pixels at high scattering angles. The weights are computed on the first
    call for the input size and kept.

    Parameters
    ----------
    *args, **kwargs
        Arguments of `torch.nn.Conv2d`, and the keywords below.
    multi_channel : bool, default False
        Use a mask per input channel (shape ``(N, C, H, W)``) instead of ``(N or 1, 1, H, W)``.
    return_mask : bool, default False
        Also return the updated mask from `forward`.
    object_size : float, default 64
        Object size in pixels of the real-space grid; the width of the model is
        ``min(H, W) / object_size``.
    weight_model : bool, default True
        Weight by the Guinier-Porod model; if False, this is the ordinary partial convolution.

    References
    ----------
    .. [1] Guinier-Porod model, https://doi.org/10.1107/S0021889810015773
    """

    def __init__(self, *args, **kwargs) -> None:

        # whether the mask is multi-channel or not
        if "multi_channel" in kwargs:
            self.multi_channel = kwargs["multi_channel"]
            kwargs.pop("multi_channel")
        else:
            self.multi_channel = False

        if "return_mask" in kwargs:
            self.return_mask = kwargs["return_mask"]
            kwargs.pop("return_mask")
        else:
            self.return_mask = False

        if "object_size" in kwargs:
            self.object_size = kwargs["object_size"]
            kwargs.pop("object_size")
        else:
            self.object_size = 64

        # Guinier-Porod model
        if "weight_model" in kwargs:
            self.trg_weight_model = kwargs["weight_model"]
            kwargs.pop("weight_model")
        else:
            self.trg_weight_model = True

        super().__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(
                self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
            )
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.last_size = (None, None, None, None)
        self.weight_model = None
        self.slide_weight = None
        self.update_mask = None
        self.mask_ratio = None

    @staticmethod
    def get_weight_model(i_max: int, j_max: int, sigma: float) -> Tensor:
        """Return the Guinier-Porod intensity profile of an ideal sphere.

        The profile is Guinier, ``exp(-(pi * q / sigma)**2 / 5)``, up to
        ``q1 = sqrt(10) / pi * sigma`` and Porod, ``(q1 / q)**4 / e**2``, beyond, where ``q``
        is the distance in pixels from index ``(i_max // 2, j_max // 2)``. Values are
        clamped to at least 1e-8.

        Parameters
        ----------
        i_max, j_max : int
            Size of the profile.
        sigma : float
            Width of the profile in pixels (the speckle size).

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``(1, 1, i_max, j_max)``.
        """
        i = torch.linspace(0, i_max - 1, steps=i_max) - i_max // 2
        j = torch.linspace(0, j_max - 1, steps=j_max) - j_max // 2
        m = torch.meshgrid(i, j, indexing="ij")
        q = torch.sqrt(m[0] ** 2 + m[1] ** 2)

        # Guinier-Porod model for ideal sphere
        q1 = np.sqrt(10) / np.pi * sigma
        kernel = torch.where(
            q <= q1,
            torch.exp(-((np.pi * q / sigma) ** 2) / 5),  # Guinier
            (q1 / q) ** 4 / np.e**2,  # Porod
        )
        kernel = torch.clamp(kernel, min=1e-8)  # limit low values

        kernel = kernel[(None,) * 2]
        return kernel

    def forward(
        self, input: Tensor, mask_in: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply the weighted partial convolution.

        Parameters
        ----------
        input : torch.Tensor
            Real tensor of shape ``(N, C, H, W)``, fftshifted.
        mask_in : torch.Tensor, optional
            Mask of valid pixels (1 or True valid, 0 or False missing) of the shape described
            in the class docstring. By default, all pixels are valid.

        Returns
        -------
        output : torch.Tensor
            Real tensor of shape ``(N, C_out, H_out, W_out)``; zero where no valid pixel is
            under the kernel.
        update_mask : torch.Tensor
            Bool mask of the output, returned only if ``return_mask`` is True.
        """
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
                        self.weight_model = self.get_weight_model(
                            size[2], size[3], sigma=min(size[2], size[3]) / self.object_size
                        ).to(input)
                    self.update_mask = F.conv2d(
                        mask * self.weight_model,
                        self.weight_maskUpdater,
                        bias=None,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=1,
                    )
                    self.slide_weight = F.conv2d(
                        self.weight_model,
                        self.weight_maskUpdater,
                        bias=None,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=1,
                    )
                else:
                    self.update_mask = F.conv2d(
                        mask.float(),
                        self.weight_maskUpdater,
                        bias=None,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=1,
                    )
                    if self.slide_weight is None:
                        self.slide_weight = (
                            self.weight_maskUpdater.shape[1]
                            * self.weight_maskUpdater.shape[2]
                            * self.weight_maskUpdater.shape[3]
                        )

                self.mask_ratio = torch.div(
                    self.slide_weight, torch.clamp(self.update_mask, min=1e-8)
                )
                self.update_mask = torch.ge(self.update_mask, 1e-8)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super().forward(torch.mul(input, mask) if mask_in is not None else input)

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
    """`WeightedPartialConv2d` followed by normalization and activation.

    Parameters
    ----------
    in_channels, out_channels, kernel_size : int
        Parameters of the convolution.
    stride, padding, dilation, groups : int
        Parameters of the convolution (defaults 1, 0, 1, 1).
    bias : bool, default False
        Bias of the convolution.
    return_mask : bool, default True
        Also return the updated mask, for the next layer.
    norm_layer : type of torch.nn.Module, default torch.nn.BatchNorm2d
        Normalization class, constructed with ``out_channels``.
    activation_layer : torch.nn.Module, optional
        Activation module. By default, the identity.
    **kwargs
        Further keywords of `WeightedPartialConv2d` and `torch.nn.Conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        return_mask: bool = True,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation_layer: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pconv = WeightedPartialConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            return_mask=True,
            **kwargs,
        )
        self.return_mask = return_mask

        self.bn = norm_layer(out_channels)
        self.act = nn.Identity() if activation_layer is None else activation_layer

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor | tuple[Tensor, Tensor]:
        """Apply the layer.

        Parameters
        ----------
        x : tuple of torch.Tensor
            Image and mask ``(input, mask)`` for `WeightedPartialConv2d.forward`.

        Returns
        -------
        output : torch.Tensor
            Output feature maps.
        mask : torch.Tensor
            Updated mask, returned only if ``return_mask`` is True.
        """
        assert type(x) is tuple, "Input should be a tuple of two tensors: image and mask."
        input, mask_in = x

        output, mask_out = self.pconv(input, mask_in)
        output = self.act(self.bn(output))

        if self.return_mask:
            return output, mask_out
        else:
            return output
