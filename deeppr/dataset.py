"""Training data for DPR: mask augmentation, simulated diffraction and the HDF5 dataset."""

import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from .network import _fft2, _ifft2


class Binarize:
    """Transform that converts an image to grayscale and binarizes it.

    Parameters
    ----------
    threshold : float
        Threshold in ``[0, 1]`` of the maximum gray level; brighter pixels become 255, the
        others 0.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = int(255 * threshold)

    def __call__(self, image: Image.Image) -> Image.Image:
        """Binarize a PIL image.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image of any mode.

        Returns
        -------
        PIL.Image.Image
            Binary image of mode ``'L'`` with values 0 and 255.
        """
        image = image.convert("L")
        image = image.point(lambda p: 255 if p > self.threshold else 0)
        return image


class Dilate:
    """Transform that dilates (or erodes) the bright regions by a random square kernel.

    The kernel size is drawn from ``numpy.random`` for every call.

    Parameters
    ----------
    ksize_range : tuple of int
        Inclusive range ``(min, max)`` of the kernel size in pixels; 0 leaves the image
        unchanged.
    inv : bool, default False
        If True, apply a minimum filter (erode the bright regions, grow the dark ones)
        instead of a maximum filter.
    """

    def __init__(self, ksize_range: tuple[int, int], inv: bool = False) -> None:
        self.ksize_range = ksize_range
        self.inv = inv

    def __call__(self, image: Image.Image) -> Image.Image:
        """Filter a grayscale PIL image.

        Parameters
        ----------
        image : PIL.Image.Image
            Grayscale image of mode ``'L'``.

        Returns
        -------
        PIL.Image.Image
            Filtered image (the input itself if the kernel size is 0).
        """
        ksize = np.random.randint(self.ksize_range[0], self.ksize_range[1] + 1)
        if ksize > 0:
            if self.inv:
                tmp = minimum_filter(np.asarray(image), ksize, mode="constant", cval=255)
            else:
                tmp = maximum_filter(np.asarray(image), ksize, mode="constant", cval=0)
            image = Image.fromarray(tmp)

        return image


class IrregularMaskDataset(Dataset):
    """Masks of missing detector pixels from the NVIDIA Irregular Mask Dataset.

    Each mask is binarized; for training, its missing regions are also grown by a random
    minimum filter, and it is randomly rotated and cropped to 512 x 512. Masks with less than
    half valid pixels are replaced by all-valid masks. A random central rectangle (half sizes
    8 to 31 pixels, offset -8 to 7 pixels) is always masked out, as for a beam stop.

    Parameters
    ----------
    root : str
        Directory containing ``irregular-mask/``: training masks in
        ``irregular-mask/irregular_mask/disocclusion_img_mask/``, test masks (512 x 512) in
        ``irregular-mask/mask/testing_mask_dataset/``.
    train : bool, default True
        Use the training masks with augmentation, or the test masks.
    """

    def __init__(self, root: str, train: bool = True) -> None:
        self.train = train
        if self.train:
            path = "irregular-mask/irregular_mask/disocclusion_img_mask/"
        else:
            path = "irregular-mask/mask/testing_mask_dataset/"
        path = os.path.join(root, path)

        self.flist = [
            os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(".png")
        ]

        if self.train:
            self.transform = transforms.Compose(
                [
                    Binarize(0.6),
                    Dilate((9, 49), True),
                    transforms.RandomAffine(90, fill=1),
                    transforms.RandomCrop(512),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    Binarize(0.6),
                    transforms.ToTensor(),
                ]
            )

    def __len__(self) -> int:
        """Return the number of mask images."""
        return len(self.flist)

    def __getitem__(self, idx: int) -> Tensor:
        """Load and augment one mask.

        Parameters
        ----------
        idx : int
            Index of the mask image.

        Returns
        -------
        torch.Tensor
            Float32 tensor of shape ``(1, 512, 512)``: 1 for valid, 0 for missing pixels.
        """
        image = Image.open(self.flist[idx])

        if self.transform is not None:
            image = self.transform(image)

        if not self.train:
            image = 1 - image

        # Additional processing
        image = torch.gt(image, 0.6).type(torch.float)
        thr = 0.5
        if torch.mean(image) < thr:
            image[:] = 1

        si, sj = image.size()[-2:]
        ri, rj = np.random.randint(8, 32, size=2)
        ti, tj = np.random.randint(-8, 8, size=2)
        image[:, si // 2 - ri + ti : si // 2 + ri + ti, sj // 2 - rj + tj : sj // 2 + rj + tj] = 0

        return image


def GenerateDiffraction(
    obj: Tensor,
    ph_ord: float = 6,
    l_coh: float = 200,
    false_scale: bool = False,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, Tensor]:
    """Simulate noisy single-pulse diffraction patterns of objects.

    The object is zero-padded to 512 x 512 and its diffraction intensity is blurred by
    partial spatial coherence (Gaussian Schell model: the autocorrelation is multiplied by
    ``exp(-r**2 / (2 * l)**2)`` with ``l`` equal to ``l_coh`` times a random factor in
    ``[0.9, 1.1)``; temporal coherence is negligible for XFELs). The intensity is scaled to a
    random total photon count in ``[1, 10) * 10**ph_ord``, and Poisson noise followed by
    Gaussian noise with a FWHM of 1 photon is added. Random numbers come from the global
    PyTorch generator.

    Parameters
    ----------
    obj : torch.Tensor
        Real objects of shape ``(N, 1, 64, 64)``, on ``device``.
    ph_ord : float, default 6
        Order of magnitude of the total photon count.
    l_coh : float, default 200
        Coherence length in pixels of the real-space grid.
    false_scale : bool, default False
        If True, scale the returned object by the intensity scale factor instead of its
        square root (the convention of the training targets).
    device : str or torch.device, default 'cpu'
        Device of ``obj`` and of the generated data.

    Returns
    -------
    inten : torch.Tensor
        Noisy intensity of shape ``(N, 1, 512, 512)``, fftshifted, in photon counts.
    obj : torch.Tensor
        Scaled object of shape ``(N, 1, 64, 64)``.
    """
    inten = torch.abs(_fft2(F.pad(obj, (224, 224, 224, 224)))) ** 2

    # Gaussian Schell-model (spatial coherence)
    # Note that temporal coherence is ignorable for XFEL
    # Unit of coherence length (l_coh) is in pixel
    ls = torch.linspace(-256, 255, steps=512)
    m = torch.meshgrid(ls, ls, indexing="ij")
    l_sq = m[0] ** 2 + m[1] ** 2
    l_sq = l_sq[None, None, ...].to(device)
    sig_mu = l_coh * (
        0.9 + 0.2 * torch.rand(inten.size(0), 1, 1, 1, device=device)
    )  # 10% deviation
    kernel = torch.exp(-l_sq / (2 * sig_mu) ** 2)
    inten = torch.abs(_fft2(_ifft2(inten) * kernel))

    # Rescale by total photon count
    flux = 10**ph_ord * (1 + 9 * torch.rand(inten.size(0), 1, 1, 1, device=device))
    scale = flux / torch.sum(inten, dim=(-2, -1), keepdim=True)

    inten = inten * scale
    if false_scale:
        obj = obj * scale
    else:
        obj = obj * torch.sqrt(scale)

    # Poisson & Gaussian noise
    sig = 1 / 2.35482  # giving FWHM = 1
    inten = torch.normal(torch.poisson(torch.clamp(inten, min=0)), sig)

    return inten, obj


class CustomDataset(Dataset):
    """Dataset of diffraction patterns stored in an HDF5 file by ``generate_dataset.py``.

    The file holds the datasets ``input`` (float32, ``(M, 1, 512, 512)``), ``target``
    (float32, ``(M, 1, 64, 64)``) and ``mask`` (bool, ``(M, 1, 512, 512)``). The file is
    opened for every item, so the dataset works with several DataLoader workers.

    Parameters
    ----------
    h5path : str
        Path of the HDF5 file.
    """

    def __init__(self, h5path: str) -> None:
        self.h5path = h5path
        with h5py.File(self.h5path, "r") as f:
            self.length = len(f["input"])

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.length

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Read one sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        input : torch.Tensor
            Intensity of shape ``(1, 512, 512)``.
        target : torch.Tensor
            Object of shape ``(1, 64, 64)``.
        mask : torch.Tensor
            Bool tensor of shape ``(1, 512, 512)``, True for valid pixels.
        """
        with h5py.File(self.h5path, "r") as f:
            input = f["input"][idx]
            target = f["target"][idx]
            mask = f["mask"][idx]

        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        mask = torch.from_numpy(mask)

        return input, target, mask
