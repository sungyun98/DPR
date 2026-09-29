"""Generate the training, validation and test datasets of DPR.

Each sample is a simulated single-pulse diffraction pattern of a random object:

* the shape of the object is an EMNIST character (balanced split), binarized, dilated,
  randomly rotated and scaled, and cropped to 64 x 64 pixels (the scaling sets the size of the
  objects, and hence the oversampling ratio, to which the network is sensitive);
* its density is a CIFAR-100 image, in grayscale, randomly cropped and resized to 64 x 64;
* the pattern is simulated with `deeppr.GenerateDiffraction` (partial coherence of 200 pixels,
  1e6 to 1e7 photons, Poisson and Gaussian noise), and the missing pixels come from the NVIDIA
  Irregular Mask Dataset (`deeppr.IrregularMaskDataset`) with a random beam stop.

The training set uses the training splits of the three datasets. The validation and test sets
use their test splits: with the same seed, the test set continues after the samples of the
validation set, so that the two do not overlap.

The HDF5 files hold the datasets ``input`` (float32, ``(N, 1, 512, 512)``), ``target``
(float32, ``(N, 1, 64, 64)``) and ``mask`` (bool, ``(N, 1, 512, 512)``), read by
`deeppr.CustomDataset` in ``train.py``.

Usage::

    python generate_dataset.py train   # ./datasets/dataset_train_n96k.h5
    python generate_dataset.py valid   # ./datasets/dataset_valid_n12k.h5
    python generate_dataset.py test    # ./datasets/dataset_test_n12k.h5

EMNIST and CIFAR-100 are downloaded by torchvision into ``--root``; the irregular masks must be
in ``--root`` as described in `deeppr.IrregularMaskDataset`.
"""

import argparse
import os
from collections.abc import Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100, EMNIST
from tqdm import tqdm

from deeppr import Binarize, Dilate, GenerateDiffraction, IrregularMaskDataset

SIZES = {"train": 96000, "valid": 12000, "test": 12000}


def build_datasets(root: str, train: bool) -> tuple[Dataset, Dataset, Dataset]:
    """Create the shape (EMNIST), density (CIFAR-100) and mask datasets.

    Parameters
    ----------
    root : str
        Directory of the datasets (EMNIST and CIFAR-100 are downloaded there if needed).
    train : bool
        Use the training splits, otherwise the test splits.

    Returns
    -------
    shape, density, mask : torch.utils.data.Dataset
        Shapes and densities give ``(image, label)``, of shape ``(1, 64, 64)``; masks give
        ``(1, 512, 512)`` tensors.
    """
    shape = EMNIST(
        root=root,
        split="balanced",
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                Binarize(0.1),
                Dilate((3, 7), False),
                transforms.Pad((18, 18), fill=0),  # EMNIST images are 28 x 28
                # the scale sets the size of the objects, i.e. the oversampling ratio
                transforms.RandomAffine(90, scale=(0.8, 1.5), fill=0),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        ),
    )
    density = CIFAR100(
        root=root,
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.RandomResizedCrop(64, antialias=True),
                transforms.ToTensor(),
            ]
        ),
    )
    mask = IrregularMaskDataset(root=root, train=train)
    return shape, density, mask


def _batches(loader: DataLoader) -> Iterator:
    """Yield the batches of ``loader`` endlessly, reshuffling at the start of every pass."""
    while True:
        yield from loader


def generate(
    datasets: tuple[Dataset, Dataset, Dataset],
    path: str,
    size: int,
    batch_size: int = 48,
    seed: int = 0,
    skip: int = 0,
    device: str | torch.device = "cpu",
    workers: int = 12,
    overwrite: bool = False,
) -> None:
    """Simulate ``size`` samples and write them to an HDF5 file.

    Parameters
    ----------
    datasets : tuple of torch.utils.data.Dataset
        Shape, density and mask datasets from `build_datasets`.
    path : str
        Output HDF5 file.
    size : int
        Number of samples, a multiple of ``batch_size``.
    batch_size : int, default 48
        Samples simulated at once.
    seed : int, default 0
        Seed of the shuffling, of the augmentations and of the simulation.
    skip : int, default 0
        Number of samples drawn and discarded first (the test set skips the validation set).
    device : str or torch.device, default 'cpu'
        Device of the simulation.
    workers : int, default 12
        Worker processes of each DataLoader.
    overwrite : bool, default False
        Replace ``path`` if it exists (otherwise an existing file is an error).

    Raises
    ------
    ValueError
        If ``size`` or ``skip`` is not a multiple of ``batch_size``.
    """
    if size % batch_size or skip % batch_size:
        raise ValueError(f"size and skip must be multiples of batch_size ({batch_size}).")
    torch.manual_seed(1 + seed)
    loaders = []
    for offset, dataset in enumerate(datasets, start=2):
        generator = torch.Generator().manual_seed(offset + seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
            generator=generator,
        )
        loaders.append(_batches(loader))
    shapes, densities, masks = loaders

    for _ in tqdm(range(skip // batch_size), desc="skipping"):
        next(shapes), next(densities), next(masks)

    with h5py.File(path, "w" if overwrite else "w-") as f:
        f.create_dataset("input", (size, 1, 512, 512), dtype=np.float32)
        f.create_dataset("target", (size, 1, 64, 64), dtype=np.float32)
        f.create_dataset("mask", (size, 1, 512, 512), dtype=np.bool_)

    for start in tqdm(range(0, size, batch_size), desc=os.path.basename(path)):
        # the loaders fork new workers at every pass through their dataset: draw the batches
        # while the file is closed, so that the workers do not inherit it
        shape, _ = next(shapes)
        density, _ = next(densities)
        mask = next(masks)

        # objects that vanish (e.g. a black crop of the density) get a uniform density
        empty = torch.sum(shape * density, dim=(-2, -1)) == 0
        if torch.any(empty):
            density[torch.nonzero(empty, as_tuple=True)[0]] = 1

        obj = (shape * density).to(device)
        pattern, target = GenerateDiffraction(
            obj, ph_ord=6, l_coh=200, false_scale=True, device=device
        )
        batch = slice(start, start + batch_size)
        with h5py.File(path, "r+") as f:
            f["input"][batch] = pattern.cpu().numpy()
            f["target"][batch] = target.cpu().numpy()
            f["mask"][batch] = (mask > 0).numpy()


def main() -> None:
    """Parse the command line and generate one dataset."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("split", choices=list(SIZES), help="dataset to generate")
    parser.add_argument("--root", default="./datasets", help="directory of the datasets")
    parser.add_argument("--size", type=int, help="samples (default: 96000 for train, else 12000)")
    parser.add_argument(
        "--valid-size", type=int, default=SIZES["valid"], help="validation samples skipped by test"
    )
    parser.add_argument("--batch-size", type=int, default=48, help="samples simulated at once")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--workers", type=int, default=12, help="DataLoader workers per dataset")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing file")
    args = parser.parse_args()

    size = args.size or SIZES[args.split]
    path = os.path.join(args.root, f"dataset_{args.split}_n{size // 1000}k.h5")
    datasets = build_datasets(args.root, train=args.split == "train")
    skip = args.valid_size if args.split == "test" else 0
    generate(
        datasets,
        path,
        size,
        args.batch_size,
        args.seed,
        skip,
        args.device,
        args.workers,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
