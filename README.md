# Deep Phase Retrieval (DPR)

> **Package renamed:** the package folder `module` is now `deeppr`. Please replace
> `from module import ...` with `from deeppr import ...`. The phase retrieval algorithms are no
> longer bundled with DPR; they come from the
> [PhaseRetrieval](https://github.com/sungyun98/PhaseRetrieval) package
> (`from phaseretrieval import PhaseRetrieval`). We apologize for the inconvenience to existing
> users. The original code remains available at the tag `v1.0-legacy`.

DPR is a deep neural network for direct phase retrievals of single-particle diffraction patterns from single-pulse coherent diffraction imaging experiments using X-ray free electron lasers.

## Requirements

Tested with Python 3.12, PyTorch 2.14.0 and torchvision 0.29.0 (CUDA 12.6 builds), NumPy 2.5.3,
SciPy 1.18.1, scikit-image 0.26.0, Pillow 12.3.0 and h5py 3.16.0; the exact versions are listed
in `requirements.txt`. The phase retrieval algorithms come from the
[PhaseRetrieval](https://github.com/sungyun98/PhaseRetrieval) package (`phaseretrieval`), which
`requirements.txt` installs from the default branch on GitHub. Minimum versions: Python 3.10, PyTorch 2.1,
torchvision 0.16, NumPy 1.26, SciPy 1.11, scikit-image 0.22.

```bash
conda env create -f environment.yml
# or, in an existing environment:
pip install -r requirements.txt && pip install -e .
```

The PyTorch builds in `requirements.txt` use CUDA 12.6 and run with NVIDIA drivers 525 or newer;
replace `cu126` with `cpu` for a CPU-only installation. The original code and the library
versions used for the paper (Python 3.11.5, PyTorch 2.1.0, CUDA 11.8) are available at the tag
`v1.0-legacy`.

## Notes

1. The pretrained network is for the diffraction patterns with oversampling ratios in the range of 10 to 20 along each axis and total diffracted intensities in the range of 10<sup>6</sup> to 10<sup>7</sup>.

2. Coefficients for the loss function might require to be adjusted for training datasets with different conditions.

3. We used NVIDIA Irregular Mask Dataset from https://research.nvidia.com/labs/adlr/publication/partialconv-inpainting. Please check the file paths in 'deeppr.dataset.IrregularMaskDataset' when using 'generate_dataset.ipynb'. Other datasets, EMNIST and CIFAR-100, are from torchvision library.

4. Third-party code included in `deeppr` is listed in [License](#license).

5. When using DPR or weighted partial convolution, please cite our paper with proper references.

    > https://doi.org/10.1038/s41524-025-01569-7
    > 

6. Contact: Sung Yun Lee, sungyun98@g.postech.edu

## License

This code is released under the BSD 2-Clause License (`LICENSE.txt`), except for the
third-party code below, which keeps its original license (full text in `LICENSES/`); the
changes made are stated at the top of each file.

| File | Source | License |
|---|---|---|
| `deeppr/weightedpartialconv2d.py` | derived from [NVIDIA/partialconv](https://github.com/NVIDIA/partialconv) `models/partialconv2d.py` | BSD 3-Clause, Copyright (c) 2018 NVIDIA Corporation |
| `deeppr/loss.py` (`VGG19Partial`, `gram_matrix`) | adapted from [NVIDIA/partialconv](https://github.com/NVIDIA/partialconv) `models/loss.py` | BSD 3-Clause, Copyright (c) 2019 NVIDIA Corporation |
| `deeppr/ffc.py` (FFC layers) | modified from [pkumivision/FFC](https://github.com/pkumivision/FFC) `model_zoo/ffc.py` | Apache 2.0, Copyright 2022 Lu Chi |
| `deeppr/ffc.py` (`FFCResNetBlock`, `ConcatTupleLayer`) | modified from [advimman/lama](https://github.com/advimman/lama) `saicinpainting/training/modules/ffc.py` | Apache 2.0, Copyright 2021 Samsung Research |
| `deeppr/asam.py` | copied from SamsungLabs/ASAM `asam.py` (repository no longer online; [Software Heritage archive](https://archive.softwareheritage.org/swh:1:rev:f156a680171db16d551c0d85cba2514fa3bff6a2)) | Apache 2.0, Copyright 2021 Samsung Research |

The pretrained VGG19 weights used by the perceptual loss are downloaded by torchvision at run
time and are not distributed with this repository.
