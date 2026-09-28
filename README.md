# Deep Phase Retrieval (DPR)

> **Package renamed:** the package folder `module` is now `deeppr`. Please replace
> `from module import ...` with `from deeppr import ...`. The phase retrieval algorithms are no
> longer bundled with DPR; they come from the
> [PhaseRetrieval](https://github.com/sungyun98/PhaseRetrieval) package
> (`from phaseretrieval import PhaseRetrieval`). We apologize for the inconvenience to existing
> users. The original code remains available at the tag `v1.0-legacy`.

DPR is a deep neural network for direct phase retrievals of single-particle diffraction patterns
from single-pulse coherent diffraction imaging experiments using X-ray free electron lasers. It
reconstructs a 64 x 64 object from a 512 x 512 diffraction pattern with missing pixels, and the
result can be refined by iterative phase retrieval.

## Installation

DPR depends on the [PhaseRetrieval](https://github.com/sungyun98/PhaseRetrieval) package
(`phaseretrieval`), which is not on PyPI; `requirements.txt` installs it from the default branch
on GitHub. Clone the repository (the pretrained weights and the demo data are in it) and create
the tested environment:

```bash
git clone https://github.com/sungyun98/DPR.git
cd DPR
conda env create -f environment.yml   # environment "deeppr"
# or, in an existing environment:
pip install -r requirements.txt && pip install -e .
```

To install only the packages into an environment that already has PyTorch and torchvision:

```bash
pip install git+https://github.com/sungyun98/PhaseRetrieval.git
pip install git+https://github.com/sungyun98/DPR.git
```

Tested with Python 3.12, PyTorch 2.14.0 and torchvision 0.29.0 (CUDA 12.6 builds), NumPy 2.5.3,
SciPy 1.18.1, scikit-image 0.26.0, Pillow 12.3.0 and h5py 3.16.0; the exact versions are in
`requirements.txt`. Minimum versions: Python 3.10, PyTorch 2.1, torchvision 0.16, NumPy 1.26,
SciPy 1.11, scikit-image 0.20. The PyTorch builds in `requirements.txt` use CUDA 12.6 and run with
NVIDIA drivers 525 or newer; replace `cu126` with `cpu` for a CPU-only installation.

## Usage

`demo.ipynb` reconstructs the measured pattern in `exp/`: it centres and crops the pattern,
runs the network, and refines the result with GPS-R from `phaseretrieval`. The network alone, on
a simulated pattern (run from the repository root):

```python
import torch

from deeppr import GenerateDiffraction, Network

model = Network(ngf=64, max_features=1024, weight_model=True, downsample_FFC=False, refinement=True)
checkpoint = torch.load("pretrained/param_dpr1.pt", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# simulated pattern of a 40 x 32 pixel object (oversampling ratios 12.8 and 16)
obj = torch.zeros(1, 1, 64, 64)
obj[..., 12:52, 16:48] = 1
intensity, target = GenerateDiffraction(obj, ph_ord=6)  # (1, 1, 512, 512), fftshifted
mask = torch.ones_like(intensity, dtype=torch.bool)  # True for valid pixels
mask[..., 244:268, 244:268] = False  # beam stop

with torch.no_grad():
    result = model(intensity * mask, mask)  # (1, 1, 64, 64)
```

Inputs are intensities in photon counts with the zero frequency at the centre, and the mask is
True (or 1) for valid pixels. `pretrained/` holds two checkpoints, `param_dpr0.pt` and
`param_dpr1.pt`; `demo.ipynb` uses `param_dpr1.pt`.

To train the network, generate the datasets with `generate_dataset.ipynb` and run `train.py` with
torchrun on one or more GPUs (the launch commands are in its docstring).

## Notes

1. The pretrained network is for the diffraction patterns with oversampling ratios in the range
   of 10 to 20 along each axis and total diffracted intensities in the range of
   10<sup>6</sup> to 10<sup>7</sup>.

2. Coefficients for the loss function might require to be adjusted for training datasets with
   different conditions.

3. We used NVIDIA Irregular Mask Dataset from
   https://research.nvidia.com/labs/adlr/publication/partialconv-inpainting. Please check the
   file paths in `deeppr.dataset.IrregularMaskDataset` when using `generate_dataset.ipynb`.
   Other datasets, EMNIST and CIFAR-100, are from torchvision library.

4. Third-party code included in `deeppr` is listed in [License](#license).

## Reproducing the paper results

The tag `v1.0-legacy` is the code used for the paper (Python 3.11.5, PyTorch 2.1.0, CUDA 11.8)
and reproduces its results:

```bash
git checkout v1.0-legacy
```

`tests/regression/environment-legacy.yml` describes a CPU environment with these library
versions. The regression tests in `tests/regression/` compare the current code with reference
outputs of `v1.0-legacy`, including the pretrained networks and the `demo.ipynb` pipeline (see
`tests/regression/README.md`). The results match within floating-point tolerance, except for
two options of the phase retrieval code that the demo does not use: the NLL error metric with
RAAR, which raised an IndexError, and ShrinkWrap, which now uses a centred Gaussian kernel.

## Citation

When using DPR or weighted partial convolution, please cite:

> S. Y. Lee, D. H. Cho, C. Jung, D. Sung, D. Nam, S. Kim, and C. Song, Deep-learning real-time
> phase retrieval of imperfect diffraction patterns from X-ray free-electron lasers,
> *npj Comput. Mater.* **11**, 68 (2025). <https://doi.org/10.1038/s41524-025-01569-7>

```bibtex
@article{lee2025npjcompumats,
  title   = {Deep-learning real-time phase retrieval of imperfect diffraction patterns from X-ray free-electron lasers},
  author  = {Lee, Sung Yun and Cho, Do Hyung and Jung, Chulho and Sung, Daeho and Nam, Daewoong and Kim, Sangsoo and Song, Changyong},
  journal = {npj Computational Materials},
  volume  = {11},
  pages   = {68},
  year    = {2025},
  doi     = {10.1038/s41524-025-01569-7}
}
```

For the phase retrieval algorithms, please also cite the references of
[PhaseRetrieval](https://github.com/sungyun98/PhaseRetrieval).

## Contact

Sung Yun Lee, sungyun98@g.postech.edu

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
