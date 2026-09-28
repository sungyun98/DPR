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
`requirements.txt` installs from GitHub. Minimum versions: Python 3.10, PyTorch 2.1,
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

4. We imported following codes.

    > Partial Convolution from https://github.com/NVIDIA/partialconv
    > 
    > Fast Fourier Convolution (FFC) from https://github.com/pkumivision/FFC
    > 
    > FFC ResNet Block from https://github.com/advimman/lama
    > 
    > Partial Convolution from https://github.com/NVIDIA/partialconv
    > 
    > Adaptive Sharpness-Aware Minimization (ASAM) from https://github.com/SamsungLabs/ASAM
    > 

5. When using DPR or weighted partial convolution, please cite our paper with proper references.

    > https://doi.org/10.1038/s41524-025-01569-7
    > 

6. Contact: Sung Yun Lee, sungyun98@postech.ac.kr
