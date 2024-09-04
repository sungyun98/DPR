# Deep Phase Retrieval (DPR)
DPR is a deep neural network for direct phase retrievals of single-particle diffraction patterns from single-pulse coherent diffraction imaging experiments using X-ray free electron lasers.

## Requirements

> Python = 3.11.5
> 
> NumPy = 1.26.0
> 
> SciPy = 1.11.3
> 
> scikit-image = 0.20.0
> 
> PIL = 10.0.1
> 
> h5py = 3.9.0
> 
> PyTorch = 2.1.0
> 
> Torchvision = 0.16.0
> 
> CUDA = 11.8
> 
> cuDNN = 8.7.0
> 

## Notes

1. The pretrained network is for the diffraction patterns with oversampling ratios in the range of 10 to 20 and total diffracted intensities in the range of 10<sup>6</sup> to 10<sup>7</sup>.

2. Coefficients for the loss function might require to be adjusted for training datasets with different conditions.

3. We used NVIDIA Irregular Mask Dataset from https://research.nvidia.com/labs/adlr/publication/partialconv-inpainting. Please check the file path in 'module.dataset.IrregularMaskDataset' when using 'generate_dataset.ipynb'. Other datasets, EMNIST and CIFAR-100, are from torchvision library.

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

6. Contact: Sung Yun Lee, sungyun98@postech.ac.kr
