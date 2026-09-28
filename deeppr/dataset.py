import os
import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from .network import _fft2, _ifft2


class Binarize(object):
    def __init__(self, threshold):
        self.threshold = int(255 * threshold)

    def __call__(self, image):
        image = image.convert('L')
        image = image.point(lambda p: 255 if p > self.threshold else 0)
        return image


class Dilate(object):
    def __init__(self, ksize_range, inv=False):
        self.ksize_range = ksize_range
        self.inv = inv

    def __call__(self, image):
        ksize = np.random.randint(self.ksize_range[0], self.ksize_range[1] + 1)
        if ksize > 0:
            if self.inv:
                tmp = minimum_filter(np.asarray(image), ksize, mode='constant', cval=255)
            else:
                tmp = maximum_filter(np.asarray(image), ksize, mode='constant', cval=0)
            image = Image.fromarray(tmp)
        
        return image


class IrregularMaskDataset(Dataset):
    def __init__(self, root, train=True):
        self.train = train
        if self.train:
            path = 'irregular-mask/irregular_mask/disocclusion_img_mask/'
        else:
            path = 'irregular-mask/mask/testing_mask_dataset/'
        path = os.path.join(root, path)
        
        self.flist = [os.path.join(path, fname)
                      for fname in os.listdir(path) if fname.endswith('.png')]
        
        if self.train:
            self.transform = transforms.Compose([Binarize(0.6),
                                                 Dilate((9, 49), True),
                                                 transforms.RandomAffine(90, fill=1),
                                                 transforms.RandomCrop(512),
                                                 transforms.ToTensor(),
                                                 ])
        else:
            self.transform = transforms.Compose([Binarize(0.6),
                                                 transforms.ToTensor(),
                                                 ])

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
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
        image[:, si//2-ri+ti:si//2+ri+ti, sj//2-rj+tj:sj//2+rj+tj] = 0

        return image


def GenerateDiffraction(obj, ph_ord=6, l_coh=200, false_scale=False, device='cpu'):
    inten = torch.abs(_fft2(F.pad(obj, (224, 224, 224, 224)))) ** 2
    
    # Gaussian Schell-model (spatial coherence)
    # Note that temporal coherence is ignorable for XFEL
    # Unit of coherence length (l_coh) is in pixel
    ls = torch.linspace(-256, 255, steps=512)
    m = torch.meshgrid(ls, ls, indexing='ij')
    l_sq = m[0] ** 2 + m[1] ** 2
    l_sq = l_sq[None, None, ...].to(device)
    sig_mu = l_coh * (0.9 + 0.2 * torch.rand(inten.size(0), 1, 1, 1, device=device)) # 10% deviation
    kernel = torch.exp(-l_sq / (2 * sig_mu) ** 2)
    inten = torch.abs(_fft2(_ifft2(inten) * kernel))
    
    # Rescale by total photon count
    flux = 10 ** ph_ord * (1 + 9 * torch.rand(inten.size(0), 1, 1, 1, device=device))
    scale = flux / torch.sum(inten, dim=(-2, -1), keepdim=True)
    
    inten = inten * scale
    if false_scale:
        obj = obj * scale
    else:
        obj = obj * torch.sqrt(scale)
    
    # Poisson & Gaussian noise
    sig = 1 / 2.35482 # giving FWHM = 1
    inten = torch.normal(torch.poisson(torch.clamp(inten, min=0)), sig)
    
    return inten, obj


class CustomDataset(Dataset):
    def __init__(self, h5path):
        self.h5path = h5path
        with h5py.File(self.h5path, 'r') as f:
            self.length = len(f['input'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5path, 'r') as f:
            input = f['input'][idx]
            target = f['target'][idx]
            mask = f['mask'][idx]
            
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        mask = torch.from_numpy(mask)
        
        return input, target, mask