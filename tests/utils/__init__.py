import copy
import matplotlib.pyplot as plt
import numpy as np
from   PIL import Image
import torch

from nonlinear_kernels.lib.dataloading import transform_16bit_single_channel

DEV_BATCH_INPUTS = [
    ("/scratch/ainaz/replica-google-objects/office_1/6/", 29, 45),
    ("/scratch/ainaz/replica-google-objects/frl_apartment_5/15", 1080, 0)
]

def equal_except_for_nan(x, y):
    x = x.detach().clone()
    y = y.detach().clone()
    isnan = (x != x)
    isnan[y != y] = 1.0
    return torch.allclose(x[~isnan], y[~isnan])

def show(x, unnormalize=True, figsize=None):
    '''
      Concatenates and displays a bunch of imags 
    '''
    if figsize is not None:
        plt.figure(figsize=figsize)
    if unnormalize:
        x = x * 255
    plt.imshow(kornia.tensor_to_image(x.byte()))
    plt.show()

def load_dev_batch(inputs=DEV_BATCH_INPUTS, mask_depth_under=0.007):
    x_stacked, d_stacked, mask_stacked, n_stacked, f_stacked = [], [], [], [], []
    for source_dir, pn, vn in inputs:
        x = Image.open(f'{source_dir}/rgb/point_{pn}_view_{vn}_domain_rgb.png')
        x = torch.from_numpy(np.asarray(x)).float().permute(2, 0, 1).unsqueeze_(0) / 255.
        d = Image.open(f'{source_dir}/depth_euclidean/point_{pn}_view_{vn}_domain_depth_euclidean.png')
        d = transform_16bit_single_channel(d).float()
        
        n = Image.open(f'{source_dir}/normal/point_{pn}_view_{vn}_domain_normal.png')
        n = torch.from_numpy(np.asarray(n)).float().permute(2, 0, 1).unsqueeze_(0) / 255.

        f = np.load(f'{source_dir}/fragments/point_{pn}_view_{vn}_domain_fragments.npy')
        f = torch.from_numpy(np.asarray(f)).long().unsqueeze_(0)

        mask = torch.zeros_like(d)
        mask[d < mask_depth_under] = 1.0

        x_stacked.append(x)
        d_stacked.append(d)
        n_stacked.append(n)
        f_stacked.append(f)
        mask_stacked.append(mask)

    x_stacked = torch.cat(x_stacked, dim=0).cuda()
    d_stacked = torch.cat(d_stacked, dim=0).cuda()
    n_stacked = torch.cat(n_stacked, dim=0).cuda()
    f_stacked = torch.cat(f_stacked, dim=0).cuda()
    mask_stacked = torch.cat(mask_stacked, dim=0).cuda()
    
    return {
        'fragments': f_stacked, 
        'rgb': x_stacked, 
        'depth': d_stacked,
        'normal': n_stacked,
    }


