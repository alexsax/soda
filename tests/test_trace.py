from soda.tests.utils import equal_except_for_nan
from soda.examples.surface_normals_homomorphism import SurfaceNormalSymmetries

from nonlinear_kernels.lib.dataloading import transform_16bit_single_channel
import numpy as np
from PIL import Image
import torch 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

def load_gso_batch(n_stacks = 3):
    source_dir, pn, vn = "/scratch/ainaz/replica-google-objects/office_1/6/", 29, 45
    source_dir, pn, vn = "/scratch/ainaz/replica-google-objects/frl_apartment_5/15", 1080, 0

    x = Image.open(f'{source_dir}/rgb/point_{pn}_view_{vn}_domain_rgb.png')
    x = torch.from_numpy(np.asarray(x).copy()).float().permute(2, 0, 1).unsqueeze_(0) / 255.
    d = Image.open(f'{source_dir}/depth_euclidean/point_{pn}_view_{vn}_domain_depth_euclidean.png')
    d = transform_16bit_single_channel(np.asarray(d).copy()).float()
    mask = torch.zeros_like(d)
    mask[d < 0.007] = 1.0
    n = Image.open(f'{source_dir}/normal/point_{pn}_view_{vn}_domain_normal.png')
    n = torch.from_numpy(np.asarray(n).copy()).float().permute(2, 0, 1).unsqueeze_(0) / 255.
    
    x_stacked = torch.cat([x] * n_stacks, dim=0).to(device)
    d_stacked = torch.cat([d] * n_stacks, dim=0).to(device)
    mask_stacked = torch.cat([mask] * n_stacks, dim=0).to(device)
    n_stacked = torch.cat([n] * n_stacks, dim=0).to(device)

    return x_stacked, d_stacked, n_stacked, mask_stacked

def test_mask():
    normals_symmetries = SurfaceNormalSymmetries()
    monoids = normals_symmetries.source._generating_monoids

    x_stacked, _, _, mask_stacked = load_gso_batch()
    x_t = x_stacked.clone().detach().to(device)
    trace = None
    for monoid in monoids:
        print(monoid, x_t.shape)
        x_t, trace = monoid.random_action(x_t, trace)
    history_trace = trace

    mask_out, trace = normals_symmetries.action_on_mask(history_trace, mask_stacked.unsqueeze(1).clone().detach())
    mask_out, trace = normals_symmetries.inv_mask(history_trace, mask_out)
    assert equal_except_for_nan(mask_out, mask_stacked.unsqueeze(1))

