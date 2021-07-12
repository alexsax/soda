import copy
import kornia
import kornia.augmentation as K
from   kornia.geometry.transform.crop.crop2d import validate_bboxes
import torch
from   typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterable
from   packaging import version


from .algebra import MAct, GActMixin, Element


# Pair transform 
# Takes in two images and the corresponding fragments
# Chooses to use or not use the fragments
# youtube-dl -o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' https://www.youtube.com/playlist?list=PL2FF649D0C4407B30

class BiFunctor:

    def __init__(self, first, second):
        '''
            See BiMap in Haskell. 
        '''
        self.first = first
        self.second = second
    
    def __call__(self, a, b):
        '''
        '''
        raise NotImplementedError('This is not abstract, but it is also not implemented')


class Product(BiFunctor):
    def __call__(self, (a, b)):
        return (self.first(a), self.second(b))

class OpticalFlow(MAct, GActMixin):

    def __init__(self, p=1.0, same_on_batch=False):
        '''
            Assumes that 
            Args:

        '''
        self.p = p
        self.same_on_batch = same_on_batch

    def random_action(self, a, b):
        assert a.shape[0] == b.shape[0]
        batch_size = a.shape[0]
        m['batch_prob'] = torch.rand(batch_size) > self.p
        return m, self.action(m, (a, b))
    
    def action(self, m, x):
        assert isinstance(x, Iterable) and len(x) == 2
        a, b = x
        result = a.clone()
        result[m['batch_prob']] = b.clone()[m['batch_prob']]
        return (a, result)

    def inverse(self, m):
        


class Diagonal(MAct, GActMixin):
    def unit(self):
        return "IdentityElem"

    def random_action(self, x: Any):
        unit = self.unit()
        return unit, self.action(unit, x)

    def action(self, m: Element, x: Any):
        return (x, x) if m == self.unit() else x[0]
    
    def inverse(self, m: Element):
        return 'Inverse'




def aggregate_labels_putonly_noscaling_helper(logits_all, faces_uniq, faces_uniq_idx_flat, n_channels, device, scaling):
    face_labels = torch.zeros((len(faces_uniq), n_channels), dtype=torch.float, device=device)
    face_counts = torch.zeros((len(faces_uniq), n_channels), dtype=torch.float, device=device)
    flatter_logits = ein.rearrange(logits_all, 'b h w c -> (b h w) c')
    face_labels.index_put_(
        (faces_uniq_idx_flat,),
        flatter_logits,
        accumulate=True
    )
    if scaling is None: return face_labels
    
    face_counts.index_put_(
        (faces_uniq_idx_flat,),
        torch.tensor(1.0, device=device),
        accumulate=True
    ) 
    nonzero = face_counts.nonzero(as_tuple=True)
    face_labels[nonzero] /= face_counts[nonzero]
    return face_labels

def aggregate_labels_putonly(fragments_all, logits_all, scaling=torch.sqrt, parallel_apply=True):
    '''
        fragments_all: N_VIEWS x W x H
        logits_all:    N_VIEWS x W x H x C
        scaling: Union[str, Callable]. 
            Default options are 'PER_VIEW', 'PER_PIXEL'
            Otherwise, calls scaling(x) -> x'

    '''
    # type: (List[Tensor],List[Tensor]) -> Tensor
    # Maybe? This only works for a single image. Need to make it work for batches. 

    device = fragments_all[0].device
    n_views, h, w, n_channels = logits_all.shape

    assert logits_all.shape[1] == logits_all.shape[2], f"predictions should be shape B x W x H x C, not {logits_all.shape}. soft_combine only handles square predictions."

    faces_uniq, faces_uniq_idx = fragments_all.unique(return_inverse=True)
    faces_uniq_idx_batched = ein.rearrange(faces_uniq_idx, 'b h w -> b (h w)').unsqueeze(-1).expand(-1, -1, n_channels)

    one = torch.tensor(1.0, device=device)
    face_labels_per_im = torch.zeros((n_views, len(faces_uniq), n_channels), dtype=torch.float, device=device)
    face_counts_per_im = torch.zeros((n_views, len(faces_uniq), n_channels), dtype=torch.float, device=device)

    # Aggregation
    flatter_logits = ein.rearrange(logits_all, 'b h w c -> b (h w) c')
    face_labels_per_im.scatter_add_(dim=1,
                            index=faces_uniq_idx_batched,
                            src=flatter_logits) # tensor[indices] += values
    # Normalization
    if scaling is not None:
        flat_ones = one.expand(n_views,h*w,n_channels)
        face_counts_per_im.scatter_add_(dim=1, index=faces_uniq_idx_batched,
                            src=flat_ones) # tensor[indices] += values
        nonzero = face_counts_per_im.nonzero(as_tuple=True)
        face_labels_per_im[nonzero] /= scaling(face_counts_per_im[nonzero]) 

    face_labels = face_labels_per_im.sum(dim=0)

    return face_labels[faces_uniq_idx].reshape(logits_all.shape)



def compute_cooccurrence(fragments, return_keys=None):
    '''
        fragments: N_VIEWS x W x H
        return_keys: Which keys to return
    '''
    # type: (List[Tensor],List[Tensor]) -> Tensor
    # Maybe? This only works for a single image. Need to make it work for batches. 
    if return_keys is None:
        return_keys = ['shared', 'masks', 'valid_and_shared_prop', 'valid_prop']
    device = fragments.device
    n_views, h, w = fragments.shape

    faces_uniq, faces_uniq_idx = fragments.unique(return_inverse=True)
    
    flat_ones    = torch.ones([n_views, h*w], dtype=torch.int32, device=device)
    total_counts = torch.zeros([n_views, len(faces_uniq)], dtype=torch.int32, device=device)
    total_counts.scatter_add_(dim=1,
                              index=faces_uniq_idx.reshape(n_views, -1),
                              src=flat_ones)
    n_views_visible = total_counts.clamp(max=1).sum(dim=0)
    target_predictions = n_views_visible[faces_uniq_idx.flatten()].reshape(fragments.shape)

    mask = (fragments != 0).to(fragments.device)    
    
    batch_size = len(fragments)
    result = target_predictions.squeeze(-1) - 1
    result[result > 0] = 1
    result = result * mask
    valid_count = mask.view(batch_size, -1).sum(dim=-1)
    props = result.view(batch_size, -1).sum(dim=-1) / (mask.numel() / batch_size)
    
    res = {
        k: v for k, v in dict(
            shared=result,
            masks=mask,
            valid_and_shared_prop=props,
            valid_prop=valid_count / mask.numel(),
        ).items()
        if k in return_keys
    }
    
    return res
