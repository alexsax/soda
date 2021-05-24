import copy
import kornia.augmentation as K
from   kornia.geometry.transform.crop.crop2d import validate_bboxes
import torch

from .algebra import MAct, GActMixin
from .trace import Trace

class KorniaMAct(MAct):
    
    def __init__(self, aug, quicktest=True):
        '''
            quicktest: Generate a few random actions (m) and make sure action(m, x) matches the expected output.
                This helps make sure that we don't have weird interactions with Kornia.
        '''
        super().__init__()
        self.aug = aug
        if self.aug.return_transform:
            raise ValueError(f"Augmentation {self.aug} must use 'return_transform' == False!")
        quicktest = False
        if quicktest:
            self.quicktest()
        # add in a sample test case?

    def quicktest(self, n_times=4):
        inputs = torch.tensor([[[0., 1., 2.],
                        [3., 4., 5.],
                         [6., 7., 8.]]]).unsqueeze(0)
        for _ in range(n_times):
            x = self.aug(inputs)
            x2 = self.aug(inputs, self.aug._params)
            assert torch.allclose(x, x2)
        return True
    
    def __repr__(self):
        return f'{type(self).__name__}.{type(self.aug).__name__}'

        
    def random_action(self, x, trace=None):
        m = self.aug.forward_parameters(x.shape)
        return self.action(m, x, trace)
    
    def action(self, m, x, trace=None):        
        if trace is None:
            trace = Trace()
#         x = self.aug(x, m)
        x = self.aug(x, m)
        cache = {'__name__': f'{self.__repr__()}', 'param': m}
        cache = self._ensure_wrap_type(cache)
        return x, trace.push(cache)


##### Crops
def bbox_transform(source: torch.Tensor, boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert 2D bounding boxes to masks. Covered area is 1. and the remaining is 0.

    Args:
        boxes (torch.Tensor): a tensor containing the coordinates of the bounding boxes to be extracted.
            The tensor must have the shape of Bx4x2, where each box is defined in the following (clockwise)
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in the x, y order.
        width (int): width of the masked image.
        height (int): height of the masked image.

    Returns:
        torch.Tensor: the output mask tensor.

    Note:
        It is currently non-differentiable.

    Examples:
        >>> boxes = torch.tensor([[
        ...        [1., 1.],
        ...        [3., 1.],
        ...        [3., 2.],
        ...        [1., 2.],
        ...   ]])  # 1x4x2
        >>> bbox_to_mask(boxes, 5, 5)
        tensor([[[0., 0., 0., 0., 0.],
                 [0., 1., 1., 1., 0.],
                 [0., 1., 1., 1., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]]])
    
    For more:
        See https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/transform/crop/crop2d.html
    """
    validate_bboxes(boxes)
    # zero padding the surroudings
    B, C, _, _ =  source.shape
    new_size = (B, C, height + 2, width + 2)
    target = torch.full(new_size, float('nan'), dtype=source.dtype, device=source.device, requires_grad=source.requires_grad)
    # push all points one pixel off
    # in order to zero-out the fully filled rows or columns
    boxes += 1

    mask_out = []
    # TODO: Looking for a vectorized way
    for s, t, box in zip(source, target, boxes):
        box = box.long()
        start_x, end_x = box[0, 0].item(), box[1, 0].item() + 1
        start_y, end_y = box[1, 1].item(), box[2, 1].item() + 1
        t[:, start_y:end_y, start_x:end_x] = s.clone()
    target = target[..., 1:-1, 1:-1]
    return target



class FixedSizeCrop(KorniaMAct, GActMixin):
    def __init__(quicktest=True, **kwargs):
        super().__init__(
            aug = K.RandomCrop(**kwargs),
            quicktest=quicktest)
    
    def action(self, m, x, trace=None):
        if 'inv' in m and m['inv']:
            x_t, trace = self._inv_action(m, x, trace)
        else:
            if 'original_shape' not in m:  # Bind this action to inputs of a certain size.
                m['original_shape'] = x.shape
            assert m['original_shape'][-2:] == x.shape[-2:] and m['original_shape'][0] == x.shape[0], f"Trying to apply crop from source image of shape {m['original_shape']}, but source image is {x.shape}"
            x_t, trace = super().action(m, x, trace)
        return x_t, trace

    def _inv_action(self, m, x, trace=None):
        assert 'inv' in m and m['inv'], 'Calling _inv_action(m, x) with m that is not an inverse element'
        if trace is None:
            trace = Trace()

        B, C, height, width = m['original_shape']
        x_t = bbox_transform(x, m['dst'], height, width)

        cache = {'__name__': f'{self.__repr__()}', 'param': m}
        cache = self._ensure_wrap_type(cache)
        return x_t, trace.push(cache)
        
    def inv(self, m):
        m_inv = copy.deepcopy(m)
        m_inv['dst'] = m['src'].clone()
        m_inv['src'] = m['dst'].clone()
        if 'inv' not in m_inv:
            m_inv['inv'] = True
        else:
            m_inv['inv'] = not m_inv['inv']
        return m_inv

    

#### Flips ###################
class HFlip(KorniaMAct, GActMixin):
    def __init__(quicktest=True):
        super().__init__(
            aug       = K.RandomHorizontalFlip(return_transform=False, same_on_batch=False, p = 0.5, p_batch = 1.0),
            quicktest = quicktest
        )
    
    def inv(self, m):
        return m
    


class VFlip(KorniaMAct, GActMixin):
    def __init__(quicktest=True):
        super().__init__(
            aug       = K.RandomVerticalFlip(return_transform=False, same_on_batch=False, p = 0.5, p_batch = 1.0),
            quicktest = quicktest
        )
    
    def inv(self, m):
        return m

    

if __name__ == '__main__':
    import kornia
    x_ = data.astronaut() / 255.0
    x_rgb = kornia.image_to_tensor(x_).float().cuda().unsqueeze(0)
    x_rgb_stacked = torch.cat([x_rgb] * 5, dim=0)
    inputs = x_rgb_stacked
    monoid = FixedSizeCrop(size=(256, 256), p=1.0, return_transform=False)
    x_t, trace = monoid.random_action(inputs)
    reconstruction, trace_back = monoid.inverse_last_action(x_t, trace)
    assert len(trace_back.history) == 0
#     m = get(trace)
#     m_inv = monoid.inv(m)
#     reconstruction, trace = monoid.action(m_inv, x_t, trace)
    show(torch.nanmedian(reconstruction, dim=0)[0])