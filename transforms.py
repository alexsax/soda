from functools import partial
from typing import Optional, List, Union, Tuple
from . import functional as F
from .base import EquivariantTransform, TransformHistory
import torch
from math import ceil

class ResizeShortestEdge(EquivariantTransform):
    """Resize images

    Args:
        sizes (List[int]): Target to resize to (by spatial image dimensions)
        original_size (int): optional, image original size for deaugmenting mask
        interpolation (str): one of "nearest"/"linear" (see more in torch.nn.interpolate)
        align_corners (bool): see more in torch.nn.interpolate
    """

    def __init__(
        self,
        sizes: List[int],
        max_size: int = 9223372036854775807,
        interpolation: str = "nearest",
        align_corners: Optional[bool] = None,
    ):
        '''
        
        '''
        # if original_size is not None and original_size not in sizes:
        #     sizes = [original_size] + list(sizes)
        self.interpolation = interpolation
        self.align_corners = align_corners
        # self.original_size = original_size
        self.max_size = max_size
        super().__init__("size", sizes)

    def pre_transform(self, image, size, history=None, **kwargs):
        
        h, w = image.shape[-2:]
        
        # size: new size
        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = round(neww)
        newh = round(newh)
        # print(f'({h}, {w}) -> ({newh}, {neww})') 
        image = F.resize(
            image,
            (newh, neww),
            interpolation=self.interpolation,
            align_corners=self.align_corners,
        )

        cache = {'__name__': f'{type(self).__name__}.pre_transform', 'original_size': (h,w)}
        history = TransformHistory.push(cache, history)
        return image, history

    def pre_transform_inverse(self, image, size, history, **kwargs):
        '''
            image: (image, cache)
        '''
        cache, history = TransformHistory.pop(history)
        original_size = cache['original_size']
        # print(f'({image.shape}) -> ({original_size})')  
        image = F.resize(
            image,
            original_size,
            interpolation=self.interpolation,
            align_corners=self.align_corners,
        )
        return image, history

    def post_transform(self, label, size=1, history=None, **kwargs):
        return self.pre_transform(label, size, history, **kwargs)

    def post_transform_inverse(self, label, size=1, history=None, **kwargs):
        return self.pre_transform_inverse(label, size, history, **kwargs)

    def mask_transform(self, mask, size, history=None, **kwargs):
        return self.pre_transform(mask, size, history, **kwargs)

    def mask_transform_inverse(self, mask, size, history=None, **kwargs):
        return self.pre_transform_inverse(mask, size, history, **kwargs)


class HorizontalFlip(EquivariantTransform):
    """Flip images horizontally (left->right)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])


    def pre_transform(self, image, apply=False, history=None, **kwargs):
        if apply:
            image = F.hflip(image)
        return image, history

    def pre_transform_inverse(self, image, apply=False, history=None,**params):
        return self.pre_transform(image, apply, history=history) # self-inverse

    def post_transform(self, label, apply=False, history=None, **params):
        return self.pre_transform(label, apply, history=history) # E=e

    def post_transform_inverse(self, label, apply=False, history=None, **params):
        return self.post_transform(label, apply, history=history) # E^{-1}=e

    def mask_transform(self, mask, apply=False, history=None, **params):
        return self.pre_transform(mask, apply, history=history)

    def mask_transform_inverse(self, mask, apply=False, history=None, **params):
        return self.pre_transform(mask, apply, history=history)


class VerticalFlip(HorizontalFlip):

    def pre_transform(self, image, apply=False, history=None, **kwargs):
        if apply:
            image = F.vflip(image)
        return image, None

class SurfaceNormalHorizontalFlip(HorizontalFlip):

    def __init__(self, dim_horizontal=0):
        self.dim_horizontal = dim_horizontal 
        super().__init__()

    def post_transform(self, label, apply=False, history=None, **kwargs):
        if apply:
            label = F.hflip(label)
            label[:, self.dim_horizontal] *= -1
        return label, None


class Resize(EquivariantTransform):
    """Resize images

    Args:
        sizes (List[Tuple[int, int]): Target to resize to (by spatial image dimensions)
        original_size Tuple(int, int): optional, image original size for deaugmenting mask
        interpolation (str): one of "nearest"/"linear" (see more in torch.nn.interpolate)
        align_corners (bool): see more in torch.nn.interpolate
    """

    def __init__(
        self,
        sizes: List[Tuple[int, int]],
        original_size: Tuple[int, int] = None,
        interpolation: str = "nearest",
        align_corners: Optional[bool] = None,
    ):
        '''
        
        '''
        # if original_size is not None and original_size not in sizes:
        #     sizes = [original_size] + list(sizes)
        self.interpolation = interpolation
        self.align_corners = align_corners
        self.original_size = original_size

        super().__init__("size", sizes)

    def pre_transform(self, image, size, history=None, **params):
        if size != self.original_size:
            image = F.resize(
                image,
                size,
                interpolation=self.interpolation,
                align_corners=self.align_corners,
            )
        return image, None

    def pre_transform_inverse(self, image, size, history=None, **params):
        if self.original_size is None:
            raise ValueError(
                "Provide original image size to make mask backward transformation"
            )
        if size != self.original_size:
            image = F.resize(
                image,
                self.original_size,
                interpolation=self.interpolation,
                align_corners=self.align_corners,
            )
        return image, None

    def post_transform(self, label, size=1, history=None, **params):
        return self.pre_transform(label, size, history=history, **params)

    def post_transform_inverse(self, label, size=1, history=None, **params):
        return self.pre_transform_inverse(label, size, history=history, **params)

    def mask_transform(self, mask, size, history=None, **params):
        return self.pre_transform(mask, size, history=history, **params)

    def mask_transform_inverse(self, mask, size, history=None, **params):
        return self.pre_transform_inverse(mask, size, history=history, **params)


class FiveCrops(EquivariantTransform):
    """
    Makes 4 crops for each corner + center crop

    Args:
        crop_scale (float): crop scale as % of pic
    """
    def __init__(self, crop_scale, pad_value=float('nan')):
        super().__init__("crop", ['left_top', 'left_bottom', 'right_bottom', 'right_top', 'center'])
        self.crop_scale = crop_scale
        self.pad_value = pad_value 

    def pre_transform(self, image, crop='center', history=None, **kwargs):
        crop_width = int(image.shape[-1] * self.crop_scale)
        crop_height = int(image.shape[-2] * self.crop_scale)
        left = int(image.shape[-1] * (1 - self.crop_scale) / 2)
        top = int(image.shape[-2] * (1 - self.crop_scale) / 2)
        if crop == 'center':
            image = F.center_crop(image, crop_h=crop_height, crop_w=crop_width)
        elif crop == 'left_top':
            image = F.crop_lt(image, crop_h=crop_height, crop_w=crop_width)
        elif crop == 'left_bottom':
            image = F.crop_lb(image, crop_h=crop_height, crop_w=crop_width)
        elif crop == 'right_bottom':
            image = F.crop_rb(image, crop_h=crop_height, crop_w=crop_width)
        elif crop == 'right_top':
            image = F.crop_rt(image, crop_h=crop_height, crop_w=crop_width)
        else:
            raise NotImplementedError(f'No matching crop in FiveCrop for "{crop}"')

        return image, None

    def pre_transform_inverse(self, image, crop='center', pad_value=None, history=None, **params):
        if pad_value == None:
            pad_value = self.pad_value

        B, C, crop_H, crop_W = image.shape               
        width = ceil(1.0 / self.crop_scale * crop_W)
        height = ceil(1.0 / self.crop_scale * crop_H)
        left = round(width * (1 - self.crop_scale) / 2)
        top = round(height * (1 - self.crop_scale) / 2)
        # print(width, width * (1 - self.crop_scale) / 2, height, width * (1 - self.crop_scale) / 2
        new_size = (B, C, height, width)
        pred = torch.full(new_size, pad_value, dtype=image.dtype, device=image.device, 
            requires_grad=image.requires_grad)
        
        if crop == 'center':
            pred[:,:, top:top+crop_H, left:left+crop_W] = image
        elif crop == 'left_top':
            pred[:,:, :crop_H, :crop_W] = image
        elif crop == 'left_bottom':
            pred[:,:, -crop_H:, :crop_W] = image
        elif crop == 'right_bottom':
            pred[:,:, -crop_H:, -crop_W:] = image
        elif crop == 'right_top':
            pred[:,:, :crop_H, -crop_W:] = image
        else:
            raise NotImplementedError(f'No matching crop in FiveCrop for "{crop}"') 

        return pred, None
        # raise NotImplementedError
        # return self.pre_transform(image, apply) # self-inverse

    def post_transform(self, label, crop='center', history=None, **params):
        return self.pre_transform(image, crop, history=history) # E=e

    def post_transform_inverse(self, label, crop='center', history=None, **params):
        return self.pre_transform_inverse(label, crop, history=history) # E^{-1}=e

    def mask_transform(self, mask, crop='center', history=None, **params):
        return self.pre_transform(mask, crop, history=history)

    def mask_transform_inverse(self, mask, crop='center', history=None, **params):
        return self.pre_transform_inverse(mask, crop, pad_value=0, history=history)

