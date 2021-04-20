import torch
import torch.nn as nn
from torch.nn.parallel import parallel_apply
from typing import Optional, Mapping, Union, Tuple

from .base import Merger, Compose
from .mergers import *


class TransformRunner:

    @classmethod
    def serial(cls, image, model, transforms, merger):
        for transform in transforms:
            image_transformed, history = transform.pre_transform(image)
            model_output = model(image_transformed)
            label_inv_transformed, _ = transform.post_inverse_transform(model_output, history=history)
            # if mask is not None:
            #     mask_restricted, _ = transform.mask_restriction(mask)

            merger.append(label_inv_transformed)
        return merger

    @classmethod
    def parallel_apply(cls, image, model, transforms, merger):
        def _thunk(image, transform):
            image_transformed, history = transform.pre_transform(image)
            model_output = model(image_transformed)
            label_inv_transformed, _ = transform.post_inverse_transform(model_output, history=history)
            return label_inv_transformed
        
        preds = parallel_apply(
                [_thunk] * len(transforms),
                list(zip([image] * len(transforms), transforms))
            )

        for pred in preds:
            merger.append(pred)

        return merger

class SurfaceNormalsTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (sf normals model) with test time augmentation transforms

    Args:
        model (torch.nn.Module): sf normals model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_mask_key (str): if model output is `dict`, specify which key belong to `mask`
    """

    def __init__(
        self,
        model: nn.Module,
        transforms: Compose,
        merger_fn: Merger = MeanMerger,
        run_mode: str = 'parallel_apply',
        output_mask_key: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merger_fn = merger_fn
        self.output_key = output_mask_key
        self.run_mode = run_mode 


    def forward(
        self, image: torch.Tensor, *args
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:
       
        merger = self.merger_fn()
        if self.run_mode == 'serial':
            TransformRunner.serial(image, self.model, self.transforms, merger)
        elif self.run_mode == 'parallel_apply':
            TransformRunner.parallel_apply(image, self.model, self.transforms, merger) 
        else:
            raise NotImplementedError(f'Unknown run mode {self.run_mode}')

        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}

        return result
