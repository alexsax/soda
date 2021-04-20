## Augmentation library
SodA: Not exactly SotA, and mostly empty calories. 

Provides augmentations (mostly for test-time training, but will probably add training aug at some point from [here](https://kornia.readthedocs.io/en/latest/augmentation.module.html#)). 

All augmentation can be done on GPU or CPU on batches. AFAIK all are differentiable, too.


### Usage:

```python
crop_transforms = tta.Product(
    [
        tta.SurfaceNormalHorizontalFlip(dim_horizontal=0),
        tta.FiveCrops(0.9),
        tta.ResizeShortestEdge([512, 256]),
    ]
)
whole_transforms = tta.Product(
    [
        tta.SurfaceNormalHorizontalFlip(dim_horizontal=0),
        tta.ResizeShortestEdge([512, 256, 320, 384, 448]),
    ]
)
transform = list(itertools.chain(crop_transforms, whole_transforms))
wrapper = tta.SurfaceNormalsTTAWrapper(model=model_fn, transforms=transforms, run_mode='parallel_apply', merger_fn=tta.MedianMerger)
```


### Idea:

Augmentation is a type of equivariance, so all the transforms subclass `EquivariantTransform`, which asserts that the following diagram should commute:

                f
        x'-------------> y'
        ^                ^
      e |                | E
        |                |
        x -------------> y
                f

    i.e. E( f( e(x) )) = y (at least for some restricted range of values S)
        e(x):      pre_transform
        e^{-1}(x): pre_transform_inverse (if exists)
        E(y):      post_transform
        E^{-1}(y): post_transform_inverse (if exists)
    
    Notes: If S is smaller than the whole image, 
        `mask_restriction` indicates the VALID pixels in S. 
    Note: to use this transform for TEST-time augmentation,
        `post_transform_inverse` must be implemented 


### Acknowledgements:
Original version was loosely based on the following repo: https://github.com/qubvel/ttach