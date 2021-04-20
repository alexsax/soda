import itertools
from functools import partial
from typing import List, Optional, Union

from . import functional as F

class TransformHistory:
    @classmethod
    def push(cls, cache, history=None):
        if history is None:
            return [cache]
        return history + [cache]
    
    @classmethod
    def pop(cls, history):
        '''
            last, history[:-1]
        '''
        if history is None or len(history) == 0:
            raise ValueError('Tried to pop from an empty history!')
        return history[-1], history[:-1]

    @classmethod
    def peek(cls, history):
        if history is None or len(history) == 0:
            raise ValueError('Tried to peek an empty history!')
        return history[-1]

class EquivariantTransform:
    '''
    The following diagram should commute:

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

    '''
    def __init__(
            self,
            name: str,
            params: Union[list, tuple],
    ):
        '''
        name:   Name of the param (used for printing/debugging)
        params: Possible parameterizations of the transform 
        '''
        self.params = params
        self.pname = name

    def pre_transform(self, image, *args, **params):
        raise NotImplementedError

    def pre_transform_inverse(self, image, *args, **params):
        raise NotImplementedError

    def post_transform(self, label, *args, **params):
        raise NotImplementedError

    def post_transform_inverse(self, label, *args, **params):
        raise NotImplementedError

    def mask_transform(self, mask, *args, **params):
        return mask 
    
    def mask_transform_inverse(self, mask, *args, **params):
        return mask

    # def mask_restriction(self, mask, *args, **params):
    #     forward_mask = self.mask_transform(self, mask, *args, **params) 
    #     restricted_mask = self.mask_transform_inverse(self, forward_mask, *args, **params) 
    #     return restricted_mask



class Chain:

    def __init__(
            self,
            functions: List[callable]
    ):
        self.functions = functions or []

    def __call__(self, x, history=None):
        for f in self.functions:
            x, history = f(x, history=history)
            # print(str(f).split(' ')[2], x.shape, x.mean())
        return x, history



######################################
#           Transformers             #
######################################

class TTATransformer:
    def __init__(
            self,
            pre_pipeline: Chain,
            mask_pipeline: Chain,
            post_inverse_pipeline: Chain,
    ):
        self.pre_pipeline = pre_pipeline
        self.mask_pipeline = mask_pipeline
        self.post_inverse_pipeline = post_inverse_pipeline

    def pre_transform(self, image, history=None):
        return self.pre_pipeline(image, history)

    def mask_restriction(self, mask, history=None):
        return self.mask_pipeline(mask, history=history)

    def post_inverse_transform(self, label, history=None):
        return self.post_inverse_pipeline(label, history)




class Transformer:
    def __init__(
            self,
            image_pipeline: Chain,
            mask_pipeline: Chain,
            label_pipeline: Chain,
            keypoints_pipeline: Chain
    ):
        self.image_pipeline = image_pipeline
        self.mask_pipeline = mask_pipeline
        self.label_pipeline = label_pipeline
        self.keypoints_pipeline = keypoints_pipeline

    def augment_image(self, image):
        return self.image_pipeline(image)

    def deaugment_mask(self, mask):
        return self.mask_pipeline(mask)

    def deaugment_label(self, label):
        return self.label_pipeline(label)

    def deaugment_keypoints(self, keypoints):
        return self.keypoints_pipeline(keypoints)



class Product:
    '''
        Returns product of the following transforms
        TODO: implement high-performance version that batches transforms

        To combine lots of transforms:
            itertools.chain(product1, product2, product3)
    '''
    def __init__(
            self,
            transforms: List[BaseTransform],
    ):
        self.aug_transforms = transforms
        self.aug_transform_parameters = list(itertools.product(*[t.params for t in self.aug_transforms]))
        self.deaug_transforms = transforms[::-1]
        self.deaug_transform_parameters = [p[::-1] for p in self.aug_transform_parameters]

    def __iter__(self) -> TTATransformer:
        for aug_params, deaug_params in zip(self.aug_transform_parameters, self.deaug_transform_parameters):
            pre_chain = Chain([partial(t.pre_transform, **{t.pname: p})
                                     for t, p in zip(self.aug_transforms, aug_params)])
            post_inverse_chain = Chain([partial(t.post_transform_inverse, **{t.pname: p})
                                       for t, p in zip(self.deaug_transforms, deaug_params)])
            mask_chain = Chain(
                [partial(t.mask_transform, **{t.pname: p})
                        for t, p in zip(self.aug_transforms, aug_params)] \
                + [partial(t.mask_transform_inverse, **{t.pname: p})
                        for t, p in zip(self.deaug_transforms, deaug_params)]
                )

            yield TTATransformer(
                pre_pipeline=pre_chain,
                mask_pipeline=mask_chain,
                post_inverse_pipeline=post_inverse_chain,
            )

    def __len__(self) -> int:
        return len(self.aug_transform_parameters)



class Merger:

    def __init__(
            self,
            n: int = 1,
    ):
        super().__init__()
        self.preds = []
        self.n = n

    def append(self, x):
        self.preds.append(x)

    @property
    def result(self):
        pass
