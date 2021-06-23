import copy
from   functools import partial
import kornia
import kornia.augmentation as K
import torch
import random
from   typing import Any, Optional, Dict, Tuple, Union, Callable

from ..soda.algebra import MAct, GActMixin, QuickWrapGroup, TrivialGroup, Element
from ..soda.trace import Trace, TraceGroup
from ..soda.kornia_wrappers import KorniaMAct, KorniaGAct, RandomErasingMask 
from ..soda.homomorphisms import NaturalMonoidHomomorphism, NaturalGroupHomomorphism

#########################################################
# Examples for surface normals
#########################################################
    
# So there's gonna be the right way and there's gonna be the fast way. 

# The right way is to define all the groups necessary. 
 
# class SurfaceNormalHFlipGroup(QuickWrapGroup):
#     def _hook_after_action(self, m, x, trace, extra):
#         x[:, self.horizontal_channel] = 1 - x[:, self.horizontal_channel]
#         return x


# Probably there is some nicer container than a Dict to map source structure to target structure.
# If no matching element is in the structure map, then the same action is expected to apply. 

class SurfaceNormalSymmetries(NaturalGroupHomomorphism):
    SURFACE_NORMALS_HORIZONTAL_CHANNEL = 0
    SURFACE_NORMALS_VERTICAL_CHANNEL = 1
    SURFACE_NORMALS_FRONT_FACING_CHANNEL = 2
    FRONT_FACING_VECTOR = torch.tensor([[0.,0.,0.]]) 
    FRONT_FACING_VECTOR[:, SURFACE_NORMALS_FRONT_FACING_CHANNEL] += 1.0
    NORMALS_RANGE = (0,1)
    
    @classmethod
    def convert_for_geometry(cls, x):
        return x * 2 - 1

    @classmethod
    def convert_for_output(cls, x):
        return (x + 1) / 2

    @classmethod
    def rot_xy(cls, m: Element, x: Any, extra: Optional[Dict]):
        degrees = m.param['angle']
        if 'do_inverse' in m.param:
            degrees = -degrees
        x = cls.convert_for_geometry(x)
        FRONT_FACING_VECTOR = torch.tensor([[0,1,0]])
        rads = kornia.geometry.conversions.deg2rad(degrees)
        axangle = cls.FRONT_FACING_VECTOR.to(rads.device) * rads.unsqueeze(-1)
        rot_mats = kornia.geometry.conversions.angle_axis_to_rotation_matrix(axangle)
        x = torch.einsum('bdc,bchw->bdhw', rot_mats.to(x.device), x)
        x = cls.convert_for_output(x)
        return m, x, extra

    @classmethod
    def invert_horiz(cls, m: Element, x: Any, extra: Optional[Dict]):
        batch_prob = m.param['batch_prob']
        x[batch_prob, cls.SURFACE_NORMALS_HORIZONTAL_CHANNEL] = 1 - x[batch_prob, cls.SURFACE_NORMALS_HORIZONTAL_CHANNEL]
        return m, x, extra

    @classmethod
    def invert_vert(cls, m: Element, x: Any, extra: Optional[Dict]):
        batch_prob = m.param['batch_prob']
        x[batch_prob, cls.SURFACE_NORMALS_VERTICAL_CHANNEL] = 1 - x[batch_prob, cls.SURFACE_NORMALS_VERTICAL_CHANNEL]
        return m, x, extra
        
    def __init__(self,
                # ignore_eps=1e-8,
                #  random_horizontal_flip_kwargs=dict(
                #      same_on_batch=False,
                #      p = 0.5,
                #      p_batch = 1.0
                #  ),
                #  random_vertical_flip_kwargs=dict(
                #      same_on_batch=False,
                #      p = 0.,
                #      p_batch = 1.0
                #  ),                
                #  random_affine_kwargs=dict(
                #      degrees=(-30,30),
                #      scale=(0.75, 1.25),
                #      same_on_batch=False,
                #      p = 0.9,
                #  ),                   
                #  random_crop_kwargs=dict(
                #      size=(512, 512),
                #      p=0.9,
                #      resample='NEAREST',
                #      cropping_mode='resample',
                #      same_on_batch=False,
                #  ),
                #  random_erasing_kwargs=dict(
                #      scale=(.02, .3),
                #      ratio=(.3, 1/.3),
                #      p=0.4
                #  ),
                #  random_grayscale_kwargs=dict(
                #      p=0.2,
                #  ),
                #  random_equalize_kwargs=dict(
                #      p=0.2,
                #  ),                 
                #  random_motion_blur_kwargs=dict(
                #      kernel_size=5,
                #      angle=35.,
                #      direction=0.5,
                #      p=0.0,
                #  ),
                #  random_gaussian_noise_kwargs=dict(
                #      mean=0.,
                #      std=0.03,
                #      p=0.8
                #  ),
                #  random_gaussian_blur_kwargs=dict(
                #      kernel_size=(5, 5),
                #      sigma=(0.1, 2.0),
                #      p=0.8
                #  ),
                #  random_sharpness_kwargs=dict(
                #      sharpness=10,
                #      p=0.0,
                #  ),
                #  random_color_jitter_kwargs=dict(
                #      brightness=0.2,
                #      contrast=0.15,
                #      saturation=0.8,
                #      hue=0.3,
                #      p=0.8
                #  ),
                #  random_posterize_kwargs=dict(
                #      bits=6,
                #      p=1.0,
                #  ),  
                #  randomize_non_invariances=False,
                #  randomize_invariances=True,
                #  resample_invariances=False,
                # ):
                 clamp=(0.,1.), # Clamp values after transform.
                 ignore_eps=1e-8,
                 random_horizontal_flip_kwargs=dict(
                     same_on_batch=False,
                     p = 0.5,
                     p_batch = 1.0
                 ),
                 random_vertical_flip_kwargs=dict(
                     same_on_batch=False,
                     p = 0.,
                     p_batch = 1.0
                 ),                
                 random_affine_kwargs=dict(
                     degrees=(-30,30),
                     scale=(0.75, 1.25),
                     same_on_batch=False,
                     p = 1.0,
                 ),                   
                 random_crop_kwargs=dict(
                     size=(512, 512),
                     p=1.0,
                     resample='NEAREST',
                     cropping_mode='resample',
                     same_on_batch=False,
                 ),
                 random_erasing_kwargs=dict(
                     scale=(.02, .3),
                     ratio=(.3, 1/.3),
                     p=1.0
                 ),
                 random_grayscale_kwargs=dict(
                     p=0.2,
                 ),
                 random_equalize_kwargs=dict(
                     p=0.2,
                 ),                 
                 random_motion_blur_kwargs=dict(
                     kernel_size=5,
                     angle=35.,
                     direction=0.5,
                     p=0.0,
                 ),
                 random_gaussian_noise_kwargs=dict(
                     mean=0.,
                     std=0.03,
                     p=0.8
                 ),
                 random_gaussian_blur_kwargs=dict(
                     kernel_size=(5, 5),
                     sigma=(0.1, 2.0),
                     p=0.8
                 ),
                 random_sharpness_kwargs=dict(
                     sharpness=10,
                     p=0.0,
                 ),
                 random_color_jitter_kwargs=dict(
                     brightness=0.2,
                     contrast=0.15,
                     saturation=0.8,
                     hue=0.3,
                     p=0.8
                 ),
                 random_posterize_kwargs=dict(
                     bits=6,
                     p=1.0,
                 ),  
                 randomize_non_invariances=False,
                 randomize_invariances=True,
                 resample_invariances=False,
                ):
        ''' Applying source structure is expected to be idempotent.'''
        source_structure, target_structure, mask_structure = [], [], []
        target_structure_map, mask_structure_map = {}, {}
        self.randomize_non_invariances = randomize_non_invariances
        self.randomize_invariances = randomize_invariances
        self.resample_invariances = resample_invariances

        ##################################
        #  EQUIVARIANCES
        ##################################
        ######  Affine transforms:  ######
        ######     Resize           ######
        ######     Rotation         ######
        if not (random_affine_kwargs['p'] < ignore_eps):
            affine_rgb = KorniaGAct(K.RandomAffine(**random_affine_kwargs))
            affine_normals = QuickWrapGroup(affine_rgb, {'hook_after_action': SurfaceNormalSymmetries.rot_xy})
            source_structure.append(affine_rgb)
            target_structure_map[affine_rgb] = affine_normals
            
        ######  Crop equivariance  ######
        # source_structure.append(FixedSizeCrop(size=(256, 256), p=1.0, return_transform=False))
        if not (random_crop_kwargs['p'] < ignore_eps):
            source_structure.append(KorniaGAct(K.RandomCrop(**random_crop_kwargs)))

        #######  Horizontal flip equivariance  ######
        if not (random_horizontal_flip_kwargs['p'] < ignore_eps):
            hflip_rgb = KorniaGAct(K.RandomHorizontalFlip(**random_horizontal_flip_kwargs))
            hflip_normals = QuickWrapGroup(hflip_rgb, {'hook_after_action': SurfaceNormalSymmetries.invert_horiz})
            source_structure.append(hflip_rgb)
            target_structure_map[hflip_rgb] = hflip_normals

        #######  Vertical flip equivariance  ######
        if not (random_vertical_flip_kwargs['p'] < ignore_eps):
            vflip_rgb = KorniaGAct(K.RandomVerticalFlip(return_transform=False, same_on_batch=False, p = 0.5, p_batch = 1.0))
            vflip_normals = QuickWrapGroup(vflip_rgb, {'hook_after_action': SurfaceNormalSymmetries.invert_vert})
            source_structure.append(vflip_rgb)
            target_structure_map[vflip_rgb] = vflip_normals


        ######  Random mask inpainting  ######
        if not (random_erasing_kwargs['p'] < ignore_eps):
            source_structure.append(RandomErasingMask(K.RandomErasing(**random_erasing_kwargs)))

        ######  Multiview/Flow equivariance  ######


        ##################################
        #  INVARIANCES
        ##################################
        trivial_group = TrivialGroup()
        invariances = {
            KorniaMAct(source, clamp=clamp): trivial_group
            for source in [
                K.RandomGrayscale(**random_grayscale_kwargs),           ######  Grayscale invariance   ######
                K.RandomEqualize(**random_equalize_kwargs),             ######  Equalization invariance   ######
                K.RandomMotionBlur(**random_motion_blur_kwargs),        ######  Motion blur invariance   ######
                K.RandomGaussianNoise(**random_gaussian_noise_kwargs),  ######  Noise invariance   ######
                K.GaussianBlur(**random_gaussian_blur_kwargs),          ######  Blur invariance   ######
                K.RandomSharpness(**random_sharpness_kwargs),           ######  Sharpen invariance   ######
                K.ColorJitter(**random_color_jitter_kwargs),            ######  Jitter invariance   ######
                K.RandomPosterize(**random_posterize_kwargs),           ######  [CPU-only] Posterize invariance   ######
#                 ######  Defocus invariance   ######
#                 K.RandomSolarize(0.001, 0.001, p=1.)                 ######  [Buggy?] Solarize invariance   ######
            ]
            if source.p > ignore_eps
        }

        ##################################
        #  INVARIANCES
        ##################################
        self.invariances = [m for m in invariances.keys()]
        self.equivariances = source_structure
        source_structure = source_structure + [m for m in invariances.keys()]
        target_structure_map.update(invariances)
        target_structure_map = {
            source: partial(Element.replace_family, family=target)
            for source, target in target_structure_map.items()
        }

        mask_structure_map.update(invariances)
        mask_structure_map = {
            source: partial(Element.replace_family, family=target)
            for source, target in mask_structure_map.items()
        }

        source_group = TraceGroup(*source_structure)
        target_group = TraceGroup(*target_structure)
        mask_group   = TraceGroup(*mask_structure)

        super().__init__(source_group, target_group, target_structure_map, mask_group, mask_structure_map)
    


    def random_action(self, x, n_actions=8):
        '''
            Generates a random action on the source domain. 
            Does equivariances first, then invariances.
        '''
        eqv = list(self.equivariances)
        if self.randomize_non_invariances:
            random.shuffle(eqv)
        inv = list(self.invariances)
        if self.randomize_invariances:
            random.shuffle(inv)
        groups_random_order = eqv + inv
#         monoid_shuffled = list(self.source._generating_monoids)
#         random.shuffle(monoid_shuffled)
        if self.resample_invariances:
            raise notImplementedError
        history = []
        for monoid in groups_random_order:
#             monoid = random.choice(self.source._generating_monoids)
            m, x = monoid.random_action(x)
            history.append(m)
            x = x.clamp(0,1)
        trace = Element(family=self.source, param=Trace(history))
        return trace, x