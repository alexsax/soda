import copy
import kornia.augmentation as K
from typing import Any, Optional, Dict, Tuple, Union, Callable

from ..soda.algebra import MAct, GActMixin, QuickWrapGroup, TrivialGroup, TraceGroup
from ..soda.trace import Trace
from ..soda.kornia_wrappers import KorniaMAct, HFlip, VFlip, FixedSizeCrop
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
    SURFACE_NORMALS_VERTICAL_CHANNEL = 2

    def __init__(self):
        ''' Applying source structure is expected to be idempotent.'''
        source_structure = []
        target_structure = []
        mask_structure = []
        mask_structure_map = {}
        target_structure_map = {}
        
        ##################################
        #  EQUIVARIANCES
        ##################################
        #######  Flip equivariance  ######
        def invert_horiz(m, x, trace, extra):
            x[m['batch_prob'], SurfaceNormalSymmetries.SURFACE_NORMALS_HORIZONTAL_CHANNEL] = 1 - x[m['batch_prob'], SurfaceNormalSymmetries.SURFACE_NORMALS_HORIZONTAL_CHANNEL]
            return m, x, trace, extra
        hflip_rgb = HFlip()
        hflip_normals = QuickWrapGroup(hflip_rgb, {'hook_after_action': invert_horiz})
        source_structure.append(hflip_rgb)
        target_structure_map[hflip_rgb] = hflip_normals
        
        def invert_vert(m, x, trace, extra):
            x[m['batch_prob'], SurfaceNormalSymmetries.SURFACE_NORMALS_VERTICAL_CHANNEL] = 1 - x[m['batch_prob'], SurfaceNormalSymmetries.SURFACE_NORMALS_VERTICAL_CHANNEL]
            return m, x, trace, extra
        vflip_rgb = VFlip()
        vflip_normals = QuickWrapGroup(vflip_rgb, {'hook_after_action': invert_vert})
        source_structure.append(vflip_rgb)
        target_structure_map[vflip_rgb] = vflip_normals


        ######  Rotation equivariance  ######
        ######  Resize equivariance  ######
        ######  Crop equivariance  ######
        source_structure.append(FixedSizeCrop(size=(256, 256), p=1.0, return_transform=False))

        
        ######  Multiview/Flow equivariance  ######


        ##################################
        #  INVARIANCES
        ##################################
        trivial_group = TrivialGroup()
        invariances = {
            KorniaMAct(source): trivial_group
            for source in [
                K.RandomEqualize(p=1.),                          ######  Equalization invariance   ######
                K.RandomMotionBlur(5, 35., 0.5, p=1.),           ######  Motion blur invariance   ######
                K.RandomGaussianNoise(mean=0., std=0.03, p=1.),  ######  Noise invariance   ######
                K.GaussianBlur((3, 3), (0.1, 2.0), p=1.),        ######  Blur invariance   ######
                K.RandomSharpness(10, p=1.),                     ######  Sharpen invariance   ######
                ######  Defocus invariance   ######
#                 K.RandomSolarize(0.1, 0.1, p=1.)                 ######  [Buggy?] Solarize invariance   ######
#                 K.RandomPosterize(3, p=1.),                      ######  [CPU-only] Posterize invariance   ######
            ]
        }
        source_structure = source_structure + [m for m in invariances.keys()]
        target_structure_map.update(invariances)
        mask_structure_map.update(invariances)

        source_group = TraceGroup(*source_structure)
        target_group = TraceGroup(*target_structure)
        mask_group   = TraceGroup(*mask_structure)

        super().__init__(source_group, target_group, target_structure_map, mask_group, mask_structure_map)
    


    def random_action(self, m):
        '''
            Generates a random action on the source domain
        '''
        pass
        
    # Handle inverses, and the fact that some elements might not have inverses