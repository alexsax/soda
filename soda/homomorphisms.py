import copy
from typing import Any, Optional, Dict, Tuple, Union, Callable

from .algebra import MAct, GActMixin, Element, QuickWrapGroup
from .trace import Trace, TraceMonoid
from .kornia_wrappers import KorniaGAct

class NaturalTransformation:
    def __init__(self, 
                 source_structure: MAct,
                 target_structure: MAct,
                 source_to_target: Dict[MAct, Callable[[Element], Element]],
                ):
        '''
            source_structure: A MAct that defines transformations on the source doimain
            target_structure: A MAct that defines transformations on the target domain
            source_to_target:
                Maps each MAct to a callable that takes a source element to a target element.
                i.e.
                >>> elem_mapper = source_to_target[source_monoid]
                >>> elem_from_target_monoid = elem_mapper(elem_from_source_monoid)

                Note that `source_to_target` acts as the identity on unnamed `MAct`s.
        '''
        self.source = source_structure
        self.target = target_structure
        self.structure_map = source_to_target

    def source_to_target(self, m: Element) -> Element:
        fam_source, param_source = m.family, m.param
        
        if isinstance(fam_source, TraceMonoid):
            history = [self.source_to_target(t) for t in param_source]
            param_source = Trace(history)

        if fam_source in self.structure_map:
            return_elem = self.structure_map[fam_source](m)
        else:
            return_elem = Element(fam_source, param_source)
        # if isinstance(fam_source, KorniaGAct) or isinstance(fam_source, QuickWrapGroup):
        #     print(m.param.keys(), return_elem.param.keys())
        return return_elem

    def __call__(self, m):
        return self.source_to_target(m)

    def action_on_codomain(self, m, x):
        return self.target.action(self(m), x)

    def action_on_domain(self, m, x):
        return self.source.action(m, x)


class NaturalMonoidHomomorphism:
    '''
        Natural transformation between two groups, or monoids. 
        Specifically, this is a triple (source, target, mask):
            source:
            target:
            mask:
        
        Such that the function respects the structure.
        
    '''
    def __init__(self, 
                 source_structure: MAct,
                 target_structure: MAct,
                 structure_map: Optional[Dict[MAct, MAct]] = None,
                 mask_structure: Optional[MAct]=None,
                 mask_structure_map: Optional[Dict[MAct, MAct]] = None,
                ):
        '''
            structure_map:
                Maps the source structure onto the target. I.e. Monoid -> Monoid
                with the appropriate '_monoid' and 'param' for the target structure.
                If None, then no mapping is used
            mask_structure_map:
                Maps the source structure onto the mask. I.e. based on '__name__', replaces {'_monoid', 'param'} in the source trace
                with the appropriate '_monoid' and 'param' for the mask structure.
                If None, then no mapping is used

        '''
        self.source = source_structure
        self.target = target_structure
        self.mask = mask_structure
        self.source_to_target = NaturalTransformation(self.source, self.target, structure_map) 
        self.source_to_mask = NaturalTransformation(self.source, self.mask, mask_structure_map) 
        # self.structure_map = structure_map
        # self.mask_structure_map = mask_structure_map
        
    def action_on_domain(self, m: Element, x: Any):
        return self.source.action(m, x)

    def action_on_codomain(self, m: Element, x: Any): # TODO: rename
        return self.target.action(self.source_to_target(m), x)

    def action_on_mask(self, m: Element, x: Any): # TODO: rename
        return self.mask.action(self.source_to_mask(m), x)

    def __repr__(self):
        return str(type(self)) + str(self.source) + str(self.target) + str(self.mask)+ str(self.mask_structure_map) + str(self.mask_structure_map)


class NaturalGroupHomomorphism(NaturalMonoidHomomorphism):
    '''
        A natural group homomorphism allows test-time augmentation
    '''
    def __init__(self, 
                 source_structure: MAct,
                 target_structure: GActMixin,
                 structure_map: Optional[Dict[MAct, GActMixin]] = None,
                 mask_structure: Optional[MAct]=None,
                 mask_structure_map: Optional[Dict[MAct, GActMixin]] = None,
                ):
        super().__init__(source_structure, target_structure, structure_map, mask_structure, mask_structure_map)

    def inverse_action_on_domain(self, g: Element, x: Any): 
        return self.action_on_domain(self.source.inverse(g), x)

    def inverse_action_on_codomain(self, g: Element, x: Any): 
        inv_action = self.target.inverse( self.source_to_target(g) )
        return self.action_on_codomain(inv_action, x)

    def inverse_action_on_mask(self, g: Element, x: Any): 
        inv_action = self.mask.inverse( self.source_to_mask(g) )
        return self.action_on_mask(inv_action, x)


# Invariance: G Nontrivial --f--> H Trivial
#   i.e. all elements G are in the kernel of f
