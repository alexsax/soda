import copy
from typing import Any, Optional, Dict, Tuple, Union, Callable

from .algebra import MAct, GActMixin
from .trace import Trace


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
        self.structure_map = structure_map
        self.mask_structure_map = mask_structure_map
        
    def action_on_domain(self, m, x, trace=None):
        return self.source.action(m, x, trace)

    def action_on_codomain(self, m, x, trace=None): # TODO: rename
        m = self._apply_recursively(self.structure_map, m)
        return self.target.action(m, x, trace)

    def action_on_mask(self, m, x, trace=None): # TODO: rename
        m = self._apply_recursively(self.mask_structure_map, m)
        return self.mask.action(m, x, trace)

    def _apply_recursively(self, mapping: Dict[MAct, MAct], trace: Union[Dict, Trace]) -> Union[Dict, Trace]:
        if isinstance(trace, dict):
            if '_monoid' not in trace:
                raise ValueError(f"Expected to find key '_monoid' in dict, but not found in {trace.keys()}")
            if trace['_monoid'] in mapping:
                return_trace = copy.copy(trace)
                return_trace['_monoid'] = mapping[trace['_monoid']]
                return return_trace
            else:
                return trace
        elif isinstance(trace, Trace):
            history = [self._apply_recursively(mapping, t) for t in trace]
            return Trace(history)
        else:
            raise NotImplementedError

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

    def inv_codomain(self, m: Trace, x: Any, is_source_trace=True) -> Tuple[Any, Trace]: 
        if is_source_trace:
            m = self._apply_recursively(self.structure_map, m)
        return self.target.inverse_pipeline(x, m)

    def inv_mask(self, m: Trace, x: Any, is_source_trace=True) -> Tuple[Any, Trace]: 
        if is_source_trace:
            m = self._apply_recursively(self.mask_structure_map, m)
        return self.mask.inverse_pipeline(x, m)

# Invariance: G Nontrivial --f--> H Trivial
#   i.e. all elements G are in the kernel of f
