from typing import Any, Optional, Dict, Tuple, Union, Callable
import copy

from .trace import Trace

###########################################
#    ABSTRACT CLASSES
###########################################
class MAct: # Action of a monoid on a set S
    '''
      Monoid acts on a set X
      For properties, see https://en.wikipedia.org/wiki/Semigroup_action#S-homomorphisms
      
      Actions should indicate the monoid in use (elements/morphisms carry type of the category.)
    '''
    def __init__(self):
        self._identity = None 

    def identity(self, x):
        return self.action(self._identity, x)

    def action(self, m, x, trace: Optional[Trace]=None):
        raise NotImplementedError()

    def random_action(self, x, trace: Optional[Trace]=None):
        '''
            Applies some random element in the monoid to x. 
        '''
        raise NotImplementedError()

    def get_representative_params(self, n_params): 
        ''' 
            Returns a number of 'representative' elements from this monoid.
            e.g. for FiveCrop, returns the corners and center [perhaps using the average sized crop]
        '''
        pass

    def _ensure_wrap_type(self, m):
        if not isinstance(m, dict):
            m = {'action': m}
        m['_monoid'] = self
        return m
    
        
    def __repr__(self):
        raise NotImplementedError()

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()
    
class GActMixin:
    ''' For properties, see: https://en.wikipedia.org/wiki/Group_action'''

    def inv(self, m):
        '''
            Returns m^{-1}, the inverse of m s.t.
               m^{-1} * m = e = m * m^{-1}
        '''
        raise NotImplementedError

    def inverse_action(self, m, x_t, trace: Optional[Trace]=None):
        m_inv = self.inv(m)
        x, trace = self.action(m_inv, x_t, trace)
        return x, trace

    def inverse_last_action(self, x_t, trace: Trace):
        '''
            Params:
                x: set item to apply inverse action
                trace: A 'word' describing the tranformatoins applied to x
        '''
        assert isinstance(trace, Trace), 'Trace must not be none (no action to invert)'
        cache, trace = trace.pop()
        x, _ = self.inverse_action(cache['param'], x_t, trace)
        return x, trace




###########################################
#    Instantiable CLASSES
###########################################
class TrivialMonoid(MAct):
    def __repr__(self):
        return str(type(self))

    def action(self, m, x, trace=None):
        if trace is None:
            trace = Trace()
        cache = {'__name__': f'{self.__repr__()}', 'param': m}
        cache = self._ensure_wrap_type(cache)
        return x, trace

class TrivialGroup(TrivialMonoid, GActMixin):
    def inv(self, m):
        return m

class QuickWrapMonoid(MAct):
    ''''
        This wrapper allows making lightweight changes to a monoid or group.
        >>> hflip_rgb = HFlip()
        >>> def invert_horiz(m, x, trace, extra, horizontal_channel=0):
        >>>     x[:, horizontal_channel] = 1 - x[:, horizontal_channel]
        >>>     return x
        >>> hflip_normals = QuickWrapGroup(hflip_rgb, {'hook_after_action': invert_horiz})
    '''
    def __init__(self, group, functions: Dict[str, Callable]={}):
        super().__init__()
        self.group = group
        
        self.overwritten_functions = {}
        for name, func in functions.items():
            setattr(self, name, func)
            self.overwritten_functions[name] = func.__repr__
    
    def random_action(self, x, trace=None):
        return self.group.random_action(x, trace)

    def action(self, m, x, trace=None):
        extra = None
        m, x, trace, extra = self._hook_before_action(m, x, trace, extra)
        x, trace = self.group.action(m, x, trace)
        self._ensure_wrap_type(trace[-1])
        m, x, trace, extra = self._hook_after_action(m, x, trace, extra)
        return x, trace 
    
    def _hook_before_action(self, m, x, trace, extra=None):
        if hasattr(self, 'hook_before_action'):
            return self.hook_before_action(m, x, trace, extra)
        else:
            return m, x, trace, extra

    def _hook_after_action(self, m, x, trace, extra=None):
        if hasattr(self, 'hook_after_action'):
            return self.hook_after_action(m, x, trace, extra)
        else:
            return m, x, trace, extra    

    def __repr__(self):
        return self.group.__repr__() + f" | funcs: {self.overwritten_functions}"

class QuickWrapGroup(QuickWrapMonoid, GActMixin):
    def inv(self, m):
        extra = None
        m, extra = self._hook_before_inv(m, extra)
        m = self.group.inv(m)
        m, extra = self._hook_after_inv(m, extra)
        return m
    
    def _hook_before_inv(self, m, extra=None):
        if hasattr(self, 'hook_before_inv'):
            return self.hook_before_inv(m, extra)
        else:
            return m, extra
    
    def _hook_after_inv(self, m, extra=None):
        if hasattr(self, 'hook_after_inv'):
            return self.hook_after_inv(m, extra)
        else:
            return m, extra


#######################################################################
# TraceMonoid and TraceGroup
#######################################################################
# This is the thing that calculates the new element (total transform)
# I'm not much of an algebraist, but the expert in algebraic topology will 
# note that this is the Free Product of groups
# We end up putting a score on the equivariance topology for each task (by how well the equivariances transfer to the task)
# This might be an interesting connection between algebraic topology and the structure of the task space
# "Algebraic Taskology" ;) 

# Theoretical way to measure topology similarity: 
#     - Set of equivariances form some group
#     - How to put a number on the distance between groups?
#     - E.g. distance between cayley graphs:  https://en.wikipedia.org/wiki/Cayley_graph#:~:text=In%20mathematics%2C%20a%20Cayley%20graph,abstract%20structure%20of%20a%20group.
#          - Like graph edit distance: https://en.wikipedia.org/wiki/Graph_edit_distance
#.         - General field of graph distances: https://en.wikipedia.org/wiki/Graph_matching
#          - Search: error-tolerant graph matching
#     - Some sort of distance betweHow to meas


class TraceMonoid(MAct):
    '''
        For definition:
            Graph Theory:
                Dependency Graph
            Parallel Computation:
                Trace Monoid: https://en.wikipedia.org/wiki/Trace_monoid
                Trace Theory: trace theory (wikipedia), http://www.cas.mcmaster.ca/~cas724/2007/paper2.pdf
            Algebraic Topology:
                Free product of monoids. For definition, see:
                    - Algebraic Topology by Allen Hatcher (http://www.math.cornell.edu/~hatcher/AT/AT.pdf) line 1 page 42
                    - Wikipedia: https://en.wikipedia.org/wiki/Free_product
                To look at simple results: http://www.math.uchicago.edu/~may/VIGRE/VIGRE2008/REUPapers/Nelson.pdf
    
        Elements are traces.

        
        Can be used for similarity learning.
        
    '''
    def __init__(self, *generating_monoids):
        '''
            generating_groups: G can be decomposed as a free product G = A1*A2*...*Ar*Fs, where s ≥ 0, r ≥ 0,
            e.g. see Grushko decomposition theorem (https://en.wikipedia.org/wiki/Grushko_theorem)
        '''
        super().__init__()
        self._generating_monoids = list(generating_monoids)  # currently unused
        self._identity = 'WRAP'

    def wrap(self, m: Union[Dict, Trace]):
        if isinstance(m, Trace):
            cache = {'__name__': f'{self.__repr__()}', 'param': m}
            cache = self._ensure_wrap_type(cache)
            print(f'returning cache: {len(m.history)}, {cache["_monoid"]}')
            return cache
        else:
            return m

    def action(self, m: Trace, x: Any, trace: Optional[Trace] = None):
        if trace is None:
            trace = Trace()
        
#         assert 'param' in m, f"Calling {type(self)}.action() but the element does not contain the key 'param'"
#         assert isinstance(m['param'], Trace), f"Calling {type(self)}.action() but the element does not look like a Trace ({type(m['param'])})"
        x_t = x
        for m_ in m:
            monoid = m_['_monoid']
            x_t, _ = monoid.action(m_['param'], x_t)
        
        cache = {'__name__': f'{self.__repr__()}', 'param': m}
        cache = self._ensure_wrap_type(cache)
        return x_t, trace.push(cache)

    def pipeline(self, m: Trace, x: Any, trace: Optional[Trace] = None):
        return self.action(m, x, trace)
    
    def random_action(self, x, trace=None):
        '''
            Applies some random element in the monoid to x. 
        '''
        raise NotImplementedError('Random action must be sampled from some distribution over the traces. \
                Not sure how to implement this (limit max length? Probablility of word ending?)')

    def __repr__(self):
        return str(type(self)) + str(self._generating_monoids)

class TraceGroup(TraceMonoid, GActMixin):
    '''
        This can be used for test-time augmentation.
    '''
    
    def inv(self, m: Union[Trace, Dict]):
        if isinstance(m, dict):
           m = m['param'] 
        history = []
        for t in m.reverse():
            monoid = t['_monoid']
            cache_copy = copy.copy(t)
            # print('monoid:', monoid)
            cache_copy['param'] = monoid.inv(t['param'])
            history.append(cache_copy)
            
        m_inv = Trace(history)
        return m_inv
    
    def inverse_pipeline(self, x_t: Any, trace: Trace):
        x_t, _ = self.inverse_action(trace, x_t)
#         m_inv = self.inv(trace)
#         x_t, _ = self.action(m_inv, x_t)
        return x_t, Trace()