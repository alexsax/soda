from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy

class Element():
    '''
        Syntactic sugar class.
        Elem(monoid, m)(x) = monoid.action(m, x)
        This effectively curries the action
        >>> monoid = MAct(params)
        >>> m = monoid.random_action()
        >>> m(x) # Error if m is like a dict or sth
        >>> monoid.action(m, x) # works
        >>> apply_m_to = Element(monoid, m)
        >>> apply_m_to(x) # works

        param: param is an element of family
    '''
    def __init__(self, family: 'MAct', param: Any): # 'MAct' Type hint is defined below
        self.param = param
        self.family = family
       
    def __call__(self, *args, **kwargs): # passthrough fn
        return self.family.action(self, *args, **kwargs)

    def __repr__(self):
        return f"{self.family} element w/ parameter: {self.param}"

    @staticmethod
    def replace_family(elem: 'Element', family: 'MAct') -> 'Element':
        return Element(family=family, param=elem.param)


###########################################
#    ABSTRACT CLASSES
###########################################

class MAct: # Action of a monoid on a set S
    '''
      Some set of actions, S that act on a set X
      For properties, see https://en.wikipedia.org/wiki/Semigroup_action#S-homomorphisms
      
      Actions should indicate the monoid in use (elements/morphisms carry type of the category.)
    '''

    def __init__(self):
        self._identity = None 

    def unit(self) -> Element: # Function that maps x -> Id_x
        raise NotImplementedError

    # I don't want to include a second pathway for generating actions just yet
    # def random_elem(self, info):
    #     raise NotImplementedError

    def compose(self, m1: Element, m2: Element) -> Element:
        pass

    def action(self, m: Element, x: Any) -> Any:
        raise NotImplementedError(f"self.action(m, x) not implemented for class {type(self)}")

    def __call__(self, *args, **kwargs) -> Any:
        return self.action(*args, **kwargs)

    def random_action(self, x: Any) -> Tuple[Element, Any]:
        '''
            Applies some random element in the monoid to x.
            Returns:
                m: Element that transforms x (i.e. m(x) = y)
                y: Transformed x
        '''
        raise NotImplementedError(f"self.random_action(x) not implemented for class {type(self)}")

    def get_representative_params(self, n_params) -> List[Element]: 
        ''' 
            Returns a number of 'representative' elements from this monoid.
            e.g. for FiveCrop, returns the corners and center [perhaps using the average sized crop]
        '''
        pass

    # def _ensure_wrap_type(self, m):
        # if not isinstance(m, dict):
        #     m = {'param': m}
        # m['__name__'] = repr(self)
        # m['_monoid']  = self
        # return m
        
    # def __repr__(self) -> str:
        # raise NotImplementedError()

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other) -> bool:
        return self.__repr__() == other.__repr__()

class GActMixin:
    ''' For properties, see: https://en.wikipedia.org/wiki/Group_action'''

    def inverse(self, m: Element) -> Element:
        '''
            Returns m^{-1}, the inverse of m s.t.
               m^{-1} * m = e = m * m^{-1}
        '''
        raise NotImplementedError(f"self.inverse(m) not implemented for class {type(self)}")




###########################################
#    Instantiable CLASSES
###########################################
class TrivialMonoid(MAct):
    def __repr__(self):
        return str(type(self))

    def unit(self) -> Element:
        return Element(self, 'IDENTITY')

    def random_elem(self):
        return self.unit()

    def action(self, m: Element, x: Any):
        return x

    def random_action(self, x: Any) -> Tuple[Element, Any]:
        return self.unit(), x

class TrivialGroup(TrivialMonoid, GActMixin):
    def inverse(self, m: Element) -> Element:
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
    
    def random_action(self, x) -> Tuple[Element, Any]:
        return self.group.random_action(x)

    def action(self, m, x):
        extra = None
        m, x, extra = self._hook_before_action(m, x, extra)
        x = self.group.action(m, x)
        m, x, extra = self._hook_after_action(m, x, extra)
        return x 
    
    def _hook_before_action(self, m, x, extra=None):
        if hasattr(self, 'hook_before_action'):
            return self.hook_before_action(m, x, extra)
        else:
            return m, x, extra

    def _hook_after_action(self, m, x, extra=None):
        if hasattr(self, 'hook_after_action'):
            return self.hook_after_action(m, x, extra)
        else:
            return m, x, extra    

    def __repr__(self):
        return self.group.__repr__() + f" | funcs: {self.overwritten_functions}"

class QuickWrapGroup(QuickWrapMonoid, GActMixin):
    def inverse(self, m):
        extra = None
        m, extra = self._hook_before_inverse(m, extra)
        m = self.group.inverse(m)
        m, extra = self._hook_after_inverse(m, extra)
        return m
    
    def _hook_before_inverse(self, m, extra=None):
        if hasattr(self, 'hook_before_inverse'):
            return self.hook_before_inverse(m, extra)
        else:
            return m, extra
    
    def _hook_after_inverse(self, m, extra=None):
        if hasattr(self, 'hook_after_inverse'):
            return self.hook_after_inverse(m, extra)
        else:
            return m, extra



