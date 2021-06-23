from typing import Any, Optional, Dict, Tuple, Union, Callable

from .algebra import MAct, GActMixin, Element

class Trace:
    ''' 
        Records a 'word' describing a tranformation history. 
        For example, if there are the transformations A={a, b, c}, then all 
        possible traces are the Kleene-star of the alphabet (A*).
        An example word w in A* would be "aabacacb".
        
        In this case, a 'cache' is considered a letter, and 'history' is
        considered a word.
    '''
    def __init__(self, history=None):
        super().__init__()
        self.history = history
        if self.history is None:
            self.history = []
        if not isinstance(self.history, list):
            self.history = [self.history]
        
    def push(self, cache):
        # assert isinstance(cache, Action) ?
        new_history = self.history + [cache]
        return Trace(new_history)
    
    def pop(self):
        if self.history is None or len(self.history) == 0:
            raise ValueError('Tried to pop from an empty history!')
        return self.history[-1], Trace(self.history[:-1])
    
    def peek(self):
        if self.history is None or len(self.history) == 0:
            raise ValueError('Tried to peek an empty history!')
        return self.history[-1]
    
    def peek_param(trace):
        return trace.peek()['param']
    
    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, slice):
            return Trace(self.history[int_or_slice])
        else:
            return self.history[int_or_slice]
    
    def reverse(self):
        return Trace(self.history[::-1])

    def __iter__(self):
        return self.history.__iter__()

    def __next__(self):
        return self.history.__next__()
    
    def __len__(self):
        return len(self.history)

    def __repr__(self):
        return "Trace: " + repr(self.history)

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
        Representation of element: Trace
        All MActs should be 'pure' in that they don't modify their state after creation.

        MAct and GActMixin should be lifted to a TagTraceGroup
    '''
    def __init__(self, *generating_objects, lift=True):
        super().__init__()
        self._generating_objects = list(generating_objects)  # currently unused

    @staticmethod
    def unit():
        return Trace([])

    @staticmethod
    def compose(m1: Element, m2: Element) -> Element:
        return Element(self, Trace(m1.history + m2.history))

    def action(self, m: Element, x: Any) -> Any:
        ''' 
            Could speed this up. e.g. do products between contiguous strings of elem from the same monoid
            e.g. if a in A, b in B,, etc. 
                aa'babacc'c"abb' -> a"babac"'ab"
            Could do some fancy associativity stuff 
            For now, just a for loop
        '''
        assert isinstance(m, Element)
        x_t = x
        for elem in m.param:
            x_t = elem(x_t)
        return x_t

    # def __rshift__(self, x):
    #     return self.compose(x)


class TraceGroup(TraceMonoid, GActMixin):
    '''

    '''
    @staticmethod
    def inverse(m: Element) -> Element:
        return Element(
            family = m.family, 
            param= Trace([
                m_.family.inverse(m_)
                for m_ in m.param.reverse()
            ])
        ) 

