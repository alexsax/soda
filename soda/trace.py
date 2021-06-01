
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