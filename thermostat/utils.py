import random
import functools

def random_permutation(n, rnd=None):
    """
    Return a random permutation of numbers 1..n,
    optionally using given random.Random instance
    """
    rnd = rnd or random
    l = list(range(n))
    rnd.shuffle(l)
    return l


class SimpleMemoClass:
    """
    A simple class memoizing the results of some methods.
    The methods should have no parameters (no parameter chaching is done.

    Decorate with:

        @SimpleCachingClass._cached
        def E(self): ...

    Invalidate with 
        
        self.invalidate_cache()
    """

    def __init__(self):
        self._cache = {}

    def invalidate_cache(self):
        self._cache = {}

    @classmethod
    def _cached(cls, meth):
        functools.wraps(meth)
        def newmeth(self):
            name = meth.__name__
            if name in self._cache:
                return self._cache[name]
            val = meth(self)
            self._cache[name] = val
            return val
        return newmeth



