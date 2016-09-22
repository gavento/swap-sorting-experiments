import random
import math
import functools
import lzma
import pickle

def write_pickle_xz(obj, fname):
    with lzma.open(fname, mode='wb', preset=4) as f:
        return pickle.dump(obj, f, protocol=-1)

def read_pickle_xz(fname):
    with lzma.open(fname, mode='rb') as f:
        return pickle.load(f)


def random_permutation(n, rnd=None):
    """
    Return a random permutation of numbers 1..n,
    optionally using given random.Random instance
    """
    rnd = rnd or random
    l = list(range(n))
    rnd.shuffle(l)
    return l


def p_err_for_T(T, W=1):
    "Compute error probability for given temperature and energy difference."
    assert T >= 0.0 - 1e-14
    if math.isinf(T) or T > 1e100:
        return 0.5
    if T < 1e-100:
        return 0.0
    ewt = math.exp(-W / T)
    return ewt / (1.0 + ewt)

def T_for_p_err(p_err, W=1):
    "Compute corresponding temperature for given energy difference and error probability."
    assert p_err <= 0.5 + 1e-14 and p_err >= 0.0 - 1e-14
    if p_err < 1e-320:
        return 0.0
    if p_err > 0.5 - 1e-15:
        return math.inf
    return -W / (math.log(p_err / (1 - p_err)))


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



