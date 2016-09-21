import random
import pickle
import copy
import enum
import joblib
import math
import functools
import itertools
import numpy as np
import pandas
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats.kde
import logging

from .utils import SimpleMemoClass, random_permutation
from . import log, DisorderMeasure

class StatProcess(SimpleMemoClass):

    def __init__(self, state=None, params={}, rnd=None, T=1.0, steps=0, N=1):
        super().__init__()
        self.state = state
        self.params = params
        self.rnd = rnd or random.Random()
        self.T = float(T)
        self.steps = steps
        self.N = N

    def copy(self):
        return copy.deepcopy(self)

    def branch(self, subkey=1):
        c = copy.deepcopy(self)
        for i in range(subkey):
            c.rnd.random()
        return c


    def __repr__(self):
        return "<%s N=%d @%d T=%.3e E=%.3e>" % (self.__class__.__name__, self.N, self.steps, self.T, self.E())

    @SimpleMemoClass._cached
    def E(self):
        raise NotImplemented

    def do_steps(self, numsteps=1):
        assert numsteps >= 0
        for i in range(numsteps):
            self.invalidate_cache()
            self.do_step()

    def do_step(self):
        self.steps += 1
        raise NotImplemented



class SortingProcess(StatProcess):
    """A process in permutation space swapping random (adjacent/any) elements
    according to the instance cmp() function.
    """

    def __init__(self, instance, measure=DisorderMeasure.INV, swap_any=False, rnd=None, T=1.0):
        self.inst = instance
        data = self.inst.get_sequence()
        self.measure = measure
        self.swap_any = swap_any
        self.swaps = 0
        super().__init__(state=np.array(data), rnd=rnd, T=T, N=len(data))

    @SimpleMemoClass._cached
    def total_dislocation(self):
        return self.inst.total_dislocation(self.state)

    @SimpleMemoClass._cached
    def total_weigted_dislocation(self):
        return self.inst.total_weighted_dislocation(self.state)

    @SimpleMemoClass._cached
    def total_inversions(self):
        return self.inst.total_inversions(self.state)

    def E(self):
        if self.measure == DisorderMeasure.INV:
            return self.total_inversions()
        elif self.measure == DisorderMeasure.DISLOC:
            return self.total_dislocation()
        elif self.measure == DisorderMeasure.W_DISLOC:
            return self.total_weigted_dislocation()
        else:
            raise ValueError("Invalid disorder measure")

    def do_step(self):
        self.steps += 1

        if self.N <= 1:
            return

        # Generate two different indices, ai < bi
        ai = self.rnd.randint(0, self.N - 2)
        if self.swap_any:
            bi = self.rnd.randint(ai + 1, self.N - 1)
        else:
            bi = ai + 1
        a = self.state[ai]
        b = self.state[bi]

        if self.inst.cmp(a, b, dist=(bi - ai)) > 0:
            self.swaps += 1
            self.state[ai] = b
            self.state[bi] = a


