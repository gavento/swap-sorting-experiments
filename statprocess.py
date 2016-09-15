import random
import pickle
import copy
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

from utils import SimpleMemoClass, random_permutation

from . import log

class StatProcessState(SimpleMemoClass):

    def __init__(self, state=None, params={}, rnd=None, T=1.0, steps=0, size=1):
        super().__init__()
        self.state = state
        self.params = params
        self.rnd = rnd or random.Random()
        self.T = float(T)
        self.steps = steps
        self.size = size

    def copy(self):
        return copy.deepcopy(self)

    def branch(self, subkey=1):
        c = copy.deepcopy(self)
        for i in range(subkey):
            c.rnd.random()
        return c


    def __repr__(self):
        return "<%s [%d] @%d T=%.3e E=%.3e>" % (self.__class__.__name__, self.size, self.steps, self.T, self.E())

    @SimpleMemoClass._cached
    def E(self):
        raise NotImplemented

    def do_steps(self, numsteps=1):
        assert numsteps >= 0
        for i in range(numsteps):
            self.invalidate_cache()
            self.do_step()
            self.steps += 1

    def do_step(self):
        raise NotImplemented

    

class SortingProcessState(StatProcessState):

    def __init__(self, sequence, rnd=None, T=1.0):
        super().__init__(state=np.array(sequence), rnd=rnd, T=T, size=len(sequence))
        assert len(sequence) == len(set(sequence)) # Assert unique values for now
        self.data_sorted = sorted(self.state)
        self.data_rank = {v: i for i, v in enumerate(self.data_sorted)}

    def dislocation(self, v, idx):
        return abs(idx - self.data_rank[v])

    @SimpleMemoClass._cached
    def total_dislocation(self):
        return sum(self.dislocation(v, i) for i, v in enumerate(self.state))

    def weighted_dislocation(self, v, idx):
        return v * (self.data_rank[v] - idx)

    @SimpleMemoClass._cached
    def total_weigted_dislocation(self):
        return sum(self.weighted_dislocation(v, i) for i, v in enumerate(self.state))

    @SimpleMemoClass._cached
    def total_inversions(self):
        return 0 # TODO


class SwapSortingProcessState(SortingProcessState):

    PARAM_W_DISLOC = 'WDis'
    PARAM_DISLOC = 'Dis'
    PARAM_INV = 'Inv'
    PARAMS = (PARAM_W_DISLOC, PARAM_DISLOC, PARAM_INV)

    STEP_ADJ = 'Adj'
    STEP_ANY = 'Any'
    STEP_REV_ANY = 'RAny'
    STEPS = (STEP_ADJ, STEP_ANY, STEP_REV_ANY)

    def __init__(self, sequence, rnd=None, T=1.0, param=PARAM_W_DISLOC, step_type=STEP_ADJ):
        assert param in self.PARAMS
        self.param = param
        assert step_type in self.STEPS
        self.step_type = step_type
        super().__init__(sequence, rnd=rnd, T=T)

    def E(self):
        if self.param == self.PARAM_W_DISLOC:
            return self.total_weigted_dislocation()
        elif self.param == self.PARAM_DISLOC:
            return self.total_dislocation()
        else:
            return self.total_inversions()

    def do_step(self):
        if self.size <= 1:
            return

        # Generate two different indices
        if self.step_type == self.STEP_ADJ:
            ai = self.rnd.randint(0, self.size - 2)
            bi = ai + 1
        else:
            ai = self.rnd.randint(0, self.size - 2)
            bi = self.rnd.randint(ai + 1, self.size - 1)
        assert ai < bi
        a = self.state[ai]
        b = self.state[bi]

        if self.param == self.PARAM_W_DISLOC:
            # Math range bounds and T == 0
            if self.T < 1e-50:
                if a > b:
                    p = 1.0
                else:
                    p = 0.0
            elif (a - b) / self.T < -100:
                p = 0.0
            elif (a - b) / self.T > 100:
                p = 1.0
            else:
                p = math.exp((a - b) / self.T) / ( math.exp((a - b) / self.T) + math.exp((b - a) / self.T) )

            if self.step_type == self.STEP_REV_ANY:
                p = p ** abs(ai - bi)

            if self.rnd.random() < p:
                self.state[ai] = b
                self.state[bi] = a

        else:
            raise NotImplemented

###

def Es_by_time(size, in_steps, samples, **kwargs):
    Es = [ [] for i in range(len(in_steps)) ]
    for i in range(samples):
        proc = SwapSortingProcessState(random_permutation(size), **kwargs)
        for sti, stval in enumerate(in_steps):
            proc.do_steps(stval - proc.steps)
            Es[sti].append(proc.E())
    for sti, stval in enumerate(in_steps):
        print("step %7d: avg(E)=%f" % (stval, sum(Es[sti]) / len(Es[sti]) ))
    return Es

def plot_Es_by_time(size, in_steps, *args, **kwargs):
    Emap = lambda e: math.log(max(1, 12 * e), size)

    Es = Es_by_time(size, in_steps, *args, **kwargs)
    dist_space = np.linspace(0, max(max(map(Emap, Ess)) for Ess in Es), 500)
    for sti, stval in enumerate(in_steps):
        kde = scipy.stats.kde.gaussian_kde(list(map(Emap, Es[sti])))
        plt.plot(dist_space, kde(dist_space), label=("step %d" % stval))
    plt.legend()
    plt.show()

###

def Es_by_T(size, in_Ts, steps, samples, **kwargs):
    Es = [ [] for i in range(len(in_Ts)) ]
    for i in range(samples):
        for Ti, Tval in enumerate(in_Ts):
            proc = SwapSortingProcessState(random_permutation(size), T=Tval, **kwargs)
            proc.do_steps(steps)
            Es[Ti].append(proc.E())
    for Ti, Tval in enumerate(in_Ts):
        print("T %7g: avg(E)=%f" % (Tval, sum(Es[Ti]) / len(Es[Ti]) ))
    return Es

def plot_Es_by_T(size, in_Ts, *args, **kwargs):
    Emap = lambda e: e

    Es = Es_by_T(size, in_Ts, *args, **kwargs)
    dist_space = np.linspace(0, max(max(map(Emap, Ess)) for Ess in Es), 500)
    for Ti, Tval in enumerate(in_Ts):
        kde = scipy.stats.kde.gaussian_kde(list(map(Emap, Es[Ti])))
        plt.plot(dist_space, kde(dist_space), label=("T=%g" % Tval))
    plt.legend()
    plt.show()

###

def DF_for_ranges__getrows(size, T, sample, steps, rnd, **kwargs):
    res = []
    proc = SwapSortingProcessState(random_permutation(size, rnd), T=T, rnd=rnd, **kwargs)
    for sti, stval in enumerate(steps):
        proc.do_steps(stval - proc.steps)
        E = proc.E()
        res.append( (size, T, sample, proc.steps, E, math.log(max(E, 1), size)) )
    return res

def DF_for_ranges(sizes, Ts, samples, steps, rnd=None, **kwargs):
    if isinstance(sizes, int):
        sizes = [sizes]
    if isinstance(Ts, int) or isinstance(Ts, float):
        Ts = [Ts]
    if isinstance(samples, int):
        samples = range(samples)
    if isinstance(steps, int):
        steps = range(0, steps + 1, max(1, steps // 500))
    if rnd is None:
        rnd = random.Random()

    parallel = joblib.Parallel(n_jobs = -1, verbose = 5)
    jobs = []
    for size, T, sample in itertools.product(sizes, Ts, samples):
#        rnd.random() # move the state forward
#        rnd2 = random.Random(str(rnd.getstate()) + str((size, T, sample)))
        rnd2 = random.Random()
        jobs.append(joblib.delayed(DF_for_ranges__getrows)(size, T, sample, steps, rnd2, **kwargs))
    log.info("Submitting %d jobs ...", len(jobs))
    res = parallel(jobs)

    df = pandas.DataFrame(np.array(list(itertools.chain(*res))), columns=['size', 'T', 'sample', 'step', 'E', 'Eexp'])
    return df

def plot_ranges_by_steps(sizes, Ts, samples, steps, rnd=None, **kwargs):
    df = DF_for_ranges(sizes, Ts, samples, steps, rnd=rnd, **kwargs)
    for size in df['size'].unique():
        ax = sns.tsplot(df[df['size'] == size], time='step', value='Eexp', ci=[68, 95], unit='sample', condition='T', err_style='ci_band', estimator=np.mean)
        ax.set_title('Energy exponent (of N) by time, N=%d, %d samples, percentiles: 68, 90' % (size, len(df['sample'].unique())))
        ax.set_ylim([0, 3])
        sns.plt.show()


