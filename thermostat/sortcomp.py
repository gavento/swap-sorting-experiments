import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
import logging
import math
import pandas

from .utils import random_permutation
from .sortinstance import RepeatedErrorInstance, IndependentErrorInstance
from .statprocess import SortingProcess
from . import log, ErrorCostType, DisorderMeasure


def sorting_process_sample(process, sampleid, steps, name=""):
    "Internal."
    steps = list(steps)
    Es = []
    logNEs = []
    for stval in steps:
        process.do_steps(stval - process.steps)
        Es.append(process.E())
        logNEs.append(math.log(max(process.E(), 1), process.N))

    label = "%s N=%s, T=%g" % (name, process.N, process.T)
    return pandas.DataFrame({
        'N': process.N,
        'T': process.T,
        'sample': sampleid,
        'label': label.strip(),
        'step': steps,
        'E': Es,
        'log_N(E)': logNEs,
        })


def listify(val):
    "Return `list(val)` if `val` is iterable. Return `[val]` otherwise."
    if hasattr(val, '__iter__'):
        return list(val)
    return [val]


def sorting_process_for_ranges(Ns, Ts, samples, steps, rndseed='42', name=None, recurrent_err=False, swap_any=False, error_type=None, measure=None):
    Ns = listify(Ns)
    Ts = listify(Ts)
    if isinstance(samples, int):
        samples = range(samples)
    if isinstance(steps, int):
        steps = range(0, steps + 1, max(1, steps // 1000))

    jobs = []
    for N, T, sample in itertools.product(Ns, Ts, samples):
        rnd = random.Random("%s-%s-%s-%s-%s-%s-%s-%s" % (rndseed, N, T, sample, recurrent_err, swap_any, error_type, measure))
        perm = random_permutation(N, rnd)
        if recurrent_err:
            inst = RepeatedErrorInstance(perm, rnd, T=T, error_cost=error_type)
        else:
            inst = IndependentErrorInstance(perm, rnd, T=T, error_cost=error_type)
        process = SortingProcess(inst, measure=measure, swap_any=swap_any, rnd=rnd, T=T)
        jobs.append(joblib.delayed(sorting_process_sample)(process, sample, steps, name))

    log.info("Submitting %d jobs ...", len(jobs))
    #res = [f(*args, **kwargs) for f, args, kwargs in jobs]
    parallel = joblib.Parallel(n_jobs = -1, verbose = 5)
    res = parallel(jobs)

    return pandas.concat(res, ignore_index=True, copy=False)
    
def plot_sorting_process(df, value='log_N(E)', title=None, ymax=None, ax=None):
    if isinstance(df, str):
        df = pandas.read_pickle(df)
    ax = sns.tsplot(df, time='step', value=value, ci=[68, 95], unit='sample',
                    condition='label', err_style='ci_band', estimator=np.median, ax=ax)
    ax.set_title(title or 'Energy by the number of steps, %d samples, percentiles: 68, 95' % len(df['sample'].unique()))
    ax.set_ylim([0, ymax or int(math.ceil(max(1, df[value].max())))])
    return ax



def main():
    Ns = 100
    Ts = [1.5 ** i for i in range(14)]
    samples = 50
    steps = 10000

    def comp(recurrent_err, swap_any, error_type, measure):
        name = "%s_%s_%s_%s" % (measure.name, error_type.name, ["IND", "REC"][recurrent_err],
                ["ADJ", "ANY"][swap_any])
        fname = "N%d_step%d_%s" % (Ns, steps, name)
        df = sorting_process_for_ranges(Ns=Ns, Ts=Ts, samples=samples, steps=steps, name=name,
                recurrent_err=recurrent_err, swap_any=swap_any, error_type=error_type, measure=measure)
        df.to_pickle(fname + '.pickle')
        plt.figure(figsize=(24, 15))
        plot_sorting_process(df)
        plt.savefig(fname + '.png', bbox_inches='tight', dpi=200)

    # ADJ

    comp(recurrent_err=False, swap_any=False, error_type=ErrorCostType.VALUE_DIST, measure=DisorderMeasure.W_DISLOC)
    
    comp(recurrent_err=True, swap_any=False, error_type=ErrorCostType.VALUE_DIST, measure=DisorderMeasure.W_DISLOC)

    # ANY - reversible (only indep)

    comp(recurrent_err=False, swap_any=True, error_type=ErrorCostType.VALUE_DIST, measure=DisorderMeasure.W_DISLOC)

    # ANY - nonreversible, error by VALUE only

    comp(recurrent_err=False, swap_any=True, error_type=ErrorCostType.VALUE, measure=DisorderMeasure.W_DISLOC)

    comp(recurrent_err=True, swap_any=True, error_type=ErrorCostType.VALUE, measure=DisorderMeasure.W_DISLOC)



