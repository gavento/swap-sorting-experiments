import joblib
import seaborn as sns
import numpy as np
import random
import itertools
import logging
import math
import pandas

from utils import random_permutation

log = logging.getLogger()
log.setLevel(logging.DEBUG)
log = logging


def sorting_process_sample(N, T, sampleid, steps, rndseed, constructor, constructor_kwargs={}, name=None):
    rnd = random.Random("%s-%s-%s-%s" % (rndseed, N, T, sampleid))
    perm = random_permutation(N, rnd)
    process = constructor(perm, T=T, rnd=rnd, **constructor_kwargs)

    if isinstance(steps, int):
        steps = range(0, steps + 1, max(1, steps // 1000))
    steps = list(steps)
    Es = []
    logNEs = []
    for stval in steps:
        process.do_steps(stval - process.steps)
        Es.append(process.E())
        logNEs.append(math.log(max(process.E(), 1), N))

    if name is None:
        name = constructor.__name__
    label = "%s N=%s, T=%g" % (name, N, T)
    return pandas.DataFrame({
        'N': N,
        'T': T,
        'sample': sampleid,
        'label': label,
        'step': steps,
        'E': Es,
        'log_N(E)': logNEs,
        })

def sorting_process_ranges(Ns, Ts, samples, steps, constructor, rndseed='42', constructor_kwargs={}, name=None):
    if isinstance(Ns, int):
        Ns = [Ns]
    if isinstance(Ts, int) or isinstance(Ts, float):
        Ts = [Ts]
    if isinstance(samples, int):
        samples = range(samples)
    parallel = joblib.Parallel(n_jobs = -1, verbose = 5)
    jobs = []
    for N, T, sample in itertools.product(Ns, Ts, samples):
        jobs.append(joblib.delayed(sorting_process_sample)(N, T, sample, steps, rndseed, constructor, constructor_kwargs, name))
    log.info("Submitting %d jobs ...", len(jobs))
    res = parallel(jobs)

#    return res
    return pandas.concat(res, ignore_index=True, copy=False)
    
def plot_sorting_process(df, value='log_N(E)', title=None, ymax=None, ax=None):
    if isinstance(df, str):
        df = pandas.read_pickle(df)
    ax = sns.tsplot(df, time='step', value=value, ci=[68, 95], unit='sample',
                    condition='label', err_style='ci_band', estimator=np.median, ax=ax)
    ax.set_title(title or 'Energy by the number of steps, %d samples, percentiles: 68, 95' % len(df['sample'].unique()))
    ax.set_ylim([0, ymax or int(math.ceil(max(1, df[value].max())))])
    return ax
    

