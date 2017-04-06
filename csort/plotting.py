import itertools
import joblib
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas
import random
import seaborn as sns
import time
from contextlib import contextmanager
from joblib import Parallel, delayed

try:
    import csort
except ImportError as e:
    raise ImportError("Module csort not found, compile 'csort.so' with 'make'") from e

############ Plotting utility

@contextmanager
def sns_plot(fbasename):
    plt.clf()
    sns.set_context('paper', rc={"lines.linewidth": 1.1, 'lines.markeredgewidth': 0.1}, font_scale=1.3)
    sns.set_style("ticks")
    cs = sns.color_palette("Set1")

    yield
    
    plt.tight_layout()
    plt.savefig(fbasename+'.pdf', dpi=200)
    plt.savefig(fbasename+'.png', dpi=200)
    plt.clf()

############# Cached computations

mc = joblib.Memory(cachedir='.cache', verbose=0)

@mc.cache
def process_time_cached(p, r, n, steps, samples, extra=None):
    dfs = []
    for i in range(samples):
        rs = csort.RandomSort(p, r, n, max(steps // 30, 1), hash((p, r, n, i)) % 2**30)
        rs.steps(steps)
        data = {
            'p': p, 'r': r, 'n': n, 'sample': i,
            'label': "p = %g r = %d n = %d" % (p,r,n),
            'T': list(rs.Ts), 'I': list(rs.Is), 'W': list(rs.Ws)
            }
        data.update(extra or {})
        dfs.append(pandas.DataFrame(data))
    return pandas.concat(dfs)

def process_time(p, r, n, steps, samples, extras=None):
    return process_time_cached(p, r, n, steps, samples, extras)

def lim_times_part(p, r, n, i, extra):
    rs = csort.RandomSort(p, r, n, -1, hash((p, r, n, i)) % 2**30)
    T = rs.converge_on_W(0.01, -1)
    I = rs.I_stab()
    W = rs.W_stab()
    data = {
        'p': p, 'r': r, 'n': n, 'sample': i,
        'label': "p = %g r = %d n = %d" % (p,r,n),
        'T': T, 'I': I, 'W': W, 'ET': rs.ET(), 'Er': rs.Er(),
        'T/n': T / n, 'I/n': I / n, 'W/n': W / n,
        'T/n^2': T / n ** 2, 'I/n^2': I / n ** 2, 'W/n^2': W / n ** 2,
        'W/n^3': W / n ** 3,
        }
    data.update(extra or {})
    return data

@mc.cache
def lim_times_cached(p, r, n, samples, extra):
    dfs = Parallel(n_jobs=-2, verbose=1, pre_dispatch='all')([delayed(lim_times_part)(p, r, n, i, extra) for i in range(samples)])
    return pandas.DataFrame(dfs)

def lim_times(p, r, n, samples, extra=None):
    return lim_times_cached(p, r, n, samples, extra)

################# Plotting functions

SMP = 4

def fig_IWT_by_n_r1():
    r = 1
    smp = SMP
    markers = ['s','D','o','^','v']
    dfs = []
    ps = [0.05, 0.1, 0.2]
    ns = [32, 64, 128, 256, 512, 1024]
    for p in ps:
        for n in ns:
            dfs.append(lim_times(p, r, n, smp))

    df = pandas.concat(dfs)
    df['Inorm'] = df['I'] / df['n']**1 #/ df['p']
    df['Wnorm'] = df['W'] / df['n']**1 #/ df['p']
    df['T/n^2'] = df['T'] / df['n']**2

    with sns_plot("fig-r1-lim-IW-by-n_s%s" % (smp, )):
        sns.pointplot(y='Inorm', x='n', hue='p', data=df, markers=markers, dodge=True)
        sns.pointplot(y='Wnorm', x='n', hue='p', data=df, markers=markers, linestyles='--', dodge=True)
        plt.ylim(ymin = 0.0)
        sns.despine(right=True, top=True, offset=1, trim=True)
        plt.ylabel("I/n, W/n (dashed)")
        plt.legend(loc=0, title="p")

    with sns_plot("fig-r1-lim-T-by-n_s%s" % (smp, )):
        sns.pointplot(y='T/n^2', x='n', hue='p', data=df, markers=markers, dodge=True)
        sns.despine(right=True, top=True, offset=1, trim=True)
        plt.legend(loc=0, title="p")

def fig_IWT_by_n_rn():

    smp = SMP
    markers = ['s','D','o','^','v']
    dfs = []
    ps = [0.05, 0.1, 0.2]
    ns = [32, 64, 128, 256, 512, 1024]
    for p in ps:
        for n in ns:
            dfs.append(lim_times(p, n, n, smp))

    df = pandas.concat(dfs)
    df['Inorm'] = df['I'] / df['n']**2
    df['Wnorm'] = df['W'] / df['n']**3
    df['T/n'] = df['T'] / df['n']

    with sns_plot("fig_IW_by_n-rn-s%s" % (smp, )):
        sns.pointplot(y='Inorm', x='n', hue='p', data=df, markers=markers, dodge=True)
        sns.pointplot(y='Wnorm', x='n', hue='p', data=df, markers=markers, linestyles='--', dodge=True)
        sns.despine(right=True, top=True, offset=1, trim=True)
        plt.ylim(ymin = 0.0)
        plt.ylabel("I/n^2, W/n^3 (dashed)")
        plt.legend(loc=0, title="p")

    with sns_plot("fig_T_by_n-rn-s%s" % (smp, )):
        sns.pointplot(y='T/n', x='n', hue='p', data=df, markers=markers, dodge=True)
        sns.despine(right=True, top=True, offset=1, trim=True)
        plt.legend(loc=0, title="n")

def fig_IW_by_p_r1():
    r = 1
    smp = SMP
    markers = ['s','D','o','^','v']
    ps = np.arange(0.0, 0.21, 0.05)
    ns = [64, 512]
    dfs = []
    for p in ps:
        for n in ns:
            dfs.append(lim_times(p, r, n, smp))

    df = pandas.concat(dfs)
    df['Inorm'] = df['I'] / df['n']**1
    df['Wnorm'] = df['W'] / df['n']**1
    dfun2 = pandas.DataFrame({'p': ps, 'EW/n': [2*p*(1-p)/(1-2*p)**2 for p in ps]})

    with sns_plot("fig_IW_by_p-r1-s%s" % (smp, )):
        sns.pointplot(y='Inorm', x='p', hue='n', data=df, markers=markers, dodge=True)
        sns.pointplot(y='Wnorm', x='p', hue='n', data=df, markers=markers, linestyles='--', dodge=True)
        sns.pointplot(y='EW/n', x='p', data=dfun2, markers='+', color='black', label='Theor. bound on E[W]')
        plt.ylim(ymin = 0)
        sns.despine(right=True, top=True, offset=1, trim=True)
        plt.legend(loc=0, title="n")

def fig_IW_by_p_rn():
    smp = SMP
    markers = ['s','D','o','^','v']
    ps = np.arange(0.02, 0.21, 0.02)
    ns = [64, 512]
    dfs = []
    for p in ps:
        for n in ns:
            dfs.append(lim_times(p, n, n, smp))

    df = pandas.concat(dfs)
    df['Inorm'] = df['I'] / df['n']**2
    df['Wnorm'] = df['W'] / df['n']**3

    with sns_plot("fig_IW_by_p-rn-s%s" % (smp, )):
        sns.pointplot(y='Inorm', x='p', hue='n', data=df, markers=markers, dodge=True)
        sns.pointplot(y='Wnorm', x='p', hue='n', data=df, markers=markers, linestyles='--', dodge=True)
        plt.ylim(ymin = 0)
        sns.despine(right=True, top=True, offset=1, trim=True)
        plt.legend(loc=0, title="n")

def fig_IWT_by_r():
    n = 512
    smp = SMP
    markers = ['s','D','o','^','v']
    dfs = []
    ps = [0.05, 0.1, 0.2, 0.3]
    rs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    for p in ps:
        for r in rs:
            dfs.append(lim_times(p, r, n, smp))

    df = pandas.concat(dfs)

    with sns_plot("fig_I_by_r-n%d-s%s" % (n, smp, )):
        sns.pointplot(y='I', x='r', hue='p', data=df, markers=markers, dodge=True)
        plt.yscale('log', basey=2)
        plt.legend(loc=4, title="p")

    with sns_plot("fig_W_by_r-n%d-s%s" % (n, smp, )):
        sns.pointplot(y='W', x='r', hue='p', data=df, markers=markers, dodge=True)
        plt.yscale('log', basey=2)
        plt.legend(loc=4, title="p")

    with sns_plot("fig_T_by_r-n%d-s%s" % (n, smp, )):
        sns.pointplot(y='T', x='r', hue='p', data=df, markers=markers, dodge=True)
        plt.yscale('log', basey=2)
        plt.legend(loc=0, title="p")

#########################

def plot_time(p, r, n, steps, samples, marker='.', color='blue', label=None, value='I'):
    df = process_time(p, r, n, steps, samples, label=label)
    sns.tsplot(df, time='T', value=value, ci=[99], unit='sample',
            condition='label', err_style=None, estimator=np.mean, marker=marker, color=color)
    return df

def fig_IW_by_T():
    T = 40000
    n = 256
    p = 0.1
    smp = SMP
    cs = sns.color_palette("Set1")
    rs = [1, 4, 16, 64, 256]
    markers = ['s', 'D', 'o', '^', 'v', 'd']

    for v in ['I', 'W']:
        with sns_plot("fig_%s_by_T-n%d-s%d" % (v, n, smp, )):
            for r, c, m in zip(rs, cs, markers):
                plot_time(p, r, n, T, smp, m, c, str(r), value=v)
            plt.ylim(ymin = -0.02 * plt.ylim()[1])
            plt.xlim(-0.02 * T, 1.02 * T)
            sns.despine(right=True, top=True, offset=0, trim=True)
            plt.legend(loc=0, title="r")

def fig_IW_by_T_single():
    T = 40000
    n = 256
    p = 0.1
    cs = sns.color_palette("Set1")
    rs = [1, 64]
    markers = ['s', 'D', 'o', '^', 'v', 'd']

    for v in ['I', 'W']:
        with sns_plot("fig_%s_by_T_single-n%d" % (v, n, )):
            for r, c, m in zip(rs, cs, markers):
                plot_time(p, r, n, T, 1, m, c, str(r), value=v) # TODO: more lines
            plt.ylim(ymin = -0.02 * plt.ylim()[1])
            plt.xlim(-0.02 * T, 1.02 * T)
            sns.despine(right=True, top=True, offset=0, trim=True)
            plt.legend(loc=0, title="r")

############### Plot all

if __name__ == '__main__':
    import sys
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    def r(f, *args, **kw):
        t0 = time.time()
        logging.info('Running %s ...' % f.__name__)
        f(*args, **kw)
        logging.info('... %s done in %gs (wall).' % (f.__name__, time.time() - t0))

    r(fig_IWT_by_n_r1)
    r(fig_IWT_by_n_rn)
    r(fig_IW_by_p_r1)
    r(fig_IW_by_p_rn)
    r(fig_IWT_by_r)
    r(fig_IW_by_T)
    r(fig_IW_by_T_single)


############### Unused

def fig_lim_T_vs_ET(p=0.1):

    smp = SMP
    dfs = []
    ns = [128, 256, 512, 1024]#, 2048]
    rnames = ['1', 'n^1/4', 'n^1/2', 'n^3/4', 'n']
    for n in ns:
        for ri, r in enumerate([1, int(n**0.25), int(n**0.5), int(n**0.75), n]):
            logging.info("Parallel: done %d of %d jobs.", len(dfs), 4 * len(ns))
            dfs.append(lim_times(p, r, n, smp, {'ri': rnames[ri]}))
    df = pandas.concat(dfs)
    df['T_conv/T_cest'] = df['T'] / df['n']**2 * df['Er'] #/ (1+0.25*np.log2(df['r'])))

    with sns_plot("fig-lim-T-vs-ET_p%g_s%d" % (p, smp,)):
        sns.boxplot(y='T_conv/T_cest', x='n', hue='ri', data=df)
        sns.despine(right=True, top=True, offset=1, trim=True)
        plt.legend(loc=0, title="Max swap distance")

def fig_process_by_p(v='I'):
    set_sns()
    cs = sns.color_palette("Set1")

    T = 1400000
    n = 512
    r = 1
    smp = SMP
    plt.clf()
    plot_time(0.0, r, n, T, smp, 's', cs[0], '0.0', value=v)
    plot_time(0.1, r, n, T, smp, 'D', cs[1], '0.1', value=v)
    plot_time(0.2, r, n, T, smp, 'o', cs[2], '0.2', value=v)
    plot_time(0.3, r, n, T, smp, '^', cs[3], '0.3', value=v)
    plot_time(0.4, r, n, T, smp, 'v', cs[4], '0.4', value=v)

    plt.ylim(ymin = -0.02 * plt.ylim()[1])
    plt.xlim(-0.02 * T, 1.02 * T)
    sns.despine(right=True, top=True, offset=0, trim=True)
    plt.axes().legend().set_title("Error probability")
    plt.tight_layout()
    plt.savefig('fig-process-' + v + '-by-p.pdf', dpi=200)

