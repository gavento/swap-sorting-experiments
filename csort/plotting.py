import joblib
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
import logging
import math
import pandas
#from dask import delayed, compute
#import dask.dataframe as dd
#import dask.multiprocessing, dask.bag
#from scoop import futures

try:
    import csort
except ImportError as e:
    raise ImportError("Module csort not found, compile 'csort.so' with 'make'") from e

#try:
#    from distributed import LocalCluster, Client
#    cluster = LocalCluster(4, 1)
#    client = Client(cluster)
#except ImportError as e:
#    raise ImportError("Module distributed not found (needed for parallel processing)") from e

mc = joblib.Memory(cachedir='.cache', verbose=0)

@mc.cache
def process_time_cached(p, r, n, steps, samples):
    dfs = []
    for i in range(samples):
        rs = csort.RandomSort(p, r, n, max(steps // 30, 1), hash((p, r, n, i)) % 2**30)
        rs.steps(steps)
        dfs.append(pandas.DataFrame({
            'p': p, 'r': r, 'n': n, 'sample': i,
            'label': label or ("p = %g r = %d n = %d" % (p,r,n)),
            'T': list(rs.Ts), 'I': list(rs.Is), 'W': list(rs.Ws)
            }))
    return pandas.concat(dfs)

def process_time(p, r, n, steps, samples):
    return process_time_cached(p, r, n, steps, samples)

@mc.cache
def lim_times_cached(p, r, n, samples, use_I=True, label=None):
    dfs = []
    for i in range(samples):
        rs = csort.RandomSort(p, r, n, -1, hash((p, r, n, i)) % 2**30)
        T = rs.run_conv(use_I)
        rs.steps(T // 2)
        dfs.append({
            'p': p, 'r': r, 'n': n, 'sample': i,
            'label': label or ("p = %g r = %d n = %d" % (p,r,n)),
            'T': T, 'I': rs.Is[-1], 'W': rs.Ws[-1],
            'T/n': T / n, 'I/n': rs.Is[-1] / n, 'W/n': rs.Ws[-1] / n,
            'T/n^2': T / n ** 2, 'I/n^2': rs.Is[-1] / n ** 2, 'W/n^2': rs.Ws[-1] / n ** 2,
            'W/n^3': rs.Ws[-1] / n ** 3,
            })
    df = pandas.DataFrame(dfs)
    return df

def lim_times(p, r, n, sample, use_I=True, label=None):
    return lim_times_cached(p, r, n, sample, use_I=use_I, label=label)

#################3

def set_sns():
    sns.set_context('paper', rc={"lines.linewidth": 1.1, 'lines.markeredgewidth': 0.1}, font_scale=1.3)
    sns.set_style("ticks")
    cs = sns.color_palette("Set1")

def fig_rn_lim_by_n():

    smp = 100
    markers = ['s','D','o','^','v']
    dfs_par = []
    for p in [0.05, 0.1, 0.15, 0.2]:
        for n in [32, 64, 128, 256, 512, 1024, 2048]:
            dfs_par.append((p, n, n, smp))

    dfs = Parallel(n_jobs=-1, verbose=5, pre_dispatch='all')([delayed(lim_times)(*pars) for pars in dfs_par])
    df = pandas.concat(dfs)
    dfavg = df.groupby((df.p, df.r, df.n), as_index=False).mean()

    set_sns()
    plt.clf()
    sns.pointplot(y='I/n^2', x='n', hue='p', data=df, markers=markers, dodge=True)
    sns.pointplot(y='W/n^3', x='n', hue='p', data=df, markers=markers, linestyles='--', dodge=True)
    sns.despine(right=True, top=True, offset=1, trim=True)
    plt.legend(loc=0, title="n")
    plt.tight_layout()
    plt.savefig('fig-rn-lim-IW-by-n_s100.pdf', dpi=200)

    plt.clf()
    sns.pointplot(y='T/n', x='n', hue='p', data=df, markers=markers, dodge=True)
    sns.despine(right=True, top=True, offset=1, trim=True)
    plt.legend(loc=0, title="n")
    plt.tight_layout()
    plt.savefig('fig-rn-lim-T-by-n_s100.pdf', dpi=200)


def fig_r1_lim_by_p():
    r = 1
    smp = 500
    markers = ['s','D','o','^','v']
    dfs_par = []
    ps = np.arange(0.0, 0.41, 0.05)
    for p in ps:
        for n in [64, 128, 256]:
            dfs_par.append((p, r, n, smp))

    dfs = Parallel(n_jobs=-1, verbose=5, pre_dispatch='all')([delayed(lim_times)(*pars) for pars in dfs_par])
    df = pandas.concat(dfs)
    dfavg = df.groupby((df.p, df.r, df.n), as_index=False).mean()
    
    dfun = pandas.DataFrame({'p': ps, 'EI/n': [2*p*(1-p)/(3*p-1)**2 for p in ps]})

    set_sns()
    plt.clf()
    sns.pointplot(y='I/n', x='p', hue='n', data=df, markers=markers, dodge=True)
    sns.pointplot(y='EI/n', x='p', data=dfun, color='black')
    sns.pointplot(y='W/n', x='p', hue='n', data=df, markers=markers, linestyles='--', dodge=True)
    plt.ylim(ymin = -0, ymax=3)
    sns.despine(right=True, top=True, offset=1, trim=True)
    plt.legend(loc=0, title="n")
    plt.tight_layout()
    plt.savefig('fig-r1-lim-IW-by-p_s100.pdf', dpi=200)

def fig_r1_lim_by_n():
    r = 1
    smp = 100
    markers = ['s','D','o','^','v']
    dfs_par = []
    for p in [0.05, 0.1, 0.15, 0.2]:
        for n in [32, 64, 128, 256, 512, 1024]:
            dfs_par.append((p, r, n, smp))

    dfs = Parallel(n_jobs=-1, verbose=5, pre_dispatch='all')([delayed(lim_times)(*pars) for pars in dfs_par])
    df = pandas.concat(dfs)
    dfavg = df.groupby((df.p, df.r, df.n), as_index=False).mean()

    set_sns()
    plt.clf()
    sns.pointplot(y='I/n', x='n', hue='p', data=df, markers=markers, dodge=True)
    sns.pointplot(y='W/n', x='n', hue='p', data=df, markers=markers, linestyles='--', dodge=True)
    plt.ylim(ymin = -0.05, ymax=1.1)
    sns.despine(right=True, top=True, offset=1, trim=True)
    plt.legend(loc=0, title="Error probability")
    plt.tight_layout()
    plt.savefig('fig-r1-lim-IW-by-n_s100.pdf', dpi=200)

    plt.clf()
    sns.pointplot(y='T/n^2', x='n', hue='p', data=df, markers=markers, dodge=True)
    sns.despine(right=True, top=True, offset=1, trim=True)
    plt.legend(loc=0, title="Error probability")
    plt.tight_layout()
    plt.savefig('fig-r1-lim-T-by-n_s100.pdf', dpi=200)


##################################

def fig_lim_by_r():
    n = 512
    smp = 100
    markers = ['s','D','o','^','v']
    dfs = []
    for p in [0.05, 0.1, 0.2, 0.3]:
        for r in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            dfs.append((p, r, n, smp))
    dfs = map(lim_times, *list(zip(*dfs)))
    df = pandas.concat(dfs)
    dfavg = df.groupby((df.p, df.r, df.n), as_index=False).mean()

    set_sns()
    plt.clf()
    sns.pointplot(y='I', x='r', hue='p', data=df, markers=markers, dodge=True)
    plt.yscale('log', basey=2)
    plt.legend(loc=4, title="Error probability")
    plt.tight_layout()
    plt.savefig('fig-lim-I-by-r_n512_s100.pdf', dpi=200)

    plt.clf()
    sns.pointplot(y='W', x='r', hue='p', data=df, markers=markers, dodge=True)
    plt.yscale('log', basey=2)
    plt.legend(loc=4, title="Error probability")
    plt.tight_layout()
    plt.savefig('fig-lim-W-by-r_n512_s100.pdf', dpi=200)

    plt.clf()
    sns.pointplot(y='T', x='r', hue='p', data=df, markers=markers, dodge=True)
    plt.yscale('log', basey=2)
    plt.legend(loc=0, title="Error probability")
    plt.tight_layout()
    plt.savefig('fig-lim-T-by-r_n512_s100.pdf', dpi=200)

#########################

def plot_time(p, r, n, steps, samples, marker='.', color='blue', label=None, value='I'):
    df = process_time(p, r, n, steps, samples)
    sns.tsplot(df, time='T', value=value, ci=[99], unit='sample',
            condition='label', err_style=None, estimator=np.mean, marker=marker, color=color)
    return df

def fig_process_by_r(v='I'):
    set_sns()
    cs = sns.color_palette("Set1")

    T = 130000
    n = 512
    p = 0.1
    smp = 100
    plot_time(p, 1, n, T, smp, 's', cs[0], '1', value=v)
    plot_time(p, 4, n, T, smp, 'D', cs[1], '4', value=v)
    plot_time(p, 16, n, T, smp, 'o', cs[2], '16', value=v)
    plot_time(p, 64, n, T, smp, '^', cs[3], '64', value=v)
    plot_time(p, 256, n, T, smp, 'v', cs[4], '256', value=v)
    plot_time(p, 512, n, T, smp, 'd', cs[5], '512', value=v)

    plt.ylim(ymin = -0.02 * plt.ylim()[1])
    plt.xlim(-0.02 * T, 1.02 * T)
    sns.despine(right=True, top=True, offset=0, trim=True)
    plt.axes().legend().set_title("Max swap distance")
    plt.tight_layout()
    plt.savefig('fig-process-' + v + '-by-r_n512_s100_p01.pdf', dpi=200)

def fig_process_by_p(v='I'):
    set_sns()
    cs = sns.color_palette("Set1")

    T = 1200000
    n = 512
    r = 1
    smp = 100
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


if __name__ == '__main__':
    fig_rn_lim_by_n()
    fig_r1_lim_by_p()
    fig_r1_lim_by_n()
    fig_lim_by_r()
    fig_process_by_n()
    fig_process_by_p()
