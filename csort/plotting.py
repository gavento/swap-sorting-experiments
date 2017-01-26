import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
import logging
import math
import pandas
from dask import delayed, compute
import dask.dataframe as dd
import dask.multiprocessing, dask.bag
from scoop import futures

try:
    import csort
except ImportError as e:
    raise ImportError("Module csort not found, compile 'csort.so' with 'make'") from e

try:
    from distributed import LocalCluster, Client
#    cluster = LocalCluster(4, 1)
#    client = Client(cluster)
except ImportError as e:
    raise ImportError("Module distributed not found (needed for parallel processing)") from e

mc = joblib.Memory(cachedir='.cache', verbose=0)

#################3
def main():
    fig_lim_by_r();


@mc.cache
def lim_times(p, r, n, samples, use_I=True, label=None):
    dfs = []
    print('computing:', p, r, samples)
    for i in range(samples):
        rs = csort.RandomSort(p, r, n, -1, hash((p, r, n, i)) % 2**30)
        T = rs.run_conv(use_I)
        dfs.append({
            'p': p, 'r': r, 'n': n, 'sample': i,
            'label': label or ("p = %g r = %d n = %d" % (p,r,n)),
            'T': T, 'I': rs.Is[-1], 'W': rs.Ws[-1],
            'T/n^2': T / n ** 2, 'I/n': rs.Is[-1] / n, 'W/n': rs.Ws[-1] / n,
            })
    df = pandas.DataFrame(dfs)
    print('done:', p, r, samples)
    return df

def lim_times2(p, r, n, samples, use_I=True, label=None):
    return lim_times(p, r, n, samples, use_I=use_I, label=label)

def fig_lim_by_r():
    sns.set_context('paper', rc={"lines.linewidth": 1.1, 'lines.markeredgewidth': 0.1}, font_scale=1.3)
    sns.set_style("ticks")
    cs = sns.color_palette("Set1")

    n = 512
    smp = 100
    markers = ['s','D','o','^','v']
    dfs = []
    for p in [0.05, 0.1, 0.2, 0.3]:
        ys = []
        xs = []
        for i, r in enumerate([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]):
#            df0 = delayed(lim_times)(p, r, n, smp)
            dfs.append((p, r, n, smp))
    dfs = futures.map(lim_times2, *list(zip(*dfs)))
    df = pandas.concat(dfs)
    dfavg = df.groupby((df.p, df.r, df.n), as_index=False).mean()

    plt.clf()
    #sns.boxplot(y='I', x='r', hue='p', data=df, whis=[5, 95])
    sns.pointplot(y='I', x='r', hue='p', data=df, markers=markers, dodge=True)
    plt.yscale('log', basey=2)
    plt.legend(loc=4, title="Error probability")
    plt.tight_layout()
    plt.savefig('fig-lim-I-by-r_n512_s100.pdf', dpi=200)

    plt.clf()
    #sns.boxplot(y='W', x='r', hue='p', data=df, whis=[5, 95])
    sns.pointplot(y='W', x='r', hue='p', data=df, markers=markers, dodge=True)
    plt.yscale('log', basey=2)
    plt.legend(loc=4, title="Error probability")
    plt.tight_layout()
    plt.savefig('fig-lim-W-by-r_n512_s100.pdf', dpi=200)

    plt.clf()
    #sns.boxplot(y='T', x='r', hue='p', data=df, whis=[5, 95])
    sns.pointplot(y='T', x='r', hue='p', data=df, markers=markers, dodge=True)
    plt.yscale('log', basey=2)
    plt.legend(loc=0, title="Error probability")
    plt.tight_layout()
    plt.savefig('fig-lim-T-by-r_n512_s100.pdf', dpi=200)

#########################

def plot_time(p, r, n, steps, samples, marker='.', color='blue', label=None, value='I'):
    dfs = []
    for i in range(samples):
        rs = csort.RandomSort(p, r, n, max(steps // 30, 1), hash((p, r, n, i)) % 2**30)
        rs.steps(steps)
        dfs.append(pandas.DataFrame({
            'p': p, 'r': r, 'n': n, 'sample': i,
            'label': label or ("p = %g r = %d n = %d" % (p,r,n)),
            'T': list(rs.Ts), 'I': list(rs.Is), 'W': list(rs.Ws)
            }))
    df = pandas.concat(dfs)
    sns.tsplot(df, time='T', value=value, ci=[99], unit='sample',
            condition='label', err_style=None, estimator=np.mean, marker=marker, color=color)
    return df

def fig_process_by_r(v='I'):
    sns.set_context('paper', rc={"lines.linewidth": 1.1, 'lines.markeredgewidth': 0.1}, font_scale=1.3)
    sns.set_style("ticks")
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
    sns.set_context('paper', rc={"lines.linewidth": 1.1}, font_scale=1.3)
    sns.set_style("ticks")
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

#        jobs.append(joblib.delayed(sorting_process_sample)(process, sample, steps, name))
#
#    log.info("Submitting %d jobs ...", len(jobs))
#    parallel = joblib.Parallel(n_jobs = 3, verbose = 5)
#    res = parallel(jobs)
#
#    return pandas.concat(res, ignore_index=True, copy=False)


def plot_sorting_process(df, value='log_N(E)', title=None, ymax=None, ax=None):
    if isinstance(df, str):
        df = pandas.read_pickle(df)
    ax = sns.tsplot(df, time='step', value=value, ci=[68, 95], unit='sample',
                    condition='label', err_style='ci_band', estimator=np.mean, ax=ax)
    ax.set_title(title or 'Energy by the number of steps, %d samples, percentiles: 68, 95' % len(df['sample'].unique()))
    ax.set_ylim([0, ymax or int(math.ceil(max(1, df[value].max())))])
    return ax


    plt.figure(figsize=(20, 20 * (9.0 / 16)))
    ax = plot_sorting_process(df, value='E')
    plt.savefig(fname + '.png', bbox_inches='tight', dpi=200)
    
    return df


if __name__ == '__main__':
    main()
