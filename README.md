# Swap-sorting experiments

Experiments for article "Sorting by Swaps with Noisy Comparisons" by Tomáš Gavenčiak, Barbara Geissmann, Johannes Lengler. GECCO 2017.

## Requirements

Python 3.5+, python packages `joblib matplotlib numpy pandas seaborn`, C++11 compiler (preferably GCC or Clang), Boost-Python library. 

## Running

```
git clone https://github.com/gavento/swap-sorting-experiments
cd swap-sorting-experiments/csort
make
```

You can configure the number of sorting process samples with `SMP=...` in `plotting.py`.
