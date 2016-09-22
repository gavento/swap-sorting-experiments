import random
import numpy as np

from . import log, ErrorCostType, DisorderMeasure
from . import utils


class SortingInstance:
    """A sorting problem instance to be a black box for any sorting alg.
    """

    def __init__(self, values):
        self.values = np.array(values) # make a copy
        self.N = len(self.values)
        self.values_index = { v: i for i, v in enumerate(self.values) }
        self.values_sorted = np.array(sorted(values))
        self.values_rank = { v: i for i, v in enumerate(self.values_sorted) }
        self.indexes = np.arange(self.N)
        self.rev_indexes = np.arange(self.N - 1, -1, -1)
        self.cmps = 0
        self.ops = 0
        self.result = None
        self.base_twd = (self.rev_indexes * self.values_sorted).sum()

    def get_sequence(self):
        return self.values[:]

    def cmp(self, a, b):
        """
        Compare a vs b.
        Return -1, 0 or +1 when a<b, a=b or a>b.
        Optionally takes distance in the permutation into account.
        """
        self.cmps += 1
        self.op()
        if (a < b):
            return -1
        if (a > b):
            return 1
        return 0

    def op(self, w=1):
        """
        Count a single performed operation.
        Calling this regularly is up to the algorithm.
        """
        self.ops += w

    def submit_result(self, sequence):
        """
        Submit the assumed positions of the elements.
        E.g. array[2] = 0 indicates that v[0] is the smallest element.
        """
        assert(len(sequence) == self.N)
        self.result = sequence

    def dislocation(self, a, index):
        "Return the dislocation of element v at pos. index. O(1) time."
        return abs(self.values_rank[a] - index)

    def total_dislocation(self, sequence):
        "Return the total (sum) dislocation of given sequence. O(n) time."
        return sum(self.dislocation(v, i) for i, v in enumerate(sequence))

    def maximum_dislocation(self, result=None):
        "Return the maximum dislocation of the elements of the given sequence. O(n) time."
        return max(self.dislocation(v, i) for i, v in enumerate(result))

    def weighted_dislocation(self, a, index):
        "Return the weighted dislocation of the given element at pos. index. O(1) time."
        return a * (self.values_rank[a] - index)

    def _total_weighted_dislocation(self, sequence):
        "Older and slower version of total_weighted_dislocation(). O(n) time."
        return sum(self.weighted_dislocation(v, i) for i, v in enumerate(sequence))

    def total_weighted_dislocation(self, sequence):
        "Return the total (sum) weighted dislocation of the given sequence. O(n) time."
        twd = (self.rev_indexes * sequence).sum() - self.base_twd
        #assert twd == self.total_weighted_dislocation_(sequence)
        return twd

    def _total_inversions_merge(self, subseq):
        "Internal for total_inversions(). Returns (sorted_seq, inversion_count)."
        if len(subseq) <= 1:
            return (subseq, 0)
        mid = len(subseq) // 2
        s1, invs1 = self._total_inversions_merge(subseq[:mid])
        s2, invs2 = self._total_inversions_merge(subseq[mid:])
        i1 = i2 = invs = 0
        s = []
        while i1 < len(s1) and i2 < len(s2):
            if s1[i1] <= s2[i2]:
                s.append(s1[i1])
                i1 += 1
            else:
                s.append(s2[i2])
                i2 += 1
                invs += len(s1) - i1
        s.extend(s1[i1:])
        s.extend(s2[i2:])
        return (s, invs + invs1 + invs2)

    def total_inversions(self, sequence):
        "Return the number of inversions in sequence. O(n log(n)) time."
        return self._total_inversions_merge(sequence)[1]

    def seqstats(self, sequence):
        "Return a stats string for the given sequence."
        wdis = self.total_weigted_dislocation(sequence)
        dis = self.total_dislocation(sequence)
        invs = self.total_inversions(sequence)
        return  "N=%d, Inv=%g (log_N=%g), Dis=%g (log_N=%g), WDis=%g (log_N=%g)" % (
                self.N, invs, math.log(invs, self.N),
                dis, math.log(dis, self.N),
                wdis, math.log(wdis, self.N),
                )

    def __repr__(self):
        return "<%s N=%d, %d ops, %d cmps>" % (self.__class__.__name__, self.N,
                self.ops, self.cmps)


class IndependentErrorInstance(SortingInstance):
    """A sorting problem instance with errors.
    The comparison error probability is given by the energy difference of the considered
    two states (swapped ot not). For some swap operations and energy costs, this is
    consistent with a statistical physics model of the permutation space.
    The energy cost is given by ErrorCostType.
    For VALUEDIST, the distance of the compared elements matters as well.
    """ 

    def __init__(self, values, rnd=None, p_err=None, T=None, error_cost=ErrorCostType.UNIT):
        assert (p_err is None) != (T is None)
        super().__init__(values)
        self.p_err = p_err
        self.T = T
        W = 1.0 # basic op weight
        if self.T is None:
            self.T = utils.T_for_p_err(self.p_err)
        if self.p_err is None:
            self.p_err = utils.p_err_for_T(self.T)
        self.rnd = rnd or random.Random()
        self.error_cost = error_cost

    def swap_E_diff(self, a, b, dist=1):
        "Return the energy increase after swapping a and b (in distance dist)."
        assert dist >= 0
        if self.error_cost == ErrorCostType.UNIT:
            return 1.0 if a < b else -1.0
        elif self.error_cost == ErrorCostType.VALUE:
            return b - a
        elif self.error_cost == ErrorCostType.DIST:
            return dist if a < b else -dist
        elif self.error_cost == ErrorCostType.VALUEDIST:
            return dist * (b - a)
        else:
            raise ValueError("Invalid energy_cost")

    def cmp_err(self, a, b, dist=1):
        """
        Return the probability of error while comparing values a and b (in distance dist).
        Does not take the current direction into account (is just the probability of error in comparison).
        """
        Ediff = abs(self.swap_E_diff(a, b, dist=dist))
        return utils.p_err_for_T(self.T, Ediff)

    def cmp(self, a, b, dist=1):
        """
        Compare the values a and b in distance dist, simulating independent errors, return -1, 0, 1.
        Never makes errors on equality (neither same values nor positions).
        """
        self.cmps += 1
        self.op()
        if a == b or dist == 0:
            return 0
        if (a < b) == (self.rnd.random() >= self.cmp_err(a, b, dist=dist)):
            return -1
        return 1

    def __repr__(self):
        return "<%s type %s, N=%d, p_err_1=%.3g T=%g, %d ops, %d cmps>" % (self.__class__.__name__,
                self.error_cost.__name__, self.n, self.p_err, self.T, self.ops, self.cmps)


class RepeatedErrorInstance(IndependentErrorInstance):
    """A sorting problem instance with errors.
    The error probability is consistent with a statistical physics model, the energy
    potential is given by ErrorCostType.""" 

    def __init__(self, values, rnd=None, p_err=None, T=None, error_cost=ErrorCostType.UNIT):
        super().__init__(values, rnd=rnd, p_err=p_err, T=T, error_cost=error_cost)
        self.comparison_result = {}

    def cmp(self, a, b, dist=1):
        if self.error_cost in (ErrorCostType.VALUEDIST, ErrorCostType.DIST) and dist != 1:
            raise ValueError("Can not combine RepeatedErrorInstance, VALUEDIST (or DIST) error type and cmp(dist >= 2)")
        if (a, b) not in self.comparison_result:
            res = super().cmp(a, b, dist=dist)
            self.comparison_result[(a, b)] = res
            self.comparison_result[(b, a)] = -res
        return self.comparison_result[(a, b)]

