import random

class SortingInstance:

    def __init__(self, values):
        self.values = values[:] # make a copy
        self.values_index = { v: i for i, v in enumerate(self.values) }
        self.values_sorted = sorted(values)
        self.values_rank = { v: i for i, v in enumerate(self.values_sorted) }
        self.n = len(self.values)
        self.cmps = 0
        self.ops = 0
        self.result = None

    def cmp(self, a, b):
        """
        Compare element values: v[a] vs v[b].
        Return -1, 0 or +1 when va<vb, va=vb or va>vb.
        """
        self.cmps += 1
        self.op()
        va = self.values[a]
        vb = self.values[b]
        if (va < vb):
            return -1
        if (va > vb):
            return 1
        return 0

    def op(self, w=1):
        "Count a single performed operation. Calling this is up to the algorithm."
        self.ops += w

    def submit_result(self, array):
        """
        Submit the assumed positions of the elements.
        E.g. array[2] = 0 indicates that v[0] is the smallest element.
        """
        assert(len(array) == self.n)
        self.result = array

    def dislocation(self, value, index):
        return abs(self.values_rank[value] - index)

    def total_dislocation(self, result=None):
        if not result:
            result = self.result
        assert result
        return sum(self.dislocation(v, i) for i, v in enumerate(result))

    def maximum_dislocation(self, result=None):
        if not result:
            result = self.result
        assert result
        return max(self.dislocation(v, i) for i, v in enumerate(result))

    def statstr(self):
        resstr = ", no res"
        if self.result:
            resstr = ", res-avg %.2f, res-max %d" % (self.total_dislocation() / self.n, self.maximum_dislocation())
        return "%d ops, %d cmps, disloc: src-avg %.2f, src-max %d%s" % (
                self.ops, self.cmps, self.total_dislocation(self.values) / self.n,
                self.maximum_dislocation(self.values), resstr)

    def __repr__(self):
        return "<%s [%d] %s>" % (self.__class__.__name__, self.n, self.statstr())

class IndependentErrorInstance(SortingInstance):

    def __init__(self, values, p_err):
        super().__init__(values)
        self.p_err = p_err

    def cmp(self, a, b):
        self.cmps += 1
        self.op()
        va = self.values[a]
        vb = self.values[b]
        if va == vb: return 0
        if (va < vb) == (random.random() >= self.p_err):
            return -1
        return 1

    def __repr__(self):
        return "<%s [%d, p_err=%.3g] %s>" % (self.__class__.__name__, self.n, self.p_err, self.statstr())

class RepeatedErrorInstance(IndependentErrorInstance):

    def __init__(self, values, p_err):
        super().__init__(values, p_err)
        self.comparison_ok = {}

    def cmp(self, a, b):
        self.cmps += 1
        self.op()
        va = self.values[a]
        vb = self.values[b]
        if va == vb: return 0
        l = min(a,b)
        u = max(a,b)
        if (l,u) not in self.comparison_ok:
            self.comparison_ok[(l,u)] = (random.random() >= self.p_err)
        if (va < vb) == self.comparison_ok[(l,u)]:
            return -1
        return 1


