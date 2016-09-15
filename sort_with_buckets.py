import random
import instance

def log2(n):
    r = 0
    while n > 1:
        r += 1
        n = n >> 1
    return r

def rank_in(inst, elem, group):
    return sum((inst.cmp(i, elem) < 0) for i in group)

def buckets_range(bkts, center, radius):
    r = []
    for bi in range(max(center - radius, 0), min(center + radius + 1, len(bkts))):
        r.extend(bkts[bi])
    return r

def sort_with_buckets(inst, radius=2):
    assert isinstance(inst, instance.SortingInstance)
    n = inst.n
    logn = log2(n)
    assert n == 1 << logn
    bkts = [inst.values]

    for step in range(0, log2(n - logn)):
        print(step, bkts)
        bkts_next = [ [] for i in range(1 << (step + 1)) ]
        presum = 0 # Sum of bkts before bi-radius
        for bi, b in enumerate(bkts):
            group = buckets_range(bkts, bi, radius)
            for el in b:
                idx = rank_in(inst, el, group) + presum
#                print(" %d has rank %d + %d" % (el, presum, rank_in(inst, el, group)))
                bkts_next[idx // (n // (1 << (step + 1)))].append(el)
            if bi >= radius:
                presum += len(bkts[bi - radius])
        bkts = bkts_next

    r = []
    for b in bkts:
        r.extend(b)
    inst.submit_result(r)
    return bkts

def test(n, p_err):
    vals = list(range(n))
    random.shuffle(vals)
    inst = instance.RepeatedErrorInstance(vals, p_err)
    bkts = sort_with_buckets(inst)
    print(list(map(len, bkts)))
    return inst


