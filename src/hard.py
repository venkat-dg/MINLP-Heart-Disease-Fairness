"""HARD problem: sparse, budgeted, fair logistic regression (MINLP).

  min_{w,b,z}  J(w,b)
  s.t.  |w_i| <= M z_{g(i)}         (column i belongs to test g(i); z binary -> test switched on/off)
        sum_j z_j <= K              (sparsity)
        sum_j c_j z_j <= B          (budget)
        |a^T w| <= eps              (fairness, convex surrogate)

For a FIXED z the problem is convex, so the MINLP is solved as: search over z, convex solve inside.
  * exact     : enumerate all 2^d subsets (d = 13 on UCI), with exact pruning.
  * heuristic : greedy forward selection + swap/drop local search (d = 54 on Z-Alizadeh).
"""
import itertools, os, time
import numpy as np
from core import fit, fairness_vec, cols_of, LAM
from data_prep import ROOT

class Problem:
    def __init__(self, d, fair="dp", lam=LAM):
        self.d, self.lam, self.fair = d, lam, fair
        self.X, self.y, self.g = d["Xtr"], d["ytr"], d["group"]
        self.nt, self.cost = len(d["tests"]), d["cost"]
        self.a = fairness_vec(self.X, self.y, d["sex_tr"], fair)
        self._cache = {}
    def cols(self, S): return cols_of(self.g, S)
    def fit(self, S, eps=np.inf):
        key = (frozenset(S), eps)
        if key not in self._cache:
            self._cache[key] = fit(self.X, self.y, self.cols(S), self.lam, self.a, eps)
        return self._cache[key]
    def feasible(self, S, K, B):
        return len(S) <= K and self.cost[list(S)].sum() <= B + 1e-9

# ------------------------------------------------------------------ exact (small d)
def all_subsets(nt):
    for r in range(nt + 1):
        yield from itertools.combinations(range(nt), r)

def unconstrained_table(P, path=None):
    """J(S) with no fairness for every subset S. Lower bound for the constrained value."""
    if path and os.path.exists(path):
        z = np.load(path, allow_pickle=True); return list(map(tuple, z["S"])), z["J"]
    subs, Js = [], []
    for S in all_subsets(P.nt):
        subs.append(S); Js.append(P.fit(S)[0])
    if path:
        S_arr = np.empty(len(subs), object); S_arr[:] = subs
        np.savez(path, S=S_arr, J=np.array(Js))
    return subs, np.array(Js)

def solve_exact(P, table, K=np.inf, B=np.inf, eps=np.inf):
    """Global optimum over subsets. Exact pruning: constrained J >= unconstrained J, so once the
    next candidate's unconstrained J exceeds the incumbent we can stop."""
    subs, Js = table
    order = np.argsort(Js)
    best = (np.inf, None, None); n_solved = 0
    for i in order:
        S = subs[i]
        if Js[i] >= best[0]: break
        if not P.feasible(S, K, B): continue
        J, th = P.fit(S, eps); n_solved += 1
        if J < best[0]: best = (J, S, th)
    return dict(J=best[0], S=best[1], theta=best[2], n_solved=n_solved)

# ------------------------------------------------------------------ heuristic (large d)
def solve_heuristic(P, K=np.inf, B=np.inf, eps=np.inf, max_pass=6):
    """Greedy forward selection (best loss decrease per step), then swap/drop local search.
    Always returns a feasible subset (the empty set is feasible)."""
    S = ()
    cur = P.fit(S, eps)[0]
    # forward
    while len(S) < K:
        cand = []
        for j in range(P.nt):
            if j in S: continue
            T = tuple(sorted(S + (j,)))
            if P.feasible(T, K, B): cand.append((P.fit(T, eps)[0], T))
        if not cand: break
        J, T = min(cand)
        if J < cur - 1e-9: cur, S = J, T
        else: break
    # local search: drop one / swap one
    for _ in range(max_pass):
        improved = False
        moves = [tuple(x for x in S if x != i) for i in S]
        moves += [tuple(sorted([x for x in S if x != i] + [j]))
                  for i in S for j in range(P.nt) if j not in S]
        moves += [tuple(sorted(S + (j,))) for j in range(P.nt) if j not in S]
        for T in moves:
            if P.feasible(T, K, B):
                J = P.fit(T, eps)[0]
                if J < cur - 1e-9: cur, S, improved = J, T, True
        if not improved: break
    J, th = P.fit(S, eps)
    return dict(J=J, S=S, theta=th)

def solve(P, table=None, **kw):
    return solve_exact(P, table, **kw) if table is not None else solve_heuristic(P, **kw)
