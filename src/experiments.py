"""Run all experiments for one dataset:  python experiments.py uci|za
Writes results/<name>_results.json."""
import sys, os, json, time
import numpy as np
from data_prep import build, ROOT
from core import evaluate, fairness_vec
from hard import Problem, unconstrained_table, solve_exact, solve_heuristic

name = sys.argv[1]
OUT = os.path.join(ROOT, "results")
EXACT = name == "uci"
t0 = time.time()
def log(*a): print(f"[{name} {time.time()-t0:6.0f}s]", *a, flush=True)

def run(P, table, **kw):
    r = solve_exact(P, table, **kw) if table is not None else solve_heuristic(P, **kw)
    d = P.d
    ev = evaluate(r["theta"], d["Xte"], d["yte"], d["sex_te"])
    S = list(r["S"])
    return dict(J=r["J"], tests=[d["tests"][i] for i in S], k=len(S), cost=float(P.cost[S].sum()),
                surrogate_gap=float(abs(P.a @ r["theta"][:-1])), **{"test_" + k: v for k, v in ev.items()})

def elbow(x, y):
    x, y = np.array(x, float), np.array(y, float)
    xn = (x - x.min()) / (np.ptp(x) or 1); yn = (y - y.min()) / (np.ptp(y) or 1)
    return int(np.argmax((1 - xn) - yn))

def setup(seed, fair, cost=None, **kw):
    d = build(name, seed=seed, **({"cost": cost} if cost is not None else {}), **kw)
    P = Problem(d, fair)
    tab = None
    if EXACT:
        tab = unconstrained_table(P, os.path.join(OUT, f"uci_table_seed{seed}{'_'+kw['sites'][0] if 'sites' in kw else ''}.npz"))
    return d, P, tab

res = dict(dataset=name)
d, P, tab = setup(0, "dp")
nt, full = P.nt, tuple(range(P.nt))
J_easy = P.fit(full)[0]
easy = run(P, tab)                          # K=B=inf, eps=inf  -> all useful tests = easy problem
res["easy"] = easy; log("easy J", round(J_easy, 4), "test acc", round(easy["test_acc"], 3), "auc", round(easy["test_auc"], 3))
theta_easy = P.fit(full)[1]
gap0 = {f: abs(fairness_vec(P.X, P.y, d["sex_tr"], f) @ theta_easy[:-1]) for f in ["dp", "eo"]}
res["gap0_surrogate"] = gap0

# ---- E1: budget sweep (Pareto) ------------------------------------------------
Bgrid = sorted(set([0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600, 1000]) if name == "uci"
               else set([0, 5, 10, 20, 30, 40, 60, 80, 100, 150, 200, 300, 450, 700]))
res["budget_sweep"] = []
for B in Bgrid:
    r = run(P, tab, B=B); r["B"] = B; res["budget_sweep"].append(r)
    log("B", B, "J", round(r["J"], 4), r["k"], "tests", "acc", round(r["test_acc"], 3))
ib = elbow([r["cost"] for r in res["budget_sweep"]], [r["J"] for r in res["budget_sweep"]])
B_star = res["budget_sweep"][ib]["B"]; res["B_star"] = B_star; log("B* (elbow)", B_star)

# ---- E2: sparsity sweep -------------------------------------------------------
Kgrid = list(range(1, nt + 1)) if name == "uci" else [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
res["k_sweep"] = []
for K in Kgrid:
    r = run(P, tab, K=K); r["K"] = K; res["k_sweep"].append(r)
    log("K", K, "J", round(r["J"], 4), "cost", r["cost"])
ik = elbow([r["K"] for r in res["k_sweep"]], [r["J"] for r in res["k_sweep"]])
K_star = res["k_sweep"][ik]["K"]; res["K_star"] = K_star; log("K* (elbow)", K_star)

# ---- E3: fairness sweep, both variants, at K*, B* ----------------------------
fracs = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
res["fair_sweep"] = {}
for fair in ["dp", "eo"]:
    Pf = Problem(d, fair); rows = []
    for fr in fracs:
        eps = fr * gap0[fair] if fr < 1 else np.inf
        r = run(Pf, tab, K=K_star, B=B_star, eps=eps); r.update(frac=fr, eps=None if fr >= 1 else eps)
        rows.append(r); log(fair, "frac", fr, "J", round(r["J"], 4), "dp_gap_prob", round(r["test_dp_gap_prob"], 3), "acc", round(r["test_acc"], 3))
    res["fair_sweep"][fair] = rows

# ---- E4: price of realism decomposition --------------------------------------
eps_h = 0.5 * gap0["dp"]
steps = {"easy": run(P, tab),
         "+sparsity K*": run(P, tab, K=K_star),
         "+budget B*": run(P, tab, K=K_star, B=B_star),
         "+fairness (eps=0.5*gap0)": run(P, tab, K=K_star, B=B_star, eps=eps_h)}
res["por"] = dict(steps=steps, K=K_star, B=B_star, eps=eps_h,
                  PoR_train_loss=steps["+fairness (eps=0.5*gap0)"]["J"] - J_easy)
log("PoR (train loss)", round(res["por"]["PoR_train_loss"], 4))

# ---- E5: heuristic vs exact (UCI only) ---------------------------------------
if EXACT:
    match = tot = 0
    for K in [2, 4, 6, 8]:
        for B in [50, 150, 400, np.inf]:
            for eps in [np.inf, 0.5 * gap0["dp"]]:
                a_ = solve_exact(P, tab, K=K, B=B, eps=eps)["J"]; h_ = solve_heuristic(P, K=K, B=B, eps=eps)["J"]
                tot += 1; match += abs(a_ - h_) < 1e-6
    res["heuristic_vs_exact"] = dict(matches=int(match), total=tot); log("heuristic matches exact:", match, "/", tot)

# ---- E6: stability over random splits (heuristic) ----------------------------
seeds = range(1, 11 if EXACT else 21)
stab = []
for s in seeds:
    ds = build(name, seed=s); Ps = Problem(ds, "dp")
    th_e = Ps.fit(tuple(range(Ps.nt)))[1]
    g0 = abs(Ps.a @ th_e[:-1]); Je = Ps.fit(tuple(range(Ps.nt)))[0]
    r = run(Ps, None, K=K_star, B=B_star, eps=0.5 * g0); r["seed"] = s
    r["J_easy"] = Je; r["PoR"] = r["J"] - Je
    ee = evaluate(th_e, ds["Xte"], ds["yte"], ds["sex_te"]); r["easy_test_acc"] = ee["acc"]; r["easy_test_auc"] = ee["auc"]
    r["easy_test_dp_gap_prob"] = ee["dp_gap_prob"]
    stab.append(r); log("seed", s, r["tests"], "acc", round(r["test_acc"], 3))
freq = {t: float(np.mean([t in r["tests"] for r in stab])) for t in d["tests"]}
res["stability"] = dict(runs=stab, freq=freq)

# ---- E7: cost sensitivity ----------------------------------------------------
rng = np.random.default_rng(0); sens = []
base_cost = dict(zip(d["tests"], d["cost"]))
for i in range(20):
    c2 = {t: (c * float(rng.uniform(0.5, 2.0)) if c > 0 else 0) for t, c in base_cost.items()}
    ds = build(name, seed=0, cost=c2); Ps = Problem(ds, "dp")
    r = run(Ps, None, K=K_star, B=B_star, eps=eps_h); sens.append(r["tests"])
res["cost_sensitivity"] = dict(panels=sens, freq={t: float(np.mean([t in p for p in sens])) for t in d["tests"]})
log("cost-sensitivity done")

# ---- E8: site robustness (UCI only) ------------------------------------------
if EXACT:
    ds, Ps, tabs = setup(0, "dp", sites=["cleveland", "hungarian"])
    full_ = tuple(range(Ps.nt)); Je = Ps.fit(full_)[0]
    th_e = Ps.fit(full_)[1]; g0 = abs(Ps.a @ th_e[:-1])
    r = run(Ps, tabs, K=K_star, B=B_star, eps=0.5 * g0)
    res["site_robustness"] = dict(sites=["cleveland", "hungarian"], J_easy=Je, hard=r, PoR=r["J"] - Je)
    log("site-robustness done", r["tests"])

json.dump(res, open(os.path.join(OUT, f"{name}_results.json"), "w"), indent=1, default=float)
log("DONE")
