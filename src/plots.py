"""Figures from results/*_results.json  ->  results/figures/*.png"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_prep import ROOT

R = os.path.join(ROOT, "results"); F = os.path.join(R, "figures"); os.makedirs(F, exist_ok=True)
TITLE = {"uci": "UCI Heart Disease (n=920)", "za": "Z-Alizadeh Sani (n=303)"}
res = {n: json.load(open(os.path.join(R, f"{n}_results.json"))) for n in ["uci", "za"]
       if os.path.exists(os.path.join(R, f"{n}_results.json"))}
C = dict(hard="#1f77b4", easy="#d62728", dp="#2ca02c", eo="#ff7f0e")

for n, r in res.items():
    Je = r["easy"]["J"]
    # 1. Pareto: cost vs loss (and test accuracy)
    b = r["budget_sweep"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot([x["cost"] for x in b], [x["J"] for x in b], "o-", color=C["hard"])
    ax[0].axhline(Je, ls="--", color=C["easy"], label="easy (all tests)")
    ib = [x["B"] for x in b].index(r["B_star"]); ax[0].plot(b[ib]["cost"], b[ib]["J"], "k*", ms=14, label=f"elbow B*={r['B_star']}")
    ax[0].set(xlabel="diagnostic cost of chosen panel", ylabel="train loss J", title="Loss vs cost (Pareto frontier)"); ax[0].legend()
    ax[1].plot([x["cost"] for x in b], [x["test_acc"] for x in b], "o-", color=C["hard"])
    ax[1].axhline(r["easy"]["test_acc"], ls="--", color=C["easy"])
    ax[1].set(xlabel="diagnostic cost of chosen panel", ylabel="test accuracy", title="Accuracy vs cost")
    fig.suptitle(TITLE[n]); fig.tight_layout(); fig.savefig(os.path.join(F, f"{n}_pareto.png"), dpi=150); plt.close(fig)

    # 2. Sparsity sweep
    k = r["k_sweep"]; fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.plot([x["K"] for x in k], [x["J"] for x in k], "o-", color=C["hard"]); ax.axhline(Je, ls="--", color=C["easy"], label="easy")
    ax.axvline(r["K_star"], color="k", ls=":", label=f"elbow K*={r['K_star']}")
    ax.set(xlabel="max number of tests K", ylabel="train loss J", title=f"{TITLE[n]}: sparsity"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(F, f"{n}_sparsity.png"), dpi=150); plt.close(fig)

    # 3. Fairness trade-off: loss and true probability gap vs eps fraction
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for fair, lab in [("dp", "demographic parity"), ("eo", "equal opportunity")]:
        rows = r["fair_sweep"][fair]; fr = [x["frac"] for x in rows]
        ax[0].plot(fr, [x["J"] for x in rows], "o-", color=C[fair], label=lab)
        ax[1].plot(fr, [x["test_acc"] for x in rows], "o-", color=C[fair], label=lab)
    for a_, yl in zip(ax, ["train loss J", "test accuracy"]):
        a_.set(xlabel="fairness bound eps (fraction of unconstrained score gap)", ylabel=yl); a_.invert_xaxis(); a_.legend()
    fig.suptitle(f"{TITLE[n]}: price of fairness (K*, B* fixed)"); fig.tight_layout()
    fig.savefig(os.path.join(F, f"{n}_fairness.png"), dpi=150); plt.close(fig)

    # 4. PoR decomposition
    s = r["por"]["steps"]; labs = list(s); vals = [s[l]["J"] for l in labs]
    fig, ax = plt.subplots(figsize=(6.5, 3.8)); ax.bar(range(len(vals)), vals, color=[C["easy"]] + [C["hard"]] * 3)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels([l.replace(" (", "\n(") for l in labs], fontsize=8)
    ax.set_ylim(min(vals) * 0.9, max(vals) * 1.03)
    for i, v in enumerate(vals): ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set(ylabel="train loss J", title=f"{TITLE[n]}: price of realism = {r['por']['PoR_train_loss']:.3f}")
    fig.tight_layout(); fig.savefig(os.path.join(F, f"{n}_por.png"), dpi=150); plt.close(fig)

    # 5. Selection frequency across random splits
    fq = r["stability"]["freq"]; items = sorted(fq.items(), key=lambda kv: -kv[1]); items = [i for i in items if i[1] > 0][:20]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(items) + 1.2))
    ax.barh([i[0] for i in items][::-1], [i[1] for i in items][::-1], color=C["hard"]); ax.set(xlim=(0, 1), xlabel="selection frequency over random splits", title=f"{TITLE[n]}: stability")
    fig.tight_layout(); fig.savefig(os.path.join(F, f"{n}_stability.png"), dpi=150); plt.close(fig)
print("figures ->", F)
