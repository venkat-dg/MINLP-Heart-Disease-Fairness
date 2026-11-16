"""EASY problem: L2-regularised logistic regression on all tests (convex), for both datasets.

  python easy.py

Solves the easy problem with the shared solver in core.py and verifies the optimum three ways:
  1. gradient ~ 0                      (stationary point)
  2. smallest Hessian eigenvalue > 0   (strictly convex -> unique global minimum)
  3. agreement with scikit-learn       (independent solver)
Writes results/easy_verification.json, results/easy_baseline.json, results/easy_theta_<name>.npy.
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from data_prep import build, ROOT
from core import LAM, sigmoid, loss_grad, fit, evaluate

OUT = os.path.join(ROOT, "results")

def verify(d, lam=LAM):
    X, y = d["Xtr"], d["ytr"]
    J, theta = fit(X, y, lam=lam)
    _, g = loss_grad(theta, X, y, lam)
    # Hessian of J at theta: X_a^T diag(p(1-p)) X_a / n + lam * I (bias not penalised)
    p = sigmoid(X @ theta[:-1] + theta[-1]); Xa = np.c_[X, np.ones(len(y))]
    H = Xa.T @ (Xa * (p * (1 - p))[:, None]) / len(y) + lam * np.diag(np.r_[np.ones(X.shape[1]), 0])
    # sklearn minimises C*sum(loss) + ||w||^2/2, i.e. our J scaled, when C = 1/(lam*n)
    sk = LogisticRegression(C=1 / (lam * len(y)), tol=1e-12, max_iter=10000).fit(X, y)
    sk_theta = np.append(sk.coef_[0], sk.intercept_[0])
    ver = dict(J=J, grad=float(np.abs(g).max()), mineig=float(np.linalg.eigvalsh(H).min()),
               sk_diff=float(np.abs(theta - sk_theta).max()))
    base = dict(lam=lam, J_train=J,
                train=evaluate(theta, X, y, d["sex_tr"]),
                test=evaluate(theta, d["Xte"], d["yte"], d["sex_te"]))
    return ver, base, theta

if __name__ == "__main__":
    verification, baseline = {}, {}
    for name in ["uci", "za"]:
        d = build(name)
        ver, base, theta = verify(d)
        verification[name], baseline[name] = ver, base
        np.save(os.path.join(OUT, f"easy_theta_{name}.npy"), theta)
        print(f"{name}: J = {ver['J']:.6f}  ||grad||_inf = {ver['grad']:.2e}  "
              f"min Hessian eig = {ver['mineig']:.2e} (>0 => strictly convex)  "
              f"max |theta - sklearn| = {ver['sk_diff']:.2e}")
        print(f"  test acc {base['test']['acc']:.3f}  test AUC {base['test']['auc']:.3f}")
        print("  largest |weights|:")
        for i in np.argsort(-np.abs(theta[:-1]))[:8]:
            print(f"    {d['names'][i]:22s} {theta[i]:+.3f}")
    json.dump(verification, open(os.path.join(OUT, "easy_verification.json"), "w"), indent=1)
    json.dump(baseline, open(os.path.join(OUT, "easy_baseline.json"), "w"), indent=1)
