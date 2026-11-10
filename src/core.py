"""Core numerics shared by the easy and hard problems.

J(w,b) = mean cross-entropy + (lam/2)||w||^2          (bias unpenalised)

Fairness (convex surrogate, linear in w):
  demographic parity : |a^T w| <= eps,  a = mean_x(male) - mean_x(female)        (all patients)
  equal opportunity  : |a^T w| <= eps,  a = same difference restricted to y = 1  (diseased patients)
The bias b cancels in the difference, so the constraint involves w only.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

LAM = 1e-2

def sigmoid(z): return 0.5 * (1 + np.tanh(z / 2))

def loss_grad(theta, X, y, lam=LAM):
    w, b = theta[:-1], theta[-1]
    z = X @ w + b
    p = sigmoid(z)
    J = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * w @ w
    g = np.append(X.T @ (p - y) / len(y) + lam * w, np.mean(p - y))
    return J, g

def fairness_vec(X, y, sex, kind="dp"):
    m = (y == 1) if kind == "eo" else np.ones(len(y), bool)
    return X[m & (sex == 1)].mean(0) - X[m & (sex == 0)].mean(0)

def fit(X, y, cols=None, lam=LAM, a=None, eps=np.inf):
    """Minimise J over the encoded columns `cols` (others fixed at 0), optionally s.t. |a^T w| <= eps.
    Returns (J, theta_full)."""
    cols = np.arange(X.shape[1]) if cols is None else np.asarray(cols, int)
    Xs, n = X[:, cols], len(cols)
    x0 = np.zeros(n + 1)
    if n == 0:
        b = np.log(y.mean() / (1 - y.mean()))
        th = np.zeros(X.shape[1] + 1); th[-1] = b
        return loss_grad(np.array([b]), Xs, y, lam)[0], th
    r = minimize(loss_grad, x0, args=(Xs, y, lam), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=1000, gtol=1e-9, ftol=1e-13))
    x, J = r.x, r.fun
    if a is not None and np.isfinite(eps):
        asub = a[cols]
        if abs(asub @ x[:-1]) > eps + 1e-9:                # fairness binding -> constrained solve
            cons = [{"type": "ineq", "fun": lambda t: eps - asub @ t[:-1], "jac": lambda t: np.append(-asub, 0)},
                    {"type": "ineq", "fun": lambda t: eps + asub @ t[:-1], "jac": lambda t: np.append(asub, 0)}]
            r2 = minimize(loss_grad, x, args=(Xs, y, lam), jac=True, method="SLSQP",
                          constraints=cons, options=dict(maxiter=300, ftol=1e-12))
            x, J = r2.x, r2.fun
    th = np.zeros(X.shape[1] + 1); th[cols] = x[:-1]; th[-1] = x[-1]
    return float(J), th

def evaluate(theta, X, y, sex):
    """Test/train metrics. Reports TRUE (sigmoid) fairness gaps, not just the surrogate."""
    s = X @ theta[:-1] + theta[-1]; p = sigmoid(s); pred = (p >= 0.5).astype(int)
    pos = y == 1
    fnr = lambda g: (pred[g & pos] == 0).mean() if (g & pos).any() else np.nan
    return dict(
        ce=float(np.mean(np.logaddexp(0, s) - y * s)), acc=float((pred == y).mean()),
        auc=float(roc_auc_score(y, p)),
        dp_gap_prob=float(abs(p[sex == 1].mean() - p[sex == 0].mean())),
        eo_gap_prob=float(abs(p[(sex == 1) & pos].mean() - p[(sex == 0) & pos].mean())),
        fnr_gap=float(abs(fnr(sex == 1) - fnr(sex == 0))))

def cols_of(group, tests_idx):
    return np.flatnonzero(np.isin(group, list(tests_idx)))
