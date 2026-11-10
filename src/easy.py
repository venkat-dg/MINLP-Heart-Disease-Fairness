"""EASY problem: L2-regularised logistic regression on all features (convex)."""
import os, numpy as np, json
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from data_prep import build, ROOT

def sigmoid(z): return 0.5*(1+np.tanh(z/2))

def loss_grad(theta, X, y, lam):
    """J = mean cross-entropy + (lam/2)||w||^2 ; theta=[w,b]; bias unpenalised."""
    w, b = theta[:-1], theta[-1]
    z = X@w + b
    ce = np.mean(np.logaddexp(0, z) - y*z)
    p = sigmoid(z)
    g = np.append(X.T@(p-y)/len(y) + lam*w, np.mean(p-y))
    return ce + 0.5*lam*w@w, g

def fit(X, y, lam=1e-2, mask=None):
    """Minimise J over columns in mask (others fixed to 0). Returns theta (full length)."""
    idx = np.arange(X.shape[1]) if mask is None else np.flatnonzero(mask)
    Xs = X[:, idx]
    r = minimize(loss_grad, np.zeros(len(idx)+1), args=(Xs, y, lam), jac=True,
                 method="L-BFGS-B", options=dict(maxiter=2000, gtol=1e-10, ftol=1e-14))
    theta = np.zeros(X.shape[1]+1); theta[idx] = r.x[:-1]; theta[-1] = r.x[-1]
    return theta, r

def evaluate(theta, X, y, sex, lam=0.0):
    w, b = theta[:-1], theta[-1]
    p = sigmoid(X@w+b)
    pred = (p >= 0.5).astype(int)
    return dict(ce=float(np.mean(np.logaddexp(0, X@w+b) - y*(X@w+b))),
                acc=float((pred==y).mean()), auc=float(roc_auc_score(y, p)),
                dp_gap=float(abs(p[sex==1].mean()-p[sex==0].mean())),       # mean-risk parity
                pos_rate_gap=float(abs(pred[sex==1].mean()-pred[sex==0].mean())),
                fnr_gap=float(abs((pred[(sex==1)&(y==1)]==0).mean()-(pred[(sex==0)&(y==1)]==0).mean())))

if __name__ == "__main__":
    d = build(); X, y = d["Xtr"], d["ytr"]; lam = 1e-2
    theta, r = fit(X, y, lam)
    J, g = loss_grad(theta, X, y, lam)
    # verification 1: gradient ~ 0 ; 2: Hessian PD ; 3: agree with sklearn
    p = sigmoid(X@theta[:-1]+theta[-1]); Xa = np.c_[X, np.ones(len(y))]
    H = Xa.T@(Xa*(p*(1-p))[:,None])/len(y) + lam*np.diag(np.r_[np.ones(X.shape[1]),0])
    ev = np.linalg.eigvalsh(H)
    sk = LogisticRegression(C=1/(lam*len(y)), tol=1e-12, max_iter=5000).fit(X, y)
    sk_theta = np.append(sk.coef_[0], sk.intercept_[0])
    Jsk, _ = loss_grad(sk_theta, X, y, lam)
    print(f"lambda={lam}  J(easy, train) = {J:.6f}   sklearn J = {Jsk:.6f}")
    print(f"||grad||_inf = {np.abs(g).max():.2e}   min Hessian eigenvalue = {ev.min():.2e} (>0 => strictly convex)")
    print(f"max |theta - theta_sklearn| = {np.abs(theta-sk_theta).max():.2e}")
    res = dict(lam=lam, J_train=J, train=evaluate(theta, X, y, d["sex_tr"]), test=evaluate(theta, d["Xte"], d["yte"], d["sex_te"]))
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(ROOT,"results","easy_baseline.json"),"w"), indent=1)
    np.save(os.path.join(ROOT,"results","easy_theta.npy"), theta)
    print("\nlargest |weights|:")
    for i in np.argsort(-np.abs(theta[:-1]))[:8]: print(f"  {d['names'][i]:14s} {theta[i]:+.3f}")
