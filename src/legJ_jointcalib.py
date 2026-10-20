"""
Leg J -- Joint prediction / calibration with coregionalization (the pivot).

Selection didn't benefit from coupling (legM). The classical home of coupling is
JOINT prediction: an ICM multi-output GP models the cross-target covariance of the
predictive distribution, which independent single-output GPs structurally cannot.
So independent models can be well-calibrated *per target* yet wrong on the JOINT
credible region whenever targets carry residual correlation.

We compare, out-of-fold, predicting all T targets from frozen features Phi:
  - Indep : one single-output GP per target -> joint predictive = diagonal Gaussian
  - ICM   : coregionalized GP (full B) -> joint predictive = full T x T Gaussian

Metrics (joint predictive N(mu_k, Sigma_k) vs true y_k):
  - per-target R^2 (mean)             does coupling help point prediction?
  - joint NLL (multivariate Gaussian) proper score for the JOINT distribution
  - joint calibration: d^2 = (y-mu)^T Sigma^-1 (y-mu) ~ chi^2_T if calibrated;
    cal_err = mean_p |emp_coverage(p) - p| over chi^2_T quantiles
  - per-target marginal coverage (sanity; both should be ~nominal)

Gate: if ICM beats Indep on joint NLL and joint cal_err (esp. where targets are
correlated), multi-output survives as a *calibration* contribution.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import chi2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# data_utils (PHO/SH/WUR loaders) is not redistributed; see CALIB_PAPER_DIR in the README
_CALIB = os.environ.get("CALIB_PAPER_DIR", "")
if _CALIB and _CALIB not in sys.path:
    sys.path.insert(0, _CALIB)


# --------------------------------------------------------------------------- #
def _rbf(A, B, ell):
    return np.exp(-cdist(A, B, "sqeuclidean") / (2 * ell ** 2))


def _neg_lml_eig(p, lam_x, Acoef, n, T):
    sig2 = np.exp(p[-1])
    w, dd = p[:T], np.exp(p[T:2 * T])
    B = np.outer(w, w) + np.diag(dd)
    lam_b, Ub = np.linalg.eigh(B)
    M = Acoef @ Ub
    denom = np.outer(lam_x, lam_b) + sig2
    return 0.5 * (np.sum(M ** 2 / denom) + np.sum(np.log(denom)) + n * T * np.log(2 * np.pi))


def fit_icm(Phi, Y, ell, sig2_cap=0.5):
    """Fit the ICM coregionalization B and noise sigma^2 at a FIXED (shared,
    robustly-fit) lengthscale ell, via the Kronecker eigen-trick. The noise is
    bounded to sig2_cap (Y is standardized to unit variance, so capping sigma^2
    at 0.5 forces the model to explain >=50% of each target's variance as signal
    -- this prevents the degenerate collapse to a wide-bar mean predictor)."""
    n, T = Y.shape
    Kx = np.exp(-squareform(pdist(Phi, "sqeuclidean")) / (2 * ell ** 2)) + 1e-6 * np.eye(n)
    lam_x, Ux = np.linalg.eigh(Kx)
    Acoef = Ux.T @ Y
    p0 = np.r_[np.full(T, 0.3), np.zeros(T), np.log(0.1)]
    bounds = [(-5.0, 5.0)] * T + [(np.log(1e-3), np.log(3.0))] * T \
        + [(np.log(1e-4), np.log(sig2_cap))]
    res = minimize(_neg_lml_eig, p0, args=(lam_x, Acoef, n, T),
                   method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
    w, dd = res.x[:T], np.exp(res.x[T:2 * T])
    return np.outer(w, w) + np.diag(dd), float(np.exp(res.x[-1]))


def icm_predict(Phi_tr, Y_tr, Phi_te, ell, B, sig2):
    """Joint posterior predictive (mean + T x T covariance incl. noise) per test point."""
    n, T = Y_tr.shape
    Kx = _rbf(Phi_tr, Phi_tr, ell) + 1e-6 * np.eye(n)
    Ktr = np.kron(B, Kx) + sig2 * np.eye(n * T)        # target-major vec ordering
    c = cho_factor(Ktr, lower=True)
    y = Y_tr.T.reshape(-1)                              # [t0 all n; t1 all n; ...]
    alpha = cho_solve(c, y)
    kxs = _rbf(Phi_te, Phi_tr, ell)                    # nte x n
    nte = len(Phi_te)
    mu = np.zeros((nte, T)); Sig = np.zeros((nte, T, T))
    for k in range(nte):
        Kk = np.kron(B, kxs[k:k + 1, :])               # T x nT
        mu[k] = Kk @ alpha
        V = cho_solve(c, Kk.T)                          # nT x T
        Sig[k] = B - Kk @ V + sig2 * np.eye(T)         # predictive (obs) cov
    return mu, Sig


def indep_predict(Phi_tr, Y_tr, Phi_te, random_state=0):
    """Per-target single-output GP -> diagonal joint predictive.
    Also returns each target's fitted RBF lengthscale so the ICM can share a
    robust kernel (the median) instead of a coarse grid -- this removes the
    degenerate-humility failure mode where a bad lengthscale forced the ICM to
    explain signal as noise and collapse to mean-prediction.

    `random_state` pins the optimizer-restart sampling so runs are reproducible
    (otherwise n_restarts_optimizer draws from the unpinned global RNG)."""
    nte, T = len(Phi_te), Y_tr.shape[1]
    mu = np.zeros((nte, T)); Sig = np.zeros((nte, T, T)); ells = []
    for t in range(T):
        k = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e2)) + WhiteKernel(0.1, (1e-4, 1e1))
        gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1,
                                      alpha=1e-6, random_state=random_state)
        gp.fit(Phi_tr, Y_tr[:, t])
        m, s = gp.predict(Phi_te, return_std=True)
        mu[:, t] = m
        Sig[:, t, t] = np.maximum(s, 1e-6) ** 2
        ells.append(gp.kernel_.k1.k2.length_scale)
    return mu, Sig, float(np.median(ells))


def joint_metrics(Y, mu, Sig):
    n, T = Y.shape
    nll, d2 = np.zeros(n), np.zeros(n)
    for k in range(n):
        S = Sig[k] + 1e-8 * np.eye(T)
        L = np.linalg.cholesky(S)
        r = Y[k] - mu[k]
        sol = np.linalg.solve(L, r)
        d2[k] = sol @ sol
        logdet = 2 * np.sum(np.log(np.diag(L)))
        nll[k] = 0.5 * (T * np.log(2 * np.pi) + logdet + d2[k])
    grid = np.linspace(0.05, 0.95, 19)
    cal = np.mean([abs(np.mean(d2 <= chi2.ppf(p, T)) - p) for p in grid])
    r2 = np.mean([r2_score(Y[:, t], mu[:, t]) for t in range(T)])
    return dict(joint_nll=float(nll.mean()), joint_cal_err=float(cal), mean_r2=float(r2))


def run_dataset(name, X, Y, n_sub=280, cv=5):  # n_sub overridable via CLI
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    if len(Xs) > n_sub:
        idx = np.random.default_rng(0).choice(len(Xs), n_sub, replace=False)
        Xs, Ys = Xs[idx], Ys[idx]
    n, T = Ys.shape
    P = {m: (np.zeros((n, T)), np.zeros((n, T, T))) for m in ("ICM", "Indep")}
    for tr, te in KFold(cv, shuffle=True, random_state=0).split(Xs):
        mu_i, Sig_i, ell = indep_predict(Xs[tr], Ys[tr], Xs[te])   # robust shared kernel
        P["Indep"][0][te], P["Indep"][1][te] = mu_i, Sig_i
        B, sig2 = fit_icm(Xs[tr], Ys[tr], ell)                     # same ell, +coupling
        mu_c, Sig_c = icm_predict(Xs[tr], Ys[tr], Xs[te], ell, B, sig2)
        P["ICM"][0][te], P["ICM"][1][te] = mu_c, Sig_c
    rows = []
    C = np.corrcoef(Ys.T)
    mc = np.mean(np.abs(C[np.triu_indices(T, 1)]))
    for m in ("ICM", "Indep"):
        rows.append(dict(dataset=name, n=n, T=T, mean_target_corr=round(float(mc), 2),
                         method=m, **joint_metrics(Ys, *P[m])))
    return rows


PLANT_DATASETS = ("PHO", "SH", "WUR", "rice", "wheat")


def dataset_available(name):
    """True if <name> can be loaded: a data_public/<name>.npz exists, or it is a
    plant set and CALIB_PAPER_DIR points at the companion data loaders."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
    if os.path.exists(os.path.join(here, f"data_public/{name.lower()}.npz")):
        return True
    return name in PLANT_DATASETS and bool(os.environ.get("CALIB_PAPER_DIR"))


def load(name):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
    if name in ("rice", "wheat"):
        # PHO split by species: each is its own homogeneous multi-target set
        import data_utils as du
        rows = [v for v in du.load_pho().values()
                if str(v["species"]).lower() == name]
        X = np.array([v["mean_spec"] for v in rows], float)
        vc = np.array([v["vcmax"] for v in rows], float)
        jx = np.array([v["jmax"] for v in rows], float)
        ok = np.all(np.isfinite(X), 1) & np.isfinite(vc) & np.isfinite(jx)
        return X[ok], np.column_stack([vc, jx])[ok]
    if name in ("PHO", "SH", "WUR"):
        from gateB_recovery import load_plants
        X, vc, jx, _ = load_plants()[name]
        return X, np.column_stack([vc, jx])
    p = os.path.join(here, f"data_public/{name.lower()}.npz")
    z = np.load(p, allow_pickle=True)
    return z["X"], z["Y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Tecator,jura,atp1d,energy,PHO")
    ap.add_argument("--n_sub", type=int, default=280)
    ap.add_argument("--out", default="results/legJ/jointcalib.csv")
    args = ap.parse_args()
    rows = []
    for name in args.datasets.split(","):
        X, Y = load(name)
        print(f"[{name}] n={len(X)} T={Y.shape[1]} p={X.shape[1]} ...", flush=True)
        rows += run_dataset(name, X, Y, n_sub=args.n_sub)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 160)
    print("\n=== Leg J: joint prediction / calibration (ICM vs Indep) ===")
    print(df.round(3).to_string(index=False))
    piv = df.pivot_table(index="dataset", columns="method",
                         values=["joint_nll", "joint_cal_err"])
    print("\nICM - Indep (negative = ICM better):")
    delta = pd.DataFrame({
        "d_joint_nll": piv["joint_nll"]["ICM"] - piv["joint_nll"]["Indep"],
        "d_joint_cal_err": piv["joint_cal_err"]["ICM"] - piv["joint_cal_err"]["Indep"],
    })
    print(delta.round(3).to_string())
    print("\nGATE: ICM should give lower joint_nll and lower joint_cal_err, esp. high mean_target_corr.")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
