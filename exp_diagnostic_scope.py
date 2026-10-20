"""
Scope of the residual diagnostic: two controlled blind spots.

Global: R_res is a single GLOBAL correlation matrix. If the residual correlation
       depends on the input region, the global estimate averages over regions.
Linear: R_res is a PEARSON correlation, so it sees only linear residual
       dependence.

Part A (nonstationary). Targets share a residual correlation +rho on one half of
the input space and -rho on the other. The stationary control uses +rho
everywhere. We report the global D_logdet, Residual-ICM with the global R_res,
and an ORACLE region-wise Residual-ICM that is told the region label.

Part B (nonlinear). Two targets whose residuals are strongly dependent but
Pearson-uncorrelated (e2 = (e1^2 - 1)/sqrt(2)). We report D_logdet, Pearson |r|,
distance correlation, Residual-ICM, and the gap between the diagonal-Gaussian
joint NLL and the TRUE generative joint NLL -- the part a non-Gaussian joint
model could still capture and a Gaussian coregionalization cannot.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legJ_jointcalib import indep_predict, joint_metrics

SEEDS = [0, 1, 2]
N, D_X, CV = 360, 4, 5


def _shrink_corr(resid):
    n, T = resid.shape
    R = np.atleast_2d(np.corrcoef(resid.T))
    R = 0.5 * (R + R.T)
    lam = min(1.0, T / max(n, 1))
    R = (1 - lam) * R + lam * np.eye(T)
    w, V = np.linalg.eigh(R)
    R = (V * np.clip(w, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def dlogdet(R):
    return float(-0.5 * np.log(max(np.linalg.det(R), 1e-12)))


def distance_corr(x, y):
    """Szekely distance correlation: 0 iff independent, catches nonlinearity."""
    x, y = np.asarray(x, float)[:, None], np.asarray(y, float)[:, None]
    a = np.abs(x - x.T)
    b = np.abs(y - y.T)
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    dvx, dvy = (A * A).mean(), (B * B).mean()
    return float(np.sqrt(max(dcov2, 0) / np.sqrt(max(dvx * dvy, 1e-12))))


def signal(X, T, rng):
    """Smooth nonlinear mean functions so the independent GPs have real signal."""
    W = rng.normal(size=(X.shape[1], T))
    return np.sin(X @ W) + 0.5 * (X @ W)


def resicm_from_resid(mu, Sig, R):
    nte, T = mu.shape
    S = np.zeros((nte, T, T))
    for k in range(nte):
        d = np.sqrt(np.maximum(np.diag(Sig[k]), 1e-12))
        S[k] = np.outer(d, d) * R
    return mu, S


def run_fold_preds(Xs, Ys, seed):
    """Out-of-fold Indep predictions plus the inner-fold residual correlation."""
    n, T = Ys.shape
    mu = np.zeros((n, T)); Sig = np.zeros((n, T, T))
    folds = []
    for tr, te in KFold(CV, shuffle=True, random_state=seed).split(Xs):
        mi, Si, _ = indep_predict(Xs[tr], Ys[tr], Xs[te], random_state=seed)
        mu[te], Sig[te] = mi, Si
        folds.append((tr, te))
    return mu, Sig, folds


# ----------------------------------------------------------------- Part A
def part_a(rho, nonstationary, seed):
    rng = np.random.default_rng(seed)
    T = 4
    X = rng.uniform(-1.5, 1.5, size=(N, D_X))
    F = signal(X, T, rng)
    region = (X[:, 0] > 0).astype(int)              # two input regions
    E = np.zeros((N, T))
    for r in (0, 1):
        m = region == r
        sgn = 1.0 if (r == 1 or not nonstationary) else -1.0
        R = (1 - sgn * rho) * np.eye(T) + sgn * rho * np.ones((T, T))
        w, V = np.linalg.eigh(R)
        L = V * np.sqrt(np.clip(w, 1e-9, None))
        E[m] = rng.normal(size=(m.sum(), T)) @ L.T
    Y = F + 0.6 * E
    Xs = (X - X.mean(0)) / X.std(0)
    Ys = (Y - Y.mean(0)) / Y.std(0)

    mu, Sig, folds = run_fold_preds(Xs, Ys, seed)
    resid = Ys - mu
    R_glob = _shrink_corr(resid)
    base = joint_metrics(Ys, mu, Sig)["joint_nll"]

    # global Residual-ICM
    _, S_glob = resicm_from_resid(mu, Sig, R_glob)
    nll_glob = joint_metrics(Ys, mu, S_glob)["joint_nll"]

    # ORACLE region-wise Residual-ICM: one R_res per region, told the label
    S_or = np.zeros_like(Sig)
    for r in (0, 1):
        m = region == r
        R_r = _shrink_corr(resid[m])
        _, S_r = resicm_from_resid(mu[m], Sig[m], R_r)
        S_or[m] = S_r
    nll_or = joint_metrics(Ys, mu, S_or)["joint_nll"]

    # what the per-region correlations actually are
    rr = [float(np.mean(np.abs(_shrink_corr(resid[region == r])[~np.eye(T, dtype=bool)])))
          for r in (0, 1)]
    return dict(rho=rho, regime="nonstationary" if nonstationary else "stationary",
                seed=seed, D_logdet_global=dlogdet(R_glob),
                mean_abs_offdiag_global=float(np.mean(np.abs(R_glob[~np.eye(T, dtype=bool)]))),
                mean_abs_offdiag_region0=rr[0], mean_abs_offdiag_region1=rr[1],
                dnll_resicm_global=nll_glob - base,
                dnll_resicm_oracle_region=nll_or - base)


# ----------------------------------------------------------------- Part B
def part_b(kind, seed):
    rng = np.random.default_rng(100 + seed)
    T = 2
    X = rng.uniform(-1.5, 1.5, size=(N, D_X))
    F = signal(X, T, rng)
    e1 = rng.normal(size=N)
    if kind == "nonlinear":
        e2 = (e1 ** 2 - 1.0) / np.sqrt(2.0)          # Pearson ~ 0, strongly dependent
    elif kind == "linear":
        e2 = 0.7 * e1 + np.sqrt(1 - 0.7 ** 2) * rng.normal(size=N)
    else:
        e2 = rng.normal(size=N)                      # independent control
    E = np.column_stack([e1, e2])
    Y = F + 0.6 * E
    Xs = (X - X.mean(0)) / X.std(0)
    Ys = (Y - Y.mean(0)) / Y.std(0)

    mu, Sig, _ = run_fold_preds(Xs, Ys, seed)
    resid = Ys - mu
    R = _shrink_corr(resid)
    base = joint_metrics(Ys, mu, Sig)["joint_nll"]
    _, S = resicm_from_resid(mu, Sig, R)
    nll_res = joint_metrics(Ys, mu, S)["joint_nll"]

    # TRUE generative joint NLL of the residual pair, vs the best DIAGONAL
    # Gaussian on the same residuals: the gap is dependence a non-Gaussian joint
    # model could exploit and a Gaussian coregionalization cannot.
    s = 0.6 * np.array([1.0, 1.0])
    z1 = resid[:, 0]
    nll_diag = 0.5 * np.sum(np.log(2 * np.pi * resid.var(0))) \
        + 0.5 * np.mean(np.sum(resid ** 2 / resid.var(0), axis=1))
    if kind == "nonlinear":
        # true model: e2 | e1 is deterministic -> use a tight conditional band
        cond_sd = 0.05
        pred_e2 = ((z1 / s[0]) ** 2 - 1) / np.sqrt(2) * s[1]
        nll_true = 0.5 * np.log(2 * np.pi * resid[:, 0].var()) \
            + 0.5 * np.mean(z1 ** 2 / resid[:, 0].var()) \
            + 0.5 * np.log(2 * np.pi * cond_sd ** 2) \
            + 0.5 * np.mean((resid[:, 1] - pred_e2) ** 2) / cond_sd ** 2
    else:
        nll_true = np.nan
    return dict(kind=kind, seed=seed, D_logdet=dlogdet(R),
                pearson_abs=float(abs(np.corrcoef(resid.T)[0, 1])),
                distance_corr=distance_corr(resid[:, 0], resid[:, 1]),
                dnll_resicm=nll_res - base,
                nll_gap_true_vs_diagonal=(nll_diag - nll_true) if kind == "nonlinear" else np.nan)


def main():
    rows_a = []
    for rho in (0.0, 0.3, 0.5, 0.7, 0.9):
        for ns in (False, True):
            for s in SEEDS:
                rows_a.append(part_a(rho, ns, s))
                print(f"[A] rho={rho} {'nonstat' if ns else 'stat  '} seed={s} "
                      f"D={rows_a[-1]['D_logdet_global']:.3f} "
                      f"glob={rows_a[-1]['dnll_resicm_global']:+.3f} "
                      f"oracle={rows_a[-1]['dnll_resicm_oracle_region']:+.3f}", flush=True)
    A = pd.DataFrame(rows_a)
    os.makedirs("results/scope", exist_ok=True)
    A.to_csv("results/scope/nonstationary.csv", index=False)

    rows_b = []
    for kind in ("independent", "linear", "nonlinear"):
        for s in SEEDS:
            rows_b.append(part_b(kind, s))
            print(f"[B] {kind:12s} seed={s} D={rows_b[-1]['D_logdet']:.3f} "
                  f"pearson={rows_b[-1]['pearson_abs']:.3f} "
                  f"dcor={rows_b[-1]['distance_corr']:.3f} "
                  f"dnll={rows_b[-1]['dnll_resicm']:+.3f}", flush=True)
    B = pd.DataFrame(rows_b)
    B.to_csv("results/scope/nonlinear.csv", index=False)

    print("\n=== Part A: input-dependent residual correlation (seed means) ===")
    sa = A.groupby(["regime", "rho"])[
        ["D_logdet_global", "mean_abs_offdiag_global", "mean_abs_offdiag_region0",
         "mean_abs_offdiag_region1", "dnll_resicm_global", "dnll_resicm_oracle_region"]
    ].mean()
    print(sa.round(3).to_string())

    print("\n=== Part B: nonlinear residual dependence (seed means) ===")
    sb = B.groupby("kind")[["D_logdet", "pearson_abs", "distance_corr", "dnll_resicm",
                            "nll_gap_true_vs_diagonal"]].mean()
    print(sb.round(3).to_string())
    print("\nSaved results/scope/{nonstationary,nonlinear}.csv")


if __name__ == "__main__":
    main()
