"""
Part B, corrected: how much residual dependence does D_logdet miss?

The first version compared against an analytic "true generative" NLL that
assumed r2 is a deterministic function of r1. That is false for the FITTED
residuals -- they carry GP estimation error on top of the generative noise -- so
the resulting gap (-26.5 nats) was meaningless and is discarded.

Instead we bound the missed dependence empirically: fit a cross-validated
nonparametric conditional model of r2 | r1 (k-NN estimate of both the
conditional mean and the conditional variance) and compare its joint NLL against
(a) the diagonal Gaussian and (b) the Gaussian coupling that Residual-ICM can
express. Everything is out-of-fold, so the comparison is honest.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_diagnostic_scope import (N, CV, D_X, SEEDS, _shrink_corr, distance_corr,
                                  dlogdet, run_fold_preds, signal)
from legJ_jointcalib import joint_metrics


def knn_conditional_nll(r1, r2, k=25, seed=0):
    """Out-of-fold NLL of r2 | r1 under a k-NN Gaussian conditional."""
    n = len(r1)
    nll = np.zeros(n)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(r1):
        nn = NearestNeighbors(n_neighbors=min(k, len(tr))).fit(r1[tr, None])
        _, idx = nn.kneighbors(r1[te, None])
        nb = r2[tr][idx]                       # (n_te, k) neighbour responses
        m, v = nb.mean(1), nb.var(1) + 1e-3
        nll[te] = 0.5 * np.log(2 * np.pi * v) + 0.5 * (r2[te] - m) ** 2 / v
    return float(nll.mean())


def gauss_nll_1d(x):
    v = x.var() + 1e-12
    return float(0.5 * np.log(2 * np.pi * v) + 0.5 * np.mean(x ** 2) / v)


def run(kind, seed):
    rng = np.random.default_rng(100 + seed)
    T = 2
    X = rng.uniform(-1.5, 1.5, size=(N, D_X))
    F = signal(X, T, rng)
    e1 = rng.normal(size=N)
    if kind == "nonlinear":
        e2 = (e1 ** 2 - 1.0) / np.sqrt(2.0)
    elif kind == "linear":
        e2 = 0.7 * e1 + np.sqrt(1 - 0.7 ** 2) * rng.normal(size=N)
    else:
        e2 = rng.normal(size=N)
    Y = F + 0.6 * np.column_stack([e1, e2])
    Xs = (X - X.mean(0)) / X.std(0)
    Ys = (Y - Y.mean(0)) / Y.std(0)

    mu, Sig, _ = run_fold_preds(Xs, Ys, seed)
    resid = Ys - mu
    R = _shrink_corr(resid)
    base = joint_metrics(Ys, mu, Sig)["joint_nll"]

    nte, _ = mu.shape
    S = np.zeros((nte, T, T))
    for k in range(nte):
        d = np.sqrt(np.maximum(np.diag(Sig[k]), 1e-12))
        S[k] = np.outer(d, d) * R
    nll_res = joint_metrics(Ys, mu, S)["joint_nll"]

    r1, r2 = resid[:, 0], resid[:, 1]
    nll_diag_pair = gauss_nll_1d(r1) + gauss_nll_1d(r2)
    nll_knn_pair = gauss_nll_1d(r1) + knn_conditional_nll(r1, r2, seed=seed)
    return dict(kind=kind, seed=seed, D_logdet=dlogdet(R),
                pearson_abs=float(abs(np.corrcoef(resid.T)[0, 1])),
                distance_corr=distance_corr(r1, r2),
                dnll_resicm=nll_res - base,
                gain_gaussian_coupling=nll_res - base,
                gain_knn_conditional=nll_knn_pair - nll_diag_pair)


def main():
    rows = [run(k, s) for k in ("independent", "linear", "nonlinear") for s in SEEDS]
    df = pd.DataFrame(rows)
    os.makedirs("results/scope", exist_ok=True)
    df.to_csv("results/scope/nonlinear_fixed.csv", index=False)
    g = df.groupby("kind")[["D_logdet", "pearson_abs", "distance_corr",
                            "gain_gaussian_coupling", "gain_knn_conditional"]].mean()
    print("\n=== Part B (corrected): nonlinear residual dependence, seed means ===")
    print(g.round(3).to_string())
    print("\ngain_* are NLL changes vs the diagonal model (negative = better).")
    print("Saved results/scope/nonlinear_fixed.csv")


if __name__ == "__main__":
    main()
