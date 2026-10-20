"""
Residual-ICM: a method the residual-correlation analysis earns.

Plain ICM re-fits the full joint GP and overfits the residual covariance in the
ceiling regime (tecator: R2_indep=0.946, residuals tiny but directionally
correlated -> ICM models noise as coupling -> dNLL=+0.89). Our diagnostic
D_logdet=-0.5 logdet(R_res) says the structure is THERE; the problem is the
estimator, not the signal.

Residual-ICM fixes the estimator. It does NOT fit a joint GP. It:
  1. fits independent single-output GPs (marginal means mu_t and variances s_t^2),
  2. estimates the residual correlation R_res from INNER-CV out-of-fold residuals
     on the training set (no leakage), with Ledoit-Wolf-style shrinkage,
  3. forms the joint predictive covariance  Sigma = diag(s) R_res diag(s),
     leaving the marginal means and variances UNTOUCHED.

Consequences by construction:
  - point prediction (R^2) is IDENTICAL to Indep (means unchanged) -> dR2 == 0,
  - joint NLL improves exactly when residuals are genuinely correlated,
  - at the ceiling the marginal fits (which are excellent) are never disturbed,
    so it cannot blow up the way joint ICM does.

We compare Indep / ICM(hardened, paper default) / ResICM on all 11 datasets and
test the DOMINANCE claim: ResICM should never be meaningfully worse than the
better of {Indep, ICM} on joint NLL, while keeping R^2 = Indep.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legJ_jointcalib import (fit_icm, icm_predict, indep_predict,
                             joint_metrics, load)

# 16 converged datasets (PHO split into rice+wheat; +scm1d/scm20d/edm/sarcos;
# oes10/oes97 dropped -- GP optimiser non-convergence at p~=n, see diagnostic).
DATASETS = ["atp1d", "atp7d", "rf1", "rf2", "WUR", "SH", "rice", "wheat",
            "jura", "wq", "energy", "tecator", "scm1d", "scm20d", "edm", "sarcos"]
from legJ_jointcalib import dataset_available  # noqa: E402
DATASETS = [d for d in DATASETS if dataset_available(d)]  # skip unavailable sets


def _shrink_corr(resid, shrink=None):
    """Residual correlation with optional shrinkage toward identity.
    Default shrink: simple Ledoit-Wolf-like intensity from sample size."""
    n, T = resid.shape
    R = np.corrcoef(resid.T)
    R = np.atleast_2d(R)
    R = 0.5 * (R + R.T)
    if shrink is None:
        # heavier shrinkage when T large relative to n (noisy off-diagonals)
        shrink = min(1.0, T / max(n, 1))
    R = (1 - shrink) * R + shrink * np.eye(T)
    # PSD guard
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-6, None)
    R = (V * w) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def residual_corr_from_train(Xtr, Ytr, inner_cv=5, seed=1):
    """Residual correlation from inner-CV out-of-fold residuals, STANDARDISED by
    the inner-fold predictive sd.

    R_res is applied downstream as Sigma(x) = diag(s(x)) R_res diag(s(x)), i.e.
    as the correlation of the predictive z-scores z_t(x) = r_t(x) / s_t(x), not
    of the raw residuals. Estimating it from raw residuals only matches that
    model when s_t(x)/sigma_t is constant in x for every target; empirically the
    within-target dispersion of s_t(x) averages ~0.40 across the 16 datasets, so
    the two estimators differ (off-diagonal |R_raw - R_z| up to 0.31) and the
    matched one is uniformly better on joint NLL. See exp_resicm_zresid.py.
    """
    n, T = Ytr.shape
    resid = np.zeros((n, T))
    sd = np.zeros((n, T))
    k = min(inner_cv, n) if n >= 4 else 2
    for itr, ite in KFold(k, shuffle=True, random_state=seed).split(Xtr):
        mi, Si, _ = indep_predict(Xtr[itr], Ytr[itr], Xtr[ite])
        resid[ite] = Ytr[ite] - mi
        sd[ite] = np.sqrt(np.maximum(np.diagonal(Si, axis1=1, axis2=2), 1e-12))
    return _shrink_corr(resid / sd)


def resicm_predict(Xtr, Ytr, Xte, R_res):
    """Indep marginals + plug-in residual correlation on the joint covariance."""
    mu, Sig_diag, _ = indep_predict(Xtr, Ytr, Xte)
    nte, T = mu.shape
    Sig = np.zeros((nte, T, T))
    for k in range(nte):
        d = np.sqrt(np.maximum(np.diag(Sig_diag[k]), 1e-12))
        Sig[k] = np.outer(d, d) * R_res
    return mu, Sig


def run_dataset(name, n_sub=280, cv=5, seed=0):
    X, Y = load(name)
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    if len(Xs) > n_sub:
        idx = np.random.default_rng(0).choice(len(Xs), n_sub, replace=False)
        Xs, Ys = Xs[idx], Ys[idx]
    n, T = Ys.shape

    P = {m: (np.zeros((n, T)), np.zeros((n, T, T)))
         for m in ("Indep", "ICM", "ResICM")}
    for tr, te in KFold(cv, shuffle=True, random_state=seed).split(Xs):
        mi, Si, ell = indep_predict(Xs[tr], Ys[tr], Xs[te])
        P["Indep"][0][te], P["Indep"][1][te] = mi, Si
        B, sig2 = fit_icm(Xs[tr], Ys[tr], ell)
        mc, Sc = icm_predict(Xs[tr], Ys[tr], Xs[te], ell, B, sig2)
        P["ICM"][0][te], P["ICM"][1][te] = mc, Sc
        R_res = residual_corr_from_train(Xs[tr], Ys[tr])
        mr, Sr = resicm_predict(Xs[tr], Ys[tr], Xs[te], R_res)
        P["ResICM"][0][te], P["ResICM"][1][te] = mr, Sr

    M = {m: joint_metrics(Ys, *P[m]) for m in P}
    base = M["Indep"]
    rows = []
    for m in ("Indep", "ICM", "ResICM"):
        rows.append(dict(
            dataset=name, T=T, method=m,
            nll=round(M[m]["joint_nll"], 3),
            cal=round(M[m]["joint_cal_err"], 3),
            r2=round(M[m]["mean_r2"], 3),
            dnll=round(M[m]["joint_nll"] - base["joint_nll"], 3),
            dr2=round(M[m]["mean_r2"] - base["mean_r2"], 3)))
    return rows


def main():
    rows = []
    for d in DATASETS:
        print(f"[{d}] ...", flush=True)
        rows += run_dataset(d)
    df = pd.DataFrame(rows)
    out = "results/legJ/residual_icm.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    piv = df.pivot(index="dataset", columns="method", values="dnll")[["ICM", "ResICM"]]
    print("\n=== dNLL vs Indep (negative = better than Indep) ===")
    print(piv.round(3).to_string())
    print("\nMean dNLL:  ICM={:+.3f}   ResICM={:+.3f}".format(
        piv["ICM"].mean(), piv["ResICM"].mean()))

    print("\n--- DOMINANCE CHECK ---")
    icm_loses = (piv["ICM"] > 0.02).sum()
    res_loses = (piv["ResICM"] > 0.02).sum()
    print(f"  ICM    worse than Indep (dNLL>+0.02): {icm_loses}/{len(piv)} datasets")
    print(f"  ResICM worse than Indep (dNLL>+0.02): {res_loses}/{len(piv)} datasets")
    best_indep_icm = np.minimum(piv["ICM"], 0.0)   # better of {Indep(0), ICM}
    res_vs_best = (piv["ResICM"] <= best_indep_icm + 0.05).sum()
    print(f"  ResICM within 0.05 of best(Indep,ICM): {res_vs_best}/{len(piv)} datasets")

    dr2 = df[df.method == "ResICM"]["dr2"].abs().max()
    print(f"\n  max |dR2| for ResICM (should be ~0 by construction): {dr2:.4f}")
    print("\n  Tecator (ceiling) detail:")
    print(df[df.dataset == "tecator"].to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
