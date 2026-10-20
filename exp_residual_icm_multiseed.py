"""
Multi-seed version of exp_residual_icm.py (Table 4 / tab:method).

WHY: Table 2 (tab:datasets_full) is averaged over 5 seeds, but the original
exp_residual_icm.py (Table 4) was a SINGLE seed=0 run with a *fixed* subsample
(default_rng(0)) and no random_state on indep_predict. That mismatch makes the
two tables disagree per-dataset -- most visibly wq, whose ICM dNLL flips sign
(+0.20 single-seed vs -0.14 over 5 seeds) because its effect is ~0 with
seed-to-seed SD larger than the mean.

This script re-runs Indep / ICM / Residual-ICM under the SAME protocol as the
diagnostic (exp_residual_diagnostic.py):
  - subsample with default_rng(SEED)        (varies by seed; was fixed rng(0))
  - KFold(random_state=SEED)
  - indep_predict(random_state=SEED)
over SEEDS=[0,1,2,3,4], and reports per-dataset seed mean +/- SD.

Consequently the ICM dNLL column here should reproduce Table 2's dNLL column
(same Indep+ICM fits, same seeds), and ResICM gets a proper 5-seed average to
replace the single-seed Table 4 values. We also emit D_logdet and r2_indep per
seed so the diagnostic-gated column can be recomputed on the same footing.
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

DATASETS = ["atp1d", "atp7d", "rf1", "rf2", "WUR", "SH", "rice", "wheat",
            "jura", "wq", "energy", "tecator", "scm1d", "scm20d", "edm", "sarcos"]
from legJ_jointcalib import dataset_available  # noqa: E402
DATASETS = [d for d in DATASETS if dataset_available(d)]  # skip unavailable sets
SEEDS = [0, 1, 2, 3, 4]


def _shrink_corr(resid, shrink=None):
    """Residual correlation with optional shrinkage toward identity (ResICM)."""
    n, T = resid.shape
    R = np.atleast_2d(np.corrcoef(resid.T))
    R = 0.5 * (R + R.T)
    if shrink is None:
        shrink = min(1.0, T / max(n, 1))
    R = (1 - shrink) * R + shrink * np.eye(T)
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-6, None)
    R = (V * w) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def _resid_corr_plain(resid):
    """Plain empirical residual correlation (matches the diagnostic's D_logdet)."""
    R = np.atleast_2d(np.corrcoef(resid.T))
    R = 0.5 * (R + R.T)
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
    mu, Sig_diag, _ = indep_predict(Xtr, Ytr, Xte)
    nte, T = mu.shape
    Sig = np.zeros((nte, T, T))
    for k in range(nte):
        d = np.sqrt(np.maximum(np.diag(Sig_diag[k]), 1e-12))
        Sig[k] = np.outer(d, d) * R_res
    return mu, Sig


def run_dataset_seed(name, seed, n_sub=280, cv=5):
    X, Y = load(name)
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    if len(Xs) > n_sub:
        idx = np.random.default_rng(seed).choice(len(Xs), n_sub, replace=False)
        Xs, Ys = Xs[idx], Ys[idx]
    n, T = Ys.shape

    P = {m: (np.zeros((n, T)), np.zeros((n, T, T)))
         for m in ("Indep", "ICM", "ResICM")}
    for tr, te in KFold(cv, shuffle=True, random_state=seed).split(Xs):
        mi, Si, ell = indep_predict(Xs[tr], Ys[tr], Xs[te], random_state=seed)
        P["Indep"][0][te], P["Indep"][1][te] = mi, Si
        B, sig2 = fit_icm(Xs[tr], Ys[tr], ell)
        mc, Sc = icm_predict(Xs[tr], Ys[tr], Xs[te], ell, B, sig2)
        P["ICM"][0][te], P["ICM"][1][te] = mc, Sc
        R_res = residual_corr_from_train(Xs[tr], Ys[tr], seed=seed)
        mr, Sr = resicm_predict(Xs[tr], Ys[tr], Xs[te], R_res)
        P["ResICM"][0][te], P["ResICM"][1][te] = mr, Sr

    M = {m: joint_metrics(Ys, *P[m]) for m in P}
    base = M["Indep"]

    # D_logdet from out-of-fold Indep residuals (plain, matches diagnostic)
    resid = Ys - P["Indep"][0]
    R_plain = _resid_corr_plain(resid)
    D_logdet = float(-0.5 * np.log(max(np.linalg.det(R_plain), 1e-12)))
    r2_indep = base["mean_r2"]

    rows = []
    for m in ("Indep", "ICM", "ResICM"):
        rows.append(dict(
            dataset=name, seed=seed, T=T, method=m,
            nll=M[m]["joint_nll"], cal=M[m]["joint_cal_err"],
            r2=M[m]["mean_r2"],
            dnll=M[m]["joint_nll"] - base["joint_nll"],
            dr2=M[m]["mean_r2"] - base["mean_r2"],
            D_logdet=D_logdet, r2_indep=r2_indep))
    return rows


def gate_column(summary, tau=0.5, ceiling=0.9):
    """Fixed-tau diagnostic gate: couple (use ICM dNLL) iff D_logdet>tau AND
    r2_indep<ceiling, else Indep (0). Computed on seed-mean D_logdet/r2_indep."""
    icm = summary.set_index("dataset")["ICM_dnll_mean"]
    g = {}
    for ds, row in summary.set_index("dataset").iterrows():
        couple = (row["D_logdet_mean"] > tau) and (row["r2_indep_mean"] < ceiling)
        g[ds] = icm[ds] if couple else 0.0
    return pd.Series(g, name="Gate_dnll")


def main():
    long_rows = []
    for d in DATASETS:
        for s in SEEDS:
            print(f"[seed {s}] {d} ...", flush=True)
            long_rows += run_dataset_seed(d, s)
    L = pd.DataFrame(long_rows)
    os.makedirs("results/legJ", exist_ok=True)
    L.to_csv("results/legJ/residual_icm_multiseed_long.csv", index=False)

    # per-dataset seed mean / SD of dNLL for each method
    piv_mean = L.pivot_table(index="dataset", columns="method", values="dnll", aggfunc="mean")
    piv_sd = L.pivot_table(index="dataset", columns="method", values="dnll", aggfunc="std")
    diag = L.groupby("dataset")[["D_logdet", "r2_indep"]].mean()

    summ = pd.DataFrame({
        "dataset": piv_mean.index,
        "ICM_dnll_mean": piv_mean["ICM"].values,
        "ICM_dnll_sd": piv_sd["ICM"].values,
        "ResICM_dnll_mean": piv_mean["ResICM"].values,
        "ResICM_dnll_sd": piv_sd["ResICM"].values,
        "D_logdet_mean": diag["D_logdet"].values,
        "r2_indep_mean": diag["r2_indep"].values,
    })
    gate = gate_column(summ)
    summ = summ.set_index("dataset")
    summ["Gate_dnll"] = gate
    summ = summ.reset_index()
    summ.to_csv("results/legJ/residual_icm_multiseed_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== 5-seed mean dNLL (negative = better than Indep) ===")
    show = summ.copy()
    for c in ["ICM_dnll_mean", "ICM_dnll_sd", "ResICM_dnll_mean",
              "ResICM_dnll_sd", "D_logdet_mean", "r2_indep_mean", "Gate_dnll"]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False))

    print("\n--- MEANS over 16 datasets (5-seed) ---")
    print("  Indep  : 0.000")
    print(f"  ICM    : {summ['ICM_dnll_mean'].mean():+.3f}  (single-seed Table 4 was -0.53)")
    print(f"  ResICM : {summ['ResICM_dnll_mean'].mean():+.3f}  (single-seed Table 4 was -0.85)")
    print(f"  Gate   : {summ['Gate_dnll'].mean():+.3f}  (fixed tau=0.5)")

    # mean rank (1=best) per dataset across the four methods
    R = summ.set_index("dataset")[["ICM_dnll_mean", "ResICM_dnll_mean", "Gate_dnll"]].copy()
    R["Indep"] = 0.0
    R = R[["Indep", "ICM_dnll_mean", "ResICM_dnll_mean", "Gate_dnll"]]
    ranks = R.rank(axis=1)
    print("\n--- mean rank (1=best) ---")
    print(ranks.mean().round(2).to_string())
    print("\nSaved: results/legJ/residual_icm_multiseed_{long,summary}.csv")


if __name__ == "__main__":
    main()
