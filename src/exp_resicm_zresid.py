"""
Standardised-residual R_res check.

The raw-residual estimator does not match the predictive-covariance model.
Residual-ICM estimates
    R_raw = corr(r)                      r = raw inner-OOF residuals
but applies it as
    Sigma(x) = diag(s(x)) R_res diag(s(x))     s(x) = GP predictive sd,
i.e. it uses R_res as the correlation of the predictive *z-scores*
z_t(x) = r_t(x) / s_t(x), not of the raw residuals. The two coincide exactly
when s_t(x)/sigma_t is constant in x for each target; they can differ when the
predictive sd varies with x differently across targets.

This script runs the matched estimator alongside the shipped one:
    ResICM_raw : R_res = shrink_corr(r)          (as published)
    ResICM_z   : R_res = shrink_corr(r / s)      (matched to the model)
under the exp_residual_icm_multiseed.py protocol (subsample rng(SEED),
KFold(random_state=SEED), indep_predict(random_state=SEED)), 16 datasets x 5
seeds, and reports both the joint-NLL difference and the entrywise distance
between R_raw and R_z. Also reports the per-target calibration ratio
std(r_t)/mean(s_t), which is the quantity that has to be x-varying for the two
estimators to disagree.

Rows are appended per (dataset, seed) so the run is resumable.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_hardening_ablation import load
from exp_residual_icm_multiseed import _shrink_corr, resicm_predict
from legJ_jointcalib import indep_predict, joint_metrics

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
OUT = os.path.join(_HERE, "results/legJ/resicm_zresid.csv")

DATASETS = ["atp1d", "atp7d", "rf1", "rf2", "WUR", "SH", "rice", "wheat",
            "jura", "wq", "energy", "tecator", "scm1d", "scm20d", "edm", "sarcos"]
from legJ_jointcalib import dataset_available  # noqa: E402
DATASETS = [d for d in DATASETS if dataset_available(d)]  # skip unavailable sets
SEEDS = [0, 1, 2, 3, 4]


def residual_corrs_from_train(Xtr, Ytr, inner_cv=5):
    """Inner-CV out-of-fold residuals -> (R_raw, R_z, calibration ratios).

    R_raw matches the shipped residual_corr_from_train (same folds, same seed=1
    default, same shrinkage); R_z applies the identical shrinkage to residuals
    standardised by the inner-fold predictive sd.
    """
    n, T = Ytr.shape
    resid = np.zeros((n, T))
    sd = np.zeros((n, T))
    k = min(inner_cv, n) if n >= 4 else 2
    for itr, ite in KFold(k, shuffle=True, random_state=1).split(Xtr):
        mi, Si, _ = indep_predict(Xtr[itr], Ytr[itr], Xtr[ite])
        resid[ite] = Ytr[ite] - mi
        sd[ite] = np.sqrt(np.maximum(np.diagonal(Si, axis1=1, axis2=2), 1e-12))
    z = resid / sd
    # per-target calibration ratio: how far the residual scale is from the
    # predictive scale. Constant-in-x ratios => R_raw == R_z exactly.
    ratio = resid.std(axis=0) / sd.mean(axis=0)
    # dispersion of s_t(x) within a target, relative to its mean: if this is ~0
    # for every target the two estimators cannot disagree.
    sd_cv = sd.std(axis=0) / np.maximum(sd.mean(axis=0), 1e-12)
    return _shrink_corr(resid), _shrink_corr(z), ratio, sd_cv


def run_job(args):
    name, seed = args
    X, Y = load(name)
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    if len(Xs) > 280:
        idx = np.random.default_rng(seed).choice(len(Xs), 280, replace=False)
        Xs, Ys = Xs[idx], Ys[idx]
    n, T = Ys.shape

    P = {m: (np.zeros((n, T)), np.zeros((n, T, T)))
         for m in ("Indep", "ResICM_raw", "ResICM_z")}
    dR_max, dR_mean, ratios, sd_cvs = [], [], [], []
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xs):
        mi, Si, _ = indep_predict(Xs[tr], Ys[tr], Xs[te], random_state=seed)
        P["Indep"][0][te], P["Indep"][1][te] = mi, Si
        R_raw, R_z, ratio, sd_cv = residual_corrs_from_train(Xs[tr], Ys[tr])
        off = ~np.eye(T, dtype=bool)
        dR_max.append(np.abs(R_raw - R_z)[off].max() if T > 1 else 0.0)
        dR_mean.append(np.abs(R_raw - R_z)[off].mean() if T > 1 else 0.0)
        ratios.append(ratio); sd_cvs.append(sd_cv)
        for m, R in (("ResICM_raw", R_raw), ("ResICM_z", R_z)):
            mr, Sr = resicm_predict(Xs[tr], Ys[tr], Xs[te], R)
            P[m][0][te], P[m][1][te] = mr, Sr

    M = {m: joint_metrics(Ys, *P[m]) for m in P}
    base = M["Indep"]
    rows = []
    for m in ("Indep", "ResICM_raw", "ResICM_z"):
        rows.append(dict(
            dataset=name, seed=seed, T=T, method=m,
            nll=M[m]["joint_nll"], cal=M[m]["joint_cal_err"], r2=M[m]["mean_r2"],
            dnll=round(M[m]["joint_nll"] - base["joint_nll"], 4),
            dR_offdiag_max=round(float(np.mean(dR_max)), 4),
            dR_offdiag_mean=round(float(np.mean(dR_mean)), 4),
            calib_ratio_mean=round(float(np.mean(np.concatenate(ratios))), 4),
            sd_cv_mean=round(float(np.mean(np.concatenate(sd_cvs))), 4)))
    return rows


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.dataset, prev.seed))
        print(f"resuming: {len(done)} (dataset, seed) cells already on disk")

    # cheapest first: this run shares the machine with the rank sweep, so the
    # T=16 cells are deliberately last.
    order = ["edm", "energy", "wheat", "rice", "WUR", "SH", "jura", "tecator",
             "atp1d", "atp7d", "sarcos", "rf1", "rf2", "wq", "scm1d", "scm20d"]
    jobs = [(d, s) for d in order for s in SEEDS if (d, s) not in done]
    print(f"{len(jobs)} jobs on 2 workers", flush=True)

    import multiprocessing as mp
    with mp.Pool(2) as pool:
        for rows in pool.imap_unordered(run_job, jobs):
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT),
                                      index=False)
            r = rows[0]
            raw = [x for x in rows if x["method"] == "ResICM_raw"][0]
            z = [x for x in rows if x["method"] == "ResICM_z"][0]
            print(f"[done] {r['dataset']:8s} seed={r['seed']}  raw={raw['dnll']:+.3f} "
                  f"z={z['dnll']:+.3f}  |dR|off_max={raw['dR_offdiag_max']:.4f}", flush=True)
    summarize()


def summarize():
    df = pd.read_csv(OUT)
    piv = df.pivot_table(index="dataset", columns="method", values="dnll", aggfunc="mean")
    piv = piv[["ResICM_raw", "ResICM_z"]]
    piv["diff"] = piv.ResICM_z - piv.ResICM_raw
    aux = df.groupby("dataset")[["dR_offdiag_max", "dR_offdiag_mean",
                                 "calib_ratio_mean", "sd_cv_mean"]].mean()
    m = piv.join(aux)
    print("\n=== 5-seed mean dNLL: shipped (raw) vs matched (z-score) R_res ===")
    print(m.round(4).to_string())
    print("\nmean over datasets:")
    print(f"  ResICM_raw {m.ResICM_raw.mean():+.3f}   ResICM_z {m.ResICM_z.mean():+.3f}"
          f"   mean |diff| {m['diff'].abs().mean():.4f}   max |diff| {m['diff'].abs().max():.4f}")
    print(f"  off-diagonal |R_raw - R_z|: mean {m.dR_offdiag_mean.mean():.4f}, "
          f"worst dataset max {m.dR_offdiag_max.max():.4f}")
    print(f"  predictive-sd dispersion within target (sd_cv): mean {m.sd_cv_mean.mean():.4f}, "
          f"max {m.sd_cv_mean.max():.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
    else:
        main()
