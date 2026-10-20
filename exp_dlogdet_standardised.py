"""
Is D_logdet consistent with the corrected Residual-ICM estimator?

After adopting R_res = corr(r/s) in Residual-ICM, the
DIAGNOSTIC still computes D_logdet from the RAW residual correlation
(exp_residual_diagnostic.py:106). Under Sigma(x) = D_s(x) R D_s(x) the
correlation matched to the Gaussian log score is that of the standardised errors
z = D_s^{-1} r, so the readout was corrected but the diagnostic was not.

This recomputes both variants on the 16 datasets x 5 seeds under the identical
protocol and compares their association with the measured ICM gain, so the
choice between "standardise everywhere" and "keep both with distinct notation"
is made on evidence.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_hardening_ablation import load
from legJ_jointcalib import indep_predict, joint_metrics

DATASETS = ["atp1d", "atp7d", "rf1", "rf2", "WUR", "SH", "rice", "wheat",
            "jura", "wq", "energy", "tecator", "scm1d", "scm20d", "edm", "sarcos"]
from legJ_jointcalib import dataset_available  # noqa: E402
DATASETS = [d for d in DATASETS if dataset_available(d)]  # skip unavailable sets
SEEDS = [0, 1, 2, 3, 4]
OUT = "results/legJ/dlogdet_standardised.csv"


def psd_corr(resid):
    R = np.atleast_2d(np.corrcoef(resid.T))
    R = 0.5 * (R + R.T)
    w, V = np.linalg.eigh(R)
    R = (V * np.clip(w, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def dlogdet(R):
    return float(-0.5 * np.log(max(np.linalg.det(R), 1e-12)))


def run(name, seed):
    X, Y = load(name)
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    if len(Xs) > 280:
        idx = np.random.default_rng(seed).choice(len(Xs), 280, replace=False)
        Xs, Ys = Xs[idx], Ys[idx]
    n, T = Ys.shape
    mu = np.zeros((n, T)); Sig = np.zeros((n, T, T))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xs):
        mi, Si, _ = indep_predict(Xs[tr], Ys[tr], Xs[te], random_state=seed)
        mu[te], Sig[te] = mi, Si
    resid = Ys - mu
    sd = np.sqrt(np.maximum(np.diagonal(Sig, axis1=1, axis2=2), 1e-12))
    R_raw, R_z = psd_corr(resid), psd_corr(resid / sd)
    off = ~np.eye(T, dtype=bool)
    return dict(dataset=name, seed=seed, T=T,
                D_raw=dlogdet(R_raw), D_z=dlogdet(R_z),
                dR_offdiag_mean=float(np.abs(R_raw - R_z)[off].mean()) if T > 1 else 0.0,
                dR_offdiag_max=float(np.abs(R_raw - R_z)[off].max()) if T > 1 else 0.0,
                r2_indep=joint_metrics(Ys, mu, Sig)["mean_r2"])


def main():
    os.makedirs("results/legJ", exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        done = set(map(tuple, pd.read_csv(OUT)[["dataset", "seed"]].values))
    jobs = [(d, s) for d in DATASETS for s in SEEDS if (d, s) not in done]
    print(f"{len(jobs)} jobs", flush=True)
    import multiprocessing as mp
    with mp.Pool(10) as pool:
        for r in pool.imap_unordered(_star, jobs):
            pd.DataFrame([r]).to_csv(OUT, mode="a", header=not os.path.exists(OUT),
                                     index=False)
            print(f"[done] {r['dataset']:8s} seed={r['seed']} "
                  f"D_raw={r['D_raw']:.3f} D_z={r['D_z']:.3f}", flush=True)
    summarize()


def _star(a):
    return run(*a)


def summarize():
    from scipy import stats
    df = pd.read_csv(OUT).groupby("dataset")[
        ["D_raw", "D_z", "dR_offdiag_mean", "dR_offdiag_max", "T"]].mean()
    pub = pd.read_csv("results/legJ/residual_icm_multiseed_summary.csv").set_index("dataset")
    g = df.join(pub[["ICM_dnll_mean", "r2_indep_mean"]])
    print("\n=== D_logdet: raw vs standardised residuals (5-seed means) ===")
    print(g.round(3).to_string())
    print("\nAssociation with the measured ICM gain:")
    for c in ("D_raw", "D_z"):
        r = stats.spearmanr(g[c], g.ICM_dnll_mean)
        pe = np.corrcoef(-g[c], g.ICM_dnll_mean)[0, 1]
        print(f"  {c:6s} Spearman {r.statistic:+.3f} (p={r.pvalue:.2e})   "
              f"Pearson(-D, dNLL) {pe:+.3f}")
    # partial correlation controlling for T, p/n, R2 (as in the paper)
    diag = pd.read_csv("results/legJ/residual_diagnostic_multiseed_summary.csv")
    diag = diag[~diag.dataset.isin(["oes97", "oes10"])].set_index("dataset")
    ctrl = pd.DataFrame({"T": diag["T"], "pn": diag.p_over_n,
                         "r2": diag.r2_indep}).loc[g.index]

    def partial(y, x):
        rk = lambda v: stats.rankdata(v)
        C = np.column_stack([rk(ctrl[c]) for c in ctrl] + [np.ones(len(g))])
        ry = rk(y) - C @ np.linalg.lstsq(C, rk(y), rcond=None)[0]
        rx = rk(x) - C @ np.linalg.lstsq(C, rk(x), rcond=None)[0]
        return stats.pearsonr(rx, ry)

    print("\nPartial (controlling T, p/n, R2_Indep):")
    for c in ("D_raw", "D_z"):
        r, p = partial(g.ICM_dnll_mean, g[c])
        print(f"  {c:6s} {r:+.3f} (p={p:.4f})")
    g.round(4).to_csv("results/legJ/dlogdet_standardised_summary.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
    else:
        main()
