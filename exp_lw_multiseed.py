"""
Multi-seed check of Ledoit-Wolf shrinkage under the standardised diagnostic.

Appendix E claims shrinkage "helps where the plain diagnostic saturates and has
little effect elsewhere". Under R_z the seed-0 tabular family instead DEGRADES
(-0.81 -> -0.60), driven by rf1, whose shrunk D_logdet collapses (1.21 -> 0.26)
despite a large ICM gain. A claim reversal on one seed is not trustworthy, so
this recomputes plain and shrunk D_logdet on standardised residuals over 5 seeds.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_dlogdet_standardised import DATASETS, SEEDS, dlogdet, psd_corr
from exp_hardening_ablation import load
from legJ_jointcalib import indep_predict

OUT = "results/legJ/lw_multiseed.csv"


def dlogdet_lw(resid):
    cov = LedoitWolf().fit(resid).covariance_
    d = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    R = cov / np.outer(d, d)
    return dlogdet(R)


def run(args):
    name, seed = args
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
    r = Ys - mu
    sd = np.sqrt(np.maximum(np.diagonal(Sig, axis1=1, axis2=2), 1e-12))
    z = r / sd
    return dict(dataset=name, seed=seed, T=T,
                D_z=dlogdet(psd_corr(z)), LW_z=dlogdet_lw(z),
                D_raw=dlogdet(psd_corr(r)), LW_raw=dlogdet_lw(r))


def main():
    done = set()
    if os.path.exists(OUT):
        done = set(map(tuple, pd.read_csv(OUT)[["dataset", "seed"]].values))
    jobs = [(d, s) for d in DATASETS for s in SEEDS if (d, s) not in done]
    print(f"{len(jobs)} jobs", flush=True)
    import multiprocessing as mp
    with mp.Pool(10) as pool:
        for r in pool.imap_unordered(run, jobs):
            pd.DataFrame([r]).to_csv(OUT, mode="a", header=not os.path.exists(OUT),
                                     index=False)
            print(f"[done] {r['dataset']:8s} seed={r['seed']}", flush=True)
    summarize()


def summarize():
    from scipy import stats
    df = pd.read_csv(OUT)
    icm = pd.read_csv("results/legJ/residual_icm_multiseed_long.csv")
    icm = icm[icm.method == "ICM"][["dataset", "seed", "dnll"]]
    d = df.merge(icm, on=["dataset", "seed"])
    print("\n=== rho_s(D, dNLL) per seed: plain vs Ledoit-Wolf, standardised ===")
    rows = []
    for s, gg in d.groupby("seed"):
        rows.append((s, stats.spearmanr(gg.D_z, gg.dnll).statistic,
                     stats.spearmanr(gg.LW_z, gg.dnll).statistic,
                     stats.spearmanr(gg.D_raw, gg.dnll).statistic,
                     stats.spearmanr(gg.LW_raw, gg.dnll).statistic))
        print("  seed %d   D_z %+.3f  LW_z %+.3f   |  D_raw %+.3f  LW_raw %+.3f" % rows[-1])
    a = np.array([r[1:] for r in rows])
    print("  mean     D_z %+.3f  LW_z %+.3f   |  D_raw %+.3f  LW_raw %+.3f" % tuple(a.mean(0)))
    g = d.groupby("dataset")[["D_z", "LW_z"]].mean()
    print("\nlargest shrinkage effect (D_z -> LW_z):")
    print((g.LW_z - g.D_z).sort_values().head(4).round(3).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
    else:
        main()
