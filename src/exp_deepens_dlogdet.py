"""
D_logdet computed entirely inside the neural experiment (Appendix C.3).

exp_deepens_coupling.py correlates the neural coupling gain against the
GP-derived D_logdet, so the claim "not specific to the GP implementation" rests
on a predictor that is still built from GP fits. This recomputes D_logdet from
the DIAGONAL neural ensemble's own out-of-fold predictive-standardised
residuals, so predictor and outcome are both neural, and reports both versions.

Protocol identical to exp_deepens_coupling.py: same datasets, subsample,
5-fold CV, seeds {0,1,2}, same architecture and training schedule.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_deepens_coupling import DATASETS, SEEDS, fit_predict
from exp_hardening_ablation import load

OUT = "results/deepens/neural_dlogdet.csv"


def psd_corr(z):
    R = np.atleast_2d(np.corrcoef(z.T))
    R = 0.5 * (R + R.T)
    w, V = np.linalg.eigh(R)
    R = (V * np.clip(w, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def run(name, seed):
    X, Y = load(name)
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    if len(Xs) > 280:
        idx = np.random.default_rng(seed).choice(len(Xs), 280, replace=False)
        Xs, Ys = Xs[idx], Ys[idx]
    n, T = Ys.shape
    mu = np.zeros((n, T)); sd = np.zeros((n, T))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xs):
        m, cov, _ = fit_predict(Xs[tr], Ys[tr], Xs[te], False, seed)   # diagonal arm
        mu[te] = m
        sd[te] = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 1e-12))
    z = (Ys - mu) / sd                      # out-of-fold predictive z-scores
    R = psd_corr(z)
    return dict(dataset=name, seed=seed, T=T,
                D_neural=float(-0.5 * np.log(max(np.linalg.det(R), 1e-12))))


def main():
    os.makedirs("results/deepens", exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        done = set(map(tuple, pd.read_csv(OUT)[["dataset", "seed"]].values))
    for d in DATASETS:
        for s in SEEDS:
            if (d, s) in done:
                continue
            r = run(d, s)
            pd.DataFrame([r]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"[done] {d:8s} seed={s} D_neural={r['D_neural']:.3f}", flush=True)
    summarize()


def summarize():
    from scipy import stats
    nd = pd.read_csv(OUT).groupby("dataset").D_neural.mean()
    de = pd.read_csv("results/deepens/deepens_coupling.csv").groupby("dataset").mean(numeric_only=True)
    gp = pd.read_csv("results/legJ/rz_sweep_inputs.csv", index_col=0).D_z
    g = pd.DataFrame({"D_neural": nd, "D_gp": gp.reindex(nd.index),
                      "dnll_resid": de.dnll_resid, "dnll_full": de.dnll_full}).dropna()
    print("\n=== D_logdet: GP-derived vs neural-derived ===")
    print(g.round(3).to_string())
    print(f"\nSpearman(D_gp, D_neural) = {stats.spearmanr(g.D_gp, g.D_neural).statistic:+.3f}")
    for pred in ("D_gp", "D_neural"):
        for out in ("dnll_resid", "dnll_full"):
            r = stats.spearmanr(g[pred], g[out])
            print(f"  Spearman({pred:9s}, {out:11s}) = {r.statistic:+.3f}  (p={r.pvalue:.4f})")


if __name__ == "__main__":
    summarize() if len(sys.argv) > 1 else main()
