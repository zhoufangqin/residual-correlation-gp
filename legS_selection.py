"""
Leg 3 -- Selection / transferability of frozen representations.

Setup: a small "representation zoo" of frozen feature extractors per dataset.
We want to PICK the one that best recovers target B, using only a cheap
TRAINING-TIME score (no expensive cross-validation). Question: does GP evidence
(log marginal likelihood, nonlinear kernel) rank the zoo better than LogME
(linear evidence) or a label-blind capacity proxy?

  cheap scores (computed on full Phi, no CV):
    - GP_LML   : RBF GaussianProcess log marginal likelihood / n
    - LogME    : linear-Gaussian evidence / n        (You et al. 2021)
    - capacity : -dim(Phi)                            (label-blind proxy)
  ground truth (expensive, what we actually want):
    - CV recovery R^2 of target B (GP head)

Metric: Kendall-tau between each cheap score and the ground-truth ranking, across
the zoo. Claim: GP_LML >= LogME > capacity, because the GP evidence is label- and
nonlinearity-aware. This is the 'select' readout of the unified instrument.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gateB_recovery import load_plants, load_public
from e0_synthetic import logme


def _gp():
    # n_restarts=0: single hyperparameter optimization from init -- enough for
    # evidence/ranking, ~3x faster than the recovery-leg GP (this leg fits many).
    k = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e2)) \
        + WhiteKernel(0.1, (1e-4, 1e1))
    return GaussianProcessRegressor(kernel=k, normalize_y=True,
                                    n_restarts_optimizer=0, alpha=1e-6)


def _mlp_feats(X, y, H, seed=0):
    net = MLPRegressor(hidden_layer_sizes=(H,), activation="tanh", max_iter=1000,
                       random_state=seed, alpha=1e-2).fit(X, y)
    return np.tanh(X @ net.coefs_[0] + net.intercepts_[0])


def make_zoo(X, yA):
    """Frozen representation zoo (mix of unsupervised and A-supervised)."""
    Xs = StandardScaler().fit_transform(X)
    zoo = {"raw": Xs}
    for k in (2, 5, 10, 20):
        if k < min(X.shape):
            zoo[f"pca{k}"] = PCA(k, random_state=0).fit_transform(Xs)
    zoo["randMLP32"] = np.tanh(Xs @ np.random.default_rng(0).standard_normal((Xs.shape[1], 32)) / np.sqrt(Xs.shape[1]))
    for H in (16, 32, 64, 128):
        zoo[f"encA{H}"] = _mlp_feats(Xs, yA, H)
    return zoo


def gp_lml(Phi, y):
    Z = StandardScaler().fit_transform(Phi)
    return float(_gp().fit(Z, y).log_marginal_likelihood_value_ / len(y))


def cv_recovery(Phi, y, cv=5, seed=0):
    Z = StandardScaler().fit_transform(Phi)
    pred = np.zeros(len(y))
    for tr, te in KFold(cv, shuffle=True, random_state=seed).split(Z):
        g = _gp().fit(Z[tr], y[tr])
        pred[te] = g.predict(Z[te])
    return r2_score(y, pred)


def run_pair(name, X, Aname, yA, Bname, yB):
    zoo = make_zoo(X, yA)
    recs = []
    for zname, Phi in zoo.items():
        recs.append(dict(
            dataset=name, dir=f"{Aname}->{Bname}", repr=zname, dim=Phi.shape[1],
            gp_lml=gp_lml(Phi, yB),
            logme=logme(Phi, yB),
            capacity=-Phi.shape[1],
            gt_recovery=cv_recovery(Phi, yB),
        ))
    d = pd.DataFrame(recs)
    taus = {s: kendalltau(d[s], d.gt_recovery).correlation
            for s in ("gp_lml", "logme", "capacity")}
    return d, dict(dataset=name, dir=f"{Aname}->{Bname}",
                   tau_gp_lml=taus["gp_lml"], tau_logme=taus["logme"],
                   tau_capacity=taus["capacity"],
                   best_repr=d.loc[d.gt_recovery.idxmax(), "repr"],
                   picked_gp=d.loc[d.gp_lml.idxmax(), "repr"],
                   picked_logme=d.loc[d.logme.idxmax(), "repr"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="PHO,Tecator,Energy")
    ap.add_argument("--out", default="results/legS/selection.csv")
    args = ap.parse_args()

    want = args.datasets.split(",")
    plants = load_plants() if any(d in want for d in ("PHO", "SH", "WUR")) else {}
    public = load_public() if any(d in want for d in ("Tecator", "Energy")) else {}

    detail, summary = [], []
    for name in want:
        if name in plants:
            X, vc, jx, _ = plants[name]
            pairs = [("Vcmax", vc, "Jmax", jx), ("Jmax", jx, "Vcmax", vc)]
        elif name in public:
            X, pairs2, _ = public[name]
            pairs = pairs2
        else:
            continue
        print(f"[{name}] building zoo + scoring ...", flush=True)
        for Aname, yA, Bname, yB in pairs:
            d, s = run_pair(name, X, Aname, yA, Bname, yB)
            detail.append(d); summary.append(s)

    det = pd.concat(detail, ignore_index=True)
    summ = pd.DataFrame(summary)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    det.to_csv(args.out, index=False)
    summ.to_csv(args.out.replace(".csv", "_summary.csv"), index=False)
    pd.set_option("display.width", 170, "display.max_rows", 300)
    print("\n=== Leg 3: representation selection (Kendall-tau vs true recovery) ===")
    print(summ.round(3).to_string(index=False))
    print("\nMean tau across datasets/dirs:")
    print(summ[["tau_gp_lml", "tau_logme", "tau_capacity"]].mean().round(3).to_string())
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
