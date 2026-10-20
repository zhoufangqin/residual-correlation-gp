"""
Leg 2 -- Calibration / UQ of the GP cross-target readout.

For each dataset and direction (encoder trained on A, recover B), collect
out-of-fold predictive (mean, std) from three uncertainty models on the SAME
frozen Phi, and ask: which gives *calibrated* cross-target uncertainty?

  - GP      : RBF GaussianProcess posterior (the proposed instrument)
  - BLR     : BayesianRidge -- linear-evidence predictive UQ (LogME's model class)
  - Ensemble: 5 MLP probes, predictive mean/std across members + residual var

Metrics (predictive N(mu, sigma) vs true y):
  - NLL          : mean negative log predictive density (lower=better)
  - cal_err      : mean |empirical - nominal| of the PIT CDF (regression ECE)
  - cov90/cov95  : central-interval coverage (target 0.90 / 0.95)
  - sharp        : mean predictive sigma (lower=sharper, only meaningful if calibrated)

Claim under test: the GP is better-calibrated (low cal_err, cov near nominal,
good NLL) than linear-evidence UQ -- the axis that carries the paper now that
the nonlinear *recovery* gap is small (see gp-iclr-derisk-results).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gateB_recovery import _encode, load_plants, load_public  # reuse


def _gp():
    k = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e2)) \
        + WhiteKernel(0.1, (1e-4, 1e1))
    return GaussianProcessRegressor(kernel=k, normalize_y=True,
                                    n_restarts_optimizer=2, alpha=1e-6)


def _predictors(Phi_tr, Phi_te, b_tr):
    """Return {method: (mu, sigma)} on the test split."""
    sc = StandardScaler().fit(Phi_tr)
    Ztr, Zte = sc.transform(Phi_tr), sc.transform(Phi_te)
    out = {}

    gp = _gp().fit(Ztr, b_tr)
    mu, sd = gp.predict(Zte, return_std=True)
    out["GP"] = (mu, np.maximum(sd, 1e-6))

    blr = BayesianRidge().fit(Ztr, b_tr)
    mu, sd = blr.predict(Zte, return_std=True)
    out["BLR"] = (mu, np.maximum(sd, 1e-6))

    preds, resid = [], []
    for s in range(5):
        m = MLPRegressor(hidden_layer_sizes=(64,), max_iter=1500, random_state=s, alpha=1e-2,
                         early_stopping=True).fit(Ztr, b_tr)
        preds.append(m.predict(Zte))
        resid.append(np.std(b_tr - m.predict(Ztr)))
    preds = np.array(preds)
    mu = preds.mean(0)
    sd = np.sqrt(preds.var(0) + np.mean(resid) ** 2)   # epistemic + aleatoric
    out["Ensemble"] = (mu, np.maximum(sd, 1e-6))
    return out


def collect_oof(X, yA, yB, cv, seed=0):
    n = len(X)
    res = {m: (np.zeros(n), np.zeros(n)) for m in ("GP", "BLR", "Ensemble")}
    splitter = LeaveOneOut() if cv == "loo" else KFold(cv, shuffle=True, random_state=seed)
    for tr, te in splitter.split(X):
        xs = StandardScaler().fit(X[tr])
        Ptr, Pte = _encode(xs.transform(X[tr]), yA[tr], xs.transform(X[te]), seed=seed)
        for m, (mu, sd) in _predictors(Ptr, Pte, yB[tr]).items():
            res[m][0][te], res[m][1][te] = mu, sd
    return res


def metrics(y, mu, sd):
    nll = float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * (y - mu) ** 2 / sd ** 2))
    z = (y - mu) / sd
    cov90 = float(np.mean(np.abs(z) <= 1.6448536269514722))
    cov95 = float(np.mean(np.abs(z) <= 1.959963984540054))
    pit = norm.cdf(z)
    grid = np.linspace(0.05, 0.95, 19)
    cal_err = float(np.mean([abs(np.mean(pit <= p) - p) for p in grid]))
    return dict(nll=nll, cal_err=cal_err, cov90=cov90, cov95=cov95, sharp=float(np.mean(sd)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="PHO,Tecator,Energy")
    ap.add_argument("--out", default="results/legC/calibration.csv")
    args = ap.parse_args()

    want = args.datasets.split(",")
    plants = load_plants() if any(d in want for d in ("PHO", "SH", "WUR")) else {}
    public = load_public() if any(d in want for d in ("Tecator", "Energy")) else {}

    rows = []
    for name in want:
        if name in plants:
            X, vc, jx, cv = plants[name]
            pairs = [("Vcmax", vc, "Jmax", jx), ("Jmax", jx, "Vcmax", vc)]
        elif name in public:
            X, pairs, cv = public[name]
        else:
            continue
        print(f"[{name}] n={len(X)} cv={cv} ...", flush=True)
        for Aname, yA, Bname, yB in pairs:
            res = collect_oof(X, yA, yB, cv)
            for m, (mu, sd) in res.items():
                mt = metrics(yB, mu, sd)
                rows.append(dict(dataset=name, dir=f"{Aname}->{Bname}", method=m, **mt))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 160, "display.max_rows", 200)
    print("\n=== Leg 2: cross-target predictive calibration ===")
    print(df.round(3).to_string(index=False))
    print("\nGP-vs-others summary (mean over datasets/dirs):")
    print(df.groupby("method")[["nll", "cal_err", "cov90", "cov95", "sharp"]]
          .mean().round(3).to_string())
    print("target: cov90~0.90, cov95~0.95, low cal_err, low NLL")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
