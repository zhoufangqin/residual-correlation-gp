"""
Gate B -- cross-target recovery on REAL data (the make-or-break test).

Protocol (mirrors E0 on real spectra, leakage-free):
  Outer CV over samples. In each fold:
    1. train an encoder (small MLP) on the TRAIN split using target A ONLY
    2. freeze it -> Phi = tanh(X W1 + b1) for train and test
    3. fit a recovery head on (Phi_train, B_train); predict B_test
  Aggregate out-of-fold predictions -> R^2 for recovering target B.

Heads: linear (RidgeCV), mlp (MLPRegressor), gp (RBF GaussianProcess).
Key quantity: gp_minus_linear = nonlinear cross-target structure the linear
probe misses. With Vcmax/Jmax correlated at ~0.9, a large share of recovery is
the trivial linear channel; the question is whether the GP adds anything beyond.

Reports both directions (Vcmax->Jmax and Jmax->Vcmax) per dataset, with
bootstrap CIs on the gap.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# The hyperspectral plant datasets (SH, WUR, PHO) are not redistributed. Point
# CALIB_PAPER_DIR at the companion code that provides data_utils.load_{sh,wur,pho}.
CALIB = os.environ.get("CALIB_PAPER_DIR", "")
if CALIB:
    sys.path.insert(0, CALIB)


# --------------------------------------------------------------------------- #
def _encode(Xtr, ytr, Xte, H=64, seed=0):
    """Train MLP on (Xtr, ytr=target A); return frozen hidden activations for
    train and test (Phi = tanh(X W1 + b1))."""
    net = MLPRegressor(hidden_layer_sizes=(H,), activation="tanh",
                       max_iter=1000, random_state=seed, alpha=1e-2)
    net.fit(Xtr, ytr)
    W1, b1 = net.coefs_[0], net.intercepts_[0]
    return np.tanh(Xtr @ W1 + b1), np.tanh(Xte @ W1 + b1)


def _gp():
    k = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e2)) \
        + WhiteKernel(0.1, (1e-4, 1e1))
    return GaussianProcessRegressor(kernel=k, normalize_y=True,
                                    n_restarts_optimizer=2, alpha=1e-6)


def _heads(Phi_tr, Phi_te, b_tr):
    sc = StandardScaler().fit(Phi_tr)
    Ztr, Zte = sc.transform(Phi_tr), sc.transform(Phi_te)
    lin = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Ztr, b_tr).predict(Zte)
    mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=1500, random_state=0, alpha=1e-2,
                       early_stopping=True).fit(Ztr, b_tr).predict(Zte)
    gp = _gp().fit(Ztr, b_tr).predict(Zte)
    return lin, mlp, gp


def cross_target(X, yA, yB, cv, seed=0):
    """Out-of-fold predictions of yB from a yA-trained encoder."""
    n = len(X)
    P = {k: np.zeros(n) for k in ("lin", "mlp", "gp")}
    splitter = LeaveOneOut() if cv == "loo" else KFold(cv, shuffle=True, random_state=seed)
    for tr, te in splitter.split(X):
        xs = StandardScaler().fit(X[tr])
        Xtr, Xte = xs.transform(X[tr]), xs.transform(X[te])
        Ptr, Pte = _encode(Xtr, yA[tr], Xte, seed=seed)
        lin, mlp, gp = _heads(Ptr, Pte, yB[tr])
        P["lin"][te], P["mlp"][te], P["gp"][te] = lin, mlp, gp
    return P


def boot_gap_ci(yB, p_gp, p_lin, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(yB)
    g = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        g.append(r2_score(yB[b], p_gp[b]) - r2_score(yB[b], p_lin[b]))
    return float(np.quantile(g, 0.025)), float(np.quantile(g, 0.975))


def run_pairs(name, X, pairs, cv, seeds=(0, 1, 2)):
    """pairs: list of (Aname, yA, Bname, yB)."""
    rows = []
    for Aname, yA, Bname, yB in pairs:
        accP = {k: [] for k in ("lin", "mlp", "gp")}
        for s in seeds:
            P = cross_target(X, yA, yB, cv, seed=s)
            for k in P:
                accP[k].append(P[k])
        # average OOF predictions across seeds, then score once
        p = {k: np.mean(accP[k], 0) for k in accP}
        r2 = {k: r2_score(yB, p[k]) for k in p}
        lo, hi = boot_gap_ci(yB, p["gp"], p["lin"])
        corr = float(np.corrcoef(yA, yB)[0, 1])
        rows.append(dict(dataset=name, encoder_target=Aname, recover_target=Bname,
                         n=len(X), corr_AB=corr,
                         r2_linear=r2["lin"], r2_mlp=r2["mlp"], r2_gp=r2["gp"],
                         gp_minus_linear=r2["gp"] - r2["lin"],
                         gap_ci_lo=lo, gap_ci_hi=hi))
    return rows


def load_public():
    """Returns {name: (X, pairs, cv)} for cached non-plant multi-target sets."""
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    tec = np.load(os.path.join(here, "data_public/tecator.npz"), allow_pickle=True)
    Xt, Yt = tec["X"], tec["Y"]            # moisture, fat, protein
    out["Tecator"] = (Xt, [("fat", Yt[:, 1], "protein", Yt[:, 2]),
                           ("protein", Yt[:, 2], "fat", Yt[:, 1])], 5)
    en = np.load(os.path.join(here, "data_public/energy.npz"), allow_pickle=True)
    Xe, Ye = en["X"], en["Y"]              # heating, cooling
    out["Energy"] = (Xe, [("heating", Ye[:, 0], "cooling", Ye[:, 1]),
                          ("cooling", Ye[:, 1], "heating", Ye[:, 0])], 5)
    return out


def load_plants():
    try:
        import data_utils as du
    except ImportError:
        print("[skip] plant datasets unavailable (set CALIB_PAPER_DIR)", flush=True)
        return {}
    out = {}
    for name, loader, cv in [("PHO", du.load_pho, 5),
                             ("SH", lambda: du.load_sh("cal_mean"), "loo"),
                             ("WUR", du.load_wur, "loo")]:
        d = loader()
        X = np.array([v["mean_spec"] for v in d.values()], float)
        vc = np.array([v["vcmax"] for v in d.values()], float)
        jx = np.array([v["jmax"] for v in d.values()], float)
        ok = np.all(np.isfinite(X), 1) & np.isfinite(vc) & np.isfinite(jx)
        out[name] = (X[ok], vc[ok], jx[ok], cv)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="PHO",
                    help="comma list: PHO,SH,WUR,Tecator,Energy")
    ap.add_argument("--out", default="results/gateB/gateB_plants.csv")
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
            print(f"unknown dataset {name}; skipping"); continue
        print(f"[{name}] n={len(X)} feat={X.shape[1]} cv={cv} ...", flush=True)
        rows += run_pairs(name, X, pairs, cv)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 160)
    print("\n=== Gate B: cross-target recovery (real plant HSI) ===")
    print(df.round(3).to_string(index=False))
    print("\nInterpretation: r2_gp ~ recovery level; gp_minus_linear (+CI) = "
          "nonlinear structure beyond the linear correlation channel.")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
