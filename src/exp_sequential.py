"""
C1 verification: does ΔNLL (ICM − Indep) grow as T' (number of coupled targets) increases?

Protocol (Approach B — pre-compute Indep once, sweep ICM):
  1. Fit all T_max per-target independent GPs once (cache per fold).
  2. For T' in {2..T_max}: draw N_DRAW random subsets of T' targets.
     For each subset: read Indep metrics from cache; fit ICM; compute metrics.
  3. Plot ΔNLL ± std vs T' for atp1d / rf1 / wq (three p/n slices).

Outputs: results/sequential/dnll_vs_T.{csv,pdf,png}
"""
import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import chi2
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legJ_jointcalib import fit_icm, icm_predict, joint_metrics

HERE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
RDIR  = os.path.join(HERE, "results", "sequential")
os.makedirs(RDIR, exist_ok=True)

N_DRAW  = 20
CV      = 5
N_SUB   = 280
RNG     = np.random.default_rng(42)

P_DIM   = dict(atp1d=411, rf1=64, rf2=576, wq=16)
DATASETS = ["atp1d", "rf1", "rf2", "wq"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _rbf_k(A, B, ell):
    return np.exp(-cdist(A, B, "sqeuclidean") / (2 * ell**2))


def load_dataset(name):
    p = os.path.join(HERE, f"data_public/{name.lower()}.npz")
    z = np.load(p, allow_pickle=True)
    return z["X"], z["Y"]


def precompute_indep(Xs, Ys, cv=CV):
    """Fit one GP per target for every fold. Return caches indexed [fold][target]."""
    n, T = Ys.shape
    folds = list(KFold(cv, shuffle=True, random_state=0).split(Xs))
    mu_cache   = [[None]*T for _ in range(cv)]
    sig2_cache = [[None]*T for _ in range(cv)]
    ell_cache  = [[None]*T for _ in range(cv)]

    for fi, (tr, te) in enumerate(folds):
        print(f"    fold {fi+1}/{cv}: fitting {T} GPs ...", flush=True)
        for t in range(T):
            k  = ConstantKernel(1.0, (1e-2,1e2)) * RBF(1.0, (1e-1,1e2)) \
               + WhiteKernel(0.1, (1e-4,1e1))
            gp = GaussianProcessRegressor(k, normalize_y=True,
                                          n_restarts_optimizer=1, alpha=1e-6)
            gp.fit(Xs[tr], Ys[tr, t])
            m, s = gp.predict(Xs[te], return_std=True)
            mu_cache[fi][t]   = m
            sig2_cache[fi][t] = np.maximum(s, 1e-6)**2
            ell_cache[fi][t]  = gp.kernel_.k1.k2.length_scale
    return folds, mu_cache, sig2_cache, ell_cache


def indep_metrics_from_cache(Ys_te, subset, fi, mu_cache, sig2_cache):
    """Assemble diagonal joint predictive from cached per-target predictions."""
    T_sub = len(subset)
    nte   = Ys_te.shape[0]
    mu  = np.stack([mu_cache[fi][t]   for t in subset], axis=1)   # nte x T_sub
    Sig = np.zeros((nte, T_sub, T_sub))
    for j, t in enumerate(subset):
        Sig[:, j, j] = sig2_cache[fi][t]
    return joint_metrics(Ys_te, mu, Sig)


def icm_metrics_for_subset(Xs, Ys, folds, ell_cache, subset, fi):
    """Fit ICM for one fold and one target subset, return joint metrics."""
    tr, te = folds[fi]
    shared_ell = float(np.median([ell_cache[fi][t] for t in subset]))
    B, sig2 = fit_icm(Xs[tr], Ys[np.ix_(tr, subset)], shared_ell)
    mu_c, Sig_c = icm_predict(Xs[tr], Ys[np.ix_(tr, subset)],
                               Xs[te], shared_ell, B, sig2)
    return joint_metrics(Ys[np.ix_(te, subset)], mu_c, Sig_c)


# ── main sweep ───────────────────────────────────────────────────────────────

def run_sequential(name):
    print(f"\n=== {name} ===", flush=True)
    X, Y = load_dataset(name)
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    if len(Xs) > N_SUB:
        idx = np.random.default_rng(0).choice(len(Xs), N_SUB, replace=False)
        Xs, Ys = Xs[idx], Ys[idx]
    n, T_max = Ys.shape
    pn = P_DIM[name] / n
    print(f"  n={n} T_max={T_max} p/n={pn:.2f}", flush=True)

    print("  Pre-computing Indep GPs ...", flush=True)
    folds, mu_cache, sig2_cache, ell_cache = precompute_indep(Xs, Ys)

    rows = []
    all_t = list(range(T_max))

    for T_prime in range(2, T_max + 1):
        print(f"  T'={T_prime} ...", flush=True)
        d_nll_draws, d_r2_draws = [], []

        for draw in range(N_DRAW):
            subset = list(RNG.choice(all_t, T_prime, replace=False))
            nll_i, r2_i, nll_c, r2_c = [], [], [], []
            for fi in range(CV):
                te = folds[fi][1]
                mi = indep_metrics_from_cache(Ys[te][:, subset], subset,
                                              fi, mu_cache, sig2_cache)
                mc = icm_metrics_for_subset(Xs, Ys, folds, ell_cache, subset, fi)
                nll_i.append(mi["joint_nll"]);  r2_i.append(mi["mean_r2"])
                nll_c.append(mc["joint_nll"]);  r2_c.append(mc["mean_r2"])

            d_nll_draws.append(np.mean(nll_c) - np.mean(nll_i))
            d_r2_draws.append(np.mean(r2_c)   - np.mean(r2_i))

        rows.append(dict(
            dataset=name, pn=round(pn,3), T_prime=T_prime,
            mean_dNLL=float(np.mean(d_nll_draws)),
            std_dNLL =float(np.std(d_nll_draws)),
            mean_dR2 =float(np.mean(d_r2_draws)),
            std_dR2  =float(np.std(d_r2_draws)),
        ))
        print(f"    ΔNLL={rows[-1]['mean_dNLL']:+.3f} ± {rows[-1]['std_dNLL']:.3f}  "
              f"ΔR²={rows[-1]['mean_dR2']:+.3f}", flush=True)

    return rows


# ── plot ─────────────────────────────────────────────────────────────────────

COLORS = {"atp1d": "#2166ac", "rf1": "#4dac26", "rf2": "#e6820e", "wq": "#d6604d"}
LABELS = {"atp1d": "atp1d  (p/n=1.47, T=6)",
          "rf1":   "rf1    (p/n=0.23, T=8)",
          "rf2":   "rf2    (p/n=2.06, T=8)",
          "wq":    "wq     (p/n=0.06, T=14)"}


def plot_results(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name in DATASETS:
        sub = df[df.dataset == name].sort_values("T_prime")
        x   = sub["T_prime"].values
        y   = sub["mean_dNLL"].values
        e   = sub["std_dNLL"].values
        c   = COLORS[name]
        ax.plot(x, y, "-o", color=c, lw=1.8, ms=5, label=LABELS[name])
        ax.fill_between(x, y-e, y+e, color=c, alpha=0.15)

    ax.axhline(0, ls="--", color="0.4", lw=0.9)
    ax.set_xlabel("$T'$  (number of coupled targets)", fontsize=13)
    ax.set_ylabel(r"$\Delta$NLL  (ICM $-$ Indep;  neg = ICM better)", fontsize=13)
    ax.set_title("Joint-uncertainty benefit grows with the number of coupled targets",
                 fontsize=13, pad=8)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xticks(range(2, 15))
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out = os.path.join(RDIR, "dnll_vs_T")
    plt.savefig(out + ".pdf", bbox_inches="tight")
    plt.savefig(out + ".png", bbox_inches="tight", dpi=160)
    print(f"\nFigure saved: {out}.pdf / .png")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_rows = []
    for name in DATASETS:
        all_rows += run_sequential(name)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(RDIR, "dnll_vs_T.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")
    print(df.round(3).to_string(index=False))

    plot_results(df)
