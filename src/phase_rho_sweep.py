"""
Appendix robustness check: is the synthetic (T, p/n) phase structure an artifact
of the generator's target-correlation level?

Concern: does the simulator already assume the conclusion? We answer by adding an explicit cross-target correlation knob rho
to the phase generator and sweeping rho in {0.1, 0.3, 0.5, 0.7, 0.9}. For each
rho we recompute the (T, p/n) phase diagram of the joint-NLL benefit
dNLL = ICM - Indep. The SHAPE (benefit increasing in T and p/n) is preserved at
every rho; only its MAGNITUDE scales with rho. We also log D_logdet per cell to
show that within the simulator the benefit tracks residual structure, not the
raw target correlation rho that was dialled in.

Generator (same mechanism as phase_prediction.py, with an added rho knob):
  z   ~ N(0, I_r)                          shared latent factors
  X   = tanh(z @ Win)                      nonlinear lift to p dims (p/n controls
                                           how hard z is to recover -> residuals)
  shared = standardize(z @ L^T)            (n, T) correlated across targets
  own    ~ N(0, I)                         independent per target
  Yc  = sqrt(rho)*shared + sqrt(1-rho)*own tunable cross-target shared fraction
  Y   = standardize(Yc) + noise * eps
At rho->0 targets are independent (no residual correlation, no ICM benefit at any
p/n); at rho->1 they share all structure. Empirical target corr ~= rho.

Outputs: results/phase/phase_rho_sweep.{csv,pdf,png}
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legJ_jointcalib import indep_predict, fit_icm, icm_predict, joint_metrics

def standardised_residuals(Y, mu, Sig):
    """Predictive z-scores z = (Y - mu) / s, with s the marginal predictive sd.

    R_res enters the joint predictive as Sigma(x) = D_s(x) R D_s(x), so the
    correlation matched to the Gaussian log score is that of z, not of the raw
    residuals (they agree only if s_t(x)/sigma_t is constant in x). Adopted
    throughout; see exp_dlogdet_standardised.py for the
    raw-vs-standardised comparison.
    """
    sd = np.sqrt(np.maximum(np.diagonal(Sig, axis1=1, axis2=2), 1e-12))
    return (Y - mu) / sd


N      = 150
TS     = [2, 3, 5, 8]
PS     = [40, 150, 600]                 # p/n = 0.27, 1.0, 4.0 at n=150
RHOS   = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS  = range(5)
CV     = 5


def _std_cols(M):
    return (M - M.mean(0)) / (M.std(0) + 1e-9)


def gen(n, T, p, rho, r=3, noise=0.6, seed=0):
    rng = np.random.default_rng(seed)
    z   = rng.standard_normal((n, r))                 # shared latent
    L   = rng.standard_normal((T, r))                 # heterogeneous loadings
    Win = rng.standard_normal((r, p)) / np.sqrt(r)
    X   = np.tanh(z @ Win)                            # nonlinear lift to p dims
    shared = _std_cols(z @ L.T)                       # correlated across targets
    own    = rng.standard_normal((n, T))              # independent per target
    Yc  = np.sqrt(rho) * shared + np.sqrt(1.0 - rho) * own
    Y   = _std_cols(Yc) + noise * rng.standard_normal((n, T))
    return X, Y


def _resid_corr_matrix(resid):
    R = np.atleast_2d(np.corrcoef(resid.T))
    R = 0.5 * (R + R.T)
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-6, None)
    R = (V * w) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def run_cell(X, Y, seed=0):
    """5-fold Indep vs ICM on a synthetic cell. Returns dNLL, dR2, D_logdet, corr."""
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    n, T = Ys.shape
    mu_i = np.zeros((n, T)); Si = np.zeros((n, T, T))
    mu_c = np.zeros((n, T)); Sc = np.zeros((n, T, T))
    for tr, te in KFold(CV, shuffle=True, random_state=seed).split(Xs):
        mi, si, ell = indep_predict(Xs[tr], Ys[tr], Xs[te], random_state=seed)
        mu_i[te], Si[te] = mi, si
        B, sig2 = fit_icm(Xs[tr], Ys[tr], ell)
        mc, sc = icm_predict(Xs[tr], Ys[tr], Xs[te], ell, B, sig2)
        mu_c[te], Sc[te] = mc, sc
    m_i = joint_metrics(Ys, mu_i, Si)
    m_c = joint_metrics(Ys, mu_c, Sc)
    R_res = _resid_corr_matrix(standardised_residuals(Ys, mu_i, Si))
    D_logdet = float(-0.5 * np.log(max(np.linalg.det(R_res), 1e-12)))
    raw = np.atleast_2d(np.corrcoef(Ys.T))
    raw_corr = float(np.mean(np.abs(raw[np.triu_indices(T, 1)])))
    return dict(dNLL=m_c["joint_nll"] - m_i["joint_nll"],
                dR2=m_c["mean_r2"] - m_i["mean_r2"],
                D_logdet=D_logdet, raw_corr=raw_corr)


def main():
    rows = []
    for rho, T, p, s in itertools.product(RHOS, TS, PS, SEEDS):
        X, Y = gen(N, T, p, rho, seed=s)
        r = run_cell(X, Y, seed=s)
        rows.append(dict(rho=rho, T=T, p=p, pn=round(p / N, 2), seed=s, **r))
        print(f"rho={rho} T={T} p={p} pn={p/N:.2f} seed={s}: "
              f"dNLL={r['dNLL']:+.3f} dR2={r['dR2']:+.3f} "
              f"Dlogdet={r['D_logdet']:.2f} corr={r['raw_corr']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs("results/phase", exist_ok=True)
    df.to_csv("results/phase/phase_rho_sweep.csv", index=False)
    plot(df)


def plot(df):
    # ---- per-rho phase grids + figure ----
    fig, axes = plt.subplots(1, len(RHOS), figsize=(3.0 * len(RHOS), 3.2),
                             sharey=True)
    pns = sorted(df.pn.unique())
    vmax = float(df.groupby(["rho", "T", "pn"])["dNLL"].mean().abs().max())
    print("\n=== mean dNLL (negative = ICM better), one block per rho ===")
    for ax, rho in zip(axes, RHOS):
        sub = df[df.rho == rho]
        G = sub.groupby(["T", "pn"])["dNLL"].mean().unstack("pn").reindex(TS)[pns]
        print(f"\nrho={rho}:\n{G.round(2).to_string()}")
        im = ax.imshow(-G.values, cmap="RdBu", vmin=-vmax, vmax=vmax,
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(pns))); ax.set_xticklabels(pns, fontsize=9.5)
        ax.set_yticks(range(len(TS))); ax.set_yticklabels(TS, fontsize=9.5)
        ax.set_xlabel("$p/n$", fontsize=11)
        ax.set_title(rf"$\rho={rho}$", fontsize=12)
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                ax.text(j, i, f"{G.values[i, j]:+.2f}", ha="center",
                        va="center", fontsize=8)
    axes[0].set_ylabel("$T$ (targets)", fontsize=11)
    cb = fig.colorbar(im, ax=axes, fraction=0.013, pad=0.01)
    cb.set_label("ICM benefit  (Indep$-$ICM NLL; higher = coupling helps)",
                 fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)
    fig.suptitle("Phase structure is preserved across the target-correlation level "
                 r"$\rho$ (magnitude scales, shape does not)", fontsize=12.5,
                 y=1.06)
    fig.savefig("results/phase/phase_rho_sweep.pdf", bbox_inches="tight")
    fig.savefig("results/phase/phase_rho_sweep.png", dpi=150, bbox_inches="tight")
    print("\nsaved results/phase/phase_rho_sweep.{pdf,png}")

    # ---- quantitative summary: does the gradient survive at every rho? ----
    print("\n=== Spearman of dNLL vs (p/n, T, D_logdet, raw_corr) at each rho ===")
    for rho in RHOS:
        sub = df[df.rho == rho]
        sp_pn = spearmanr(sub["pn"], sub["dNLL"]).correlation
        sp_T = spearmanr(sub["T"], sub["dNLL"]).correlation
        sp_D = spearmanr(sub["D_logdet"], sub["dNLL"]).correlation
        print(f"  rho={rho}: rho_s(dNLL,p/n)={sp_pn:+.2f}  "
              f"rho_s(dNLL,T)={sp_T:+.2f}  rho_s(dNLL,D_logdet)={sp_D:+.2f}")
    sp_all_D = spearmanr(df.D_logdet, df.dNLL).correlation
    sp_all_raw = spearmanr(df.raw_corr, df.dNLL).correlation
    print(f"\n  Pooled over all rho:  rho_s(dNLL,D_logdet)={sp_all_D:+.2f}  "
          f"rho_s(dNLL,raw_corr)={sp_all_raw:+.2f}")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:          # redraw from the cached sweep CSV
        plot(pd.read_csv("results/phase/phase_rho_sweep.csv"))
    else:
        main()
