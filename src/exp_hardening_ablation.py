"""
#2 -- ICM hardening ablation.

Concern: the Indep-vs-ICM comparison may be "engineered" by the two
hardening choices (shared lengthscale fixed to the Indep median; noise capped at
sig2 <= 0.5). We show the OPPOSITE: hardening HANDICAPS ICM. Relaxing either knob
makes ICM's apparent NLL advantage LARGER (because an unconstrained ICM overfits
the training residual covariance), so the numbers we report in the paper are the
conservative ones. We also sweep the coregionalization rank r in {1, 2, T-1} to
settle whether rank matters (the shipped fit_icm is rank-1).

Variants (all share the same Indep baseline per fold):
  hard_r1     : fixed median ell, sig2<=0.5, rank 1     (paper default)
  hard_r2     : fixed median ell, sig2<=0.5, rank 2
  hard_rfull  : fixed median ell, sig2<=0.5, rank T-1
  nocap_r1    : fixed median ell, sig2<=3.0, rank 1     (cap relaxed)
  optell_r1   : OPTIMISED shared ell, sig2<=0.5, rank 1 (ell freed)

Report per dataset: dNLL, dR2, dCalErr (ICM_variant - Indep). Negative dNLL = ICM
better. The headline check: nocap/optell should make dNLL MORE negative (or at
least not less) than hard_r1 -> hardening is conservative, not favourable.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legJ_jointcalib import icm_predict, indep_predict, joint_metrics
from legJ_jointcalib import load as _load_orig

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root


def load(name):
    """npz-first loader. The plant sets (SH/WUR/rice/wheat) are frozen to
    data_public/<name>.npz from the LOCAL machine (where data_utils/gateB
    resolve), so the cluster run reads the *identical* (X,Y) used by the
    diagnostic/method tables instead of the unwired cluster plant loaders.
    Public MTR sets already live as npz; anything else falls back to load()."""
    p = os.path.join(_HERE, f"data_public/{name.lower()}.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        return z["X"], z["Y"]
    return _load_orig(name)

# Conservativeness ablation on the full 16-dataset converged set (matches the
# diagnostic / method tables). The rank-(T-1) + optimize-ell variants are
# O((nT)^3) per fold, so the high-dim / T=16 additions (scm1d/scm20d, T=16) make
# the 5-variant sweep run for hours -- executed on Snellius, not locally.
# PHO is split into its two homogeneous species (rice, wheat) as elsewhere.
# oes10/oes97 stay excluded (GP optimiser does not converge at p~=n~=300, T=16).
DATASETS = ["atp1d", "atp7d", "rf1", "rf2", "WUR", "SH", "rice", "wheat",
            "jura", "wq", "energy", "tecator", "scm1d", "scm20d", "edm", "sarcos"]
from legJ_jointcalib import dataset_available  # noqa: E402
DATASETS = [d for d in DATASETS if dataset_available(d)]  # skip unavailable sets


def _neg_lml_rank(p, lam_x, Acoef, n, T, r):
    """Eigen-trick neg-LML for a rank-r coregionalization B = W W^T + diag(dd)."""
    sig2 = np.exp(p[-1])
    W = p[:T * r].reshape(T, r)
    dd = np.exp(p[T * r:T * r + T])
    B = W @ W.T + np.diag(dd)
    lam_b, Ub = np.linalg.eigh(B)
    lam_b = np.clip(lam_b, 1e-8, None)
    M = Acoef @ Ub
    denom = np.outer(lam_x, lam_b) + sig2
    return 0.5 * (np.sum(M ** 2 / denom) + np.sum(np.log(denom)) + n * T * np.log(2 * np.pi))


def fit_icm_rank(Phi, Y, ell, rank, sig2_cap=0.5, optimize_ell=False):
    """Fit rank-r ICM at fixed ell (or jointly optimise ell). Returns B, sig2, ell."""
    n, T = Y.shape
    D = squareform(pdist(Phi, "sqeuclidean"))

    def _objective_fixed(p, ell_):
        Kx = np.exp(-D / (2 * ell_ ** 2)) + 1e-6 * np.eye(n)
        lam_x, Ux = np.linalg.eigh(Kx)
        lam_x = np.clip(lam_x, 1e-10, None)
        Acoef = Ux.T @ Y
        return _neg_lml_rank(p, lam_x, Acoef, n, T, rank)

    p0 = np.r_[np.full(T * rank, 0.3 / np.sqrt(rank)), np.zeros(T), np.log(0.1)]
    bnds = [(-5.0, 5.0)] * (T * rank) + [(np.log(1e-3), np.log(3.0))] * T \
        + [(np.log(1e-4), np.log(sig2_cap))]

    if optimize_ell:
        log_ell0 = np.log(ell)
        q0 = np.r_[p0, log_ell0]
        bq = bnds + [(np.log(1e-1), np.log(1e2))]
        res = minimize(lambda q: _objective_fixed(q[:-1], np.exp(q[-1])),
                       q0, method="L-BFGS-B", bounds=bq, options={"maxiter": 300})
        p, ell = res.x[:-1], float(np.exp(res.x[-1]))
    else:
        res = minimize(_objective_fixed, p0, args=(ell,),
                       method="L-BFGS-B", bounds=bnds, options={"maxiter": 300})
        p = res.x
    W = p[:T * rank].reshape(T, rank)
    dd = np.exp(p[T * rank:T * rank + T])
    return W @ W.T + np.diag(dd), float(np.exp(p[-1])), ell


VARIANTS = {
    "hard_r1":    dict(rank=1, sig2_cap=0.5, optimize_ell=False),
    "hard_r2":    dict(rank=2, sig2_cap=0.5, optimize_ell=False),
    "hard_rfull": dict(rank=None, sig2_cap=0.5, optimize_ell=False),   # None -> T-1
    "nocap_r1":   dict(rank=1, sig2_cap=3.0, optimize_ell=False),
    "optell_r1":  dict(rank=1, sig2_cap=0.5, optimize_ell=True),
}


def run_dataset(name, n_sub=280, cv=5, seed=0):
    X, Y = load(name)
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    if len(Xs) > n_sub:
        idx = np.random.default_rng(0).choice(len(Xs), n_sub, replace=False)
        Xs, Ys = Xs[idx], Ys[idx]
    n, T = Ys.shape

    mu_ind = np.zeros((n, T)); Sig_ind = np.zeros((n, T, T))
    preds = {v: (np.zeros((n, T)), np.zeros((n, T, T))) for v in VARIANTS}
    for tr, te in KFold(cv, shuffle=True, random_state=seed).split(Xs):
        mi, Si, ell = indep_predict(Xs[tr], Ys[tr], Xs[te], random_state=seed)
        mu_ind[te], Sig_ind[te] = mi, Si
        for v, cfg in VARIANTS.items():
            r = cfg["rank"] if cfg["rank"] is not None else max(1, T - 1)
            if r >= T and v == "hard_rfull" and T == 2:
                r = 1   # T-1=1 already covered; keep stable
            B, sig2, ell_used = fit_icm_rank(
                Xs[tr], Ys[tr], ell, rank=r,
                sig2_cap=cfg["sig2_cap"], optimize_ell=cfg["optimize_ell"])
            mc, Sc = icm_predict(Xs[tr], Ys[tr], Xs[te], ell_used, B, sig2)
            preds[v][0][te], preds[v][1][te] = mc, Sc

    m_ind = joint_metrics(Ys, mu_ind, Sig_ind)
    rows = []
    for v in VARIANTS:
        m = joint_metrics(Ys, *preds[v])
        rows.append(dict(dataset=name, T=T, variant=v,
                         dnll=round(m["joint_nll"] - m_ind["joint_nll"], 3),
                         dr2=round(m["mean_r2"] - m_ind["mean_r2"], 3),
                         dcal=round(m["joint_cal_err"] - m_ind["joint_cal_err"], 3)))
    return rows


def main():
    rows = []
    for d in DATASETS:
        print(f"[{d}] ...", flush=True)
        rows += run_dataset(d)
    df = pd.DataFrame(rows)
    out = "results/legJ/hardening_ablation.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    piv = df.pivot(index="dataset", columns="variant", values="dnll")
    piv = piv[list(VARIANTS)]
    print("\n=== dNLL (ICM_variant - Indep), negative = ICM better ===")
    print(piv.round(3).to_string())
    print("\nMean dNLL by variant (lower=ICM better on average):")
    print(piv.mean().round(3).to_string())

    print("\n--- CONSERVATIVENESS CHECK: is hardening (hard_r1) WORSE for ICM than relaxed variants? ---")
    base = piv["hard_r1"]
    for v in ["nocap_r1", "optell_r1", "hard_rfull"]:
        better = (piv[v] < base).sum()   # variant gives more-negative (better) dNLL
        print(f"  {v:11s}: more-negative dNLL than hard_r1 in {better}/{len(piv)} datasets "
              f"(mean {piv[v].mean():+.3f} vs hard_r1 {base.mean():+.3f})")
    print("  If relaxed variants are >= as good for ICM, the paper's hardened numbers are conservative.")

    print("\n--- RANK effect (mean dNLL): r1 vs r2 vs rfull ---")
    print(f"  r1={piv['hard_r1'].mean():+.3f}  r2={piv['hard_r2'].mean():+.3f}  "
          f"rfull={piv['hard_rfull'].mean():+.3f}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
