"""
#1 -- Residual-correlation diagnostic.

The paper claims ICM's joint-NLL benefit is driven by RESIDUAL correlation (what
independent GPs leave unexplained), not raw target correlation, and uses p/n as a
crude proxy (Spearman -0.54 vs raw-corr 0.00 across 11 datasets).

This experiment replaces the proxy with a DIRECT, PRINCIPLED measurement.

For each dataset we run the same out-of-fold CV as legJ, collect the independent-GP
residuals  r = Y - mu_indep,  and form the residual correlation matrix R_res. Then:

  Under a Gaussian model where ICM and Indep differ ONLY in whether the residual
  covariance is full vs diagonal, the expected per-point joint-NLL reduction is
        D_logdet = -0.5 * log det(R_res)            (>= 0, =0 iff residuals uncorrelated)
  This is not a heuristic: it is the exact expected NLL gap between the correct
  correlated-Gaussian predictive and its diagonal approximation.

We compare four candidate predictors of the measured dNLL = NLL_ICM - NLL_Indep:
  (a) raw_corr     : mean |off-diag| of corr(Y)          [paper baseline, rho_s ~ 0]
  (b) p_over_n     : feature/sample ratio               [paper proxy,    rho_s ~ -0.54]
  (c) resid_corr   : mean |off-diag| of R_res            [direct residual measure]
  (d) D_logdet     : -0.5 log det(R_res)                 [principled, predicted dNLL]

Output: per-dataset table + Spearman of each predictor against dNLL.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legJ_jointcalib import (fit_icm, icm_predict, indep_predict,
                             joint_metrics, load)


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

# Expanded set (2026-06-10): PHO split into rice+wheat (per-species, homogeneous);
# added Mulan MTR sets (scm1d/scm20d/edm/oes10/oes97) and SARCOS for robustness.
DATASETS = ["atp1d", "atp7d", "rf1", "rf2", "WUR", "SH", "rice", "wheat",
            "jura", "wq", "energy", "tecator",
            "scm1d", "scm20d", "edm", "oes10", "oes97", "sarcos"]
from legJ_jointcalib import dataset_available  # noqa: E402
DATASETS = [d for d in DATASETS if dataset_available(d)]  # skip unavailable sets


def _resid_corr_matrix(resid):
    """Correlation matrix of the per-target residuals (T x T)."""
    R = np.corrcoef(resid.T)
    R = np.atleast_2d(R)
    # numerical guard: symmetrise and floor eigenvalues for a stable log-det
    R = 0.5 * (R + R.T)
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-6, None)
    R = (V * w) @ V.T
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)            # renormalise to unit diagonal
    return R


def dlogdet_lw(resid):
    """D_logdet computed from a Ledoit-Wolf shrinkage estimate of the residual
    covariance instead of the raw empirical correlation. Ledoit-Wolf shrinks
    adaptively toward a scaled identity -- strongly when the residual matrix is
    ill-conditioned (collinear targets / n<T, the AFLW saturation regime),
    negligibly when it is well-conditioned -- so it restores resolution where the
    plain estimator floors at -0.5 log(det_floor) without disturbing the
    well-sampled tabular cases. Returns -0.5 log det of the shrunk correlation."""
    from sklearn.covariance import LedoitWolf
    resid = np.atleast_2d(np.asarray(resid, float))
    T = resid.shape[1]
    if T < 2:
        return 0.0
    cov = LedoitWolf().fit(resid).covariance_
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    R = cov / np.outer(d, d)
    _, logdet = np.linalg.slogdet(R)
    return float(-0.5 * logdet)


def run_dataset(name, n_sub=280, cv=5, seed=0):
    X, Y = load(name)
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    if len(Xs) > n_sub:
        idx = np.random.default_rng(seed).choice(len(Xs), n_sub, replace=False)
        Xs, Ys = Xs[idx], Ys[idx]
    n, T = Ys.shape
    p = X.shape[1]

    mu_icm = np.zeros((n, T)); Sig_icm = np.zeros((n, T, T))
    mu_ind = np.zeros((n, T)); Sig_ind = np.zeros((n, T, T))
    for tr, te in KFold(cv, shuffle=True, random_state=seed).split(Xs):
        mi, Si, ell = indep_predict(Xs[tr], Ys[tr], Xs[te], random_state=seed)
        mu_ind[te], Sig_ind[te] = mi, Si
        B, sig2 = fit_icm(Xs[tr], Ys[tr], ell)
        mc, Sc = icm_predict(Xs[tr], Ys[tr], Xs[te], ell, B, sig2)
        mu_icm[te], Sig_icm[te] = mc, Sc

    m_icm = joint_metrics(Ys, mu_icm, Sig_icm)
    m_ind = joint_metrics(Ys, mu_ind, Sig_ind)
    dnll = m_icm["joint_nll"] - m_ind["joint_nll"]
    dr = m_icm["mean_r2"] - m_ind["mean_r2"]

    resid = standardised_residuals(Ys, mu_ind, Sig_ind)   # out-of-fold z-scores
    R_res = _resid_corr_matrix(resid)
    off = R_res[np.triu_indices(T, 1)]
    resid_corr = float(np.mean(np.abs(off)))
    D_logdet = float(-0.5 * np.log(max(np.linalg.det(R_res), 1e-12)))
    D_logdet_lw = dlogdet_lw(resid)

    raw = np.corrcoef(Ys.T)
    raw = np.atleast_2d(raw)
    raw_corr = float(np.mean(np.abs(raw[np.triu_indices(T, 1)])))

    return dict(dataset=name, n=n, T=T, p=p, p_over_n=round(p / n, 3),
                raw_corr=round(raw_corr, 3), resid_corr=round(resid_corr, 3),
                D_logdet=round(D_logdet, 4), D_logdet_lw=round(D_logdet_lw, 4),
                r2_indep=round(m_ind["mean_r2"], 3),
                dnll=round(dnll, 3), dr=round(dr, 3))


def main():
    rows = [run_dataset(d) for d in DATASETS
            for _ in [print(f"[{d}] ...", flush=True)]]
    df = pd.DataFrame(rows).sort_values("dnll").reset_index(drop=True)
    out = "results/legJ/residual_diagnostic.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print("\n=== Residual-correlation diagnostic (sorted by dNLL) ===")
    print(df.to_string(index=False))

    print("\n--- Spearman rho_s of each predictor vs measured dNLL (negative=ICM better) ---")
    print("  (we want |rho_s| LARGE for a good predictor of ICM benefit)")
    for col, sign in [("raw_corr", "baseline, paper says ~0"),
                      ("p_over_n", "paper proxy, ~ -0.54"),
                      ("resid_corr", "direct residual measure"),
                      ("D_logdet", "principled, predicted dNLL")]:
        rho, pval = spearmanr(df[col], df["dnll"])
        print(f"  {col:12s} rho_s = {rho:+.3f}  (p={pval:.3f})   [{sign}]")

    # the principled diagnostic should also predict dNLL near-linearly in magnitude;
    # report Pearson r between -D_logdet and dnll as a calibration check
    r_lin = np.corrcoef(-df["D_logdet"], df["dnll"])[0, 1]
    print(f"\n  Pearson(-D_logdet, dNLL) = {r_lin:+.3f}  "
          f"(D_logdet predicts the *magnitude* of the NLL gain, not just rank)")

    # robustness of the headline D_logdet correlation (small-n / cherry-pick concern)
    rng = np.random.default_rng(0)
    D = df["D_logdet"].to_numpy(); Nv = df["dnll"].to_numpy()
    rho0, _ = spearmanr(D, Nv)
    boots = []
    for _ in range(20000):
        idx = rng.integers(0, len(D), len(D))
        if len(np.unique(D[idx])) < 3:
            continue
        r, _ = spearmanr(D[idx], Nv[idx])
        if np.isfinite(r):
            boots.append(r)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    P = 50000
    cnt = sum(spearmanr(D, rng.permutation(Nv))[0] <= rho0 for _ in range(P))
    perm_p = (cnt + 1) / (P + 1)
    # leave-one-dataset-out jackknife: is any single dataset driving the result?
    lodo = []
    for i in range(len(D)):
        m = np.ones(len(D), bool); m[i] = False
        lodo.append(spearmanr(D[m], Nv[m])[0])
    lodo = np.array(lodo)
    imax = int(np.argmax(np.abs(lodo - rho0)))
    print(f"\n  [robustness] D_logdet vs dNLL on n={len(D)} datasets: "
          f"rho_s={rho0:+.3f}, bootstrap 95% CI=[{lo:.2f}, {hi:.2f}], "
          f"permutation p={perm_p:.4f}")
    print(f"  [robustness] leave-one-dataset-out rho_s in [{lodo.min():+.3f}, "
          f"{lodo.max():+.3f}]; most influential = -{df['dataset'].iloc[imax]} "
          f"(rho_s={lodo[imax]:+.3f})")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
