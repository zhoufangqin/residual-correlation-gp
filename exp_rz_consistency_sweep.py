"""
Full R_z consistency sweep: every statistic derived from D_logdet, recomputed
under both the original raw-residual definition and the standardised-error one,
with a SINGLE sign convention throughout.

Sign convention (uniform here and in the paper):
every correlation is reported against dNLL = NLL_ICM - NLL_Indep, so a NEGATIVE
value means a larger diagnostic predicts a larger improvement.

Covers: headline rho and r, per-target-normalised rho, partial correlations,
the multivariate regression, bootstrap CI, leave-one-dataset-out, permutation
test, per-seed values, the LODO-tau gate, and the fixed-tau ceiling sweep --
because the gate THRESHOLDS on D_logdet, so changing its definition can move
gate decisions even when the overall correlation barely moves.
"""
import numpy as np
import pandas as pd
from scipy import stats

EXCLUDED = ["oes97", "oes10"]
rng = np.random.default_rng(0)

dz = pd.read_csv("results/legJ/dlogdet_standardised.csv")
long = pd.read_csv("results/legJ/residual_icm_multiseed_long.csv")
icm_long = long[long.method == "ICM"][["dataset", "seed", "dnll"]]
per_seed = dz.merge(icm_long, on=["dataset", "seed"])

summ = pd.read_csv("results/legJ/residual_diagnostic_multiseed_summary.csv")
summ = summ[~summ.dataset.isin(EXCLUDED)].set_index("dataset")
g = dz.groupby("dataset")[["D_raw", "D_z"]].mean()
pub = pd.read_csv("results/legJ/residual_icm_multiseed_summary.csv").set_index("dataset")
g = g.join(pub[["ICM_dnll_mean", "r2_indep_mean", "D_logdet_mean"]])
g["T"] = summ["T"]; g["pn"] = summ.p_over_n; g["raw_corr"] = summ.raw_corr
dn = g.ICM_dnll_mean


def partial(y, x, ctrl):
    """Rank-based partial correlation, with the degrees of freedom corrected.

    scipy's pearsonr on the residual vectors assumes df = n - 2, but partialling
    out k controls costs k further degrees of freedom, so the correct null has
    df = n - 2 - k. At n=16 with three controls that is 11 rather than 14, which
    moves the headline partial from p=0.013 to p=0.028 -- not negligible.
    """
    rk = stats.rankdata
    k = len(ctrl)
    C = np.column_stack([rk(g[c]) for c in ctrl] + [np.ones(len(g))])
    ry = rk(y) - C @ np.linalg.lstsq(C, rk(y), rcond=None)[0]
    rx = rk(x) - C @ np.linalg.lstsq(C, rk(x), rcond=None)[0]
    r = stats.pearsonr(rx, ry).statistic
    df = len(g) - 2 - k
    t = r * np.sqrt(df / max(1 - r ** 2, 1e-12))
    return r, 2 * (1 - stats.t.cdf(abs(t), df))


def ols(cols):
    X = np.column_stack([np.ones(len(g))] +
                        [((g[c] - g[c].mean()) / g[c].std()).values for c in cols])
    y = ((dn - dn.mean()) / dn.std()).values
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    dof = len(g) - X.shape[1]
    se = np.sqrt(np.diag((r @ r / dof) * np.linalg.inv(X.T @ X)))
    t = b / se
    return 1 - (r @ r) / (y @ y), dict(zip(cols, zip(b[1:], t[1:],
                                        2 * (1 - stats.t.cdf(abs(t[1:]), dof)))))


def lodo_gate(D, R2, ICM, ceiling=0.9):
    grid = np.unique(np.concatenate([[0.0], np.sort(D), [D.max() + 1]]))
    f = lambda t: np.where((D > t) & (R2 < ceiling), ICM, 0.0)
    out, taus = np.zeros(len(D)), []
    for i in range(len(D)):
        m = np.ones(len(D), bool); m[i] = False
        tau = min(grid, key=lambda t: f(t)[m].mean())
        taus.append(tau); out[i] = f(tau)[i]
    return out, taus


for name, D in (("R_raw (original)", g.D_raw), ("R_z  (revised)", g.D_z)):
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    r_s = stats.spearmanr(D, dn); r_p = stats.pearsonr(D, dn)
    print(f"  headline vs dNLL      Spearman {r_s.statistic:+.3f} (p={r_s.pvalue:.1e})"
          f"   Pearson {r_p.statistic:+.3f}")
    rn = stats.spearmanr(D / g["T"], dn / g["T"])
    print(f"  per-target normalised Spearman {rn.statistic:+.3f} (p={rn.pvalue:.1e})")
    for ctrl in (["T"], ["T", "pn", "r2_indep_mean"]):
        rr, pp = partial(dn, D, ctrl)
        print(f"  partial | {','.join(ctrl):22s} {rr:+.3f} (p={pp:.4f})")
    rr, pp = partial(dn, g["T"], ["D_raw" if "raw" in name else "D_z"])
    print(f"  partial rho(T | D)            {rr:+.3f} (p={pp:.3f})")

    col = "D_raw" if "raw" in name else "D_z"
    r2a, _ = ols(["T", "pn", "r2_indep_mean"])
    r2b, co = ols(["T", "pn", "r2_indep_mean", col])
    print(f"  regression R2  {r2a:.3f} -> {r2b:.3f}   "
          f"beta_D={co[col][0]:+.2f} (t={co[col][1]:+.1f})  "
          f"beta_T={co['T'][0]:+.2f} (p={co['T'][2]:.2f})")

    bs = []
    for _ in range(10000):
        i = rng.integers(0, len(g), len(g))
        if len(set(D.values[i])) < 3:
            continue
        bs.append(stats.spearmanr(D.values[i], dn.values[i]).statistic)
    bs = np.array([b for b in bs if not np.isnan(b)])
    print(f"  bootstrap 95% CI      [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
    lo = [stats.spearmanr(D.drop(i), dn.drop(i)).statistic for i in g.index]
    print(f"  leave-one-dataset-out [{min(lo):+.3f}, {max(lo):+.3f}]")
    perm = np.array([stats.spearmanr(rng.permutation(D.values), dn.values).statistic
                     for _ in range(10000)])
    print(f"  permutation p         {(perm <= r_s.statistic).mean():.5f}")
    ps = [stats.spearmanr(x[col], x.dnll).statistic
          for _, x in per_seed[~per_seed.dataset.isin(EXCLUDED)].groupby("seed")]
    print(f"  per-seed rho          [{min(ps):+.3f}, {max(ps):+.3f}]")

    gate, taus = lodo_gate(D.values, g.r2_indep_mean.values, g.ICM_dnll_mean.values)
    print(f"  LODO gate: mean {gate.mean():+.3f}  couples {int((gate!=0).sum())}/16  "
          f"harms {int((gate>0).sum())}/16 for {gate.clip(min=0).sum():.2f} nats")
    changed = [d for d, a, b in zip(g.index, lodo_gate(g.D_raw.values,
               g.r2_indep_mean.values, g.ICM_dnll_mean.values)[0], gate) if (a != 0) != (b != 0)]
    if "z" in name:
        print(f"  gate decisions changed vs R_raw: {changed if changed else 'none'}")
    sweep = []
    for c in (0.80, 0.85, 0.90, 0.926, 0.93, 0.935, 1.0):
        gg = np.where((D.values > 0.5) & (g.r2_indep_mean.values < c),
                      g.ICM_dnll_mean.values, 0.0)
        sweep.append(f"{c}:{gg.mean():+.3f}")
    print("  fixed-tau=0.5 ceiling sweep  " + "  ".join(sweep))

print(f"\n{'='*72}\nUnchanged predictors (do not depend on D_logdet):")
print(f"  Spearman(raw target corr, dNLL) {stats.spearmanr(g.raw_corr, dn).statistic:+.3f}")
print(f"  Spearman(log p/n, dNLL)         {stats.spearmanr(np.log(g.pn), dn).statistic:+.3f}")
g.round(4).to_csv("results/legJ/rz_sweep_inputs.csv")
