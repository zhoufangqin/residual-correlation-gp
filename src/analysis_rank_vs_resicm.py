"""Merge the 5-seed ICM rank sweep with the Table 3 columns (Appendix B, tab:rank_resicm).

Compares rank-1 / rank-2 / rank-(T-1) / oracle-rank ICM against Residual-ICM on
the SAME 5 seeds and the same 16 datasets, on mean dNLL and on the tail (worst
case, number of datasets harmed).
"""
import numpy as np
import pandas as pd

rank = pd.read_csv("results/legJ/hardening_multiseed.csv")
res = pd.read_csv("results/legJ/residual_icm_multiseed_long.csv")

seeds = sorted(set(rank.seed) & set(res.seed))
rank = rank[rank.seed.isin(seeds)]
res = res[res.seed.isin(seeds)]
print(f"seeds: {seeds}   datasets: {rank.dataset.nunique()}")

piv = rank.pivot_table(index="dataset", columns="variant", values="dnll", aggfunc="mean")
piv = piv[["hard_r1", "hard_r2", "hard_rfull"]]
piv["oracle_rank"] = piv.min(axis=1)
# For T=2, rank T-1 = 1, so hard_rfull is hard_r1 by construction (6 of the 16
# datasets: SH, WUR, rice, wheat, energy, edm). The rank sweep therefore only
# has three genuinely distinct levels on the T>=3 datasets; report both.
T2 = sorted(rank[rank["T"] == 2].dataset.unique())
print(f"T=2 datasets (rfull == r1 by construction): {T2}")
resicm = res[res.method == "ResICM"].pivot_table(index="dataset", values="dnll", aggfunc="mean")
icm_t4 = res[res.method == "ICM"].pivot_table(index="dataset", values="dnll", aggfunc="mean")
m = piv.join(resicm.rename(columns={"dnll": "ResICM"})).join(icm_t4.rename(columns={"dnll": "ICM_Table4"}))

# The paper reports the predictive-STANDARDISED Residual-ICM (CR1). Its
# CANONICAL values are the ones behind Table 5, i.e. table4_matched.csv --
# use those in the appendix table so the paper quotes one value for
# Residual-ICM everywhere (-1.13 / -1.54), rather than a second realisation.
# Residual-ICM does not depend on the ICM rank, so re-running it here would
# only add optimiser noise: the two realisations differ on 12 of 16 datasets,
# by up to 0.097 on rf1, which is why the appendix used to read -1.12.
canon = pd.read_csv("results/legJ/table4_matched.csv", index_col=0)
m = m.join(canon["ResICM"].rename("ResICM_canonical"))
zr = pd.read_csv("results/legJ/resicm_zresid.csv")
zr = zr[zr.seed.isin(seeds)]
for meth in ("ResICM_z", "ResICM_raw"):
    col = zr[zr.method == meth].pivot_table(index="dataset", values="dnll", aggfunc="mean")
    m = m.join(col.rename(columns={"dnll": meth}))
m = m.dropna()

print("\n=== 5-seed mean dNLL per dataset (negative favours coupling) ===")
print(m.round(3).to_string())

cols = ["hard_r1", "hard_r2", "hard_rfull", "oracle_rank", "ResICM_canonical", "ResICM_z", "ResICM"]
print("\n=== summary over %d datasets ===" % len(m))
print(f"{'variant':13s} {'mean':>8s} {'median':>8s} {'worst':>8s} {'#harmed':>8s} {'harm':>7s} {'rank':>6s}")
ranks = m[["hard_r1", "hard_r2", "hard_rfull", "ResICM"]].rank(axis=1).mean()
for c in cols:
    x = m[c]
    r = f"{ranks[c]:.2f}" if c in ranks.index else "  -"
    print(f"{c:13s} {x.mean():+8.3f} {x.median():+8.3f} {x.max():+8.3f} "
          f"{(x > 0).sum():4d}/{len(m):<3d} {x.clip(lower=0).sum():7.2f} {r:>6s}")

print("\nsanity: hard_r1 vs Table 4 ICM column (should be near-identical; "
      "fit_icm maxiter=200 vs fit_icm_rank maxiter=300)")
d = (m.hard_r1 - m.ICM_Table4).abs()
print(f"  max |diff| = {d.max():.3f} on {d.idxmax()};  mean |diff| = {d.mean():.3f}")

big = m.drop(index=[d for d in T2 if d in m.index])
print(f"\n=== same summary on the {len(big)} datasets with T>=3 (rank is non-degenerate) ===")
for c in cols:
    x = big[c]
    print(f"{c:13s} {x.mean():+8.3f} {x.median():+8.3f} {x.max():+8.3f} "
          f"{(x > 0).sum():4d}/{len(big):<3d} {x.clip(lower=0).sum():7.2f}")

print("\nhead-to-head, Residual-ICM vs each ICM rank:")
for c in ["hard_r1", "hard_r2", "hard_rfull", "oracle_rank"]:
    w = (m.ResICM_canonical < m[c]).sum()
    print(f"  Residual-ICM better than {c:12s}: {w:2d}/{len(m)}  "
          f"(mean {m.ResICM_canonical.mean():+.3f} vs {m[c].mean():+.3f})")

m.round(4).to_csv("results/legJ/rank_vs_resicm_5seed.csv")
print("\nSaved: results/legJ/rank_vs_resicm_5seed.csv")
print("The ResICM_canonical column and the two summary blocks above are the source of\nthe rank-sweep table in the appendix (ICM ranks + hindsight rank vs\nResidual-ICM, all 16 and T>=3, mean and sum of positive dNLL).")
