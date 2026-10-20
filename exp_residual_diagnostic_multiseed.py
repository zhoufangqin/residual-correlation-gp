"""Multi-seed robustness wrapper for the residual-correlation diagnostic.

The single-output GP uses one optimizer restart, so each run varies slightly
(worst for the high-dim T=16 sets). We therefore run every dataset under S seeds
(each varies the 280-point subsample, the CV split, and the GP restart), report
seed-averaged per-dataset metrics (stable table values), and characterise the
stability of the headline Spearman rho_s three ways:
  - across seeds (mean +/- SD of the per-seed rho_s),
  - leave-one-dataset-out on the seed-averaged values,
  - label permutation + dataset bootstrap on the seed-averaged values.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from exp_residual_diagnostic import run_dataset, DATASETS

SEEDS = [0, 1, 2, 3, 4]
OUT = "results/legJ"


def main():
    os.makedirs(OUT, exist_ok=True)
    long = []
    for s in SEEDS:
        for d in DATASETS:
            print(f"[seed {s}] {d} ...", flush=True)
            r = run_dataset(d, seed=s)
            r["seed"] = s
            long.append(r)
    L = pd.DataFrame(long)
    L.to_csv(f"{OUT}/residual_diagnostic_multiseed_long.csv", index=False)

    # per-dataset seed means/SDs (the values that go in the paper table)
    g = L.groupby("dataset")
    summ = g.agg(n=("n", "first"), p=("p", "first"), T=("T", "first"),
                 p_over_n=("p_over_n", "first"), raw_corr=("raw_corr", "mean"),
                 D_logdet=("D_logdet", "mean"), D_logdet_sd=("D_logdet", "std"),
                 dnll=("dnll", "mean"), dnll_sd=("dnll", "std"),
                 dr=("dr", "mean"), r2_indep=("r2_indep", "mean")).reset_index()
    summ = summ.sort_values("dnll").reset_index(drop=True)
    summ.to_csv(f"{OUT}/residual_diagnostic_multiseed_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== seed-averaged per-dataset diagnostic (sorted by dNLL) ===")
    print(summ.round(3).to_string(index=False))

    # (1) rho_s stability across seeds
    per_seed = []
    for s in SEEDS:
        sub = L[L.seed == s]
        per_seed.append(spearmanr(sub["D_logdet"], sub["dnll"])[0])
    per_seed = np.array(per_seed)
    print(f"\n[stability] per-seed rho_s(D_logdet, dNLL) = {per_seed.round(3).tolist()}")
    print(f"[stability]   mean +/- SD over {len(SEEDS)} seeds = "
          f"{per_seed.mean():+.3f} +/- {per_seed.std(ddof=1):.3f}")

    # (2) rho_s on seed-averaged values + LODO + permutation + bootstrap
    D = summ["D_logdet"].to_numpy(); N = summ["dnll"].to_numpy()
    rho0, _ = spearmanr(D, N)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(20000):
        idx = rng.integers(0, len(D), len(D))
        if len(np.unique(D[idx])) < 3:
            continue
        r, _ = spearmanr(D[idx], N[idx])
        if np.isfinite(r):
            boots.append(r)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    P = 50000
    cnt = sum(spearmanr(D, rng.permutation(N))[0] <= rho0 for _ in range(P))
    perm_p = (cnt + 1) / (P + 1)
    lodo = np.array([spearmanr(np.delete(D, i), np.delete(N, i))[0]
                     for i in range(len(D))])
    imax = int(np.argmax(np.abs(lodo - rho0)))
    pear = np.corrcoef(-D, N)[0, 1]
    print(f"\n[on seed-means] rho_s(D_logdet, dNLL) = {rho0:+.3f}  (Pearson -D vs dNLL = {pear:+.3f})")
    print(f"[on seed-means] bootstrap 95% CI = [{lo:.2f}, {hi:.2f}], permutation p = {perm_p:.4f}")
    print(f"[on seed-means] leave-one-dataset-out rho_s in [{lodo.min():+.3f}, {lodo.max():+.3f}]; "
          f"most influential = -{summ['dataset'].iloc[imax]} (rho_s={lodo[imax]:+.3f})")
    for col in ["raw_corr", "p_over_n"]:
        rr, pp = spearmanr(summ[col], summ["dnll"])
        print(f"[on seed-means] rho_s({col}, dNLL) = {rr:+.3f} (p={pp:.3f})")
    print(f"\nSaved: {OUT}/residual_diagnostic_multiseed_summary.csv")


if __name__ == "__main__":
    main()
