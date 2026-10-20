"""
Multi-seed ICM rank sweep (Table 3 rank-(T-1) column; Appendix B).

exp_hardening_ablation.py ran the rank sweep {1, 2, T-1} at a SINGLE seed with a
FIXED subsample (default_rng(0)), while Table 4's ICM / Residual-ICM columns are
5-seed means under the exp_residual_icm_multiseed.py protocol. The higher-rank /
rank-selected ICM is reported in the main results, so the rank variants need the same 5-seed footing as the columns they are compared against.

Protocol matched to exp_residual_icm_multiseed.py:
  - subsample with default_rng(SEED)   (was fixed rng(0) in the ablation)
  - KFold(random_state=SEED)
  - indep_predict(random_state=SEED)
over SEEDS = [0, 1, 2, 3, 4].

Variants (all share the same Indep baseline per fold, as in the ablation):
  hard_r1     rank 1      (paper default; should reproduce Table 4's ICM column)
  hard_r2     rank 2
  hard_rfull  rank T-1

Rows are appended per (dataset, seed) so the run is resumable and parallel.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_hardening_ablation import fit_icm_rank, load
from legJ_jointcalib import icm_predict, indep_predict, joint_metrics

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "results/legJ/hardening_multiseed.csv")

DATASETS = ["atp1d", "atp7d", "rf1", "rf2", "WUR", "SH", "rice", "wheat",
            "jura", "wq", "energy", "tecator", "scm1d", "scm20d", "edm", "sarcos"]
from legJ_jointcalib import dataset_available  # noqa: E402
DATASETS = [d for d in DATASETS if dataset_available(d)]  # skip unavailable sets
SEEDS = [0, 1, 2, 3, 4]
# rank sweep plus the two relaxation variants, so Appendix B's table can be
# reported on the same five seeds as Table 4 (it was previously single-seed,
# which left the same quantity printed twice with different values).
RANK_VARIANTS = {"hard_r1": dict(rank=1, sig2_cap=0.5, optimize_ell=False),
                 "hard_r2": dict(rank=2, sig2_cap=0.5, optimize_ell=False),
                 "hard_rfull": dict(rank=None, sig2_cap=0.5, optimize_ell=False),
                 "nocap_r1": dict(rank=1, sig2_cap=3.0, optimize_ell=False),
                 "optell_r1": dict(rank=1, sig2_cap=0.5, optimize_ell=True)}


def run_job(args):
    """One (dataset, seed) cell: fit Indep once per fold, then all rank variants."""
    name, seed = args
    X, Y = load(name)
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    if len(Xs) > 280:
        idx = np.random.default_rng(seed).choice(len(Xs), 280, replace=False)
        Xs, Ys = Xs[idx], Ys[idx]
    n, T = Ys.shape

    mu_ind = np.zeros((n, T)); Sig_ind = np.zeros((n, T, T))
    preds = {v: (np.zeros((n, T)), np.zeros((n, T, T))) for v in RANK_VARIANTS}
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xs):
        mi, Si, ell = indep_predict(Xs[tr], Ys[tr], Xs[te], random_state=seed)
        mu_ind[te], Sig_ind[te] = mi, Si
        for v, cfg in RANK_VARIANTS.items():
            r = cfg["rank"] if cfg["rank"] is not None else max(1, T - 1)
            if v == "hard_rfull" and T == 2:
                r = 1                      # T-1 = 1; identical to hard_r1 by construction
            B, sig2, ell_used = fit_icm_rank(Xs[tr], Ys[tr], ell, rank=r,
                                             sig2_cap=cfg["sig2_cap"],
                                             optimize_ell=cfg["optimize_ell"])
            mc, Sc = icm_predict(Xs[tr], Ys[tr], Xs[te], ell_used, B, sig2)
            preds[v][0][te], preds[v][1][te] = mc, Sc

    m_ind = joint_metrics(Ys, mu_ind, Sig_ind)
    rows = []
    for v in RANK_VARIANTS:
        m = joint_metrics(Ys, *preds[v])
        rows.append(dict(dataset=name, T=T, seed=seed, variant=v,
                         dnll=round(m["joint_nll"] - m_ind["joint_nll"], 4),
                         dr2=round(m["mean_r2"] - m_ind["mean_r2"], 4),
                         dcal=round(m["joint_cal_err"] - m_ind["joint_cal_err"], 4)))
    return rows


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # resume at (dataset, seed, variant) granularity: the file already holds the
    # three rank variants for all seeds, and only nocap/optell are outstanding.
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = {(d, s) for (d, s), grp in prev.groupby(["dataset", "seed"])
                if set(RANK_VARIANTS) <= set(grp.variant)}
        print(f"resuming: {len(done)} complete (dataset, seed) cells on disk")

    # Longest-processing-time-first: cost grows ~(nT)^3, so start the T=16 cells
    # early or they become the makespan tail.
    order = ["scm1d", "scm20d", "wq", "rf1", "rf2", "sarcos", "atp7d", "atp1d",
             "tecator", "jura", "SH", "WUR", "rice", "wheat", "energy", "edm"]
    jobs = [(d, s) for d in order for s in SEEDS if (d, s) not in done]
    print(f"{len(jobs)} jobs to run on {min(10, len(jobs))} workers", flush=True)

    import multiprocessing as mp
    with mp.Pool(10) as pool:
        for rows in pool.imap_unordered(run_job, jobs):
            df = pd.DataFrame(rows)
            df.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            r = rows[0]
            print(f"[done] {r['dataset']:8s} seed={r['seed']}  "
                  + "  ".join(f"{x['variant']}={x['dnll']:+.3f}" for x in rows), flush=True)

    summarize()


def summarize():
    df = pd.read_csv(OUT)
    piv = df.pivot_table(index="dataset", columns="variant", values="dnll", aggfunc="mean")
    piv = piv[[c for c in RANK_VARIANTS if c in piv.columns]]
    print("\n=== 5-seed mean dNLL (ICM_rank - Indep); negative = ICM better ===")
    print(piv.round(3).to_string())
    print("\nMean over datasets:")
    print(piv.mean().round(3).to_string())
    print("\nWorst case / # datasets harmed:")
    for v in piv.columns:
        print(f"  {v:11s} worst={piv[v].max():+.3f}  harmed={(piv[v] > 0).sum()}/{len(piv)}"
              f"  total_harm={piv[v].clip(lower=0).sum():.2f}")
    orc = piv.min(axis=1)
    print(f"  {'oracle-rank':11s} mean={orc.mean():+.3f}  worst={orc.max():+.3f}"
          f"  harmed={(orc > 0).sum()}/{len(piv)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
    else:
        main()
