"""
Regenerate Table 3 (tab:method) with the matched (standardised-residual) R_res.

Only the Residual-ICM column changes: the estimator switch does not touch the
Indep or ICM fits, so those are reused verbatim from
results/legJ/residual_icm_multiseed_long.csv (re-fitting ICM here would also
re-expose the fit_icm maxiter/BLAS sensitivity on SH/WUR, which is a separate
open issue and must not leak into this table).

Indep IS recomputed and checked against the published NLL, so the reuse is
verified rather than assumed.

Protocol identical to exp_residual_icm_multiseed.py: subsample rng(SEED),
KFold(random_state=SEED), indep_predict(random_state=SEED), and the inner
residual CV seeded with `seed` (NOT the hardcoded 1 used by the exploratory
exp_resicm_zresid.py).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_hardening_ablation import load
from exp_residual_icm_multiseed import residual_corr_from_train, resicm_predict
from legJ_jointcalib import indep_predict, joint_metrics

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "results/legJ/resicm_matched_long.csv")
PUBLISHED = os.path.join(_HERE, "results/legJ/residual_icm_multiseed_long.csv")

DATASETS = ["atp1d", "atp7d", "rf1", "rf2", "WUR", "SH", "rice", "wheat",
            "jura", "wq", "energy", "tecator", "scm1d", "scm20d", "edm", "sarcos"]
from legJ_jointcalib import dataset_available  # noqa: E402
DATASETS = [d for d in DATASETS if dataset_available(d)]  # skip unavailable sets
SEEDS = [0, 1, 2, 3, 4]


def run_job(args):
    name, seed = args
    X, Y = load(name)
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    if len(Xs) > 280:
        idx = np.random.default_rng(seed).choice(len(Xs), 280, replace=False)
        Xs, Ys = Xs[idx], Ys[idx]
    n, T = Ys.shape

    P = {m: (np.zeros((n, T)), np.zeros((n, T, T))) for m in ("Indep", "ResICM")}
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xs):
        mi, Si, _ = indep_predict(Xs[tr], Ys[tr], Xs[te], random_state=seed)
        P["Indep"][0][te], P["Indep"][1][te] = mi, Si
        R_res = residual_corr_from_train(Xs[tr], Ys[tr], seed=seed)
        mr, Sr = resicm_predict(Xs[tr], Ys[tr], Xs[te], R_res)
        P["ResICM"][0][te], P["ResICM"][1][te] = mr, Sr

    M = {m: joint_metrics(Ys, *P[m]) for m in P}
    base = M["Indep"]
    return [dict(dataset=name, seed=seed, T=T, method=m,
                 nll=M[m]["joint_nll"], cal=M[m]["joint_cal_err"],
                 r2=M[m]["mean_r2"],
                 dnll=round(M[m]["joint_nll"] - base["joint_nll"], 4))
            for m in ("Indep", "ResICM")]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.dataset, prev.seed))
        print(f"resuming: {len(done)} cells on disk")
    order = ["scm1d", "scm20d", "wq", "rf1", "rf2", "sarcos", "atp7d", "atp1d",
             "tecator", "jura", "SH", "WUR", "rice", "wheat", "energy", "edm"]
    jobs = [(d, s) for d in order for s in SEEDS if (d, s) not in done]
    print(f"{len(jobs)} jobs on 10 workers", flush=True)

    import multiprocessing as mp
    with mp.Pool(10) as pool:
        for rows in pool.imap_unordered(run_job, jobs):
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT),
                                      index=False)
            r = [x for x in rows if x["method"] == "ResICM"][0]
            print(f"[done] {r['dataset']:8s} seed={r['seed']}  ResICM={r['dnll']:+.3f}",
                  flush=True)
    build_table()


def build_table():
    new = pd.read_csv(OUT)
    pub = pd.read_csv(PUBLISHED)

    # verify the reused arms: Indep must reproduce exactly
    a = new[new.method == "Indep"].set_index(["dataset", "seed"]).nll
    b = pub[pub.method == "Indep"].set_index(["dataset", "seed"]).nll
    d = (a - b.reindex(a.index)).abs()
    print(f"\nIndep NLL reproduction check: max |diff| = {d.max():.6f}, "
          f"mean {d.mean():.6f}  ({'OK' if d.max() < 1e-6 else 'MISMATCH'})")

    icm = pub[pub.method == "ICM"].groupby("dataset").dnll.mean()
    res_new = new[new.method == "ResICM"].groupby("dataset").dnll.mean()
    res_old = pub[pub.method == "ResICM"].groupby("dataset").dnll.mean()
    diag = pub.groupby("dataset")[["D_logdet", "r2_indep"]].mean()

    t = pd.DataFrame({"Indep": 0.0, "ICM": icm, "ResICM_old": res_old,
                      "ResICM": res_new}).join(diag)
    # gate: couple iff D_logdet > tau and r2_indep < 0.9 (fixed tau=0.5)
    t["Gate"] = np.where((t.D_logdet > 0.5) & (t.r2_indep < 0.9), t.ICM, 0.0)

    print("\n=== Table 3 inputs (5-seed mean dNLL) ===")
    print(t[["Indep", "ICM", "ResICM_old", "ResICM", "Gate"]].round(3).to_string())

    ranks = t[["Indep", "ICM", "ResICM", "Gate"]].rank(axis=1).mean()
    print("\nMean dNLL over 16 datasets:")
    print(t[["Indep", "ICM", "ResICM_old", "ResICM", "Gate"]].mean().round(3).to_string())
    print("\nMean rank (1=best):")
    print(ranks.round(2).to_string())
    print(f"\nResICM: harmed {(t.ResICM > 0).sum()}/16, worst {t.ResICM.max():+.3f}, "
          f"total harm {t.ResICM.clip(lower=0).sum():.2f}")

    t.round(4).to_csv(os.path.join(_HERE, "results/legJ/table3_matched.csv"))

    rows = [("scm20d (strong)", "scm20d"), ("atp1d (strong)", "atp1d"),
            (r"rf1 ($R^2_{\mathrm{Indep}}{=}0.93$)", "rf1"),
            ("sarcos (ceiling)", "sarcos"), ("tecator (ceiling)", "tecator"),
            ("wq (low struct.)", "wq")]

    def f(v):
        return (r"$\phantom{+}0.00$" if abs(v) < 5e-3
                else (f"${v:+.2f}$".replace("+", r"\phantom{-}") if v > 0 else f"${v:.2f}$"))

    best = t[["Indep", "ICM", "ResICM", "Gate"]].mean().idxmin()
    lines = []
    for label, ds in rows:
        r = t.loc[ds]
        lines.append(f"{label} & {f(0.0)} & {f(r.ICM)} & {f(r.ResICM)} & {f(r.Gate)} \\\\")
    mn = t[["Indep", "ICM", "ResICM", "Gate"]].mean()
    def bold(v, col):
        s = f"{v:.2f}" if v < 0 else f"+{v:.2f}"
        return rf"$\mathbf{{{s}}}$" if col == best else f"${s}$"
    lines.append(r"\midrule")
    lines.append(r"\textbf{Mean $\dnll$} (16 sets) & $\phantom{+}0.00$ & "
                 + " & ".join(bold(mn[c], c) for c in ["ICM", "ResICM", "Gate"]) + r" \\")
    rk = ranks
    bestr = rk.idxmin()
    lines.append(r"\textbf{Mean rank} (1=best) & "
                 + " & ".join((rf"$\mathbf{{{rk[c]:.2f}}}$" if c == bestr else f"${rk[c]:.2f}$")
                              for c in ["Indep", "ICM", "ResICM", "Gate"]) + r" \\")
    print("\n=== LaTeX body for tab:method ===")
    print("\n".join(lines))
    with open(os.path.join(_HERE, "results/legJ/table3_matched.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nSaved results/legJ/table3_matched.{csv,tex}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "table":
        build_table()
    else:
        main()
