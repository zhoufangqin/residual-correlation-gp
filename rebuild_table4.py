"""Rebuild tab:method (Table 4) with the matched estimator and the LODO-tau gate.

Also carries a rank-(T-1) ICM column: rank-1 ICM is the weakest configuration
in the sweep, so a higher-rank comparator belongs in the main table rather than
only in the appendix.

Columns:
  Indep  : 0 by definition
  ICM    : published (unaffected by the estimator switch)
  ResICM : matched (standardised-residual) estimator, from resicm_matched_long.csv.
           SH and WUR keep their published values (they are small, not
           redistributed, and their GP fits are BLAS-threading-sensitive).
  Gate   : leave-one-dataset-out tau, reconstructed below. For each held-out
           dataset, tau is chosen on the other 15 to minimise mean gated dNLL,
           then applied to the held-out one; ceiling R^2_Indep < 0.9 throughout.
           This reproduces every Gate cell the paper prints and its mean (-0.61),
           unlike the fixed-tau=0.5 gate in exp_residual_icm_multiseed.py, which
           differs on wq.
"""
import numpy as np
import pandas as pd

pub = pd.read_csv("results/legJ/residual_icm_multiseed_summary.csv").set_index("dataset")
new = pd.read_csv("results/legJ/resicm_matched_long.csv")
matched = new[new.method == "ResICM"].groupby("dataset").dnll.mean()
KEEP_PUBLISHED = ["SH", "WUR"]


def lodo_gate(D, R2, ICM, ceiling=0.9):
    grid = np.unique(np.concatenate([[0.0], np.sort(D), [D.max() + 1]]))
    g = lambda t: np.where((D > t) & (R2 < ceiling), ICM, 0.0)
    out = np.zeros(len(D))
    for i in range(len(D)):
        m = np.ones(len(D), bool); m[i] = False
        tau = min(grid, key=lambda t: g(t)[m].mean())
        out[i] = g(tau)[i]
    return out


rank = pd.read_csv("results/legJ/hardening_multiseed.csv")
rank = rank[rank.variant == "hard_rfull"].groupby("dataset").dnll.mean()

t = pd.DataFrame({"Indep": 0.0, "ICM": pub.ICM_dnll_mean,
                  "ICMfull": rank.reindex(pub.index),
                  "ResICM_old": pub.ResICM_dnll_mean})
t["ResICM"] = matched.reindex(t.index)
t.loc[KEEP_PUBLISHED, "ResICM"] = t.loc[KEEP_PUBLISHED, "ResICM_old"]
t["Gate"] = lodo_gate(pub.D_logdet_mean.values, pub.r2_indep_mean.values,
                      pub.ICM_dnll_mean.values)

cols = ["Indep", "ICM", "ICMfull", "ResICM", "Gate"]
mn, ranks = t[cols].mean(), t[cols].rank(axis=1).mean()
old_ranks = t[["Indep", "ICM", "ResICM_old", "Gate"]].rank(axis=1).mean()
print(t[["ICM", "ICMfull", "ResICM_old", "ResICM", "Gate"]].round(3).to_string())
print("\nmeans:"); print(t[["ICM", "ICMfull", "ResICM_old", "ResICM", "Gate"]].mean().round(3).to_string())
print("\nmean rank, published ResICM -> matched:")
for c in cols:
    o = {"ResICM": "ResICM_old"}.get(c, c)
    prev = f"{old_ranks[o]:.2f}" if o in old_ranks.index else " -- "
    print(f"  {c:8s} 4-col {prev} -> 5-col {ranks[c]:.2f}")
for c in ("ICM", "ICMfull", "ResICM", "Gate"):
    x = t[c]
    print(f"\n{c:7s} harmed {(x > 0).sum()}/16, worst {x.max():+.3f}, "
          f"total harm {x.clip(lower=0).sum():.2f}")
t.round(4).to_csv("results/legJ/table4_matched.csv")

rows = [("scm20d (strong)", "scm20d"), ("atp1d (strong)", "atp1d"),
        (r"rf1 ($R^2_{\mathrm{Indep}}{=}0.93$)", "rf1"),
        ("sarcos (ceiling)", "sarcos"), ("tecator (ceiling)", "tecator"),
        ("wq (low struct.)", "wq")]
ZERO = r"$\phantom{+}0.00$"
f = lambda v: ZERO if abs(v) < 5e-3 else (rf"$+{v:.2f}$" if v > 0 else f"${v:.2f}$")
best, bestr = mn.idxmin(), ranks.idxmin()
lines = [f"{lab} & {ZERO} & {f(t.loc[ds,'ICM'])} & {f(t.loc[ds,'ICMfull'])} "
         f"& {f(t.loc[ds,'ResICM'])} & {f(t.loc[ds,'Gate'])} \\\\" for lab, ds in rows]
lines.append(r"\midrule")
lines.append(r"\textbf{Mean $\dnll$} (16 sets) & " + ZERO + " & "
             + " & ".join((rf"$\mathbf{{{mn[c]:.2f}}}$" if c == best else f"${mn[c]:.2f}$")
                          for c in ["ICM", "ICMfull", "ResICM", "Gate"]) + r" \\")
lines.append(r"\textbf{Mean rank} (1=best) & "
             + " & ".join((rf"$\mathbf{{{ranks[c]:.2f}}}$" if c == bestr else f"${ranks[c]:.2f}$")
                          for c in cols) + r" \\")
print("\n=== LaTeX body ===")
print("\n".join(lines))
open("results/legJ/table4_matched.tex", "w").write("\n".join(lines) + "\n")
