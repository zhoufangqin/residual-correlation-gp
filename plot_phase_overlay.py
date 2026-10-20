"""
Combined figure: synthetic phase diagram (background) + real-data points (foreground).

Design:
  Background — synthetic (T × p/n) pcolormesh, normalised to its own range so the
               subtle gradient is visible.
  Foreground — real datasets: circle colour = blue (ICM wins) / red (Indep wins);
               circle size ∝ |metric|; labelled with name + value.

Two panels: ΔNLL (left) and ΔR² (right).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
RDIR = os.path.join(HERE, "results")

# ── data ─────────────────────────────────────────────────────────────────────
syn  = pd.read_csv(os.path.join(RDIR, "phase/phase_prediction.csv"))
real = pd.read_csv(os.path.join(RDIR, "legJ/realdata_full.csv"))

P_DIM = dict(SH=252, WUR=150, rice=204, wheat=204, tecator=100, energy=8, jura=15,
             atp1d=411, atp7d=411, rf1=64, rf2=576, wq=16,
             scm1d=280, scm20d=61, edm=16, sarcos=21)

icm   = real[real.method == "ICM"].set_index("dataset")
indep = real[real.method == "Indep"].set_index("dataset")
rd = pd.DataFrame({
    "T":    icm["T"].astype(int),
    "n":    icm["n"].astype(int),
    "pn":   pd.Series({ds: P_DIM[ds] / int(icm.loc[ds, "n"]) for ds in icm.index}),
    "corr": icm["mean_target_corr"],
    "dNLL": icm["joint_nll"]     - indep["joint_nll"],
    "dR2":  icm["mean_r2"]       - indep["mean_r2"],
})

# ── aggregate synthetic grid ──────────────────────────────────────────────────
syn_agg = syn.groupby(["T", "pn"])[["dNLL", "dR2"]].mean().reset_index()
T_vals  = sorted(syn_agg["T"].unique())    # [2, 3, 5, 8]
pn_vals = sorted(syn_agg["pn"].unique())   # [0.27, 1.0, 4.0]

def build_grid(col):
    g = np.zeros((len(T_vals), len(pn_vals)))
    for i, T in enumerate(T_vals):
        for j, pn in enumerate(pn_vals):
            row = syn_agg[(syn_agg["T"] == T) & (syn_agg["pn"] == pn)]
            g[i, j] = row[col].values[0] if len(row) else np.nan
    return g

grid_nll = build_grid("dNLL")
grid_r2  = build_grid("dR2")

# ── pcolormesh bin edges (log10 p/n × T) ─────────────────────────────────────
log_pn    = np.log10(pn_vals)
log_edges = np.r_[log_pn[0] - (log_pn[1] - log_pn[0]) / 2,
                   (log_pn[:-1] + log_pn[1:]) / 2,
                   log_pn[-1] + (log_pn[-1] - log_pn[-2]) / 2]
T_arr     = np.array(T_vals, dtype=float)
T_edges   = np.r_[T_arr[0]  - (T_arr[1]  - T_arr[0])  / 2,
                   (T_arr[:-1] + T_arr[1:]) / 2,
                   T_arr[-1]  + (T_arr[-1]  - T_arr[-2]) / 2]

# ── per-dataset: (label_dx, label_dy, dot_T_jitter)
# label offsets in (log10_pn, T) space; dot_T_jitter shifts the plotted dot
# so overlapping pairs (atp1d/atp7d, same T and p/n) are visually separated.
OFFSETS = {
    #         label_dx  label_dy  dot_jitter_T
    "scm20d": (-0.30,   0.60,    0.0),
    "scm1d":  (  -0.26,  -1.15,    0.0),
    "atp1d":  ( 0.32,   0.65,   +0.30),
    "atp7d":  ( 0.42,  -0.85,   -0.30),
    "rf1":    (-0.25,   0.55,    0.0),
    "rf2":    ( 0.15,   0.65,    0.0),
    "wq":     (-0.25,   0.55,    0.0),
    "sarcos": (-0.32,   0.55,    0.0),
    "WUR":    ( -0.08,   0.95,    0.0),
    "SH":     ( 0.05,   0.95,    0.0),
    "rice":   (-0.22,  -0.85,   -0.15),
    "wheat":  ( 0.32,  -0.95,   +0.15),
    "energy": (-0.02,   0.85,    0.0),
    "jura":   (0.10,  0.35,    0.0),
    "edm":    ( 0.12,  -0.85,    0.0),
    "tecator":(-0.05,   0.75,    0.0),
}

# ── figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6.0))
fig.subplots_adjust(wspace=0.20)

PANELS = [
    # (grid,      col,   sign_for_bg, title,                          dot_scale)
    (grid_nll, "dNLL",  -1,
     r"$\Delta\mathrm{NLL}$  (ICM $-$ Indep;  negative = ICM better)",
     200),
    (grid_r2,  "dR2",   +1,
     r"$\Delta R^2$  (ICM $-$ Indep;  positive = ICM better)",
     60),
]

for ax, (grid, col, sign, title, dot_scale) in zip(axes, PANELS):
    # ── background: synthetic, normalised to its OWN data range so the full
    #    blue ramp is used (the gain is small in absolute terms, so a fixed
    #    wide clim washed the gradient out to near-white). ────────────────────
    bg_data = sign * grid   # +ve = ICM better
    bg_clim = float(np.nanmax(np.abs(bg_data)))
    norm_bg = TwoSlopeNorm(vmin=-bg_clim, vcenter=0.0, vmax=bg_clim)
    pcm = ax.pcolormesh(log_edges, T_edges, bg_data,
                        cmap="RdBu", norm=norm_bg, shading="flat", alpha=0.85)
    cb = plt.colorbar(pcm, ax=ax, pad=0.02, shrink=0.80,
                      label=f"Synthetic ICM benefit\n(±{bg_clim:.2f})", aspect=20)
    cb.ax.tick_params(labelsize=7)

    # ── real-data scatter ────────────────────────────────────────────────────
    for ds, (dx, dy, jT) in OFFSETS.items():
        row      = rd.loc[ds]
        logpn    = np.log10(float(row["pn"]))
        Tv       = float(row["T"]) + jT   # apply dot jitter
        val      = float(row[col])
        icm_wins = (sign * val) > 0        # True → blue

        # size: min 60, max 400, proportional to |val|
        vals_all = rd[col].abs()
        s = 60 + dot_scale * float(abs(val)) / float(vals_all.max() + 1e-9)
        s = np.clip(s, 60, 400)

        facecolor = "#2166ac" if icm_wins else "#d6604d"   # blue / red
        ax.scatter(logpn, Tv, s=s, color=facecolor,
                   edgecolors="k", linewidths=0.7, zorder=5)

        # label: name + value, with a thin connecting line
        tx, ty = logpn + dx, Tv + dy
        ax.annotate(f"{ds}\n({val:+.2f})",
                    xy=(logpn, Tv), xytext=(tx, ty),
                    fontsize=8.5, zorder=6, ha="center",
                    arrowprops=dict(arrowstyle="-", color="0.55", lw=0.5,
                                    connectionstyle="arc3,rad=0.1"))

    # ── axes & grid lines ────────────────────────────────────────────────────
    ax.set_xlabel(r"$p/n$  (feature dim / samples)", fontsize=12)
    ax.set_ylabel("$T$  (number of targets)", fontsize=12)
    ax.set_title(title, fontsize=12.5, pad=7)

    tick_log  = [np.log10(v) for v in [0.03, 0.27, 1.0, 4.0, 5.0]]
    tick_labs = ["0.03", "0.27", "1.0", "4.0", "5.0"]
    ax.set_xticks(tick_log);  ax.set_xticklabels(tick_labs, fontsize=10)
    ax.set_yticks([2, 3, 5, 6, 8, 14, 16])
    ax.set_yticklabels(["2", "3", "5", "6", "8", "14", "16"], fontsize=10)
    ax.set_ylim(0.5, 17.8)
    ax.set_xlim(np.log10(0.018), np.log10(8))
    ax.axvline(0.0, ls="--", color="0.40", lw=0.9)
    ax.text(-0.26, 17.4, "$p/n=1$", fontsize=7,
            ha="left", va="top", color="0.40")

    # ── legend ───────────────────────────────────────────────────────────────
    leg = [
        mpatches.Patch(facecolor="steelblue", alpha=0.4, label="Synthetic (bkgd)"),
        mpatches.Patch(facecolor="#2166ac",   ec="k",    label="Real: ICM better"),
        mpatches.Patch(facecolor="#d6604d",   ec="k",    label="Real: Indep better"),
    ]
    ax.legend(handles=leg, fontsize=7.5, loc="upper right",
              framealpha=0.88, handlelength=1.2, handletextpad=0.4)

# ── save ─────────────────────────────────────────────────────────────────────
out = os.path.join(RDIR, "phase/phase_overlay_combined")
plt.savefig(out + ".pdf", bbox_inches="tight")
plt.savefig(out + ".png", bbox_inches="tight", dpi=180)
print(f"Saved: {out}.pdf / .png")
