"""
MPII figure: residual-structure diagnostic D_logdet vs ICM joint-NLL benefit, on
frozen transformer/CNN image embeddings. Each point is one (encoder, keypoint-
subset) condition; marker size encodes T (number of coupled targets), colour the
encoder. Points below the dashed line (dNLL<0) are where ICM beats independent GPs.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "results/mpii/mpii_predict_calibrate.csv"))

ENC_LABEL = {
    "dinov2_vits14": "DINOv2 ViT-S", "dinov2_vitb14": "DINOv2 ViT-B",
    "dinov2_vitl14": "DINOv2 ViT-L", "vit_base_patch16_224": "ViT-B (sup.)",
    "vit_large_patch16_224": "ViT-L (sup.)", "vit_base_patch16_clip_224": "CLIP ViT-B",
    "resnet50": "ResNet-50", "convnext_base": "ConvNeXt-B",
    "swin_base_patch4_window7_224": "Swin-B", "deit3_base_patch16_224": "DeiT3-B",
}
colors = dict(zip(ENC_LABEL, plt.cm.tab10(np.linspace(0, 1, 10))))

rho, p = spearmanr(df["D_logdet"], df["dnll_icm"])

fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.axhline(0, ls="--", c="0.5", lw=1, zorder=1)
for enc, g in df.groupby("encoder"):
    ax.scatter(g["D_logdet"], g["dnll_icm"], s=18 + 6 * g["T"],
               color=colors[enc], alpha=0.8, edgecolor="k", linewidth=0.4,
               label=ENC_LABEL.get(enc, enc), zorder=3)
# trend line (least squares) for visual guidance
b, a = np.polyfit(df["D_logdet"], df["dnll_icm"], 1)
xs = np.linspace(df["D_logdet"].min(), df["D_logdet"].max(), 50)
ax.plot(xs, a + b * xs, c="0.3", lw=1.2, zorder=2)

ax.set_xlabel(r"$D_{\mathrm{logdet}} = -\frac{1}{2}\log\det\,R_{\mathrm{res}}$")
ax.set_ylabel(r"$\Delta\mathrm{NLL}$ (ICM $-$ Indep)")
ax.text(0.04, 0.06, rf"Spearman $\rho_s={rho:.2f}$ ($p={p:.1e}$)",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
ax.annotate("ICM better", (0.97, 0.02), xycoords="axes fraction", ha="right",
            va="bottom", fontsize=8, color="0.4")
ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9, title="encoder",
          title_fontsize=7.5, markerscale=0.5, ncol=2)
# marker-size legend for T
for T in (8, 16, 24):
    ax.scatter([], [], s=18 + 6 * T, c="0.6", edgecolor="k", linewidth=0.4,
               label=f"T={T}")
ax.text(0.5, 1.02, "marker size $\\propto T$ (coupled targets)",
        transform=ax.transAxes, ha="center", fontsize=7.5, color="0.4")
fig.tight_layout()
out = os.path.join(HERE, "results/mpii/mpii_dlogdet_scatter")
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
print("Spearman:", round(rho, 3), "p=", p)
print("Saved", out + ".pdf")
