"""
Synthetic prediction phase diagram: when does coregionalization (ICM) beat
independent GPs on multi-target point prediction, as a function of the number of
targets T and the feature/sample ratio p/n?

Controlled generator: r shared latent factors -> T correlated targets (loadings L),
and a nonlinear lift of the factors to p observed features (intrinsic dim stays r
while p grows -- models high-dim spectra with low intrinsic dimension). Per-target
noise makes single-target signal weak so sharing across targets can pay off.

For each (T, p/n) cell we run the hardened Leg J pipeline (shared robust kernel,
+coupling) and record ICM-minus-Indep joint R^2 and NLL. Expected: the prediction
benefit concentrates at LARGE T and LARGE p/n (the data-scarce, many-target
regime) and vanishes at T=2 -- the synthetic analogue of atp1d/atp7d (win) vs the
2-trait HSI (no win).
"""
import itertools
import os

import numpy as np
import pandas as pd

from legJ_jointcalib import run_dataset


def gen(n, T, p, r=3, noise=0.6, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, r))                 # shared latent factors
    L = rng.standard_normal((T, r))                 # heterogeneous loadings
    Win = rng.standard_normal((r, p)) / np.sqrt(r)
    X = np.tanh(z @ Win)                            # nonlinear lift to p dims
    Yc = z @ L.T
    Yc = (Yc - Yc.mean(0)) / (Yc.std(0) + 1e-9)
    Y = Yc + noise * rng.standard_normal((n, T))
    return X, Y


def main():
    n = 150
    Ts = [2, 3, 5, 8]
    ps = [40, 150, 600]                            # p/n = 0.27, 1.0, 4.0
    seeds = range(5)
    rows = []
    for T, p, s in itertools.product(Ts, ps, seeds):
        X, Y = gen(n, T, p, seed=s)
        r = run_dataset(f"T{T}p{p}", X, Y, n_sub=10000)   # no subsample (n=150)
        icm = next(d for d in r if d["method"] == "ICM")
        ind = next(d for d in r if d["method"] == "Indep")
        rows.append(dict(T=T, p=p, pn=round(p / n, 2), seed=s,
                         corr=icm["mean_target_corr"],
                         dR2=icm["mean_r2"] - ind["mean_r2"],
                         dNLL=icm["joint_nll"] - ind["joint_nll"],
                         icm_r2=icm["mean_r2"], ind_r2=ind["mean_r2"]))
        print(f"T={T} p={p} pn={p/n:.2f} seed={s}: "
              f"dR2={rows[-1]['dR2']:+.3f} dNLL={rows[-1]['dNLL']:+.3f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("results/phase", exist_ok=True)
    df.to_csv("results/phase/phase_prediction.csv", index=False)

    gR2 = df.groupby(["T", "pn"])["dR2"].mean().unstack("pn")
    gNLL = df.groupby(["T", "pn"])["dNLL"].mean().unstack("pn")
    pd.set_option("display.width", 140)
    print("\n=== mean ICM-minus-Indep R^2 over (T rows x p/n cols) ===")
    print(gR2.round(3).to_string())
    print("\n=== mean ICM-minus-Indep NLL (negative=ICM better) ===")
    print(gNLL.round(3).to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, G, title, cmap in [(axes[0], gR2, "ICM$-$Indep  joint $R^2$  (higher=coupling helps)", "RdBu_r"),
                                   (axes[1], -gNLL, "Indep$-$ICM  joint NLL  (higher=coupling helps)", "RdBu_r")]:
            vmax = np.nanmax(np.abs(G.values))
            im = ax.imshow(G.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
            ax.set_xticks(range(len(G.columns))); ax.set_xticklabels(G.columns)
            ax.set_yticks(range(len(G.index))); ax.set_yticklabels(G.index)
            ax.set_xlabel("p / n"); ax.set_ylabel("T (number of targets)")
            ax.set_title(title, fontsize=10)
            for i in range(G.shape[0]):
                for j in range(G.shape[1]):
                    ax.text(j, i, f"{G.values[i,j]:+.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle("Prediction phase diagram: coregionalization helps at large T and large p/n", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig("results/phase/phase_prediction.pdf")
        fig.savefig("results/phase/phase_prediction.png", dpi=130)
        print("\nsaved results/phase/phase_prediction.{pdf,png}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
