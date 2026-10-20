"""
Appendix robustness: does a shrinkage residual-covariance estimator restore the
resolution that D_logdet loses on highly collinear targets (the AFLW saturation)?

Mechanism in a controlled setting. We draw n residual vectors of dimension T from
an equicorrelation population R_pop(c) = (1-c) I + c 11^T (off-diagonal = c), so
the *true* per-sample full-vs-diagonal NLL gain is the closed form
    D_pop(c) = -1/2 log det R_pop(c)
             = -1/2 [ log(1 + (T-1) c) + (T-1) log(1-c) ],
which is finite and monotone increasing in c. We compare two estimators of
D_logdet from the sampled residuals:
  plain : empirical correlation, eigenvalues clipped at 1e-6, det floored at 1e-12
          (the estimator used in the main paper)
  LW    : Ledoit-Wolf shrinkage covariance -> correlation -> -1/2 log det
At n < T the sample correlation matrix is rank-deficient, so the plain estimator
saturates at its numerical floor for a whole range of c (no resolution), whereas
LW stays finite and tracks D_pop monotonically.

Outputs: results/shrinkage/synthetic_collinearity.{csv,pdf,png}
"""
import os
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T       = 24                                   # face-landmark-scale target count
N_REGS  = [15, 60]                             # n<T (rank-deficient) and n>T
CS      = np.round(np.linspace(0.0, 0.95, 13), 3)
SEEDS   = range(20)
CLIP    = 1e-6
DET_FLR = 1e-12


def equicorr(c, T):
    return (1.0 - c) * np.eye(T) + c * np.ones((T, T))


def d_pop(c, T):
    return -0.5 * (np.log(1.0 + (T - 1) * c) + (T - 1) * np.log(max(1.0 - c, 1e-300)))


def plain_dlogdet(resid):
    """Main-paper estimator: empirical corr, eigen-clip, det floor."""
    R = np.atleast_2d(np.corrcoef(resid.T))
    R = 0.5 * (R + R.T)
    w, V = np.linalg.eigh(R)
    w = np.clip(w, CLIP, None)
    R = (V * w) @ V.T
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    return float(-0.5 * np.log(max(np.linalg.det(R), DET_FLR)))


def lw_dlogdet(resid):
    """Ledoit-Wolf shrinkage covariance -> correlation -> -1/2 log det."""
    cov = LedoitWolf().fit(resid).covariance_
    d = np.sqrt(np.diag(cov))
    R = cov / np.outer(d, d)
    sign, logdet = np.linalg.slogdet(R)
    return float(-0.5 * logdet)


def main():
    rng0 = np.random.default_rng(0)
    rows = []
    for n in N_REGS:
        for c in CS:
            R_pop = equicorr(c, T)
            L = np.linalg.cholesky(R_pop + 1e-9 * np.eye(T))
            for s in SEEDS:
                rng = np.random.default_rng(1000 * n + int(c * 1000) + s)
                resid = rng.standard_normal((n, T)) @ L.T
                rows.append(dict(n=n, c=c, seed=s,
                                 d_plain=plain_dlogdet(resid),
                                 d_lw=lw_dlogdet(resid),
                                 d_pop=d_pop(c, T)))
    df = pd.DataFrame(rows)
    os.makedirs("results/shrinkage", exist_ok=True)
    df.to_csv("results/shrinkage/synthetic_collinearity.csv", index=False)

    g = df.groupby(["n", "c"]).agg(
        plain_m=("d_plain", "mean"), plain_s=("d_plain", "std"),
        lw_m=("d_lw", "mean"), lw_s=("d_lw", "std"),
        popm=("d_pop", "mean")).reset_index()

    print("=== mean D_logdet by (n, c): plain vs LW vs population ===")
    for n in N_REGS:
        sub = g[g.n == n]
        print(f"\n n={n} (T={T}, {'rank-deficient n<T' if n < T else 'n>T'}):")
        print(sub[["c", "plain_m", "plain_s", "lw_m", "lw_s", "popm"]]
              .round(2).to_string(index=False))

    fig, axes = plt.subplots(1, len(N_REGS), figsize=(5.4 * len(N_REGS), 4.0),
                             sharey=False)
    for ax, n in zip(axes, N_REGS):
        sub = g[g.n == n]
        ax.plot(sub["c"], sub["popm"], "k--", lw=1.6,
                label=r"true $-\frac{1}{2}\log\det R$")
        ax.errorbar(sub["c"], sub["plain_m"], yerr=sub["plain_s"], fmt="-o", ms=4,
                    color="#d6604d", capsize=2, label="plain (paper estimator)")
        ax.errorbar(sub["c"], sub["lw_m"], yerr=sub["lw_s"], fmt="-s", ms=4,
                    color="#2166ac", capsize=2, label="Ledoit-Wolf shrinkage")
        reg = "rank-deficient" if n < T else "well-sampled"
        ax.set_title(f"$n={n}$, $T={T}$  ({reg})", fontsize=12)
        ax.set_xlabel("residual collinearity $c$", fontsize=12)
        ax.set_ylabel(r"$D_{\mathrm{logdet}}$", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9.5, loc="upper left")
    fig.suptitle("Shrinkage restores $D_{\\mathrm{logdet}}$ resolution under high "
                 "residual collinearity", fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("results/shrinkage/synthetic_collinearity.pdf", bbox_inches="tight")
    fig.savefig("results/shrinkage/synthetic_collinearity.png", dpi=150,
                bbox_inches="tight")
    print("\nsaved results/shrinkage/synthetic_collinearity.{pdf,png}")

    # quantitative: resolution = Spearman(estimate, c) in the high-collinearity tail
    from scipy.stats import spearmanr
    print("\n=== resolution in the high-collinearity tail (c >= 0.6) ===")
    for n in N_REGS:
        sub = df[(df.n == n) & (df.c >= 0.6)]
        sp_plain = spearmanr(sub.c, sub.d_plain).correlation
        sp_lw = spearmanr(sub.c, sub.d_lw).correlation
        print(f" n={n}: Spearman(c, plain)={sp_plain:+.2f}  "
              f"Spearman(c, LW)={sp_lw:+.2f}")


if __name__ == "__main__":
    main()
