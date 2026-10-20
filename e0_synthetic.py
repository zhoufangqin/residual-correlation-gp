"""
E0 -- Synthetic correlated-target backbone (Gate A de-risk).

Question this answers (RQ2, de-risked before any real data):
    A single-target encoder is trained on y1 ONLY. Does its frozen latent Phi
    retain recoverable information about a *correlated* target y2 -- and is that
    information recoverable by a nonlinear GP while a linear probe misses it?

Two controllable knobs:
    rho  -- target correlation corr(y1, y2)            (shared, linear signal)
    lam  -- nonlinearity of y2's *own* (non-shared) part in latent space

Predicted pattern (the thing Gate A must confirm):
    1. Recovery R^2(y2) rises with rho for ALL methods (shared linear signal).
    2. The GP-minus-linear gap rises with lam: the nonlinear own-part of y2 is
       recoverable by an RBF GP but invisible to a linear probe / LogME.
    3. At lam=0 (linear), GP ~= linear probe.
    4. (Bonus) a y1-TRAINED encoder retains nearly as much y2 signal as a RANDOM
       encoder -> single-target supervision does not destroy correlated info.

No new algorithm. Same instrument (GP on frozen features) vs the baseline panel.
Pure numpy/sklearn; exact GP via sklearn GaussianProcessRegressor.
"""

import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The optimal value found for dimension")


# --------------------------------------------------------------------------- #
# Data generator
# --------------------------------------------------------------------------- #
def _std(v):
    """Zero-mean, unit-variance."""
    v = v - v.mean()
    s = v.std()
    return v / s if s > 1e-12 else v


def generate(n, rho, lam, m=8, D=64, noise=0.1, seed=0):
    """
    z ~ N(0, I_m)            latent factors
    X = z @ A                linear lift to D dims (the 'raw input' the encoder sees)
    y1 = std(z @ u1) + noise                                  (linear in z)
    own2 = (1-lam)*std(z @ u2)  +  lam*std(quadratic(z))      (u2 _|_ u1)
    y2  = rho*std(z @ u1) + sqrt(1-rho^2)*own2 + noise

    own2 is uncorrelated with the y1 signal (u2 _|_ u1; centered quadratic has
    zero covariance with a linear form under Gaussian z), so corr(y1,y2) ~= rho.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, m))

    A = rng.standard_normal((m, D)) / np.sqrt(m)
    X = z @ A

    # y1 direction and an orthogonal y2 linear direction
    u1 = rng.standard_normal(m)
    u1 /= np.linalg.norm(u1)
    u2 = rng.standard_normal(m)
    u2 = u2 - (u2 @ u1) * u1          # orthogonalize against u1
    u2 /= np.linalg.norm(u2)

    s1 = _std(z @ u1)                 # shared linear signal

    # nonlinear own-part: random quadratic form, centered -> ortho to linear s1
    R = rng.standard_normal((m, 4)) / np.sqrt(m)
    quad = _std((z @ R) ** 2 @ np.ones(4))
    own2 = (1.0 - lam) * _std(z @ u2) + lam * quad
    own2 = _std(own2)

    y1 = s1 + noise * rng.standard_normal(n)
    y2 = rho * s1 + np.sqrt(max(1.0 - rho**2, 0.0)) * own2 + noise * rng.standard_normal(n)
    return X, y1, y2


# --------------------------------------------------------------------------- #
# Encoder: 1-hidden-layer net. Phi = tanh(X W1 + b1).
#   'trained' -> W1 copied from an MLP trained on y1 only (the real phenomenon)
#   'random'  -> W1 sampled randomly (retention guaranteed; isolates readout)
# --------------------------------------------------------------------------- #
def encode(X, y1, mode, H=64, seed=0):
    rng = np.random.default_rng(seed + 9999)
    if mode == "raw":
        return X
    if mode == "random":
        W1 = rng.standard_normal((X.shape[1], H)) / np.sqrt(X.shape[1])
        b1 = rng.standard_normal(H) * 0.1
    elif mode == "trained":
        net = MLPRegressor(hidden_layer_sizes=(H,), activation="tanh",
                           max_iter=800, random_state=seed, alpha=1e-3)
        net.fit(X, y1)
        W1, b1 = net.coefs_[0], net.intercepts_[0]
    else:
        raise ValueError(mode)
    return np.tanh(X @ W1 + b1)


# --------------------------------------------------------------------------- #
# Recovery heads (predict y2 from frozen Phi) + LogME selection score
# --------------------------------------------------------------------------- #
def _gp():
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e2)) \
        + WhiteKernel(0.1, (1e-4, 1e1))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=2, alpha=1e-6)


def recover(Phi_tr, Phi_te, y_tr, y_te):
    sc = StandardScaler().fit(Phi_tr)
    Ztr, Zte = sc.transform(Phi_tr), sc.transform(Phi_te)
    out = {}

    out["linear"] = r2_score(y_te, RidgeCV(alphas=np.logspace(-3, 3, 13))
                             .fit(Ztr, y_tr).predict(Zte))

    mlp = MLPRegressor(hidden_layer_sizes=(64,), activation="relu", max_iter=1500,
                       random_state=0, alpha=1e-3, early_stopping=True)
    out["mlp"] = r2_score(y_te, mlp.fit(Ztr, y_tr).predict(Zte))

    out["gp"] = r2_score(y_te, _gp().fit(Ztr, y_tr).predict(Zte))
    return out


def logme(F, y, max_iter=500, tol=1e-6):
    """LogME (You et al., 2021): per-target log maximum evidence of the linear-
    Gaussian model y = F w + e,  w ~ N(0, a^-1 I),  e ~ N(0, b^-1), maximized
    over (a, b) by the standard MacKay evidence fixed point. Returns evidence/n.

    Uses the economy SVD F = U diag(s) V^T; eig(F^T F) = s^2. The linear-kernel
    GP's log marginal likelihood equals this evidence up to constants -- the
    'GP evidence : LogME :: nonlinear probe : linear probe' positioning line.
    """
    F = StandardScaler().fit_transform(F)
    n, D = F.shape
    y = y - y.mean()
    U, s, _ = np.linalg.svd(F, full_matrices=False)
    sig = s ** 2                       # eigenvalues of F^T F (len r = min(n,D))
    r = len(sig)
    z = U.T @ y
    z2 = z ** 2
    res_out = float(y @ y - z2.sum())  # energy of y outside span(U)

    a, b = 1.0, 1.0
    for _ in range(max_iter):
        denom = a + b * sig
        gamma = float((b * sig / denom).sum())
        m2 = float((b ** 2 * sig * z2 / denom ** 2).sum())          # ||m||^2
        res_in = float(((a / denom) ** 2 * z2).sum())               # in-span resid
        res = res_in + res_out
        a_new = gamma / (m2 + 1e-12)
        b_new = (n - gamma) / (res + 1e-12)
        if abs(a_new - a) < tol and abs(b_new - b) < tol:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new

    denom = a + b * sig
    m2 = float((b ** 2 * sig * z2 / denom ** 2).sum())
    res = float(((a / denom) ** 2 * z2).sum()) + res_out
    logdetA = float(np.log(denom).sum() + (D - r) * np.log(a))
    L = (0.5 * D * np.log(a) + 0.5 * n * np.log(b)
         - 0.5 * b * res - 0.5 * a * m2
         - 0.5 * logdetA - 0.5 * n * np.log(2 * np.pi))
    return float(L / n)


# --------------------------------------------------------------------------- #
# Experiment grid
# --------------------------------------------------------------------------- #
def run(rhos, lams, ns, encoders, seeds, test_frac=0.3):
    rows = []
    for rho, lam, n, enc, seed in itertools.product(rhos, lams, ns, encoders, seeds):
        X, y1, y2 = generate(n, rho, lam, seed=seed)
        Phi = encode(X, y1, enc, seed=seed)

        rng = np.random.default_rng(1000 + seed)
        idx = rng.permutation(n)
        n_te = int(round(n * test_frac))
        te, tr = idx[:n_te], idx[n_te:]

        r2 = recover(Phi[tr], Phi[te], y2[tr], y2[te])
        rows.append(dict(
            rho=rho, lam=lam, n=n, encoder=enc, seed=seed,
            emp_corr=float(np.corrcoef(y1, y2)[0, 1]),
            r2_linear=r2["linear"], r2_mlp=r2["mlp"], r2_gp=r2["gp"],
            logme_y2=logme(Phi[tr], y2[tr]),
        ))
    return pd.DataFrame(rows)


def summarize(df):
    g = (df.groupby(["encoder", "lam", "rho"])
           [["r2_linear", "r2_mlp", "r2_gp", "emp_corr"]]
           .mean().reset_index())
    g["gp_minus_linear"] = g["r2_gp"] - g["r2_linear"]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grid for a smoke test")
    ap.add_argument("--out", default="results/e0/e0_synthetic.csv")
    args = ap.parse_args()

    if args.quick:
        rhos, lams, ns, encoders, seeds = [0.0, 0.9], [0.0, 1.0], [120], ["random"], [0, 1]
    else:
        rhos = [0.0, 0.3, 0.6, 0.9]
        lams = [0.0, 0.5, 1.0]
        ns = [100, 300]
        encoders = ["trained", "random", "raw"]
        seeds = [0, 1, 2, 3, 4]

    df = run(rhos, lams, ns, encoders, seeds)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    summ = summarize(df)
    pd.set_option("display.width", 140, "display.max_rows", 200)
    print("\n=== Mean over seeds & n (recovery R^2 of y2 from frozen Phi) ===")
    print(summ.round(3).to_string(index=False))

    # ---- headline checks (printed verdicts) ----
    print("\n=== GATE A CHECKS ===")
    enc_for_check = "trained" if "trained" in summ.encoder.values else summ.encoder.iloc[0]
    tr = summ[summ.encoder == enc_for_check]
    print(f"(encoder used for checks: {enc_for_check})")

    # (1) recovery rises with rho (GP), averaged over lam
    by_rho = tr.groupby("rho")["r2_gp"].mean()
    inc_rho = by_rho.is_monotonic_increasing
    print(f"(1) GP recovery rises with rho (trained enc): {by_rho.round(3).to_dict()} "
          f"-> {'PASS' if inc_rho else 'CHECK'}")

    # (2) GP-minus-linear gap rises with lam
    by_lam = tr.groupby("lam")["gp_minus_linear"].mean()
    inc_lam = by_lam.is_monotonic_increasing and by_lam.iloc[-1] > 0.03
    print(f"(2) GP-minus-linear gap rises with lam: {by_lam.round(3).to_dict()} "
          f"-> {'PASS' if inc_lam else 'CHECK'}")

    # (3) at lam=0, GP ~= linear (gap small)
    gap0 = tr[tr.lam == 0.0]["gp_minus_linear"].mean()
    print(f"(3) at lam=0, GP~=linear (gap={gap0:.3f}) "
          f"-> {'PASS' if abs(gap0) < 0.06 else 'CHECK'}")

    # (4) trained retains ~ as much as random (nonlinear cell, high rho)
    cell = (summ.lam == 1.0) & (summ.rho == 0.9)
    rt = summ[cell & (summ.encoder == "trained")]["r2_gp"].mean()
    rr = summ[cell & (summ.encoder == "random")]["r2_gp"].mean()
    print(f"(4) trained vs random GP recovery @ (lam=1,rho=.9): "
          f"trained={rt:.3f} random={rr:.3f} -> "
          f"{'PASS' if rt > 0.5 * rr and rt > 0.1 else 'CHECK'}")

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
