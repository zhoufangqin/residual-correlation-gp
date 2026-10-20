"""
Does the "coupling helps uncertainty, not point prediction" finding survive
OUTSIDE the GP-ICM family?

In other multi-task paradigms, parameter sharing can improve prediction as
well, so we test
the coupling question in a parametric model that already shares parameters
across targets: a multi-output deep ensemble whose Gaussian head is either

  diag : predicts mean (T) and log-variance (T); joint covariance is diagonal
  full : predicts mean (T) and a Cholesky factor (T(T+1)/2); joint covariance is
         unrestricted, and the coupling is learned JOINTLY with the mean
  resid: the `diag` ensemble, with a residual correlation estimated post hoc from
         held-out standardised residuals and applied to the off-diagonal only

`resid` is the parametric analogue of Residual-ICM: it leaves the `diag` means
and marginal variances untouched, so its dR^2 is exactly 0 by construction and
any NLL change is attributable to cross-target covariance alone. `full` is the
analogue of ICM. Reporting both separates "can coupling help uncertainty here?"
from "does learning the coupling jointly disturb the mean fit?", which the
smoke test showed it can (scm1d dR^2 = -0.40 for `full`).

Each member holds out 10%% of its training split for early stopping; without it
the ensemble overfits n<=224 and its predictive variances collapse, which
inflates joint NLL for BOTH arms and makes the comparison meaningless.

Both share the identical trunk, ensemble size, and training schedule, so the ONLY
difference is whether the predictive distribution can express cross-target
covariance -- the same isolation as Indep vs ICM, but with parameter sharing
present in both arms.

Ensemble members are combined by moment matching. For the `diag` arm the
off-diagonal entries of the moment-matched covariance are zeroed, so that arm
really is a product of marginals (mirroring the independent GP); otherwise
disagreement between members would smuggle in cross-target covariance and the
comparison would no longer isolate the head.

Protocol matches the GP experiments: subsample 280 per seed, 5-fold CV,
standardised X and Y, joint NLL / mean R^2 from legJ_jointcalib.joint_metrics.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_hardening_ablation import load
from legJ_jointcalib import joint_metrics

DATASETS = ["atp1d", "atp7d", "rf1", "rf2", "WUR", "SH", "rice", "wheat",
            "jura", "wq", "energy", "tecator", "scm1d", "scm20d", "edm", "sarcos"]
from legJ_jointcalib import dataset_available  # noqa: E402
DATASETS = [d for d in DATASETS if dataset_available(d)]  # skip unavailable sets
SEEDS = [0, 1, 2]
M_ENSEMBLE, EPOCHS, HID, LR, WD = 5, 400, 128, 3e-3, 1e-4
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "results/deepens/deepens_coupling.csv"


class Net(nn.Module):
    def __init__(self, p, T, full):
        super().__init__()
        self.T, self.full = T, full
        self.trunk = nn.Sequential(nn.Linear(p, HID), nn.ReLU(),
                                   nn.Linear(HID, HID), nn.ReLU())
        self.mu = nn.Linear(HID, T)
        n_scale = T * (T + 1) // 2 if full else T
        self.scale = nn.Linear(HID, n_scale)
        nn.init.zeros_(self.scale.weight); nn.init.zeros_(self.scale.bias)
        if full:
            self.tril = torch.tril_indices(T, T)

    def forward(self, x):
        h = self.trunk(x)
        mu, s = self.mu(h), self.scale(h)
        if not self.full:
            return mu, torch.diag_embed(torch.nn.functional.softplus(s) + 1e-3)
        L = x.new_zeros(len(x), self.T, self.T)
        L[:, self.tril[0], self.tril[1]] = s
        d = torch.arange(self.T)
        L[:, d, d] = torch.nn.functional.softplus(L[:, d, d].clone()) + 1e-3
        return mu, L


def nll_loss(mu, S, y, full):
    r = (y - mu).unsqueeze(-1)
    if full:
        z = torch.linalg.solve_triangular(S, r, upper=False)
        logdet = torch.log(torch.diagonal(S, dim1=1, dim2=2)).sum(1)
    else:
        sd = torch.diagonal(S, dim1=1, dim2=2)
        z = r.squeeze(-1) / sd
        logdet = torch.log(sd).sum(1)
    return (0.5 * (z ** 2).sum((1, 2) if full else 1) + logdet).mean()


def fit_predict(Xtr, Ytr, Xte, full, seed, val_frac=0.1, patience=40):
    """Train an ensemble with early stopping; return moment-matched mean/cov on
    Xte, and the same for the training split's held-out validation points (used
    by the `resid` arm to estimate the residual correlation without leakage)."""
    p, T = Xtr.shape[1], Ytr.shape[1]
    rng = np.random.default_rng(seed)
    nval = max(8, int(val_frac * len(Xtr)))
    perm = rng.permutation(len(Xtr))
    vi, ti = perm[:nval], perm[nval:]
    xt = torch.tensor(Xtr[ti], dtype=torch.float32, device=DEV)
    yt = torch.tensor(Ytr[ti], dtype=torch.float32, device=DEV)
    xv = torch.tensor(Xtr[vi], dtype=torch.float32, device=DEV)
    yv = torch.tensor(Ytr[vi], dtype=torch.float32, device=DEV)
    xe = torch.tensor(Xte, dtype=torch.float32, device=DEV)

    mus, covs, vmus, vcovs = [], [], [], []
    for m in range(M_ENSEMBLE):
        torch.manual_seed(1000 * seed + 17 * m + int(full))
        net = Net(p, T, full).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
        best, best_state, bad = np.inf, None, 0
        for ep in range(EPOCHS):
            net.train(); opt.zero_grad()
            mu, S = net(xt)
            nll_loss(mu, S, yt, full).backward()
            opt.step()
            if ep % 5 == 0:
                net.eval()
                with torch.no_grad():
                    v = float(nll_loss(*net(xv), yv, full))
                if v < best - 1e-4:
                    best, bad = v, 0
                    best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience // 5:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            for src, ms, cs in ((xe, mus, covs), (xv, vmus, vcovs)):
                mu, S = net(src)
                cov = S @ S.transpose(1, 2) if full else S ** 2
                ms.append(mu.cpu().numpy()); cs.append(cov.cpu().numpy())

    def combine(ms, cs, zero_offdiag):
        ms, cs = np.stack(ms), np.stack(cs)
        mu_bar = ms.mean(0)
        spread = ms - mu_bar
        cov = cs.mean(0) + np.einsum("mnt,mns->nts", spread, spread) / len(ms)
        if zero_offdiag:
            d = np.arange(cov.shape[1])
            keep = cov[:, d, d].copy()
            cov = np.zeros_like(cov)
            cov[:, d, d] = keep
        return mu_bar, cov

    mu_te, cov_te = combine(mus, covs, not full)
    mu_va, cov_va = combine(vmus, vcovs, not full)
    return mu_te, cov_te, (Ytr[vi], mu_va, cov_va)


def _matched_corr(y, mu, cov):
    """Residual correlation from held-out residuals standardised by predictive sd
    (the matched estimator adopted in exp_residual_icm.py)."""
    T = y.shape[1]
    sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 1e-12))
    z = (y - mu) / sd
    R = np.atleast_2d(np.corrcoef(z.T))
    R = 0.5 * (R + R.T)
    lam = min(1.0, T / max(len(z), 1))
    R = (1 - lam) * R + lam * np.eye(T)
    w, V = np.linalg.eigh(R)
    R = (V * np.clip(w, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def run(name, seed):
    X, Y = load(name)
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    if len(Xs) > 280:
        idx = np.random.default_rng(seed).choice(len(Xs), 280, replace=False)
        Xs, Ys = Xs[idx], Ys[idx]
    n, T = Ys.shape
    P = {k: (np.zeros((n, T)), np.zeros((n, T, T)))
         for k in ("diag", "full", "resid")}
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xs):
        for k, full in (("diag", False), ("full", True)):
            mu, cov, val = fit_predict(Xs[tr], Ys[tr], Xs[te], full, seed)
            P[k][0][te], P[k][1][te] = mu, cov
            if k == "diag":
                R = _matched_corr(*val)
                sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 1e-12))
                P["resid"][0][te] = mu
                P["resid"][1][te] = sd[:, :, None] * R[None] * sd[:, None, :]
    M = {k: joint_metrics(Ys, *P[k]) for k in P}
    b = M["diag"]
    return dict(dataset=name, seed=seed, T=T,
                nll_diag=b["joint_nll"], nll_full=M["full"]["joint_nll"],
                nll_resid=M["resid"]["joint_nll"],
                dnll_full=round(M["full"]["joint_nll"] - b["joint_nll"], 4),
                dnll_resid=round(M["resid"]["joint_nll"] - b["joint_nll"], 4),
                r2_diag=b["mean_r2"], r2_full=M["full"]["mean_r2"],
                dr2_full=round(M["full"]["mean_r2"] - b["mean_r2"], 4))


def main():
    os.makedirs("results/deepens", exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.dataset, prev.seed))
        print(f"resuming: {len(done)} cells")
    for d in DATASETS:
        for s in SEEDS:
            if (d, s) in done:
                continue
            r = run(d, s)
            pd.DataFrame([r]).to_csv(OUT, mode="a", header=not os.path.exists(OUT),
                                     index=False)
            print(f"[done] {d:8s} seed={s} T={r['T']:2d} "
                  f"dNLL_full={r['dnll_full']:+.3f} dNLL_resid={r['dnll_resid']:+.3f} "
                  f"dR2_full={r['dr2_full']:+.4f}", flush=True)
    summarize()


def summarize():
    from scipy import stats
    df = pd.read_csv(OUT)
    g = df.groupby("dataset")[["dnll_full", "dnll_resid", "dr2_full", "T"]].mean()
    diag = pd.read_csv("results/legJ/residual_icm_multiseed_summary.csv").set_index("dataset")
    # D_logdet on predictive-standardised residuals (R_z), matching the rest of
    # the analysis; the multiseed summary predates that change.
    dz = pd.read_csv("results/legJ/rz_sweep_inputs.csv", index_col=0)
    g = g.join(diag[["ICM_dnll_mean"]])
    g["D_logdet_mean"] = dz["D_z"].reindex(g.index)
    print("\n=== deep ensemble: coupled heads vs diagonal head ===")
    print(g.round(3).to_string())
    for c in ("dnll_full", "dnll_resid"):
        print(f"\n{c}: mean {g[c].mean():+.3f}, improved on {(g[c] < 0).sum()}/{len(g)}")
    print(f"dr2_full : mean {g.dr2_full.mean():+.4f}, "
          f"reduced on {(g.dr2_full < 0).sum()}/{len(g)}, "
          f"|dR2| < 0.01 on {(g.dr2_full.abs() < 0.01).sum()}/{len(g)}   "
          f"(resid arm is 0 by construction)")
    # NOTE: the "reduced on N/16" count is on SEED-MEANS. Per seed it is
    # 12, 14, 12 of 16, so a single-seed run reports 14/16; the paper quotes
    # the seed-mean count (12/16) to match the seed-mean dR2 printed beside it.
    ok = g.dropna(subset=["D_logdet_mean"])
    for c in ("dnll_full", "dnll_resid"):
        r = stats.spearmanr(ok.D_logdet_mean, ok[c])
        print(f"Spearman(D_logdet, {c}) = {r.statistic:+.3f} (p={r.pvalue:.4f})")
    r = stats.spearmanr(ok.ICM_dnll_mean, ok.dnll_resid)
    print(f"Spearman(GP-ICM dNLL, dnll_resid) = {r.statistic:+.3f} (p={r.pvalue:.4f})")
    g.round(4).to_csv("results/deepens/summary.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
    else:
        main()
