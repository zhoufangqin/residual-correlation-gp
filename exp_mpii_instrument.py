"""
COCO keypoint instrument: real-image test of (A) the calibration mechanism at
large T and (B) selection generality, from cached person-crop embeddings
(embed_coco.py).

Keypoint subsets of varying size/structure form the conditions; targets are the
normalized (x,y) coordinates of the subset's keypoints (instances where all are
visible). Body joints co-vary kinematically, so a frozen non-pose encoder leaves
correlated residuals.

(A) Uncertainty / mechanism: per (subset x encoder), compute Indep/ICM/ResICM
    joint NLL, R^2, and D_logdet = -0.5 logdet(R_res). Test Spearman(D_logdet,
    dNLL) across conditions and whether dNLL grows with T (our T-scaling claim).
(B) Selection: for the full-body target set, rank the encoders by GP-LML / GP-ICM
    / LogME against true CV R^2 (Kendall tau).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legJ_jointcalib import fit_icm, icm_predict, indep_predict, joint_metrics
from exp_residual_diagnostic import _resid_corr_matrix, dlogdet_lw
from exp_residual_icm import residual_corr_from_train, resicm_predict

def standardised_residuals(Y, mu, Sig):
    """Predictive z-scores z = (Y - mu) / s, with s the marginal predictive sd.

    R_res enters the joint predictive as Sigma(x) = D_s(x) R D_s(x), so the
    correlation matched to the Gaussian log score is that of z, not of the raw
    residuals (they agree only if s_t(x)/sigma_t is constant in x). Adopted
    throughout; see exp_dlogdet_standardised.py for the
    raw-vs-standardised comparison.
    """
    sd = np.sqrt(np.maximum(np.diagonal(Sig, axis1=1, axis2=2), 1e-12))
    return (Y - mu) / sd


EMB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/mpii_embed")
N_SUB, CV, PCA_DIM, MIN_N = 250, 5, 64, 120

# MPII 16-joint indices: 0 r_ankle 1 r_knee 2 r_hip 3 l_hip 4 l_knee 5 l_ankle
# 6 pelvis 7 thorax 8 upper_neck 9 head_top 10 r_wrist 11 r_elbow 12 r_shoulder
# 13 l_shoulder 14 l_elbow 15 l_wrist
SUBSETS = {
    "torso(4kp)":   [2, 3, 12, 13],
    "arms(6kp)":    [10, 11, 12, 13, 14, 15],
    "legs(6kp)":    [0, 1, 2, 3, 4, 5],
    "limbs(8kp)":   [0, 1, 4, 5, 10, 11, 14, 15],
    "body(12kp)":   [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
}


def load_encoder(path):
    z = np.load(path, allow_pickle=True)
    return str(z["encoder"]), z["X"], z["Ykp"], z["vis"]


def make_condition(X, Ykp, vis, kps, seed=0):
    mask = (vis[:, kps] > 0).all(1)
    Xc, Yc = X[mask], Ykp[mask][:, kps, :].reshape(mask.sum(), -1)
    if len(Xc) < MIN_N:
        return None
    if len(Xc) > N_SUB:
        idx = np.random.default_rng(seed).choice(len(Xc), N_SUB, replace=False)
        Xc, Yc = Xc[idx], Yc[idx]
    Xs = StandardScaler().fit_transform(Xc)
    if Xs.shape[1] > PCA_DIM:
        Xs = StandardScaler().fit_transform(PCA(PCA_DIM, random_state=0).fit_transform(Xs))
    return Xs, StandardScaler().fit_transform(Yc)


WITH_RESICM = os.environ.get("WITH_RESICM", "1") == "1"


def predict_calibrate(Xs, Ys, seed=0):
    n, T = Ys.shape
    methods = ("Indep", "ICM", "ResICM") if WITH_RESICM else ("Indep", "ICM")
    P = {m: (np.zeros((n, T)), np.zeros((n, T, T))) for m in methods}
    for tr, te in KFold(CV, shuffle=True, random_state=seed).split(Xs):
        mi, Si, ell = indep_predict(Xs[tr], Ys[tr], Xs[te])
        P["Indep"][0][te], P["Indep"][1][te] = mi, Si
        B, s2 = fit_icm(Xs[tr], Ys[tr], ell)
        mc, Sc = icm_predict(Xs[tr], Ys[tr], Xs[te], ell, B, s2)
        P["ICM"][0][te], P["ICM"][1][te] = mc, Sc
        if WITH_RESICM:
            R = residual_corr_from_train(Xs[tr], Ys[tr], inner_cv=2)
            mr, Sr = resicm_predict(Xs[tr], Ys[tr], Xs[te], R)
            P["ResICM"][0][te], P["ResICM"][1][te] = mr, Sr
    M = {m: joint_metrics(Ys, *P[m]) for m in P}
    resid = standardised_residuals(Ys, P["Indep"][0], P["Indep"][1])
    D = float(-0.5 * np.log(max(np.linalg.det(_resid_corr_matrix(resid)), 1e-12)))
    D_lw = dlogdet_lw(resid)
    return M, D, D_lw


def gp_lml(Xs, y, dot=False):
    base = DotProduct(1.0) if dot else RBF(1.0, (1e-1, 1e2))
    k = ConstantKernel(1.0, (1e-2, 1e2)) * base + WhiteKernel(0.1, (1e-4, 1e1))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1,
                                    alpha=1e-6).fit(Xs, y).log_marginal_likelihood_value_


def icm_neglml(Phi, Y, ell, B, sig2):
    from scipy.spatial.distance import pdist, squareform
    n, T = Y.shape
    Kx = np.exp(-squareform(pdist(Phi, "sqeuclidean")) / (2 * ell ** 2)) + 1e-6 * np.eye(n)
    lam_x, Ux = np.linalg.eigh(Kx)
    Ac = Ux.T @ Y
    lam_b, Ub = np.linalg.eigh(B); lam_b = np.clip(lam_b, 1e-8, None)
    Mp = Ac @ Ub
    den = np.outer(lam_x, lam_b) + sig2
    return 0.5 * (np.sum(Mp ** 2 / den) + np.sum(np.log(den)) + n * T * np.log(2 * np.pi))


def cv_r2(Xs, Ys):
    n, T = Ys.shape
    mu = np.zeros((n, T))
    for tr, te in KFold(CV, shuffle=True, random_state=0).split(Xs):
        mu[te], _, _ = indep_predict(Xs[tr], Ys[tr], Xs[te])
    return float(np.mean([r2_score(Ys[:, t], mu[:, t]) for t in range(T)]))


def main():
    files = sorted(glob.glob(os.path.join(EMB_DIR, "*.npz")))
    if not files:
        print("No COCO embeddings in", EMB_DIR); return
    encs = {}
    for f in files:
        name, X, Ykp, vis = load_encoder(f)
        encs[name] = (X, Ykp, vis)
        print(f"loaded {name} X{X.shape}", flush=True)

    # ---- (A) mechanism ----
    rows = []
    for name, (X, Ykp, vis) in encs.items():
        for sub, kps in SUBSETS.items():
            cond = make_condition(X, Ykp, vis, kps)
            if cond is None:
                continue
            Xs, Ys = cond
            M, D, D_lw = predict_calibrate(Xs, Ys)
            base = M["Indep"]["joint_nll"]
            rows.append(dict(encoder=name, subset=sub, T=Ys.shape[1], n=len(Xs),
                             D_logdet=round(D, 3), D_logdet_lw=round(D_lw, 3),
                             r2_indep=round(M["Indep"]["mean_r2"], 3),
                             dnll_icm=round(M["ICM"]["joint_nll"] - base, 3),
                             dnll_resicm=(round(M["ResICM"]["joint_nll"] - base, 3) if WITH_RESICM else None),
                             dr2_icm=round(M["ICM"]["mean_r2"] - M["Indep"]["mean_r2"], 3)))
            print(f"  [{name}|{sub}] T={Ys.shape[1]} n={len(Xs)} D={D:.2f} "
                  f"r2={M['Indep']['mean_r2']:.2f} dNLLicm={rows[-1]['dnll_icm']:+.2f}", flush=True)
    dfA = pd.DataFrame(rows)
    os.makedirs("results/mpii", exist_ok=True)
    dfA.to_csv("results/mpii/mpii_predict_calibrate.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== (A) MPII keypoint mechanism ===")
    print(dfA.to_string(index=False))
    if len(dfA) >= 4:
        rd, pd_ = spearmanr(dfA["D_logdet"], dfA["dnll_icm"])
        rlw, plw = spearmanr(dfA["D_logdet_lw"], dfA["dnll_icm"])
        rt, pt = spearmanr(dfA["T"], dfA["dnll_icm"])
        rr, pr = spearmanr(dfA["r2_indep"], dfA["dnll_icm"])
        print(f"\nSpearman(D_logdet,    dNLL_icm) = {rd:+.3f} (p={pd_:.3f})  [plain]")
        print(f"Spearman(D_logdet_LW, dNLL_icm) = {rlw:+.3f} (p={plw:.3f})  [shrinkage]")
        print(f"Spearman(T,        dNLL_icm) = {rt:+.3f} (p={pt:.3f})  [T-scaling]")
        print(f"Spearman(r2_indep, dNLL_icm) = {rr:+.3f} (p={pr:.3f})  [ceiling]")
        print(f"max |dR2_icm| = {dfA['dr2_icm'].abs().max():.3f}")

    # ---- (B) selection on the largest populated subset ----
    big = "body(12kp)"
    kps = SUBSETS[big]
    true_r2, gpl, icm_ev, logme = {}, {}, {}, {}
    for name, (X, Ykp, vis) in encs.items():
        cond = make_condition(X, Ykp, vis, kps)
        if cond is None:
            continue
        Xs, Ys = cond
        true_r2[name] = cv_r2(Xs, Ys)
        gpl[name] = sum(gp_lml(Xs, Ys[:, t]) for t in range(Ys.shape[1]))
        logme[name] = sum(gp_lml(Xs, Ys[:, t], dot=True) for t in range(Ys.shape[1]))
        _, _, ell = indep_predict(Xs, Ys, Xs[:2])
        B, s2 = fit_icm(Xs, Ys, ell)
        icm_ev[name] = -icm_neglml(Xs, Ys, ell, B, s2)
    order = sorted(true_r2)
    tr = [true_r2[e] for e in order]
    selrows = []
    for label, sc in [("GP-LML", gpl), ("GP-ICM", icm_ev), ("LogME", logme)]:
        tau, _ = kendalltau([sc[e] for e in order], tr)
        selrows.append(dict(method=label, kendall_tau=round(tau, 3)))
    dfB = pd.DataFrame(selrows)
    dfB.to_csv("results/mpii/mpii_selection.csv", index=False)
    print(f"\n=== (B) selection on {big} ===")
    print("true CV R2:", {e: round(true_r2[e], 3) for e in order})
    print(dfB.to_string(index=False))
    print("\nSaved results/mpii/mpii_predict_calibrate.csv, coco_selection.csv")


if __name__ == "__main__":
    main()
