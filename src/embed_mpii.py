"""
Embed MPII Human Pose person crops with frozen encoders (second large-T real-image
dataset for the calibration-mechanism test; corroborates COCO).

MPII annotations come as a MATLAB .mat RELEASE struct. We parse training images
with annotated people, crop each person to a square around objpos sized by the
MPII scale convention (person height approx 200*scale px), and regress the 16
joint (x,y) coordinates normalized to the crop. Reuses the encoder + embedding
code from embed_coco.py.

Output (one file per encoder): results/mpii_embed/{encoder}.npz
  X (n x d), Ykp (n x 16 x 2), vis (n x 16)
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image
from scipy.io import loadmat

import embed_coco as ec   # get_encoder, embed

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
MPII_DIR = os.path.join(_HERE, "imgdata", "mpii")
N_JOINTS = 16


def _find_mat():
    hits = glob.glob(os.path.join(MPII_DIR, "**", "mpii_human_pose_v1_u12_1.mat"),
                     recursive=True)
    if not hits:
        raise FileNotFoundError("MPII .mat not found under " + MPII_DIR)
    return hits[0]


def _find_imgdir():
    for d in glob.glob(os.path.join(MPII_DIR, "**", "images"), recursive=True):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.jpg")):
            return d
    # fallback: any dir with jpgs
    for root, _, files in os.walk(MPII_DIR):
        if any(f.endswith(".jpg") for f in files):
            return root
    raise FileNotFoundError("MPII images dir not found under " + MPII_DIR)


def _aslist(x):
    """squeeze_me turns single-element arrays into scalars; normalize to list."""
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return list(x.ravel()) if x.dtype == object else [x]
    return [x]


def build_instances(n_max, pad=1.25, min_joints=12, seed=0):
    mat = loadmat(_find_mat(), struct_as_record=False, squeeze_me=True)
    R = mat["RELEASE"]
    annolist = _aslist(R.annolist)
    img_train = np.atleast_1d(R.img_train)
    img_dir = _find_imgdir()

    order = np.random.default_rng(seed).permutation(len(annolist))
    crops, Ykp, vis = [], [], []
    for i in order:
        if i >= len(img_train) or img_train[i] != 1:
            continue
        a = annolist[i]
        try:
            name = a.image.name
        except Exception:
            continue
        for rect in _aslist(getattr(a, "annorect", None)):
            ap = getattr(rect, "annopoints", None)
            objpos = getattr(rect, "objpos", None)
            scale = getattr(rect, "scale", None)
            if ap is None or objpos is None or scale is None or np.size(scale) == 0:
                continue
            pts = _aslist(getattr(ap, "point", None))
            J = np.full((N_JOINTS, 2), np.nan); V = np.zeros(N_JOINTS)
            for p in pts:
                jid = int(p.id)
                if 0 <= jid < N_JOINTS:
                    J[jid] = [float(p.x), float(p.y)]
                    iv = getattr(p, "is_visible", 1)
                    V[jid] = 1.0 if (np.size(iv) == 0 or int(np.atleast_1d(iv)[0]) == 1
                                     or str(iv) == "1") else 0.0
            if np.isfinite(J).all(1).sum() < min_joints:
                continue
            s = float(np.atleast_1d(scale)[0]) * 200.0 * pad
            cx, cy = float(objpos.x), float(objpos.y)
            x0, y0 = cx - s / 2, cy - s / 2
            try:
                im = Image.open(os.path.join(img_dir, name)).convert("RGB")
            except Exception:
                continue
            crop = im.crop((x0, y0, x0 + s, y0 + s)).resize((224, 224), Image.BILINEAR)
            nx = (J[:, 0] - x0) / s
            ny = (J[:, 1] - y0) / s
            present = np.isfinite(J).all(1)
            nx = np.where(present, nx, 0.0); ny = np.where(present, ny, 0.0)
            crops.append(np.asarray(crop, dtype=np.uint8))
            Ykp.append(np.stack([nx, ny], 1).astype(np.float32))
            vis.append((present & (V > 0)).astype(np.float32))
            if len(crops) >= n_max:
                break
        if len(crops) >= n_max:
            break
    return np.stack(crops), np.stack(Ykp), np.stack(vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--encoders", default=("dinov2_vits14,dinov2_vitb14,"
                    "vit_base_patch16_224.augreg2_in21k_ft_in1k,"
                    "vit_base_patch16_clip_224.openai,resnet50.a1_in1k"))
    ap.add_argument("--out", default=os.path.join(_HERE, "results/mpii_embed"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)
    crops, Ykp, vis = build_instances(args.n)
    print(f"instances: {len(crops)}  mean visible joints={vis.sum(1).mean():.1f}", flush=True)

    for enc in args.encoders.split(","):
        short = enc.split(".")[0]
        model, mean, std, size = ec.get_encoder(enc, device)
        X = ec.embed(model, crops, mean, std, size, device)
        np.savez(os.path.join(args.out, short + ".npz"),
                 X=X.astype(np.float32), Ykp=Ykp, vis=vis, encoder=short)
        print(f"  [{enc}] X{X.shape} -> {short}.npz", flush=True)
        del model
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
