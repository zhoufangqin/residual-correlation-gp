"""
Embed AFLW2000-3D face crops with frozen encoders (second real-image dataset, a
DIFFERENT domain from MPII pose: faces). Targets are the 68 facial landmarks'
(x,y) coordinates normalized to a face crop -- large T, strongly correlated,
canonical framing -> expected moderate-residual regime. Reuses embed_coco code.

Output (one file per encoder): results/aflw_embed/{encoder}.npz
  X (n x d), Ykp (n x 68 x 2), vis (n x 68)
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image
from scipy.io import loadmat

import embed_coco as ec

_HERE = os.path.dirname(os.path.abspath(__file__))
AFLW_DIR = os.path.join(_HERE, "imgdata", "aflw")


def _find_dir():
    for d in glob.glob(os.path.join(AFLW_DIR, "**", "*.mat"), recursive=True):
        return os.path.dirname(d)
    raise FileNotFoundError("no AFLW .mat under " + AFLW_DIR)


def build_instances(n_max=2000, pad=0.4):
    base = _find_dir()
    mats = sorted(glob.glob(os.path.join(base, "*.mat")))
    crops, Ykp, vis = [], [], []
    for mf in mats:
        img = mf[:-4] + ".jpg"
        if not os.path.exists(img):
            continue
        m = loadmat(mf)
        if "pt3d_68" not in m:
            continue
        lm = m["pt3d_68"][:2].T.astype(float)        # 68 x 2 (x, y) in pixels
        x0_, y0_ = lm.min(0); x1_, y1_ = lm.max(0)
        cx, cy = (x0_ + x1_) / 2, (y0_ + y1_) / 2
        s = max(x1_ - x0_, y1_ - y0_) * (1 + 2 * pad)
        if s < 8:
            continue
        x0, y0 = cx - s / 2, cy - s / 2
        try:
            im = Image.open(img).convert("RGB")
        except Exception:
            continue
        crop = im.crop((x0, y0, x0 + s, y0 + s)).resize((224, 224), Image.BILINEAR)
        nx = (lm[:, 0] - x0) / s
        ny = (lm[:, 1] - y0) / s
        crops.append(np.asarray(crop, dtype=np.uint8))
        Ykp.append(np.stack([nx, ny], 1).astype(np.float32))
        vis.append(np.ones(68, np.float32))
        if len(crops) >= n_max:
            break
    return np.stack(crops), np.stack(Ykp), np.stack(vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--encoders", default=("dinov2_vits14,dinov2_vitb14,dinov2_vitl14,"
                    "vit_base_patch16_224.augreg2_in21k_ft_in1k,"
                    "vit_large_patch16_224.augreg_in21k_ft_in1k,"
                    "vit_base_patch16_clip_224.openai,resnet50.a1_in1k,"
                    "convnext_base.fb_in22k_ft_in1k,"
                    "swin_base_patch4_window7_224.ms_in22k_ft_in1k,"
                    "deit3_base_patch16_224.fb_in22k_ft_in1k"))
    ap.add_argument("--out", default=os.path.join(_HERE, "results/aflw_embed"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)
    crops, Ykp, vis = build_instances(args.n)
    print(f"instances: {len(crops)}  Ykp{Ykp.shape}", flush=True)

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
