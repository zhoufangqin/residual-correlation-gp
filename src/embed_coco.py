"""
Embed COCO val2017 person crops with frozen transformer/CNN encoders, for the
real-image calibration-mechanism test (large-T, kinematically-correlated targets).

Each annotated person instance is cropped to a padded square bbox and resized;
the regression targets are the 17 COCO keypoints' (x,y) coordinates normalized to
[0,1] within the crop. Body keypoints co-vary via kinematics, so a frozen (non-
pose) encoder leaves correlated residuals -- the regime where the residual-
correlation diagnostic D_logdet should predict the ICM joint-NLL benefit, and at
large T (up to 34 coordinates), the regime our T-scaling claim cares about most.

We save the full 17-keypoint target matrix + visibility per instance, so the
analysis (exp_coco_instrument.py) can form keypoint-subset conditions without
re-embedding. Multiple encoders give the "encoder variants" axis.

Output (one file per encoder):
  results/coco_embed/{encoder}.npz  with X (n x d), Ykp (n x 17 x 2), vis (n x 17)

GPU job (gpu_mig); encoder weights pre-cached (see prefetch_encoders.py).
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
COCO_DIR = os.path.join(_HERE, "imgdata", "coco")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _overlap_frac(box, others):
    """Fraction of the square crop `box`=(x0,y0,s) covered by other bboxes (xywh)."""
    x0, y0, s = box
    area = s * s
    cov = 0.0
    for (ox, oy, ow, oh) in others:
        ix = max(0.0, min(x0 + s, ox + ow) - max(x0, ox))
        iy = max(0.0, min(y0 + s, oy + oh) - max(y0, oy))
        cov += ix * iy
    return cov / max(area, 1e-6)


def build_instances(n_max, min_kpts=14, min_area=64 * 64, pad=0.15, seed=0,
                    single_person=False, max_other_frac=0.10, mask_others=False):
    """Return (crops_uint8 [n,H,W,3], Ykp [n,17,2] normalized, vis [n,17]).
    single_person: keep only crops where other people's bboxes cover < max_other_frac
    of the crop. mask_others: gray out other people's bbox regions inside the crop."""
    from pycocotools.coco import COCO
    ann_file = os.path.join(COCO_DIR, "annotations", "person_keypoints_val2017.json")
    coco = COCO(ann_file)
    pcat = coco.getCatIds(catNms=["person"])
    anns = coco.loadAnns(coco.getAnnIds(catIds=pcat, iscrowd=False))
    rng = np.random.default_rng(seed)
    rng.shuffle(anns)
    # cache per-image person bboxes for the "others" computation
    by_img = {}
    for a in coco.loadAnns(coco.getAnnIds(catIds=pcat)):
        by_img.setdefault(a["image_id"], []).append((a["id"], a["bbox"]))

    crops, Ykp, vis = [], [], []
    for a in anns:
        if a.get("num_keypoints", 0) < min_kpts or a["area"] < min_area:
            continue
        x, y, w, h = a["bbox"]
        if w < 8 or h < 8:
            continue
        cx, cy = x + w / 2, y + h / 2
        s = max(w, h) * (1 + 2 * pad)
        x0, y0 = cx - s / 2, cy - s / 2
        others = [bb for (aid, bb) in by_img.get(a["image_id"], []) if aid != a["id"]]
        if single_person and _overlap_frac((x0, y0, s), others) > max_other_frac:
            continue
        img_info = coco.loadImgs(a["image_id"])[0]
        path = os.path.join(COCO_DIR, "val2017", img_info["file_name"])
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            continue
        crop = im.crop((x0, y0, x0 + s, y0 + s)).resize((224, 224), Image.BILINEAR)
        if mask_others and others:
            arr = np.asarray(crop).copy()
            for (ox, oy, ow, oh) in others:
                # map other bbox into the 224 crop frame and gray it out
                gx0 = int(np.clip((ox - x0) / s * 224, 0, 224))
                gy0 = int(np.clip((oy - y0) / s * 224, 0, 224))
                gx1 = int(np.clip((ox + ow - x0) / s * 224, 0, 224))
                gy1 = int(np.clip((oy + oh - y0) / s * 224, 0, 224))
                arr[gy0:gy1, gx0:gx1] = 128
            crop = Image.fromarray(arr)
        kp = np.array(a["keypoints"], float).reshape(17, 3)
        nx = (kp[:, 0] - x0) / s          # normalized x in crop
        ny = (kp[:, 1] - y0) / s
        v = (kp[:, 2] > 0).astype(np.float32)
        crops.append(np.asarray(crop, dtype=np.uint8))
        Ykp.append(np.stack([nx, ny], 1).astype(np.float32))
        vis.append(v)
        if len(crops) >= n_max:
            break
    return np.stack(crops), np.stack(Ykp), np.stack(vis)


def get_encoder(name, device):
    if name.startswith("dinov2"):
        repo = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        if os.path.isdir(repo):
            m = torch.hub.load(repo, name, source="local", pretrained=True)
        else:
            m = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
        return m.to(device).eval(), IMAGENET_MEAN, IMAGENET_STD, 224
    import timm
    m = timm.create_model(name, pretrained=True, num_classes=0)
    cfg = timm.data.resolve_model_data_config(m)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    return m.to(device).eval(), mean, std, cfg["input_size"][-1]


@torch.no_grad()
def embed(model, crops, mean, std, size, device, bs=128):
    mean, std = mean.to(device), std.to(device)
    out = []
    for i in range(0, len(crops), bs):
        x = torch.from_numpy(crops[i:i + bs]).float().to(device) / 255.0   # b,H,W,3
        x = x.permute(0, 3, 1, 2)
        if x.shape[-1] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = (x - mean) / std
        feat = model(x)
        if feat.ndim > 2:
            feat = feat.flatten(1)
        out.append(feat.float().cpu().numpy())
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--encoders", default=("dinov2_vits14,dinov2_vitb14,"
                    "vit_base_patch16_224.augreg2_in21k_ft_in1k,"
                    "vit_base_patch16_clip_224.openai,resnet50.a1_in1k"))
    ap.add_argument("--out", default=os.path.join(_HERE, "results/coco_embed"))
    ap.add_argument("--single_person", action="store_true")
    ap.add_argument("--mask_others", action="store_true")
    ap.add_argument("--min_kpts", type=int, default=14)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, "single_person", args.single_person,
          "mask_others", args.mask_others, flush=True)
    crops, Ykp, vis = build_instances(args.n, min_kpts=args.min_kpts, min_area=96 * 96,
                                      single_person=args.single_person,
                                      mask_others=args.mask_others)
    print(f"instances: {len(crops)}  Ykp{Ykp.shape}  mean visible kpts={vis.sum(1).mean():.1f}",
          flush=True)

    for enc in args.encoders.split(","):
        short = enc.split(".")[0]
        model, mean, std, size = get_encoder(enc, device)
        X = embed(model, crops, mean, std, size, device)
        np.savez(os.path.join(args.out, short + ".npz"),
                 X=X.astype(np.float32), Ykp=Ykp, vis=vis, encoder=short)
        print(f"  [{enc}] X{X.shape} -> {short}.npz", flush=True)
        del model
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
