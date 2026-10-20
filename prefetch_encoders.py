"""Instantiate every encoder once so its pretrained weights cache to ~/.cache.
Run on a LOGIN node (has internet); the gpu_mig job then loads offline."""
import os
import torch

ENCODERS = ["dinov2_vits14", "dinov2_vitb14",
            "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "vit_base_patch16_clip_224.openai", "resnet50.a1_in1k"]


def main():
    for name in ENCODERS:
        try:
            if name.startswith("dinov2"):
                m = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
            else:
                import timm
                m = timm.create_model(name, pretrained=True, num_classes=0)
            n = sum(p.numel() for p in m.parameters())
            print(f"OK   {name}  ({n/1e6:.1f}M params)", flush=True)
            del m
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__} {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
