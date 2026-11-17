"""导出一张局部放大的三联对比图：bicubic / SRCNN / GT。

    python scripts/compare_crop.py --config configs/srcnn_x2.yaml `
        --checkpoint $env:DLSS_RUNS\srcnn_x2_skimage\best.pt `
        --image coffee --center 0.42 0.55 --size 160 --zoom 2 `
        --out experiments/2026-11-18_srcnn_skimage_2x/outputs/coffee_crop.png

center 用相对坐标（0~1），方便同一位置反复取图做对照实验。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import data_root, get_device, load_config  # noqa: E402
from data.sr_dataset import (  # noqa: E402
    EvalImageSet,
    load_skimage_images,
    split_sources,
    ycbcr_to_rgb,
)
from eval import build_model  # noqa: E402


def panel(img: np.ndarray, label: str, zoom: int) -> np.ndarray:
    img = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bar = np.full((24, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", default=None, help="验证集里的图片名，默认第一张")
    ap.add_argument("--center", nargs=2, type=float, default=[0.5, 0.5])
    ap.add_argument("--size", type=int, default=160, help="裁剪边长（HR 像素）")
    ap.add_argument("--zoom", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("train", {}).get("device", "cuda"))
    _, val_names = split_sources(cfg["data"]["sources"], cfg["data"]["val_images"])
    images = load_skimage_images(val_names, data_root() / "cache")
    ds = EvalImageSet(images, scale=cfg["data"]["scale"])

    item = None
    for it in ds:
        if args.image is None or it["name"] == args.image:
            item = it
            break
    if item is None:
        raise SystemExit(f"{args.image} 不在验证集 {val_names} 里")

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    lr = item["lr"].unsqueeze(0).to(device)
    with torch.no_grad():
        sr = model(lr).clamp(0.0, 1.0)

    h, w = item["hr"].shape[-2:]
    s = min(args.size, h, w)
    cy, cx = int(args.center[0] * h), int(args.center[1] * w)
    y0, x0 = max(0, min(cy - s // 2, h - s)), max(0, min(cx - s // 2, w - s))
    sl = (slice(y0, y0 + s), slice(x0, x0 + s))

    cbcr = item["cbcr"].permute(1, 2, 0).numpy()[sl]
    hr_rgb = (item["hr_rgb"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)[sl]
    bic_rgb = ycbcr_to_rgb(item["lr"][0].numpy()[sl], cbcr)
    sr_rgb = ycbcr_to_rgb(sr[0, 0].cpu().numpy()[sl], cbcr)

    sep = np.full((s * args.zoom + 24, 4, 3), 255, np.uint8)
    out = np.hstack(
        [
            panel(bic_rgb, "bicubic x2", args.zoom),
            sep,
            panel(sr_rgb, "SRCNN x2", args.zoom),
            sep,
            panel(hr_rgb, "GT", args.zoom),
        ]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    print(f"saved -> {out_path}  ({item['name']} crop {s}x{s} @ ({y0},{x0}) x{args.zoom})")


if __name__ == "__main__":
    main()
