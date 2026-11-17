"""评估：在验证集整图上比 PSNR（ours vs bicubic），可选导出对比图。

用法（训练完自动调用一次，也可以手动跑）：
    python src/eval.py --config configs/srcnn_x2.yaml \
        --checkpoint $DLSS_RUNS/srcnn_x2/best.pt --save-images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import data_root, get_device, load_config, save_json  # noqa: E402
from data.sr_dataset import (  # noqa: E402
    EvalImageSet,
    load_skimage_images,
    split_sources,
    ycbcr_to_rgb,
)
from metrics import psnr  # noqa: E402
from models.srcnn import SRCNN  # noqa: E402


def build_model(cfg: dict) -> SRCNN:
    m = cfg["model"]
    return SRCNN(
        in_channels=m.get("in_channels", 1),
        channels=tuple(m.get("channels", [64, 32])),
        kernels=tuple(m.get("kernels", [9, 5, 5])),
        residual=m.get("residual", True),
    )


def build_eval_set(cfg: dict, which: str = "val") -> tuple[EvalImageSet, dict[str, np.ndarray]]:
    d = cfg["data"]
    train_names, val_names = split_sources(d["sources"], d["val_images"])
    names = train_names if which == "train" else val_names
    images = load_skimage_images(names, data_root() / "cache")
    return EvalImageSet(images, scale=d["scale"]), images


@torch.no_grad()
def evaluate_model(model: SRCNN, dataset: EvalImageSet, device: torch.device) -> dict:
    """返回整体 PSNR、bicubic 基线、逐图结果。"""
    model.eval()
    border = model.border
    rows = []
    for item in dataset:
        lr = item["lr"].unsqueeze(0).to(device)
        hr = item["hr"].unsqueeze(0).to(device)
        out = model(lr).clamp(0.0, 1.0)
        rows.append(
            {
                "name": item["name"],
                "psnr": psnr(out[0], hr[0], border=border),
                "psnr_bicubic": psnr(lr[0], hr[0], border=border),
            }
        )
    mean_ours = float(np.mean([r["psnr"] for r in rows]))
    mean_bicubic = float(np.mean([r["psnr_bicubic"] for r in rows]))
    return {
        "psnr": mean_ours,
        "psnr_bicubic": mean_bicubic,
        "gain_db": mean_ours - mean_bicubic,
        "border": border,
        "images": rows,
    }


def _panel(img_rgb: np.ndarray, label: str) -> np.ndarray:
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    bar = np.full((22, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


@torch.no_grad()
def save_comparison(model: SRCNN, dataset: EvalImageSet, out_dir: Path, device: torch.device) -> None:
    """每张验证图输出 LR(bicubic) / SRCNN / HR 三联图。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for item in dataset:
        lr = item["lr"].unsqueeze(0).to(device)
        cbcr = item["cbcr"].permute(1, 2, 0).numpy()
        hr_rgb = (item["hr_rgb"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

        lr_rgb = ycbcr_to_rgb(item["lr"][0].numpy(), cbcr)
        sr_y = model(lr).clamp(0.0, 1.0)[0, 0].cpu().numpy()
        sr_rgb = ycbcr_to_rgb(sr_y, cbcr)

        sep = np.full((lr_rgb.shape[0] + 22, 4, 3), 255, np.uint8)
        triple = np.hstack(
            [
                _panel(lr_rgb, "bicubic input"),
                sep,
                _panel(sr_rgb, "srcnn x2"),
                sep,
                _panel(hr_rgb, "ground truth"),
            ]
        )
        cv2.imwrite(str(out_dir / f"{item['name']}_compare.png"), cv2.cvtColor(triple, cv2.COLOR_RGB2BGR))


def main() -> None:
    ap = argparse.ArgumentParser(description="SRCNN 评估")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--out", default=None, help="结果目录，默认 checkpoint 同级 /eval")
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("train", {}).get("device", "cuda"))

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    dataset, _ = build_eval_set(cfg, args.split)
    res = evaluate_model(model, dataset, device)

    print(f"split = {args.split}, border = {res['border']}")
    print(f"{'image':<12}{'bicubic':>10}{'srcnn':>10}{'gain':>8}")
    for r in res["images"]:
        print(f"{r['name']:<12}{r['psnr_bicubic']:>10.3f}{r['psnr']:>10.3f}{r['psnr'] - r['psnr_bicubic']:>8.3f}")
    print(f"{'mean':<12}{res['psnr_bicubic']:>10.3f}{res['psnr']:>10.3f}{res['gain_db']:>8.3f}")

    out_dir = Path(args.out) if args.out else Path(args.checkpoint).parent / "eval"
    save_json(res, out_dir / "metrics.json")
    if args.save_images:
        save_comparison(model, dataset, out_dir / "images", device)
        print(f"对比图 -> {out_dir / 'images'}")
    print(f"指标   -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
