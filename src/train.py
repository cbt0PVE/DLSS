"""训练入口。

    python src/train.py --config configs/srcnn_x2.yaml

固定随机种子、按 iteration 训练（patch 在线采样，没有 epoch 概念），
每 val_every 步在验证集整图上算 PSNR，同时记录 bicubic 基线。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    AverageMeter,
    data_root,
    get_device,
    human_time,
    load_config,
    runs_root,
    save_json,
    set_seed,
)
from data.sr_dataset import (  # noqa: E402
    PairedPatchDataset,
    load_skimage_images,
    split_sources,
)
from eval import build_eval_set, build_model, evaluate_model, save_comparison  # noqa: E402
from losses.losses import build_loss  # noqa: E402


def infinite(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def crop_border(x: torch.Tensor, b: int) -> torch.Tensor:
    return x[..., b:-b, b:-b] if b > 0 else x


def main() -> None:
    ap = argparse.ArgumentParser(description="SRCNN 训练")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="", help="给输出目录加后缀，便于对照实验")
    ap.add_argument("--iters", type=int, default=None, help="覆盖配置里的训练步数（冒烟测试用）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg["experiment"]["name"] + (f"_{args.tag}" if args.tag else "")
    set_seed(cfg["experiment"].get("seed", 42))

    device = get_device(cfg.get("train", {}).get("device", "cuda"))
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        print(f"[env] {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

    dcfg, tcfg = cfg["data"], cfg["train"]
    iters = args.iters or tcfg["iters"]
    batch_size = int(dcfg.get("batch_size", 16))

    # ---- 数据
    train_names, val_names = split_sources(dcfg["sources"], dcfg["val_images"])
    print(f"[data] train={train_names} val={val_names}")
    train_imgs = load_skimage_images(train_names, data_root() / "cache")
    ds = PairedPatchDataset(
        list(train_imgs.values()),
        scale=dcfg["scale"],
        patch_size=dcfg["patch_size"],
        length=max(batch_size * 50, 512),
        augment=dcfg.get("augment", True),
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dcfg.get("num_workers", 0),
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    eval_ds, _ = build_eval_set(cfg, "val")

    # ---- 模型 / 优化器
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] SRCNN params={n_params / 1e3:.1f}K border={model.border}")

    criterion = build_loss(tcfg.get("loss", "l1"))
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=tcfg["lr"] * 0.01)

    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- 输出目录
    run_dir = runs_root() / exp_name
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / "tb"))
    (run_dir / "config.yaml").write_text(
        Path(args.config).read_text(encoding="utf-8"), encoding="utf-8"
    )

    print(f"[run] {run_dir} | iters={iters} bs={batch_size} amp={use_amp}")
    log_every, val_every = tcfg.get("log_every", 50), tcfg.get("val_every", 500)
    best = {"psnr": -1.0, "step": 0, "metrics": None}
    loss_meter = AverageMeter()
    t0 = time.time()
    data_iter = infinite(loader)

    for step in range(1, iters + 1):
        batch = next(data_iter)
        lr, hr = batch["lr"].to(device, non_blocking=True), batch["hr"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            pred = model(lr)
            loss = criterion(crop_border(pred, model.border), crop_border(hr, model.border))

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 1.0))
        scaler.step(opt)
        scaler.update()
        sched.step()

        loss_meter.update(loss.item())
        if step % log_every == 0:
            writer.add_scalar("train/loss", loss_meter.avg, step)
            writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
            speed = step / (time.time() - t0)
            print(
                f"step {step:>6}/{iters} | loss {loss_meter.avg:.5f} | "
                f"{speed:.1f} it/s | elapsed {human_time(time.time() - t0)}"
            )
            loss_meter.reset()

        if step % val_every == 0 or step == iters:
            res = evaluate_model(model, eval_ds, device)
            writer.add_scalar("val/psnr", res["psnr"], step)
            writer.add_scalar("val/psnr_bicubic", res["psnr_bicubic"], step)
            print(
                f"  [val @ {step}] psnr {res['psnr']:.3f} dB "
                f"(bicubic {res['psnr_bicubic']:.3f}, gain {res['gain_db']:+.3f})"
            )
            torch.save(
                {"model": model.state_dict(), "step": step, "metrics": res, "config": cfg},
                run_dir / "latest.pt",
            )
            if res["psnr"] > best["psnr"]:
                best = {"psnr": res["psnr"], "step": step, "metrics": res}
                torch.save(
                    {"model": model.state_dict(), "step": step, "metrics": res, "config": cfg},
                    run_dir / "best.pt",
                )
            model.train()

    # ---- 收尾：用 best 权重复核 + 出对比图
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    final = evaluate_model(model, eval_ds, device)
    save_comparison(model, eval_ds, run_dir / "images", device)

    summary = {
        "experiment": exp_name,
        "iters": iters,
        "params": n_params,
        "border": model.border,
        "best_step": best["step"],
        "best_psnr": best["psnr"],
        "final": final,
        "wall_time_sec": round(time.time() - t0, 1),
        "device": str(device),
        "torch": torch.__version__,
        "config": cfg,
    }
    save_json(summary, run_dir / "metrics.json")
    writer.close()

    print("\n=== done ===")
    print(f"best step {best['step']} | psnr {best['psnr']:.3f} dB")
    print(f"bicubic   {final['psnr_bicubic']:.3f} dB | gain {final['gain_db']:+.3f} dB")
    print(f"metrics -> {run_dir / 'metrics.json'}")
    print(f"images  -> {run_dir / 'images'}")


if __name__ == "__main__":
    main()
