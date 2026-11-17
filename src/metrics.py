"""评价指标。PSNR 按 Y 通道、[0,1] 范围计算，并裁掉边界像素。"""

from __future__ import annotations

import math

import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, border: int = 0, max_val: float = 1.0) -> float:
    """单张图 PSNR（dB）。pred/target 形状 1xHxW 或 HxW，同尺寸。"""
    pred = pred.detach().float()
    target = target.detach().float()
    if pred.shape != target.shape:
        raise ValueError(f"尺寸不一致：{tuple(pred.shape)} vs {tuple(target.shape)}")
    if border > 0:
        if pred.ndim == 3:
            pred = pred[..., border:-border, border:-border]
            target = target[..., border:-border, border:-border]
        else:
            pred = pred[border:-border, border:-border]
            target = target[border:-border, border:-border]
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / mse)


def psnr_batch(pred: torch.Tensor, target: torch.Tensor, border: int = 0, max_val: float = 1.0) -> float:
    """batch 内逐张算 PSNR 再取平均（超分领域的标准做法）。"""
    n = pred.shape[0]
    return sum(psnr(pred[i], target[i], border=border, max_val=max_val) for i in range(n)) / n
