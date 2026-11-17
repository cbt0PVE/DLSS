"""损失函数。超分重建默认用 L1，比 MSE 更不糊。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """sqrt(x^2 + eps^2)，对离群值比 L2 稳，比 L1 平滑。"""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


def build_loss(name: str = "l1") -> callable:
    name = name.lower()
    if name == "l1":
        return F.l1_loss
    if name == "l2" or name == "mse":
        return F.mse_loss
    if name == "charbonnier":
        return CharbonnierLoss()
    raise ValueError(f"未知损失函数：{name}")
