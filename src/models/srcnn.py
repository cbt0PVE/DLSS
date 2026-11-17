"""SRCNN（Dong et al., ECCV 2014）的外推版本。

结构：conv(9x9, 64) - ReLU - conv(5x5, 32) - ReLU - conv(5x5, 1)，
默认预测残差（base + net(x)），最后一个卷积零初始化，所以训练开始时输出
严格等于 bicubic 输入 —— 这样任何 val PSNR 提升都能归因到网络本身。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (64, 32),
        kernels: tuple[int, ...] = (9, 5, 5),
        residual: bool = True,
    ) -> None:
        super().__init__()
        if len(channels) + 1 != len(kernels):
            raise ValueError("kernels 数量应为 channels 数量 + 1")

        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out, k in zip(channels, kernels[:-1]):
            layers += [nn.Conv2d(c_in, c_out, k, padding=k // 2), nn.ReLU(inplace=True)]
            c_in = c_out
        last = nn.Conv2d(c_in, in_channels, kernels[-1], padding=kernels[-1] // 2)
        layers.append(last)

        self.net = nn.Sequential(*layers)
        self.residual = residual
        self.border = sum((k - 1) // 2 for k in kernels)

        self._init_weights()
        if residual:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.residual:
            out = out + x
        return out
