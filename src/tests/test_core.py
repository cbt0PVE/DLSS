"""最小测试：保证 shape、PSNR、降质、残差初始化这些易错点不变。

    python -m pytest src/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.sr_dataset import (  # noqa: E402
    PairedPatchDataset,
    downsample_bicubic,
    rgb_to_y,
    rgb_to_ycbcr,
    upsample_bicubic,
    ycbcr_to_rgb,
)
from metrics import psnr  # noqa: E402
from models.srcnn import SRCNN  # noqa: E402


def _fake_rgb(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_rgb_to_y_range_and_shape():
    y = rgb_to_y(_fake_rgb())
    assert y.shape == (64, 64)
    assert y.dtype == np.float32
    assert 0.0 <= y.min() and y.max() <= 1.0


def test_ycbcr_roundtrip():
    rgb = _fake_rgb()
    y, cbcr = rgb_to_ycbcr(rgb)
    back = ycbcr_to_rgb(y, cbcr)
    assert back.shape == rgb.shape and back.dtype == np.uint8
    assert np.abs(back.astype(int) - rgb.astype(int)).max() <= 2


def test_down_up_sample_shapes():
    y = rgb_to_y(_fake_rgb(64, 96))
    lr = downsample_bicubic(y, 2)
    assert lr.shape == (32, 48)
    up = upsample_bicubic(lr, (64, 96))
    assert up.shape == y.shape


def test_patch_dataset_shapes():
    imgs = [_fake_rgb(80, 100, seed=i) for i in range(3)]
    ds = PairedPatchDataset(imgs, scale=2, patch_size=32, length=8)
    item = ds[0]
    assert item["lr"].shape == (1, 32, 32)
    assert item["hr"].shape == (1, 32, 32)
    assert 0.0 <= item["hr"].min() and item["hr"].max() <= 1.0


def test_psnr_identity_is_inf():
    x = torch.rand(1, 32, 32)
    assert psnr(x, x) == float("inf")


def test_psnr_monotonic():
    target = torch.zeros(1, 32, 32)
    near = torch.full((1, 32, 32), 0.01)
    far = torch.full((1, 32, 32), 0.10)
    assert psnr(near, target) > psnr(far, target)


def test_psnr_border_crop():
    # 边界故意加噪声，裁掉后应得到 inf
    x = torch.zeros(1, 32, 32)
    x[..., :2, :] = 0.5
    x[..., -2:, :] = 0.5
    assert psnr(x, torch.zeros(1, 32, 32), border=2) == float("inf")


def test_srcnn_shapes_and_border():
    model = SRCNN()
    x = torch.rand(2, 1, 40, 48)
    y = model(x)
    assert y.shape == x.shape
    assert model.border == 8


def test_srcnn_residual_init_is_identity():
    """残差 + 末层零初始化 => 初始输出应等于输入（bicubic）。"""
    model = SRCNN(residual=True).eval()
    x = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(x), x, atol=1e-6)


def test_srcnn_non_residual_differs():
    model = SRCNN(residual=False).eval()
    x = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        assert not torch.allclose(model(x), x, atol=1e-2)
