"""数据管线：降质、YCbCr 转换、成对 patch 采样、整图评估集。

X2 的降质方式固定为 bicubic 下采样（和 SRCNN/VDSR 等工作的标准设定一致）：
    HR --bicubic_down(2)--> LR --bicubic_up(2)--> 网络输入
网络输入和目标都在 Y 通道上，PSNR 也按 Y 通道算，这是超分领域的通行口径。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------- 颜色空间

def rgb_to_y(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB -> HxW float32 亮度，范围 [0,1]。"""
    rgb = rgb.astype(np.float32) / 255.0
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (Y, CbCr)，都是 float32；Y 在 [0,1]，CbCr 约在 [-0.5,0.5]。

    色度按 BT.601 反变换的系数归一化（Cb=(B-Y)/1.772, Cr=(R-Y)/1.402），
    这样 ycbcr_to_rgb 用标准逆矩阵就能无损还原。
    """
    y = rgb_to_y(rgb)
    rgb_f = rgb.astype(np.float32) / 255.0
    cb = (rgb_f[..., 2] - y) / 1.772
    cr = (rgb_f[..., 0] - y) / 1.402
    return y, np.stack([cb, cr], axis=-1).astype(np.float32)


def ycbcr_to_rgb(y: np.ndarray, cbcr: np.ndarray) -> np.ndarray:
    """Y[0,1] + CbCr[-0.5,0.5] -> HxWx3 uint8 RGB。"""
    y = y[..., None]
    cb = cbcr[..., 0:1]
    cr = cbcr[..., 1:2]
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = np.concatenate([r, g, b], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------- 降质 / 上采样

def imresize(img: np.ndarray, size_hw: tuple[int, int], interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """按 (H, W) 缩放，保持 float32/uint8 类型不变。"""
    h, w = int(size_hw[0]), int(size_hw[1])
    out = cv2.resize(img, (w, h), interpolation=interpolation)
    if img.ndim == 3 and out.ndim == 2:  # 单通道输入防御
        out = out[..., None]
    return out.astype(img.dtype, copy=False)


def downsample_bicubic(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    return imresize(img, (h // scale, w // scale))


def upsample_bicubic(img: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    return imresize(img, size_hw)


# ---------------------------------------------------------------- 数据集来源

# 从 scikit-image 自带的示例图里取真实照片/纹理，避免依赖外网下载，保证离线可复现。
# 注意：horse() 是二值掩膜、cat() 和 chelsea() 是同一张图，都已排除；
# eagle/lily/skin 需要额外装 pooch 才能下载，不用。
# 括号里是 x2 bicubic 的 Y-PSNR，用来判断这张图难不难。
SKIMAGE_SOURCES = [
    "astronaut",             # 512x512 人像           ~30.8 dB
    "chelsea",               # 300x451 猫             ~33.6 dB
    "coffee",                # 400x600 静物           ~29.9 dB
    "rocket",                # 427x640 火箭           ~31.0 dB
    "camera",                # 512x512 灰度           ~30.0 dB
    "grass",                 # 512x512 草地纹理       ~23.7 dB（最难）
    "gravel",                # 512x512 碎石纹理       ~28.3 dB
    "brick",                 # 512x512 砖墙纹理       ~37.4 dB
    "clock",                 # 300x400 钟表细线       ~31 dB
    "immunohistochemistry",  # 512x512 彩色纹理       ~30 dB
]


def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """统一成 HxWx3 uint8；灰度复制成三通道，RGBA 丢掉 alpha。"""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def load_skimage_images(names: list[str], cache_dir: str | Path) -> dict[str, np.ndarray]:
    """读取示例图，第一次读时缓存成 .npy（skimage 解码比较慢）。"""
    from skimage import data as skdata

    cache_dir = Path(cache_dir) / "skimage"
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, np.ndarray] = {}
    for name in names:
        cache = cache_dir / f"{name}.npy"
        if cache.exists():
            out[name] = np.load(cache)
            continue
        if not hasattr(skdata, name):
            raise AttributeError(f"scikit-image 没有示例图 {name}")
        img = _to_rgb_uint8(getattr(skdata, name)())
        np.save(cache, img)
        out[name] = img
    return out


# ---------------------------------------------------------------- Dataset

def _to_tensor(y: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0).float()


class PairedPatchDataset(Dataset):
    """训练用：每次随机取一张图、随机裁 HR patch，再在线生成 LR。

    length 设成远大于图片数，等价于无限流式采样；配合 DataLoader 的 shuffle 即可。
    """

    def __init__(
        self,
        images: list[np.ndarray],
        scale: int = 2,
        patch_size: int = 64,
        length: int = 2000,
        augment: bool = True,
    ) -> None:
        if not images:
            raise ValueError("images 不能为空")
        if patch_size % scale != 0:
            raise ValueError("patch_size 必须是 scale 的整数倍，否则 LR 尺寸不整")
        self.images = images
        self.scale = scale
        self.patch_size = patch_size
        self.length = length
        self.augment = augment

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img = self.images[np.random.randint(len(self.images))]
        p = self.patch_size
        h, w = img.shape[:2]
        if h < p or w < p:
            pad_h, pad_w = max(0, p - h), max(0, p - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            h, w = img.shape[:2]

        y0 = np.random.randint(0, h - p + 1)
        x0 = np.random.randint(0, w - p + 1)
        hr = img[y0:y0 + p, x0:x0 + p]

        if self.augment:
            k = np.random.randint(4)
            if k:
                hr = np.rot90(hr, k)
            if np.random.rand() < 0.5:
                hr = hr[:, ::-1]
        hr = np.ascontiguousarray(hr)

        y_hr = rgb_to_y(hr)
        # 在 Y 上做降质，避免 RGB 分别插值引入色偏
        lr = downsample_bicubic(y_hr, self.scale)
        lr_up = upsample_bicubic(lr, y_hr.shape)
        return {"lr": _to_tensor(lr_up), "hr": _to_tensor(y_hr)}


class EvalImageSet(Dataset):
    """评估用：整图，不做 patch。附带 RGB 便于生成对比图。"""

    def __init__(self, images: dict[str, np.ndarray], scale: int = 2) -> None:
        self.items = list(images.items())
        self.scale = scale

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, object]:
        name, rgb = self.items[idx]
        y_hr, cbcr = rgb_to_ycbcr(rgb)
        lr = downsample_bicubic(y_hr, self.scale)
        lr_up = upsample_bicubic(lr, y_hr.shape)
        return {
            "name": name,
            "lr": _to_tensor(lr_up),
            "hr": _to_tensor(y_hr),
            "hr_rgb": torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0),
            "cbcr": torch.from_numpy(np.ascontiguousarray(cbcr)).permute(2, 0, 1).float(),
        }


def split_sources(sources: list[str], val_names: list[str]) -> tuple[list[str], list[str]]:
    train = [s for s in sources if s not in val_names]
    val = [s for s in sources if s in val_names]
    if not train or not val:
        raise ValueError("训练集或验证集为空，检查 data.val_images")
    return train, val
