"""跨模块公用工具：配置、路径、随机种子、日志。

约定（见 src/README.md）：
- 路径不写死，从配置或环境变量读。
- 数据集目录默认 $DLSS_DATA，训练输出默认 $DLSS_RUNS。
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value):
    """把配置里的 ${VAR} 展开成环境变量，没定义就报错，避免静默写错路径。"""
    if isinstance(value, str):
        def repl(m):
            name = m.group(1)
            if name not in os.environ:
                raise KeyError(f"配置引用了未定义的环境变量 ${{{name}}}")
            return os.environ[name]

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | Path) -> dict:
    """读取 YAML 配置，并展开其中的环境变量。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式不对：{path}")
    return _expand_env(cfg)


def data_root() -> Path:
    root = Path(os.environ.get("DLSS_DATA", "data"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def runs_root() -> Path:
    root = Path(os.environ.get("DLSS_RUNS", "runs"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "cuda") -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，退回 CPU")
        return torch.device("cpu")
    return torch.device(name)


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


class AverageMeter:
    """滑动平均，训练日志用。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def human_time(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
