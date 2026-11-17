# M0 基础阶段

状态：进行中  
时间：2026-11 ~ 2027-01  
阶段：Phase 0

## 目标

- 配好 Windows CUDA / PyTorch 环境
- 掌握 Python、NumPy、PyTorch 基础
- 理解 MLP、CNN、损失函数、反向传播
- 跑通 SRCNN 2x baseline

## 验收标准

| 项目 | 目标 |
|---|---|
| PyTorch CUDA | `torch.cuda.is_available() == True` |
| MLP MNIST | 测试集准确率 > 97% |
| CNN MNIST | 测试集准确率 > 98% |
| SRCNN | PSNR 至少比 bicubic 高 0.5 dB |
| Git | 每周至少 3 次有意义的 commit |
| 笔记 | 每周周报 + 月末复盘 |

## 当前进度

- [x] 仓库骨架
- [x] Windows 环境（torch 2.11.0+cu128 / RTX 4060 / CUDA 12.8）
- [x] SRCNN 2x baseline：27.390 dB vs bicubic 26.797 dB = **+0.593 dB**
- [ ] Python / NumPy 基础（第 1 周学习任务）
- [ ] PyTorch 基础（第 3 周）
- [ ] MLP / CNN MNIST（第 4 周）

> 说明：工程链路先跑通了（数据管线 → 训练 → 评估 → 实验记录），
> 基础理论按 `docs/guides/stage-01-phase0.md` 的 8 周计划补。
> MLP / CNN 的验收没有单独建 experiment，会在第 4 周补跑并记录。

## 已有证据

| 项目 | 证据 |
|---|---|
| 环境 | `docs/setup_windows.md` 第 5 节实测输出 |
| SRCNN | `experiments/2026-11-18_srcnn_skimage_2x/`（config + metrics + 对比图） |
| 复现 | `python src/train.py --config configs/srcnn_x2.yaml`，519 s |
| 测试 | `python -m pytest src/tests -q`，10 passed |

## 风险

- 线代生疏：遇到再补，不系统补课。
- 8GB 显存：先小图、小模型。
- 三分钟热度：用 2 周 sprint 和每日 commit 保持节奏。

## 复盘

阶段结束后填写。重点回答：

1. 能不能独立写出训练循环？
2. 能不能解释卷积和反向传播？
3. 遇到报错能不能自己定位？
4. 是否准备好进入图形学阶段？
