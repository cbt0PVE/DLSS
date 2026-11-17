# HANDOFF

最后更新：2026-11-18  
当前阶段：Phase 0 — 基础闭环（工程链路已跑通，理论待补）  
当前机器：Windows（训练 / 开发）

## 当前目标

工程侧：M0 已达成（SRCNN +0.593 dB）。
学习侧：按 `docs/guides/stage-01-phase0.md` 的 8 周计划补基础，学完做验收。

## 已完成

- GitHub 仓库 `DLSS`（`git@github.com:cbt0PVE/DLSS.git`）
- Windows 环境：conda `dlss`（Python 3.11.15）+ torch 2.11.0+cu128 + CUDA 12.8
- 数据管线：`src/data/sr_dataset.py`（bicubic x2 降质、Y 通道、在线 patch 采样）
- 模型：`src/models/srcnn.py`（9-5-5 / 64-32 通道 / 残差，57.3K 参数）
- 训练 + 评估：`src/train.py`、`src/eval.py`（AMP、余弦退火、best.pt、metrics.json）
- 实验记录：`experiments/2026-11-18_srcnn_skimage_2x/`（含局部放大对比图）
- 单元测试：`src/tests/test_core.py`，10 passed
- 文档：`docs/guides/stage-01-phase0.md`（2 个月学习清单）

## 正在运行

- 无

## 下一步

1. **人**：学 `docs/guides/stage-01-phase0.md`，第 8 周做三个消融实验。
2. **代码**：换 BSDS500（200 张）复跑同配置，验证增益方向一致且更大。
3. 补 MLP / CNN MNIST 两个验收项，写进 `notes/reviews/M0-foundations.md`。

## 卡点

- 只有 10 张示例图，train 8 / val 2，指标方差大，不能和论文数字比。
- 第 10000 步后 val PSNR 不再涨（27.35 附近），说明当前数据量下模型已到顶。

## 相关 commit

- `f9f0c48` chore: initial project scaffold
- `db4f757` docs: update handoff after initial push
- `bbc8c6e` chore(win): Day 1 environment verified
- 本次提交：SRCNN baseline + 阶段学习清单
