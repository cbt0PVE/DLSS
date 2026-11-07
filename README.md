# DLSS

> **DLSS-inspired temporal neural upscaler and simplified frame generation prototype.**
>
> 这是一个学习项目，不是 NVIDIA DLSS 的复刻，也不隶属于 NVIDIA。
> DLSS 是 NVIDIA 的商标。

## 项目状态

**Phase 0 — 环境搭建 / 基础准备**

- [x] GitHub 仓库创建
- [x] 本地项目骨架
- [ ] Windows 环境验证
- [ ] 第一个 SRCNN baseline

当前重点：Python / PyTorch 基础，暂不碰 Unity。

## 目标

用 20 个月，从零开始做一个 DLSS 风格的原型：

1. **D：时序神经超分**
   - 2x，540p → 1080p
   - Unity URP 采集数据与实时 demo
   - PyTorch 训练，ONNX / TensorRT 部署

2. **E：简化帧生成**
   - 30 → 60fps，两帧插一帧
   - 先离线验证，再尝试实时
   - 不做 DLSS 3/4 级别的多帧生成

## 非目标

- 不复刻完整 NVIDIA DLSS 2/3/4。
- 不做 AAA 级画质、商业级泛化和多帧生成。
- 不把 DLSS SDK 或商业游戏的效果说成自己的成果。

## 技术栈

- **训练**：Python、PyTorch、NumPy、OpenCV
- **引擎**：Unity URP + C#（第二年可选 Unreal）
- **部署**：ONNX、TensorRT、Unity 推理插件
- **工具**：Git、VS Code、TensorBoard

## 仓库结构

```text
.
├─ README.md
├─ ROADMAP.md              # 20 个月路线图
├─ HANDOFF.md              # Mac / Windows 交接单
├─ docs/
│  ├─ setup_windows.md     # Windows 环境配置
│  ├─ learning_path.md     # 学习路径与资源
│  └─ decisions.md         # 技术决策记录
├─ notes/
│  ├─ weekly/              # 周报
│  ├─ monthly/             # 月报
│  ├─ reviews/             # 阶段复盘
│  └─ papers/              # 论文笔记
├─ src/                    # 训练与推理代码
├─ configs/                # 实验配置
├─ scripts/                # 辅助脚本
├─ experiments/            # 每次实验的配置、命令、指标
├─ assets/samples/         # 少量可公开的示例图
├─ data/                   # 本地数据，gitignore
├─ checkpoints/            # 模型权重，gitignore
└─ runs/                   # TensorBoard 日志，gitignore
```

## 快速开始

- Windows 环境配置：[docs/setup_windows.md](docs/setup_windows.md)
- 20 个月路线：[ROADMAP.md](ROADMAP.md)
- 学习路径：[docs/learning_path.md](docs/learning_path.md)
- 当前交接状态：[HANDOFF.md](HANDOFF.md)

## 两台机器

- **Windows（RTX 4060 Laptop）**：训练、Unity、TensorRT、实验。
- **Mac**：规划、文档、周报/月报、代码审阅。

两边通过 Git 同步。每次开始先 `git pull --rebase`，结束前 commit + push。

## 免责声明

本项目仅用于学习与研究，与 NVIDIA 无关联。DLSS 是 NVIDIA Corporation 的商标。
仓库中的代码、数据和素材应遵守各自的许可证；不要提交商业游戏资产或 NVIDIA 专有 SDK。
