# ROADMAP

起始时间：2026-11  
总周期：20 个月  
目标：从零基础完成一个 DLSS-inspired 的时序神经超分 + 简化帧生成原型。

## 总原则

1. **D 优先，E 扩展。** D 是复试主线，E 是加分项。
2. **每 2 周一个 sprint**，必须有可演示的产出。
3. **每 8 周留 1 周缓冲**，不把计划排满。
4. **进度落后先砍 E，不砍 D。**
5. **不追求 SOTA，追求可复现、可解释、可演示。**

## 阶段规划

| 阶段 | 时间 | 目标 | 验收标准 |
|---|---|---|---|
| Phase 0 | 第 1–2 月 | Python / PyTorch / 数学基础 | MLP MNIST > 97%；CNN MNIST > 98%；SRCNN PSNR 比 bicubic 高 0.5 dB |
| Phase 1 | 第 3–6 月 | 图形学 + Unity + 视频超分 | Unity 能输出 depth / motion vector；复现一个小规模 BasicVSR 或 RAFT |
| Phase 2 | 第 7–12 月 | D 核心原型 | 2x 540p→1080p 可训练、可推理；有 PSNR/LPIPS 和消融 |
| Phase 3 | 第 13–15 月 | D 部署与评测 | TensorRT FP16 延迟可接受；Unity demo 可运行 |
| Phase 4 | 第 16–18 月 | E 简化帧生成 | 30→60fps 插帧 demo；有遮挡处理和延迟数据 |
| Phase 5 | 第 19–20 月 | 收尾 | demo 视频、技术报告、复试 slides、模拟答辩 |

## Milestones

### M0：基础闭环（第 1–2 月）

- 配好 Windows CUDA / PyTorch 环境
- 能独立训练 MLP、CNN
- 能解释卷积、损失、反向传播
- 跑通 SRCNN 2x baseline
- 产出：`notes/reviews/M0-foundations.md`

### M1：图形学与视频超分（第 3–6 月）

- 理解坐标变换、光栅化、深度、采样/走样
- Unity URP 能输出 color / depth / motion vector
- 手写一个简单 TAA
- 复现一个小规模 BasicVSR 或 RAFT
- 产出：`notes/reviews/M1-graphics-vsr.md`

### M2：D 核心（第 7–12 月）

- 完成数据采集管线
- 完成时序超分模型：历史帧 warp + 运动矢量 + 遮挡 mask
- 2x 540p→1080p 可训练、可推理
- 完成消融：无运动矢量 / 无历史帧 / 无遮挡 mask
- 产出：`notes/reviews/M2-d-core.md`

**Gate：** 如果第 12 个月 D 还不稳定，砍掉 E，把 D 做完整。

### M3：D 部署（第 13–15 月）

- ONNX 导出
- TensorRT FP16 推理
- Unity demo
- 延迟、显存、FPS 测试
- 产出：`notes/reviews/M3-d-deploy.md`

### M4：E 扩展（第 16–18 月）

- 两帧插一帧，30→60fps
- 光流 / 运动矢量 + 遮挡 mask
- 先离线，再实时
- 产出：`notes/reviews/M4-frame-generation.md`

### M5：收尾（第 19–20 月）

- 整合 D + E
- demo 视频
- 技术报告
- 复试 slides
- 模拟答辩
- 产出：`notes/reviews/M5-final.md`

## 当前状态

| 项目 | 状态 |
|---|---|
| 仓库骨架 | 完成 |
| Windows 环境 | 完成（torch 2.11.0+cu128 / RTX 4060 Laptop / CUDA 12.8） |
| Phase 0 | 进行中（Day 2：Python + NumPy） |
| 下一个任务 | Day 2：Python + NumPy 基础练习 |

## 风险与应对

| 风险 | 应对 |
|---|---|
| 数学/图形学基础拖慢进度 | 只学当前任务用得到的部分，不系统补课 |
| 8GB 显存不够 | 2x、小模型、256 patch、AMP、梯度累积 |
| Unity 集成耗时 | 先 Python 离线跑通，再做 Unity demo |
| E 烂尾 | 第 12 个月 gate，D 不稳定就砍 E |
| 范围膨胀 | 每月 review，冻结非核心功能 |
| 两台机器不同步 | `git pull --rebase` + `HANDOFF.md` |

## 复盘节奏

- **每周日**：周报，20–30 分钟。
- **每 4 周**：月报，40 分钟。
- **每个 milestone**：阶段复盘，1–2 小时。
- **每次复盘必须带证据**：commit、配置、指标、截图。
