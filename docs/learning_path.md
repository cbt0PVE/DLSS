# 学习路径

这份文档回答两个问题：**现在学什么**、**学到什么程度算够**。

## 原则

1. **项目驱动，不系统补课。** 遇到问题再补数学/图形学。
2. **每 2 周必须有可演示产出。**
3. **先跑通，再优化，再讲原理。**
4. **不追 SOTA。** 能复现一个 baseline 并解释清楚，比堆模型更有价值。

## 阶段与资源

### Phase 0：Python + PyTorch + 数学（第 1–2 月）

- Python 基础、NumPy
- 张量、autograd、优化器
- MLP、CNN、损失函数、反向传播
- 线代复习：向量、矩阵乘法、转置、范数
- 微积分：偏导、链式法则

资源：

- [d2l.ai](https://zh.d2l.ai/)：主力教材，先看线性回归、MLP、CNN
- PyTorch 官方教程：Tensors、Autograd、Training a classifier
- 3Blue1Brown：线代系列挑 4–5 集
- CS231n：卷积、反向传播部分

验收：MLP MNIST > 97%，CNN MNIST > 98%，SRCNN 比 bicubic 高 0.5 dB。

### Phase 1：图形学 + Unity + 视频超分（第 3–6 月）

- 坐标变换、投影、光栅化、深度缓冲
- 采样、走样、重建
- Unity 编辑器、C#、URP、Renderer Feature
- 运动矢量、jitter、TAA
- 视频超分、光流

资源：

- GAMES101：光栅化、几何、光线追踪基础
- LearnOpenGL：渲染管线、坐标系统、深度测试
- Unity URP 官方文档
- BasicVSR、EDVR、RAFT 论文

验收：Unity 能输出 depth / motion vector；手写一个简单 TAA；复现小规模 BasicVSR 或 RAFT。

### Phase 2：D 核心（第 7–12 月）

- 数据采集与配对
- 历史帧 warp
- 运动矢量对齐
- 遮挡 / disocclusion mask
- 时序损失、感知损失
- 消融实验

关键论文：

- SRCNN、ESPCN、VDSR
- RCAN、SwinIR（选读）
- BasicVSR、BasicVSR++
- RAFT（光流）

验收：2x 540p→1080p 可训练、可推理，有 PSNR/LPIPS 和消融。

### Phase 3：D 部署（第 13–15 月）

- ONNX 导出
- TensorRT FP16
- Unity 推理插件
- 延迟、显存、FPS 测试

资源：

- ONNX 官方文档
- TensorRT Developer Guide
- `trtexec` 文档

验收：TensorRT 延迟可接受，Unity demo 可运行。

### Phase 4：E 简化帧生成（第 16–18 月）

- 光流 / 运动矢量
- 前后帧 warp
- 遮挡 mask
- 插帧、混合、refinement
- 延迟与 frame pacing

验收：30→60fps 插帧 demo，有遮挡处理和延迟数据。

### Phase 5：收尾（第 19–20 月）

- 整合 D + E
- demo 视频
- 技术报告
- 复试 slides
- 模拟答辩

## 数学补课顺序

1. 向量、矩阵乘法、转置、范数
2. 偏导、链式法则
3. 点积、叉积、齐次坐标
4. 采样、走样、傅里叶直觉

不用一次学完。每遇到一个公式，回到对应知识点补 30 分钟即可。

## 论文阅读顺序

1. SRCNN — 图像超分开山工作
2. ESPCN — 亚像素卷积、实时超分
3. VDSR — 深层超分、残差
4. BasicVSR — 视频超分的时序传播
5. RAFT — 光流估计
6. TAA / TAAU 相关资料 — 时序累积与 jitter

读论文的方法：

1. 先看摘要、图、结论。
2. 用自己的话写 5 行总结。
3. 只复现核心模块，不追求完整复现。
4. 记录“它解决了什么问题、代价是什么”。

## 每周节奏

- 周一至周六：每天 3 小时。
- 周日：周报 + 休息，或补进度。
- 每 2 周：一个 sprint demo。
- 每 8 周：1 周缓冲。

每天 3 小时建议分配：

- 30 分钟：理论 / 论文 / 视频
- 2 小时：写代码
- 30 分钟：笔记 + Git commit

## 不要做的事

- 不要先花半年学完数学。
- 不要一上来读 DLSS 专利。
- 不要同时学 Unity + Unreal + Vulkan + DL。
- 不要用 C++ 写训练框架。
- 不要在没有 baseline 的情况下堆模型。
- 不要用 PSNR 单独判断画质。
