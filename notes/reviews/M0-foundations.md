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
- [ ] Windows 环境
- [ ] Python / NumPy 基础
- [ ] PyTorch 基础
- [ ] MLP / CNN
- [ ] SRCNN

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
