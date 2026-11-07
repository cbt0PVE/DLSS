# src

训练与推理代码。**Windows 是主开发环境，Mac 只做审阅和小改动。**

## 目录约定

```text
src/
├─ data/        # 数据集、dataloader、预处理
├─ models/      # 模型定义
├─ losses/      # 损失函数
├─ train.py     # 训练入口（后续添加）
└─ eval.py      # 评估入口（后续添加）
```

## 约定

- Python 3.11。
- 所有路径通过配置或环境变量传入，不写死 `D:\...`。
- 数据集路径默认从 `DLSS_DATA` 读取。
- 训练配置放 `configs/`，每次实验复制一份到 `experiments/`。
- 不提交 checkpoint、数据集、ONNX、TensorRT engine。
- 每个实验必须能用一个命令复现。

## 代码风格

- 先写能跑的，再重构。
- 函数职责单一，训练循环和模型定义分开。
- 关键超参写进 config，不散落在代码里。
- 每个模块至少有一个最小测试或可运行示例。
