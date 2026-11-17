# src

训练与推理代码。**Windows 是主开发环境，Mac 只做审阅和小改动。**

## 目录约定

```text
src/
├─ data/            # 数据集、dataloader、预处理（含 x2 降质）
├─ models/          # 模型定义（srcnn.py）
├─ losses/          # 损失函数（l1 / l2 / charbonnier）
├─ tests/           # pytest 冒烟测试
├─ common.py        # 配置、路径、种子、日志工具
├─ metrics.py       # PSNR
├─ train.py         # 训练入口
└─ eval.py          # 评估入口（也可被 train 复用）
```

## 当前能跑的东西（M0）

```powershell
conda activate dlss

# 单元测试
python -m pytest src/tests -q

# 训练（6000 步，4060 上约 5 分钟）
python src/train.py --config configs/srcnn_x2.yaml

# 只看跑通：100 步
python src/train.py --config configs/srcnn_x2.yaml --iters 100 --tag smoke

# 单独评估某个 checkpoint
python src/eval.py --config configs/srcnn_x2.yaml `
  --checkpoint $env:DLSS_RUNS\srcnn_x2_skimage\best.pt --save-images
```

输出写到 `$DLSS_RUNS/<实验名>/`：`best.pt` / `latest.pt` / `metrics.json` /
`images/`（bicubic vs SRCNN vs GT 三联图）/ `tb/`（TensorBoard）。

配置里的路径只允许用 `${DLSS_DATA}`、`${DLSS_RUNS}` 这类环境变量，
`common.load_config` 遇到未定义变量会直接报错，避免静默写错目录。

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
