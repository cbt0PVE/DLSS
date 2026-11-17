# SRCNN x2 baseline（skimage 10 张图）

- 日期：2026-11-18
- Git commit：`e2447ed`
- 目标：M0 验收项 —— x2 超分 PSNR 比 bicubic 高 0.5 dB，且整条链路可复现。

## 配置

- 数据：`skimage.data` 10 张真实照片/纹理，train 8 / val 2（coffee、grass）
- 降质：HR → Y 通道 → bicubic ↓2 → bicubic ↑2 → 网络输入（同 bicubic 基线输入）
- 模型：SRCNN 9-5-5，64/32 通道，残差预测（末层零初始化），57.3K 参数
- 训练：L1 loss，Adam，lr 1e-4，余弦退火到 1e-6，20000 步，batch 16，patch 96，AMP fp16
- 评估：Y 通道 PSNR，裁边界 8px

## 命令

```powershell
python src/train.py --config configs/srcnn_x2.yaml
python scripts/compare_crop.py --config configs/srcnn_x2.yaml `
  --checkpoint $env:DLSS_RUNS\srcnn_x2_skimage\best.pt `
  --image coffee --center 0.52 0.46 --size 160 --zoom 2 `
  --out experiments/2026-11-18_srcnn_skimage_2x/outputs/coffee_crop.png
```

## 结果

| step | val PSNR | bicubic | gain |
|---|---|---|---|
| 2000 | 27.02 | 26.797 | +0.22 |
| 6000 | 27.25 | 26.797 | +0.46 |
| 10000 | **27.39** | 26.797 | **+0.59** |
| 14000 | 27.35 | 26.797 | +0.56 |
| 20000 | 27.35 | 26.797 | +0.56 |

逐图（best @ 10000）：

| image | bicubic | SRCNN | gain |
|---|---|---|---|
| coffee | 29.869 | 30.765 | +0.897 |
| grass | 23.725 | 24.015 | +0.289 |
| **mean** | **26.797** | **27.390** | **+0.593** |

总耗时：519 s（RTX 4060 Laptop，约 38 it/s 稳态）。
对比图：`outputs/coffee_crop.png`（bicubic 边缘发虚、有振铃；SRCNN 边缘更实）。

## 观察

1. **6000 步到 +0.46 dB，10000 步到 +0.59 dB，之后不再涨到 20000 步反而略降。**
   说明在 8 张训练图上，模型容量/数据量在第 10k 步左右就到顶了，继续训只是过拟合。
2. 平滑的 coffee 涨 0.90 dB，高频纹理 grass 只涨 0.29 dB。
   SRCNN 只学高频修正，而纹理区的信息已经被 bicubic 降质阶段丢掉了 —— 这是"经典降质设定"的天花板。
3. 训练 loss 从 0.0133 只降到 0.0120，降幅很小；PSNR 却涨了 0.6 dB。
   再次说明 loss 和 PSNR 不是线性关系，看 loss 判断好坏不可靠。

## 结论

M0 验收达成：**+0.593 dB > 0.5 dB 目标**，且 baseline 公平（同输入、同评价脚本）。
但结论只在这套自建数据上成立，不能和论文数字比。

## 下一步

1. 换 BSDS500（200 张训练）重跑同配置，看增益是否同向、是否更大。
2. 消融：(a) 去掉残差预测；(b) L1 换 L2；(c) kernel 改 5-3-3。
3. 加 SSIM，为后面 M2 的 LPIPS 铺路。
