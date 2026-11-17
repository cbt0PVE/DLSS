# 阶段一 / Phase 0：基础闭环（第 1–2 月）

> 这一份文档替代了原来的 `M0-foundations.md`。
> 阶段长度：**8 周 / 约 2 个月**，每周 6 天 × 3 小时 ≈ 144 小时。
> 学习配比建议：**60% 动手改代码、25% 读资料、15% 写笔记**。
> 只读不写是这个阶段最常见的坑：看得懂 ≠ 改得动。

---

## 这个阶段已经产出了什么

| 产出 | 状态 |
|---|---|
| conda 环境 + CUDA PyTorch | ✅ torch 2.11.0+cu128 / RTX 4060 |
| 可训练的数据管线（bicubic x2 降质） | ✅ `src/data/sr_dataset.py` |
| SRCNN baseline，比 bicubic **+0.59 dB** | ✅ `experiments/2026-11-18_srcnn_skimage_2x/` |
| 评估脚本 + 三联对比图 | ✅ `src/eval.py`、`scripts/compare_crop.py` |
| 实验记录规范 + 单元测试 | ✅ `experiments/`、`src/tests/` |

**所以这个阶段的重点不是"从零跑通"，而是"把跑通的东西彻底想明白"。**
下面第 5、6 周的内容你已经见过一遍了，那两周的任务是补原理、能复述、能改坏再修好。

---

## 第 1 周：Python + NumPy

**目标**：能写脚本处理图像数组，不用查教程。

- `list/dict/切片/函数/类`、`with open`、`pathlib`、f-string
- NumPy：`shape/dtype/axis`、视图 vs 拷贝、广播、`uint8↔float32`
- 图像数组的 `HWC` vs 张量的 `CHW`
- 向量化思维：用 `np.mean((a-b)**2)` 代替 for 循环

**对应代码**：`src/data/sr_dataset.py` 全文、`src/metrics.py`

**自测**
1. `a[::2, ::2]` 是视图还是拷贝？之后 `a[:]=0` 会不会影响它？
2. `(400,600,3)` 的 uint8 图归一化后参与减法，为什么会得到 dtype 错误？
3. 手写：读一张图 → 转 float32 → 算它和自身翻转版的 PSNR。

---

## 第 2 周：线代、微积分直觉、手写梯度下降

**目标**：解释"反向传播在干什么"，不用推公式但要看懂链式法则。

- 矩阵乘法、转置、范数；卷积写成矩阵乘法的直觉
- 偏导、链式法则、梯度方向
- 用 NumPy 实现线性回归 + 手写 SGD，观察 loss 曲线
- 学习率过大/过小的典型曲线形状（发散 / 爬得慢）

**对应代码**：无（这是唯一一个纯 NumPy 周，建议在 `src/tests/` 下写个 `train_linreg.py` 练手，别提交也行）

**资料**：d2l.ai 线性回归一章；3Blue1Brown 线代 3–4 集

**自测**
1. 为什么参数更新是 `w -= lr * grad` 而不是 `+=`？
2. lr=1e-1 时 loss 变成 NaN，最可能的原因是什么？
3. 写出交叉熵对 logits 的梯度（结果是 `softmax - onehot`）。

---

## 第 3 周：PyTorch 基础

**目标**：能默写训练五步循环。

- `tensor`、`device`、`requires_grad`
- `autograd`：为什么 `loss.backward()` 之后梯度挂在 `.grad` 上
- `optimizer`、`zero_grad`、`state_dict`
- **用 PyTorch 重写第 2 周的线性回归**，和 NumPy 版对拍 loss

**对应代码**：`src/train.py` 的 118–140 行（训练循环）

**自测**
1. 忘记 `zero_grad()` 时，梯度是"被覆盖"还是"被累加"？哪个数字能立刻看出来？
2. `with torch.no_grad():` 省的是什么？为什么验证必须加？
3. `model.train()` 和 `model.eval()` 在什么层上有区别？

---

## 第 4 周：CNN 与 MNIST

**目标**：MNIST 测试集 > 97%（M0 原定验收线）。

- 卷积核、stride、padding、输出尺寸 `O=(I+2P−K)/S+1`
- 感受野递推 `RF += (K−1)×之前的 stride 积`
- 池化 vs stride 卷积
- `DataLoader`、`CrossEntropyLoss`、训练/测试循环

**对应代码**：`src/models/srcnn.py`（看 `padding=k//2` 和 `self.border` 的算法）

**自测**
1. 9-5-5 三层的感受野是多少？为什么等于 `border*2+1`？
2. 把池化换成 stride=2 卷积，参数量和感受野各怎么变？
3. 本模型 57,281 个参数，手算一遍对不对得上。

---

## 第 5 周：图像、插值、降质、PSNR（补原理）

**目标**：解释清楚"为什么 bicubic 后的图看着糊，PSNR 却不低"。

- 最近邻/双线性/双三次：邻居数 1/4/16 与核函数
- 降采样 = 低通 + 抽样；不低通 → 混叠（摩尔纹）
- `cv2.resize` 缩小时会自动放大滤波核（自带抗混叠），和"每 2 个取 1 个"不同
- **PSNR 强依赖降质协议**，跨论文比数字没意义
- PSNR 的盲区：越模糊越安全

**对应代码**：`src/data/sr_dataset.py` → `imresize/downsample_bicubic/upsample_bicubic`

**自测**
1. MSE 减半，PSNR 涨多少 dB？
2. 换成 `INTER_NEAREST` 降质，PSNR 会涨还是会跌？为什么？
3. 为什么评估要裁掉边界 8 像素？（提示：感受野）

---

## 第 6 周：超分任务与 SRCNN

**目标**：读完 SRCNN 论文，能说出它每一层在干什么。

- SISR 是病态问题：一个 LR 像素对应多个 HR 解
- 经典设定 vs 真实降质（blur + noise + JPEG）→ domain gap
- SRCNN 三段式：patch 提取（9×9）→ 非线性映射（5×5）→ 重建（5×5）
- 为什么本项目用**残差预测 + 末层零初始化**：让初始输出严格等于 bicubic
- AMP 是什么、为什么需要 GradScaler、为什么裁剪梯度前要 `unscale_`

**对应代码**：`src/models/srcnn.py`、`src/train.py` 的 AMP 部分

**资料**：SRCNN 原文（arXiv:1501.00092）第 3 节；PyTorch `torch.amp` 文档

**自测**
1. 为什么第一层用 9×9 大核、后面用 5×5？
2. 去掉残差预测，初始 PSNR 会变成多少？为什么？
3. fp16 训练时梯度为什么会下溢？GradScaler 怎么解决？

---

## 第 7 周：实验方法（比调参重要）

**目标**：能设计一个"能证伪自己想法"的消融实验。

- baseline 三要素：同输入、同评价脚本、同数据划分
- 读曲线：训练降+验证涨=欠拟合；训练降+验证平=过拟合或数据问题
- 消融：一次只改一个变量
- 记录规范：`experiments/<日期>_<模型>_<数据>_<设置>/`，必须含 config / 命令 / 指标
- 为什么 `metrics.json` 要存 config：三个月后你会忘了当时改了什么

**对应代码**：`experiments/2026-11-18_srcnn_skimage_2x/README.md`（照着写你自己的）

**自测**
1. 观察记录里"10000 步之后不再涨"，用三句话说明你的判断和下一步。
2. 想证明"残差预测有用"，你改哪一行、看哪个数字、预期差多少？
3. 训练 loss 降了但 PSNR 不涨，可能有哪些原因（列 3 个）？

---

## 第 8 周：实战消融 + 复盘

**任务（三选二，做完写进 `experiments/`）**

1. **去残差**：`model.residual: false`，跑 3000 步，比较 1000/2000/3000 步的 PSNR 曲线。
   预测：前期明显更慢，长期可能追平。
2. **换损失**：`loss: l2`，跑 3000 步。预测：PSNR 可能更高，但边缘更软。
   用 `compare_crop.py` 出图验证。
3. **换核**：`kernels: [5,3,3]`，先手算 border 和参数量，再跑 3000 步验证预测。

**复盘产出**：填 `notes/reviews/M0-foundations.md`，必须带 commit、配置、指标、截图。
这是复试材料里最值钱的一份东西。

---

## 阶段验收（8 周后找我）

我会给你 **10 道口述题**（每模块 1–2 道）和 **2 个现场改代码任务**，例如：

- 把 `border` 改成 4，解释 PSNR 会升还是降，然后跑一次验证；
- 让 `PairedPatchDataset` 支持 4x，指出哪几行必须改，改完跑 20 步确认不炸。

两题都过 → 进阶段二（图形学 + Unity + 时序超分）。

## 这个阶段**不要**碰

- Unity / 渲染管线 / TAA / DLSS 专利 → 阶段二
- Transformer / SwinIR / 扩散 → 阶段三之后
- CUDA 编程 / TensorRT → 阶段三
- 分布式训练、混合精度深度调优 → 真正训不动再说

## 阶段二预告（第 3–6 月）

坐标变换、投影、光栅化、深度缓冲 → Unity URP 输出 color/depth/motion vector →
手写 TAA → 复现小规模 BasicVSR 或 RAFT。那一阶段的知识密度比这一阶段高，
现在把卷积、感受野、评估口径打牢，否则阶段二会一直卡在"为什么 warp 完是错的"。
