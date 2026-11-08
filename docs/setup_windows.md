# Windows 环境配置

目标机器：Windows 笔记本，RTX 4060 Laptop 8GB，32GB RAM。  
目标：PyTorch 能调用 CUDA，能跑通 Day 1。

## 0. 前置检查

打开 PowerShell：

```powershell
nvidia-smi
```

能看到 `RTX 4060 Laptop GPU` 和驱动版本即可。

> 先不要安装完整 CUDA Toolkit。PyTorch 的 CUDA wheel 自带运行时，TensorRT 阶段再处理 CUDA Toolkit / cuDNN / TensorRT 的版本匹配。

## 1. 安装基础工具

- Git for Windows
- Miniconda
- VS Code
  - 扩展：Python、Jupyter
- 可选：GitHub Desktop（不熟悉 Git 时可用）

本项目仓库固定放在：

```text
C:\Users\10545\Desktop\DLSS
```

数据集和训练输出放在 D 盘（体积大、不进 Git）：

```text
D:\DLSS_data
D:\DLSS_runs
```

要求：路径全英文、无空格、不在 OneDrive 同步目录里。
（本机桌面 `C:\Users\10545\Desktop` 未接入 OneDrive，可以使用。）

## 2. Clone 仓库

```powershell
cd C:\Users\10545\Desktop
git clone git@github.com:cbt0PVE/DLSS.git DLSS
cd C:\Users\10545\Desktop\DLSS
```

> 本机已用 SSH 方式 clone（`git@github.com:cbt0PVE/DLSS.git`），SSH key 已配置并验证通过。
> 如果换机器没有 SSH key，用 HTTPS 地址 `https://github.com/cbt0PVE/DLSS.git`。

如果仓库里已有初始提交，直接 clone 即可；如果之前 clone 过空仓库，先：

```powershell
git pull --rebase
```

## 3. 创建 Python 环境

打开 **Anaconda Prompt**：

```powershell
conda create -n dlss python=3.11 -y
conda activate dlss
python -m pip install --upgrade pip
```

> **本机已完成**：Miniconda 在 `C:\Users\10545\miniconda3`，环境 `dlss` 已创建（Python 3.11.15）。
> 新开终端后先 `conda activate dlss` 再干活。

## 4. 安装 PyTorch CUDA 版

打开 <https://pytorch.org/get-started/locally/>，选择：

- OS: Windows
- Package: Pip
- Language: Python
- Compute Platform: CUDA 12.x

复制网站给出的命令执行。下面只是示例，版本可能已经过时：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> **本机实际使用**（官方源下载会卡住，实测 10 分钟无进度）：
>
> ```powershell
> pip install --no-cache-dir torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu128
> ```
>
> 装到的是 `torch 2.11.0+cu128` / `torchvision 0.26.0+cu128` / `torchaudio 2.11.0+cu128`，
> 下载速度约 26 MB/s。如果这个镜像也慢，可以换 `https://mirror.sjtu.edu.cn/pytorch-wheels/cu128`
> 或清华的 anaconda 镜像。
> 注意：`mirrors.aliyun.com/pytorch-wheels/` 只是文件列表，不能直接当 pip index 用。

不要安装成 CPU 版。装完后再装项目依赖：

```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 5. 验证

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

期望输出包含：

```text
True
NVIDIA GeForce RTX 4060 Laptop GPU
```

**本机 2026-11-08 实测结果（Day 1 已通过）：**

```text
python      : 3.11.15 64bit
torch       : 2.11.0+cu128
cuda avail  : True
cuda ver    : 12.8
device      : NVIDIA GeForce RTX 4060 Laptop GPU
cudnn       : 91900
capability  : (8, 9)
vram total  : 8.0 GB
driver      : 617.14
```

完整验证脚本：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); x=torch.randn(2048,2048,device='cuda'); print((x@x).sum().item())"
```

如果 `torch.cuda.is_available()` 是 `False`：

1. 确认装的是 CUDA wheel，不是 CPU wheel。
2. 卸载重装：
   ```powershell
   pip uninstall torch torchvision torchaudio -y
   ```
   然后按 pytorch.org 的命令重装。
3. 更新 NVIDIA 驱动。
4. 确认没有在 WSL 里装错环境。

## 6. Git 配置

如果这台 Windows 还没配置过 Git：

```powershell
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

建议使用与 Mac 相同的 GitHub 账号。

## 7. 电源与性能设置

- 插电使用。
- 打开厂商控制中心的高性能 / 独显直连模式。
- Windows 电源计划：高性能 / 最佳性能。
- 训练时关闭游戏、浏览器大页面、Unity 编辑器（如果不在用）。
- 笔记本垫高，注意温度。

## 8. 目录约定

| 路径 | 用途 |
|---|---|
| `C:\Users\10545\Desktop\DLSS` | Git 仓库，代码和文档 |
| `D:\DLSS_data` | 数据集，不提交 |
| `D:\DLSS_runs` | checkpoint、TensorBoard、临时输出，不提交 |

可以把数据路径写进环境变量：

```powershell
setx DLSS_DATA "D:\DLSS_data"
```

新开终端后生效。本机已设置：`DLSS_DATA = D:\DLSS_data`。

## 9. Day 1 验收

- [x] `nvidia-smi` 正常（RTX 4060 Laptop GPU，驱动 617.14，8188 MiB）
- [x] `torch.cuda.is_available()` 为 `True`
- [x] GPU 名称为 `RTX 4060 Laptop GPU`
- [x] `git status` 干净
- [x] 把验证结果写入 `notes/weekly/2026-W46.md`

## 10. 常见问题

**pip 下载很慢**  
换时间段、挂代理，或使用可信镜像；不要从来路不明的 wheel 安装。
本机实测：官方 `download.pytorch.org` 下 torch 会卡死，改用南大镜像
`https://mirrors.nju.edu.cn/pytorch/whl/cu128` 后 26 MB/s。

**conda 创建环境很慢**  
可以改用 `python -m venv .venv`，但团队统一用 conda 更省事。

**`torch.cuda.is_available()` 为 False**  
大概率装成了 CPU 版，按第 5 节重装。

**显存不足**  
先用 256×256 patch、batch 4–8、AMP；不要一上来跑 1080p 全图。

**路径有中文或空格**  
本机路径已是纯英文（`C:\Users\10545\Desktop\DLSS`），无需更换。
