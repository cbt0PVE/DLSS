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

项目路径使用英文，例如：

```text
D:\DLSS
D:\DLSS_data
D:\DLSS_runs
```

不要放在 OneDrive、中文路径或桌面同步目录里。

## 2. Clone 仓库

```powershell
cd /d D:
git clone https://github.com/<YOUR_GITHUB_USERNAME>/DLSS.git D:\DLSS
cd /d D:\DLSS
```

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

不要安装成 CPU 版。装完后再装项目依赖：

```powershell
pip install -r requirements.txt
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
| `D:\DLSS` | Git 仓库，代码和文档 |
| `D:\DLSS_data` | 数据集，不提交 |
| `D:\DLSS_runs` | checkpoint、TensorBoard、临时输出，不提交 |

可以把数据路径写进环境变量：

```powershell
setx DLSS_DATA "D:\DLSS_data"
```

新开终端后生效。

## 9. Day 1 验收

- [ ] `nvidia-smi` 正常
- [ ] `torch.cuda.is_available()` 为 `True`
- [ ] GPU 名称为 `RTX 4060 Laptop GPU`
- [ ] `git status` 干净
- [ ] 把验证结果写入 `notes/weekly/2026-W46.md`

## 10. 常见问题

**pip 下载很慢**  
换时间段、挂代理，或使用可信镜像；不要从来路不明的 wheel 安装。

**conda 创建环境很慢**  
可以改用 `python -m venv .venv`，但团队统一用 conda 更省事。

**`torch.cuda.is_available()` 为 False**  
大概率装成了 CPU 版，按第 5 节重装。

**显存不足**  
先用 256×256 patch、batch 4–8、AMP；不要一上来跑 1080p 全图。

**路径有中文或空格**  
换到 `D:\DLSS` 这类纯英文路径。
