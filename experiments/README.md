# experiments

每次实验一个目录，保证可复现。

## 命名

```text
YYYY-MM-DD_<模型>_<数据集>_<关键设置>
```

例如：

```text
2026-11-08_srcnn_bsds500_2x
2026-12-01_bicubic_bsds500_2x
```

## 每个实验必须包含

```text
experiments/<name>/
├─ README.md       # 目标、结论、观察
├─ config.yaml     # 完整配置
├─ command.txt     # 复现命令
├─ metrics.json    # 指标
└─ outputs/        # 小体积可视化结果（可选）
```

## 提交规则

可以提交：

- `README.md`
- `config.yaml`
- `command.txt`
- `metrics.json`
- 小体积对比图

不要提交：

- checkpoint
- 大体积视频
- TensorBoard 日志
- 数据集

## 实验 README 模板

```markdown
# 实验名称

- 日期：
- Git commit：
- 目标：
- 配置：
- 命令：
- 结果：
- 观察：
- 结论：
- 下一步：
```
