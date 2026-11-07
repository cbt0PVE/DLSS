# configs

实验配置。建议用 YAML，一次实验一份配置。

示例：

```yaml
experiment:
  name: srcnn_baseline
  seed: 42

data:
  dataset: bsds500
  scale: 2
  patch_size: 32
  batch_size: 8
  num_workers: 0

model:
  name: srcnn
  in_channels: 3
  hidden_channels: 64

train:
  epochs: 100
  lr: 0.0001
  amp: true

output:
  dir: experiments/2026-11-08_srcnn_baseline
```

规则：

- 配置里不写绝对路径。
- 数据路径用 `DLSS_DATA` 环境变量。
- 每次实验把配置复制到 `experiments/<date>_<name>/config.yaml`。
