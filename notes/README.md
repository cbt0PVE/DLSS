# 笔记系统

这个目录负责“阶段性总结和规划”，不是流水账。

## 三层总结

| 类型 | 位置 | 频率 | 目的 |
|---|---|---|---|
| 周报 | `weekly/` | 每周日 | 本周产出、卡点、下周计划 |
| 月报 | `monthly/` | 每 4 周 | 进度、范围、风险、下月重点 |
| 阶段复盘 | `reviews/` | 每个 milestone | 目标 vs 实际，是否进入下一阶段 |

## 写作要求

- 每次总结不超过 1 页。
- 必须带证据：commit hash、配置、指标、截图。
- 不要写“学习了 CNN”，要写“CNN 在 MNIST 上到 98.6%，比 MLP 高 1.2 个点”。
- 失败记录比成功记录更有价值：写清楚报错、原因、怎么解决。

## 节奏

- 周日：20–30 分钟周报。
- 每 4 周：40 分钟月报。
- 每个 milestone：1–2 小时阶段复盘。
- 复盘结束后，更新 `ROADMAP.md` 和 `HANDOFF.md`。

## 模板

- `weekly/TEMPLATE.md`
- `monthly/TEMPLATE.md`
- `reviews/TEMPLATE.md`
- `papers/TEMPLATE.md`
