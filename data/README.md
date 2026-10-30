# 数据约定

原始 Speck 样例保留在 `../../speck-lab/data/`；不覆盖或重复生成已有实验文件。

- `timestamped-capture.csv`：10,873 个真实事件，5.004639 秒，有效设备时间。
- `first-capture.csv`：首次失败样例，设备时间全零，不能当作正常时间回放。
- 两者均没有主机接收时刻，不能用于计算 USB 延迟或端到端反应。

正式新数据规划放在 `data/raw/<run_id>/`、派生数据放在 `data/processed/<run_id>/`，目前尚未创建真实采集回合。大数据、模型权重与 `outputs/` 不进 Git；需要另行安排备份。

合成演示结果写入 `outputs/`，明确标记 synthetic，不与真实硬件实验数据混用。
