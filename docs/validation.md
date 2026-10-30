# 框架检查（2026-10-30）

- 执行环境：WSL Ubuntu-20.04，Linux Conda minist，Python 3.12.16。
- editable 安装成功，未安装或升级额外模型依赖。
- `python -m unittest discover -s tests -v`：10 项行为检查通过。
- 50/100 Hz 逻辑控制时刻的合成演示通过，输出分别为 `outputs/demo-50hz.json`、`outputs/demo-100hz.json`。
- 事件开/关对照表明两帧间的 toy 动作可不同；没有调用传感器、学习网络或机器人。
- 迁移前后原始数据 SHA-256 一致；本次没有重连 Speck。
- 文档链接、项目根、正式代码目录与 M01 历史通过状态核对完成。

这些结果不表示真实模型、实时频率、端到端延迟或机械响应通过验收。
