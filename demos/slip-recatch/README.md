# Slip Recatch：滑动重抓取

本 demo 固定机械臂，夹爪从启动实测位置逐步松开；原 SNN 的 `P(slide)` 达到阈值时立即回夹到本轮起点。
默认使用 Speck 片上卷积骨干，上位机执行原 EMA、64维 z 和分类头。所有入口收敛为 `run.sh`。

## 运行

在本机任意目录运行，会实际使能并控制夹爪：

```bash
bash /home/diaohh/neurofusion/neurofusion/demos/slip-recatch/run.sh
```

先使用 `scripts/gripper.sh` 手动调整好物体夹持位置，再运行本 demo。
收到首条新鲜分类后，程序在使能前读取夹爪位置，固定为本轮松开和回夹基准。
检出滑动后回夹，到位后保持使能；托住物体再 Ctrl+C 结束。
未检出滑动直到超时，也会保护回夹，并以 `no_slide_timeout` 退出；这种回夹不表示成功检出滑动。

以下命令从项目根目录执行：

```bash
# 仅预演：检查配置、模型路径和子进程命令，不打开设备
bash demos/slip-recatch/run.sh --dry-run
# 只读联调：读取真实触觉和夹爪反馈，不使能、不发送目标
bash demos/slip-recatch/run.sh --shadow
# CPU 骨干对照：沿用同一套配置、分类头与回夹控制
bash demos/slip-recatch/run.sh --backend cpu
# 所有选项
bash demos/slip-recatch/run.sh --help
```

shadow 不会执行松开和回夹，不能把它的超时当作实物抓取结果。

## 配置

日常只改 `configs/config.local.json`。本机设置已经迁移到此处，当前为：

```json
{
  "format": "counts-v1",
  "grip_reference": "current",
  "max_open_counts": 40,
  "open_step_counts": 1,
  "open_step_interval_s": 0.5,
  "extra_observation_s": 3,
  "slide_threshold_percent": 60,
  "expected_direction": "up"
}
```

| 字段 | 含义 |
| --- | --- |
| `grip_reference` | `"current"` 使用启动实测位置；整数则使用固定夹持/回夹编码位置 |
| `max_open_counts` | 总共最多松开多少个编码计数 |
| `open_step_counts` | 每次松开多少个计数 |
| `open_step_interval_s` | 两次松开之间的最小间隔，单位秒 |
| `extra_observation_s` | 在理论松爪时间之外增加的观察预算，单位秒 |
| `slide_threshold_percent` | 滑动触发百分比，60表示60% |
| `expected_direction` | 模型坐标中的方向标签；即时 P(slide) 触发不额外限制方向 |

程序自动计算：

- 超时 = `ceil(max_open_counts / open_step_counts) × open_step_interval_s + extra_observation_s`。
  当前为23秒；机械到位等待也占用该预算，额外观察预算不是到位后精确等待的时间。
- 会话范围 = `max_open_counts + 5`，当前45计数。
- 执行时读取真实 `min_raw/max_raw`，再把计数转换为内部闭合比例，无需手算除以1150。

例如起点197时最多松到157，检测触发和超时回夹均以197为目标。
只把 `max_open_counts` 改成80，超时自动成为43秒，会话范围成为85计数。
请求范围触及硬件下限时在使能前拒绝；现有全局收紧边界、单条目标限制、过期检查继续生效。

需要时可加入高级参数：

| 参数 | 默认值 | 单位 |
| --- | --- | --- |
| `tracking_limit_counts` | 16 | 控制器单条目标相对实测位置的计数限制 |
| `arrival_tolerance_counts` | 3 | 到位容差，编码计数 |
| `session_margin_counts` | 5 | 会话额外余量，编码计数 |
| `grip_settle_ms` | 60 | 到位稳定时间，毫秒 |
| `brake_timeout_s` | 0.8 | 回夹超时，秒 |
| `max_prediction_age_ms` / `max_feedback_age_ms` | 150 | 分类/反馈最大年龄，毫秒 |
| `max_tick_gap_ms` | 100 | 控制调度最大间隔，毫秒 |
| `slide_confirm_frames` | 1 | 连续原生窗口确认数 |

修改下次启动生效，运行中不热更新。`--config <文件>` 可临时选用其他配置。
没有本机配置时自动读取 `configs/config.example.json`；换机调参可先复制：

```bash
cp demos/slip-recatch/configs/config.example.json demos/slip-recatch/configs/config.local.json
```

旧比例格式仍可读取，用于离线回放已有日志；日常参数使用上面的计数格式。

## 模型、环境与日志

共享运行环境为项目根目录的 `.venv/live`，芯片工具叠加在 `.venv/chip-tools`。
训练权重保持在共享的 `artifacts/checkpoints/gelneuro-snn-20261111/final/all-data.pt`。
片上导出已迁移到本 demo 的 `models/speck/`，含配置、mapping和量化模型。
新机器需要准备训练权重；若导出目录尚不存在，启动器自动调用共享导出器。
每轮打开 Speck 后重新加载配置，并校验 checkpoint 与 mapping 的哈希一致。
环境重建和独立部署见 [Speck工具](../../tools/speck/README.md)。

每轮在 `outputs/slip-recatch-<时间>-<随机标识>/` 新建独立记录：

| 文件 | 内容 |
| --- | --- |
| `run.json` | 后端、启动参数、退出状态和实际分类发布频率 |
| `config.json` | 人编辑的配置快照 |
| `control/manifest.json` | 启动位置、实际换算配置和运行结果 |
| `control/trace.jsonl` | 每周期概率、状态、目标回执和夹爪实测反馈 |
| `events/predictions.jsonl` | 发布的分类和 P(slide) |
| `events/event-windows.jsonl` | 原生窗口；片上后端记录的是末层特征脉冲，不是原始 DVS |
| `events/chip-applied.json` | 本轮实际加载的芯片配置 |
| `events/report.json` | 窗口数、耗时和实际发布频率 |
| `inference.stderr.log` | 感知进程的诊断输出 |

本次整理只保留最近3轮实机记录：2026-11-18的171201、171226、171247。
历史目录名称已迁入本 demo；日志内容中的原路径作为历史信息保留，不做改写。
以后新运行都写入本目录；不会自动删除后续实验数据。

1 ms表示原生处理窗口，当前主机新分类发布仍约99 Hz、夹爪控制100 Hz。
同一批次的多个原生窗口先确认并锁存滑动，再合并发布最新结果；不能称为1000 Hz实时输出。
时序标签采用主机接收时刻，无法单凭它确定实物开始滑动的时刻。

## 文件边界

```text
demos/slip-recatch/          入口、配置、说明、导出模型与运行记录
src/neurofusion/demos/slip_recatch/
  launch.py                 子进程编排与路径管理
  runtime.py                实机控制循环与离线回放
  slip.py                   滑动确认、状态机与一维命令
  config.py                 可读配置与硬件单位换算
  live.py                   触觉窗口与分类发布
  chip_readout.py            片上输出的主机 EMA/分类头
  snn.py                    CPU 骨干对照
  session.py                日志写入
tests/demos/slip_recatch/    对应软件测试
```

通用 `adapters/robotis.py`、`adapters/speck_chip.py`、训练网络和导出器仍是共享模块。
旧的两个slip-catch脚本、纸盒/候选配置、计时教师入口、BC采集草案、重复历史报告和临时探测输出已经删除。
