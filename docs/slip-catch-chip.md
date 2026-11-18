# 使用片上模型运行滑移重抓取 demo

2026-11-18。新入口沿用原 demo 的状态机、0.6 单窗滑动门限、
每0.5秒松1计数、最多松40计数和100 Hz夹爪控制。松爪超时自动计算为23秒。
感知骨干在片上执行。
片上入口默认以每次启动时的实测夹爪位置作为本轮夹持和回夹参考。
机械臂不参与；原 `slip-catch.sh` 默认继续使用 CPU。

## 运行

本机已具备工具依赖和模型，直接运行下面的命令会实际控制夹爪：

```bash
bash /home/diaohh/neurofusion/neurofusion/scripts/slip-catch-chip.sh
```

脚本自动进入项目根目录，读取原 `configs/slip-catch-search.local.json`。
首次运行若 `outputs/speck-chip-model/speck2f-config.json` 不存在，会先调用原导出器
导出模型；每次运行都在持有 Speck 的同一进程内重新下发配置，随后开始分类。
完整环境重建步骤见 [片上部署文档](speck-on-chip.md)。

更换物体后，先用手动夹爪脚本调整好夹持位置，再启动本脚本。收到首个新鲜分类后，
控制器在使能前读取夹爪位置并固定为本轮参考；松爪过程中不会反复更新参考。
例如启动实测197，则从197开始分步松至最低157，滑动触发或超时均回夹至197。
达到40计数的松开上限后保持该位置继续检测，直到检出滑动或本轮松爪满23秒。
终端会打印本轮参考，`control/manifest.json` 的 `config.grip_closure` 保存实际生效值，
`metadata.startup_position_raw` 保存启动实测位置；顶层 `config.json` 保留原始配置快照。
会话范围自动设为总松开量加5计数余量，本轮为45计数；单条运动限制和全部故障检查继续生效。
如需固定夹持参考，在下述配置中把 `grip_reference` 改成整数编码位置，例如213。
命令行 `--grip-from-current` 可临时覆盖为实测参考；`--no-grip-from-current` 要求配置已有固定位置。

## 直接使用计数和秒调参

只编辑 `configs/slip-catch-search.local.json`，下次启动自动生效；运行中的会话不热更新。
该文件使用 `counts-v1` 格式，当前完整内容为：

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

| 字段 | 人直接填写的含义 |
| --- | --- |
| `grip_reference` | `"current"` 表示启动实测位置；整数表示固定夹持/回夹编码位置 |
| `max_open_counts` | 总共最多松开多少个编码计数；本轮40，是上轮20的两倍 |
| `open_step_counts` | 每次松开的编码计数 |
| `open_step_interval_s` | 两次松开之间至少间隔多少秒 |
| `extra_observation_s` | 在理论松爪时间上增加的观察预算，单位秒 |
| `slide_threshold_percent` | 滑动触发阈值，60表示60% |
| `expected_direction` | 模型坐标中的预期方向；原即时P(slide)触发规则不增加方向门槛 |

程序自动完成以下计算，无需手动同步填写其他字段：

- 总松爪超时 = `ceil(max_open_counts / open_step_counts) × open_step_interval_s + extra_observation_s`。
  当前为 `40 × 0.5 + 3 = 23` 秒。机械跟踪延迟占用同一预算；它不保证到达最松位置后精确等待3秒。
- 会话编码范围 = `max_open_counts + session_margin_counts`，余量默认5，本轮为45。
- 读到设备的真实 `min_raw/max_raw` 后，执行端才把编码计数转换为内部闭合比例，未写死1150。
  实测起点也只在本轮开始时读取一次。
- 默认跟踪限制16计数、到位容差3计数、稳定等待60 ms、回夹超时0.8秒，保持原控制策略。
  如有需要可添加 `tracking_limit_counts`、`arrival_tolerance_counts`、`session_margin_counts`、
  `grip_settle_ms`、`brake_timeout_s`；分类/反馈新鲜度及调度限制分别为
  `max_prediction_age_ms=150`、`max_feedback_age_ms=150`、`max_tick_gap_ms=100`。
  连续确认窗口数可用 `slide_confirm_frames` 设置，默认1。

例如只把 `max_open_counts` 改成80，超时自动成为43秒，会话范围自动成为85计数。
实际执行仍须通过硬件量程、全局收紧边界和驱动单条行程检查；请求松开范围到达硬件下限时在使能前拒绝。

启动与 `--dry-run` 都会打印可读的推导结果。顶层 `config.json` 保存原始可读输入；
`control/manifest.json` 的 `metadata.requested_config` 保存包含高级默认值的可读设置，
`config` 保存按真实量程和本轮参考换算的实际参数，便于复现。
新检出可从 `configs/slip-catch-search.example.json` 创建本地文件。
旧的比例配置及历史会话快照仍然受支持，旧格式不会自动改写。

离线回放推荐使用该轮 `control/manifest.json` 中已经换算好的 `config`。
若直接回放可读配置，需要通过 `--gripper-limits <该轮control/manifest.json>` 提供真实量程，
执行端会使用输入第一条反馈作为起点；回放数据须包含原来的 `now_ns/feedback/snn` 字段。

放置物体、夹持参考、会话行程边界沿用原 demo 的现场要求。成功回夹后继续保持使能；
托住物体后 Ctrl+C 结束并关闭使能。这是原控制器的既有行为。

不发送夹爪运动目标的联调命令：

```bash
bash /home/diaohh/neurofusion/neurofusion/scripts/slip-catch-chip.sh --shadow
```

shadow 不会实际完成初始夹持或回夹，因此可能以 `grip_timeout` 或
`no_slide_timeout` 结束；不能把这种结果当成真实抓取成功或失败。

不连接设备、只查看将执行的命令：

```bash
bash /home/diaohh/neurofusion/neurofusion/scripts/slip-catch-chip.sh --dry-run
```

需要自定义配置时使用原有 `--config`；已有其他导出模型时使用 `--chip-config`。
识别 checkpoint 与导出目录的 `mapping.json` 中的 checkpoint SHA256 必须一致，
否则在打开设备前拒绝运行，防止错误地混用骨干和分类头。

## 数据路径

```text
DVS → 片上4×4池化 → 片上四层Conv/IAF → 末层16通道Spike
    → 主机原四组时间迹/log1p → 64维z → 原分类头
    → 原P(slide)确认与锁存 → 原夹爪控制器
```

CPU 不再运行四层卷积；主机只根据真实末层 Spike 累加输出计数、推进原时间迹和分类。
每1 ms格的输出脉冲数使用原 `rate_scale=50`，时间迹衰减和分类头均沿用原模型。
空格继续衰减时间迹。NumPy实现与原PyTorch时间迹、分类概率做了数值对照测试。
固定位宽量化与异步片上运算仍可能改变实际分类，实物抓取效果需要现场观察。

`outputs/slip-catch-<session>/run.json` 记录 `backend=chip`；
`events/report.json` 应包含：

```json
{
  "backend": "chip",
  "cpu_backbone_executed": false,
  "chip_configuration_applied": true,
  "output_core": 2
}
```

`events/chip-applied.json` 保存本次部署的完整配置；`predictions.jsonl` 中
`model_id` 以 `chip-sha256:` 开头。原 `control/trace.jsonl` 的命令回执、夹爪反馈
和故障记录不变。该后端 `event-windows.jsonl` 的 `events` 是片上输出脉冲，
列为 `[feature_id,0,device_timestamp_us,0]`，不是原始DVS。
这些输出不能当成同期双触觉原始训练数据。

## 频率实测与限制

本次六秒真实静止输入测试：585条分类，5904个1 ms窗口，独立发布约98.98 Hz；
主机时间迹/分类单窗口耗时中位约0.018 ms。没有CNN末层脉冲时，分类仍基于
真实硬件水位推进空窗口，不把上一条分类重复发布成新观测。
完整shadow联调约99.22 Hz，682条反馈中均未使能且无命令回执，结束为
`grip_timeout`（参考327，实测330，shadow不发送初始夹持命令）。

另做了无夹爪的合成链路试验：向首层输入10,240个合成脉冲，末层收到7,774个，
主机z最大5.73、P(slide)最大约0.981，验证片上输出实际进入分类。
这些人工输入不是物理滑动或分类准确率证据。

超过100 Hz的实验：在与训练模型无连接的空闲核中产生时钟脉冲；既尝试
主机实时发送，也尝试由FPGA按未来设备时间每1 ms调度。后者的返回时间戳确实
间隔1 ms，但主机仍约每10 ms收到一批，约99.13个独立批次/秒。
使用SDK `read_write()` 手动调度也未改善（约98.74 Hz）。
这证明本次测试条件下更快的设备时间格没有转化成更快的主机新结果交付，
不能据此区分具体是哪一层固件、USB或SDK造成了批量交付。
没有确认可用的低延迟设置，因此正式demo没有加入额外时钟核心、人工感知事件、
主机猜测设备时间或刷新旧分类的逻辑，仍使用原硬件时间基准和过期检查。

**本次尚未实现超过100 Hz的独立感知结果发布。** 芯片内部异步计算频率、
每秒推进的1 ms时间格数、主机实际分类交付频率和100 Hz夹爪控制是不同指标。
脚本会按实际发布间隔打印结果，不能把单轮补算多个窗口称为1000 Hz实时输出。
真实运动下的持续吞吐、感知到命令延迟和保持成功率尚未测量。

原始证据在本机：

- `outputs/chip-live-smoke-20261118/`
- `outputs/chip-demo-shadow-20261118/`
- `outputs/chip-readout-chain-20261118/`
- `outputs/chip-clock-probe-20261118/`

结构化摘要见 [本次验证](validation/slip-catch-chip-20261118.json)。
