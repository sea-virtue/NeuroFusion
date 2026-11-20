# Balloon Grip（砝码振荡下的气球夹持）

气球夹在 GelSight 和 GelNeuro 之间，砝码挂在气球下方。砝码下拉时，气球拉长，夹持处变细，夹爪小步收紧。回弹时，夹持处变宽，夹爪小步张开。控制器从连续 GelSight 帧估计形变趋势，默认向前预测 60 毫秒。回弹已经开始、形变仍偏低时，可以撤回尚未到位的收紧目标。

GelSight 使用相对空载图的颜色和纹理变化作为接触形变代理量。启动时在砝码静止、气球轻夹的状态下取约两秒的中位数。GelNeuro 记录短时间内的正负极性事件计数，其持续更新的设备时钟用于检测断流。上下晃动可能同时产生形变和剪切，需用影子数据核对每次下拉和回弹分别对应的触觉变化。绝对压力和两侧压力差需要独立标定。

## 运行

在项目根目录执行，沿用 `.venv/live` 和已有的 GelSight、Speck、ROBOTIS 驱动。

```bash
# 两路触觉及夹爪反馈连通性
bash demos/balloon-grip/run.sh --probe --seconds 5
# 暂时让气球脱离 GelSight 胶面，采集空载图
bash demos/balloon-grip/run.sh --reference
# 重新固定气球和砝码，保持轻夹。READY 后上下晃动砝码，查看 close / open 提案
bash demos/balloon-grip/run.sh --shadow --seconds 45
# 确认方向后运行实机闭环
bash demos/balloon-grip/run.sh --execute --seconds 45
```

默认是影子模式。每轮启动时先让砝码停稳。实机与 `scripts/gripper.sh` 共用 ROBOTIS 位置驱动和限速轮廓，速度与加速度取本机设备读回值。使能后先验证 0.5 秒原位保持。等终端打印 `READY` 再开始振荡。实机执行到时或按 Ctrl+C 后会关闭扭矩，试验时让气球保持悬挂或托住。完全失去接触或触觉断流时会结束实机闭环。空载图默认保存在 `outputs/gelsight-unloaded.local.png`。重新采集时用 `--reference-image <新图路径>`，并在后续运行传相同路径。默认按 USB by-id 找 GelSight，也可以用 `--camera <设备编号>`；夹爪端口可通过 `--port` 指定。

## 调节

从 `configs/config.example.json` 复制到 `configs/config.local.json` 后可调整运行参数，也可用 `--config <文件>`。启动时的编码位置限定本轮活动范围，夹爪的共享机械边界同样生效。

| 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `deadband_fraction` | 0.12 | 相对初始形变的保持带 |
| `loss_fraction` | 0.20 | 低于该形变比时判定失去接触 |
| `min_startup_score` | 1.2 | 建立目标所需的最低形变分数 |
| `open_step_counts` / `close_step_counts` | 2 / 1 | 每步开合的编码计数 |
| `min_command_interval_s` | 0.035 | 连续目标的最短间隔，单位秒 |
| `lookahead_s` / `max_projection_delta` | 0.060 / 0.20 | 形变预测时距（秒）及单次预测修正上限（相对目标） |
| `max_open_counts` / `max_close_counts` | 25 / 15 | 相对启动位置的最大行程，单位编码计数 |
| `tracking_limit_counts` | 4 | 目标与当前实测位置允许的最大差，单位编码计数 |
| `current_cap_a` | 0.20 | 电机电流上限，单位 A |
| `motion_window_s` | 0.120 | GelNeuro 事件计数的主机接收窗口，单位秒 |
| `max_sensor_age_s` | 0.25 | 触觉和反馈允许的最大年龄，单位秒 |

每轮在 `outputs/` 下保存 `report.json` 和 `trace.jsonl`。逐帧日志包括形变实测值、预测值、两路数据时序和夹爪位置。执行模式还记录目标和写入回执，可用后续位置反馈核实运动。报告记录 GelSight 的实际帧间隔。软件仿真频率、图像更新频率和夹爪实测闭环带宽分别验证。

## 录制学习数据和导出 Sparsh 特征

第一次先做小幅、可重复的伸长。沿砝码运动方向放一把尺，记录砝码静止时的高度为 0 mm，并在气球靠近夹口处选一个可从侧面观察的位置。用这个位置判断气球是否整体往下滑。托住部分重量或先用更轻的砝码，在砝码下方留接落空间并限住最大下移量。记录砝码重量、支撑设置和每次装夹编号。

1. **空载和静态标定。** 原始空载录制时让气球与 GelSight 脱离。恢复装夹后，录当前静止位，再试向下额外移约 1–2 mm，在仍稳定接触的范围内保持约 5 秒。确认这个级别可重复，再增加一级。绝对位移取决于装夹，观察到滑移就回到上一级。每级重新回零后可录 2–3 次。
2. **限幅下拉和缓慢松回。** 先做一段可控的下拉、保持、松回，便于比较 GelNeuro 在两种变化方向上的事件。若需要顶部助力，从开录前就保持同样的支撑，并用 `--top-assist` 标注；有助力和无助力的试次分别记录。
3. **单次下拉和自然回弹。** 用已确认不会滑脱的幅度，先做一次下拉、松手、自然回弹，停稳后再重复。双触觉同时保存；GelNeuro 的动作信息主要在拉动和回弹段。
4. **短时连续振荡。** 接触连续稳定后再提高频率，先录 5–10 秒小幅振荡。若要监督学习下拉和回弹时刻，侧面拍到尺和砝码，或同步接入位移传感器。操作提示只标记录制阶段。脱手试次可保留给失去接触检测，成功保持接触的动作数据应单独标注。

```bash
# 胶面无接触时采原始空载背景，记下程序输出的 OUTPUT 路径
bash demos/balloon-grip/run.sh --collect background
# 保持在静止基准或实测 2 mm 位移。每条命令录一段
bash demos/balloon-grip/run.sh --collect hold --extension-mm 0
bash demos/balloon-grip/run.sh --collect hold --extension-mm 2
# 依照终端提示缓慢下拉、保持、松回；顶部持续助力时加 --top-assist
bash demos/balloon-grip/run.sh --collect load-cycle
# 看到 DO MIXED / DOWN 再做自然回弹或振荡
bash demos/balloon-grip/run.sh --collect pull-rebound
bash demos/balloon-grip/run.sh --collect oscillation
# 用上面 OUTPUT 显示的两个目录离线提取特征
bash demos/balloon-grip/run.sh --encode <试次目录> --background <空载试次目录>
```

`--collect` 打开共享双触觉采集器，存原分辨率 JPEG、GelNeuro 逐事件数据、主机时间戳和试次条件。输出在 `outputs/raw/`，采集器不连接夹爪串口。`--notes` 可记录砝码重量和支撑条件，`--object-id` 可区分不同气球或装夹，`--repetition` 记录重复编号。

`--encode` 调用项目现有的 `ImageBackbone('sparsh_dino_small')` 和 `prepare_sparsh`。它使用同一台 GelSight 在同一设置下的空载录制，对当前帧和较早一帧做因果配对，默认间隔一帧，可用 `--history-frames` 调整。另存 `outputs/<试次名>.sparsh.npz`，每组含 384 维编码、帧编号、两帧的主机接收时间及参考帧索引。原始 GelNeuro 事件由对应试次目录读取。主机接收时间可用于近似对齐，传感器之间的精确对时和学习动作标签仍需另行采集。Sparsh 特征本身需要训练伸长、相位或动作预测头后才能用于网络控制。


## 第一版 Sparsh 闭环

第二遍 16 秒下拉试次已拟合本机的 `models/contact-head.npz`。冻结项目原有的 Sparsh ViT-Small，用单试次的 GelSight 形变分数监督线性接触头。该头输出相对接触估计，当前数据无牛顿力标签，也没有独立试次验收。

```bash
# 回放已录的原始帧和事件，复用实机控制器的目标计算
bash demos/balloon-grip/run.sh --replay demos/balloon-grip/outputs/raw/20261120T050453Z_press_release_fab1236b
# 两路实物输入，夹爪仅提供位置反馈
bash demos/balloon-grip/run.sh --shadow --contact-head demos/balloon-grip/models/contact-head.npz --seconds 30
# 真机试验，沿用 gripper.sh 的设备电流上限和限速配置
# 托住气球和砝码，待 READY 后小幅下拉、保持、松回
bash demos/balloon-grip/run-live.sh
# 默认 70 秒，亦可传 --seconds 25
```

每次启动使用本轮的静止接触作为参考。估计下降时增加闭合位置，估计恢复时撤回部分闭合量。`run-live.sh` 加载 `configs/fast-close-20.json`，每帧闭合目标最多前进 20 个编码计数，松开步幅为 2 个计数。相对使能前的位置，最多收紧 24 个计数、松开 4 个计数。胶面形变低于初始值的 80%，且单帧下降至少初始值的 12% 时，可以在该帧 Sparsh 推理完成前发出收紧目标。软件目标步幅和实际机械速度分别验证。之前的 5 计数版可用 `--config demos/balloon-grip/configs/fast-close.json` 选用。

阈值附近保留 0.75 计数滞回。低接触时需要读数从本轮最低点回升至少初始值的 5% 才逐步撤回闭合目标。GelNeuro 在 120 毫秒内至少出现 3 个不同的事件位置时，形变外推从 60 毫秒增加到 90 毫秒。GelSight 原图的形变分数低于初始值的 80% 时辅助收紧，快闭版辅助维持到形变回升至 90%。确认接触丢失时停止。直接使用 `run.sh --execute --contact-head ...` 并省略快闭配置时，仍为每帧最多收紧 1 个计数的基础配置。

回放假定每条目标立即到位，原触觉记录保持不变。实机结果以实际夹爪位置、扭矩状态和动作回执为准。实机在使能前校验显式电流上限的寄存器回读。Sparsh 气球试验默认采用 0.20 A 的显式电流上限，启动原位保持容差为 1 个编码计数。使用 `--manual-current-limit` 时，电流上限与原手动位控入口一致，使能后通过共享位置命令收紧至多 2 个计数，按原驱动的到位判定等待，再检查位置保持。启动等待最多 0.35 秒，期间逐次验证会话位置范围。实机读到初始位置漂移、电流越限、触觉断流或接触丢失会退出并关闭扭矩。

有新试次时可用 `--fit-contact <试次目录> --contact-head <新模型路径>` 生成另一个模型。预测头与拟合使用的空载图绑定，重新采集空载参考后需重新拟合。

## GelNeuro 主机 CPU 检查

```bash
bash demos/balloon-grip/run-gelneuro.sh
```

该入口直接复用 slip-recatch 的 CPU 分类发布器，读取 GelNeuro 原始 DVS 事件及板载时钟。默认输出 10 秒的分类、概率及原始事件窗口，目录为 `outputs/gelneuro-cpu-live-*`。程序仅打开 GelNeuro，运行时不接入夹爪。在已有权重中，分类头输出 `quiet`、`slide`、`rotate` 和 `escape_wriggle`。压力变化需要对应的监督数据和输出头。当前气球闭环在 GelNeuro 近期出现多点事件时延长 GelSight 的趋势外推时间，动作判断仍随 GelSight 更新。
