# Speck2f 片上骨干部署与事件速率说明

更新：2026-11-18。设备为 Speck2fModuleDevKit，USB `337d:5bc9`。

同日后续已新增 [片上重抓取 demo](slip-catch-chip.md)：使用 `scripts/slip-catch-chip.sh` 在同一会话加载模型并接入原分类头和控制器；原CPU入口保留。

## 部署范围

现有 `artifacts/checkpoints/gelneuro-snn-20261111/final/all-data.pt`
是训练好的 PyTorch 权重，不是可直接下发的硬件配置。
`final/sdk/quantized-model.pt` 也是离线量化权重；本机归档没有保留配套的硬件 JSON。
原有导出器 `src/neurofusion/training/export_gelneuro_unified_snn.py`
调用官方 Sinabs 生成 SDK 配置，但原先不连接芯片。

部署后的计算分工：

```text
DVS 128×128 双极性事件
  → 片上 DVS 4×4 sum pooling（32×32，保留事件数量）
  → 片上四层 Conv/IAF（8 bit 权重、整数阈值）
  → 末层 16 个输出通道的 Spike
  → 主机：四组因果时间迹 → log1p → 64 维特征 → 分类头
```

时间迹和分类头没有搬到芯片。现有 `scripts/slip-catch.sh` 的识别器
仍读取 RAW DVS，并在 CPU 运行原 SNN；本次部署不自动切换滑移策略的数据源。
部署工具不导入夹爪或机械臂控制模块。

## 环境

沿用 `.venv/live/bin/python` 的 Python 3.12、PyTorch 2.6.0+cpu、
NumPy 2.2.6、Samna 0.48.6。导出工具依赖单独放在 `.venv/chip-tools`，
启动脚本自动加入 PYTHONPATH，不改原闭环环境的包。

重新配置机器时，先按 `development.md` 建好 live 环境，再安装工具：

```bash
cd /home/diaohh/neurofusion/neurofusion
.venv/live/bin/python -m pip install --no-deps --target .venv/chip-tools \
  -r configs/speck-chip-tools.requirements.txt
```

工具路径均相对项目根目录，项目搬家后脚本无需修改绝对路径。
USB 权限沿用 `../speck-lab/configure-usb.sh` 的专用设备规则。
使用前关闭占用 Speck 的采集 GUI、`control.live` 或旧部署进程。

## 导出、部署、检查与恢复

输出目录必须是新目录，不会覆盖既有记录。

```bash
cd /home/diaohh/neurofusion/neurofusion

# 1. 仅导出，不打开设备
bash scripts/speck-deploy.sh export \
  artifacts/checkpoints/gelneuro-snn-20261111/final/all-data.pt \
  --out outputs/speck-export-next

# 2. 下发、SDK 配置读回比较、片上合成输入测试、5 秒真实输入监测
bash scripts/speck-deploy.sh deploy \
  --config outputs/speck-export-next/speck2f-config.json \
  --self-test --seconds 5 --output outputs/speck-deploy-next

# 3. 检查新 SDK 会话的初始配置（不是片内 RAM 持久性读回）
bash scripts/speck-deploy.sh inspect --output outputs/speck-inspect-next

# 4. 恢复第 2 步部署前保存的配置
bash scripts/speck-deploy.sh restore \
  --config outputs/speck-deploy-next/before.json \
  --output outputs/speck-restore-next
```

默认成功后不主动回滚，下发的配置在本次 SDK 会话内使用。
实测关闭进程后重新打开设备，SDK 返回初始默认配置：池化 1×1、无 CNN 监测、
传感器事件转发关闭。因此不能把“成功退出部署脚本”理解成之后任意进程直接可用。
任何需要片上推理的新进程，都必须先打开设备，再 `apply_configuration` 加载导出的
JSON，随后在同一会话连接输出图、读取末层 Spike。`inspect` 只记录新 SDK 会话
返回的配置，不提供独立硬件 RAM 持久性证明。加 `--restore-on-exit` 则只做临时验证；
出现异常或 Ctrl+C 时尝试恢复部署前配置。
`before.json` 保存部署前完整配置，`applied.json` 保存部署配置读回，
`report.json` 保存版本、配置哈希、下发/恢复状态与监测计数。
`live-events.jsonl` 分别记录临时监测的原始 DVS 与 CNN Spike。
实时监测结束后关闭临时 RAW 监测，只保留导出配置中的末层监测。

这是运行时配置下发，没有刷写固件或 Flash。断电/复位后应重新运行部署并检查，
不要将其当作已经设置了开机自动加载。SDK `get_configuration()` 读回比较
不能代替独立的硬件 RAM 电气读回测试。Samna 0.48.6 开启监测时实测会把
`factoryConfig.ioSel` 从 0 调整为 32；工具仅允许这一明确变化并记入报告，
权重、阈值、路由及其他配置仍全部严格比较。

合成测试向首层 CNN 输入 20,480 个已池化事件，检查末层是否实际输出脉冲，
从而检查四层片上计算链。它绕过真实传感器和 DVS 池化，不是触觉准确率测试，
也不是软硬件逐事件数值等价测试或吞吐量测试。真实输入过少时末层没有输出
可以是正常现象；报告分别保留真实输入和合成输入计数。

本机 2026-11-18 实测：核心映射 0→1→3→2，20,480 个合成输入事件得到
13,872 个末层输出脉冲；随后 5 秒真实输入得到 5 个 RAW DVS 事件、零末层
脉冲。新导出的整数权重与归档量化模型逐层一致。9 项部署/时间水位测试通过，
工具叠加环境 `pip check` 通过。原始报告在
`outputs/speck-deploy-20261118-check4/report.json`；
重开 SDK 会话记录在 `outputs/speck-inspect-20261118/report.json`。

## 之前到底读取了什么

`adapters/live_devices.py:SpeckSource` 克隆设备配置，打开
`dvs_layer.raw_monitor_enable`，通过 `Speck2fOutputEventTypeFilter`
选择 `speck2f::event::DvsEvent`，保存 `[x,y,timestamp_us,polarity]`。
`get_events()` 一次排空 SDK 当前缓冲中的一批事件，批大小不固定。
它不是相机按固定帧率产生的一帧，也不是 SNN 的输出特征包。

厂商区分：

| 配置 | 主机收到的类型 | 含义 |
|---|---|---|
| `raw_monitor_enable=True` | `DvsEvent` | DVS 预处理前的原始事件 |
| `monitor_enable=True` | `Spike`，layer=13 | DVS 预处理层监测输出 |
| `cnn_layers[k].monitor_enable=True` | `Spike`，layer=k | 对应神经计算核心的输出脉冲 |

因此，代码使用 `EventFilterGraph` 不代表经过片上 Event Pre-processing。
这里的图在主机做类型筛选。原始事件不会因预处理层池化、ROI、镜像等设置
而变成已池化的输出。DVS 模拟阈值、传感器与传输本身的影响仍然存在；
“原始”不是无噪声或已经证明零丢失。

## 事件吞吐、编码步长、发布率应分开

| 层次 | 原项目实现 | 应如何理解 |
|---|---|---|
| DVS 产生事件 | 异步 `(x,y,t,p)` | 用 events/s 描述，不用固定帧率描述 |
| SDK 读取 | `sink.get_events()` | 一次取得多条或零条事件，不是一次一条 |
| 原始录制轮询 | 默认约 5 ms | 请求约 200 次读取/秒，不是只读 200 events/s |
| 最新闭环请求轮询 | 启动脚本 0.25 ms | 请求上限约 4,000 次/秒；实际含调度、推理和写盘耗时 |
| 主机空间编码 | x、y 各整除 4，计数相加 | 128×128 → 32×32；不表示每 16 条只保留一条 |
| SNN 原生时间格 | `native_feature_step_us=1000` | 每 1 ms 汇总计数；格内细粒度先后次序未保留 |
| 旧识别调用窗口 | 默认 20 ms | 每次连续推进 20 个原生格；不是改成 20 ms 的神经元步长 |
| 最新脚本窗口 | 1 ms | 按原生格逐个推进状态 |
| 辅助硬件时间水位 | 功耗测量请求最多 100 Hz | 这是辅助时钟通道的限制，不是 DVS 或 CNN 上限 |
| 最新分类发布 | 历史实测约 99.45 Hz | 同一批可推进多个 1 ms 格，只发布最新结果 |
| 夹爪控制周期 | 最新脚本 100 Hz | 主机控制调度，与异步事件吞吐不同 |

代码 `models/snn_adapter.py:count_bins` 用 scatter_add 累加每条事件，
没有在这里限每格最大事件数或每格只保留一个事件；但格内时序被汇总。
空格也推进状态，主机分类发布前处理全部就绪窗口，滑动确认在合并发布前完成。
`control.live` 默认参数与 `control.launch` 启动时覆盖参数不同，比较时要看实际入口。

硬件、监测接口、USB、SDK 缓冲、Python 逐事件转换、推理与写盘都有有限能力。
原项目没有测出这块板在本机下的“最大可持续无丢失 events/s”，也没有完整链路
逐事件序号可用来证明硬件端绝对零丢包。不能从 100 Hz、USB3 的 5 Gbit/s 枚举速率、
静止时的少量事件或某次读取耗时推导这个上限。
片上 CNN 还应检查输入拥塞丢弃 `input_congestion_balancer_enable` 和
输出抽取 `output_decimator_enable`；它们会影响实际事件传递，不能混同为采样帧率。

原模型文件同时保留 `bin_us=50000` 与 `native_feature_step_us=1000`，
在线识别以后者为准，加载时重算时间迹衰减。导出器现已使用同一加载逻辑，
保留 input_gain 并折叠进第一层整数权重；量化模型也保留类别与方向元信息。

片上骨干按事件异步运行，不需要主机每 1 ms 将计数张量重新发到芯片。
主机时间迹/分类发布仍需自己的时间基准，所以部署骨干本身不会自动消除
当前 100 Hz 辅助水位所带来的发布批次和 10 ms 水位保护延迟。

## 官方依据

- [Samna 0.48.6 Speck2f：RAW、DVS 预处理、CNN 层](https://synsense-sys-int.gitlab.io/samna/0.48.6/models/speckSeries/speck2f.html)
- [厂商开发板手册：预处理、拥塞控制、抽取与功耗监测](https://www.synsense.ai/wp-content/uploads/2024/12/Speck-Dev-Kit-Manual.pdf)
- [Sinabs 官方部署示例](https://sinabs.readthedocs.io/v2.0.2/speck/notebooks/nmnist_quick_start.html)

本次实机结果见配套 `docs/validation/speck-on-chip-20261118.json`，
原始输出保存在本机 `outputs/`，不随 Git 自动备份。
