# 独立夹爪闭环开发说明 v1

更新：2026-11-12。本文记录 Speck 事件相机、CPU SNN 分类与 ROBOTIS 夹爪的独立闭环；不依赖 Franka，不要求片上部署。详细操作见 [滑移控制说明](slip-teacher.md)，当天硬件排查见 [实机记录](tactile-live-check-20261112.md)。

## 启动与控制过程

本机已配置环境和标定，一行启动：

```bash
bash /home/diaohh/neurofusion/neurofusion/scripts/slip-catch.sh
```

此命令会实际使能并控制夹爪。脚本自动进入项目根目录、调用 `.venv/live/bin/python`、创建独立日志目录，并管理推理和控制两个子进程。可从任意工作目录运行。

```bash
# 不打开设备、不创建日志目录，只检查参数并显示命令
bash scripts/slip-catch.sh --dry-run
# 读取真实相机和夹爪反馈，不使能、不发送运动目标
bash scripts/slip-catch.sh --shadow
# 参数与路径覆盖选项
bash scripts/slip-catch.sh --help
```

默认过程：等待第一条新鲜分类，回到夹持参考562；稳定后每至少0.5秒将目标减小1计数，最多松开8计数至554。每次继续松开还要求位置反馈进入上一步目标的3计数容差。SNN持续分类；第一条新鲜 `task=slide` 且任务置信度至少0.6的分类，在同一控制周期触发回夹562，不等待0.5秒间隔结束，也不附加连续确认或方向门限。没有检测到滑动时，达到行程边界后保持并等待；6秒未滑超时会回夹并报告失败。

到位后的 `FINISHED` 保持使能并继续检查反馈。扶好直尺后 Ctrl+C，脚本协调两个子进程退出，驱动关闭使能。`FINISHED`仅说明位置序列结束，不证明直尺仍被夹住；故障退出也可能关闭使能。

## 模块与数据流

```text
Speck USB3 / Samna
  → control.live：原始事件、真实硬件时钟水位、连续窗口
  → control.snn.SNNRecognizer：CPU分类，保留跨窗口状态
  → stdout SlipPrediction JSONL / stdin
  → control stream：100 Hz反馈查询与状态机
  → adapters.robotis：有界一维位置目标与实测反馈
```

| 文件 | 职责 |
| --- | --- |
| `scripts/slip-catch.sh` / `control/launch.py` | 环境入口、配置快照、双进程管理、退出和频率报告 |
| `control/live.py` | Speck独占读取、原生事件分窗、原始记录及分类发布 |
| `adapters/live_devices.py` | SDK连接、可选板载辅助时间戳与资源恢复 |
| `control/snn.py` | checkpoint分类头、膜状态/trace、等价的空窗口快速推进 |
| `control/slip.py` | 不依赖Torch/硬件的状态机和输入合同 |
| `control/__main__.py` | 固定周期调度、输入队列、反馈、执行与保持 |
| `control/session.py` | 每周期教师trace、manifest与BC样本准入 |
| `adapters/robotis.py` | 位置/电流模式区分、目标和实测边界、串口及使能清理 |

同一时刻只能有一个进程持有Speck，也不要让手动夹爪工具与闭环同时占用串口。stdout管道只承载预测JSONL。GelSight不参与当前一维回夹判定；完整多模态示教需要另行接入同步原始采集，不能只用这些诊断日志代替。

## 频率与延迟

启动脚本选择1 ms推理窗口、0.25 ms SDK轮询、100 Hz辅助硬件时间戳请求及100 Hz夹爪控制。底层 `control.live` 直接启动时仍默认20 ms窗口；请用脚本或显式指定参数。模型原生时间步为1 ms，未改权重或训练定义。

1 ms窗口不等于每秒发布1000条新分类：SDK辅助时间戳最高100 Hz且可能批量送达。同一批关闭多个窗口时推进所有窗口，但仅发布最新分类；本机只读联调实测分类约99.47 Hz、控制约99.99 Hz，周期中位10.004 ms、p95 10.044 ms、最大11.019 ms。内部追算频率和实际发布频率分别记录。

板载功耗测量只提供与DVS共用的真实设备时钟，功耗值不输入SNN。水位保留10 ms设备时间滞后，容纳独立SDK队列；迟到事件失败退出。`captured_ns`是水位关闭窗口的主机接收时间，不是物理滑动起点。消费者仍检查150 ms分类/反馈新鲜度与100 ms控制tick间隔；冷启动最多等第一条分类15秒，不会提前使能。

旧50 Hz控制、20 ms窗口的用户实跑已记录到滑动分类触发回夹：分类可用至目标写入7.93 ms，决策至位置回到至少561约259.68 ms。新版100 Hz仅完成只读联调，尚未进行新版带尺动作验收。机械速度、加速度和电流设置沿用原值；软件提高频率不会消除机械响应时间。原始滑动起点、直尺保持成功率和分类准确率未验证。指标摘要见 [验证记录](validation/slip-catch-20261112.json)。

## 配置与换机恢复

| 配置 | 用途 |
| --- | --- |
| `configs/slip-teacher.example.json` | 通用软件夹具，不能视为实机夹持标定 |
| `configs/slip-catch-search.example.json` | 本机逐计数搜索参数参考，夹持562、最多松8计数 |
| `configs/gripper-position-limits.example.json` | 本机位置边界参考：目标最高562、实测容许575 |
| `configs/slip-catch-search.local.json` | 启动脚本实际读取的本机控制配置 |
| `configs/gripper-position-limits.local.json` | 驱动自动读取的本机位置边界 |

`*.local.json`被Git忽略，推送不会携带这些文件。新检出且本地文件不存在时，可以从模板创建，再按实际硬件核对或重新标定；已有本地文件不要覆盖：

```bash
cp configs/slip-catch-search.example.json configs/slip-catch-search.local.json
cp configs/gripper-position-limits.example.json configs/gripper-position-limits.local.json
```

边界模板对应型号35073、ID 1、`/dev/ttyUSB0`、硬件位置范围0～1150。闭合比例仍按完整硬件范围计算，562对应562/1150，不把562重新映射为1。实测上界575减命令余量13得到目标上界562；575是用户接受的反馈抖动边界，不是允许发送的收紧目标。3计数到位容差不代表1计数机械定位精度；计数不能直接换算成夹爪开口毫米数。默认单条20计数、会话25计数范围限制保留。

环境和运行输出也不随Git上传。当前工作组合为Python 3.12.16、Torch 2.6.0+cpu、NumPy 2.2.6、Pillow 11.3.0、OpenCV headless 4.11.0.86、Samna 0.48.6、dynamixel-sdk 3.8.4、pyserial 3.5。下面是换机安装参考，未在全新机器重做安装验收；本机已有 `.venv/live` 可直接使用。

```bash
python3.12 -m venv .venv/live
.venv/live/bin/python -m pip install 'setuptools>=68' wheel
.venv/live/bin/python -m pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/live/bin/python -m pip install numpy==2.2.6 pillow==11.3.0 opencv-python-headless==4.11.0.86 dynamixel-sdk==3.8.4 pyserial==3.5
.venv/live/bin/python -m pip install samna==0.48.6 --index-url https://gitlab.com/api/v4/projects/27423070/packages/pypi/simple
.venv/live/bin/python -m pip install -e . --no-deps --no-build-isolation
.venv/live/bin/python -m pip check
```

确认Speck以USB3枚举为 `337d:5bc9`，当前用户拥有USB读写权限；夹爪串口同样需要读写权限。Samna不能打开设备时，先检查接线、枚举速率和权限。本机排查过程见当天实机记录；不要把USB2枚举或空闲时事件稀疏直接当作模型错误。

默认checkpoint已在Git跟踪：`artifacts/checkpoints/gelneuro-snn-20261111/final/all-data.pt`，SHA256 `d28753b7bb9494bb6e21bbf32d096421090f473675bddd9e5cc0a44eb65e2ee8`。原训练数据集不在本仓库，运行推理不要求训练数据集。

## 测试与日志

不连接硬件的常用检查：

```bash
bash -n scripts/slip-catch.sh
bash scripts/slip-catch.sh --dry-run
.venv/live/bin/python -m unittest discover -s tests -p 'test_slip*.py' -v
.venv/live/bin/python -m unittest discover -s tests -p 'test_live_slip.py' -v
.venv/live/bin/python -m unittest discover -s tests -p 'test_speck_watermark.py' -v
.venv/live/bin/python -m unittest discover -s tests -p 'test_snn_empty_advance.py' -v
.venv/live/bin/python -m unittest discover -s tests -p 'test_gripper*.py' -v
```

本轮归档还检查了采集器和ROBOTIS适配器。最小运行环境没有torchvision，两个依赖融合策略 `Observation` 的ROBOTIS测试未执行；不把局部回归称为全项目测试通过。测试范围与计数保存在上述验证JSON中。

每轮 `outputs/slip-catch-*` 保存配置和边界快照、`run.json`、`events/`原始窗口/分类/频率报告、`control/`状态/反馈/目标回执，以及推理stderr。这些原始输出和 `.venv/` 均被Git忽略；只归档精简验证摘要。完整BC训练另要求同步双相机原始数据、质量检查和人工接受的真实示教，不接受synthetic、shadow或未经核验的捕获结果。后续优先验证新版100 Hz带尺回夹、模型漏检与机械响应，再整合同期双相机示教。
