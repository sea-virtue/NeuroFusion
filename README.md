> 最新夹爪实机（2026-11-12）：修正使能后目标刷新与 USB 延迟后，三次保持、八次小行程往返及正式驱动接口验证通过。当次复测结束已关闭使能；电流上限沿用设备值。详见[本次复测](docs/gripper-retest-20261112.md)。

手动开合入口：`./scripts/gripper.sh status`、`./scripts/gripper.sh close --steps 12 --hold-seconds 2`、`./scripts/gripper.sh open --steps 12`。操作和实测说明见[夹爪手动控制](docs/gripper-manual-control.md)。

# NeuroFusion

本机代码目录：`/home/diaohh/neurofusion/neurofusion`。已实现可训练主机链、双触觉采集与独立夹爪SNN闭环；完整多模态策略的实机任务训练和验收仍待完成。

当前独立夹爪一行启动（会实际运动）：

```bash
bash /home/diaohh/neurofusion/neurofusion/scripts/slip-catch.sh
```

默认100 Hz控制、1 ms推理窗口，本机只读实测分类发布约99.47 Hz。`--dry-run`检查命令，`--shadow`读取设备但不发送运动目标。换机环境、配置模板、代码结构和验证范围见[开发说明 v1](docs/development.md)，控制流程见[滑移控制说明](docs/slip-teacher.md)。

核心设计：RGB 编码后缓存任务上下文 `c`；GelSight 新帧、GelNeuro 事件包、机器人状态及已执行动作，通过确定顺序更新同一份接触记忆 `h`；控制时刻独立读取最新状态，经统一动作头生成动作提案。

## 早期合成演示（历史入口）

以下为原WSL研究环境；当前独立夹爪使用上文 `.venv/live` 入口，安装见开发说明。

在 Ubuntu WSL 终端中使用已经验证的 Linux 环境，无需重装 Samna 或新建环境：

```bash
cd ~/neurofusion/neurofusion
source ~/miniconda3/etc/profile.d/conda.sh
conda activate minist
python -m pip install -e . --no-deps --no-build-isolation
python -m unittest discover -s tests -v
python -m neurofusion demo --config configs/baseline.toml --output outputs/demo-50hz.json
python -m neurofusion demo --control-hz 100 --output outputs/demo-100hz.json
```

`(base)` 是终端当前环境，不是项目文件夹的一部分；运行时仍使用已验证的 `minist`。若不安装包，可在仓库根使用 `PYTHONPATH=src python -m neurofusion demo`。

演示使用**合成特征与手写 toy 运算**，用于检查接口顺序。帧在逻辑 0/40 ms 到达，事件在 5/15/25/35 ms 更新记忆；动作按逻辑 50 Hz（或指定 100 Hz）读取，不等待下一帧。`--without-events` 可作接线对照。没有调用相机、Speck、训练网络或机器人；50/100 Hz 是逻辑时间设置，不是实测吞吐量。输出 JSON 也标明 `synthetic=true`。

## 初版参考目录（后续新增模块见开发说明）

```text
configs/baseline.toml        # 候选维度、逻辑频率与队列容量
src/neurofusion/
  contracts.py              # 时钟、原始数据、特征消息、状态快照、动作提案
  config.py                 # 配置与维度检查
  models/interfaces.py      # 编码器、三路状态更新、统一动作头协议
  models/pending.py         # 学习模型待实现入口，调用时明确报错
  runtime/engine.py         # 单写入者参考顺序驱动，非实时调度器
  runtime/clock.py          # 独立逻辑控制时刻
  adapters/interfaces.py    # 相机/事件/机器人边界与仅记录提案的 ShadowSink
  adapters/pending.py       # Speck、GelSight、RGB、真实机器人接口待实现
  data/pending.py           # H01 所需的真实输入读取等仍待实现
  training/pending.py       # 训练与评估待实现
  demo.py                   # 合成数据与 toy 组件，只验证接线
  cli.py                    # 当前仅有 demo 命令
tests/test_runtime.py       # 因果顺序、帧间影响、单写入者等行为检查
docs/architecture.md        # 数据流、时钟、调度、输出和待实现边界
docs/implementation-status.md
data/README.md              # 数据、来源与排除样例约定
```

历史硬件验证保存在同级 [speck-lab](../speck-lab/README.md)，研究规划见同级 [项目规划](../项目规划/README.md)。正式项目不复制一个新的驱动实现；未来通过适配器接入已经验证的采集路径。硬件已跑通不表示 NeuroFusion 已完成片上推理、训练或闭环。

初版规划（历史记录）：按[当前任务](../项目规划/当前任务.md)完成 H01 真实输入接入与最小追踪，随后尽早建立 H02 最小可训练主机链；逐组件单测并接回整体。验证方法见[组件实现与验证](../项目规划/组件实现与验证.md)。不等待完整GUI、REACT复现、片上推理或机器鱼平台才开始模型。模型层数、编码方式、门控公式、实际控制频率和执行器映射仍待确定。

## 可切换的编码器与动作头（2026-11-01）

已下载六个官方源码仓库，接入 ResNet18、DINOv2-Small、Sparsh-Small，以及 MLP、ACT 条件适配、DP U-Net、DP Transformer 动作头。配置见 `configs/candidates.toml`。图像骨干默认冻结，附带官方预训练权重；动作头和投影尚未做任务训练。

```bash
python -m neurofusion.models.smoke --all --output outputs/candidates-check.json
```

参数量、权重大小、预算建议、张量训练接口及现有调度器桥接见 [候选模型说明](docs/model-candidates.md)。源码版本与恢复方法见 [third_party](third_party/README.md)。新入口不替代原合成演示；事件编码器、接触记忆学习更新和完整训练链仍待实现。


## 2026-11-03 可训练主机链

新主链见[已实现结构](docs/implemented-network.md)。配置`configs/pipeline.toml`；入口`python -m neurofusion.training.pipeline --steps 3`。默认attention记忆+预训练ResNet/Sparsh+MLP，支持plain/gated及ACT/两种DP。动作改为7位Franka提案（6有效+1预留零）与独立一维ROBOTIS提案；新主链总8槽位，旧5维demo仅作历史参考。22项测试、12组合及检查点检查通过；真实数据与硬件闭环待实现，辅助预测未实现。


## 2026-11-04 实时双触觉接线

GelNeuro/Speck与GelSight真实输入已同时接入正式编码器、记忆和影子动作输出，证据见项目规划/实验记录/2026-11-04_双触觉接入正式网络.md；新入口`python -m neurofusion.live_check --seconds 20 --output outputs/live-next`。夹爪尚未正常枚举；本次没有发送机械指令，不构成训练效果、严格时钟同步或实时控制通过。

## 2026-11-04 独立原始触觉采集

采集入口：`python -m neurofusion.capture gui`，也支持 `record` 与 `inspect`。事件和相机分进程读取、独立写盘，无模型推理/机械指令。保存逐事件二进制、SDK批次时间索引、相机原生MJPEG逐帧JPEG、操作条件与提示标注；支持仅事件采集。不是旧live_check的首末帧摘要。操作方法、39段建议清单、格式与边界见[触觉采集说明](docs/tactile-capture.md)。

8秒真实诊断保存16个事件、1558个批次、124张3280×2464图像，文件复核通过；本机观测约17.2Hz，非25Hz保证。现有45项测试通过（含11项采集测试）。无动作真值或精确同步，不把弱标签或采集成功称为训练完成。

## 2026-11-11 SNN主链

训练入口默认使用`configs/pipeline-snn.toml`，已接入归档的训练后SNN与跨包膜电位/历史迹，输出64维特征进入融合和动作头。先冻结SNN与视觉骨干，训练投影、融合和动作头；CNN保留作对照。52项测试、12种组合、三步合成训练和真实双触觉离线前向通过。详情、复现命令及真实训练缺口见[本轮验证](docs/snn-flow-validation.md)。旧live_check仍是CNN联调入口，不把离线通过视为SNN已在线/上芯片或机器人闭环完成。


## 2026-11-11：SNN 计时滑移教师（历史记录）

2026-11-12已加入实时发布、默认即时回夹与100 Hz启动脚本，当前行为以[开发说明](docs/development.md)为准。以下描述首版实现。

新增独立一维尺子滑移控制器：夹持、有限松开、SNN方向确认、默认0.5秒墙钟计时、夹紧及教师日志。代码`src/neurofusion/control/`；[使用与数据边界](docs/slip-teacher.md)，配置`configs/slip-teacher.example.json`。保持现有机械臂/ROBOTIS接口不变，本控制器未启动实机；2026-11-12上游驱动已通过空载小行程复测并清除旧故障锁，见[复测记录](docs/gripper-retest-20261112.md)。新增16项控制测试在拉取后的驱动上通过，当前SNN checkpoint离线加载通过。原始采集程序的实时窗口/预测发布仍需接线，未宣称整机闭环完成。
