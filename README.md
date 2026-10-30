# NeuroFusion

正式代码目录：`/home/diaohh23/neurofusion/neurofusion`。当前为研究框架，学习模型与真实机器人控制待实现。

核心设计：RGB 编码后缓存任务上下文 `c`；GelSight 新帧、GelNeuro 事件包、机器人状态及已执行动作，通过确定顺序更新同一份接触记忆 `h`；控制时刻独立读取最新状态，经统一动作头生成动作提案。

## 快速开始

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

## 目录

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

下一优先工作：按[当前任务](../项目规划/当前任务.md)完成 H01 真实输入接入与最小追踪，随后尽早建立 H02 最小可训练主机链；逐组件单测并接回整体。验证方法见[组件实现与验证](../项目规划/组件实现与验证.md)。不等待完整GUI、REACT复现、片上推理或机器鱼平台才开始模型。模型层数、编码方式、门控公式、实际控制频率和执行器映射仍待确定。
