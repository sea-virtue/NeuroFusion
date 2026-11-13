# SNN 驱动的一维尺子滑移与即时回夹

当前实现直接使用任务softmax中的P(slide)，由 `slide_probability_threshold`控制即时回夹门限，不要求slide为胜出类别。预测JSON中的 `task`/`task_confidence`仍表示胜出类别及其概率；独立 `slide_probability`才是滑动概率。新阈值省略时回退 `confidence`；旧消息缺少滑动概率时只接受slide胜出类别的置信度。`confidence`另用于其他运动类别和显式计时方式，不能把quiet置信度当作滑动概率。

确认由 `slide_confirm_frames`配置，在发布器遍历全部原生窗口时先确认并锁存，再合并发布最新预测。连续性由源窗口边界和序号检查；重复控制tick不计作新窗口。后续quiet不能取消确认，原始证据时间不被刷新，过期检查仍保留。默认即时方式没有0.5秒滑动计时或额外连续墙钟确认延迟。

本机当前纸盒参数：夹持参考327，6秒未滑超时，每0.5秒松开1计数，最多12计数至315，P(slide)门限0.6，单窗触发，会话范围25。归档快照为 `configs/slip-catch-paperbox.example.json`。Bash脚本实际读取 `configs/slip-catch-search.local.json`，该文件被Git忽略；`slip-catch-search.example.json`是另外调参的探索模板，不替代本机配置。启动脚本默认100 Hz、1 ms窗口、RN型号35073的direct profile（Profile Velocity=0），可传 `--position-profile bounded`回退机械profile。

修改持续松开时间和间隔时，还应按“步数×每步计数/1150”同步设置 `max_open_delta`；`gripper_session_radius_raw`须容纳初始位置偏差和总释放量。每条目标跟踪限制仍为20计数，回夹会按实测位置逐次追赶。全局位置标定继续限制命令≤562、反馈≤575；本轮327是物体夹持参考，不会因此把562重新定义为闭合比例1。

滑动概率门限曾试用0.35/3窗，释放范围曾试用15秒/60计数及10秒/20计数，现均不作为纸盒当前参数。实验与回退依据见[当前验证摘要](validation/slip-catch-20261113.json)。这些概率不是已校准的抓持成功率，纸盒和直尺的滑移识别/保持成功率仍待核验。

首版0.5秒方式仍可显式使用：设 `slide_seconds=0.5`、`confirm_seconds=0.03`；其方向确认与连续墙钟计时规则保留。`open_step=null` 使用旧的连续松开方式。项目为独立实现，不是 SlipTack 作者源码，不声称下落了固定距离。

## 代码边界

- `src/neurofusion/control/slip.py`：纯 Python、无 Torch/SDK/机械臂依赖的状态机、参数、预测/反馈合同及一维命令。
- `control/snn.py`：保留已有 `GelNeuroUnifiedSNN` 的分类头，按 checkpoint 标签读取任务和方向，连续保留膜状态与 host trace。与主策略去掉分类头的 `SNNPacketEncoder` 分开，不修改已有模型。
- `control/session.py`：独立版本的教师 trace sidecar，以及经审核真实示教的一维 BC 标签读取；原始触觉录制格式不变。
- `control/__main__.py`：软件 fixture、离线回放、接收在线 SNN 预测的周期控制循环。默认不发送动作；只有 `stream --execute` 才尝试通过现有独立 ROBOTIS 适配器运动。

没有修改 Franka/ROBOTIS 已解耦的布局。`GripperCommand.robotis` 仅含一维绝对闭合比例，`normalized=False`，兼容现有 `RobotisGripper.command()`；不需要伪造 Franka 占位字段。0 为设备最小位置端，1 为最大位置端，不是牛顿、毫米或每步位移。

## 状态与计时

默认即时方式：`GRIPPING → OPENING → BRAKING → FINISHED`。新鲜P(slide)达到门限并满足配置的窗口数后，直接产生BRAKING回夹目标，不经过SLIDING。回夹目标受每步跟踪范围和用户收紧上限约束；过期预测、通信/夹爪故障按原规则处理，不用旧预测制造心跳。

“立即”是控制软件不再额外等待；仍包含事件窗口关闭、CPU推理、控制调度、串口通信和机械响应。`FINISHED`依旧需要位置反馈到达，并不自动证明直尺仍在夹爪中。静止输入的设备时间水位现已通过辅助硬件时钟处理。旧50 Hz用户实跑日志已验证滑动分类触发回夹，但尚未核验直尺保持成功；新版100 Hz仅完成只读联调。

以下为显式计时方式（slide_seconds>0）：`GRIPPING → OPENING → SLIDING → BRAKING → FINISHED`

1. 向经现场确定的 `grip_closure` 有界缓慢移动，实测位置进入容差并保持一段时间。
2. 以 `open_rate` 逐步减小闭合比例，最多松开 `max_open_delta`，等待 SNN 的滑动及预期方向判断。
3. 第一条高置信度、预期方向的滑动预测出现时停止继续松开；连续独立预测确认后进入计时状态。**计时起点为该连续序列第一条预测的主机可用时间**，不是物理滑动起点。重复读取同一预测不会完成确认。
4. 保持当前开度，距起点经过 `slide_seconds-close_delay_seconds` 后向原夹持位置收紧。`close_delay_seconds` 默认 0，只有本机测出响应后再设置，不拷原论文的 120 ms。
5. 实测夹爪回到夹持位置并稳定后记为 FINISHED。它只说明命令序列结束，不证明尺子未滑脱、成功停住或没有变形，真实回合仍需人工/外部测量审核。

计时是连续墙钟时间，不是“累计识别为 slide 的窗口数”。中间 quiet 不重置或暂停计时；quiet 也不作为尺子静止/仍在接触的证明。识别从开始就持续向下时同样可触发。方向置信度只在 task=slide 时使用。

未滑动超时、异常方向/旋转、SNN 过期、控制 tick 超时均进入带失败原因的夹紧流程，不会恢复松开；夹爪反馈过期则停止产生新命令并报告 FAULT。实际驱动异常由现有适配器清理处理，不能保证故障时物体保持不掉落。周期调度目标 100 Hz，不是已验证的硬实时性能。

## 本机一行启动（100 Hz控制，会实际运动）

```bash
bash /home/diaohh/neurofusion/neurofusion/scripts/slip-catch.sh
```

`scripts/`也是已有 `gripper.sh` 的位置。新脚本自动进入项目、使用 `.venv/live`、创建独立日志目录并启动现有推理发布器和 `control stream --execute`，没有复制一份夹爪状态机。默认读取本机 `configs/slip-catch-search.local.json`，当前纸盒参考327、逐计数松开、0.6单窗触发，机械profile为direct，电流与562/575边界沿用现有设置。

控制循环现为100 Hz，包括最终保持阶段，按固定10 ms截止时间调度。推理使用1 ms窗口、0.25 ms轮询；保留10 ms设备时间滞后及150 ms过期检查。板载功耗时间戳接口最高100 Hz采样，静止时按批送达；因此1 ms窗口不等于1000 Hz持续分类发布。本机六秒只读测试实测分类发布99.47 Hz、控制99.99 Hz，控制周期中位10.004 ms、p95 10.044 ms、最大11.019 ms，零运动命令。1 ms模型窗口是当前原生时间步允许的最细设置，实时输出频率保存在每轮 `events/report.json` 和 `run.json` 中。

脚本加 `--shadow` 仅采集分类和夹爪反馈，不使能；`--dry-run` 仅显示完整命令，不连接设备。`--help` 查看配置、频率及输出路径选项。脚本正常结束或Ctrl+C时协调两个子进程退出；Ctrl+C关闭使能，结束前扶好直尺。日志目录保存本轮控制配置、位置边界快照、原始事件窗口、分类、反馈和目标回执。脚本的命令/影子模式、跨目录启动、自动收尾及Ctrl+C双进程清理均有软件回归测试。100 Hz/direct已做真实小行程与分类触发回夹记录；物体保持成功率尚未验收，Ctrl+C确认异常见开发说明。

## 无硬件检查与离线回放

在项目根目录使用已安装项目的Python环境（本机可用 `.venv/live/bin/python` 代替下述 `python`）：

```bash
python -m neurofusion.control demo --config configs/slip-teacher.example.json --output outputs/slip-fixture-first
python -m unittest discover -s tests -p 'test_slip_control.py'
```

demo 是显式 synthetic 的固定输入夹具，没有摩擦/电机模型，不可用于 BC，也不代表真实抓持成功。输出目录必须不存在。示例 `grip_closure=0.44` 仅用于 fixture，**不是已标定的尺子夹持目标**。

回放输入每行含 `now_ns`、`feedback`、`snn`，字段对应 dataclass；回放不连接设备、不发送命令。时间必须严格递增，反馈/预测只使用该时刻已经可用的数据。

```bash
python -m neurofusion.control replay --config configs/slip-teacher.example.json --input prepared-trial.jsonl --output outputs/slip-replay-first
```

## 在线分类如何接入

现有 checkpoint 的 task 为 quiet/slide/rotate/escape_wriggle，direction 为 up/down/left/right。不同安装方向下，尺子物理向下可能对应另一分类；显式计时模式须用 `expected_direction` 明确映射，不能默认图像坐标等于世界坐标；默认即时模式记录方向，但任何达标的slide任务都回夹，不由方向阻挡。

在已持有原始事件的进程中调用：

```python
from dataclasses import asdict
import json
from neurofusion.control.snn import SNNRecognizer

recognizer = SNNRecognizer(
    'artifacts/checkpoints/gelneuro-snn-20261111/final/all-data.pt',
    expected_direction='down',
)
# events 是现有采集器提供的 Nx4 原始事件，不要再开第二个进程抢设备。
# 窗口必须连续，含整数个 checkpoint 原生 bin；空窗口也要处理。
# window_received_host_ns 由采集器记录，不能把 device_us 直接乘1000充作主机时间。
prediction = recognizer.predict(events, start_us, end_us, window_received_host_ns)
print(json.dumps(asdict(prediction)), flush=True)  # 专用 stdout，只输出预测 JSONL
```

在线输入可以是管道，也可以在自己的采集进程中直接调用 `Episode.tick()`。辨识器不擅自缩短 checkpoint 的 native_feature_step_us。现在可用下述 `control.live` 独立入口持有 Speck 并发布预测；它与采集 GUI 不能同时打开同一块 Speck。

### 本机实时事件入口

本机 CPU 环境位于 `.venv/live`（Python 3.12、CPU PyTorch 2.6.0、Samna 0.48.6）。在项目根目录运行十秒真实事件分类诊断：

```bash
PYTHONPATH=src .venv/live/bin/python -m neurofusion.control.live \
  --checkpoint artifacts/checkpoints/gelneuro-snn-20261111/final/all-data.pt \
  --expected-direction down --seconds 10 --output outputs/speck-live-first
```

此入口只读取 Speck、在 CPU 上分类，不连接夹爪。stdout 只发布 `SlipPrediction` JSONL；输出目录保存对应原始事件窗口、分类和诊断报告，目录必须不存在。`--seconds 0` 持续运行，Ctrl+C 结束。方向参数必须在实际安装中确认，不能因示例写了 down 就当作已标定。

事件按 checkpoint 原生1 ms bin、默认20 ms连续窗口处理。`control.live`额外读取板载功耗测量的硬件时间戳（请求100 Hz），作为静止DVS的真实设备时间水位；本机复位实验确认它与DVS时钟同步复位。功耗值不输入分类器，不产生虚构触觉事件。普通采集器默认不开启此辅助测量；若本入口临时开启，退出时关闭自己启动的测量并恢复原配置。

为容纳独立SDK队列，辅助水位保留10 ms设备时间滞后，先读辅助时间再排空DVS，落在已关闭窗口的迟到事件报错并触发消费者原有故障处理；10 ms是本机已测试的接线参数，不是所有USB负载下的延迟保证。`captured_ns`表示设备水位关闭窗口的主机接收时刻，不是相机曝光时间。没有用主机时间直接外推设备时钟。空窗口仍实际推进SNN原生时间状态；该无偏置、无膜泄漏的IAF骨干在零输入时膜状态不变，仅host trace按原生衰减推进，已与完整运算做数值等价测试。

同一次SDK读取关闭多个窗口时，保留全部窗口并推进状态，只发布最后一个预测；追算跨度最多10秒、缓冲最多100万个事件，最后预测仍须满足150 ms新鲜度。设备时钟回绕/明显倒退、迟到事件或积压超界均停止。本机静止八秒已完成397条新分类，主机可用时间间隔中位20.76 ms、最大48.23 ms；没有放宽消费者150 ms过期检测。完整滑移捕获成功仍需单独验证。

与在线消费者联调时，可将该命令 stdout 接至下述 `stream`。**先使用默认 shadow**，按现场实际位置准备控制配置；加 `--execute` 后才会发送动作。此诊断目录仅含事件和分类，没有同期 GelSight 和完整质量报告，不能作为完整 BC 数据。

在线消费者（默认读取夹爪反馈并做 shadow，不运动）：

```bash
python -m neurofusion.control.snn --checkpoint artifacts/checkpoints/gelneuro-snn-20261111/final/all-data.pt --expected-direction down < live-event-windows.jsonl | python -m neurofusion.control stream --config configs/slip-teacher.example.json --output outputs/slip-live-shadow
```

`control.snn` 是已实现的窗口到分类桥接程序。上例 `live-event-windows.jsonl` 必须是采集程序**实时**提供的输入（可用命名管道），每行包含 `events`（Nx4数组，空窗口可为[]）、`start_us`、`end_us`、`captured_host_ns`。不能将历史文件当在线输入，因为其时间戳会过期。现有采集 GUI 还没有添加这个实时窗口发布接口；可直接在其持有设备的进程中调用前述 `recognizer.predict`，无需走 JSON，或单独使用 `control.live`。stdout 不能混入调试文字。所有进程应在同一主机单调时钟上；跨机器/原始设备时钟输入不能直接使用。分类消息保留 checkpoint SHA256，供回合追溯。

控制器自身定期查询夹爪，不接受 stdin 伪造的夹爪反馈。stdin 阻塞时 watchdog 仍执行；输入结束、解码失败、积压或序号逆序终止正常松开并尝试收紧。最多缓存64条分类，超出即失败，不偷偷丢弃数据当成功。

底层 `control stream` 的物理执行必须另加 `--execute`（上文Bash脚本默认已添加），它调用现有 `device.arm()`，不会解除 `configs/gripper-motion-hold.json`。2026-11-12拉取的驱动已通过空载小行程复测并清除了旧故障阻断，见[最新记录](gripper-retest-20261112.md)；随后已做带尺试验，旧50 Hz实跑出现滑动分类并回夹；物体保持成功率尚未核验。执行前须按夹爪几何/夹持条件重设 grip 及范围；现有 commissioning 适配器的每步和会话行程限制均保留，过大任务被拒绝。它不是完整生产控制器。

FINISHED 后执行入口保持夹持并继续读反馈，直到操作者扶住尺子并 Ctrl+C，才按现有驱动退出流程关闭使能。异常退出也可能关闭使能，不能把失电状态当作可靠持稳。旧50 Hz实跑和新版100 Hz只读联调的范围见[当天实机记录](tactile-live-check-20261112.md)。

## 示教记录与训练边界

每周期记录：SNN 类别/方向/两项置信度与来源、可用时间；实测闭合量与查询时间；完整夹爪原始反馈（在线模式）；状态与计时；教师动作意图；成功下发的命令和设备回执。ACK 不代表机械动作完成。终端 hold 期间继续记录反馈，不制造新的 BC 命令标签。

`manifest.json` 保留目标计时、控制参数、运行方式、synthetic 标志、原始触觉目录链接和人工 outcome。实际录制可用 `--tactile-recording` 指定同期原始数据目录。旧 capture GUI 与发布器不能重复占用 Speck；发布器需在同一原始采集链输出分类。没有对应原始事件/GelSight 的教师 trace 不能当作完整多模态训练集。

`bc_rows(path)` 仅接受 `synthetic=false`、`status=finished`、人工核验后设 `outcome=accepted` 且链接到完整并通过 quality 检查的原始触觉回合。它输出一维 `target_robotis`，对应决策当时可用的观测及实际下发动作，不把实测闭合量或 shadow 提案作专家标签。人工接受还需核对原始录制确实与本回合同步，记录 checkpoint/参数版本。自动日志完整性不证明示教正确。

现有8槽位网络训练器要求完整动作标签；独立夹爪示教没有机械臂动作，不应捏造七个零当作实测标签。后续接具身训练时需要夹爪损失 mask/专项数据适配，此次没有改训练器。目标计时也应作为学生任务条件，尤其是在混用0.3/0.5/0.8秒示教时。

## 验证与未完成项

软件测试覆盖计时从 SNN 起点开始、去除重复预测确认、方向异常、未滑超时、输入过期、控制超时、松开上界、提前补偿、失败命令记录、synthetic 数据隔离、无机械臂/Torch依赖的控制核心和 checkpoint 分类头接入。

已有本机562夹持参考、逐计数松开和旧50 Hz滑动分类触发回夹日志；尚未完成新版100 Hz带尺验收、分类准确率/物理方向标定、物体保持成功率、准确下落距离及完整双相机录制与教师trace集成。不声称整机闭环已经验收。

## 2026-11-12 拉取后兼容性复核

基于 `515a5d4` 的新版 ROBOTIS 初始化与电流上限逻辑，早期16项控制测试再次通过，后续扩展到22项。位置模式连接后由驱动把 `config.current_cap_a` 更新为设备现有电流上限，因此在线消费者读取的是实际采用的上限，不会使用过期的0.2A值。没有改回旧驱动。现已添加独立的实时事件分类发布入口 `control.live` 和9项分窗测试；示例 grip_closure=0.44 不可直接视为当前尺子的夹持位置。实机检查结果另见本次诊断记录。

开发环境、可提交的本机参数模板、模块职责和本轮完整回归范围见[开发说明 v1](development.md)。
