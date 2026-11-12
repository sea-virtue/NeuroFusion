# SNN 驱动的一维尺子滑移教师

2026-11-11：独立实现，不是 SlipTack 作者源码。首版按用户要求采用 **识别到下滑后计时 0.5 秒再夹紧**；不声称下落了固定距离，也不需要夹爪动力学仿真或 MPC。

## 代码边界

- `src/neurofusion/control/slip.py`：纯 Python、无 Torch/SDK/机械臂依赖的状态机、参数、预测/反馈合同及一维命令。
- `control/snn.py`：保留已有 `GelNeuroUnifiedSNN` 的分类头，按 checkpoint 标签读取任务和方向，连续保留膜状态与 host trace。与主策略去掉分类头的 `SNNPacketEncoder` 分开，不修改已有模型。
- `control/session.py`：独立版本的教师 trace sidecar，以及经审核真实示教的一维 BC 标签读取；原始触觉录制格式不变。
- `control/__main__.py`：软件 fixture、离线回放、接收在线 SNN 预测的周期控制循环。默认不发送动作；只有 `stream --execute` 才尝试通过现有独立 ROBOTIS 适配器运动。

没有修改 Franka/ROBOTIS 已解耦的布局。`GripperCommand.robotis` 仅含一维绝对闭合比例，`normalized=False`，兼容现有 `RobotisGripper.command()`；不需要伪造 Franka 占位字段。0 为设备最小位置端，1 为最大位置端，不是牛顿、毫米或每步位移。

## 状态与计时

`GRIPPING → OPENING → SLIDING → BRAKING → FINISHED`

1. 向经现场确定的 `grip_closure` 有界缓慢移动，实测位置进入容差并保持一段时间。
2. 以 `open_rate` 逐步减小闭合比例，最多松开 `max_open_delta`，等待 SNN 的滑动及预期方向判断。
3. 第一条高置信度、预期方向的滑动预测出现时停止继续松开；连续独立预测确认后进入计时状态。**计时起点为该连续序列第一条预测的主机可用时间**，不是物理滑动起点。重复读取同一预测不会完成确认。
4. 保持当前开度，距起点经过 `slide_seconds-close_delay_seconds` 后向原夹持位置收紧。`close_delay_seconds` 默认 0，只有本机测出响应后再设置，不拷原论文的 120 ms。
5. 实测夹爪回到夹持位置并稳定后记为 FINISHED。它只说明命令序列结束，不证明尺子未滑脱、成功停住或没有变形，真实回合仍需人工/外部测量审核。

计时是连续墙钟时间，不是“累计识别为 slide 的窗口数”。中间 quiet 不重置或暂停计时；quiet 也不作为尺子静止/仍在接触的证明。识别从开始就持续向下时同样可触发。方向置信度只在 task=slide 时使用。

未滑动超时、异常方向/旋转、SNN 过期、控制 tick 超时均进入带失败原因的夹紧流程，不会恢复松开；夹爪反馈过期则停止产生新命令并报告 FAULT。实际驱动异常由现有适配器清理处理，不能保证故障时物体保持不掉落。周期调度目标 50 Hz，不是已验证的硬实时性能。

## 先运行无硬件检查

在 WSL 正式项目根目录，使用 `minist` 环境：

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

现有 checkpoint 的 task 为 quiet/slide/rotate/escape_wriggle，direction 为 up/down/left/right。不同安装方向下，尺子物理向下可能对应另一分类；用 `expected_direction` 明确映射，不能默认图像坐标等于世界坐标。

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

在线输入可以是管道，也可以在自己的采集进程中直接调用 `Episode.tick()`。新增代码没有启动事件设备、没有替换当前 GUI/原始采集器；需要由持有设备的采集进程发布上述预测。辨识器不擅自缩短 checkpoint 的 native_feature_step_us。

在线消费者（默认读取夹爪反馈并做 shadow，不运动）：

```bash
python -m neurofusion.control.snn --checkpoint artifacts/checkpoints/gelneuro-snn-20261111/final/all-data.pt --expected-direction down < live-event-windows.jsonl | python -m neurofusion.control stream --config configs/slip-teacher.example.json --output outputs/slip-live-shadow
```

`control.snn` 是已实现的窗口到分类桥接程序。上例 `live-event-windows.jsonl` 必须是采集程序**实时**提供的输入（可用命名管道），每行包含 `events`（Nx4数组，空窗口可为[]）、`start_us`、`end_us`、`captured_host_ns`。不能将历史文件当在线输入，因为其时间戳会过期。现有采集 GUI 还没有添加这个实时窗口发布接口；可直接在其持有设备的进程中调用前述 `recognizer.predict`，无需走 JSON。stdout 不能混入调试文字。所有进程应在同一 WSL 主机单调时钟上；跨机器/原始设备时钟输入不能直接使用。分类消息保留 checkpoint SHA256，供回合追溯。

控制器自身定期查询夹爪，不接受 stdin 伪造的夹爪反馈。stdin 阻塞时 watchdog 仍执行；输入结束、解码失败、积压或序号逆序终止正常松开并尝试收紧。最多缓存64条分类，超出即失败，不偷偷丢弃数据当成功。

物理执行必须另加 `--execute`，它调用现有 `device.arm()`，不会解除 `configs/gripper-motion-hold.json`。2026-11-12拉取的驱动已通过空载小行程复测并清除了旧故障阻断，见[最新记录](gripper-retest-20261112.md)；本控制器尚未做带尺子闭环试验。执行前须按夹爪几何/夹持条件重设 grip 及范围；现有 commissioning 适配器的每步和会话行程限制均保留，过大任务被拒绝。它不是完整生产控制器。

FINISHED 后执行入口保持夹持并继续读反馈，直到操作者扶住尺子并 Ctrl+C，才按现有驱动退出流程关闭使能。异常退出也可能关闭使能，不能把失电状态当作可靠持稳。这里是说明实现的实际行为，不代表本次已经运行过设备。

## 示教记录与训练边界

每周期记录：SNN 类别/方向/两项置信度与来源、可用时间；实测闭合量与查询时间；完整夹爪原始反馈（在线模式）；状态与计时；教师动作意图；成功下发的命令和设备回执。ACK 不代表机械动作完成。终端 hold 期间继续记录反馈，不制造新的 BC 命令标签。

`manifest.json` 保留目标计时、控制参数、运行方式、synthetic 标志、原始触觉目录链接和人工 outcome。实际录制可用 `--tactile-recording` 指定同期原始数据目录。旧 capture GUI 与发布器不能重复占用 Speck；发布器需在同一原始采集链输出分类。没有对应原始事件/GelSight 的教师 trace 不能当作完整多模态训练集。

`bc_rows(path)` 仅接受 `synthetic=false`、`status=finished`、人工核验后设 `outcome=accepted` 且链接到完整并通过 quality 检查的原始触觉回合。它输出一维 `target_robotis`，对应决策当时可用的观测及实际下发动作，不把实测闭合量或 shadow 提案作专家标签。人工接受还需核对原始录制确实与本回合同步，记录 checkpoint/参数版本。自动日志完整性不证明示教正确。

现有8槽位网络训练器要求完整动作标签；独立夹爪示教没有机械臂动作，不应捏造七个零当作实测标签。后续接具身训练时需要夹爪损失 mask/专项数据适配，此次没有改训练器。目标计时也应作为学生任务条件，尤其是在混用0.3/0.5/0.8秒示教时。

## 验证与未完成项

软件测试覆盖计时从 SNN 起点开始、去除重复预测确认、方向异常、未滑超时、输入过期、控制超时、松开上界、提前补偿、失败命令记录、synthetic 数据隔离、无机械臂/Torch依赖的控制核心和 checkpoint 分类头接入。

尚未实机验证：夹持位置、松开速度/行程、分类置信度及物理方向映射、机械制动响应、准确下落距离，以及触觉采集程序的在线分类发布接线。当前提供可执行的控制核心/在线消费者/推理接口和记录合同，不声称整机闭环已经验收。

## 2026-11-12 拉取后兼容性复核

基于 `515a5d4` 的新版 ROBOTIS 初始化与电流上限逻辑，新增16项控制测试再次通过。位置模式连接后由驱动把 `config.current_cap_a` 更新为设备现有电流上限，因此在线消费者读取的是实际采用的上限，不会使用过期的0.2A值。没有改回旧驱动，也没有重新执行任何硬件动作。实时采集预测发布接线仍未完成；示例 grip_closure=0.44 不可直接视为当前尺子的夹持位置。
