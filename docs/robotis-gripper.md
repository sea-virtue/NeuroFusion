> 2026-11-11 软件修正见[位置与电流控制模式](gripper-control-modes.md)：已分离 Mode 5 开合与 Mode 0 电流接口，修正初始化和块读取；默认动作仍是开合比例。没有新增真机通过结论，正常运动锁保留。下面保留原接入与实测记录。

# ROBOTIS 独立夹爪接入（2026-11-04）

正式入口是 `src/neurofusion/adapters/robotis.py`，使用官方 DYNAMIXEL SDK，不重新实现串口协议。当前实机返回 RH-P12-RN，型号号35073、固件25、ID1、2Mbps；不能按名称中的“(A)”猜测控制地址。适配器按PING型号选择RN/RN(A)控制表；RN(A)分支尚未实机验证。夹爪独立连接上位机，不经过Franka。

## 已验证范围

- Windows USB `0403:6014`、COM8；转接WSL后 `/dev/ttyUSB0`，重新插拔时设备号可能变化。
- 读取真实位置、运动状态、目标位置、电流、电压、温度、使能状态和硬件错误。
- 实机33组反馈已通过正式模型的`step`，产生影子动作；没有发送网络预测动作。
- SDK写指令、目标寄存器预置、电流/速度设置、电机使能及关闭均收到回复。完整开合往返仍待实测确认，不能称为运动闭环已通过。
- 第一次20计数目标被步幅保护拦截；第二次尚未运动即遇到串口超时。随后按官方建议把当前FTDI延迟从16ms调至1ms，并将SDK单包容许超时至少设为150ms，不自动重发写指令；连续只读复核通过。
- 位置约207/208，与此前503不同；用户确认没有手动移动，目视观察到夹爪自行张开。原因未确认，不能以软件步幅限制宣称物理运动已受限。已添加`configs/gripper-motion-hold.json`，`arm()`在任何运动写入前拒绝使能，`--exercise`当前也会被拒绝。只读反馈仍可用。保护触发和失败报告完整保留。

## 框架中的数据流与单位

```text
U2D2实测位置 -> GripperSample -> q[15] -> 时间传播/接触记忆 -> 动作头影子提案
独立夹爪物理提案 -> command -> WRITE接收确认 -> read/wait实测到达 -> gripper_executed
```

本次明确临时合同：`q[15]`为夹爪绝对闭合比例，其余15个q字段保持原值；夹爪动作也是绝对闭合比例，0对应硬件最小位置（张开）、1对应最大位置（闭合）。按读取到的硬件上下限进行线性映射。该数不是毫米开口宽度、夹持力，也不是已经完成触觉标定；今后真实训练数据须采用同一字段和单位。旧合成训练检查点不因此获得真实控制能力。

串口接收时间使用主机单调时钟，标记`host_receipt_proxy`；不是设备测量时间，不据此声称物理响应时延。传输ACK只证明指令接收；只有后续实测位置连续进入容差范围，才产生`gripper_executed`，而非立即把目标回填成已执行动作。当前没有自动重试运动目标，没有独立实时调度。

驱动默认只读。运动会话必须显式开启，要求现有模式为5、电机最初关闭、写回复开启；不改EEPROM、不升级固件。使能前先写当前位置目标，暂时限制电流约0.2A、低速，单步20计数、会话起点附近25计数；默认测试10计数。异常或超时关闭电机使能，确认关闭后恢复原电流/速度/加速度RAM参数。运行测试会改变目标寄存器，但不会在退出时追赶旧目标；设备端保留最近目标，重新使能前再次预置实测位置。

动作适配器拒绝`normalized=True`的网络提案。当前模型的随机或合成训练输出始终只记录；测试用人工明确指定的小幅物理目标。Franka分支不发送任何指令。

## 使用方法

```bash
cd ~/neurofusion/neurofusion
conda activate minist
python -m pip install -e '.[hardware]' --no-build-isolation
# 只读反馈接回网络；输出目录必须是新的
python -m neurofusion.gripper_check --seconds 3 --output outputs/gripper-read-next
# 当前被异常锁阻止；须先查明意外张开原因并完成运动验证，不能直接运行
python -m neurofusion.gripper_check --exercise --delta 10 --output outputs/gripper-motion-next
# 两路触觉重新连接并转接WSL后，可同时加只读夹爪
python -m neurofusion.live_check --seconds 20 --gripper-port /dev/ttyUSB0 --output outputs/three-input-next
```

最后一个三设备入口已编写，本轮没有重新连接两路触觉，所以尚未联合实测。`gripper_check`本身只测试夹爪输入，RGB/Franka/双触觉缺失均在报告中列出。USB重插后需要重新`usbipd attach`；FTDI延迟设置也可能重置。

## 官方参考

- [RH-P12-RN控制表](https://emanual.robotis.com/docs/en/platform/rh_p12_rn/)：当前实机；使能562、目标位置596、实测位置611。
- [RH-P12-RN(A)控制表](https://emanual.robotis.com/docs/en/platform/rh_p12_rna/)：不同型号表，使能512、目标564、实测580。
- [U2D2及USB延迟设置](https://emanual.robotis.com/docs/en/parts/interface/u2d2/#usb-latency-setting)
- [官方DYNAMIXEL SDK](https://github.com/ROBOTIS-GIT/DynamixelSDK)

9项新增硬件边界测试、全套31项测试通过。软件测试不等于真实机械响应验证。

## 用户确认后的收尾

已在电机使能持续为0时验证目标寄存器：写202读回202、写212读回212；两次实测位置均207，没有运动。最后将遗留目标503改为当前实测约207，电机仍关闭、电流0、硬件错误0。证据`outputs/gripper-20261104-disabled-write/report.json`。这证明目标信号双向通信可用，不证明实际闭合控制已通过。

最初测试曾短暂使能电机，不能排除它与意外张开的关系；官方旧版手册提醒使能切换可能影响位置读数，但本轮证据无法区分坐标变化、控制行为或机械回弹等原因。没有继续通过增加电流/速度尝试解决。后续应先检查固定/负载情况和使能时的实际坐标行为，再在有现场观察的受控条件下重新验证；保留当前运动锁。
