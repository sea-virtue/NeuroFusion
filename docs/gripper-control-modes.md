# 夹爪位置模式、电流模式与参考代码修正（2026-11-11）

本次根据用户提供的四份夹爪参考脚本修正正式适配器，保留官方 DYNAMIXEL SDK、现有模型动作合同和实机异常锁。只完成软件实现与离线验证，没有在本次修改中连接设备、切换模式或执行夹爪运动。历史测试未通过的根因仍未确认。

## 两种模式及模型接口

| 项目 | 位置模式 | 电流模式 |
|---|---|---|
| Operating Mode | 5：Current-based Position | 0：Current |
| 配置 | `GripperConfig(control_mode='position')`，默认 | `GripperConfig(control_mode='current')` |
| 指令 | `command(proposal)`：一维绝对闭合比例 0..1 | `command_current(current_a)`：带符号安培值 |
| Goal Current 含义 | 内部位置控制器可使用的最大电流 | 内部电流闭环的目标电流 |
| 启动预置 | 当前位置、正的电流上限、正的速度参数 | 目标电流 0，再开启使能 |
| 反馈 | 位置、速度、电流、电压、温度、使能、错误 | 同左；位置目标寄存器不代表本模式的控制目标 |
| 完成判据 | `wait()` 检查连续实测到位；ACK 不代表到位 | 没有位置到达语义；电流 ACK 不会生成 `gripper_executed` |
| 退出 | 关闭使能并确认，再恢复原电流/速度/加速度参数 | 尝试目标电流归零，然后无论归零是否报错都尝试关闭使能；不恢复旧的非零电流目标 |

模型及滑移教师仍输出开合比例，`q[15]` 仍是实测闭合比例，现有训练标签与 Franka 布局不变。电流接口是独立的手动调试接口；不能把已有开合输出直接当作安培值。以后若训练电流动作，需另行定义标签、归一化与已执行反馈。

电流控制不是 MIT 接口。当前没有刚度、阻尼、前馈力矩的组合指令，也没有上位机阻抗控制器。电流与指尖夹持力之间尚未标定，不能从安培值宣称牛顿数或安全接触力。

## 正式代码的变化

- `src/neurofusion/adapters/robotis.py`：明确分离两种控制模式和回执。动作模式不匹配、非有限值、超出电流上限、越过会话范围等情况会拒绝发送。保持默认只读、运动锁、初始关闭使能、写回复和型号识别等约束。
- 旧版 RN（型号 35073）在使能前一次写入连续的 10 字节：位置 596/4 字节、速度 600/4 字节、电流上限 604/2 字节。仍由官方 SDK 编码、CRC 校验和字节填充，并等待单播应答。
- RN 的反馈改为一次读取 596..625 的 30 字节，再读取使能和硬件错误。正确处理速度与电流之间的保留字节，以及有符号电流/速度。短包拒绝解析。它降低查询次数，不保证数据原子快照或硬实时性。
- RN(A)（型号 35074）的布局不同，保持独立地址路径，不能套用 RN 的连续写。两种型号的电流单位分别为约 4.02 mA/raw、1 mA/raw；RN(A) 路径仅离线测试。
- `set_mode('current'/'position')` 是显式 EEPROM 配置接口：要求允许写入、异常锁解除、无活动运动会话、使能关闭，并读取确认。`arm()` 和下面的命令行入口均不会自动切换模式。
- `hold_current(current_a, seconds)` 只发送一次电流目标，持续读反馈并检查状态，结束或异常时归零并关闭使能。时长不得超过配置的 `timeout_s`。这不是设备端看门狗，主机崩溃或 USB 阻塞时仍需要独立停机措施。
- `src/neurofusion/gripper_benchmark.py`：删除零电流保持、使能后强制改目标及电流爬升的旧初始化；改为在关闭使能时预置实测位置、100.5 mA 上限、0.456 rpm 速度参数，再使能检查。保留原有小行程、电流和异常停止边界，没有照搬参考脚本更高的限值。

零电流不提供有效的位置保持能力，过去零电流下的漂移不能单独证明位置控制器损坏。历史原始测量和异常锁保留，本次软件修正也不能证明过去约 1 A 的反馈读数已经得到解释。

## 使用入口与边界

现有位置/影子流程不变：

```bash
python -m neurofusion.gripper_check --seconds 3 --output outputs/gripper-read-new
```

新增独立电流入口，默认也只读，不修改当前硬件模式：

```bash
python -m neurofusion.gripper_current_check --output outputs/gripper-current-read-new
```

只有后续完成现场确认、解除异常锁并将设备显式配置为 Mode 0 后，才使用以下带电试验。这里仅记录命令，本次未执行：

```bash
python -m neurofusion.gripper_current_check \
  --exercise --operator-ready --current-a 0.05 --seconds 0.25 \
  --output outputs/gripper-current-trial-new
```

入口限制目标电流绝对值不超过 0.2 A、时长不超过 1 秒，安培值按设备分辨率向零量化。正负方向需实物核对。成功报告只表示这段有界程序完成，不表示位置精度、夹持力或机械响应性能通过。

## 频率与参考代码的取舍

固定位置目标可写一次，由内部位置闭环继续追踪；固定电流目标也会持续生效，直到新目标或关闭使能。持续发包用于更新轨迹，不是维持同一目标的必要条件。

参考脚本的 500 Hz 是请求的上位机发送节奏，100 Hz 是目标反馈节奏。其广播写没有 ACK，串口 write 返回时间不等于设备收到时间或机械响应时间。本次没有直接引入广播或宣称 500 Hz 已通过，正式接口继续使用可确认的单播写；独立实时调度器仍待实现和实测。

参考目录中的测试脚本引用原作者的 `data_collecetion/1_robotis_rh.py`，不能直接原样启动；benchmark 还会在启动时使能，再倒计时自动执行轨迹。本次借鉴控制语义和寄存器布局，不将这些自动运动流程接入模型。

## 来源和验证

用户提供目录：`文献资料/04_硬件手册与参考代码/`，参考文件：

- `1_robotis_rh.py`
- `1_robotis_rh_current_debug.py`
- `01_gripper_position_debug_v4.py`
- `01_rhp12_streaming_benchmark.py`

官方依据：[RN 控制表](https://emanual.robotis.com/docs/en/platform/rh_p12_rn/)、[RN(A) 控制表](https://emanual.robotis.com/docs/en/platform/rh_p12_rna/)、[DYNAMIXEL SDK](https://github.com/ROBOTIS-GIT/DynamixelSDK)。

离线测试覆盖 RN/RN(A) 地址分离、正负电流及单位、模式混用拒绝、非零位置预置、块读取和短包、使能后实测电流检查、清理故障后的关闭使能、异常锁、已有位置动作兼容性和电流入口报告。

WSL 的 `minist` 环境中，本次提交对应项目的 71 项测试通过，其中夹爪相关 31 项；另有本地未提交的 16 项滑移兼容测试通过，共 87 项。验证记录见 [gripper-control-modes-20261111.json](validation/gripper-control-modes-20261111.json)。这些测试不替代真机开合、夹持力及延迟验证。
