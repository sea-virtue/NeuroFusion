# 夹爪手动控制

在本机仓库目录运行以下命令。`scripts/gripper.sh` 使用本项目 `.venv` 的 Python 和官方 DYNAMIXEL SDK，不需要激活 conda、加载模型或连接 Franka。夹爪由 U2D2 和独立电源连接；默认串口 `/dev/ttyUSB0`、2 Mbps、ID 1。

```bash
cd /home/diaohh/neurofusion/neurofusion

# 只读取状态；不使能、不移动。
./scripts/gripper.sh status

# 从实测当前位置再闭合 12 个计数，到位后保持使能 2 秒，再关闭使能。
./scripts/gripper.sh close --steps 12 --hold-seconds 2

# 从实测当前位置再张开 12 个计数。
./scripts/gripper.sh open --steps 12

# 移动到指定绝对位置，例如 230；距当前位置必须不超过 20 个计数。
./scripts/gripper.sh move --position 230
```

`close/open` 的 `--steps` 为编码器位置计数，默认 12、允许 1..20，不是毫米；表示单次小幅闭合/张开。位置计数增大为闭合，减小为张开。`move --position` 表示绝对位置，同样保留单次 20 个计数的限幅，目标还须在设备读回的位置范围内。命令输出会显示初始位置、目标和实测到位结果。

运动命令执行顺序：读取状态 → 预置当前位置和速度/电流参数 → 开启使能 → 立即刷新当前位置 → 发送位置目标 → 持续读取并确认到位 → 可选保持 → 关闭使能并读回确认、恢复原 RAM 参数。每条命令都从实测位置计算，不直接依赖上一次的目标值。

默认到位后关闭使能；`--hold-seconds 2` 会先保持使能两秒。关闭使能后不再持续提供主动夹持力。此入口用于已固定、空载、运动区域清空的手动小行程调试，未测试带物体持续夹持或全行程开合。

位置模式的电流上限沿用设备现有 Current Limit，本机读值为 820 raw，约 3.2964 A。该值不是持续输出电流。命令不会自动切换设备模式、修改 EEPROM、PID 或固件。

每次运行自动创建一个新的 `outputs/gripper-control/<动作>-<随机标识>/`，保存 `report.json` 和带阶段标记的实测 `trace.jsonl`。可用 `--output <新目录>` 指定位置，已有目录不会覆盖。

## 本机演示（2026-11-12）

实际执行：

```bash
PYTHONPATH=src .venv/bin/python -m neurofusion.gripper_control close --steps 12 --hold-seconds 2
```

初始位置 225、使能 0；使能后目标为 237。首次确认到位时实测 236，保持结束时实测 237，随后确认使能 0、电流 0、速度 0。没有执行回到初始位置的命令，本次演示是单次闭合。

5 项新增离线测试通过，覆盖只读状态、开合方向与绝对目标、越界目标在使能前拒绝、等待失败后的关闭/恢复，以及非法步幅/保持参数不打开串口。真实闭合/保持/退出也已通过上述演示验证。

证据归档：[演示记录](validation/gripper-manual-control-20261112.json)。
