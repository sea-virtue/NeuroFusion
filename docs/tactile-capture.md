# 触觉先导数据采集：现在可以开始录

正式代码：`~/neurofusion/neurofusion`。此模块只采数据，不运行编码器、融合或动作头，不发送夹爪/机械臂命令，也不需要外部 RGB 相机。保留原始事件供之后训练事件编码器；当前并不把未训练 CNN 的特征当作数据集。

## 打开录制窗口

在 WSL 终端中：

```bash
cd ~/neurofusion/neurofusion
conda activate minist
python -m neurofusion.capture gui
```

先关闭旧的 GelNeuro/GelSight 查看窗口，避免两个程序抢同一设备。GUI 的预览与录制使用同一条输入。**建议两路一起录**；只研究事件时可取消 `Also record GelSight`，或运行 `python -m neurofusion.capture gui --events-only`。

## 界面怎么操作

1. 固定装置，约定实际物体移动的上下左右，并在 `direction_convention` 填清楚。例如“以正对 GelSight 的操作者视角，上为靠近桌上标记 A”。不把两路画面的像素方向直接当作同一物理方向。
2. 填物体编号 `object_id`、材质 `material`、安装关系 `mounting_notes`。安装关系至少说明两传感器是否同时接触同一物体、是否在不同位置；无法同时接触时记下来，不能当作对应接触真值。
3. 选 `task`、`direction`、`speed_level`（slow/fast 仅操作等级）、`contact_level`（light 等仅主观接触程度），确认独立重复编号 `repetition`，点 **Record new trial**。
4. 等待设备打开，看到 **HOLD STILL** 才算开始计时。默认每段 8 秒：前 2 秒不动，中间 4 秒操作，后 2 秒不动。背景任务全程无接触，静止接触任务全程保持接触不动。开始停止、反转、按压释放会有进一步分段提示，并提示音；这些提示的实际主机时刻会保存，但不是已测得的物体动作时刻。
5. 可选点 `Mark: motion starts/stops` 留下人工标注；按按钮的延迟与误差仍存在。`notes` 可以写本段实际执行情况或失误。字段在点击 Record 时固定；若录制后发现错误，可在下一段的 notes 记明作废的目录名，原数据保留。
6. 结束后等待 **SAVED AND VERIFIED**。程序会全量解码照片并核对事件索引。若出现 **CHECK FAILED**，不要直接放入训练集，保留该段供排查。`Stop` 提前停止会保存为 `interrupted`，不会伪装成完整试验。
7. 每次重复重新开始接触，成功后重复编号自动加一；切换方向/任务时自行把重复编号改回 1。每段都会生成独立新目录，不覆盖以前数据。

事件预览只累计近期事件，静止时少甚至为空属正常现象。**第一轮一定做一段有接触变化的试样，确认按压/移动时事件计数明显增长**；完整性检查通过只说明文件可读，不代表你按计划完成了动作或模型能识别动作。

## 第一批采什么

先用一个容易稳定接触的物体，接触程度尽量一致。不要先混杂太多材质和条件。

| task 选项 | 实际操作 | 建议完整批次 |
|---|---|---|
| background | 两触觉表面都明确无接触，全程静置；背景不能用不明状态的第一帧代替 | 3 段，8–10 秒 |
| static_contact | 接触后保持不动，不要横向滑动 | 3 段 |
| slide | 上、下、左、右；物体相对触觉表面滑动，移动方向沿用统一物理定义 | 4 方向 × 2 速度等级 × 3 次独立重复 = 24 段 |
| start_stop | 按提示移动、停住、再移动；建议先固定同一方向 | 3 段 |
| reverse | 按提示先向初始方向移动，再反向；direction 记初始方向 | 3 段 |
| press_release | 按压、保持、释放，尽量不横向滑动 | 3 段 |

共 **39 段左右**。先采 12 段小试：背景 2、静止接触 2、四方向各 1、按压释放 2、开始停止 1、反转 1；检查事件响应、图像、操作标注和方向约定后，再补足上表数量。不要只录滑动，避免把“事件多”学成滑动。旋转和混合运动放第二批。

无外部运动观测时，这批提供的是**试验条件/提示层面的弱标签**。暂不声称毫米/秒、接触力、精确滑动起止或真实抓持失稳。训练时不能把整段 slide 都标成正在滑动：它含前后静止段；中间提示段也存在人的反应延迟。先做整段/宽窗口方向识别或人工复核，精细时序监督需后续补证据。无机器人执行记录，不拿这批训练动作头；影子动作也不能作专家标签。

GelSight 不是事件编码器训练的强制依赖，但同期记录能用于冻结编码器评估、双触觉融合比较和之后的辅助预测。不同位置的接触图像不能直接监督另一传感器的逐像素触觉状态。

## 保存了什么

默认：`data/raw/tactile/<UTC时间>_<task>_<唯一ID>/`，整个目录不进 Git。目录名使用 UTC（比北京时间早 8 小时），manifest 同时保存时间锚点。建议整段目录备份，按完整段/物体划分训练和测试，不能把同一段切出来的相邻窗口随机分到两边。

```text
manifest.json            # 格式版本、设备/相机配置、操作条件、时间语义、状态/错误、计数
annotations.jsonl        # 指令提示、人工按钮标注；都不是测量真值
quality.json             # 自动复核、实际图像频率/间隔、数量、字节数、提示/问题
events/
  format.json            # 二进制字段、类型、128×128 尺寸与设备时间单位
  events.bin             # 小端 int64，N×4：[x,y,timestamp_us,polarity]，每事件 32 字节
  batches.jsonl          # 含空轮询；批号、偏移/数量、读开始/主机收到时刻、设备首末时间
gelsight/                # events-only 时整个目录不存在
  frames.jsonl           # 每帧编号、主机读取/收到/保存时刻、原尺寸、文件名、字节数
  frames/00000000.jpg    # 原生 MJPEG 的逐帧 JPEG 字节，不缩放、不重压缩
```

相机本身用 JPEG 压缩；本模块保存它给出的原字节，并不声称这是 Bayer RAW 或无损传感器采样。数据路径不做背景相减、裁剪、分辨率缩小或特征提取。窗口预览单独缩小，丢预览不影响文件；原始写盘队列满则明确标为失败。最终关闭会刷新文件；这不是突然断电下的存储事务保证，异常后可检查部分数据，不自动宣布零丢包。

两传感器各有独立进程与有界写盘线程；没有模型推理。Speck 复用已跑通的 `SpeckSource`，启动等待期间的缓冲在共同主机启动门处排空并记录排除数量；之后保存每个 SDK 读取批次，包括空批次。5ms 是请求的**轮询间隔**，不是硬件每 5ms 生成的同步事件包，离线可依设备时间重新封窗。

主机时间用同一个 `monotonic_ns`，设备事件保留微秒计数。相机曝光时间当前未知，`device_timestamp=null`。共同启动门也不表示精确同步，不能直接用两个设备的时间差声称物理时延。

## 命令行与离线读取

```bash
# 默认两路同时录；只录事件时加 --events-only
python -m neurofusion.capture record \
  --task slide --direction left --speed-level slow \
  --object-id object01 --contact-level light --repetition 1 \
  --direction-convention 'operator-facing-GelSight; up=toward-mark-A' \
  --mounting-notes 'same-object; opposite-contact-patches' --seconds 8

# 复查一整段，生成/更新 quality.json；不会更改原始观测
python -m neurofusion.capture inspect data/raw/tactile/实际目录名 --save
```

```python
from neurofusion.capture.reader import iter_event_batches, iter_frames

trial = 'data/raw/tactile/实际目录名'
for metadata, events in iter_event_batches(trial):
    # events 是 N×4 int64 [x,y,设备微秒,极性]；可能 N=0
    # metadata 保留该 SDK 批次主机读开始/完成时刻
    pass
for metadata, rgb in iter_frames(trial):
    # 解码一张原分辨率 RGB uint8；不一次载入所有图像
    pass
```

## 本机验证与边界（2026-11-04）

验证目录 `outputs/capture-native-validation-20261104`：8 秒诊断采集，保存 16 个真实事件、1,558 个读取批次和 124 张 3280×2464 图像（约 22.85 MB，十进制）。全部照片可解码、事件与索引一致，写盘队列峰值 1，无应用层拒收记录。实际相邻帧平均约 17.2 Hz，最大接收间隔约 165ms；相机报告 25Hz **不等于本机已达到 25Hz**。该段基本静止，只用于工程验证，不能作为已标注滑动训练段。

最初逐帧 PNG 编码的诊断段 `outputs/capture-validation-20261104` 发生队列溢出，明确标为 failed 并保留，不能进入训练集。原生字节保存路径已替代该录制方案。11 项新测试与全部 45 项项目测试通过。

另实测 GUI 完整流程（打开预览→按钮录制→自动结束→后台全量检查）：`outputs/capture-gui-validation-20261104/` 的诊断回合保存123张图像、9个事件，检查通过。`outputs/capture-stop-validation-20261104/` 验证仅事件模式和提前停止：原始批次保留，无GelSight目录，状态为interrupted，不会误标成完整训练段。两类诊断都不计入人工标注训练数据。

尚未实现：自动动作真值、精确跨设备时钟映射、固定设备时间窗重放到训练链、事件编码器监督训练/数据划分工具、数据集规模与泛化验证。当前提供原始采集、元信息、独立读取和完整性复核；不把它称为完整训练平台。
