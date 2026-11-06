# 2026-11-04 GelNeuro 与 GelSight 首次接入正式网络

## 已完成的真实输入链

Speck2fModuleDevKit（337d:5bc9）和GelSight Mini（0c45:636d）由Windows usbipd转接到Ubuntu-20.04，在既有minist中同时读取。SpeckSource复用了speck-lab中的配置克隆、原始DVS过滤、计时器启动及退出恢复流程；没有刷写固件、片上推理或更换Samna。

真实相机帧→六帧历史缓存→已有Sparsh预处理/编码器；真实事件→事件包统计网格/CNN；两路进入已有attention接触记忆，按实际循环产生统一动作头的影子提案。此次没有接RGB场景相机、Franka或夹爪反馈，相关状态保留缺失说明，不把它们称为有效真实观测。没有机械指令发送。

本次20.36秒：相机22帧，14个SDK批次、756个真实事件，31次观测记忆更新，17次影子动作提案。真实图像已人工目视检查，可见触觉表面标记点和局部形变。22项既有测试复核通过。

证据：`neurofusion/outputs/live-20261104-first/report.json`、`trace.jsonl`、`gelsight-first.png`和`gelsight-last.png`。报告中的passed仅指两路读取/网络接线通过，不包括夹爪或闭环。

## 运行入口

```bash
cd ~/neurofusion/neurofusion
conda activate minist
python -m neurofusion.live_check --seconds 20 --output outputs/live-next
```

输出目录不覆盖旧记录。相机节点变化时使用`--camera`指定，默认0。设备退出恢复Speck配置、停止采集并释放相机；USB附着保留供下一次使用。

## 本次边界与尚未通过的设备

- 夹爪控制板没有枚举为有效USB串口；Windows另发现`0000:0002`、Code 43（设备描述符请求失败）。该错误设备尚不能确认就是夹爪控制板，因此不能把其身份或原因写成已确定事实。仍需用户确认板型号、供电和数据线，成功枚举后才能读夹爪状态。未尝试盲发夹爪命令。
- 网络使用已下载的图像骨干，但融合、投影和动作头仍未经过真实任务训练。提案只是检查数值贯通，不能用于自动闭合夹爪。
- 首张真实帧暂作背景参考，未确认空载，不作为正式背景标定；Sparsh需要6帧历史才开始帧更新。
- 当前相机实际返回3280×2464，640×480请求未生效；串行CPU联调循环未优化，也不是50/100Hz实时调度器。动作生成与相机读取在该诊断入口中同循环进行，之后再实现独立采集/控制工作者。
- SDK每次轮询返回的一批事件作为一个诊断包，设备窗按实际最小时间戳到最大时间戳加1定义，暂非固定5ms因果封包器。此入口不代替H01完整数据封包及重放工作。
- 设备微秒时间戳保留在事件内容中；当前采集/曝光时刻未知，`captured_ns`明确标为`host_receipt_proxy`（主机取到该批数据的时刻），不能拿它声称设备曝光时间、跨设备精准同步或物理端到端时延。软件只做因果顺序联调，后续时钟映射仍待实现。

源码：`adapters/live_devices.py`和`live_check.py`。旧`adapters/pending.py`保留历史占位，新输入入口不调用那些占位类。
