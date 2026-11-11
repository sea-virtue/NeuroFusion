# SNN 主链接入与验证（2026-11-11）

主训练入口默认读取 `configs/pipeline-snn.toml`：加载已归档 GelNeuro SNN，冻结脉冲主干与主机迹读出，64维连续特征进入共享接触记忆；不使用分类头输出代替接触特征。RGB/GelSight继续使用现成冻结骨干，投影、融合、动作头可训练。`configs/pipeline.toml`及直接构造CandidateConfig的旧默认保留CNN对照兼容；旧live_check仍是CNN联调入口，本轮验证为离线SNN主链，不宣称实时入口或芯片已迁移。

## 可复现命令

```bash
cd ~/neurofusion/neurofusion
conda activate minist
python -m neurofusion.training.pipeline --steps 3 --output outputs/snn-next.json --checkpoint outputs/snn-next.pt
python -m unittest discover -s tests -v
python scripts/verify_snn_flow.py --recording data/raw/tactile/20261104T082310Z_slide_5f021487 --matrix --output outputs/snn-flow-next.json
```

输出路径必须是新路径。源码/训练权重在Git存档中，视觉权重与原始录制需使用既有本地文件或按来源另行恢复；原始录制不上传远端。此验证脚本的背景参考来自 `outputs/sparsh-contact-probe-20261110/background.jpg`，该文件缺失时明确报错，不把首张接触帧自动当成空载背景。

## SNN 接口和状态

- 原事件列为 x,y,device_us,polarity，保持整数、非负计数；128×128按4×4求和到32×32，ON/OFF分通道，不能沿用旧CNN的log1p体素前处理。
- 校准检查点使用1ms原生特征步长；输入窗必须包含完整时间格。相邻事件窗必须连续，即使没有事件也传空窗，重叠或缺口报错。调用方负责关闭因果窗口和设备/主机时钟关系，本适配器不猜测跨源同步。
- 四层膜电位和四组主机迹随每条TensorState传递；每回合initial_state重置，detach同时截断SNN状态。SNN的历史与融合记忆是两个不同状态，不互相替代。
- 先冻结SNN，动作误差训练视觉投影、融合及动作头。允许显式解冻，但未做真实任务联合微调评估；解冻会改变已验证的分类表征。
- 完整策略检查点包含SNN参数、迹衰减及增益，恢复时不需要再次找到原独立SNN文件。它是模型检查点；没有保存优化器/在线运行状态，不能当作精确训练续跑或实时状态快照。

## 本轮证据

52项纯软件测试通过；新增测试检查SNN计数、整窗与跨包一致性、显式空窗、状态重置、冻结梯度边界和完整策略恢复。三个融合、四个动作头的12组组合在合成输入/标签上均完成一次训练及推理。默认attention+MLP的三步合成损失0.32553→0.28368→0.26133，视觉投影、融合、动作头有梯度，冻结SNN梯度为零，模型恢复一致。

真实录制的GelSight当前/历史帧与59个原始事件、64个1ms步完成离线前向。SNN迹非零；与同时间的空事件对照，记忆最大差约0.1547，归一化动作最大差约0.00768。这里只证明真实输入能经SNN影响输出；融合与动作头仍未用真实动作训练，没有正确控制结论。

录制缺少场景RGB、机器人反馈与专家动作，回放保留明确缺失说明；初始占位保持零，不能称为真实观测。时间使用原录制的主机接收代理，窗口选择使用操作提示仅用于挑选诊断区间，不能当精确运动真值或物理延迟。没有连接硬件或发送机械命令，也未部署SNN到芯片。完整摘要见 [验证记录](validation/snn-flow-20261111.json)。

## 怎样开始真实整体训练

1. 固定已训练SNN和图像骨干；先用小型attention+MLP，避免同时更换动作生成器。
2. 准备按完整回合划分的训练/验证序列：原始事件因果分窗、已预处理触觉图像、可得RGB、机器人与真实执行反馈、每个控制时刻的专家动作。缺失来源要明确；不能把接触类别、手动滑动方向或影子动作当机器人专家标签。
3. 当前训练数据合同仍是纯张量字典observations/targets；MLP标签B×8，块头B×16×8，第7槽为预留零，6个臂分量与1个独立夹爪分量须先明确单位和语义。归一化只从训练集拟合。数据转换、实际回合采样、长序列预热/截断与真实动作数据验收仍待完成，不把已有原始采集自动当作可训练动作数据。
4. 有可靠动作标签后运行 `python -m neurofusion.training.pipeline --data train.pt --validation-data val.pt --checkpoint outputs/task.pt`。未传--data使用合成调试样例，必须与真实实验区别。当前入口是小序列拟合原型，长数据训练需增加回合/窗口调度；当前验证不保证大规模训练或实时频率。
5. 真实动作标签不足时，可以用已有接触标签验证融合表征；未来触觉辅助预测仍未实现，不在本次补成隐式功能。
