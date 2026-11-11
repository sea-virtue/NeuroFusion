# SNN 接入前检查点（2026-11-11）

本目录是本地 Git 检查点的一部分，保存三折评估模型、全数据试用模型、量化试用模型与验证摘要。原始 outputs 文件保留不动，SHA256 与来源见 manifest.json。

此时普通/门控/attention 融合、四种动作头及张量训练链已有实现，NeuroFusionPolicy 仍使用旧 PacketEncoder CNN，尚未接入这份 SNN。融合和动作头没有真实任务训练通过的结论；未来触觉辅助预测尚未实现。

summary.json 的分数来自留出折评估。final/all-data.pt 使用全部录制训练，没有独立评分，不能把交叉验证分数直接写成该模型的独立测试成绩。

这些是 PyTorch 检查点，不代表已经上芯片或完成机器人控制。原始采集和大型视觉骨干权重不包含在此提交中；视觉权重来源仍见 third_party/weights.lock.json。本地 Git 提交不等于远程备份。
