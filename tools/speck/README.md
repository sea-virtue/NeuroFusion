# Speck 通用工具

此目录提供各 demo 可共享的模型导出、临时RAM配置部署、检查与恢复。
当前滑动重抓取的入口与参数见 [slip-recatch](../../demos/slip-recatch/README.md)。

## 常用命令

从项目根目录执行。`export`只导出文件；`inspect/deploy/restore`会打开Speck，均不控制夹爪。
每次输出路径必须是尚不存在的新目录。

```bash
# 导出新的片上配置；已有导出目录不会被覆盖
bash tools/speck/deploy.sh export \
  artifacts/checkpoints/gelneuro-snn-20261111/final/all-data.pt \
  --out demos/slip-recatch/models/speck-next

# 检查新SDK会话中的初始配置
bash tools/speck/deploy.sh inspect --output tools/speck/outputs/inspect-next

# 独立部署，并保存 before.json、applied.json 和 report.json
bash tools/speck/deploy.sh deploy \
  --config demos/slip-recatch/models/speck/speck2f-config.json \
  --seconds 5 --output tools/speck/outputs/deploy-next

# 恢复同一板卡此前保存的配置
bash tools/speck/deploy.sh restore \
  --config tools/speck/outputs/deploy-next/before.json \
  --output tools/speck/outputs/restore-next
```

`deploy --self-test`会向CNN首层注入人工脉冲，只用于链路诊断，不表示触觉识别准确率。
日常重抓取直接运行对应demo即可，它会在持有设备的同一进程内重新加载配置。
独立部署后退出进程不代表RAM配置在下一次SDK会话中仍然有效，也没有写入Flash。

## 环境

使用共享的 `.venv/live`（Python3.12、Torch2.6CPU、NumPy2、Samna0.48.6）和
`.venv/chip-tools` 工具叠加目录。独立工具的精确版本在 [requirements.txt](requirements.txt)。
已有 `.venv/live` 后可重建工具叠加目录：

```bash
.venv/live/bin/python -m pip install --no-deps --target .venv/chip-tools \
  -r tools/speck/requirements.txt
```

`deploy.sh`和demo入口会自行设置 `PYTHONPATH`，不需要修改全局Python环境。
训练权重与设备SDK仍需在本机准备，不随Git提交。

## 数据路径与验证边界

当前训练网络导出为DVS4×4池化和四层Conv/IAF，最后输出16个特征通道的Spike。
原四组EMA、log1p、64维z及分类头在主机执行；16通道不是16个分类结果。
模型导出使用训练时的量化权重、阈值和增益，mapping记录原checkpoint哈希。

部署前执行SDK配置校验；应用后逐字段比较SDK返回配置。
Samna0.48.6在开启监测时观察到 `factoryConfig.ioSel` 从0规范化为32，该特定差异被明确处理；
其他字段差异仍拒绝。SDK读回对象不是独立电气测量。

设备辅助时钟与DVS共用StopWatch，当前请求100 Hz并保留10 ms水位滞后。
已尝试额外核心定时脉冲与手动SDK轮询，主机批次仍约每10 ms到达；当前没有确认可用的超过100 Hz分类发布方案。
合成脉冲通过片上网络不等于物理滑动分类有效。更换导出模型后仍需用实物确认。

通用实现保留在 `src/neurofusion/speck_deploy.py`、`adapters/speck_chip.py` 和原训练导出模块，
其他demo可直接复用，不需要复制进任务目录。
