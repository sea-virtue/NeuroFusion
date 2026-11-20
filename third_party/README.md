# 官方候选源码

这些目录已在 WSL 下载完成。源码不改动；`.gitignore` 排除上游工作树，Git 记录 `sources.lock.json` 中的精确提交与本项目适配层。换机器后运行：

```bash
python scripts/fetch_upstream.py
python scripts/download_weights.py
```

| 目录 | 官方来源 | 本次用途 |
|---|---|---|
| dinov2 | https://github.com/facebookresearch/dinov2 | 接入 ViT-S/14 编码器 |
| sparsh | https://github.com/facebookresearch/sparsh | 接入 Sparsh-DINO ViT-Small 编码器 |
| act | https://github.com/tonyzhaozh/act | 复用 mlp()；CVAE/Transformer 接共享状态条件 |
| diffusion_policy | https://github.com/real-stanford/diffusion_policy | 接入条件一维 U-Net 与 Transformer 扩散动作头 |
| reactive_diffusion_policy | https://github.com/xiaoxiaoxh/reactive_diffusion_policy | 下载保留，作为 GRU/PCA/慢快策略参考，未接入完整策略 |
| ImplicitRDP | https://github.com/Chen-Wendi/ImplicitRDP | 下载保留，作为力反馈/因果注意力参考，未接入完整策略 |

ResNet18 使用 Torchvision 0.21.0 实现，来源为 https://github.com/pytorch/vision/tree/v0.21.0 。

每个上游工作树保留完整 LICENSE 与版权说明。Sparsh 权重许可为 CC BY-NC 4.0；不同项目和文件的许可并不完全相同，以各自原文为准。

`weights.lock.json` 记录官方权重地址、字节数和 SHA256。权重保存在被 Git 排除的 `checkpoints/encoders/`，不下载任务动作策略或大模型权重。Sparsh 选择 87.6MB 的纯编码器 safetensors，未下载 734MB 的训练检查点。

上游导入的特别处理：ACT 的 `detr_vae.py` 顶层会连带导入相机/DETR/IPython。适配层用 AST 读取原文件的类/函数定义并原样执行，只省略无关的顶层导入；Transformer/CVAE/MLP 定义未重写。ACT 输入边界从图像特征改接共享状态 token，并替换原来硬编码的机器人/动作维度。这属于 ACT 结构的条件适配，不等同于原论文完整视觉 ACT 复现。

2026-11-04新增官方夹爪示例`robotis_rh_p12_rn_example`（Apache-2.0），固定提交见sources.lock.json。仅核对旧版RN控制表和使能顺序，未运行上游自动开合。
