# 编码器与动作头候选（2026-11-01）

本轮完成官方代码与编码器权重下载、候选工厂、现有运行时协议适配。保持 `runtime/engine.py`、原数据协议和合成演示不变；事件编码器、接触记忆更新和完整训练主链仍待实现。

## 规模与训练预算

`M=百万参数，B=十亿参数`，10B 即 10000M。下表是本仓库具体配置的参数量，不把论文完整策略的参数量算作某个动作头的参数量。

| 候选 | 参数量（含本项目投影） | 默认可训练参数 | FP32 参数本体约 MB | 已下载的官方权重文件 |
|---|---:|---:|---:|---|
| RGB / GelSight 单帧基线 ResNet18 | 11.242M | 0.066M | 45.0 | 46,830,571 bytes，约46.8MB；含原始分类层，加载后移除 |
| RGB DINOv2 ViT-S/14 | 22.106M | 0.049M | 88.4 | 88,283,115 bytes，约88.3MB |
| GelSight Sparsh-DINO ViT-Small | 21.943M | 0.049M | 87.8 | 87,593,104 bytes，约87.6MB |
| ACT 仓库 MLP 基线，400→256→256→5 | 0.170M | 0.170M | 0.7 | 无本任务训练权重 |
| ACT 条件适配，宽256，编码/解码/潜变量编码器各2层 | 5.476M | 5.476M | 21.9 | 无本任务训练权重 |
| DP 条件一维 U-Net，通道64/128/256 | 4.433M | 4.433M | 17.7 | 无本任务训练权重 |
| DP Transformer，4层，宽256，4头 | 4.850M | 4.850M | 19.4 | 无本任务训练权重 |

编码器默认冻结，仅训练输出投影；改变 `output_dim` 会改变投影参数。表中的动作头是轻量参数配置，采用上游实现，不宣称与各论文原始配置、性能等同。动作块长度16，动作5维，条件为 h256+c128+q16=400维。

最重的当前组合约49.5M（0.050B）参数，尚不包含待实现的事件编码器和状态更新模块。默认 ResNet+Sparsh+MLP 约33.4M，其中约0.285M可训练。这两者都远低于10B。

权重文件大小不是训练显存。FP32 参数本体约为参数数×4字节；标准FP32 Adam 还要存梯度与两个动量，参数相关内存合计常按可训练参数×16字节估算，不含激活、缓存及实现差异。冻结骨干的 no_grad 前向避免保存骨干反向激活，投影保留梯度。

**预算建议是工程估计，未在4090/A100实测**：先用224×224、冻结两个图像骨干、动作长度16、小批量8～16；需要更大有效批量时使用梯度累积。4090 24GB 是这些候选的合理首选目标，单张 A100 也可用于后续更大序列实验。以后训练接触记忆时，长序列反向传播可能成为主要显存来源；优先预计算冻结骨干的原始特征、训练投影和状态模块，再考虑更长序列或解冻骨干。不能由模型参数量保证任何批量一定能放下或达到控制频率。

本机实际为 RTX4060 8GB，本轮在 minist 安装 **CPU PyTorch 2.6.0** 做结构验证；没有在 A100/4090 做 GPU 实验。GPU 机器应先安装同版本兼容的 CUDA PyTorch/Torchvision，然后安装本项目 models extra；CPU 版不会自动获得 CUDA 支持。

## 切换与调用

选择在 `configs/candidates.toml`，维度仍由 `configs/baseline.toml` 管理。可独立组合：

- RGB：`resnet18` / `dinov2_vits14`
- GelSight：`resnet18` / `sparsh_dino_small`
- 动作头：`mlp` / `act` / `diffusion_unet` / `diffusion_transformer`

共有16种候选组合；本轮逐个验证组件和所有动作头的运行时接入，没有比较16种组合的任务成功率。

```bash
cd ~/neurofusion/neurofusion
conda activate minist
python -m pip install -e '.[models]' --no-build-isolation
python -m neurofusion.models.smoke --all --output outputs/candidates-check.json
python -m unittest discover -s tests -v
# CUDA机器安装CUDA版PyTorch后，可验证相同接口：
python -m neurofusion.models.smoke --all --device cuda --batch-size 8
```

`--all` 对每种组件分别做前向/反向，使用合成输入，并严格加载编码器官方权重；不代表完成任务训练、实时性测试或融合模型。输出报告包含实际参数量、可训练参数、权重字节数、合成损失与条件梯度检查。报告文件不覆盖已有文件。

```python
import torch
from neurofusion.config import Config
from neurofusion.models.candidates import CandidateConfig, build_encoders, build_action_head
from neurofusion.models.encoders import prepare_sparsh

dims = Config.load('configs/baseline.toml')
choice = CandidateConfig.load('configs/candidates.toml')
rgb_encoder, tactile_encoder = build_encoders(choice, dims)
head = build_action_head(choice, dims)

# scene_images: [B,3,H,W]，float RGB [0,1]。
c = rgb_encoder(scene_images)
# current/previous/background: HWC uint8 RGB，同一触觉传感器。
# previous 由采集缓存按实际5帧间隔提供，不能每次都用当前帧代替。
touch = prepare_sparsh(current, previous, background)[None]
frame_feature = tactile_encoder(touch)
# frame_feature 交给帧校正模块，不直接替代接触记忆。
# h 来自之后要实现的统一状态更新链，q 是对应时刻机器人状态。
condition = torch.cat([h, c, q], dim=-1)
loss = head.loss(condition, normalized_actions)
loss.backward()
```

MLP 的标签为 `[B,5]`；ACT/扩散头的标签为 `[B,16,5]`。标签按训练集统计归一化；扩散采样裁剪在[-1,1]，因此推荐用训练集动作范围的中心/半范围做可逆归一化。ACT 当前接口要求完整动作块，数据集必须明确选择有效窗口，不接受未声明的填充标签。

`ImageBackbone.forward()` 输出的是投影后的特征。投影随机初始化，需要训练；加载了骨干预训练权重，不代表投影已经学会本任务。冻结骨干同时固定BatchNorm统计；投影仍可反向训练。不允许权重丢失时静默退回随机模型。只有显式 `pretrained=false` 才创建随机骨干，用于结构检查或未来从头训练。

## 接入原数据流

- `ImageEncoderBridge` 满足现有 `ImageEncoder.encode(ImageSample) -> Vector`。RGB绑定 `scene_rgb`，触觉绑定 `gelsight`；Sparsh的 `sample.image` 为包含 `current/previous/background` 的字典。RGB/单帧触觉传入PIL或HWC数组。
- RGB 输出维度为 `context_dim`，发布到现有 `Kind.CONTEXT`；GelSight 输出为 `frame_feature_dim`，发布到 `Kind.FRAME`。应在编码完成后记录可用时间，不把编码前时间伪装成完成时间。
- `ActionHeadBridge` 满足 `UnifiedActionHead.predict(Snapshot) -> Vector`，拼接最新 `[h,c,q]`。必须提供训练集得到的 `action_mean` 和正的 `action_scale`，桥接返回物理动作单位。
- ACT/扩散生成长度16的动作块；首版桥接每次控制时刻重新以最新h推理，只取第一个动作，不缓存执行剩余动作。这样事件有机会参与下一次计算，但推理是否赶得上周期尚未测量。离线任务实验之后再决定是否需要别的块执行方案。
- 元组桥接仅供推理。训练直接调用张量 `forward/loss`，保留梯度；原tuple参考调度器不承担跨时间训练。

```python
from neurofusion.models.bridges import ActionHeadBridge, ImageEncoderBridge

scene = ImageEncoderBridge(rgb_encoder, 'scene_rgb')
gel = ImageEncoderBridge(tactile_encoder, 'gelsight')
runtime_head = ActionHeadBridge(head, action_mean=train_action_center,
                              action_scale=train_action_half_range, dimensions=dims)
# 用 runtime_head 替换现有 Engine(..., head=..., sink=ShadowSink()) 的动作头。
# 其余 frame/event/propagation 必须由真实实现或明确标注的测试组件提供。
```

## Sparsh 预处理与比较边界

采用上游背景差分 `(frame-background)/255 + 0.5`，裁剪并量化成uint8后，复用其方向处理/中心裁剪/224 resize；通道顺序为当前帧在前、历史帧在后，不额外施加ImageNet归一化。骨干使用sinusoidal位置编码、一个register token；取patch tokens均值，再投影。这一池化是本项目的简单接口选择，并非Sparsh所有下游头的统一配置。背景和历史帧必须来自同一传感器；采样率变化时记录真实时间间隔。

如果选GelSight单帧ResNet18基线，输入原始单张RGB触觉帧，使用ImageNet预训练及归一化。这与Sparsh的时序输入不同，比较时应记录输入历史差异，不能把所有收益归因于骨干结构。

## 现有证据

`outputs/candidates-initial.json`：严格加载3种官方编码器，验证投影梯度和冻结边界；4种动作头前向/反向、采样，检查条件输入梯度。初轮16项unittest通过，包括4种动作头接入未改动的Engine。完整H02训练链、真实事件利用、成功率、GPU峰值显存、机器人闭环和实时频率均待后续验证。

代码来源和固定提交见 [third_party](../third_party/README.md)。模型大小报告来自本机实例化统计；官方模型说明见 [Sparsh](https://huggingface.co/facebook/sparsh-dino-small)、[DINOv2](https://github.com/facebookresearch/dinov2)、[ACT](https://github.com/tonyzhaozh/act)、[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)。
