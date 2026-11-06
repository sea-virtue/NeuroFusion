"""Frozen pretrained image backbones plus small trainable feature projections.

    RGB: float [B,3,H,W] in [0,1], RGB order. Tactile: prepared [B,6,224,224].
    forward() retains projection gradients; no tuple conversion during training.
"""
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn
from torchvision import models
from torchvision.transforms import functional as TF
from .upstream import import_upstream, project_root


class ImageBackbone(nn.Module):
    def __init__(self, name, output_dim=None, *, pretrained=True, frozen=True, checkpoint=None):
        super().__init__()
        self.name, self.frozen = name, frozen
        weights = project_root() / 'checkpoints' / 'encoders'
        if name == 'resnet18':
            self.backbone = models.resnet18(weights=None)
            checkpoint = checkpoint or weights / 'resnet18-f37072fd.pth'
            self.raw_dim = 512
        elif name == 'dinov2_vits14':
            module = import_upstream('dinov2', 'dinov2.hub.backbones')
            self.backbone = module.dinov2_vits14(pretrained=False)
            checkpoint = checkpoint or weights / 'dinov2_vits14_pretrain.pth'
            self.raw_dim = 384
        elif name == 'sparsh_dino_small':
            module = import_upstream('sparsh', 'tactile_ssl.model.vision_transformer')
            self.backbone = module.vit_small(img_size=(224, 224), in_chans=6,
                                            pos_embed_fn='sinusoidal', num_register_tokens=1)
            checkpoint = checkpoint or weights / 'dino_vitsmall.safetensors'
            self.raw_dim = 384
        else:
            raise ValueError(f'Unknown image encoder: {name}')
        if pretrained:
            checkpoint = Path(checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(f'Pretrained weights required: {checkpoint}; run scripts/download_weights.py')
            if checkpoint.suffix == '.safetensors':
                from safetensors.torch import load_file
                state = load_file(str(checkpoint))
            else:
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=True)
        if name == 'resnet18':
            self.backbone.fc = nn.Identity()
        self.pretrained_loaded = pretrained
        self.output_dim = output_dim or self.raw_dim
        self.projection = nn.Identity() if output_dim is None else nn.Linear(self.raw_dim, output_dim)
        self.backbone.requires_grad_(not frozen)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()  # Also freeze ResNet BatchNorm running statistics.
        return self

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != (6 if self.name == 'sparsh_dino_small' else 3):
            raise ValueError('Expected BCHW with 6 tactile or 3 RGB channels')
        if self.name == 'sparsh_dino_small':
            if images.shape[-2:] != (224, 224):
                raise ValueError('Prepared Sparsh inputs must be 224x224')
            inputs = images
        else:
            # ResNet weights' standard resize + center crop; same RGB transform for DINOv2.
            inputs = TF.center_crop(TF.resize(images, 256, antialias=True), [224, 224])
            inputs = TF.normalize(inputs, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        with torch.no_grad() if self.frozen else nullcontext():
            if self.name == 'sparsh_dino_small':
                feature = self.backbone.forward_features(inputs)['x_norm_patchtokens'].mean(dim=1)
            else:
                feature = self.backbone(inputs)
        return self.projection(feature)


def prepare_sparsh(current, previous, background):
    """Explicit current + historical frame; caller chooses the actual sample stride.

    HWC uint8 RGB, reproducing upstream compute_diff(offset=.5), orientation,
    center crop and PIL resize. Background must be the same sensor's no-contact
    frame. No silent substitution of current for the historical frame.
    """
    import numpy as np
    from PIL import Image

    def prepare(frame):
        frame, bg = np.asarray(frame), np.asarray(background)
        if frame.dtype != np.uint8 or bg.dtype != np.uint8 or frame.shape != bg.shape:
            raise ValueError('Frames/background must be matching HWC uint8 RGB arrays')
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Expected HWC RGB')
        # Exact upstream integer subtraction and uint8 quantization before resize.
        image = np.uint8(np.clip((frame.astype(np.int32) - bg.astype(np.int32)) / 255 + .5, 0, 1) * 255)
        if image.shape[0] < image.shape[1]:
            image = np.rot90(image, k=-1)
        h, w, _ = image.shape
        if h / w != 4 / 3:
            h2, w2 = int(h / (4 / 3)), w
            image = image[int((h-h2)/2):int((h+h2)/2), int((w-w2)/2):int((w+w2)/2)]
        image = TF.resize(Image.fromarray(image), [224, 224], antialias=True)
        return TF.to_tensor(image)

    return torch.cat([prepare(current), prepare(previous)], dim=0)
