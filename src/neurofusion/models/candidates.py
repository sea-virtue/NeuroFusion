"""Configuration factory; importing this module alone does not require PyTorch."""
from dataclasses import dataclass
from pathlib import Path
import tomllib
from ..config import Config


@dataclass(frozen=True)
class CandidateConfig:
    rgb_encoder: str = 'resnet18'
    tactile_encoder: str = 'sparsh_dino_small'
    action_head: str = 'mlp'
    frozen_encoders: bool = True
    pretrained: bool = True
    horizon: int = 16
    inference_steps: int = 10
    event_encoder: str = 'cnn'
    event_checkpoint: str | None = None
    frozen_event_encoder: bool = True
    event_bin_us: int = 1000

    def __post_init__(self):
        if self.event_encoder not in ('cnn', 'snn'):
            raise ValueError('event_encoder must be cnn or snn')
        if type(self.frozen_event_encoder) is not bool or type(self.event_bin_us) is not int or self.event_bin_us <= 0:
            raise ValueError('Event freeze flag/time step invalid')
        if self.rgb_encoder not in ('resnet18', 'dinov2_vits14'):
            raise ValueError('RGB candidate must be resnet18 or dinov2_vits14')
        if self.tactile_encoder not in ('resnet18', 'sparsh_dino_small'):
            raise ValueError('Tactile candidate must be resnet18 or sparsh_dino_small')
        if self.action_head not in ('mlp', 'act', 'diffusion_unet', 'diffusion_transformer'):
            raise ValueError('Unknown action head')
        if type(self.horizon) is not int or self.horizon < 4 or self.horizon % 4:
            raise ValueError('horizon must be a positive multiple of four')
        if type(self.inference_steps) is not int or not 1 <= self.inference_steps <= 100:
            raise ValueError('inference_steps must be between 1 and 100')
        if type(self.pretrained) is not bool or type(self.frozen_encoders) is not bool:
            raise TypeError('pretrained and frozen_encoders must be booleans')

    @classmethod
    def load(cls, path):
        with Path(path).open('rb') as stream:
            return cls(**tomllib.load(stream))


def build_encoders(candidate=CandidateConfig(), dimensions=Config()):
    from .encoders import ImageBackbone
    kwargs = dict(pretrained=candidate.pretrained, frozen=candidate.frozen_encoders)
    return (ImageBackbone(candidate.rgb_encoder, dimensions.context_dim, **kwargs),
            ImageBackbone(candidate.tactile_encoder, dimensions.frame_feature_dim, **kwargs))


def build_action_head(candidate=CandidateConfig(), dimensions=Config()):
    from .action_heads import MLPActionHead, DiffusionActionHead, ACTActionHead
    condition_dim = dimensions.memory_dim + dimensions.context_dim + dimensions.robot_dim
    if candidate.action_head == 'mlp':
        return MLPActionHead(condition_dim, dimensions.action_dim)
    if candidate.action_head == 'act':
        return ACTActionHead(condition_dim, dimensions.robot_dim, dimensions.action_dim, candidate.horizon)
    return DiffusionActionHead(candidate.action_head, condition_dim, dimensions.action_dim,
                               candidate.horizon, candidate.inference_steps)
