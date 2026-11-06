"""Trainable recurrent contact memory, with three controlled fusion variants."""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class FusionConfig:
    kind: str = 'attention'
    memory_dim: int = 256
    context_dim: int = 128
    frame_dim: int = 128
    event_dim: int = 64
    robot_dim: int = 16
    executed_dim: int = 8
    tokens: int = 4
    heads: int = 4

    def __post_init__(self):
        if self.kind not in ('plain', 'gated', 'attention'):
            raise ValueError('Fusion kind must be plain, gated or attention')
        if any(type(v) is not int or v <= 0 for k, v in vars(self).items() if k != 'kind'):
            raise ValueError('Dimensions must be positive integers')
        if self.memory_dim % self.tokens or (self.memory_dim // self.tokens) % self.heads:
            raise ValueError('Memory must divide into tokens and attention heads')


class ContactFusion(nn.Module):
    """frame/event/propagation are functional roles sharing one update network.

    time_features: elapsed, old frame age/valid, old event age/valid, new source age.
    All time values are seconds; no future source times are accepted by pipeline.
    """
    def __init__(self, config=FusionConfig()):
        super().__init__()
        self.config = config
        width = config.memory_dim // config.tokens if config.kind == 'attention' else 128
        self.width = width
        self.frame = nn.Linear(config.frame_dim, width)
        self.event = nn.Linear(config.event_dim, width)
        self.context = nn.Linear(config.context_dim, width)
        self.robot_action = nn.Linear(config.robot_dim + config.executed_dim, width)
        self.time = nn.Sequential(nn.Linear(6, width), nn.SiLU(), nn.Linear(width, width))
        self.kind = nn.Embedding(3, width)
        if config.kind == 'attention':
            self.slots = nn.Parameter(torch.randn(1, config.tokens, width) * .02)
            self.modality = nn.Parameter(torch.randn(1, 4, width) * .02)
            self.attention = nn.MultiheadAttention(width, config.heads, dropout=0., batch_first=True)
            self.norm1 = nn.LayerNorm(width)
            self.ffn = nn.Sequential(nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width))
            self.norm2 = nn.LayerNorm(width)
        else:
            input_dim = config.memory_dim + width*4
            self.candidate = nn.Sequential(nn.Linear(input_dim, 256), nn.SiLU(), nn.Linear(256, config.memory_dim))
            if config.kind == 'gated':
                self.gate = nn.Linear(input_dim, config.memory_dim)

    def forward(self, memory, context, robot, executed, time_features, kind, feature=None):
        if kind not in ('frame', 'event', 'propagation'):
            raise ValueError(kind)
        identity = {'frame': 0, 'event': 1, 'propagation': 2}[kind]
        observed = torch.zeros(memory.shape[0], self.width, device=memory.device, dtype=memory.dtype)
        if kind != 'propagation':
            if feature is None:
                raise ValueError('Observation update needs encoded features')
            observed = (self.frame if kind == 'frame' else self.event)(feature)
        observed = observed + self.kind.weight[identity]
        parts = [observed, self.context(context), self.robot_action(torch.cat([robot, executed], -1)),
                 self.time(torch.log1p(time_features))]
        if self.config.kind == 'attention':
            old = memory.reshape(-1, self.config.tokens, self.width)
            query = old + self.slots
            inputs = torch.stack(parts, dim=1) + self.modality
            keys = torch.cat([query, inputs], dim=1)
            read, _ = self.attention(query, keys, keys, need_weights=False)
            output = self.norm1(old + read)
            output = self.norm2(output + self.ffn(output)).flatten(1)
        else:
            inputs = torch.cat([memory, *parts], dim=-1)
            candidate = self.candidate(inputs)
            output = candidate if self.config.kind == 'plain' else memory + self.gate(inputs).sigmoid() * (candidate-memory)
        if kind == 'propagation':
            # No double propagation for messages with the same availability time.
            output = torch.where(time_features[:, :1] > 0, output, memory)
        return output
