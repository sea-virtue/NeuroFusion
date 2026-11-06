"""Unified learned policy; separate Franka and ROBOTIS output destinations."""
from dataclasses import dataclass
import torch
from torch import nn
from ..config import Config
from .candidates import build_action_head, CandidateConfig

ACTION_LAYOUT = ('arm_motion_0', 'arm_motion_1', 'arm_motion_2', 'arm_motion_3',
                 'arm_motion_4', 'arm_motion_5', 'arm_reserved_zero', 'robotis_gripper')
ACTIVE = (0,1,2,3,4,5,7)


def pack_actions(active):
    if active.shape[-1] != 7:
        raise ValueError('Six arm components and one independent gripper component required')
    return torch.cat([active[..., :6], torch.zeros_like(active[..., :1]), active[..., 6:]], -1)


@dataclass(frozen=True)
class DeviceProposal:
    time_ns: int
    revision: int
    franka: tuple
    robotis: tuple
    normalized: bool
    # No SDK calls and no assumption that the two devices execute atomically.


class SplitActionHead(nn.Module):
    def __init__(self, candidate=CandidateConfig(), *, memory_dim=256, context_dim=128, robot_dim=16):
        super().__init__()
        dimensions = Config(memory_dim=memory_dim, context_dim=context_dim, robot_dim=robot_dim, action_dim=7)
        self.head = build_action_head(candidate, dimensions)
        self.candidate = candidate
        self.register_buffer('center', torch.zeros(7))
        self.register_buffer('scale', torch.ones(7))
        self.register_buffer('calibrated', torch.tensor(False))

    @torch.no_grad()
    def fit_normalizer(self, training_actions):
        active = training_actions[..., ACTIVE].reshape(-1,7)
        if not active.numel() or not torch.isfinite(active).all():
            raise ValueError('Finite, nonempty training actions required')
        low, high = active.amin(0), active.amax(0)
        self.center.copy_((high+low)/2)
        self.scale.copy_(((high-low)/2).clamp_min(1e-6))
        self.calibrated.fill_(True)

    def normalize(self, actions):
        if actions.shape[-1] != 8 or not bool(self.calibrated):
            raise ValueError('Fit training-set normalization before using physical action labels')
        active = actions[..., ACTIVE]
        if not torch.isfinite(active).all():
            raise ValueError('Active labels must be finite')
        return (active-self.center)/self.scale

    def loss(self, condition, actions):
        return self.head.loss(condition, self.normalize(actions))

    def forward(self, condition):
        if self.candidate.action_head != 'mlp':
            raise RuntimeError('Use sample() for ACT/diffusion; loss() for their training')
        return pack_actions(self.head(condition))

    @torch.no_grad()
    def sample(self, condition, *, physical=False, generator=None):
        if self.training:
            raise RuntimeError('Use eval() for inference')
        active = self.head.sample(condition, generator=generator) if hasattr(self.head,'sample') else self.head(condition)
        if physical:
            if not bool(self.calibrated):
                raise RuntimeError('Physical units require training-set calibration')
            active = active*self.scale+self.center
        return pack_actions(active)

    @staticmethod
    def proposal(actions, time_ns, revision, normalized=True):
        if actions.shape != (8,) or not torch.isfinite(actions).all() or actions[6].item() != 0:
            raise ValueError('Expected a finite eight-slot action with reserved zero')
        values = actions.detach().cpu().tolist()
        return DeviceProposal(time_ns, revision, tuple(values[:7]), (values[7],), normalized)
