"""Differentiable, causal message replay. Not a live hard-real-time scheduler."""
from dataclasses import dataclass, replace
import torch
from torch import nn
from .candidates import CandidateConfig
from .encoders import ImageBackbone
from .events import PacketEncoder, voxelize
from .snn_adapter import SNNPacketEncoder
from .fusion import ContactFusion, FusionConfig
from .split_actions import SplitActionHead


@dataclass(frozen=True)
class Observation:
    kind: str
    available_ns: int
    value: object = None
    captured_ns: int | None = None
    start_us: int | None = None
    end_us: int | None = None
    capture_time_basis: str = "host_mapped_device"


@dataclass(frozen=True)
class TensorState:
    memory: torch.Tensor
    context: torch.Tensor
    robot: torch.Tensor
    executed: torch.Tensor
    time_ns: int
    revision: int = 0
    frame_ns: int | None = None
    event_ns: int | None = None
    event_state: object = None
    event_end_us: int | None = None
    def detach(self):
        def detach(value):
            if isinstance(value, torch.Tensor): return value.detach()
            if isinstance(value, tuple): return tuple(detach(item) for item in value)
            return value
        return replace(self, **{key:detach(getattr(self,key)) for key in ('memory','context','robot','executed','event_state')})


class NeuroFusionPolicy(nn.Module):
    def __init__(self, fusion=FusionConfig(), candidate=CandidateConfig(), *, load_event_checkpoint=True):
        super().__init__()
        self.fusion_config, self.candidate = fusion, candidate
        kwargs = dict(pretrained=candidate.pretrained, frozen=candidate.frozen_encoders)
        self.rgb = ImageBackbone(candidate.rgb_encoder, fusion.context_dim, **kwargs)
        self.tactile = ImageBackbone(candidate.tactile_encoder, fusion.frame_dim, **kwargs)
        if candidate.event_encoder == 'snn':
            if fusion.event_dim != 64:
                raise ValueError('Trained SNN provides 64 features')
            self.events = SNNPacketEncoder(candidate.event_checkpoint, bin_us=candidate.event_bin_us,
                frozen=candidate.frozen_event_encoder, load_checkpoint=load_event_checkpoint)
        else:
            self.events = PacketEncoder(fusion.event_dim)
        self.fusion = ContactFusion(fusion)
        self.actions = SplitActionHead(candidate, memory_dim=fusion.memory_dim,
                                      context_dim=fusion.context_dim, robot_dim=fusion.robot_dim)
        if fusion.executed_dim != 8:
            raise ValueError('Current split-action contract requires eight feedback slots')

    def initial_state(self, batch_size, time_ns=0):
        if batch_size <= 0 or type(time_ns) is not int or time_ns < 0:
            raise ValueError('Positive batch and nonnegative integer time required')
        parameter = next(self.parameters())
        zero = lambda n: parameter.new_zeros(batch_size,n)
        cfg = self.fusion_config
        return TensorState(zero(cfg.memory_dim),zero(cfg.context_dim),zero(cfg.robot_dim),zero(8),time_ns)

    def condition(self, state):
        return torch.cat([state.memory,state.context,state.robot],-1)

    def step(self, state, observation):
        if observation.capture_time_basis not in ("host_mapped_device", "host_receipt_proxy"):
            raise ValueError("Unknown capture time basis")
        now = observation.available_ns
        if type(now) is not int or now < state.time_ns:
            raise ValueError('Replay messages must have nondecreasing integer availability times')
        if observation.kind not in ('rgb','frame','event','robot','arm_executed','gripper_executed','control'):
            raise ValueError('Unknown observation kind')
        if observation.kind in ('rgb','frame','event') and observation.captured_ns is None:
            raise ValueError('Map and supply source capture/window-close time; do not invent unknown timestamps')
        captured = observation.captured_ns if observation.captured_ns is not None else now
        if type(captured) is not int or not 0 <= captured <= now:
            raise ValueError('Captured time must be mapped to host clock and cannot be in the future')
        ages = [(now-state.time_ns)/1e9, 0 if state.frame_ns is None else (now-state.frame_ns)/1e9,
                float(state.frame_ns is not None), 0 if state.event_ns is None else (now-state.event_ns)/1e9,
                float(state.event_ns is not None), (now-captured)/1e9]
        timing = state.memory.new_tensor(ages)[None].expand(state.memory.shape[0],-1)
        memory = self.fusion(state.memory,state.context,state.robot,state.executed,timing,'propagation')
        current = replace(state,memory=memory,time_ns=now,revision=state.revision+1)
        kind, value = observation.kind, observation.value
        if kind == 'rgb':
            if value.ndim != 4 or value.shape[0] != memory.shape[0] or not torch.isfinite(value).all():
                raise ValueError('Images need a matching batch and finite BCHW tensor')
            current = replace(current,context=self.rgb(value))
        elif kind in ('frame','event'):
            if kind == 'frame':
                if value.ndim != 4 or value.shape[0] != memory.shape[0] or not torch.isfinite(value).all():
                    raise ValueError('Tactile images need a matching batch and finite BCHW tensor')
                feature = self.tactile(value)
            else:
                if observation.start_us is None or observation.end_us is None:
                    raise ValueError('Event packet needs an explicit device-time window')
                if len(value) != state.memory.shape[0]:
                    raise ValueError('One raw event packet per batch member required')
                if self.candidate.event_encoder == 'snn':
                    if state.event_end_us is not None and observation.start_us != state.event_end_us:
                        raise ValueError('SNN windows must be contiguous, including empty windows; reset per episode')
                    feature, event_state = self.events(value, observation.start_us, observation.end_us, state.event_state)
                    current = replace(current, event_state=event_state, event_end_us=observation.end_us)
                else:
                    feature = self.events(voxelize(value,observation.start_us,observation.end_us,device=memory.device))
            memory = self.fusion(memory,current.context,current.robot,current.executed,timing,kind,feature)
            # A late source must not replace the timestamp of fresher evidence.
            key = 'frame_ns' if kind == 'frame' else 'event_ns'
            old = getattr(state,key)
            current = replace(current,memory=memory,**{key:captured if old is None else max(old,captured)})
        elif kind in ('robot','arm_executed','gripper_executed'):
            width = {'robot':self.fusion_config.robot_dim,'arm_executed':7,'gripper_executed':1}[kind]
            if value.shape != (memory.shape[0],width) or not torch.isfinite(value).all():
                raise ValueError('Feedback has wrong shape or nonfinite values')
            if kind == 'robot':
                current = replace(current,robot=value)
            elif kind == 'arm_executed':
                if (value[:,6] != 0).any():
                    raise ValueError('Arm feedback slot 6 is reserved, not a gripper signal')
                current = replace(current,executed=torch.cat([value,current.executed[:,7:]],-1))
            else:
                current = replace(current,executed=torch.cat([current.executed[:,:7],value],-1))
        return current

    def rollout(self, observations, state):
        """Caller owns sequence/reset; equal-time order is explicit iterable order."""
        controls = []
        for observation in observations:
            state = self.step(state,observation)
            if observation.kind == 'control':
                controls.append((state,self.condition(state)))
        return state, controls

    @torch.no_grad()
    def propose(self, state, physical=False):
        actions = self.actions.sample(self.condition(state),physical=physical)
        if actions.ndim == 3:
            actions = actions[:,0]  # No open-loop chunk execution; replan with the latest state.
        return [self.actions.proposal(action,state.time_ns,state.revision,not physical) for action in actions]
