"""Inference-only bridges to the existing tuple-based reference runtime.

Training uses tensors and forward()/loss() directly, never these tuple bridges.
"""
import torch
from ..config import Config, vector
from .encoders import prepare_sparsh


class ImageEncoderBridge:
    def __init__(self, model, source):
        self.model = model.eval()
        self.source = source

    @torch.inference_mode()
    def encode(self, sample):
        if sample.source != self.source:
            raise ValueError(f'Expected image source {self.source}')
        if self.model.name == 'sparsh_dino_small':
            if not isinstance(sample.image, dict):
                raise ValueError('Sparsh image must contain current, previous and background')
            image = prepare_sparsh(**sample.image)
        else:
            from torchvision.transforms.functional import to_tensor
            image = to_tensor(sample.image)
        device = next(self.model.parameters()).device
        return tuple(self.model(image[None].to(device))[0].cpu().tolist())


class ActionHeadBridge:
    def __init__(self, model, action_mean, action_scale, dimensions=Config()):
        self.model = model.eval()
        self.dimensions = dimensions
        self.mean = vector(action_mean, dimensions.action_dim, 'action_mean')
        self.scale = vector(action_scale, dimensions.action_dim, 'action_scale')
        if any(value <= 0 for value in self.scale):
            raise ValueError('Action scales must be positive and fitted from training data')

    @torch.inference_mode()
    def predict(self, state):
        memory = vector(state.memory, self.dimensions.memory_dim, 'memory')
        context = vector(state.context, self.dimensions.context_dim, 'context')
        robot = vector(state.robot, self.dimensions.robot_dim, 'robot')
        device = next(self.model.parameters()).device
        condition = torch.tensor([memory + context + robot], device=device)
        if hasattr(self.model, 'sample'):
            values = self.model.sample(condition)[0, 0]  # Replan at every control tick with latest h.
        else:
            values = self.model(condition)[0]
        normalized = values.cpu().tolist()
        return tuple(v * scale + mean for v, scale, mean in zip(normalized, self.scale, self.mean))
