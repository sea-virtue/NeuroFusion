"""Independent, one-dimensional teacher controllers (no robot-arm dependency)."""
from .slip import SlipConfig, SlipController, SlipPrediction, GripperFeedback

__all__ = ['SlipConfig', 'SlipController', 'SlipPrediction', 'GripperFeedback']
