"""Explicit placeholders: no random weights or pretend learned behavior."""


class SceneImageEncoder:
    def encode(self, sample):
        raise NotImplementedError('Scene RGB encoder and task-context caching require implementation')


class TactileImageEncoder:
    def encode(self, sample):
        raise NotImplementedError('GelSight image encoder requires implementation')


class PacketEventEncoder:
    def encode(self, packet):
        raise NotImplementedError('GelNeuro event representation and encoder require implementation')


class FrameCorrector:
    def correct(self, memory, feature, context):
        raise NotImplementedError('Learned frame correction requires implementation')


class GatedEventUpdater:
    def update(self, memory, feature, context, robot, timing):
        raise NotImplementedError('Learned event update and gate require implementation')


class ActionTimePropagator:
    def propagate(self, memory, robot, executed_action, timing):
        raise NotImplementedError('Learned action/time propagation requires implementation')


class RegressionActionHead:
    def predict(self, state):
        raise NotImplementedError('Candidate concat(h,c,q) -> MLP -> 5D action; train and validate first')


class FutureTactileAuxiliaryHead:
    def predict(self, state):
        raise NotImplementedError('Optional training-only tactile prediction; never a controller')
