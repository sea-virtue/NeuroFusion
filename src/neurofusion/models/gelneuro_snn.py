"""Speck-shaped GelNeuro backbone and causal host readout."""
import torch
from torch import nn


class _Spikes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage, threshold):
        ctx.save_for_backward(voltage / threshold - 1)
        return (voltage / threshold).floor().clamp_min(0)

    @staticmethod
    def backward(ctx, grad_output):
        distance, = ctx.saved_tensors
        slope = 0.3 * (1 + distance.abs()).square().reciprocal()
        return grad_output * slope, None


class GelNeuroSNN(nn.Module):
    """Simulate native ON/OFF events as nonnegative counts, in B,T,2,120,120."""
    roi = (4, 4, 124, 124)
    output_shape = (4, 2, 2)
    threshold = 0.15
    min_membrane = -1.0

    def __init__(self):
        super().__init__()
        self.convolutions = nn.ModuleList([
            nn.Conv2d(2, 8, 3, 2, 1, bias=False),
            nn.Conv2d(8, 16, 3, 2, 1, bias=False),
            nn.Conv2d(16, 24, 3, 2, 1, bias=False),
            nn.Conv2d(24, 32, 3, 2, 1, bias=False),
            nn.Conv2d(32, 4, 4, 4, bias=False),
        ])

    def forward(self, counts, state=None):
        if counts.ndim != 5 or counts.shape[2:] != (2, 120, 120) or not counts.shape[1]:
            raise ValueError("Expected B,T,2,120,120 with at least one time bin")
        if state is not None and len(state) != len(self.convolutions):
            raise ValueError("Expected one membrane tensor per convolution")
        batch, time = counts.shape[:2]
        spikes, next_state = counts, []
        for index, convolution in enumerate(self.convolutions):
            currents = convolution(spikes.flatten(0, 1)).unflatten(0, (batch, time)).float()
            membrane = torch.zeros_like(currents[:, 0]) if state is None else state[index]
            if membrane.shape != currents[:, 0].shape:
                raise ValueError("Membrane shape differs from current batch")
            output = []
            for current in currents.unbind(1):
                voltage = (membrane + current).clamp_min(self.min_membrane)
                spike = _Spikes.apply(voltage, self.threshold)
                membrane = voltage - self.threshold * spike.detach()
                output.append(spike)
            spikes = torch.stack(output, dim=1)
            next_state.append(membrane)
        return spikes, tuple(next_state)


class SpikeReadout(nn.Module):
    """Read the last 16 ms as two ordered 8 ms summaries, then project to 64."""
    history_us = 16_000

    def __init__(self, bin_us):
        super().__init__()
        if bin_us < 1 or 8_000 % bin_us:
            raise ValueError("bin_us must evenly divide 8000")
        self.half_steps = 8_000 // bin_us
        self.projection = nn.Linear(32, 64)

    def forward(self, spikes):
        steps = self.half_steps
        if spikes.shape[1] < 2 * steps or spikes.shape[2:] != (4, 2, 2):
            raise ValueError("Expected at least 16 ms of B,T,4,2,2 output spikes")
        history = spikes[:, -2 * steps:].reshape(len(spikes), 2, steps, 16)
        return self.projection(history.sum(dim=2).log1p().flatten(1))
