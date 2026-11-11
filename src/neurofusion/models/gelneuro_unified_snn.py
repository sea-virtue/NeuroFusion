"""Causal Speck-shaped IAF encoder with an explicitly host-side trace readout."""
from itertools import permutations
import math

import torch
from torch import nn


@torch.jit.script
def _iaf_forward(currents: torch.Tensor, membrane: torch.Tensor, alpha: float):
    output = torch.jit.annotate(list[torch.Tensor], [])
    positive = torch.jit.annotate(list[torch.Tensor], [])
    unclipped = torch.jit.annotate(list[torch.Tensor], [])
    for current in currents.unbind(1):
        raw = membrane + current
        voltage = raw.clamp_min(-1.)
        spike = voltage.floor().clamp_min(0.)
        membrane = voltage - spike
        output.append(alpha * spike + (1. - alpha) * current.relu())
        positive.append(voltage > 0.)
        unclipped.append(raw > -1.)
    return torch.stack(output,1),membrane,torch.stack(positive,1),torch.stack(unclipped,1)


@torch.jit.script
def _iaf_backward(grad: torch.Tensor, end: torch.Tensor, positive: torch.Tensor,
                  unclipped: torch.Tensor, current_positive: torch.Tensor, alpha: float):
    result = torch.empty_like(grad)
    carried = end
    for t in range(grad.size(1)-1,-1,-1):
        g=positive[:,t].to(grad.dtype)
        voltage_grad=(alpha*grad[:,t]*g+carried*(1.-g))*unclipped[:,t]
        result[:,t]=voltage_grad+(1.-alpha)*grad[:,t]*current_positive[:,t]
        carried=voltage_grad
    return result,carried


class _Integrate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,currents,membrane,alpha):
        output,end,positive,unclipped=_iaf_forward(currents,membrane,alpha)
        ctx.save_for_backward(positive,unclipped,currents>0.)
        ctx.alpha=alpha
        return output,end
    @staticmethod
    def backward(ctx,grad,end):
        positive,unclipped,current_positive=ctx.saved_tensors
        dx,dm=_iaf_backward(grad,end,positive,unclipped,current_positive,ctx.alpha)
        return dx,dm,None


@torch.jit.script
def _trace_forward(spikes: torch.Tensor, trace: torch.Tensor, decay: torch.Tensor):
    history = torch.jit.annotate(list[torch.Tensor], [])
    for current in spikes.unbind(1):
        trace=trace*decay+current.unsqueeze(1)*(1.-decay)
        history.append(trace)
    all_trace=torch.stack(history,1)
    return all_trace.log1p().flatten(2),trace,all_trace


@torch.jit.script
def _trace_backward(grad: torch.Tensor, end: torch.Tensor, history: torch.Tensor, decay: torch.Tensor):
    local=grad.reshape_as(history)/(1.+history)
    carried=end
    result=history.new_empty((history.size(0),history.size(1),history.size(3)))
    for t in range(history.size(1)-1,-1,-1):
        current=local[:,t]+carried
        result[:,t]=(current*(1.-decay)).sum(1)
        carried=current*decay
    return result,carried


class _Trace(torch.autograd.Function):
    @staticmethod
    def forward(ctx,spikes,trace,decay):
        out,end,history=_trace_forward(spikes,trace,decay)
        ctx.save_for_backward(history,decay)
        return out,end
    @staticmethod
    def backward(ctx,grad,end):
        history,decay=ctx.saved_tensors
        dx,dt=_trace_backward(grad,end,history,decay)
        return dx,dt,None


def integrate(currents,membrane,alpha):
    return _Integrate.apply(currents,membrane,alpha)


def trace_readout(spikes,trace,decay):
    return _Trace.apply(spikes,trace,decay)


class GelNeuroUnifiedSNN(nn.Module):
    """Four Conv/IAF cores, 16 monitored outputs, 4 host traces -> 64 features.

    Input is raw nonnegative event counts in B,T,2,32,32; native DVS pooling
    is 4x4 sum pooling. No log operation or GRU is part of the chip backbone.
    alpha<1 is training-only continuation; deployment/inference uses alpha=1.
    State is (four membrane tensors, host trace tensor).
    """
    def __init__(self, bin_us=50_000):
        super().__init__()
        self.bin_us = bin_us
        self.input_gain = .25
        self.rate_scale = 50_000. / bin_us
        self.convolutions = nn.ModuleList([
            nn.Conv2d(2, 16, 3, 2, 1, bias=False),
            nn.Conv2d(16, 32, 3, 2, 1, bias=False),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.Conv2d(64, 16, 4, bias=False),
        ])
        for layer in self.convolutions:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        self.head = nn.Linear(64, 8)
        taus = torch.tensor([50.,150.,500.,1500.])
        self.register_buffer("decay", (-float(bin_us)/1000./taus).exp().reshape(1,4,1))
        self.last_activity = None

    @classmethod
    def from_checkpoint(cls, checkpoint, bin_us=None):
        """Load a training dt or an explicit replay dt without overwriting its decay."""
        dt=bin_us or checkpoint.get("native_feature_step_us",checkpoint["bin_us"])
        if dt<=0:
            raise ValueError("bin_us must be positive")
        model=cls(dt)
        model.load_state_dict(checkpoint["model"])
        model.input_gain=checkpoint.get("input_gain",.25)
        model.decay.copy_((-dt/1000./torch.tensor([50.,150.,500.,1500.])).exp().reshape(1,4,1))
        return model

    def encode(self, counts, state=None, alpha=1.):
        if counts.ndim != 5 or counts.shape[2:] != (2,32,32) or not counts.shape[1]:
            raise ValueError("Expected B,T,2,32,32 nonnegative event counts")
        batch, time = counts.shape[:2]
        values = counts * self.input_gain
        membranes, rates = [], []
        for i, conv in enumerate(self.convolutions):
            current = conv(values.flatten(0,1)).unflatten(0,(batch,time))
            mem = torch.zeros_like(current[:,0]) if state is None else state[0][i]
            values, mem = integrate(current, mem, float(alpha))
            membranes.append(mem)
            rates.append(values.mean())
        spikes = values.flatten(2)
        self.last_output_spikes = spikes.detach()
        trace = spikes.new_zeros(batch,4,16) if state is None else state[1]
        features, trace = trace_readout(spikes*self.rate_scale,trace,self.decay)
        self.last_activity = torch.stack(rates)
        return features,(tuple(membranes),trace)

    def forward(self, counts, state=None, alpha=1.):
        z, state = self.encode(counts,state,alpha)
        logits = self.head(z)
        return logits[...,:4],logits[...,4:],state

    def quantized_copy(self):
        """Joint 8-bit weights/16-bit thresholds; no additional quantization training."""
        result = GelNeuroUnifiedSNN(self.bin_us).to(next(self.parameters()).device)
        result.load_state_dict(self.state_dict())
        result.input_gain = 1.
        layers = []
        for i, conv in enumerate(result.convolutions):
            weight = conv.weight.detach().clone()
            if i == 0:
                weight *= self.input_gain  # Fold fixed input gain into chip weights.
            scale = min(127./max(float(weight.abs().max()),1e-9),32767.)
            threshold = max(1,round(scale))
            integer = (weight*scale).round().clamp(-128,127)
            conv.weight.data.copy_(integer/threshold)
            layers.append(dict(layer=i,threshold_high=threshold,threshold_low=-threshold,
                               weight_min=int(integer.min()),weight_max=int(integer.max()),
                               zero_fraction=float(integer.eq(0).float().mean()),
                               weights=integer.to(torch.int8).cpu()))
        return result,layers

    def resource_estimate(self):
        width, layers = 32, []
        rounded = lambda n: 1 << (n-1).bit_length()
        for conv in self.convolutions:
            width=(width+2*conv.padding[0]-conv.kernel_size[0])//conv.stride[0]+1
            layers.append(dict(shape=[conv.out_channels,width,width],
                kernel_entries=conv.in_channels*rounded(conv.out_channels)*rounded(math.prod(conv.kernel_size)),
                neuron_entries=conv.out_channels*rounded(width)**2))
        budgets=[(16,64)]*3+[(32,32)]*2+[(64,16)]*2+[(16,16)]*2
        cores=next((p for p in permutations(range(9),len(layers)) if all(
            row["kernel_entries"]<=budgets[c][0]*1024 and
            row["neuron_entries"]<=budgets[c][1]*1024
            for row,c in zip(layers,p))),None)
        return dict(layers=layers,candidate_cores=cores,chip_parameters=sum(p.numel() for p in self.convolutions.parameters()),
                    host_parameters=sum(p.numel() for p in self.head.parameters()),
                    dvs_pool=[4,4],output_features=16,
                    scope="Static budgets; SDK mapping and asynchronous hardware replay require separate checks.")
