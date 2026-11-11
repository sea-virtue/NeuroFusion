"""Stateful raw-event adapter for the trained host SNN; no chip execution claim."""
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn
from .gelneuro_unified_snn import GelNeuroUnifiedSNN
from .upstream import project_root


def count_bins(packets, start_us, end_us, bin_us, device):
    if not packets or end_us <= start_us or (end_us-start_us) % bin_us:
        raise ValueError('SNN windows must contain whole positive time bins')
    steps = (end_us-start_us)//bin_us
    counts = torch.zeros(len(packets), steps, 2, 32, 32, device=device)
    for i, raw in enumerate(packets):
        raw = torch.as_tensor(raw, device=device)
        if raw.ndim != 2 or raw.shape[1] != 4 or not torch.isfinite(raw).all():
            raise ValueError('Expected finite Nx4 events: x,y,device_us,polarity')
        if not len(raw): continue
        x,y,t,p = raw.unbind(-1)
        if ((x<0)|(x>=128)|(y<0)|(y>=128)|(t<start_us)|(t>=end_us)|
            ((p!=0)&(p!=1))|(raw!=raw.floor()).any(-1)).any() or (t[1:]<t[:-1]).any():
            raise ValueError('Events must be ordered integers inside sensor and half-open window')
        index = ((t.long()-start_us)//bin_us)*2048+p.long()*1024+(y.long()//4)*32+x.long()//4
        counts[i].view(-1).scatter_add_(0,index,torch.ones(len(raw),device=device))
    return counts


class SNNPacketEncoder(nn.Module):
    def __init__(self, checkpoint=None, *, bin_us=1000, frozen=True, load_checkpoint=True):
        super().__init__()
        self.frozen, self.bin_us = frozen, bin_us
        if load_checkpoint:
            if not checkpoint: raise ValueError('SNN requires an explicit trained checkpoint')
            path=Path(checkpoint)
            if not path.is_absolute(): path=project_root()/path
            saved=torch.load(path,map_location='cpu',weights_only=True)
            native=saved.get('native_feature_step_us',saved['bin_us'])
            if bin_us != native:
                raise ValueError(f'Checkpoint requires {native} us feature steps; explicit retraining/calibration needed')
            self.encoder=GelNeuroUnifiedSNN.from_checkpoint(saved,bin_us)
        else:
            self.encoder=GelNeuroUnifiedSNN(bin_us)
        # The recognition head is not part of the contact-memory pathway.
        self.encoder.head=nn.Identity()
        self.register_buffer('input_gain',torch.tensor(float(self.encoder.input_gain)))
        self.encoder.requires_grad_(not frozen)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen: self.encoder.eval()
        return self

    def forward(self, packets, start_us, end_us, state=None):
        device=next(self.encoder.parameters()).device
        counts=count_bins(packets,start_us,end_us,self.bin_us,device)
        self.encoder.input_gain=float(self.input_gain)
        with torch.no_grad() if self.frozen else nullcontext():
            features,state=self.encoder.encode(counts,state,alpha=1.)
        return features[:,-1],state
