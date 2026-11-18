"""Original host EMA/log1p/head applied to real chip output, no CPU CNN."""
import hashlib
from pathlib import Path
import time

import numpy as np

from .slip import SlipPrediction


class ChipReadout:
    def __init__(self, checkpoint, *, expected_direction):
        import torch
        saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.tasks,self.directions = tuple(saved['classes']),tuple(saved['directions'])
        if (set(self.tasks) != {'quiet','slide','rotate','escape_wriggle'}
                or set(self.directions) != {'up','down','left','right'}
                or expected_direction not in self.directions):
            raise ValueError('Checkpoint labels do not match slip controller')
        self.bin_us = saved.get('native_feature_step_us',saved['bin_us'])
        self.weight = saved['model']['head.weight'].numpy().copy()
        self.bias = saved['model']['head.bias'].numpy().copy()
        self.decay = (-float(self.bin_us)/1000./torch.tensor([50.,150.,500.,1500.])).exp().numpy()[:,None]
        self.rate_scale = 50000./self.bin_us
        self.model_id = 'chip-sha256:'+hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self.reset()

    def reset(self):
        self.trace = np.zeros((4,16),dtype=np.float32)
        self.z = np.zeros(64,dtype=np.float32)
        self.end_us = None
        self.sequence = 0

    def predict(self, spikes, start_us, end_us, captured_host_ns):
        if end_us <= start_us or (end_us-start_us)%self.bin_us:
            raise ValueError('Readout requires whole native bins')
        if self.end_us is not None and start_us != self.end_us:
            raise ValueError('Readout windows must be contiguous')
        spikes = np.asarray(spikes,dtype=np.int64).reshape(-1,4)
        steps = (end_us-start_us)//self.bin_us
        counts = np.zeros((steps,16),dtype=np.float32)
        if len(spikes):
            features,times = spikes[:,0],spikes[:,2]
            if ((features<0)|(features>=16)|(times<start_us)|(times>=end_us)).any():
                raise ValueError('Spike outside readout window or feature bounds')
            np.add.at(counts,((times-start_us)//self.bin_us,features),1.)
        for count in counts:
            self.trace = self.trace*self.decay+count[None,:]*self.rate_scale*(1-self.decay)
        self.z = np.log1p(self.trace).reshape(64)
        logits = self.weight@self.z+self.bias
        def softmax(values):
            exp=np.exp(values-values.max())
            return exp/exp.sum()
        tp,dp = softmax(logits[:4]),softmax(logits[4:])
        ti,di = int(tp.argmax()),int(dp.argmax())
        self.end_us,self.sequence = end_us,self.sequence+1
        return SlipPrediction(self.sequence,time.monotonic_ns(),captured_host_ns,
                              self.tasks[ti],self.directions[di],float(tp[ti]),float(dp[di]),self.model_id,
                              slide_probability=float(tp[self.tasks.index('slide')]))
