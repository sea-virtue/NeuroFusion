"""Recognition adapter retaining the trained head, unlike SNNPacketEncoder."""
from pathlib import Path
import hashlib
import time
from .slip import SlipPrediction


class SNNRecognizer:
    def __init__(self, checkpoint, device='cpu', *, expected_direction, fast_empty=True):
        import torch
        from ..models.gelneuro_unified_snn import GelNeuroUnifiedSNN
        self.torch,self.device = torch,device
        saved = torch.load(Path(checkpoint),map_location='cpu',weights_only=True)
        self.tasks = tuple(saved['classes'])
        self.directions = tuple(saved['directions'])
        if set(self.tasks) != {'quiet','slide','rotate','escape_wriggle'} or set(self.directions) != {'up','down','left','right'}:
            raise ValueError('Checkpoint labels do not match the slip teacher contract')
        if expected_direction not in self.directions: raise ValueError('Unknown physical slide direction mapping')
        self.bin_us = saved.get('native_feature_step_us',saved['bin_us'])
        self.model = GelNeuroUnifiedSNN.from_checkpoint(saved,self.bin_us).to(device).eval()
        self.model_id = 'sha256:'+hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self.state,self.end_us,self.sequence = None,None,0
        self.fast_empty = fast_empty

    def reset(self):
        self.state,self.end_us,self.sequence = None,None,0

    def predict(self, events, start_us, end_us, captured_host_ns):
        """captured_host_ns is caller-measured host window-close/receipt time.

        Preserve the source-time uncertainty in the episode metadata. No device
        microsecond to host nanosecond conversion is guessed here.
        """
        from ..models.snn_adapter import count_bins
        if self.end_us is not None and start_us != self.end_us:
            raise ValueError('Recognition windows must be contiguous; reset per episode')
        events = self.torch.as_tensor(events,device=self.device)
        if events.numel() == 0: events = events.reshape(0,4)
        counts = count_bins([events],start_us,end_us,self.bin_us,self.device)
        with self.torch.inference_mode():
            if self.fast_empty and events.numel() == 0:
                # These Conv/IAF cores have no bias and no membrane leak.
                # With zero input, every membrane stays in [-1,1), no core
                # spikes, and only the host trace decays. This advances every
                # native bin without running redundant zero convolutions.
                state = self.state
                steps = counts.shape[1]
                if state is None:
                    _,_,state = self.model(counts[:,:1],None,alpha=1.)
                    steps -= 1
                state = (state[0],state[1]*self.model.decay.pow(steps))
                logits = self.model.head(state[1].log1p().flatten(1))
                task,direction = logits[:,:4].unsqueeze(1),logits[:,4:].unsqueeze(1)
                self.model.last_activity = counts.new_zeros(4)
                self.model.last_output_spikes = counts.new_zeros(1,counts.shape[1],16)
            else:
                task,direction,state = self.model(counts,self.state,alpha=1.)
            tp,dp = task[0,-1].softmax(-1),direction[0,-1].softmax(-1)
            ti,di = int(tp.argmax()),int(dp.argmax())
        self.state,self.end_us = state,end_us
        self.sequence += 1
        return SlipPrediction(self.sequence,time.monotonic_ns(),captured_host_ns,
                              self.tasks[ti],self.directions[di],float(tp[ti]),float(dp[di]),self.model_id,
                              slide_probability=float(tp[self.tasks.index('slide')]))


def main():
    """JSONL event-window bridge; hardware ownership stays with the raw collector."""
    import argparse
    from dataclasses import asdict
    import json
    import sys
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--expected-direction',required=True,choices=('up','down','left','right'))
    args = parser.parse_args()
    recognizer = SNNRecognizer(args.checkpoint,expected_direction=args.expected_direction)
    for line in sys.stdin:
        window = json.loads(line)
        prediction = recognizer.predict(window['events'],window['start_us'],window['end_us'],
                                        window['captured_host_ns'])
        print(json.dumps(asdict(prediction)),flush=True)


if __name__ == '__main__': main()
