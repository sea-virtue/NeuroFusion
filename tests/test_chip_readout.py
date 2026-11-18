from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from neurofusion.control.chip_readout import ChipReadout
from neurofusion.models.gelneuro_unified_snn import trace_readout


class ChipReadoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'head.pt'
        self.weight = torch.arange(8*64,dtype=torch.float32).reshape(8,64)/3000
        self.bias = torch.arange(8,dtype=torch.float32)/100
        torch.save(dict(classes=('quiet','slide','rotate','escape_wriggle'),
                        directions=('up','down','left','right'),bin_us=50000,native_feature_step_us=1000,
                        model={'head.weight':self.weight,'head.bias':self.bias}),self.path)

    def test_chip_counts_match_original_host_trace_and_head_including_empty_bins(self):
        readout = ChipReadout(self.path,expected_direction='up')
        counts = torch.zeros(1,5,16)
        counts[0,0,2]=2
        counts[0,2,11]=3
        counts[0,4,0]=1
        decay = torch.exp(-1/torch.tensor([50.,150.,500.,1500.])).reshape(1,4,1)
        features,trace = trace_readout(counts*50,torch.zeros(1,4,16),decay)
        for t in range(5):
            spikes = []
            for channel in range(16):
                spikes.extend([[channel,0,t*1000+10,0]]*int(counts[0,t,channel]))
            pred = readout.predict(spikes,t*1000,(t+1)*1000,123)
            np.testing.assert_allclose(readout.z,features[0,t].numpy(),rtol=1e-5,atol=1e-6)
            logits = self.weight@features[0,t]+self.bias
            self.assertAlmostEqual(pred.slide_probability,float(logits[:4].softmax(0)[1]),places=6)
            self.assertEqual(pred.captured_ns,123)
        self.assertEqual(readout.sequence,5)
        self.assertTrue(readout.model_id.startswith('chip-sha256:'))

    def test_multiple_native_bins_match_single_bin_steps(self):
        a,b = [ChipReadout(self.path,expected_direction='up') for _ in range(2)]
        events = [[1,0,10,0],[2,0,2100,0]]
        a.predict(events,0,4000,123)
        for t in range(4):
            b.predict([e for e in events if t*1000<=e[2]<(t+1)*1000],t*1000,(t+1)*1000,123)
        np.testing.assert_allclose(a.z,b.z)

    def test_old_or_outside_spikes_cannot_be_relabelled_as_current(self):
        r = ChipReadout(self.path,expected_direction='up')
        with self.assertRaisesRegex(ValueError,'outside'):
            r.predict([[0,0,1000,0]],0,1000,123)
        r.predict([],0,1000,123)
        with self.assertRaisesRegex(ValueError,'contiguous'):
            r.predict([],2000,3000,123)
        r.reset()
        self.assertEqual(r.sequence,0)
        self.assertFalse(r.z.any())
