import importlib.util
from pathlib import Path
import time
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch required')
class EmptyAdvanceTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2)
        from neurofusion.control.snn import SNNRecognizer
        path=Path(__file__).resolve().parents[1]/'artifacts/checkpoints/gelneuro-snn-20261111/final/all-data.pt'
        self.fast=SNNRecognizer(path,expected_direction='up')
        self.reference=SNNRecognizer(path,expected_direction='up',fast_empty=False)

    def check_window(self,events,start,end):
        import torch
        a=self.fast.predict(events,start,end,time.monotonic_ns())
        b=self.reference.predict(events,start,end,time.monotonic_ns())
        self.assertEqual(a.task,b.task)
        self.assertEqual(a.direction,b.direction)
        self.assertAlmostEqual(a.task_confidence,b.task_confidence,places=5)
        self.assertAlmostEqual(a.direction_confidence,b.direction_confidence,places=5)
        for x,y in zip(self.fast.state[0],self.reference.state[0]):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
        torch.testing.assert_close(self.fast.state[1],self.reference.state[1],rtol=1e-4,atol=1e-6)

    def test_empty_initial_state_and_single_native_bin(self):
        self.check_window([],0,1000)
        self.check_window([],1000,21000)

    def test_long_empty_gap_after_real_activity_preserves_state_and_next_prediction(self):
        events=[[(i*13)%128,(i*29)%128,(i//8)*20,i%2] for i in range(4000)]
        self.check_window(events,0,20000)
        self.check_window([],20000,1020000)
        self.check_window([[64,64,1020001,1],[63,64,1039999,0]],1020000,1040000)

    def test_empty_fast_path_still_rejects_non_native_or_noncontiguous_windows(self):
        with self.assertRaises(ValueError):self.fast.predict([],0,1500,time.monotonic_ns())
        self.fast.predict([],0,1000,time.monotonic_ns())
        with self.assertRaises(ValueError):self.fast.predict([],2000,3000,time.monotonic_ns())
