import unittest
import tempfile
from pathlib import Path
from dataclasses import replace
import torch
from neurofusion.models.candidates import CandidateConfig
from neurofusion.models.policy import NeuroFusionPolicy, Observation
from neurofusion.models.fusion import FusionConfig
from neurofusion.models.snn_adapter import SNNPacketEncoder, count_bins
from neurofusion.training.pipeline import smoke_data, sequence_loss, save_checkpoint, restore_checkpoint

CHECKPOINT='artifacts/checkpoints/gelneuro-snn-20261111/final/all-data.pt'

class SNNPolicyTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2)

    def test_counts_and_streaming_continuity(self):
        net=SNNPacketEncoder(CHECKPOINT).eval()
        events=torch.tensor([[2,3,500,1],[60,80,1500,0],[70,90,2500,1]])
        counts=count_bins([events],0,3000,1000,'cpu')
        self.assertEqual(counts.sum().item(),3)
        full,end=net([events],0,3000)
        _,state=net([events[:1]],0,1000)
        tail,state=net([events[1:]],1000,3000,state)
        torch.testing.assert_close(full,tail)
        torch.testing.assert_close(end[1],state[1])
        with self.assertRaises(ValueError): net([events],0,2500)
        with self.assertRaises(ValueError): net([events.flip(0)],0,3000)

    def test_end_to_end_training_reset_and_checkpoint(self):
        choice=CandidateConfig(event_encoder='snn',event_checkpoint=CHECKPOINT,pretrained=False)
        model=NeuroFusionPolicy(candidate=choice)
        obs,targets=smoke_data(choice)
        model.actions.fit_normalizer(torch.cat(targets))
        loss,state=sequence_loss(model,obs,targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(state.event_state)
        self.assertIsNone(model.initial_state(2).event_state)
        self.assertTrue(all(p.grad is None for p in model.events.parameters()))
        for module in (model.fusion,model.actions,model.rgb.projection,model.tactile.projection):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None),0)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'policy.pt'
            save_checkpoint(path,model)
            # Restore must not depend on the external SNN checkpoint path.
            restored=restore_checkpoint(path).eval()
            model.eval()
            a,_=model.rollout(obs,model.initial_state(2))
            b,_=restored.rollout(obs,restored.initial_state(2))
            torch.testing.assert_close(a.memory,b.memory)
            torch.testing.assert_close(a.event_state[1],b.event_state[1])
            self.assertEqual(len(restored.propose(b)),2)

    def test_gap_requires_explicit_empty_windows(self):
        choice=CandidateConfig(event_encoder='snn',event_checkpoint=CHECKPOINT,pretrained=False)
        model=NeuroFusionPolicy(candidate=choice).eval()
        state=model.initial_state(1)
        first=Observation('event',1000000,[torch.empty(0,4)],captured_ns=1000000,start_us=0,end_us=1000)
        state=model.step(state,first)
        with self.assertRaises(ValueError):
            model.step(state,replace(first,available_ns=3000000,captured_ns=3000000,start_us=2000,end_us=3000))
        state=model.step(state,replace(first,available_ns=2000000,captured_ns=2000000,start_us=1000,end_us=2000))
        self.assertEqual(state.event_end_us,2000)
