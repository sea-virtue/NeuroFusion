"""Integration/gradient boundaries introduced by optional upstream adapters."""
import importlib.util
import unittest
from dataclasses import replace
from neurofusion.config import Config
from neurofusion.contracts import Kind, Message, Stamp
from neurofusion.demo import ToyFrameCorrection, ToyEventUpdate, ToyPropagation
from neurofusion.runtime.engine import Engine
from neurofusion.adapters.interfaces import ShadowSink
from neurofusion.models.candidates import CandidateConfig


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional models dependencies not installed')
class CandidateIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def test_frozen_backbone_allows_projection_training(self):
        import torch
        from neurofusion.models.encoders import ImageBackbone
        model = ImageBackbone('resnet18', 128, pretrained=False)
        model.train()
        before = model.backbone.bn1.running_mean.clone()
        model(torch.rand(2, 3, 224, 224)).square().mean().backward()
        self.assertIsNotNone(model.projection.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertTrue(torch.equal(before, model.backbone.bn1.running_mean))

    def test_all_heads_condition_gradients_and_output(self):
        import torch
        from neurofusion.models.candidates import build_action_head
        candidate = CandidateConfig(inference_steps=2)
        for name in ('mlp', 'act', 'diffusion_unet', 'diffusion_transformer'):
            with self.subTest(name=name):
                model = build_action_head(replace(candidate, action_head=name))
                condition = torch.randn(2, 400, requires_grad=True)
                target = torch.rand((2, 5) if name == 'mlp' else (2, 16, 5)) * 2 - 1
                model.loss(condition, target).backward()
                self.assertGreater(condition.grad[:, :256].abs().sum().item(), 0)
                model.eval()
                with torch.no_grad():
                    output = model.sample(condition) if hasattr(model, 'sample') else model(condition)
                self.assertEqual(output.shape, target.shape)
                self.assertTrue(torch.isfinite(output).all())

    def test_each_head_is_accepted_by_existing_engine(self):
        from neurofusion.models.candidates import build_action_head
        from neurofusion.models.bridges import ActionHeadBridge
        for name in ('mlp', 'act', 'diffusion_unet', 'diffusion_transformer'):
            with self.subTest(name=name):
                model = build_action_head(CandidateConfig(action_head=name, inference_steps=2))
                head = ActionHeadBridge(model, (0.,)*5, (0.001,)*5)
                sink = ShadowSink()
                engine = Engine(Config(), ToyFrameCorrection(), ToyEventUpdate(), ToyPropagation(), head, sink)
                engine.submit(Message(Kind.EVENT, Stamp(5_000_000), (.1,)+(0.,)*63))
                engine.submit(Message(Kind.CONTROL, Stamp(10_000_000)))
                engine.run_until()
                self.assertEqual(len(sink.proposals), 1)
                self.assertEqual(len(sink.proposals[0].values), 5)
                self.assertEqual(engine.state.last_event_ns, 5_000_000)

    def test_sparsh_preprocess_matches_background_difference(self):
        import numpy as np
        import torch
        from neurofusion.models.encoders import prepare_sparsh
        bg = np.full((320, 240, 3), 100, dtype=np.uint8)
        value = prepare_sparsh(bg+20, bg-20, bg)
        self.assertEqual(value.shape, (6, 224, 224))
        self.assertTrue(torch.allclose(value[:3], torch.full_like(value[:3], 147/255)))
        self.assertTrue(torch.allclose(value[3:], torch.full_like(value[3:], 107/255)))

    def test_pretrained_missing_fails_instead_of_silent_random(self):
        from neurofusion.models.encoders import ImageBackbone
        with self.assertRaises(FileNotFoundError):
            ImageBackbone('resnet18', checkpoint='/nonexistent/neurofusion-weights.pth')

    def test_candidate_limits_and_action_scale(self):
        from neurofusion.models.candidates import build_action_head
        from neurofusion.models.bridges import ActionHeadBridge
        with self.assertRaises(ValueError):
            CandidateConfig(rgb_encoder='dinov2_vitg14')
        with self.assertRaises(ValueError):
            CandidateConfig(horizon=3)
        with self.assertRaises(ValueError):
            ActionHeadBridge(build_action_head(), (0.,)*5, (0.,)*5)
