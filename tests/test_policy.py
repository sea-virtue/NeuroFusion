import unittest
from dataclasses import replace
import torch
from neurofusion.models.candidates import CandidateConfig
from neurofusion.models.fusion import ContactFusion, FusionConfig
from neurofusion.models.policy import NeuroFusionPolicy, Observation
from neurofusion.models.split_actions import SplitActionHead
from neurofusion.models.events import voxelize
from neurofusion.training.pipeline import smoke_data, sequence_loss


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_sequence_gradients_all_fusion_variants(self):
        for kind in ('plain','gated','attention'):
            with self.subTest(fusion=kind):
                torch.manual_seed(7)
                choice = CandidateConfig()
                model = NeuroFusionPolicy(FusionConfig(kind=kind),choice)
                observations,targets = smoke_data(choice)
                model.actions.fit_normalizer(torch.cat(targets))
                loss,state = sequence_loss(model,observations,targets)
                loss.backward()
                for module in (model.rgb.projection,model.tactile.projection,model.events,model.fusion,model.actions):
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None),0)
                self.assertTrue(all(p.grad is None for p in model.rgb.backbone.parameters()))
                self.assertTrue(all(p.grad is None for p in model.tactile.backbone.parameters()))
                self.assertIsNotNone(state.memory.grad_fn)
                self.assertIsNone(state.detach().memory.grad_fn)

    def test_split_heads_reserved_slot_and_independent_destinations(self):
        for kind in ('mlp','act','diffusion_unet','diffusion_transformer'):
            with self.subTest(head=kind):
                choice = CandidateConfig(action_head=kind,inference_steps=2)
                head = SplitActionHead(choice)
                shape = (2,8) if kind == 'mlp' else (2,16,8)
                labels = torch.rand(shape)
                head.fit_normalizer(labels)
                condition = torch.randn(2,400,requires_grad=True)
                torch.manual_seed(9)
                loss = head.loss(condition,labels)
                changed = labels.clone(); changed[...,6] = 99999
                torch.manual_seed(9)
                self.assertTrue(torch.allclose(loss,head.loss(condition,changed)))
                loss.backward()
                self.assertGreater(condition.grad.abs().sum(),0)
                head.eval()
                actions = head.sample(condition.detach(),physical=True)
                self.assertEqual(actions.shape,labels.shape)
                self.assertTrue((actions[...,6] == 0).all())
                single = actions[0] if kind == 'mlp' else actions[0,0]
                proposal = head.proposal(single,10,2,False)
                self.assertEqual(len(proposal.franka),7)
                self.assertEqual(proposal.franka[6],0)
                self.assertEqual(len(proposal.robotis),1)

    def test_causal_prefix_and_separate_execution_feedback(self):
        model = NeuroFusionPolicy(candidate=CandidateConfig(pretrained=False)).eval()
        state = model.initial_state(1)
        initial = state.memory.clone()
        state = model.step(state,Observation('arm_executed',0,torch.tensor([[1.,2,3,4,5,6,0]])))
        self.assertEqual(state.executed[0,7],0)
        state = model.step(state,Observation('gripper_executed',0,torch.ones(1,1)))
        self.assertEqual(state.executed[0,0],1)
        self.assertEqual(state.executed[0,7],1)
        self.assertTrue(torch.equal(initial,state.memory))
        state = model.step(state,Observation('control',10))
        before = state.executed.clone()
        model.propose(state)
        self.assertTrue(torch.equal(before,state.executed))
        with self.assertRaises(ValueError): model.step(state,Observation('control',9))
        with self.assertRaises(ValueError): model.step(state,Observation('rgb',11,None,captured_ns=12))
        with self.assertRaises(ValueError): model.step(state,Observation('frame',11,None))
        self.assertTrue((model.initial_state(1).memory == 0).all())

    def test_event_window_bounds_empty_and_order(self):
        packet = torch.tensor([[0,0,0,0],[127,127,4999,1]])
        grid = voxelize([packet,torch.empty(0,4)],0,5000)
        self.assertEqual(grid.shape,(2,8,32,32))
        self.assertEqual(grid[1].sum(),0)
        self.assertAlmostEqual(grid[0].expm1().sum().item(),2,places=5)
        with self.assertRaises(ValueError): voxelize([torch.tensor([[0,0,5000,1]])],0,5000)
        with self.assertRaises(ValueError): voxelize([packet.flip(0)],0,5000)

    def test_zero_interval_and_bounded_attention_memory(self):
        for kind in ('plain','gated','attention'):
            net = ContactFusion(FusionConfig(kind=kind))
            h = torch.randn(2,256)
            inputs = (h,torch.randn(2,128),torch.randn(2,16),torch.randn(2,8),torch.zeros(2,6))
            self.assertTrue(torch.equal(net(*inputs,'propagation'),h))
            next_h = net(*inputs,'event',torch.randn(2,64))
            self.assertEqual(next_h.shape,h.shape)
            self.assertFalse(torch.equal(next_h,h))

    def test_event_content_affects_later_memory_and_action(self):
        model = NeuroFusionPolicy(candidate=CandidateConfig(pretrained=False)).eval()
        initial = model.initial_state(1)
        event = Observation('event',5_000_000,[torch.tensor([[30,40,1000,1]])],
                            captured_ns=5_000_000,start_us=0,end_us=5000)
        empty = replace(event,value=[torch.empty(0,4)])
        a = model.step(model.step(initial,event),Observation('control',10_000_000))
        b = model.step(model.step(initial,empty),Observation('control',10_000_000))
        self.assertFalse(torch.equal(a.memory,b.memory))
        self.assertFalse(torch.equal(model.actions(model.condition(a)),model.actions(model.condition(b))))
        self.assertTrue((initial.memory == 0).all())
