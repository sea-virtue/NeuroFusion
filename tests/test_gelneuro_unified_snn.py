"""Causality, state continuity and custom BPTT checks for the unified SNN."""
import unittest
import torch
from neurofusion.models.gelneuro_unified_snn import (
    GelNeuroUnifiedSNN, integrate, trace_readout,
)


class UnifiedSNNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(2)

    def test_iaf_bptt_matches_reference(self):
        for alpha in (0.,.5,1.):
            a=torch.randn(2,9,3,2,2,requires_grad=True)
            b=(torch.randn(2,3,2,2)*.1).requires_grad_()
            x=a.detach().clone().requires_grad_()
            m0=b.detach().clone().requires_grad_()
            m=m0
            ref=[]
            for current in x.unbind(1):
                v=(m+current).clamp_min(-1.)
                hard=v.floor().clamp_min(0.)
                spike=hard.detach()+v.relu()-v.relu().detach()
                m=v-spike
                ref.append(alpha*spike+(1.-alpha)*current.relu())
            y,end=integrate(a,b,alpha)
            expected=torch.stack(ref,1)
            torch.testing.assert_close(y,expected,atol=2e-6,rtol=1e-5)
            torch.testing.assert_close(end,m,atol=2e-6,rtol=1e-5)
            (y.square().sum()+end.square().sum()).backward()
            (expected.square().sum()+m.square().sum()).backward()
            torch.testing.assert_close(a.grad,x.grad,atol=1e-5,rtol=1e-5)
            torch.testing.assert_close(b.grad,m0.grad,atol=1e-5,rtol=1e-5)

    def test_trace_bptt_matches_reference(self):
        x=torch.rand(2,7,16,requires_grad=True)
        initial=torch.rand(2,4,16,requires_grad=True)
        decay=torch.tensor([.1,.5,.8,.95]).reshape(1,4,1)
        a=x.detach().clone().requires_grad_()
        b=initial.detach().clone().requires_grad_()
        y,end=trace_readout(x,initial,decay)
        state=b
        ref=[]
        for current in a.unbind(1):
            state=state*decay+current.unsqueeze(1)*(1.-decay)
            ref.append(state.log1p().flatten(1))
        expected=torch.stack(ref,1)
        torch.testing.assert_close(y,expected)
        (y.square().sum()+end.square().sum()).backward()
        (expected.square().sum()+state.square().sum()).backward()
        torch.testing.assert_close(x.grad,a.grad)
        torch.testing.assert_close(initial.grad,b.grad)

    def test_streaming_causality_silence_quantization(self):
        model=GelNeuroUnifiedSNN().eval()
        x=torch.poisson(torch.full((2,9,2,32,32),.2))
        with torch.inference_mode():
            full,state=model.encode(x)
            first,s=model.encode(x[:,:4])
            last,s=model.encode(x[:,4:],s)
            torch.testing.assert_close(full,torch.cat([first,last],1),atol=1e-6,rtol=1e-5)
            changed=x.clone(); changed[:,4:]=100.
            z,_=model.encode(changed)
            torch.testing.assert_close(full[:,:4],z[:,:4],atol=1e-6,rtol=1e-5)
            quiet,_=model.encode(torch.zeros_like(x))
            self.assertEqual(float(quiet.abs().max()),0.)
            q,layers=model.quantized_copy()
            out,_=q.encode(x)
            self.assertTrue(torch.isfinite(out).all())
            self.assertEqual(tuple(out.shape),(2,9,64))
            for layer in layers:
                self.assertGreaterEqual(layer["weight_min"],-128)
                self.assertLessEqual(layer["weight_max"],127)
                self.assertLessEqual(layer["threshold_high"],32767)
            self.assertIsNotNone(model.resource_estimate()["candidate_cores"])
            fine=GelNeuroUnifiedSNN.from_checkpoint(
                {"model":model.state_dict(),"bin_us":50000},bin_us=1000)
            torch.testing.assert_close(fine.decay,(-1./torch.tensor([50.,150.,500.,1500.])).exp().reshape(1,4,1))
            self.assertEqual(fine.rate_scale,50.)

    def test_balanced_batches_and_physical_trace_time(self):
        from neurofusion.training.train_gelneuro_unified_snn import BalancedBatches
        originals=[(None,torch.tensor([c]),None,None) for c in [0]*4+[1]*30+[2]*2+[3]*2]
        sampler=BalancedBatches(originals,31)
        batches=list(sampler)
        self.assertEqual(len(batches),10)
        for batch in batches:
            self.assertEqual({int(originals[i][1][0]) for i in batch},{0,1,2,3})
        self.assertEqual(batches,list(sampler))
        sampler.epoch=1
        self.assertNotEqual(batches,list(sampler))
        taus=torch.tensor([50.,150.,500.,1500.]).reshape(1,4,1)
        # A constant physical spike rate must have the same trace at equal times.
        coarse,_=trace_readout(torch.ones(1,3,16),torch.zeros(1,4,16),(-50./taus).exp())
        fine,_=trace_readout(torch.ones(1,15,16),torch.zeros(1,4,16),(-10./taus).exp())
        torch.testing.assert_close(coarse,fine[:,4::5],rtol=2e-5,atol=2e-6)


if __name__=="__main__":
    unittest.main()
