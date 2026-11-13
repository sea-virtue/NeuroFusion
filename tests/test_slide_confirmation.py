from dataclasses import asdict,replace
import unittest

from neurofusion.control.slip import (SlipConfig,SlipPrediction,SlideConfirmation,
                                     SlipController,GripperFeedback)


BASE = 1_000_000_000


def prediction(sequence, task='slide', confidence=.8, receipt=BASE):
    # All source windows may be delivered by the same USB read.
    return SlipPrediction(sequence,receipt+sequence*10_000,receipt,
                          task,'up',confidence,.1,'test-model')


class SlideConfirmationTests(unittest.TestCase):
    def test_slide_probability_can_confirm_below_winning_quiet_probability(self):
        latch=SlideConfirmation(3,.35,.15)
        for i,score in enumerate((.36,.42,.37),1):
            p=replace(prediction(i,'quiet',confidence=1-score),slide_probability=score)
            latest=latch.observe(p,(i-1)*1000,i*1000,5000)
            if i<3:self.assertIsNone(latest.slide_evidence)
        # The newer quiet label must not erase an already confirmed event.
        p=replace(prediction(4,'quiet',confidence=.99),slide_probability=.01)
        latest=latch.observe(p,3000,4000,5000)
        self.assertAlmostEqual(latest.slide_evidence.confidence,.36)
        wire=SlipPrediction(**asdict(latest))
        cfg=self.config(slide_probability_threshold=.35)
        self.assertEqual(cfg.confidence,.6)
        controller=SlipController(cfg)
        row=controller.step(BASE+1_000_000,GripperFeedback(BASE,560/1150),wire)
        self.assertEqual(row['phase'],'BRAKING')
        self.assertEqual(row['command']['closure'],562/1150)
        self.assertIsNone(row['fault_reason'])

    def test_subthreshold_slide_probability_and_legacy_quiet_do_not_trigger(self):
        for score in (.34,None):
            controller=SlipController(self.config(slide_probability_threshold=.35))
            for i in range(1,5):
                p=replace(prediction(i,'quiet',confidence=.66 if score else .99,
                                     receipt=BASE+i*1_000_000),slide_probability=score)
                row=controller.step(p.available_ns+1000,GripperFeedback(p.captured_ns,562/1150),p)
                self.assertEqual(row['phase'],'GRIPPING')

    def test_direct_input_can_confirm_three_probability_windows_with_quiet_winner(self):
        controller=SlipController(self.config(slide_probability_threshold=.35))
        for i in range(1,4):
            p=replace(prediction(i,'quiet',confidence=.64,receipt=BASE+i*1_000_000),slide_probability=.36)
            row=controller.step(p.available_ns+1000,GripperFeedback(p.captured_ns,562/1150),p)
            self.assertEqual(row['phase'],'BRAKING' if i==3 else 'GRIPPING')

    def test_invalid_slide_probability_and_threshold_are_rejected(self):
        for value in (-.1,1.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):replace(prediction(1),slide_probability=value)
            with self.assertRaises(ValueError):self.config(slide_probability_threshold=value)
        with self.assertRaises(ValueError):self.config(slide_probability_threshold=0)

    def config(self, **changes):
        settings=dict(grip_closure=562/1150,confidence=.6,slide_confirm_frames=3)
        settings.update(changes)
        return SlipConfig(**settings)

    def batch(self, tasks, **options):
        latch=SlideConfirmation(3,.6,.15)
        latest=None
        for i,task in enumerate(tasks,1):
            latest=latch.observe(prediction(i,task,**options),(i-1)*1000,i*1000,len(tasks)*1000)
        return latest

    def test_confirm_before_coalescing_and_catch_even_when_latest_is_quiet(self):
        latest=self.batch(['slide','slide','slide','quiet','quiet'])
        self.assertEqual(latest.task,'quiet')
        wire=SlipPrediction(**asdict(latest))
        self.assertEqual(wire.slide_evidence.frames,3)
        self.assertEqual(wire.slide_evidence.last_sequence,3)
        controller=SlipController(self.config())
        row=controller.step(BASE+1_000_000,GripperFeedback(BASE,560/1150),wire)
        self.assertEqual(row['phase'],'BRAKING')
        self.assertEqual(row['command']['closure'],562/1150)
        self.assertIsNone(row['fault_reason'])
        # Later quiet cannot reopen or change the tightening target.
        quiet=prediction(6,'quiet',receipt=BASE+10_000_000)
        row=controller.step(BASE+11_000_000,GripperFeedback(BASE+10_000_000,561/1150),quiet)
        self.assertEqual(row['phase'],'BRAKING')
        self.assertEqual(row['command']['closure'],562/1150)

    def test_quiet_or_low_confidence_breaks_unconfirmed_streak(self):
        self.assertIsNone(self.batch(['slide','slide','quiet','slide','slide']).slide_evidence)
        latch=SlideConfirmation(3,.6,.15)
        for i,score in enumerate((.9,.9,.59,.9,.9),1):
            latest=latch.observe(prediction(i,confidence=score),(i-1)*1000,i*1000,5000)
        self.assertIsNone(latest.slide_evidence)

    def test_repeated_prediction_gap_or_changed_model_cannot_count_as_contiguous(self):
        for mode in ('duplicate','sequence_gap','source_gap','model_change'):
            latch=SlideConfirmation(3,.6,.15)
            latch.observe(prediction(1),0,1000,5000)
            latch.observe(prediction(2),1000,2000,5000)
            p=prediction(2 if mode=='duplicate' else 4 if mode=='sequence_gap' else 3)
            if mode=='model_change':p=replace(p,model_id='other-model')
            start=3000 if mode=='source_gap' else 2000
            latest=latch.observe(p,start,start+1000,5000)
            self.assertIsNone(latest.slide_evidence,mode)

    def test_old_catch_up_windows_cannot_create_live_evidence(self):
        latch=SlideConfirmation(3,.6,.15)
        for i in range(1,4):
            latest=latch.observe(prediction(i),(i-1)*1000,i*1000,1_000_000)
        self.assertIsNone(latest.slide_evidence)

    def test_new_quiet_cannot_refresh_time_of_old_latched_evidence(self):
        old=self.batch(['slide','slide','slide']).slide_evidence
        current=replace(prediction(4,'quiet',receipt=BASE+200_000_000),slide_evidence=old)
        controller=SlipController(self.config())
        row=controller.step(BASE+201_000_000,GripperFeedback(BASE+200_000_000,562/1150),current)
        self.assertEqual(row['phase'],'GRIPPING')
        self.assertIsNone(row['onset_available_ns'])

    def test_external_predictions_confirm_only_new_consecutive_frames_and_ignore_direction(self):
        controller=SlipController(self.config())
        for i in range(1,4):
            p=prediction(i,receipt=BASE+i*1_000_000)
            row=controller.step(p.available_ns+1000,GripperFeedback(p.captured_ns,561/1150),p)
            self.assertEqual(row['phase'],'BRAKING' if i==3 else 'GRIPPING')

    def test_repeated_control_ticks_do_not_complete_confirmation(self):
        controller=SlipController(self.config())
        p=prediction(1)
        for i in range(1,5):
            row=controller.step(BASE+i*1_000_000,GripperFeedback(BASE+i*1_000_000,562/1150),p)
            self.assertEqual(row['phase'],'GRIPPING')

    def test_evidence_must_meet_controller_frame_and_confidence_settings(self):
        original=self.batch(['slide','slide','slide','quiet'])
        for evidence in (replace(original.slide_evidence,confidence=.59),
                         replace(original.slide_evidence,first_sequence=2,frames=2)):
            controller=SlipController(self.config())
            latest=replace(original,slide_evidence=evidence)
            row=controller.step(BASE+1_000_000,GripperFeedback(BASE,562/1150),latest)
            self.assertEqual(row['phase'],'GRIPPING')

    def test_invalid_confirmation_and_future_wire_evidence_are_rejected(self):
        for frames in (0,-1,True,1.5):
            with self.assertRaises(ValueError):self.config(slide_confirm_frames=frames)
        latest=self.batch(['slide','slide','slide','quiet'])
        wire=asdict(latest)
        wire['slide_evidence']['available_ns']=latest.available_ns+1
        with self.assertRaises(ValueError):SlipPrediction(**wire)


if __name__=='__main__':unittest.main()
