import unittest
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from neurofusion.gripper_benchmark import decode_block,validate_sample,trial_metrics,Bench,run


class GripperBenchmarkTests(unittest.TestCase):
    def test_block_decodes_signed_current_velocity_and_units(self):
        data = bytearray(30)
        for offset,size,value in ((0,4,20),(4,4,4),(8,2,25),(10,4,1),(15,4,10),
                                  (19,4,-3),(25,2,-2),(27,2,243),(29,1,43)):
            data[offset:offset+size] = value.to_bytes(size,'little',signed=value<0)
        sample = decode_block(data)
        self.assertEqual(sample['position_raw'],10)
        self.assertEqual(sample['velocity_raw'],-3)
        self.assertAlmostEqual(sample['current_a'],-.00804)
        with self.assertRaises(ValueError): decode_block(data[:-1])

    def test_goal_reset_out_of_range_and_current_spike_abort(self):
        sample = dict(goal_raw=10,position_raw=10,current_raw=0,voltage_v=24.,temperature_c=43)
        validate_sample(sample,target=10,low=6,high=35,current_raw=25)
        for key,value in (('goal_raw',0),('position_raw',36),('current_raw',40),('voltage_v',0),('temperature_c',60)):
            with self.assertRaises(RuntimeError):
                validate_sample(dict(sample,**{key:value}),target=10,low=6,high=35,current_raw=25)

    def test_metrics_do_not_mistake_ack_for_motion_or_convert_counts_to_mm(self):
        samples = [dict(position_raw=p,query_end_ns=t,current_a=.02,velocity_raw=0)
                   for t,p in [(10_000_000,10),(20_000_000,11),(30_000_000,13),
                               (40_000_000,20),(200_000_000,20),(220_000_000,20)]]
        result = trial_metrics(samples,10,20,0,2_000_000)
        self.assertEqual(result['command_ack_ms'],2.)
        self.assertEqual(result['first_observed_3_count_change_ms'],30.)
        self.assertEqual(result['first_within_2_counts_ms'],40.)
        self.assertEqual(result['steady_mean_error_raw'],0)
        self.assertNotIn('speed_mm_s',result)

    def test_position_profile_rejects_zero_current_instead_of_testing_a_hold(self):
        device = SimpleNamespace(limits={'min_raw':0,'max_raw':1150,'current_raw':820},table=None)
        bench = Bench(device,io.StringIO(),503)
        with self.assertRaisesRegex(ValueError,'positive ceiling'):
            bench.profile(0,4)

    def test_hold_startup_uses_nonzero_preload_without_changed_goal_or_zero_ramp(self):
        import sys
        import types
        from test_robotis import FakePort,FakePacket
        packet = FakePacket()
        sdk = types.SimpleNamespace(PortHandler=FakePort,PacketHandler=lambda protocol:packet)
        with tempfile.TemporaryDirectory() as root,patch.dict(sys.modules,{'dynamixel_sdk':sdk}),patch(
                'neurofusion.gripper_benchmark.Bench.monitor',return_value=[]) as monitor,patch(
                'neurofusion.gripper_benchmark.time.sleep'),contextlib.redirect_stdout(io.StringIO()):
            report = run(Path(root)/'hold','hold',operator_ready=True)
        self.assertTrue(report['passed'])
        self.assertTrue(report['cleanup']['torque_off_confirmed'])
        profile = next(value for address,value in packet.log if address == 596 and isinstance(value,bytes))
        self.assertEqual(int.from_bytes(profile[:4],'little'),503)
        self.assertEqual(int.from_bytes(profile[8:],'little'),820)
        self.assertNotIn((604,0),packet.log)
        monitor.assert_called_once_with(503,.6,'enable-hold')


if __name__ == '__main__': unittest.main()
