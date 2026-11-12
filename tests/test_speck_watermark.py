from types import SimpleNamespace
import unittest

from neurofusion.adapters.live_devices import SpeckSource


class SpeckWatermarkTests(unittest.TestCase):
    def test_read_drains_dvs_after_clock_and_detects_counter_reset(self):
        calls=[]
        times=iter((200000,220000,1000))
        def clock():
            calls.append('clock')
            return [SimpleNamespace(timestamp=next(times))]
        def dvs():
            calls.append('dvs')
            return [SimpleNamespace(x=1,y=2,timestamp=210000,p=True)]
        source=SpeckSource(clock_hz=100.)
        source.clock_graph=object()
        source.clock_sink=SimpleNamespace(get_events=clock)
        source.sink=SimpleNamespace(get_events=dvs)
        _,values=source.read()
        self.assertEqual(calls,['clock','dvs'])
        self.assertEqual(values.tolist(),[[1,2,210000,1]])
        self.assertEqual(source.watermark_us,190000)
        source.read()
        self.assertEqual(source.watermark_us,210000)
        with self.assertRaisesRegex(RuntimeError,'clock reset'):
            source.read()

    def test_clock_cleanup_still_runs_when_configuration_restore_fails(self):
        calls=[]
        source=SpeckSource(clock_hz=100.)
        def fail(_):raise RuntimeError('restore failed')
        source.original=object()
        source.device=SimpleNamespace(get_model=lambda:SimpleNamespace(apply_configuration=fail))
        source.power_monitor=SimpleNamespace(stop_auto_power_measurement=lambda:calls.append('power'))
        source.clock_owned=True
        source.clock_graph=SimpleNamespace(stop=lambda:calls.append('clock_graph'))
        source.graph=SimpleNamespace(stop=lambda:calls.append('dvs_graph'))
        source.watch=SimpleNamespace(stop=lambda:calls.append('watch'))
        source.samna=SimpleNamespace(device=SimpleNamespace(close_device=lambda _:calls.append('close')))
        with self.assertRaisesRegex(RuntimeError,'restore failed'):
            source.__exit__(None,None,None)
        self.assertEqual(calls,['power','clock_graph','dvs_graph','watch','close'])

    def test_invalid_clock_configuration_fails_before_opening_device(self):
        for kwargs in (dict(clock_hz=0),dict(clock_hz=101),dict(clock_hz=float('nan')),dict(clock_lag_us=-1)):
            with self.assertRaises(ValueError):SpeckSource(**kwargs)
