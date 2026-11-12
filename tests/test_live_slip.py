import unittest

import numpy as np

from neurofusion.control.live import EventWindows


def events(*times):
    return np.asarray([[1, 2, t, 1] for t in times], dtype=np.int64).reshape(-1, 4)


class LiveWindowsTests(unittest.TestCase):
    def test_boundary_event_belongs_to_next_window_and_receipt_is_conservative(self):
        windows = EventWindows(1000)
        windows.push(events(101, 19999), 100)
        self.assertEqual(list(windows.ready()), [])
        windows.push(events(20000, 39999, 40000), 200)
        rows = list(windows.ready())
        self.assertEqual([(a, b, c) for _, a, b, c in rows], [(0, 20000, 200), (20000, 40000, 200)])
        self.assertEqual(rows[0][0][:, 2].tolist(), [101, 19999])
        self.assertEqual(rows[1][0][:, 2].tolist(), [20000, 39999])
        self.assertEqual(windows.pending[:, 2].tolist(), [40000])

    def test_empty_windows_between_events_preserve_device_time(self):
        windows = EventWindows(1000)
        windows.push(events(5010, 65000), 100)
        rows = list(windows.ready())
        self.assertEqual([(a, b) for _, a, b, _ in rows], [(5000, 25000), (25000, 45000), (45000, 65000)])
        self.assertEqual([len(row[0]) for row in rows], [1, 0, 0])

    def test_empty_polls_do_not_invent_device_time(self):
        windows = EventWindows(1000)
        windows.push(events(100), 100)
        windows.push([], 10_000_000_000)
        self.assertEqual(list(windows.ready()), [])

    def test_hardware_watermark_closes_empty_windows_without_host_clock_guess(self):
        windows=EventWindows(1000)
        windows.push([],100)
        windows.advance_watermark(10000,100)
        self.assertEqual(list(windows.ready()),[])
        windows.push([],200)
        windows.advance_watermark(70000,200)
        rows=list(windows.ready())
        self.assertEqual([(a,b,c) for _,a,b,c in rows],[(10000,30000,200),(30000,50000,200),(50000,70000,200)])
        self.assertTrue(all(len(v)==0 for v,_,_,_ in rows))

    def test_dvs_arriving_after_hardware_closed_window_is_rejected(self):
        windows=EventWindows(1000)
        windows.advance_watermark(10000,100)
        windows.advance_watermark(30000,200)
        list(windows.ready())
        with self.assertRaisesRegex(ValueError,'Late DVS'):
            windows.push(events(29999),300)

    def test_device_reset_and_wrap_are_rejected(self):
        windows = EventWindows(1000)
        windows.push(events(10000), 100)
        with self.assertRaises(ValueError):
            windows.push(events(10), 200)

    def test_backlog_and_invalid_events_fail(self):
        windows = EventWindows(1000)
        with self.assertRaises(RuntimeError):
            windows.push(events(0, 10000001), 100)
        for value in ([[128, 0, 1, 0]], [[0, 0, 1, 2]], [[0, 0, 1.5, 0]], [[0, 0, float('nan'), 0]]):
            with self.assertRaises(ValueError):
                EventWindows(1000).push(value, 100)

    def test_realistic_sparse_polls_preserve_all_intervening_empty_windows(self):
        windows = EventWindows(1000)
        windows.push(events(1000), 100)
        windows.push(events(901000), 900_000_100)
        rows = list(windows.ready())
        self.assertEqual(len(rows), 45)
        self.assertEqual(sum(len(row[0]) for row in rows), 1)
        self.assertEqual(rows[-1][1:], (881000, 901000, 900_000_100))
        self.assertEqual(windows.pending[:, 2].tolist(), [901000])

    def test_invalid_bin_and_reversed_host_time(self):
        with self.assertRaises(ValueError):
            EventWindows(3000)
        windows = EventWindows(1000)
        windows.push([], 100)
        with self.assertRaises(ValueError):
            windows.push([], 99)


if __name__ == '__main__':
    unittest.main()
