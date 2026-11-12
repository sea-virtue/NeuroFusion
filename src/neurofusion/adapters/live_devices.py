"""Read-only hardware inputs; Speck configuration sequence follows speck-lab."""
import time
import math
import numpy as np


class SpeckSource:
    def __init__(self, *, clock_hz=None, clock_lag_us=10_000):
        if clock_hz is not None and (not math.isfinite(clock_hz) or not 0 < clock_hz <= 100):
            raise ValueError('Device clock measurement rate must be in (0,100] Hz')
        if type(clock_lag_us) is not int or clock_lag_us < 0:
            raise ValueError('Nonnegative integer device clock lag required')
        self.clock_hz,self.clock_lag_us = clock_hz,clock_lag_us
        self.clock_graph = self.power_monitor = None
        self.clock_owned = False
        self.actual_clock_hz = None
        self.watermark_us = self.watermark_host_ns = self.last_clock_us = None

    def __enter__(self):
        import samna
        self.samna = samna
        devices = [d for d in samna.device.get_unopened_devices() if d.device_type_name == 'Speck2fModuleDevKit']
        if len(devices) != 1:
            raise RuntimeError(f'Expected one unopened Speck2f board, found {len(devices)}')
        self.device = samna.device.open_device(devices[0])
        self.graph = samna.graph.EventFilterGraph()
        self.original = None
        self.watch = None
        try:
            model = self.device.get_model()
            self.original = model.get_configuration()
            config = samna.speck2f.configuration.SpeckConfiguration()
            config.from_json(self.original.to_json())
            config.dvs_layer.raw_monitor_enable = True
            _,filter_,self.sink = self.graph.sequential([self.device.get_model_source_node(),
                'Speck2fOutputEventTypeFilter',samna.BasicSinkNode_speck2f_event_output_event()])
            filter_.set_desired_type('speck2f::event::DvsEvent')
            self.graph.start()
            self.watch = self.device.get_stop_watch()
            self.watch.start()
            self.sink.get_events()
            model.apply_configuration(config)
            if self.clock_hz is not None:
                # Power measurements carry the same resettable FPGA timestamp
                # as DVS events. They advance even when the DVS is silent.
                self.power_monitor = self.device.get_power_monitor()
                self.clock_graph = samna.graph.EventFilterGraph()
                _,self.clock_sink = self.clock_graph.sequential([
                    self.power_monitor.get_source_node(),
                    samna.BasicSinkNode_unifirm_modules_events_measurement()])
                self.clock_graph.start()
                if not self.power_monitor.is_auto_power_measurement_active():
                    self.clock_owned = True
                    self.actual_clock_hz = self.power_monitor.start_auto_power_measurement(self.clock_hz)
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def read(self):
        measurements = self.clock_sink.get_events() if self.clock_graph is not None else []
        # Drain DVS after reading the clock channel, allowing a short guarded
        # delay for independent graph/USB queues. Late DVS events are rejected
        # by the window buffer rather than silently reassigned to future bins.
        events = self.sink.get_events()
        received_ns = time.monotonic_ns()
        if measurements:
            latest = max(int(e.timestamp) for e in measurements)
            if self.last_clock_us is not None and latest < self.last_clock_us-50_000:
                raise RuntimeError('Speck hardware clock reset or wrapped during capture')
            if self.last_clock_us is None or latest > self.last_clock_us:
                self.last_clock_us = latest
                self.watermark_us = max(0,latest-self.clock_lag_us)
                self.watermark_host_ns = received_ns
        values = np.asarray([[e.x,e.y,e.timestamp,int(e.p)] for e in events],dtype=np.int64).reshape(-1,4)
        return received_ns,values

    def __exit__(self,*args):
        try:
            if self.original is not None:
                self.device.get_model().apply_configuration(self.original)
        finally:
            try:
                try:
                    if self.clock_owned: self.power_monitor.stop_auto_power_measurement()
                finally:
                    if self.clock_graph is not None: self.clock_graph.stop()
            finally:
                try: self.graph.stop()
                finally:
                    try:
                        if self.watch is not None: self.watch.stop()
                    finally: self.samna.device.close_device(self.device)


class GelSightSource:
    def __init__(self, index=0):
        self.index = index

    def __enter__(self):
        import cv2
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(self.index,cv2.CAP_V4L2)
        if not self.capture.isOpened():
            raise RuntimeError(f'GelSight camera /dev/video{self.index} unavailable')
        self.capture.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.capture.set(cv2.CAP_PROP_FPS,30)
        return self

    def read(self):
        ok,bgr = self.capture.read()
        received_ns = time.monotonic_ns()
        if not ok:
            raise RuntimeError('GelSight camera frame read failed')
        return received_ns,self.cv2.cvtColor(bgr,self.cv2.COLOR_BGR2RGB)

    def __exit__(self,*args):
        self.capture.release()
