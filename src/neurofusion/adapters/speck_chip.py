"""Deployed CNN output, retaining the existing hardware time watermark."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .live_devices import SpeckSource
from ..speck_deploy import apply_verified, load_config


class SpeckChipSource(SpeckSource):
    def __init__(self, config, checkpoint, *, clock_hz=100., clock_lag_us=10000):
        super().__init__(clock_hz=clock_hz,clock_lag_us=clock_lag_us)
        self.path,self.checkpoint = Path(config),Path(checkpoint)

    def __enter__(self):
        import samna
        self.samna = samna
        mapping = json.loads(self.path.with_name('mapping.json').read_text())
        checkpoint_sha = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        if mapping['checkpoint_sha256'] != checkpoint_sha:
            raise ValueError('Chip configuration and recognition checkpoint do not match')
        config = load_config(self.path,samna)
        outputs = [i for i,c in enumerate(config.cnn_layers) if c.monitor_enable]
        if len(outputs) != 1:
            raise ValueError('Expected one monitored CNN output core')
        self.output_core = outputs[0]
        shape = config.cnn_layers[self.output_core].dimensions.output_shape
        if (shape.feature_count,shape.size.x,shape.size.y) != (16,1,1):
            raise ValueError('Expected 16x1x1 CNN output')
        config.dvs_layer.raw_monitor_enable = False
        config.dvs_layer.monitor_enable = False
        config.dvs_layer.pass_sensor_events = True
        devices = [d for d in samna.device.get_unopened_devices() if d.device_type_name=='Speck2fModuleDevKit']
        if len(devices) != 1:
            raise RuntimeError(f'Expected one unopened Speck board, found {len(devices)}')
        self.device = samna.device.open_device(devices[0])
        self.graph = samna.graph.EventFilterGraph()
        self.original = self.watch = None
        try:
            self.original = self.device.get_model().get_configuration()
            actual = apply_verified(self.device.get_model(),config,samna)
            self.applied_json = actual.to_json()
            self.config_id = hashlib.sha256(self.path.read_bytes()).hexdigest()
            self.sink = samna.BasicSinkNode_speck2f_event_output_event()
            self.graph.sequential([self.device.get_model_source_node(),self.sink])
            self.graph.start()
            self.watch = self.device.get_stop_watch()
            self.watch.start()
            self.sink.get_events()
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
        measurements = self.clock_sink.get_events()
        batch = self.sink.get_events()
        received = time.monotonic_ns()
        if measurements:
            latest = max(int(e.timestamp) for e in measurements)
            if self.last_clock_us is not None and latest < self.last_clock_us-50000:
                raise RuntimeError('Chip hardware clock reset/wrapped')
            if self.last_clock_us is None or latest > self.last_clock_us:
                self.last_clock_us = latest
                self.watermark_us = max(0,latest-self.clock_lag_us)
                self.watermark_host_ns = received
        rows = []
        for event in batch:
            if isinstance(event,self.samna.speck2f.event.Spike):
                if event.layer != self.output_core:
                    raise RuntimeError('Unexpected monitored core')
                if not (0 <= event.feature < 16 and event.x == event.y == 0):
                    raise RuntimeError('Unexpected CNN output coordinates')
                # Existing contiguous-window buffer: x stores output feature ID.
                rows.append((event.feature,0,event.timestamp,0))
        values = np.asarray(rows,dtype=np.int64).reshape(-1,4)
        if len(values):
            values = values[np.argsort(values[:,2],kind='stable')]
        return received,values
