"""Read-only hardware inputs; Speck configuration sequence follows speck-lab."""
import time
import numpy as np


class SpeckSource:
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
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def read(self):
        events = self.sink.get_events()
        received_ns = time.monotonic_ns()
        values = np.asarray([[e.x,e.y,e.timestamp,int(e.p)] for e in events],dtype=np.int64).reshape(-1,4)
        return received_ns,values

    def __exit__(self,*args):
        try:
            if self.original is not None:
                self.device.get_model().apply_configuration(self.original)
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
