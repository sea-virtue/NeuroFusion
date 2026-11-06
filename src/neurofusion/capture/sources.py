"""Recorder-specific native camera reader; existing model adapter stays unchanged."""
from pathlib import Path
import time

from .schema import NativeFrame


class NativeGelSightSource:
    """Keep original MJPEG bytes, avoiding decode/re-encode on the acquisition path."""
    def __init__(self, index=0):
        self.index = index

    def __enter__(self):
        import cv2
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        try:
            if not self.capture.isOpened():
                raise RuntimeError(f"GelSight /dev/video{self.index} unavailable; close other viewers and check USB attachment")
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            if int(self.capture.get(cv2.CAP_PROP_FOURCC)) != cv2.VideoWriter_fourcc(*"MJPG"):
                raise RuntimeError("Camera does not provide MJPG; this recorder will not silently re-encode frames")
            if not self.capture.set(cv2.CAP_PROP_CONVERT_RGB, 0):
                raise RuntimeError("Camera backend cannot return native MJPEG bytes")
            self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            name_path = Path(f"/sys/class/video4linux/video{self.index}/name")
            self.info = {"camera_index": self.index, "storage_encoding": "native_mjpeg_per_frame",
                         "driver_reported_size": [self.width, self.height],
                         "driver_reported_fps": self.capture.get(cv2.CAP_PROP_FPS),
                         "camera_name": name_path.read_text().strip() if name_path.exists() else None,
                         "requested_size": None, "host_resize": False, "host_recompression": False}
            return self
        except BaseException:
            self.capture.release()
            raise

    def read(self):
        ok, raw = self.capture.read()
        received = time.monotonic_ns()
        if not ok:
            raise RuntimeError("GelSight native frame read failed")
        if raw.dtype.name != "uint8" or raw.ndim != 2 or raw.shape[0] != 1:
            raise RuntimeError(f"Backend returned decoded/unsupported data {raw.shape}; expected native MJPEG bytes")
        payload = raw.tobytes()
        if not payload.startswith(b"\xff\xd8"):
            raise RuntimeError("Camera frame lacks JPEG start marker")
        return received, NativeFrame(payload, self.width, self.height)

    def __exit__(self, *args):
        self.capture.release()
