"""GelSight deformation scoring and independent camera acquisition."""

from collections import Counter, deque
from pathlib import Path
import re
import threading
import time

import cv2
import numpy as np

from ...adapters.live_devices import GelSightSource

SIZE = (192, 144)


def camera_index(index):
    if index is not None:
        return index
    paths = sorted(Path('/dev/v4l/by-id').glob('*GelSight*-video-index0'))
    if len(paths) != 1:
        raise RuntimeError('Expected one GelSight by-ID capture device; use --camera INDEX')
    match = re.fullmatch(r'video(\d+)', paths[0].resolve(strict=True).name)
    if match is None:
        raise RuntimeError(f'Unexpected GelSight device path: {paths[0]}')
    return int(match[1])


def downsample(rgb):
    return cv2.resize(rgb, SIZE, interpolation=cv2.INTER_AREA)


class ContactImage:
    """Spatial RGB change from the unloaded gel, after an edge-based offset."""

    def __init__(self, unloaded_rgb):
        if unloaded_rgb.shape != (SIZE[1], SIZE[0], 3):
            raise ValueError(f'Reference must be {SIZE[0]}x{SIZE[1]} RGB')
        self.reference = self._smooth(unloaded_rgb)

    @staticmethod
    def _smooth(rgb):
        return cv2.GaussianBlur(rgb, (3, 3), 0).astype(np.float32)

    def score(self, rgb):
        diff = self._smooth(rgb) - self.reference
        edges = np.concatenate((diff[28:118, 5:15], diff[28:118, 177:187]), axis=1)
        offset = np.median(edges.reshape(-1, 3), axis=0)
        contact = diff[24:124, 26:164] - offset
        return float(np.sqrt(np.mean(contact**2)))


class GelSightReader:
    """Keep blocking V4L2 reads outside the gripper loop."""

    def __init__(self, index, contact, predictor=None):
        self.index = index
        self.contact = contact
        self.predictor = predictor
        self.latest = None
        self.latest_guard = None
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)

    def _read(self):
        try:
            with GelSightSource(self.index) as source:
                for _ in range(12):
                    if self.stop.is_set():
                        return
                    source.read()
                while not self.stop.is_set():
                    stamp, rgb = source.read()
                    if self.contact is None:
                        self.latest = stamp, None, None, time.monotonic_ns()
                        continue
                    image = downsample(rgb)
                    guard_score = self.contact.score(image)
                    self.latest_guard = stamp, guard_score, time.monotonic_ns()
                    score = self.predictor.update(image) if self.predictor is not None else guard_score
                    if score is not None:
                        self.latest = stamp, score, guard_score, time.monotonic_ns()
        except Exception as error:
            self.error = error

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=1)


class RecentEvents:
    """Summarize both DVS polarities in a rolling host-receipt window."""

    def __init__(self, window_ns):
        self.window_ns = window_ns
        self.batches = deque()
        self.count = self.positive = 0
        self.sites = Counter()

    def update(self, now_ns, events):
        count = len(events)
        if count:
            positive = int(np.count_nonzero(events[:, 3]))
            sites = {tuple(position) for position in events[:, :2]}
            self.sites.update(sites)
            self.batches.append((now_ns, count, positive, sites))
            self.count += count
            self.positive += positive
        while self.batches and self.batches[0][0] <= now_ns - self.window_ns:
            _, expired, positive, sites = self.batches.popleft()
            self.sites.subtract(sites)
            for site in sites:
                if self.sites[site] == 0:
                    del self.sites[site]
            self.count -= expired
            self.positive -= positive
        return self.count, self.positive


def capture_reference(index, path):
    if path.exists():
        raise FileExistsError(f'Reference already exists: {path}; choose another --reference-image')
    path.parent.mkdir(parents=True, exist_ok=True)
    with GelSightSource(index) as source:
        for _ in range(12):
            source.read()
        frames = [downsample(source.read()[1]) for _ in range(13)]
    reference = np.rint(np.median(frames, axis=0)).astype(np.uint8)
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(reference, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError('Could not encode unloaded GelSight reference')
    with path.open('xb') as output:
        output.write(encoded.tobytes())
