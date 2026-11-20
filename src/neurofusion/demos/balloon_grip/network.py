"""Frozen Sparsh encoder and a small contact head for the balloon pilot."""

from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

from ...capture.reader import iter_frames, rows
from .learning import verified_recording
from .tactile import ContactImage, downsample

ENCODER = 'sparsh_dino_small'
FEATURE_DIM = 384


def load_reference(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Cannot read unloaded reference: {path}')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ContactImage(rgb)  # Check the same reference dimensions used by the contact proxy.
    return rgb


class ContactEncoder:
    def __init__(self, reference):
        import torch

        from ...models.encoders import ImageBackbone, prepare_sparsh

        self.torch = torch
        self.prepare = prepare_sparsh
        self.reference = reference
        torch.set_num_threads(min(torch.get_num_threads(), 3))
        self.model = ImageBackbone(ENCODER, pretrained=True, frozen=True).eval()

    def encode(self, pairs):
        torch = self.torch
        with torch.inference_mode():
            inputs = torch.stack([self.prepare(current, previous, self.reference)
                                  for current, previous in pairs])
            features = self.model(inputs)
            return torch.nn.functional.normalize(features, dim=1).numpy()


class ContactHead:
    def __init__(self, checkpoint, reference_path, reference):
        with np.load(checkpoint, allow_pickle=False) as archive:
            version = int(archive['version'])
            encoder = archive['encoder'].item()
            reference_hash = archive['reference_sha256'].item()
            self.weight = archive['weight'].astype(np.float64)
            self.bias = float(archive['bias'])
        if version != 1 or encoder != ENCODER or self.weight.shape != (FEATURE_DIM,):
            raise ValueError('Incompatible Sparsh contact head')
        if not np.isfinite(self.weight).all() or not np.isfinite(self.bias):
            raise ValueError('Nonfinite contact head parameters')
        if reference_hash != sha256(reference_path.read_bytes()).hexdigest():
            raise ValueError('Contact head requires the same unloaded image used at fit time')
        self.encoder = ContactEncoder(reference)
        self.previous = None

    def update(self, rgb):
        previous, self.previous = self.previous, rgb
        if previous is None:
            return None
        features = self.encoder.encode([(rgb, previous)])[0]
        value = float(features @ self.weight + self.bias)
        if not np.isfinite(value):
            raise RuntimeError('Sparsh contact estimate is nonfinite')
        return max(0.0, value)


def fit_ridge(features, targets, penalty=0.002):
    center = features.mean(axis=0)
    centered = features - center
    mean = targets.mean()
    gram = centered.T @ centered
    gram.flat[::len(gram)+1] += penalty
    weight = np.linalg.solve(gram, centered.T @ (targets - mean))
    return weight, float(mean - center @ weight)


def recording_start(trial):
    return next(mark['host_marked_ns'] for mark in rows(trial / 'annotations.jsonl')
                if mark['name'] == 'recording_start')


def fit_trial(trial, reference_path, checkpoint):
    verified_recording(trial)
    if checkpoint.exists():
        raise FileExistsError(f'Contact head already exists: {checkpoint}')
    reference = load_reference(reference_path)
    encoder = ContactEncoder(reference)
    teacher = ContactImage(reference)
    start_ns = recording_start(trial)
    values, targets, times, pairs = [], [], [], []
    previous = None
    for frame, image in iter_frames(trial):
        current = downsample(image)
        sec = (frame['host_received_ns'] - start_ns) / 1e9
        if previous is not None and sec >= .9:
            pairs.append((current, previous))
            targets.append(teacher.score(current))
            times.append(sec)
            if len(pairs) == 8:
                values.append(encoder.encode(pairs))
                pairs.clear()
        previous = current
    if pairs:
        values.append(encoder.encode(pairs))
    if len(targets) < 24:
        raise ValueError('Need at least 24 usable GelSight frame pairs to fit contact')
    features = np.concatenate(values).astype(np.float64)
    if not np.isfinite(features).all():
        raise RuntimeError('Nonfinite Sparsh training features')
    targets = np.asarray(targets)
    times = np.asarray(times)
    # Hold out contiguous blocks within this single trial. The metrics compare
    # the image proxy, with no force or independent-trial ground truth.
    block = np.floor(times / .7).astype(int)
    held = block % 4 == 2
    fit = (block % 4 == 0)
    if held.any() and fit.any():
        probe_weight, probe_bias = fit_ridge(features[fit], targets[fit])
        errors = features[held] @ probe_weight + probe_bias - targets[held]
        held_mae = float(np.mean(np.abs(errors)))
    else:
        held_mae = None
    weight, bias = fit_ridge(features, targets)
    fitted = features @ weight + bias
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open('xb') as stream:
        np.savez_compressed(
            stream, version=1, encoder=ENCODER, weight=weight, bias=bias,
            reference_sha256=sha256(reference_path.read_bytes()).hexdigest(),
            source_trial=str(trial.resolve()), train_pairs=len(targets),
        )
    report = {
        'model': str(checkpoint.resolve()), 'trial': str(trial.resolve()),
        'frames': len(targets), 'teacher': 'unloaded-image deformation RMS',
        'fitted_mae': float(np.mean(np.abs(fitted-targets))),
        'blocked_same_trial_mae': held_mae, 'blocked_frames': int(held.sum()),
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)
