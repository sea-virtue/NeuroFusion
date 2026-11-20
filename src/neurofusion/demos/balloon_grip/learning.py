"""Raw balloon trials and offline features from the project GelSight encoder."""

from collections import deque
import json
import os
from pathlib import Path
import sys

import numpy as np

from ...capture.reader import iter_frames, rows, safe_path


CAPTURE_TASKS = {
    'background': 'background',
    'hold': 'static_contact',
    'load-cycle': 'press_release',
    'pull-rebound': 'mixed',
    'oscillation': 'mixed',
}


def collect(args, demo, index):
    """Run the shared dual-sensor recorder without opening the motor port."""
    protocol = args.collect
    notes = json.dumps({
        'balloon_protocol': protocol,
        'extension_mm': args.extension_mm,
        'extension_basis': 'operator measured from resting position' if args.extension_mm is not None else None,
        'top_assist': args.top_assist,
        'operator_notes': args.notes,
    }, ensure_ascii=False)
    duration = args.seconds
    direction = 'down' if protocol in ('load-cycle', 'pull-rebound', 'oscillation') else 'none'
    command = [
        sys.executable, '-m', 'neurofusion.capture', 'record',
        '--root', str(demo / 'outputs' / 'raw'),
        '--task', CAPTURE_TASKS[protocol], '--camera', str(index),
        '--seconds', str(duration), '--object-id', args.object_id,
        '--direction', direction, '--repetition', str(args.repetition),
        '--material', 'balloon', '--direction-convention', 'down follows gravity',
        '--mounting-notes', 'GelSight and GelNeuro face opposite sides of balloon; weight below',
        '--notes', notes,
    ]
    if protocol == 'load-cycle':
        print(
            'HOLD STILL 静止 → PRESS WITHOUT SLIDING 缓慢下拉 → HOLD CONTACT 保持 → '
            'RELEASE CONTACT 缓慢松回原位，保持两侧接触 → STOP MOTION 静止。',
            flush=True,
        )
    elif protocol == 'pull-rebound':
        print('看到 DO MIXED / DOWN 后轻拉砝码再松手，每次回稳后再做下一次。', flush=True)
    elif protocol == 'oscillation':
        print('看到 DO MIXED / DOWN 后让砝码小幅连续上下振荡。', flush=True)
    elif protocol == 'hold':
        print('在已标记的位移保持静止，直到录制结束。', flush=True)
    os.execv(sys.executable, command)


def verified_recording(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    quality = json.loads((path / 'quality.json').read_text())
    if manifest['status'] != 'complete' or not quality['passed']:
        raise ValueError(f'Recording must be complete and pass integrity checks: {path}')
    if not (path / 'gelsight' / 'frames.jsonl').is_file():
        raise ValueError(f'Recording is missing GelSight frames: {path}')
    return manifest


def encode_trial(path, background_path, output, history_frames, device):
    """Encode causal GelSight pairs and retain keys for joining raw observations."""
    from PIL import Image

    trial = verified_recording(path)
    manifest = verified_recording(background_path)
    if manifest['config']['labels']['task'] != 'background':
        raise ValueError('Select a verified unloaded --background recording')
    trial_camera = trial.get('sensors', {}).get('gelsight', {}).get('camera_name')
    background_camera = manifest.get('sensors', {}).get('gelsight', {}).get('camera_name')
    if trial_camera and background_camera and trial_camera != background_camera:
        raise ValueError('The trial and unloaded reference must use the same GelSight camera')
    references = list(rows(background_path / 'gelsight' / 'frames.jsonl'))
    if not references:
        raise ValueError(f'No unloaded images in {background_path}')
    reference = references[len(references) // 2]
    with Image.open(safe_path(background_path, reference['path'])) as frame:
        background = np.array(frame.convert('RGB'))

    if output.exists():
        raise FileExistsError(f'Features already exist: {output}')
    import torch

    from ...models.encoders import ImageBackbone, prepare_sparsh

    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = ImageBackbone('sparsh_dino_small', pretrained=True, frozen=True).to(device).eval()
    recent = deque(maxlen=history_frames + 1)
    features = []
    metadata = []
    inputs = []
    with torch.inference_mode():
        for frame, rgb in iter_frames(path):
            recent.append((frame, rgb))
            if len(recent) < recent.maxlen:
                continue
            old, old_rgb = recent[0]
            inputs.append(prepare_sparsh(rgb, old_rgb, background))
            metadata.append((frame['frame_id'], old['frame_id'],
                             frame['host_received_ns'], old['host_received_ns']))
            if len(inputs) == 4:
                features.append(model(torch.stack(inputs).to(device)).cpu().numpy())
                inputs.clear()
        if inputs:
            features.append(model(torch.stack(inputs).to(device)).cpu().numpy())

    if not features:
        raise ValueError(f'Trial has no causal frame pairs at stride {history_frames}')
    values = np.concatenate(features)
    if not np.isfinite(values).all():
        raise RuntimeError('Nonfinite Sparsh embedding')
    frame_ids, old_ids, host_ns, old_ns = np.asarray(metadata, dtype=np.int64).T
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(
            stream, features=values, frame_id=frame_ids, history_frame_id=old_ids,
            host_received_ns=host_ns, history_host_received_ns=old_ns,
            encoder='sparsh_dino_small', history_frames=history_frames,
            trial=str(path.resolve()), background_trial=str(background_path.resolve()),
            background_frame_id=reference['frame_id'],
        )
    print(f'Sparsh 特征 {values.shape[0]} 帧 × {values.shape[1]} 维，已保存 {output}', flush=True)
