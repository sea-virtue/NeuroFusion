"""Versioned teacher trace sidecar; raw tactile recordings remain unchanged."""
from dataclasses import asdict
import json
from pathlib import Path
import time
from .slip import GripperCommand


class Episode:
    def __init__(self, path, config, *, synthetic, tactile_recording=None, metadata=None):
        self.path = Path(path)
        self.path.mkdir(parents=True,exist_ok=False)
        self.manifest = {'format':'neurofusion.slip_teacher','version':1,
                         'status':'recording','synthetic':synthetic,'outcome':'unreviewed',
                         'control_mode':'snn_immediate_reclose' if config.slide_seconds == 0 else 'snn_onset_wall_timer','distance_measured':False,
                         'action_units':'absolute_closure_fraction_0_to_1',
                         'config':asdict(config),'tactile_recording':str(Path(tactile_recording).resolve()) if tactile_recording else None,
                         'host_monotonic_anchor_ns':time.monotonic_ns(),
                         'metadata':metadata or {},'commands_sent':0}
        self.stream = (self.path/'trace.jsonl').open('x',encoding='utf-8')
        self.save()

    def save(self):
        temporary = self.path/'manifest.json.tmp'
        temporary.write_text(json.dumps(self.manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        temporary.replace(self.path/'manifest.json')

    def tick(self, controller, now_ns, feedback, prediction, sink=None, telemetry=None):
        """sink=None is shadow only. No inferred ACK or execution label."""
        decision = controller.step(now_ns,feedback,prediction)
        row = {'feedback':asdict(feedback),'snn':asdict(prediction) if prediction else None,
               'decision':decision,'sent_command':None,'receipt':None,'gripper_telemetry':telemetry}
        try:
            if decision['command'] and sink is not None:
                command = GripperCommand(**decision['command'])
                receipt = sink.command(command)
                row['sent_command'] = asdict(command)
                row['receipt'] = asdict(receipt)
                self.manifest['commands_sent'] += 1
        except BaseException as exc:
            row['error'] = repr(exc)
            self.manifest['status'] = 'failed'
            raise
        finally:
            self.stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            self.stream.flush()
        return decision

    def close(self, status, error=None):
        self.manifest['status'] = status
        self.manifest['error'] = error
        self.manifest['closed_host_ns'] = time.monotonic_ns()
        self.stream.close()
        self.save()


def bc_rows(path):
    """Explicitly reviewed real trials only; yields one-dimensional BC labels.

    Training the full eight-slot policy needs its own masked gripper loss; do not
    fabricate seven measured arm targets from a gripper-only demonstration.
    """
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    if (manifest.get('synthetic') or manifest.get('status') != 'finished' or
        manifest.get('outcome') != 'accepted' or not manifest.get('tactile_recording')):
        raise ValueError('BC requires an accepted real completed trial and raw tactile recording')
    raw = Path(manifest['tactile_recording'])
    raw_manifest = json.loads((raw/'manifest.json').read_text())
    quality = json.loads((raw/'quality.json').read_text())
    if raw_manifest.get('synthetic') or raw_manifest.get('status') != 'complete' or not quality.get('passed'):
        raise ValueError('Raw tactile recording has not passed its audit')
    for line in (path/'trace.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('sent_command') is not None and row.get('receipt') is not None and not row.get('error'):
            yield {'decision_host_ns':row['decision']['time_ns'],
                   'feedback':row['feedback'],'snn':row['snn'],
                   'target_robotis':[row['sent_command']['closure']],
                   'receipt':row['receipt'],'tactile_recording':str(raw),
                   'phase':row['decision']['phase']}
