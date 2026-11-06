"""Check all 3 fusion x 4 action head paths on synthetic inputs, CPU only."""
from dataclasses import replace
import json
from pathlib import Path
from neurofusion.training.pipeline import load_configuration, run

fusion,candidate = load_configuration('configs/pipeline.toml')
reports = []
for kind in ('plain','gated','attention'):
    for head in ('mlp','act','diffusion_unet','diffusion_transformer'):
        report = run(replace(fusion,kind=kind),replace(candidate,action_head=head),steps=1)
        assert all(proposal['franka'][6] == 0 and len(proposal['robotis']) == 1 for proposal in report['proposals'])
        reports.append(report)
        print(kind,head,'PASS',flush=True)
output = Path('outputs/pipeline-matrix.json')
output.parent.mkdir(exist_ok=True)
with output.open('x') as stream:
    json.dump(reports,stream,indent=2)
