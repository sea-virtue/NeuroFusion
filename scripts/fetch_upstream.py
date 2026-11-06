"""Restore exact official sources in third_party/sources.lock.json."""
from pathlib import Path
import json
import subprocess

root = Path(__file__).resolve().parents[1]
sources = json.loads((root / 'third_party/sources.lock.json').read_text())
for name, spec in sources.items():
    target = root / spec['path']
    if not target.resolve().is_relative_to(root / 'third_party'):
        raise ValueError(f'Checkout must stay inside third_party: {target}')
    if target.exists():
        actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(target), 'status', '--porcelain'], text=True).strip()
        if actual != spec['commit'] or dirty:
            raise RuntimeError(f'{name}: existing checkout differs; preserve it and resolve manually')
    else:
        subprocess.run(['git', 'init', str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'remote', 'add', 'origin', spec['url']], check=True)
        subprocess.run(['git', '-C', str(target), 'fetch', '--depth', '1', 'origin', spec['commit']], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    print(name, spec['commit'])
