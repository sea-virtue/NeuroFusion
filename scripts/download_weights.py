"""Download pinned encoder-only weights; validate size and SHA256."""
from pathlib import Path
import hashlib
import json
import urllib.request

root = Path(__file__).resolve().parents[1]
target_dir = root / 'checkpoints/encoders'
target_dir.mkdir(parents=True, exist_ok=True)
items = json.loads((root / 'third_party/weights.lock.json').read_text())
for name, spec in items.items():
    target = target_dir / spec['file']
    if not target.exists():
        part = target.with_suffix(target.suffix + '.part')
        urllib.request.urlretrieve(spec['url'], part)
        if part.stat().st_size != spec['bytes']:
            raise RuntimeError(f'Unexpected file size for {name}: {part}')
        with part.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != spec['sha256']:
                raise RuntimeError(f'Checksum mismatch for {name}: {part}')
        part.replace(target)
    with target.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != spec['sha256']:
            raise RuntimeError(f'Checksum mismatch for existing {name}')
    print(name, target.stat().st_size, 'bytes', 'SHA256 verified')
