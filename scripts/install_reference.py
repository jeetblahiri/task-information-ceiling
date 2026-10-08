#!/usr/bin/env python3
"""Restore the hash-verified published output subset; never overwrite a changed file."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    manifest = json.loads((ROOT / 'reference/manifest.json').read_text())
    archive = ROOT / 'reference/frozen_results.tar.gz'
    if sha(archive.read_bytes()) != manifest['archive_sha256']:
        raise RuntimeError('Reference archive hash mismatch')
    expected = manifest['files']
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        if {m.name for m in members} != set(expected) or len(members) != len(expected):
            raise RuntimeError('Archive members do not match the publication manifest')
        for member in members:
            relative = Path(member.name)
            if not member.isfile() or relative.is_absolute() or '..' in relative.parts:
                raise RuntimeError('Invalid archive member: ' + member.name)
            data = stream.extractfile(member).read()
            if sha(data) != expected[member.name]:
                raise RuntimeError('Reference file hash mismatch: ' + member.name)
            target = ROOT / relative
            if not target.resolve().is_relative_to(ROOT.resolve()):
                raise RuntimeError('Archive destination escapes repository: ' + member.name)
            if target.exists():
                if not target.is_file() or sha(target.read_bytes()) != expected[member.name]:
                    raise RuntimeError('Refusing to overwrite changed file: ' + member.name)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    print(f'Restored or verified {len(expected)} published reference files; large simulation arrays are not included.')

if __name__ == '__main__':
    main()
