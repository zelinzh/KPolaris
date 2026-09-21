#!/usr/bin/env python3
"""Compatibility entry point for the shared, read-only KPolaris doctor."""
import argparse
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('repo', nargs='?', type=Path, default=Path('.'))
p.add_argument('--json', action='store_true')
p.add_argument('--backend', choices=('cpu', 'openmp', 'cuda'), default='cpu')
a = p.parse_args()
entry = a.repo.resolve() / 'scripts/kpolaris.py'
if not entry.is_file():
    p.error('repo must point to a KPolaris source tree containing scripts/kpolaris.py')
raise SystemExit(subprocess.call([sys.executable, str(entry), 'doctor', f'--backend={a.backend}', *(['--json'] if a.json else [])]))
