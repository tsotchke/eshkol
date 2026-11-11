#!/usr/bin/env python3
"""Prove the actual readiness gate rejects a successful but wrong backend."""
from pathlib import Path
import subprocess
import sys
import tempfile

BASE = Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory(prefix='staging-readiness-') as work:
        backend = Path(work) / 'wrong-backend'
        backend.write_text('#!' + sys.executable + '\nimport sys\nsys.stdout.buffer.write(b"\\0")\n')
        backend.chmod(0o755)
        result = subprocess.run(
            [sys.executable, str(BASE / 'run_gate.py'), '--engine', 'reference',
             '--reference-bin', str(backend), '--case', 'm0-infer0'],
            cwd=BASE, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            raise AssertionError('Readiness gate accepted wrong returned bytes')
        expected = "AssertionError: ('reference', 'm0-infer0', 'generic', [0],"
        if expected not in result.stderr:
            raise AssertionError('Unexpected infrastructure failure: ' + result.stderr[-2000:])
    print('STAGING READINESS SELF-TEST PASS: wrong returned bytes rejected')


if __name__ == '__main__':
    main()
