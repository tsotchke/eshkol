#!/usr/bin/env python3
"""Prove the actual readiness gate rejects equal but false native receipts."""
from pathlib import Path
import subprocess
import sys
import tempfile

BASE = Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory(prefix='bb-readiness-') as work:
        backend = Path(work) / 'wrong-backend'
        emitted = '#!' + sys.executable + '\nprint("not-a-certificate")\n'
        backend.write_text(
            '#!' + sys.executable + '\nimport sys\nfrom pathlib import Path\n'
            'if "-o" in sys.argv:\n'
            '    output = Path(sys.argv[sys.argv.index("-o") + 1])\n'
            '    output.write_text(' + repr(emitted) + ')\n'
            '    output.chmod(0o755)\n'
            'else:\n    print("not-a-certificate")\n')
        backend.chmod(0o755)
        result = subprocess.run(
            [sys.executable, str(BASE / 'run_gate.py'), '--eshkol-bin', str(backend),
             '--source-root', work], cwd=BASE, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            raise AssertionError('Readiness gate accepted false native receipts')
        if "assert 'BB5-CERTIFICATE (47176870 -12242 4098 15 4094)\\n' in jit" not in result.stderr:
            raise AssertionError('Unexpected infrastructure failure: ' + result.stderr[-2000:])
    print('BUSY BEAVER READINESS SELF-TEST PASS: false native receipts rejected')


if __name__ == '__main__':
    main()
