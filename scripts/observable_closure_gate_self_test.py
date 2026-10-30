#!/usr/bin/env python3
"""Exercise the real gate with equal but incomplete compiler output."""
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='observable-false-') as directory:
        compiler = Path(directory) / 'compiler'
        compiler.write_text('#!' + sys.executable + '\nprint("PASS observable_closure 159")\n')
        compiler.chmod(0o755)
        r = subprocess.run([sys.executable, str(root / 'scripts/run_observable_closure_gate.py'),
                            '--eshkol-bin', str(compiler)], capture_output=True, text=True, timeout=30)
        if not r.returncode or 'receipt is incomplete' not in r.stderr:
            raise AssertionError('The actual gate did not reject incomplete native receipts')
    print('OBSERVABLE READINESS PASS: false native receipts rejected')


if __name__ == '__main__':
    main()
