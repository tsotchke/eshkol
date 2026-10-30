#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='resource-false-') as directory:
        compiler = Path(directory)/'compiler'
        compiler.write_text('#!'+sys.executable+'\nprint("PASS sblc_resource 84")\n')
        compiler.chmod(0o755)
        result = subprocess.run([sys.executable, str(root/'scripts/run_sblc_resource_gate.py'),
                                 '--eshkol-bin', str(compiler)], capture_output=True, text=True, timeout=30)
        if not result.returncode or 'receipt is incomplete' not in result.stderr:
            raise AssertionError('Resource gate accepted incomplete native output')
    print('SBLC RESOURCE READINESS PASS: false native receipts rejected')


if __name__ == '__main__':
    main()
