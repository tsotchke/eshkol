#!/usr/bin/env python3
"""Prove the actual readiness gate rejects equal but false native receipts."""
from pathlib import Path
import importlib.util
import os
from unittest.mock import patch
import subprocess
import sys
import tempfile

BASE = Path(__file__).resolve().parent



def check_child_cache_policy(work):
    spec = importlib.util.spec_from_file_location('bb_gate_boundary', BASE / 'run_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    class ChildBoundary(Exception):
        pass
    for inherited in (None, '1'):
        calls = []
        def intercept(command, **kwargs):
            calls.append((command, kwargs))
            raise ChildBoundary()
        with patch.dict(os.environ, {}, clear=False):
            if inherited is None:
                os.environ.pop('ESHKOL_JIT_CACHE', None)
            else:
                os.environ['ESHKOL_JIT_CACHE'] = inherited
            argv = ['run_gate.py', '--eshkol-bin', 'boundary-must-not-run', '--source-root', str(work)]
            with patch.object(sys, 'argv', argv), patch.object(gate.subprocess, 'run', side_effect=intercept):
                try:
                    gate.main()
                except ChildBoundary:
                    pass
                else:
                    raise AssertionError('Gate did not reach its first child boundary')
            if len(calls) != 1 or calls[0][0][1] != '-r':
                raise AssertionError('Expected only the first JIT child boundary')
            if calls[0][1]['env'].get('ESHKOL_JIT_CACHE') != '0':
                raise AssertionError('JIT cache must be forced off at the first child boundary')
            if os.environ.get('ESHKOL_JIT_CACHE') != inherited:
                raise AssertionError('Gate mutated its caller environment')
    print('BUSY BEAVER ENVIRONMENT CONTROLS PASS: unset/enabled callers forced off; no compiler or BB workload')


def main():
    with tempfile.TemporaryDirectory(prefix='bb-readiness-') as work:
        check_child_cache_policy(work)
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
