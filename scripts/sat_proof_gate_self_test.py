#!/usr/bin/env python3
"""The real SAT gate must force uncached children and reject false receipts."""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

def check_child_cache_policy(gate_path):
    spec = importlib.util.spec_from_file_location('native_gate_boundary', gate_path)
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
            argv = [str(gate_path), '--eshkol-bin', 'boundary-must-not-run']
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
    print('ENVIRONMENT CONTROLS PASS: unset/enabled callers forced off; first child intercepted')


def main():
    root = Path(__file__).resolve().parents[1]
    gate = root / 'scripts/run_sat_proof_gate.py'
    check_child_cache_policy(gate)
    with tempfile.TemporaryDirectory(prefix='sat-proof-readiness-') as directory:
        compiler = Path(directory) / 'compiler'
        emitted = '#!' + sys.executable + '\nprint("not-a-real-receipt")\n'
        compiler.write_text('#!' + sys.executable + '\nimport sys\nfrom pathlib import Path\n'
            'if "-o" in sys.argv:\n'
            ' p=Path(sys.argv[sys.argv.index("-o")+1]);p.write_text(' + repr(emitted) + ');p.chmod(0o755)\n'
            'else: print("not-a-real-receipt")\n')
        compiler.chmod(0o755)
        result = subprocess.run([sys.executable, str(gate), '--eshkol-bin', str(compiler)],
                                capture_output=True, text=True, timeout=30)
        if not result.returncode or 'Existing SAT receipt incomplete' not in result.stderr:
            raise AssertionError('Gate accepted false output or failed for infrastructure: ' + result.stderr[-1200:])
    print('SAT PROOF READINESS PASS: equal false native receipts rejected')


if __name__ == '__main__':
    main()
