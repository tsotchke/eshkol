#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import os
from unittest.mock import patch
import subprocess
import sys
import tempfile


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
    check_child_cache_policy(Path(__file__).resolve().parent / 'run_gate.py')
    base = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='bb-false-') as directory:
        compiler = Path(directory)/'compiler'
        compiler.write_text('#!'+sys.executable+'\nprint("PASS bb_maximality 16")\n')
        compiler.chmod(0o755)
        r = subprocess.run([sys.executable, str(base/'run_gate.py'), '--eshkol-bin', str(compiler)],
                           capture_output=True, text=True, timeout=30)
        if not r.returncode or 'Incomplete or false maximality receipt' not in r.stderr:
            raise AssertionError('Actual maximality gate accepted incomplete output')
    print('BB MAXIMALITY READINESS PASS: false native receipts rejected')


if __name__ == '__main__':
    main()
