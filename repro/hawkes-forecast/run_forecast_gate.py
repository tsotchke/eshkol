#!/usr/bin/env python3
"""Reduced jets must equal derivatives of the independently expanded full map."""
import argparse
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from oracle import full_forecast, source_tools

BASE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eshkol-bin', required=True)
    p.add_argument('--source-root', type=Path, default=BASE.parents[1])
    p.add_argument('--compile-linalg-source', action='store_true')
    p.add_argument('--results', type=Path)
    args = p.parse_args()
    root = args.source_root.resolve()
    env = dict(os.environ, ESHKOL_PATH=str(root/'lib'), ESHKOL_JIT_CACHE='0',ESHKOL_JIT_COMPILE_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='forecast-native-') as directory:
        work = Path(directory)

        def run(command):
            r = subprocess.run(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=240)
            if r.returncode:
                raise RuntimeError(r.stdout[-2000:]+r.stderr[-2000:])
            return r.stdout

        source = BASE/'forecast_test.esk'
        jit = run([args.eshkol_bin, '-r', str(source)])
        lines = jit.splitlines()
        if len(lines)!=17 or lines[-1]!='PASS forecast 23' or not lines[-2].startswith('SOURCE '):
            raise AssertionError('Incomplete forecast receipt')
        tools = source_tools(root)
        full_source = tools.read(lines[-2][7:])
        expected_contexts = [(w, point) for w in [F(0), F(1,4), F(1,2), F(3,4), F(1)]
                             for point in [[0,0,0,0], [1,2,1,0], [2,1,0,2]]]
        nonzero_second = 0
        for line, context in zip(lines[:15], expected_contexts):
            assert line.startswith('FORECAST ')
            w, point, mean, covariance_queries, covariance = tools.read(line[9:])
            assert (w, point)==context
            wanted, wanted_covariance, _ = full_forecast(root, full_source, point, w)
            assert mean==[wanted[0]] and covariance_queries==wanted and covariance==wanted_covariance
            nonzero_second += int(wanted[0][2]!=0)
        assert nonzero_second==15
        extra = [str(root/'lib/core/exact_linalg.esk')] if args.compile_linalg_source else []
        run([args.eshkol_bin, str(source), *extra, '-O', '0', '-o', str(work/'forecast')])
        assert run([str(work/'forecast')])==jit
        result = {'schema_version':1, 'engines':['jit','aot-O0'], 'checks_per_engine':23,
            'contexts':15, 'ambient_dimension':15, 'mean_dimension':4, 'covariance_dimension':11,
            'nonzero_mean_second_coefficients':nonzero_second, 'method':'one RK4 step, h=1/16',
            'oracle':'differentiate exact polynomials from the full 15-coordinate source model',
            'linalg_source_compiled_explicitly':args.compile_linalg_source,
            'stdout_sha256':hashlib.sha256(jit.encode()).hexdigest(),
            'compiler_sha256':hashlib.sha256(Path(args.eshkol_bin).read_bytes()).hexdigest(),
            'files_sha256':{str((BASE/name).resolve().relative_to(root)):hashlib.sha256((BASE/name).read_bytes()).hexdigest()
                for name in ['source.esk','forecast.esk','forecast_test.esk','oracle.py','run_forecast_gate.py']}}
        if args.results:
            args.results.write_text(json.dumps(result, indent=2)+'\n')
    print('FORECAST PASS: exact values and first/second coefficients, full 15 -> 4/11, JIT/AOT')


if __name__=='__main__':
    main()
