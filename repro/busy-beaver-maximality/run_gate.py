#!/usr/bin/env python3
"""Replay complete proof data, missing-proof control, and independent class."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from reference import whole_class

BASE = Path(__file__).resolve().parent
EXPECTED = ('BB2 (2 20736 20736 0 9784 10952 6 6 4 4 445 0 0)\n'
            'BB3 (3 16777216 16777216 0 7571840 9205376 21 21 6 6 115409 512 216)\n'
            'WITHDRAWN (3 16777216 16777215 1 7571840 9205375 21 unknown 6 unknown 115409 511 216)\n'
            'PASS bb_maximality 16\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eshkol-bin', required=True)
    p.add_argument('--source-root', type=Path, default=BASE.parents[1])
    p.add_argument('--results', type=Path)
    args = p.parse_args()
    env = dict(os.environ, ESHKOL_PATH=str(args.source_root.resolve()/'lib'), ESHKOL_JIT_CACHE='0')
    with tempfile.TemporaryDirectory(prefix='bb-maximality-') as directory:
        work = Path(directory)
        for name in ('bb2-cover.sexp', 'bb3-cover.sexp'):
            shutil.copyfile(BASE/name, work/name)
        lines = (BASE/'bb3-cover.sexp').read_text().splitlines()
        regular = next(i for i, line in enumerate(lines) if line.startswith('(regular '))
        lines[regular] = '(unknown)'
        (work/'bb3-withdrawn.sexp').write_text('\n'.join(lines)+'\n')

        def run(command):
            r = subprocess.run(command, cwd=work, env=env, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=240)
            if r.returncode:
                raise RuntimeError(r.stdout[-2000:]+r.stderr[-2000:])
            return r.stdout

        source = str(BASE/'maximality_test.esk')
        jit = run([args.eshkol_bin, '-r', source])
        if jit != EXPECTED:
            raise AssertionError('Incomplete or false maximality receipt: '+jit[-700:])
        run([args.eshkol_bin, source, '-O', '2', '-o', str(work/'maximality')])
        aot = run([str(work/'maximality')])
        if aot != EXPECTED:
            raise AssertionError('AOT maximality receipt differs')
        independent = whole_class()
        result = {'schema_version': 1, 'jit_executable_cache': 'disabled', 'engines': ['jit', 'aot-O2'], 'checks_per_engine': 16,
                  'two_state_independent': independent, 'three_state_machines': 16777216,
                  'three_state_halts': 7571840, 'three_state_nonhalts': 9205376,
                  'three_state_unknown': 0, 'time': 21, 'marks': 6,
                  'partition_nodes': 115409, 'regular_certificates': 512, 'ngram_certificates': 216,
                  'withdrawn_unknown': 1, 'withdrawn_upper_bounds': ['unknown', 'unknown'],
                  'stdout_sha256': hashlib.sha256(jit.encode()).hexdigest(),
                  'compiler_sha256': hashlib.sha256(Path(args.eshkol_bin).read_bytes()).hexdigest(),
                  'files_sha256': {name: hashlib.sha256((BASE/name).read_bytes()).hexdigest()
                    for name in ['checker.esk', 'maximality_test.esk', 'bb2-cover.sexp', 'bb3-cover.sexp',
                                 'reference.py', 'run_gate.py']}}
        if args.results:
            args.results.write_text(json.dumps(result, indent=2)+'\n')
    print('BB MAXIMALITY PASS: complete 2/3-state classes, 21/6, missing-proof upper-bound withdrawal, JIT/AOT')


if __name__ == '__main__':
    main()
