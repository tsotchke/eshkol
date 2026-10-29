#!/usr/bin/env python3
"""Compare exact JIT/AOT proofs with a full independent raw-machine execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

BASE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eshkol-bin', required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--results', type=Path)
    args = p.parse_args()
    env = dict(os.environ, ESHKOL_PATH=str(args.source_root.resolve() / 'lib'))
    with tempfile.TemporaryDirectory(prefix='bb5-gate-') as work:
        work = Path(work)
        def run(command):
            r = subprocess.run(command, env=env, cwd=BASE, capture_output=True,
                               text=True, timeout=180)
            if r.returncode:
                raise RuntimeError(r.stdout + r.stderr)
            return r.stdout
        source = str(BASE / 'bb5_certificate_test.esk')
        jit = run([args.eshkol_bin, '-r', source])
        run([args.eshkol_bin, source, '-o', str(work / 'bb5')])
        aot = run([str(work / 'bb5')])
        assert jit == aot, 'JIT/AOT bytes differ'
        assert 'BB5-CERTIFICATE (47176870 -12242 4098 15 4094)\n' in jit
        assert 'PASS bb5_certificate 34\n' in jit, jit[-300:]
        queries = [list(map(int, line[7:-1].split())) for line in jit.splitlines() if line.startswith('(QUERY ')]
        assert len(queries) == 18
        for n, k, r, nxt, cost, head, ones in queries:
            assert n == 3*k+r and r in (0, 1, 2)
            expected = [(5*k+6, 5*k*k+19*k+15, -2*k-3, 5*k+6),
                        (5*k+9, 5*k*k+25*k+27, -2*k-5, 5*k+9),
                        (-1, 6*k+12, 0, k+4)][r]
            assert (nxt, cost, head, ones) == expected
        run(['cc', '-O2', '-std=c11', '-Wall', '-Wextra', '-Werror',
             str(BASE / 'direct_reference.c'), '-o', str(work / 'reference')])
        started = time.monotonic()
        summary = run([str(work / 'reference'), str(work / 'tape')]).strip()
        reference_seconds = time.monotonic() - started
        assert summary == '47176870 -12242 5 4098', summary
        expected = bytearray(131072)
        head = len(expected)//2 - 12242
        expected[head-1] = expected[head+1] = 1
        # Halt: ... 1 (Z0) 1 (001)^(4094+1) 1 ...
        start = head + 2
        expected[start:start+3*4095] = bytes([0, 0, 1])*4095
        expected[start+3*4095] = 1
        tape = (work/'tape').read_bytes()
        assert tape == expected, 'Complete final tape differs'
        result = {'schema_version': 1, 'engines': ['jit', 'aot'], 'checks_per_engine': 34,
                  'bb6_scope': '12-variable polynomial embedding; 16 dense/structured powers',
                  'query_count': 18, 'raw_steps': 47176870, 'macros': 15,
                  'head': -12242, 'ones': 4098, 'tape_cells_compared': len(tape),
                  'tape_sha256': hashlib.sha256(tape).hexdigest(),
                  'stdout_sha256': hashlib.sha256(jit.encode()).hexdigest(),
                  'reference_seconds': reference_seconds,
                  'runner_sha256': hashlib.sha256(Path(args.eshkol_bin).read_bytes()).hexdigest(),
                  'files_sha256': {name: hashlib.sha256((BASE/name).read_bytes()).hexdigest()
                      for name in ['bb5_certificate.esk', 'bb6_cost_algebra.esk',
                                   'bb5_certificate_test.esk', 'direct_reference.c', 'run_gate.py']},
                  'source_sha256': hashlib.sha256((BASE/'bb5_certificate.esk').read_bytes()).hexdigest(),
                  'symbolic_sha256': hashlib.sha256((args.source_root/'lib/core/symbolic.esk').read_bytes()).hexdigest(),
                  'ring_sha256': hashlib.sha256((args.source_root/'lib/core/ring.esk').read_bytes()).hexdigest()}
        if args.results:
            args.results.write_text(json.dumps(result, indent=2)+'\n')
        print('BB5 PASS: exact JIT/AOT, 18 queries, complete 47,176,870-step reference tape')


if __name__ == '__main__':
    main()
