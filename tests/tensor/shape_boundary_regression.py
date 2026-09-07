#!/usr/bin/env python3
"""Exercise shape input/refusal boundaries with a real source-engine runner."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runner', required=True)
    p.add_argument('--engine', choices=('jit', 'vm'), required=True)
    p.add_argument('--artifact-root', required=True)
    args = p.parse_args()
    root = Path(args.artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cases = [
        ('fractional-list', '(make-tensor (list 2.5 2) 0.0)', True, {}),
        ('fractional-vector', '(make-tensor #(2.5 2) 0.0)', True, {}),
        ('non-numeric', '(make-tensor (list #t 2) 0.0)', True, {}),
        ('reshape-count', '(reshape #(1.0 2.0 3.0) 2 2)', True, {}),
        ('reshape-empty-count', '(reshape #(1.0 2.0 3.0) 0 3)', True, {}),
        ('negative', '(make-tensor (list -1 2) 0.0)', True, {}),
        ('overflow', '(make-tensor (list 9223372036854775807 2) 0.0)', True, {}),
        ('limit', '(make-tensor (list 2 3) 0.0)', True,
         {'ESHKOL_MAX_TENSOR_ELEMS': '5'}),
        ('limit-exact', '(make-tensor (list 2 3) 0.0)', False,
         {'ESHKOL_MAX_TENSOR_ELEMS': '6'}),
        ('empty', '(make-tensor (list 0 3) 0.0)', False, {}),
    ]
    failures = []
    with tempfile.TemporaryDirectory(prefix='shape-boundary-', dir=root) as temp:
        for name, expression, reject, overrides in cases:
            src = Path(temp) / (name + '.esk')
            # Guard success is evidence of a language-level refusal, not merely
            # an abnormal subprocess exit or a compiler/linker failure.
            src.write_text('(guard (e (#t (display "REFUSED") (newline)))\n'
                           '  ' + expression + '\n'
                           '  (display "ACCEPTED") (newline))\n')
            cmd = [str(Path(args.runner).resolve())]
            if args.engine == 'jit':
                cmd.append('-r')
            env = os.environ.copy()
            env.pop('ESHKOL_MAX_TENSOR_ELEMS', None)
            env.update(overrides)
            env['ESHKOL_VM_NO_DISASM'] = '1'
            try:
                r = subprocess.run(cmd + [str(src)], capture_output=True,
                                   text=True, timeout=30, env=env)
            except subprocess.TimeoutExpired:
                failures.append(name + ': timeout')
                continue
            markers = [x.strip() for x in r.stdout.splitlines()
                       if x.strip() in ('REFUSED', 'ACCEPTED')]
            expected = 'REFUSED' if reject else 'ACCEPTED'
            hard_limit = (name == 'limit' and args.engine == 'jit' and
                          r.returncode == 122 and not markers and
                          'ESHKOL_MAX_TENSOR_ELEMS' in r.stderr)
            if not hard_limit and (r.returncode != 0 or markers != [expected]):
                failures.append(f'{name}: rc={r.returncode}, markers={markers}, '
                                f'stderr={r.stderr[-300:]}')
    for failure in failures:
        print('FAIL:', failure)
    if not failures:
        print(f'PASS: shape boundaries ({args.engine}, {len(cases)} cases)')
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
