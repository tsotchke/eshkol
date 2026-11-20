#!/usr/bin/env python3
"""Replay staging, rebuild binaries, and compare raw-token executions exactly."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import tempfile
import time
try:
    import tomllib
except ImportError:
    import tomli as tomllib

BASE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fraction_bytes(q):
    return [int(q < 0)] + list(abs(q.numerator).to_bytes(64, 'little')) + list(q.denominator.to_bytes(64, 'little'))


def oracle(case, spec):
    frame = case['input']
    p, context, vocabulary = spec['parameters'], spec['context'], spec['vocabulary']
    mode = spec['mode']
    if (len(frame) != p+context+int(bool(mode)) or any(x > 128 for x in frame[:p])
            or any(x >= vocabulary for x in frame[p:])):
        return [255]
    model = [Fraction(x//2 if x % 2 == 0 else -(x+1)//2, 16) for x in frame[:p]]
    tokens = frame[p:p+context]
    target = frame[-1] if mode else 0
    point = model + [Fraction(token == i) for token in tokens for i in range(vocabulary)]
    point += [Fraction(bool(mode) and target == i) for i in range(vocabulary)]
    values = []
    graph = spec['graph'] if mode == 0 else spec['graph'][:spec['objective']+1]
    for op, left, right in graph:
        if op == 0:
            value, gradient = point[left], [Fraction(i == left) for i in range(p)]
        elif op == 1:
            value = Fraction((left+1)//2 * (-1 if left % 2 else 1), right)
            gradient = [Fraction(0)]*p
        else:
            a, da = values[left]
            b, db = values[right]
            if op == 2:
                value, gradient = a+b, [x+y for x, y in zip(da, db)]
            elif op == 3:
                value, gradient = a*b, [x*b+a*y for x, y in zip(da, db)]
            else:
                assert op == 4
                if b == 0:
                    return [1]
                value, gradient = a/b, [(x*b-a*y)/(b*b) for x, y in zip(da, db)]
        values.append((value, gradient))
    loss, gradient = values[-1]
    if mode == 0:
        answers = [values[i][0] for i in spec['outputs']]
        prefix = [0, max(range(len(answers)), key=lambda i: answers[i])]
    elif mode == 1:
        answers, prefix = [loss]+gradient, [0]
    else:
        def quantize(x):
            shifted = abs(x)*16 + Fraction(1, 2)
            return Fraction((-1 if x < 0 else 1)*(shifted.numerator//shifted.denominator), 16)
        answers = [quantize(w-g/16) for w, g in zip(model, gradient)]
        if any(abs(w) > 4 for w in answers):
            return [1]
        prefix = [0, 0, 1]
    return prefix+[b for q in answers for b in fraction_bytes(q)]


def command(argv, payload, cwd, env, timeout):
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (2*1024**3,)*2)
        resource.setrlimit(resource.RLIMIT_STACK, (64*1024**2,)*2)
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout)+1,)*2)
    start = time.monotonic()
    child = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             start_new_session=True, preexec_fn=limits)
    try:
        out, err = child.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.communicate()
        raise RuntimeError('Wall budget exhausted; execution is not completion')
    if child.returncode:
        raise RuntimeError(f'Child exit {child.returncode}: '+err.decode(errors='replace')[-3000:])
    return out, err, time.monotonic()-start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=['reference', 'jit', 'aot', 'all'], default='all')
    parser.add_argument('--eshkol-bin')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--reference-bin', type=Path)
    parser.add_argument('--case', action='append')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--optimization', type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument('--results', type=Path)
    args = parser.parse_args()
    d = tomllib.loads((BASE/'fixtures.toml').read_text())
    specs = {x['id']: x for x in d['topology']}
    cases = [c for c in d['case'] if not args.case or c['id'] in args.case]
    assert cases and (not args.case or set(args.case) == {c['id'] for c in cases})
    for p in d['program']:
        bits = (BASE/(p['id']+'.blc')).read_text().strip()
        packed = (BASE/(p['id']+'.blc8')).read_bytes()
        assert set(bits) <= {'0', '1'} and len(bits) == p['logical_bits']
        assert len(packed) == p['packed_bytes'] and packed == int(bits.ljust(len(packed)*8, '0'), 2).to_bytes(len(packed), 'big')
    producer = subprocess.run(['sbcl', '--noinform', '--script', str(BASE/'producer.lisp')],
                              capture_output=True, text=True, timeout=30)
    assert producer.returncode == 0, producer.stderr[-3000:]
    assert 'six residuals and seven closed binary rebuilds' in producer.stdout
    engines = ['reference', 'jit', 'aot'] if args.engine == 'all' else [args.engine]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               ESHKOL_JIT_CACHE='0', ESHKOL_JIT_COMPILE_THREADS='1')
    if args.source_root:
        env['ESHKOL_PATH'] = str(args.source_root.resolve()/'lib')
    rows, parity = [], {}
    with tempfile.TemporaryDirectory(prefix='staging-gate-') as temp:
        temp = Path(temp)
        for engine in engines:
            for case in cases:
                expected = oracle(case, specs[case['program']])
                assert case['expected_refusal'] is False or expected == case['expected_refusal']
                for route in ['generic', 'residual']:
                    name = 'generic' if route == 'generic' else case['program']
                    frame = case['generic_input'] if route == 'generic' else case['input']
                    bits = (BASE/(name+'.blc')).read_text().strip()
                    if engine == 'reference':
                        assert args.reference_bin
                        out, err, elapsed = command([str(args.reference_bin.resolve()), '-x', '-x', '-x', '-x'],
                            (BASE/(name+'.blc8')).read_bytes()+bytes(frame), temp, env, args.timeout)
                        actual = list(out)
                        steps = re.search(rb'\bsteps (\d+)', err)
                        counters = {'graph_steps': int(steps[1])} if steps else {}
                    else:
                        assert args.eshkol_bin and args.source_root
                        probe = temp/'probe.esk'
                        text = ('(require core.blc_bytes)\n'
                          '(write (blc-run-bytes (blc-decode '+json.dumps(bits)+") '"+
                          '('+' '.join(map(str, frame))+') 1000000000 2000000 4096)) (newline)\n')
                        probe.write_text(text)
                        prefix = [args.eshkol_bin, '--no-stdlib', '-I', str(args.source_root.resolve()/'lib'), '-O', str(args.optimization)]
                        if engine == 'jit':
                            out, err, elapsed = command(prefix+['-r', str(probe)], b'', temp, env, args.timeout)
                        else:
                            executable = temp/'probe'
                            command(prefix+[str(probe), '-o', str(executable)], b'', temp, env, args.timeout)
                            out, err, elapsed = command([str(executable)], b'', temp, env, args.timeout)
                        match = re.fullmatch(rb'\(normal \(([\d ]*)\)((?: \d+){6})\)\n', out)
                        assert match, 'Incomplete or malformed byte-observer receipt: '+out[:300].decode(errors='replace')
                        actual = [int(x) for x in match[1].split()]
                        counters = {'observer_counters': [int(x) for x in match[2].split()]}
                        key = case['id'], route
                        if key in parity:
                            assert parity[key] == out, 'JIT/AOT receipt differs'
                        parity[key] = out
                    assert actual == expected, (engine, case['id'], route, actual[:20], expected[:20])
                    rows.append({'engine': engine, 'case': case['id'], 'route': route,
                                 'output_sha256': sha(bytes(actual)), 'receipt_sha256': sha(out),
                                 'seconds': elapsed, **counters})
                    print(engine, case['id'], route, 'PASS', flush=True)
    result = {'schema_version': 1, 'producer': 'six exact residuals; seven exact binary rebuilds',
              'case_count': len(cases), 'execution_count': len(rows), 'engines': engines,
              'optimization': args.optimization,
              'fixture_sha256': sha((BASE/'fixtures.toml').read_bytes()), 'executions': rows}
    if args.results:
        args.results.write_text(json.dumps(result, indent=2)+'\n')
    print('STAGING GATE PASS', len(rows), 'exact executions')


if __name__ == '__main__':
    main()
