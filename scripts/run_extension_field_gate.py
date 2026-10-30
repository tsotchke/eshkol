#!/usr/bin/env python3
"""Independent coefficient-array oracle and native JIT/AOT qualification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def mul(a, b, p, modulus):
    m = len(modulus) - 1
    out = [0] * (2 * m - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % p
    for i in range(len(out) - 1, m - 1, -1):
        for j in range(m):
            out[i - m + j] = (out[i - m + j] - out[i] * modulus[j]) % p
    return out[:m]


def power(a, k, p, modulus):
    out = [1] + [0] * (len(a) - 1)
    while k:
        if k & 1:
            out = mul(out, a, p, modulus)
        a = mul(a, a, p, modulus)
        k //= 2
    return out


def digits(i, p, m):
    return [(i // p**j) % p for j in range(m)]


def index(cs, p):
    return sum(c * p**i for i, c in enumerate(cs))


def minimal(a, p, modulus):
    orbit, x = [], a
    while x not in orbit:
        orbit.append(x)
        x = power(x, p, p, modulus)
    assert x == a
    zero = [0] * len(a)
    coefficients = [[1] + zero[1:]]
    for root in orbit:
        out = [zero[:] for _ in range(len(coefficients) + 1)]
        for j, c in enumerate(coefficients):
            product = mul(c, root, p, modulus)
            out[j] = [(u - v) % p for u, v in zip(out[j], product)]
            out[j + 1] = [(u + v) % p for u, v in zip(out[j + 1], c)]
        coefficients = out
    assert all(not any(c[1:]) for c in coefficients)
    return [c[0] for c in coefficients]


def expected_lines():
    lines = []
    for p, modulus in [(2, [1, 1, 1]), (3, [1, 0, 1]),
                       (2, [1, 1, 0, 0, 1]), (5, [2, 0, 1])]:
        m = len(modulus) - 1
        for i in range(p**m):
            a = digits(i, p, m)
            for j in range(p**m):
                b = digits(j, p, m)
                add = index([(u + v) % p for u, v in zip(a, b)], p)
                product = index(mul(a, b, p, modulus), p)
                lines.append(f'PAIR ({p} {m} {i} {j} {add} {product})')
            polynomial = ' '.join(map(str, minimal(a, p, modulus)))
            lines.append(f'MINPOLY ({p} {m} {i} ({polynomial}))')
    lines.append('PASS extension_field 1162')
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eshkol-bin', required=True)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--results', type=Path)
    args = parser.parse_args()
    root = args.source_root.resolve()
    env = dict(os.environ, ESHKOL_PATH=str(root / 'lib'), ESHKOL_JIT_CACHE='0')
    expected = expected_lines()
    with tempfile.TemporaryDirectory(prefix='extension-field-') as directory:
        binary = Path(directory) / 'field-test'
        source = root / 'tests/stdlib/extension_field_test.esk'

        def run(command):
            result = subprocess.run(command, env=env, cwd=root, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=240)
            if result.returncode:
                raise RuntimeError(result.stdout[-2000:] + result.stderr[-2000:])
            return result.stdout

        jit = run([args.eshkol_bin, '-r', str(source)])
        run([args.eshkol_bin, str(source), '-O', '2', '-o', str(binary)])
        aot = run([str(binary)])
        if jit.splitlines() != expected or aot.splitlines() != expected:
            raise AssertionError('Native field receipt differs from independent oracle')
        result = {'schema_version': 1, 'jit_executable_cache': 'disabled', 'checks_per_engine': 1162,
                  'pair_oracles': 978, 'minimal_polynomial_oracles': 54,
                  'engines': ['jit', 'aot-O2'],
                  'stdout_sha256': hashlib.sha256(jit.encode()).hexdigest(),
                  'compiler_sha256': hashlib.sha256(Path(args.eshkol_bin).read_bytes()).hexdigest(),
                  'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                      for name in ['lib/core/extension_field.esk', 'lib/core/ring.esk',
                                   'tests/stdlib/extension_field_test.esk', 'scripts/run_extension_field_gate.py']}}
        if args.results:
            args.results.write_text(json.dumps(result, indent=2) + '\n')
    print('EXTENSION FIELD PASS: 1162 checks/engine, 978 pair and 54 minimal-polynomial oracles')


if __name__ == '__main__':
    main()
