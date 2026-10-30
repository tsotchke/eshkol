#!/usr/bin/env python3
"""Independent exact matrix/polynomial oracle and producer-free native replay."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def read(source):
    tokens = re.findall(r'#\(|[()]|[^\s()]+', source)
    position = 0

    def parse():
        nonlocal position
        token = tokens[position]
        position += 1
        if token in ('(', '#('):
            result = []
            while tokens[position] != ')':
                result.append(parse())
            position += 1
            return result
        if re.fullmatch(r'-?\d+', token):
            return int(token)
        if re.fullmatch(r'-?\d+/\d+', token):
            return Fraction(token)
        return token

    result = parse()
    if position != len(tokens):
        raise ValueError('Trailing source data')
    return result


def rank(rows, width):
    a = [[Fraction(x) for x in row] for row in rows]
    pivot = 0
    for j in range(width):
        found = next((i for i in range(pivot, len(a)) if a[i][j]), None)
        if found is None:
            continue
        a[pivot], a[found] = a[found], a[pivot]
        scale = a[pivot][j]
        a[pivot] = [x / scale for x in a[pivot]]
        for i in range(len(a)):
            if i != pivot:
                scale = a[i][j]
                a[i] = [x - scale * y for x, y in zip(a[i], a[pivot])]
        pivot += 1
    return pivot


def act(row, matrix, width):
    return [sum((x * line[j] for x, line in zip(row, matrix)), Fraction(0)) for j in range(width)]


def closure_dimension(queries, actions):
    width = len(queries[0])
    basis = []
    for q in queries:
        if rank(basis + [q], width) > len(basis):
            basis.append(q)
    while True:
        previous = len(basis)
        candidates = [act(row, matrix, width) for row in basis[:] for matrix in actions]
        for candidate in candidates:
            if rank(basis + [candidate], width) > len(basis):
                basis.append(candidate)
        if len(basis) == previous:
            return len(basis)


def add(a, b):
    out = dict(a)
    for e, c in b.items():
        out[e] = out.get(e, 0) + c
    return {e: c for e, c in out.items() if c}


def scale(a, c):
    return {e: c * x for e, x in a.items() if c * x}


def mul(a, b):
    out = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            e = tuple(x + y for x, y in zip(ea, eb))
            out[e] = out.get(e, 0) + ca * cb
    return {e: c for e, c in out.items() if c}


def power(a, n, zero):
    out = {zero: Fraction(1)}
    for _ in range(n):
        out = mul(out, a)
    return out


def polynomial(expression, names):
    zero = (0,) * len(names)
    if isinstance(expression, (int, Fraction)):
        return {zero: Fraction(expression)} if expression else {}
    if isinstance(expression, str):
        return {tuple(int(name == expression) for name in names): Fraction(1)}
    op, *args = expression
    if op == 'expt':
        return power(polynomial(args[0], names), args[1], zero)
    values = [polynomial(x, names) for x in args]
    if op == '*':
        out = {zero: Fraction(1)}
        for x in values:
            out = mul(out, x)
        return out
    out = {} if op == '+' else values.pop(0)
    if op == '-' and not values:
        return scale(out, -1)
    for x in values:
        out = add(out, x if op == '+' else scale(x, -1))
    return out


def source_oracle(source):
    spec = {row[0]: row[1:] for row in source}
    variables, parameters, basis = spec['variables'], spec['parameters'], spec['basis']
    names = variables + parameters
    zero = (0,) * len(names)
    states = len(variables)
    drift = [polynomial(x, names) for x in spec['drift']]
    images = []
    for powers in basis:
        monomial = {tuple(powers) + (0,) * len(parameters): Fraction(1)}
        out = {}
        for i, exponent in enumerate(powers):
            if exponent:
                e = list(powers) + [0] * len(parameters)
                e[i] -= 1
                out = add(out, mul(drift[i], {tuple(e): Fraction(exponent)}))
        for rate, updates in spec['jumps']:
            changed = {zero: Fraction(1)}
            for e, update in zip(powers, updates):
                changed = mul(changed, power(polynomial(update, names), e, zero))
            out = add(out, mul(polynomial(rate, names), add(changed, scale(monomial, -1))))
        images.append(out)
    family = sorted({e[states:] for image in images for e in image} or {(0,) * len(parameters)})
    matrices = []
    for parameter in family:
        matrices.append([[image.get(tuple(state) + parameter, 0) for state in basis] for image in images])
    queries = []
    for query in spec['observations']:
        p = polynomial(query, names)
        assert all(e[:states] in map(tuple, basis) and not any(e[states:]) for e in p)
        queries.append([p.get(tuple(state) + (0,) * len(parameters), 0) for state in basis])
    return family, matrices, queries


def check_certificate(source, certificate):
    family, matrices, queries = source_oracle(source)
    tag, powers, actual_queries, actions, linear = certificate
    assert tag == 'generator-closure-v1'
    assert [tuple(p) for p in powers] == family and actual_queries == queries
    assert [a[1:] for a in actions] == matrices
    tag, n, basis, coordinates, reduced, origins, inverse = linear
    assert tag == 'observable-closure-v1' and n == len(queries[0])
    r = len(basis)
    assert r == rank(basis, n) == closure_dimension(queries, matrices) == 4
    for row, coordinate in zip(queries, coordinates):
        assert row == act(coordinate, basis, n)
    for i, row in enumerate(basis):
        assert act(row, inverse, r) == [int(i == j) for j in range(r)]
        qid, word = origins[i]
        replayed = queries[qid]
        for a in word:
            replayed = act(replayed, matrices[a], n)
        assert replayed == row
        for matrix, small in zip(matrices, reduced):
            assert act(row, matrix, n) == act(small[i + 1], basis, n)
    covariance = [row[:] for row in source]
    covariance[-1] = ['observations', 'n1', 'n2', ['*', 'n1', 'n2']]
    _, gs, qs = source_oracle(covariance)
    assert closure_dimension(qs, gs) == 11


def disabled_source(source):
    forbidden = {'compile-observable-closure', 'compile-generator-closure', 'oc-act',
                 'oc-solve', 'oc-coordinates', 'oc-source-image', 'oc-generator-lowering'}
    starts, forms, level, string, comment, escape = [], [], 0, False, False, False
    for i, c in enumerate(source):
        if comment:
            comment = c != '\n'
            continue
        if c == ';' and not string:
            comment = True
            continue
        if c == '"' and not escape:
            string = not string
        if not string:
            if c == '(':
                if level == 0:
                    starts.append(i)
                level += 1
            elif c == ')':
                level -= 1
                if level == 0:
                    forms.append(source[starts[-1]:i + 1])
        escape = c == '\\' and not escape
    assert level == 0
    replaced = set()
    for i, form in enumerate(forms):
        match = re.match(r'\(define\s+\(([^\s()]+)', form)
        if match and match[1] in forbidden:
            forms[i] = form[:form.index(')') + 1] + ' (error "Producer disabled during replay"))'
            replaced.add(match[1])
    assert replaced == forbidden
    return '\n'.join(forms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eshkol-bin', required=True)
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--results', type=Path)
    p.add_argument('--compile-linalg-source', action='store_true',
                   help='Explicitly compile this checkout\'s linalg module when a packaged stdlib object predates exact-affine-solve')
    args = p.parse_args()
    root = args.source_root.resolve()
    env = dict(os.environ, ESHKOL_PATH=str(root / 'lib'), ESHKOL_JIT_CACHE='0')
    with tempfile.TemporaryDirectory(prefix='observable-gate-') as directory:
        work = Path(directory)

        def run(command):
            r = subprocess.run(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=240)
            if r.returncode:
                raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
            return r.stdout

        source = root / 'tests/stdlib/observable_closure_test.esk'
        jit = run([args.eshkol_bin, '-r', str(source)])
        lines = jit.splitlines()
        if len(lines) != 68 or lines[-1] != 'PASS observable_closure 159':
            raise AssertionError('Native closure receipt is incomplete')
        expected_spans = []
        for mask in range(16):
            matrix = [[(mask >> j) & 1 for j in (0, 1)], [(mask >> j) & 1 for j in (2, 3)]]
            for code in range(4):
                dimension = closure_dimension([[(code >> j) & 1 for j in (0, 1)]], [matrix])
                expected_spans.append(f'SPAN ({mask} {code} {dimension})')
        assert lines[:64] == expected_spans and lines[64] == 'HAWKES (15 4 11)'
        assert lines[65].startswith('SOURCE ') and lines[66].startswith('CERTIFICATE ')
        check_certificate(read(lines[65][7:]), read(lines[66][12:]))
        binary = work / 'observable-test'
        extra = [str(root / 'lib/core/exact_linalg.esk')] if args.compile_linalg_source else []
        run([args.eshkol_bin, *extra, str(source), '-O', '2', '-o', str(binary)])
        if run([str(binary)]) != jit:
            raise AssertionError('JIT/AOT bytes differ')
        blocked = disabled_source((root / 'lib/core/observable_closure.esk').read_text())
        replay = work / 'replay.esk'
        replay.write_text(blocked + '\n(define spec \'' + lines[65][7:] + ')\n'
            '(define proof \'' + lines[66][12:] + ')\n'
            '(if (not (verify-generator-closure spec proof)) (error "Replay refused"))\n'
            '(display "PASS observable_producer_free") (newline)\n')
        replay_jit = run([args.eshkol_bin, '-r', str(replay)])
        run([args.eshkol_bin, *extra, str(replay), '-O', '2', '-o', str(work / 'replay')])
        assert replay_jit == run([str(work / 'replay')]) == 'PASS observable_producer_free\n'
        result = {'schema_version': 1, 'jit_executable_cache': 'disabled', 'engines': ['jit', 'aot-O2'], 'checks_per_engine': 159,
                  'independent_span_cases': 64, 'hawkes_dimensions': [15, 4, 11],
                  'producer_free_replay': True, 'stdout_sha256': hashlib.sha256(jit.encode()).hexdigest(),
                  'linalg_source_compiled_explicitly': args.compile_linalg_source,
                  'compiler_sha256': hashlib.sha256(Path(args.eshkol_bin).read_bytes()).hexdigest(),
                  'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in ['lib/core/observable_closure.esk', 'tests/stdlib/observable_closure_test.esk',
                                 'scripts/run_observable_closure_gate.py', 'lib/core/exact_linalg.esk', 'lib/core/symbolic.esk']}}
        if args.results:
            args.results.write_text(json.dumps(result, indent=2) + '\n')
    print('OBSERVABLE CLOSURE PASS: exact spans, source-bound Hawkes 4/15 and 11/15, producer-free JIT/AOT')


if __name__ == '__main__':
    main()
