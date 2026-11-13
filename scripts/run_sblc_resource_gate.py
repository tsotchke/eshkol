#!/usr/bin/env python3
"""Exact PGF oracle, separate beta reducer, and producer-free native replay."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def read(source):
    tokens = re.findall(r'"[^"\\]*"|#\(|[()]|[^\s()]+', source)
    i = 0

    def parse():
        nonlocal i
        t = tokens[i]
        i += 1
        if t in ('(', '#('):
            out = []
            while tokens[i] != ')':
                out.append(parse())
            i += 1
            return out
        if t.startswith('"'):
            return json.loads(t)
        if re.fullmatch(r'-?\d+(?:/\d+)?', t):
            return F(t)
        return t

    out = parse()
    if i != len(tokens):
        raise ValueError('Trailing datum')
    return out


def product(a, b, d):
    return [sum((a[i] * b[k-i] for i in range(k+1)), F(0)) for k in range(d+1)]


def repeat(a, m, d):
    out = [F(1)] + [F(0)] * d
    for _ in range(m):
        out = product(out, a, d)
    return out


def branch(k, d):
    p = F(k, 4)
    q = p / (1-p) if p < F(1, 2) else F(1)
    denominator = 1 - 2*(1-p)*q
    beta = (p + 2*(1-p)*q*q) / denominator if denominator else 'infinite'
    coins = (2*p + 2*(1-p)*q*q) / denominator if denominator else 'infinite'
    a = [F(0)] * (d+1)
    for j in range(d+1):
        a[j] = (p if j == 3 else 0) + ((1-p)*sum((a[i]*a[j-4-i] for i in range(j-3)), F(0)) if j >= 4 else 0)
    return [F(k), q, beta, coins, a]


def weighted(w, x):
    return F(0) if w == 0 else ('infinite' if x == 'infinite' else w*x)


def plus(a, b):
    return 'infinite' if 'infinite' in (a, b) else a+b


def moments_product(a, b):
    return [a[0]*b[0], plus(weighted(b[0], a[1]), weighted(a[0], b[1])),
            plus(weighted(b[0], a[2]), weighted(a[0], b[2]))]


def oracle(descriptor, d):
    mode, k0, k1, m = descriptor
    k0, k1, m = int(k0), int(k1), int(m)
    a, b = branch(k0, d), branch(k1, d)
    avg = [(x+y)/2 for x, y in zip(a[4], b[4])]
    if mode == 'branch':
        coefficients, offset = a[4], 0
    elif mode == 'shared':
        coefficients = [(x+y)/2 for x, y in zip(repeat(a[4], m, d), repeat(b[4], m, d))]
        offset = m
    else:
        coefficients, offset = repeat(avg, m, d), 2*m-1
    coefficients = [F(0) if i < offset else coefficients[i-offset] for i in range(d+1)]
    ta, tb = a[1:4], b[1:4]

    def repetitions(t):
        out = [F(1), F(0), F(0)]
        for _ in range(m):
            out = moments_product(out, t)
        return out

    def mixture(x, y):
        return [(x[0]+y[0])/2, weighted(F(1, 2), plus(x[1], y[1])), weighted(F(1, 2), plus(x[2], y[2]))]

    if mode == 'branch':
        total = ta
    elif mode == 'shared':
        total = mixture(repetitions(ta), repetitions(tb))
        total[1] = plus(total[1], (m-1)*total[0])
        total[2] = plus(total[2], total[0])
    else:
        mixed = mixture(ta, tb)
        mixed[2] = plus(mixed[2], mixed[0])
        total = repetitions(mixed)
        total[1] = plus(total[1], (m-1)*total[0])
    q, beta, coins = total
    work = plus(beta, coins)
    summary = [q, beta, coins, 'infinite' if q < 1 else work,
               'undefined' if q == 0 else weighted(1/q, work)]
    return [a, b], coefficients, summary


def term(value):
    return tuple(term(x) if isinstance(x, list) else int(x) if isinstance(x, F) else x for x in value)


def shift(t, amount, depth=0):
    if t[0] == 'var':
        return ('var', t[1] + (amount if t[1] > depth else 0))
    return (t[0], *(shift(x, amount, depth+int(t[0] == 'lam')) for x in t[1:]))


def substitute(body, argument, depth=0):
    if body[0] == 'var':
        if body[1] == depth+1:
            return shift(argument, depth)
        return ('var', body[1] - int(body[1] > depth+1))
    return (body[0], *(substitute(x, argument, depth+int(body[0] == 'lam')) for x in body[1:]))


@lru_cache(maxsize=65536)
def step(t):
    tag = t[0]
    if tag == 'var':
        return None
    if tag == 'choice':
        return ('coin', t[1], t[2])
    if tag == 'app' and t[1][0] == 'lam':
        return ('beta', substitute(t[1][1], t[2]))
    for i, child in enumerate(t[1:], 1):
        event = step(child)
        if event:
            successors = []
            for out in event[1:]:
                changed = list(t)
                changed[i] = out
                successors.append(tuple(changed))
            return (event[0], *successors)
    return None


def normal_mass(source, bound):
    frontier, complete = {source: F(1)}, F(0)
    for depth in range(bound+1):
        next_layer = defaultdict(F)
        for t, mass in frontier.items():
            event = step(t)
            if event is None:
                assert t == ('lam', ('var', 1))
                complete += mass
            elif depth < bound:
                for successor in event[1:]:
                    next_layer[successor] += mass/(2 if event[0] == 'coin' else 1)
        frontier = next_layer
    return complete


def branching(k):
    identity = ('lam', ('var', 1))
    xx = ('app', ('var', 1), ('var', 1))
    leaves = [identity if i < k else ('app', xx, xx) for i in range(4)]
    d = ('lam', ('choice', ('choice', *leaves[:2]), ('choice', *leaves[2:])))
    return ('app', d, d)


def sequence(t, n):
    out = t
    for _ in range(n-1):
        out = ('app', out, t)
    return out


def source_for(descriptor):
    mode, a, b, m = descriptor
    a, b, m = int(a), int(b), int(m)
    if mode == 'branch':
        return branching(a)
    if mode == 'shared':
        return ('choice', sequence(branching(a), m), sequence(branching(b), m))
    return sequence(('choice', branching(a), branching(b)), m)


def encode(t):
    if t[0] == 'var':
        return '0' + '1'*(t[1]-1) + '0'
    return {'lam': '10', 'app': '110', 'choice': '111'}[t[0]] + ''.join(encode(x) for x in t[1:])


def disabled_source(source):
    forbidden = {'sblc-branching-source', 'sblc-latent-source', 'sblc-resource-analyze', 'sr-sequence',
                 'sr-power', 'sr-convolve', 'sr-series-power', 'sr-coefficients', 'sr-branch-record', 'sr-summary', 'sr-output'}
    forms, level, start, string, comment = [], 0, 0, False, False
    for i, c in enumerate(source):
        if comment:
            comment = c != '\n'
            continue
        if c == ';' and not string:
            comment = True
            continue
        if c == '"':
            string = not string
        if not string:
            if c == '(':
                if level == 0:
                    start = i
                level += 1
            elif c == ')':
                level -= 1
                if level == 0:
                    forms.append(source[start:i+1])
    assert level == 0
    replaced = set()
    for i, form in enumerate(forms):
        match = re.match(r'\(define\s+\(([^\s()]+)', form)
        if match and match[1] in forbidden:
            forms[i] = form[:form.index(')')+1] + ' (error "Resource producer disabled"))'
            replaced.add(match[1])
    assert replaced == forbidden
    return '\n'.join(forms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eshkol-bin', required=True)
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--results', type=Path)
    args = p.parse_args()
    root = args.source_root.resolve()
    env = dict(os.environ, ESHKOL_PATH=str(root/'lib'), ESHKOL_JIT_CACHE='0')
    with tempfile.TemporaryDirectory(prefix='sblc-resource-') as directory:
        work = Path(directory)

        def run(command):
            r = subprocess.run(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=240)
            if r.returncode:
                raise RuntimeError(r.stdout[-2000:]+r.stderr[-2000:])
            return r.stdout

        source = root/'tests/stdlib/sblc_resource_test.esk'
        jit = run([args.eshkol_bin, '-r', str(source)])
        lines = jit.splitlines()
        if len(lines) != 30 or lines[-1] != 'PASS sblc_resource 84':
            raise AssertionError('Native resource receipt is incomplete')
        for line in lines[:11]:
            assert line.startswith('RESOURCE ')
            label, summary, coefficients = read(line[9:])
            descriptor = [label[0], label[1], label[1], 1] if label[0] == 'branch' else label
            _, wanted, moments = oracle(descriptor, 24)
            assert coefficients == wanted and summary == moments
        for line in lines[11:27]:
            assert line.startswith('OBS ')
            descriptor, bound, completed, timeout = read(line[4:])
            bound = int(bound)
            _, coefficients, _ = oracle(descriptor, 24)
            assert completed == sum(coefficients[:bound+1]) == normal_mass(source_for(descriptor), bound)
            assert completed+timeout == 1
        assert lines[27].startswith('SOURCE ') and lines[28].startswith('CERTIFICATE ')
        raw_source, raw_proof = lines[27][7:], lines[28][12:]
        actual, proof = term(read(raw_source)), read(raw_proof)
        assert actual == source_for(['shared', 1, 3, 2])
        assert proof[1] == 'sblc-v1:'+encode(actual)
        records, coefficients, summary = oracle(proof[2], int(proof[3]))
        assert proof[4:] == [records, coefficients, summary]
        run([args.eshkol_bin, str(source), '-O', '2', '-o', str(work/'test')])
        assert run([str(work/'test')]) == jit
        replay = work/'replay.esk'
        replay.write_text(disabled_source((root/'lib/core/sblc/resource.esk').read_text())
            + '\n(define source \''+raw_source+')\n(define proof \''+raw_proof+')\n'
            '(if (not (sblc-resource-verify source proof)) (error "Resource replay refused"))\n'
            '(display "PASS resource_producer_free") (newline)\n')
        replay_jit = run([args.eshkol_bin, '-r', str(replay)])
        run([args.eshkol_bin, str(replay), '-O', '2', '-o', str(work/'replay')])
        assert replay_jit == run([str(work/'replay')]) == 'PASS resource_producer_free\n'
        result = {'schema_version': 1, 'jit_executable_cache': 'disabled', 'checks_per_engine': 84, 'cost_profiles': 11,
                  'independent_operational_observations': 16, 'engines': ['jit', 'aot-O2'],
                  'termination_probabilities': ['5/9', '4/9'], 'conditional_total_work': [15, 16],
                  'producer_free_replay': True, 'stdout_sha256': hashlib.sha256(jit.encode()).hexdigest(),
                  'compiler_sha256': hashlib.sha256(Path(args.eshkol_bin).read_bytes()).hexdigest(),
                  'source_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                      for name in ['lib/core/sblc/resource.esk', 'tests/stdlib/sblc_resource_test.esk',
                                   'scripts/run_sblc_resource_gate.py', 'lib/core/sblc.esk', 'lib/core/blc.esk']}}
        if args.results:
            args.results.write_text(json.dumps(result, indent=2)+'\n')
    print('SBLC RESOURCE PASS: exact beta+coin laws, independent reducer, 5/9 vs 4/9, producer-free JIT/AOT')


if __name__ == '__main__':
    main()
