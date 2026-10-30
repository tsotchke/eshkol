#!/usr/bin/env python3
"""Independent qualification of upstream SAT proof/input admission (MIT)."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def read(source):
    tokens = re.findall(r'#\(|[()]|[^\s()]+', source)
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
        if t in ('#t', '#f'):
            return t == '#t'
        return int(t) if re.fullmatch(r'-?\d+', t) else t

    out = parse()
    if i != len(tokens):
        raise ValueError('Trailing datum')
    return out


def model_valid(clauses, model):
    return all(any(model[abs(lit)] == (lit > 0) for lit in row) for row in clauses)


def truth(n, clauses):
    return 'sat' if any(model_valid(clauses, (False,)+bits)
                        for bits in itertools.product((False, True), repeat=n)) else 'unsat'


def php(pigeons, holes):
    def v(p, h):
        return 1+p*holes+h
    clauses = [[v(p, h) for h in range(holes)] for p in range(pigeons)]
    clauses += [[-v(p, h), -v(q, h)] for h in range(holes)
                for p in range(pigeons) for q in range(p+1, pigeons)]
    return sorted(sorted(row) for row in clauses)


def expected_random():
    seed = 20261030

    def next_value():
        nonlocal seed
        seed = (seed*1103515245+12345) % 2147483648
        return seed

    for i in range(96):
        n, clauses = 3+i % 3, []
        for _ in range(4+i % 12):
            row = []
            for _ in range(1+next_value() % 4):
                lit = 1+next_value() % n
                row.insert(0, lit if next_value() % 2 == 0 else -lit)
            clauses.insert(0, row)
        yield i, n, clauses


def rup_conflict(database, row):
    assignment = {}
    for lit in row:
        value = lit < 0
        if abs(lit) in assignment and assignment[abs(lit)] != value:
            return True
        assignment[abs(lit)] = value
    while True:
        changed = False
        for clause in database:
            if any(assignment.get(abs(lit)) == (lit > 0) for lit in clause
                   if abs(lit) in assignment):
                continue
            open_lits = [lit for lit in clause if abs(lit) not in assignment]
            if not open_lits:
                return True
            if len(open_lits) == 1:
                lit = open_lits[0]
                assignment[abs(lit)] = lit > 0
                changed = True
        if not changed:
            return False


def proof_valid(clauses, proof):
    def literals(row):
        return isinstance(row, list) and all(type(x) is int and x != 0 for x in row)
    if not isinstance(clauses, list) or not all(literals(row) for row in clauses):
        return False
    if not isinstance(proof, list) or not all(
            isinstance(step, list) and step and step[0] in ['add', 'delete']
            and literals(step[1:]) and not any(-x in step[1:] for x in step[1:])
            for step in proof):
        return False
    database = [set(row) for row in clauses if not any(-x in row for x in row)]
    for step in proof:
        row = set(step[1:])
        if step[0] == 'delete':
            if row in database:
                database.pop(database.index(row))
            continue
        pivot = step[1] if row else None
        rat = bool(row) and all(rup_conflict(database, row | (other - {-pivot}))
                               for other in database if -pivot in other)
        if not rup_conflict(database, row) and not rat:
            return False
        if not row:
            return True
        database.append(row)
    return False


def signed_model_valid(n, clauses, model):
    if not isinstance(model, list) or len(model) != n:
        return False
    if any(type(x) is not int or not 1 <= abs(x) <= n for x in model):
        return False
    if len({abs(x) for x in model}) != n:
        return False
    values = {abs(x): x > 0 for x in model}
    return model_valid(clauses, (False,) + tuple(values[i] for i in range(1, n + 1)))


def datum(value):
    if isinstance(value, list):
        return '(' + ' '.join(datum(x) for x in value) + ')'
    if value is True or value is False:
        return '#t' if value else '#f'
    return str(value)


def cases():
    pool = [[], [1], [-1], [2], [-2], [1, 2], [1, -2], [-1, -2]]
    rows = [(mask, 2, [row for i, row in enumerate(pool) if mask >> i & 1])
            for mask in range(256)]
    return rows + [(256 + i, n, clauses) for i, n, clauses in expected_random()]


def solver_free(source):
    producers = {'sat-solve', 'sat-luby', 'sat-brute-force',
                 'cnf-pigeonhole', 'cnf-random-ksat', 'cnf-queens'}
    forms, start, depth, string, comment = [], 0, 0, False, False
    for i, ch in enumerate(source):
        if comment:
            comment = ch != '\n'
            continue
        if ch == ';' and not string:
            comment = True
            continue
        if ch == '"' and (not i or source[i - 1] != '\\'):
            string = not string
        if string:
            continue
        if ch == '(':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                forms.append(source[start:i + 1])
    assert depth == 0
    disabled = set()
    for i, form in enumerate(forms):
        match = re.match(r'\(define\s+\(([^\s()]+)', form)
        if match and match[1] in producers:
            forms[i] = form[:form.index(')') + 1] + ' (error "SAT producer disabled"))'
            disabled.add(match[1])
    assert disabled == producers
    return '\n'.join(forms), sorted(disabled)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eshkol-bin', required=True)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--results', type=Path)
    args = parser.parse_args()
    root = args.source_root.resolve()
    env = dict(os.environ, ESHKOL_PATH=str(root / 'lib'), ESHKOL_JIT_CACHE='0',
               ESHKOL_JIT_COMPILE_THREADS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='sat-proof-gate-') as directory:
        work = Path(directory)
        def run(command):
            result = subprocess.run(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=240)
            if result.returncode:
                raise RuntimeError(result.stdout[-1600:] + result.stderr[-1600:])
            return result.stdout
        def both(source, name):
            jit = run([args.eshkol_bin, '-r', str(source)])
            binary = work / name
            run([args.eshkol_bin, str(source), '-O', '0', '-o', str(binary)])
            aot = run([str(binary)])
            if jit != aot:
                raise AssertionError(name + ': uncached JIT/AOT receipts differ')
            return jit
        existing = both(root / 'tests/stdlib/reasoning_sat_test.esk', 'existing')
        if not re.fullmatch(r'Passed: [1-9][0-9]* Failed: 0\nPASS: reasoning_sat_test\nRESULT: PASS\n', existing):
            raise AssertionError('Existing SAT receipt incomplete')
        admission = both(root / 'tests/stdlib/reasoning_sat_admission_test.esk', 'admission')
        if admission != 'PASS sat_admission 49\n':
            raise AssertionError('SAT admission receipt incomplete')
        expected = cases()
        driver = work / 'cases.esk'
        driver.write_text('''(require core.reasoning.sat)
(for-each (lambda (entry)
 (let* ((i (car entry)) (n (cadr entry)) (cnf (caddr entry))
        (r (sat-solve cnf (list (cons 'num-vars n)))))
  (display "CASE ") (write (list i n cnf (sat-status r) (sat-model r) (sat-proof r))) (newline)))
''' + "'" + datum([[i, n, cnf] for i, n, cnf in expected]) + ''')
(for-each (lambda (h)
 (let* ((cnf (cnf-pigeonhole h)) (r (sat-solve cnf)))
  (if (not (eq? (sat-status r) 'unsat)) (error "PHP verdict"))
  (display "PHP ") (write (list (+ h 1) h cnf (sat-proof r))) (newline))) '(3 4))
(define rat-cnf (cnf-pigeonhole 3))
(define rat-proof (append '((add 100 -1) (add -100 1)) (sat-proof (sat-solve rat-cnf))))
(if (not (drat-check rat-cnf rat-proof)) (error "RAT replay"))
(display "RAT ") (write (list rat-cnf rat-proof)) (newline)
(define model-cnf '((1 2) (-1 2)))
(display "MODEL ") (write (list model-cnf (sat-model (sat-solve model-cnf)))) (newline)
(display "PASS sat_proof_cases 352") (newline)
''')
        output = both(driver, 'cases')
        lines = output.splitlines()
        if len(lines) != 357 or lines[-1] != 'PASS sat_proof_cases 352':
            raise AssertionError('SAT case receipt incomplete')
        for (i, n, cnf), line in zip(expected, lines[:352]):
            assert line.startswith('CASE ')
            actual_i, actual_n, actual_cnf, status, model, proof = read(line[5:])
            assert (actual_i, actual_n, actual_cnf) == (i, n, cnf)
            assert status == truth(n, cnf)
            assert signed_model_valid(n, cnf, model) if status == 'sat' else proof_valid(cnf, proof)
        php_records = []
        for line, p, h in zip(lines[352:354], [4, 5], [3, 4]):
            assert line.startswith('PHP ')
            actual_p, actual_h, cnf, proof = read(line[4:])
            assert (actual_p, actual_h) == (p, h)
            assert sorted(sorted(row) for row in cnf) == php(p, h)
            assert proof_valid(cnf, proof) and not proof_valid(cnf, proof[:-1])
            php_records.append([cnf, proof])
        assert lines[354].startswith('RAT ') and lines[355].startswith('MODEL ')
        rat_cnf, rat_proof = read(lines[354][4:])
        assert proof_valid(rat_cnf, rat_proof)
        database = [set(row) for row in rat_cnf]
        for step in rat_proof[:2]:
            assert not rup_conflict(database, set(step[1:])), 'RAT control became RUP-only'
            database.append(set(step[1:]))
        model_cnf, model = read(lines[355][6:])
        assert model_cnf == [[1, 2], [-1, 2]] and signed_model_valid(2, model_cnf, model)
        blocked, disabled = solver_free((root / 'lib/core/reasoning/sat.esk').read_text())
        replay = work / 'replay.esk'
        replay.write_text(blocked + '''
(define (refuses? thunk) (guard (e (else #t)) (thunk) #f))
''' + '\n'.join('(if (not (refuses? (lambda () ' + call + '))) (error "Producer still enabled"))'
                         for call in ['(sat-solve \'())', '(sat-luby 0)', '(sat-brute-force \'())',
                                      '(cnf-pigeonhole 2)', '(cnf-random-ksat 2 1 1 7)', '(cnf-queens 1)'])
                         + '\n(if (not (drat-check \'' + datum(php_records[1][0]) + " '" + datum(php_records[1][1]) + ')) (error "PHP replay"))'
                         + '\n(if (not (drat-check \'' + datum(rat_cnf) + " '" + datum(rat_proof) + ')) (error "RAT replay"))'
                         + '\n(if (drat-check \'((1 2)) \'((add))) (error "Forged proof"))'
                         + '\n(if (not (sat-satisfies? \'' + datum(model_cnf) + " '" + datum(model) + ')) (error "Model replay"))'
                         + '\n(display "PASS sat_solver_free 10") (newline)\n')
        replay_output = both(replay, 'replay')
        assert replay_output == 'PASS sat_solver_free 10\n'
        names = ['lib/core/reasoning/sat.esk', 'tests/stdlib/reasoning_sat_test.esk',
                 'tests/stdlib/reasoning_sat_admission_test.esk', 'scripts/run_sat_proof_gate.py',
                 'scripts/sat_proof_gate_self_test.py', 'repro/sat-proof-admission/SOURCE.toml']
        result = {'schema_version': 1, 'engines': ['jit', 'aot-O0'], 'jit_executable_cache': 'disabled',
                  'admission_checks_per_engine': 49, 'independent_truth_cases': 352,
                  'independent_pigeonhole_proofs': [[4, 3], [5, 4]], 'non_RUP_RAT_steps': 2,
                  'solver_free_checks_per_engine': 10, 'disabled_producers': disabled,
                  'existing_suite_stdout_sha256': hashlib.sha256(existing.encode()).hexdigest(),
                  'case_stdout_sha256': hashlib.sha256(output.encode()).hexdigest(),
                  'compiler_sha256': hashlib.sha256(Path(args.eshkol_bin).read_bytes()).hexdigest(),
                  'files_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}
        if args.results:
            args.results.write_text(json.dumps(result, indent=2) + '\n')
    print('SAT PROOF PASS: existing suite, 49 admission controls, 352 truth cases, independent DRAT, solver-free JIT/AOT')


if __name__ == '__main__':
    main()
