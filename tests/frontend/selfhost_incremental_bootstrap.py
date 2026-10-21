"""Reproduce the bounded compiler-pass bootstrap and exercise private gateways."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def run(command, env, timeout=180, input_text=None):
    start = time.monotonic()
    result = subprocess.run(command, env=env, input=input_text, text=True, capture_output=True, timeout=timeout)
    output = result.stdout + result.stderr
    if result.returncode or 'FAIL:' in output or 'RESULT: FAIL' in output:
        raise AssertionError((command, result.returncode, output))
    return output, time.monotonic() - start


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('runner', 'seed', 'packager', 'plan-test', 'seed-test', 'link-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--baseline-runner')
    parser.add_argument('--baseline-link-dir')
    parser.add_argument('--repl-runner')
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, ESHKOL_PATH=str(source_root / 'lib'), ESHKOL_JIT_CACHE='0',
               ESHKOL_TARGET_CPU='generic', ESHKOL_TARGET_FEATURES='')
    native, _ = run([args.plan_test, '--emit'], env)
    seed, _ = run([args.seed_test, '--emit'], env)
    assert native == seed and len(native.splitlines()) == 168
    print('PLAN_PARITY cases=168 sha256=' + hashlib.sha256(native.encode()).hexdigest(), flush=True)
    if args.repl_runner:
        forms = [
            '(define (check ok) (if (not ok) (error "staged compiler check failed")))',
            '(define (incremental-plan graph) 99)',
            '(define (__eshkol_compiler_incremental_state a b c) 91)',
            '(define (__eshkol_compiler_incremental_check value) 92)',
            "(define state (differentiate 'incremental (lambda (x y) (+ (* x y) (* x x))) (vector 2 3)))",
            "(define other (differentiate 'incremental (lambda (x) (* x x)) (vector 4)))",
            '(check (= (procedure-arity state) 2))',
            "(check (= (state 'output #f) 10))",
            "(define next (vector-ref (state 'step (vector 1 2)) 0))",
            "(check (= (next 'output #f) 24))",
            "(check (= (state 'output #f) 10))",
            '(check (= (__eshkol_compiler_incremental_state 0 0 0) 91))',
            '(check (= (__eshkol_compiler_incremental_check 0) 92))',
            '(check (= (incremental-plan #f) 99))',
            '(display "STAGED_REPL_PASS\\n")',
        ]
        output, _ = run([args.repl_runner, '--machine'], env, input_text='\n'.join(forms) + '\n')
        assert 'STAGED_REPL_PASS' in output and 'EREPL FAIL' not in output, output
        print('INTEGRATION staged-repl private-bindings and retained-state', flush=True)
    with tempfile.TemporaryDirectory(prefix='eshkol-pass-bootstrap-') as directory:
        root = Path(directory)
        for part, source, entry in (
            ('plan', source_root / 'tools/compiler/incremental_plan_entry.esk', 'eshkol_incremental_plan_entry'),
            ('runtime', source_root / 'lib/core/compiler/incremental_runtime.esk', '-'),
        ):
            hashes = []
            for stage, runner in (('seed', args.seed), ('final', args.runner)):
                base = root / (part + '-' + stage)
                run([runner, '--no-stdlib', '--strict-types', '--shared-lib', '-c', '-O1', '-o', str(base), str(source)], env)
                packaged = base.with_suffix('.native.o')
                run([args.packager, str(base.with_suffix('.bc')), str(packaged), '__eshkol_incremental_' + part, entry], env)
                hashes.append(sha(packaged))
            assert hashes[0] == hashes[1], (part, hashes)
            print('BOOTSTRAP ' + part + ' object_sha256=' + hashes[0], flush=True)

        prefix = '''(define (check ok) (if (not ok) (error "FAIL: self-hosted compiler integration")))
'''
        computation = '''
(define state (differentiate 'incremental (lambda (x y) (+ (* x y) (* x x))) (vector 2 3)))
(check (= (procedure-arity state) 2))
(define gradient-at-point (differentiate 'numeric (lambda (x y) (+ (* x y) (* x x))) (vector 2 3)))
(check (= (vector-ref gradient-at-point 0) 7))
(check (= (vector-ref gradient-at-point 1) 2))
(define result (state 'step (vector 1 2)))
(check (= ((vector-ref result 0) 'output #f) 24))
(check (= (vector-ref result 1) 14))
(check (= (state 'output #f) 10))
(check (= (vector-ref (vector-ref result 2) 0) 3))
'''
        fixtures = {
            'source-bindings': prefix + '''
(define (__eshkol_compiler_incremental_state a b c) 91)
(define (__eshkol_compiler_incremental_check value) 92)
(define (__eshkol_incremental_state) 93)
(define (__eshkol_incremental_check) 94)
''' + computation + '''
(check (= (__eshkol_compiler_incremental_state 0 0 0) 91))
(check (= (__eshkol_compiler_incremental_check 0) 92))
(check (= (__eshkol_incremental_state) 93))
(check (= (__eshkol_incremental_check) 94))
(display "RESULT: PASS\\n")
''',
            'source-macros': prefix + '''
(define-syntax __eshkol_incremental_state (syntax-rules () ((_ a b) (error "hijacked state gateway"))))
(define-syntax __eshkol_incremental_check (syntax-rules () ((_ a) (error "hijacked check gateway"))))
''' + computation + '''
(check (guard (condition (#t #t)) (__eshkol_incremental_check 1) #f))
(display "RESULT: PASS\\n")
''',
        }
        for name, text in fixtures.items():
            source, binary = root / (name + '.esk'), root / (name + '.bin')
            source.write_text(text)
            for mode in ('jit', 'aot'):
                command = [args.runner, '--no-stdlib', '--strict-types', '-L' + args.link_dir]
                command += ['-r', str(source)] if mode == 'jit' else ['-O1', '-o', str(binary), str(source)]
                output, _ = run(command, env)
                if mode == 'aot': output, _ = run([str(binary)], env)
                assert 'RESULT: PASS' in output
                print('INTEGRATION ' + name + ' ' + mode, flush=True)

        if args.baseline_runner:
            assert args.baseline_link_dir
            measurements = []
            for count in (1, 4, 8):
                source = root / ('size-' + str(count) + '.esk')
                source.write_text(''.join(
                    "(define s%d (differentiate 'incremental (lambda (x y) (+ (* x y) (* x x))) (vector 2 3)))\n" % i
                    for i in range(count)) + "(display (s%d 'output #f)) (newline)\n" % (count - 1))
                for label, runner, link_dir in (
                    ('baseline', args.baseline_runner, args.baseline_link_dir),
                    ('shared', args.runner, args.link_dir),
                ):
                    binary = root / (label + '-' + str(count))
                    local_env = dict(env, ESHKOL_LIB_DIR=link_dir)
                    _, seconds = run([runner, '--no-stdlib', '--strict-types', '-O1', '-o', str(binary), str(source), '-L' + link_dir], local_env, 600)
                    output, _ = run([str(binary)], local_env)
                    assert output.strip() == '10'
                    measurements.append(dict(forms=count, implementation=label, compile_seconds=round(seconds, 4), executable_bytes=binary.stat().st_size))
            print('MEASUREMENTS ' + json.dumps(measurements), flush=True)
    print('RESULT: PASS')


if __name__ == '__main__':
    main()
