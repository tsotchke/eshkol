#!/usr/bin/env python3
"""Exercise entry signatures and entry identity through the actual compiler."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--vm-runner', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    runner = args.runner.resolve()
    evidence = root / '.scratch' / f'program-entry-{time.time_ns()}-{os.getpid()}'
    evidence.mkdir(parents=True, exist_ok=False)
    temporary = evidence / 'tmp'
    temporary.mkdir()
    env = dict(os.environ, TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary),
               ESHKOL_JIT_CACHE='0', ESHKOL_VM_NO_DISASM='1')
    env['ESHKOL_LIB_DIR'] = str(runner.parent)
    env['ESHKOL_PATH'] = str(root / 'lib')
    results = []

    def source(name, text):
        path = evidence / f'{name}.esk'
        path.write_text(text)
        return path

    def run(name, command, expected=0, output=None, diagnostic=False):
        start = time.monotonic()
        completed = subprocess.run([str(x) for x in command], capture_output=True,
                                   text=True, env=env, timeout=120)
        log = completed.stdout + completed.stderr
        (evidence / f'{name}.log').write_text(log)
        result = {'case': name, 'exit_code': completed.returncode,
                  'expected_exit': expected, 'seconds': round(time.monotonic() - start, 3)}
        results.append(result)
        assert completed.returncode == expected, (name, completed.returncode, log)
        if output is not None:
            assert completed.stdout.strip() == output, (name, completed.stdout)
        if diagnostic:
            assert 'main must be defined with zero parameters' in log, (name, log)
            assert 'LLVM module verification failed' not in log, (name, log)
            assert 'Incorrect number of arguments' not in log, (name, log)
            assert 'SHOULD-NOT-RUN' not in completed.stdout, (name, log)
        result['status'] = 'PASS'
        return completed

    try:
        invalid = {
            'fixed': '(define (main a b) (+ a b))\n(display "SHOULD-NOT-RUN")\n',
            'rest': '(define (main . args) 0)\n(display "SHOULD-NOT-RUN")\n',
            'fixed-rest': '(define (main a . args) a)\n(display "SHOULD-NOT-RUN")\n',
            'macro': '(define-syntax make-main (syntax-rules () ((_ ) (define (main a) a))))\n(make-main)\n(display "SHOULD-NOT-RUN")\n',
        }
        for name, text in invalid.items():
            path = source(name, text)
            for optimization in (0, 2):
                prefix = f'{name}-o{optimization}'
                output = evidence / f'{prefix}.o'
                run(prefix + '-aot', [runner, '--no-stdlib', '-O', optimization,
                                     '--compile-only', '-o', output, path],
                    expected=1, diagnostic=True)
                assert not output.exists(), ('failed compilation created output', output)
                run(prefix + '-jit', [runner, '--no-stdlib', '-O', optimization,
                                     '-r', path], expected=1, diagnostic=True)

        valid = {
            'nullary': ('(define (main) (display "MAIN-ZERO") (newline) 7)\n', 'MAIN-ZERO', 7),
            'collision': ('(define (scheme_main) 91)\n(define (main) (display (scheme_main)) (newline) 7)\n', '91', 7),
            'ordinary': ('(define (f a b) (+ a b))\n(display (f 1 2))\n(newline)\n', '3', 0),
            'name-identity': ('(define (remaining x) (+ x 1))\n(define (main-menu x) (+ x 2))\n(define (run-main x) (+ x 3))\n(display (+ (remaining 0) (main-menu 0) (run-main 0)))\n(newline)\n', '6', 0),
        }
        for name, (text, expected_output, exit_code) in valid.items():
            path = source(name, text)
            run(name + '-jit', [runner, '--no-stdlib', '-r', path], output=expected_output)
            binary = evidence / (name + ('.exe' if os.name == 'nt' else '.out'))
            run(name + '-compile', [runner, '--no-stdlib', '-L' + str(runner.parent),
                                   '-o', binary, path])
            run(name + '-execute', [binary], expected=exit_code, output=expected_output)

        library = source('library-main', '(define (main a b) (+ a b))\n')
        object_file = evidence / 'library-main.o'
        run('library-main', [runner, '--no-stdlib', '--shared-lib', '--compile-only',
                             '-o', object_file, library])
        assert object_file.is_file() and object_file.stat().st_size > 0

        if args.vm_runner:
            vm_source = source('vm-main', '(define (main a b) (+ a b))\n(display (main 1 2))\n(newline)\n')
            run('vm-main-top-level-contract', [args.vm_runner.resolve(), vm_source], output='3')
        status = 'PASS'
        error = None
    except Exception as exception:
        status = 'FAIL'
        error = str(exception)
    report = {'status': status, 'error': error, 'runner': str(runner),
              'runner_sha256': hashlib.sha256(runner.read_bytes()).hexdigest(),
              'cases': results, 'evidence': str(evidence)}
    (evidence / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': status, 'cases': len(results), 'error': error,
                      'evidence': str(evidence)}, separators=(',', ':')))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
