"""Compile refusal controls; a rejected form must produce no artifact or effects."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

CASES = {
    'capture': ('(define outside 3)\n(differentiate \'numeric (lambda (x) (+ x outside)) (vector 1.0))', 'free capture'),
    'opaque': ('(differentiate \'numeric (lambda (x) (sin x)) (vector 1.0))', 'opaque call'),
    'effect': ('(differentiate \'numeric (lambda (x) (begin (display x) x)) (vector 1.0))', 'one expression'),
    'unused-effect': ('(differentiate \'numeric (lambda (x) (let ((unused (display x))) x)) (vector 1.0))', 'opaque call'),
    'control': ('(differentiate \'numeric (lambda (x) (if (> x 0) x (- x))) (vector 1.0))', 'control flow'),
    'variadic': ('(differentiate \'numeric (lambda (x . rest) x) (vector 1.0))', 'fixed scalar'),
    'named-function': ('(define (f x) (* x x))\n(differentiate \'numeric f (vector 1.0))', 'inline lambda'),
    'nonfinite': ('(differentiate \'numeric (lambda (x) (+ x +inf.0)) (vector 1.0))', 'nonfinite'),
    'global-operator': ('(define (+ a b) (* a b))\n(differentiate \'numeric (lambda (x) (+ x 1)) (vector 1.0))', 'helper'),
    'outer-operator': ('(let ((+ (lambda (a b) (* a b)))) (differentiate \'numeric (lambda (x) (+ x 1)) (vector 1.0)))', 'helper'),
    'global-multiply': ('(define (* a b) (+ a b))\n(differentiate \'numeric (lambda (x) (* x x)) (vector 1.0))', 'not bound to its primitive'),
    'outer-multiply': ('(let ((* (lambda (a b) (+ a b)))) (differentiate \'incremental (lambda (x) (* x x)) (vector 1.0)))', 'opaque call'),
    'inner-operator': ('(differentiate \'numeric (lambda (x) (let ((* 3)) (* x x))) (vector 1.0))', 'shadows an arithmetic operator'),
    'helper-shadow': ('(let ((vector-ref (lambda (a b) 99))) (differentiate \'numeric (lambda (x) (* x x)) (vector 1.0)))', 'helper'),
    'guard-helper-shadow': ('(let ((vector-ref (lambda (a b) 99))) (guard (e (else #f)) (differentiate \'numeric (lambda (x) (* x x)) (vector 1.0))))', 'helper'),
    'incremental-capture': ('(define outside 3)\n(differentiate \'incremental (lambda (x) (+ x outside)) (vector 1.0))', 'free capture'),
    'incremental-effect': ('(differentiate \'incremental (lambda (x) (begin (display x) x)) (vector 1.0))', 'one expression'),
    'incremental-helper': ('(let ((eqv? (lambda (a b) #t))) (differentiate \'incremental (lambda (x) (* x x)) (vector 1.0)))', 'helper'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner', required=True)
    parser.add_argument('--link-dir', required=True)
    parser.add_argument('--vm-runner')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='eshkol-differentiation-') as directory:
        root = Path(directory)
        for name, (program, diagnostic) in CASES.items():
            source = root / (name + '.esk')
            binary = root / (name + '.bin')
            program = program.replace('(vector 1.0)', '(begin (display "FORBIDDEN-RUNTIME-EFFECT\\n") (vector 1.0))')
            source.write_text('(require stdlib)\n' + program + '\n')
            for mode in ('aot', 'jit'):
                command = [args.runner, '-O', '1', '-o', str(binary), str(source), '-L' + args.link_dir] if mode == 'aot' else [args.runner, '-r', str(source), '-L' + args.link_dir]
                result = subprocess.run(command, text=True, capture_output=True, timeout=180)
                output = result.stdout + result.stderr
                assert result.returncode != 0 and diagnostic in output and not binary.exists(), (name, mode, output)
                assert 'FORBIDDEN-RUNTIME-EFFECT' not in output.splitlines(), (name, mode, output)
                print('PASS:', name, mode, flush=True)
        for mode in ('numeric', 'incremental'):
            source = root / ('vm-refusal-' + mode + '.esk')
            source.write_text("(differentiate '%s (lambda (x) (* x x)) (vector 1.0))\n" % mode)
            binary = source.with_suffix('.eskb')
            result = subprocess.run([args.runner, '--profile=hosted-vm', '-B', str(binary), str(source)],
                                    text=True, capture_output=True, timeout=180)
            output = result.stdout + result.stderr
            assert result.returncode != 0 and 'native compiler frontend' in output and not binary.exists(), output
            print('PASS: explicit VM refusal', mode, flush=True)
        if args.vm_runner:
            source = root / 'vm-rebound.esk'
            source.write_text("(define (differentiate mode function point) 77)\n"
                "(if (= (differentiate 'numeric (lambda (x) x) (vector 1)) 77)\n"
                "  (display \"VM rebound differentiate PASS\\n\") (error \"FAIL: VM rebinding\"))\n")
            result = subprocess.run([args.vm_runner, str(source)], text=True, capture_output=True, timeout=180)
            output = result.stdout + result.stderr
            assert result.returncode == 0 and 'VM rebound differentiate PASS' in output and 'FAIL:' not in output, output
            print('PASS: VM user-rebound differentiate', flush=True)
    print('RESULT: PASS')


if __name__ == '__main__':
    main()
