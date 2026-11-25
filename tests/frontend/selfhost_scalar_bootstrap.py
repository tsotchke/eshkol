"""Qualify scalar lowering parity, reproducible bootstrap and native integration."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile

from selfhost_incremental_bootstrap import run, sha


def main():
    parser = argparse.ArgumentParser()
    for name in ('runner', 'seed', 'packager', 'lower-test', 'seed-test', 'link-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--repl-runner')
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, ESHKOL_PATH=str(source_root / 'lib'), ESHKOL_JIT_CACHE='0',
               ESHKOL_TARGET_CPU='generic', ESHKOL_TARGET_FEATURES='')
    native, _ = run([args.lower_test, '--emit'], env)
    seed, _ = run([args.seed_test, '--emit'], env)
    assert native == seed and len(native.splitlines()) == 168
    print('LOWERING_PARITY cases=168 sha256=' + hashlib.sha256(native.encode()).hexdigest(), flush=True)
    with tempfile.TemporaryDirectory(prefix='eshkol-scalar-bootstrap-') as directory:
        root = Path(directory)
        hashes = []
        for stage, runner in (('seed', args.seed), ('final', args.runner)):
            base = root / stage
            run([runner, '--no-stdlib', '--strict-types', '--shared-lib', '-c', '-O1', '-o', str(base),
                 str(source_root / 'tools/compiler/scalar_lower_entry.esk')], env)
            packaged = base.with_suffix('.native.o')
            run([args.packager, str(base.with_suffix('.bc')), str(packaged), '__eshkol_incremental_scalar',
                 'eshkol_scalar_lower_entry', 'eshkol_scalar_lower_call'], env)
            hashes.append(sha(packaged))
        assert hashes[0] == hashes[1], hashes
        print('BOOTSTRAP scalar object_sha256=' + hashes[0], flush=True)
        forms = [
            '(define (check ok) (if (not ok) (error "scalar lowering integration failed")))',
            '(define (scalar-lower graph) 91)',
            '(define (sl-leaf kind slot) 92)',
            '(define (eshkol_scalar_lower_entry graph) 93)',
            '(define (eshkol_scalar_lower_call graph) 94)',
            "(define state (differentiate 'incremental (lambda (x y) (/ (- (* x y) 7/3) (+ y 1))) (vector 2 3)))",
            "(check (= (state 'output #f) 11/12))",
            "(define next (vector-ref (state 'step (vector 1 2)) 0))",
            "(check (= (next 'output #f) 19/9))",
            "(check (= (state 'output #f) 11/12))",
            "(define g (differentiate 'numeric (lambda (x y) (- (* x y) (/ x y))) (vector 6 2)))",
            '(check (= (vector-ref g 0) 1.5))',
            '(check (= (vector-ref g 1) 7.5))',
            "(define rounded (differentiate 'incremental (lambda (x y) (+ (- (+ x y) x) 1)) (vector 1e16 1.0)))",
            "(check (= (rounded 'output #f) 1.0))",
            "(check (= ((vector-ref (rounded 'step (vector 0.0 1.0)) 0) 'output #f) 3.0))",
            '(check (= (scalar-lower #f) 91))',
            '(check (= (sl-leaf #f #f) 92))',
            '(check (= (eshkol_scalar_lower_entry #f) 93))',
            '(check (= (eshkol_scalar_lower_call #f) 94))',
            '(display "SCALAR_INTEGRATION_PASS\\n")',
        ]
        source, binary = root / 'integration.esk', root / 'integration.bin'
        source.write_text('\n'.join(forms) + '\n')
        for mode in ('jit', 'aot'):
            command = [args.runner, '--no-stdlib', '--strict-types', '-L' + args.link_dir]
            command += ['-r', str(source)] if mode == 'jit' else ['-O1', '-o', str(binary), str(source)]
            output, _ = run(command, env)
            if mode == 'aot': output, _ = run([str(binary)], env)
            assert 'SCALAR_INTEGRATION_PASS' in output
            print('INTEGRATION scalar ' + mode, flush=True)
        if args.repl_runner:
            output, _ = run([args.repl_runner, '--machine'], env, input_text='\n'.join(forms) + '\n')
            assert 'SCALAR_INTEGRATION_PASS' in output and 'EREPL FAIL' not in output, output
            print('INTEGRATION scalar staged-repl', flush=True)
    print('RESULT: PASS')


if __name__ == '__main__':
    main()
