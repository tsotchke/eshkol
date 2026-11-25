"""Native locality witness: cold graph growth must not introduce a per-tick scan/copy."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def program(cold_nodes):
    bindings = ' '.join('(c%d (+ %s %d))' % (i, 'cold' if i == 1 else 'c%d' % (i - 1), i)
                        for i in range(1, cold_nodes + 1))
    return '''(require stdlib)
(define passed 0)
(define (check ok) (if ok (set! passed (+ passed 1)) (error "FAIL: incremental locality")))
(define initial (differentiate 'incremental (lambda (hot cold) (let* (%s) (+ (* hot hot) c%d))) (vector 2 7)))
(define info (initial 'info #f))
(define height (vector-ref info 3))
(check (= (vector-ref info 0) %d))
(let ticks ((i 0) (state initial)) (if (< i 64)
  (let* ((step (state 'step (vector 1 0))) (next (vector-ref step 0)) (work (vector-ref step 2))
         (x (+ 3 i)) (batch (+ (* x x) %d)))
    (check (= (next 'output #f) batch))
    (check (= (vector-ref work 0) 2))
    (check (= (vector-ref work 1) 3))
    (check (= (vector-ref work 4) 2))
    (check (= (vector-ref work 5) 2))
    (check (<= (vector-ref work 3) (* 12 height)))
    (check (<= (vector-ref work 2) (* 24 (+ height 1))))
    (check (= (state 'output #f) (- batch (- (* x x) (* (- x 1) (- x 1))))))
    (ticks (+ i 1) next))))
(check (= (initial 'output #f) %d))
(display "LOCALITY cold=") (display %d) (display " nodes=") (display (vector-ref info 0))
(display " height=") (display height) (display " checks=") (display passed) (newline)
(display "RESULT: PASS\\n")
''' % (bindings, cold_nodes, 2 * cold_nodes + 4, 7 + cold_nodes * (cold_nodes + 1) // 2,
       11 + cold_nodes * (cold_nodes + 1) // 2, cold_nodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner', required=True)
    parser.add_argument('--link-dir', required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='eshkol-incremental-locality-') as directory:
        root = Path(directory)
        for count in (16, 64, 256):
            source, binary = root / ('cold-%d.esk' % count), root / ('cold-%d.bin' % count)
            source.write_text(program(count))
            for mode in ('jit', 'aot'):
                command = [args.runner, '--strict-types', '-r', str(source), '-L' + args.link_dir] if mode == 'jit' else [
                    args.runner, '--strict-types', '-O', '1', '-o', str(binary), str(source), '-L' + args.link_dir]
                result = subprocess.run(command, text=True, capture_output=True, timeout=300)
                output = result.stdout + result.stderr
                assert result.returncode == 0, (count, mode, output)
                if mode == 'aot':
                    result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=120)
                    output += result.stdout + result.stderr
                assert result.returncode == 0 and 'RESULT: PASS' in output and 'FAIL:' not in output, (count, mode, output)
                print(mode, next(line for line in output.splitlines() if line.startswith('LOCALITY ')), flush=True)
    print('RESULT: PASS')


if __name__ == '__main__':
    main()
