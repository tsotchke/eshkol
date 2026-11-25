"""Reject malformed whole programs before side effects or ESKB publication."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner', required=True)
    parser.add_argument('--vm', required=True)
    parser.add_argument('--link-dir', required=True)
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, ESHKOL_PATH=str(source_root / 'lib'), ESHKOL_VM_NO_DISASM='1', ESHKOL_JIT_CACHE='0')
    malformed = {
        'list': '(list 1 2',
        'nested-list': '(list (list 1 2)',
        'extra-close': ')',
        'vector': '#(1 2',
        'nested-vector': '(list #(1 2)',
        'string': '"unterminated',
        'escaped-quote-eof': r'"unterminated\"',
        'dangling-string-escape': '"unterminated' + chr(92),
        'quote': "'",
        'quasiquote': '`',
        'unquote': ',',
        'splice': ',@',
        'symbol': '|unterminated',
        'character': '#\\',
        'interpolation': '"prefix ~{(+ 1 2)',
        'interpolation-with-prior-parts': '"first ~{(+ 1 2)} next ~{(',
    }
    stages = 0
    with tempfile.TemporaryDirectory(prefix='eshkol-parser-refusal-') as directory:
        root = Path(directory)
        for name, tail in malformed.items():
            source = root / (name + '.esk')
            # Even if the valid prefix compiles, no part may execute.
            source.write_text('(display "PREFIX_EXECUTED\\n")\n' + tail)
            for mode in ('vm-source', 'eskb'):
                artifact = root / (name + '.eskb')
                command = [args.vm, str(source)] if mode == 'vm-source' else [
                    args.runner, '--profile', 'hosted-vm', '--emit-eskb', str(artifact), str(source), '-L' + args.link_dir]
                result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
                output = result.stdout + result.stderr
                assert result.returncode > 0, (name, mode, result.returncode, output)
                assert 'PREFIX_EXECUTED' not in result.stdout, (name, mode, output)
                assert 'ERROR' in output or 'error' in output, (name, mode, output)
                if mode == 'eskb': assert not artifact.exists(), (name, 'rejected source published bytecode')
                stages += 1
        source = root / 'valid.esk'
        source.write_text(r'''(require stdlib)
(define quoted '(a (b c)))
(define vector '#(1 2))
(define empty '#())
(define empty-list '())
(define escaped "a \"b\" \\z")
(define text "parentheses )(|")
(define ch #\))
(define |odd name| 17)
(if (not (and (= (length quoted) 2) (= (vector-length vector) 2)
              (= (vector-length empty) 0) (null? empty-list)
              (= (string-length escaped) 8) (= (char->integer (string-ref escaped 2)) 34)
              (= (char->integer (string-ref escaped 6)) 92) (= (char->integer ch) 41)
              (= |odd name| 17) (= (char->integer (string-ref text 14)) 124)))
    (error "valid reader control failed"))
(display "CONTROL_PASS\n") ; complete final datum followed by comment/EOF
''')
        for mode in ('vm-source', 'eskb', 'native-jit', 'native-aot'):
            artifact = root / ('valid.eskb' if mode == 'eskb' else 'valid.bin')
            if mode == 'vm-source': command = [args.vm, str(source)]
            elif mode == 'eskb': command = [args.runner, '--profile', 'hosted-vm', '--emit-eskb', str(artifact), str(source), '-L' + args.link_dir]
            elif mode == 'native-jit': command = [args.runner, '-r', str(source), '-L' + args.link_dir]
            else: command = [args.runner, '-O1', '-o', str(artifact), str(source), '-L' + args.link_dir]
            result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
            assert result.returncode == 0, (mode, result.stdout, result.stderr)
            if mode in ('eskb', 'native-aot'):
                result = subprocess.run(([args.vm, str(artifact)] if mode == 'eskb' else [str(artifact)]), env=env, text=True, capture_output=True, timeout=30)
            assert result.returncode == 0 and result.stdout.count('CONTROL_PASS\n') == 1, (mode, result.stdout, result.stderr)
            stages += 1
    assert len(malformed) == 16 and stages == 36
    print('PARSER_REFUSAL malformed=16 routes=2 valid_routes=4 stages=36')
    print('RESULT: PASS')


if __name__ == '__main__':
    main()
