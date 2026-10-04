#!/usr/bin/env python3
"""Exercise CUDA-capable CPU native/JIT/VM paths and inspect actual imports."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile


def run(argv, **kwargs):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=300, **kwargs)
    if result.returncode:
        raise RuntimeError(f'{argv[0]} exited {result.returncode}: {result.stdout}\n{result.stderr}')
    return result.stdout


def imports(binary, reader):
    magic = binary.read_bytes()[:4]
    if magic == b'\x7fELF':
        return run([reader, '--needed-libs', str(binary)])
    if magic[:2] == b'MZ':
        return run([reader, '--coff-imports', str(binary)])
    raise RuntimeError(f'Unexpected CUDA executable format: {binary}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner', required=True)
    parser.add_argument('--readobj', required=True)
    args = parser.parse_args()
    runner = Path(args.runner).resolve()
    with tempfile.TemporaryDirectory(prefix='cuda-cpu-link-') as directory:
        root = Path(directory)
        source = root / 'cpu.esk'
        source.write_text('(display (+ 19 23)) (newline)\n')
        output = root / ('cpu.exe' if os.name == 'nt' else 'cpu')
        run([str(runner), str(source), '-o', str(output)])
        for binary in (runner, output):
            listing = imports(binary, args.readobj)
            if re.search(r'cublas(?:lt|64)?[._-]', listing, re.IGNORECASE):
                raise RuntimeError(f'CPU executable eagerly imports cuBLAS: {binary}\n{listing}')
        for command in ([str(output)], [str(runner), '-r', str(source)],
                        [str(runner), '--vm', str(source)]):
            answer = run(command)
            if answer.strip() != '42':
                raise RuntimeError(f'CPU path produced {answer!r}, expected 42')
    print('PASS: CUDA-capable native/JIT/VM CPU paths omit eager cuBLAS imports')


if __name__ == '__main__':
    main()
