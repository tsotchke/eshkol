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
    with binary.open('rb') as stream:
        magic = stream.read(4)
    if magic == b'\x7fELF':
        listing = run([reader, '--needed-libs', str(binary)])
        libraries = re.search(r'NeededLibraries\s*\[([^]]*)\]', listing)
        if not libraries:
            raise RuntimeError(f'Missing ELF dependency records: {binary}')
        return [line.strip() for line in libraries.group(1).splitlines() if line.strip()]
    if magic[:2] == b'MZ':
        listing = run([reader, '--coff-imports', str(binary)])
        libraries = re.findall(r'^\s*Name:\s*(\S+)', listing, re.MULTILINE)
        if not libraries:
            raise RuntimeError(f'Missing PE import records: {binary}')
        return libraries
    raise RuntimeError(f'Unexpected CUDA executable format: {binary}')


def eager_cublas(libraries):
    return [name for name in libraries
            if name.lower().startswith(('libcublas', 'cublas'))]


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
            libraries = imports(binary, args.readobj)
            if eager_cublas(libraries):
                raise RuntimeError(f'CPU executable eagerly imports cuBLAS: {binary}\n{libraries}')
        for command in ([str(output)], [str(runner), '-r', str(source)],
                        [str(runner), '--vm', str(source)]):
            answer = run(command)
            if answer.strip() != '42':
                raise RuntimeError(f'CPU path produced {answer!r}, expected 42')
    print('PASS: CUDA-capable native/JIT/VM CPU paths omit eager cuBLAS imports')


if __name__ == '__main__':
    main()
