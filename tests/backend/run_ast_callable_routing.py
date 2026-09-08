#!/usr/bin/env python3
"""Verify both native execution routes against explicit callable assertions."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build = args.build.resolve()
    source = root/'tests/backend/ast_callable_routing_test.esk'
    expected = re.findall(r'\(check "([^"]+)"', source.read_text())
    env = dict(os.environ, ESHKOL_JIT_CACHE='0')
    with tempfile.TemporaryDirectory(prefix='ast-callable-', dir=build) as directory:
        binary = Path(directory)/'program'
        common = [str(build/'eshkol-run'), '-n', '-L', str(build), '-I', str(root/'lib'), '-O0']
        commands = [('jit', [*common, '-r', str(source)], True),
                    ('aot-compile', [*common, str(source), '-o', str(binary)], False),
                    ('aot', [str(binary)], True)]
        for label, command, verify in commands:
            result = subprocess.run(command, cwd=root, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
            log = build/('ast-callable-'+label+'.log');log.write_text(result.stdout)
            observed = re.findall(r'^PASS: (.+)$', result.stdout, re.M)
            if result.returncode or (verify and (observed != expected or 'RESULT: PASS' not in result.stdout)):
                raise SystemExit(f'FAIL: {label}, see {log}')
            print(f'PASS: {label}'+(f' ({len(observed)} assertions)' if verify else ''), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
