#!/usr/bin/env python3
"""Run public subprocess stream cases with actual JIT/AOT and explicit VM rejection."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--child', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--support-library', type=Path,
                        help='Optional test-only native closure for an agent-FFI-off build')
    parser.add_argument('--evidence-dir', type=Path)
    args = parser.parse_args()
    source, build = args.source.resolve(), args.build_dir.resolve()
    env = os.environ.copy()
    env.update(ESHKOL_STREAM_TEST_CHILD=str(args.child.resolve()),
               ESHKOL_JIT_CACHE='0', ESHKOL_PATH=str(source / 'lib'),
               ESHKOL_LIB_DIR=str(build))
    extra = []
    if args.support_library:
        support = args.support_library.resolve(strict=True)
        if not support.name.startswith('lib') or not support.name.endswith('.so'):
            parser.error('test support must be a libNAME.so library')
        env['LD_PRELOAD'] = str(support)
        env['LD_LIBRARY_PATH'] = str(support.parent) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
        extra = ['-L', str(support.parent), '-l', support.name[3:-3]]
    evidence = args.evidence_dir.resolve() if args.evidence_dir else None
    if evidence:
        evidence.mkdir(parents=True, exist_ok=True)

    def run(label, command, *, negative=None, run_env=env):
        done = subprocess.run([str(x) for x in command], cwd=source, env=run_env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=90)
        if evidence:
            (evidence / f'{label}.log').write_text(done.stdout)
        if negative:
            good = done.returncode == 1 and negative in done.stdout
        else:
            good = done.returncode == 0 and not re.search(r'^FAIL:|JIT session error|LLVM ERROR:', done.stdout, re.M)
        if not good:
            raise RuntimeError(f'{label}: exit={done.returncode}\n{done.stdout}')
        return done.stdout

    fixtures = [('subprocess_stream_public', 'PASS: subprocess_stream_public'),
                ('subprocess_stream_large', 'PASS: subprocess_stream_large'),
                ('subprocess_stream_reentry', None)]
    with tempfile.TemporaryDirectory(prefix='eshkol-stream-public-') as tmp:
        for mode in ('jit', 'aot'):
            for name, marker in fixtures:
                fixture = source / 'tests/agent' / f'{name}.esk'
                executable = Path(tmp) / f'{name}-{mode}'
                base = [args.runner.resolve(), '-L', build, *extra]
                if mode == 'jit':
                    command = [*base, '-r', fixture]
                else:
                    run(f'{name}-{mode}-compile', [*base, '-O0', '-o', executable, fixture])
                    command = [executable]
                output = run(f'{name}-{mode}', command,
                             negative='closed session re-entry' if marker is None else None)
                if marker and output.splitlines().count(marker) != 1:
                    raise RuntimeError(f'{name}-{mode}: exact success marker absent\n{output}')
                print(f'PASS: {name} {mode}')
                for line in output.splitlines():
                    if line.startswith(('PUBLIC STREAM CHECKS:', 'LARGE STREAM BYTES:')):
                        print(line)
        # The closed typed-extern table rejects the agent bridge before VM execution.
        vm_env = env.copy()
        vm_env.pop('LD_PRELOAD', None)
        module = Path(tmp) / 'unsupported.eskb'
        run('subprocess-stream-vm-unsupported',
            [args.runner.resolve(), '--profile', 'hosted-vm', '--emit-eskb', module,
             source / 'tests/agent/subprocess_stream_vm_unsupported.esk'],
            negative='unsupported or malformed VM extern declaration', run_env=vm_env)
        print('PASS: standalone VM explicitly rejects unsupported stream externs')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
