#!/usr/bin/env python3
"""Host-compile unavailable socket branches; this is not target-platform execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--cc', default=os.environ.get('CC', 'clang'))
    parser.add_argument('--nm', default=os.environ.get('NM', 'nm'))
    args = parser.parse_args()
    source = args.source_dir.resolve()
    args.work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='socket-host-refusal-', dir=args.work_root)).resolve()
    production = source / 'lib/agent/c/agent_unix_socket_stream.c'
    fixture = source / 'tests/agent/unix_socket_stream_unsupported_test.c'
    rows = []
    inputs = (production, fixture, source / 'inc/eshkol/core/unix_socket_stream.h',
              source / 'inc/eshkol/core/resource_lease.h', source / 'inc/eshkol/core/handle.h',
              source / 'tests/agent/test_unix_socket_unsupported.py')
    forbidden = {'socket', 'socketpair', 'accept', 'accept4', 'connect', 'bind', 'listen',
                 'recv', 'send', 'read', 'write', 'poll', 'select', 'epoll_wait', 'kqueue',
                 'kevent', 'nanosleep', 'sched_yield', 'pipe', 'pipe2', 'fcntl', 'getsockopt'}
    flags = ['-std=c11', '-O1', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
             '-I' + str(source / 'inc')]

    def run(label, argv):
        done = subprocess.run([str(x) for x in argv], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=30)
        (work / (label + '.log')).write_text(done.stdout)
        rows.append({'command': [str(x) for x in argv], 'exit_code': done.returncode,
                     'log': label + '.log'})
        if done.returncode:
            raise RuntimeError(label + ': ' + done.stdout[-8000:])
        return done.stdout

    result = {'kind': 'HOST_SOURCE_BRANCH_CONTRACT', 'status': 'FAIL', 'records': rows,
              'scope': 'Host compiler selects _WIN32, __EMSCRIPTEN__ and ESHKOL_VM_WASM branches; no Windows, bytecode VM or WASM target execution claim',
              'sources_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in inputs}}
    try:
        for macro in ('_WIN32', '__EMSCRIPTEN__', 'ESHKOL_VM_WASM'):
            label = macro.strip('_').lower()
            obj, executable = work / (label + '.o'), work / label
            run(label + '-compile', [args.cc, *flags, '-D' + macro, '-c', production, '-o', obj])
            names = {line.split()[-1].lstrip('_').split('@', 1)[0].split('$', 1)[0] for line in
                     run(label + '-undefined', [args.nm, '-u', obj]).splitlines() if line.split()}
            if names & forbidden:
                raise RuntimeError(macro + ': unsupported branch references native I/O: ' + str(sorted(names & forbidden)))
            run(label + '-link', [args.cc, *flags, '-D' + macro, fixture, obj, '-o', executable])
            output = run(label + '-run', [executable])
            if output.splitlines().count('PASS: HOST source-branch socket refusal; not target-platform execution') != 1:
                raise RuntimeError(macro + ': exact refusal marker absent')
        if any(hashlib.sha256((source / name).read_bytes()).hexdigest() != digest
               for name, digest in result['sources_sha256'].items()):
            raise RuntimeError('qualification sources changed during execution')
        result['status'] = 'PASS'
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        result['reason'] = str(error)
    finally:
        (work / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'kind': result['kind'], 'receipt': str(work / 'RESULT.json')}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
