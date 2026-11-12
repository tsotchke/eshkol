#!/usr/bin/env bash
# Standalone hosted native qualification; no LLVM compiler or agent dependency bundle.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
OUT="${ESHKOL_SOCKET_QUALIFICATION_DIR:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/eshkol-socket-native-$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$OUT"
exec python3 - "$ROOT" "$OUT" <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

root, output = (Path(x).resolve() for x in sys.argv[1:])
system = platform.system()
if system not in ('Linux', 'Darwin'):
    raise SystemExit('Unsupported: this qualification needs Linux epoll or macOS kqueue')
backend = 'epoll' if system == 'Linux' else 'kqueue'
sanitize = os.environ.get('ESHKOL_SOCKET_SANITIZE', '0') == '1'
cc, cxx = os.environ.get('CC', 'clang'), os.environ.get('CXX', 'clang++')
flags = ['-O1', '-g', '-Wall', '-Wextra', '-pthread', '-I' + str(root / 'inc')]
sanflags = ['-fsanitize=address,undefined', '-fno-omit-frame-pointer'] if sanitize else []
cflags = ['-std=c11', '-D_GNU_SOURCE', '-D_XOPEN_SOURCE=700', *flags, *sanflags]
if system == 'Darwin':
    cflags += ['-D_DARWIN_C_SOURCE=1']
cxxflags = ['-std=c++17', *flags, *sanflags]
env = os.environ.copy()
for name in ('LD_PRELOAD', 'DYLD_INSERT_LIBRARIES', 'ESHKOL_UNIX_SOCKET_TEST_PENDING_PATH'):
    env.pop(name, None)
env['ASAN_OPTIONS'] = ('detect_leaks=1' if system == 'Linux' else 'detect_leaks=0') + ':halt_on_error=1'
env['UBSAN_OPTIONS'] = 'halt_on_error=1:print_stacktrace=1'
records = []
sources = [
    'tests/agent/run_unix_socket_stream_native.sh', '.github/workflows/unix-socket-native.yml',
    'tests/agent/unix_socket_stream_test.c', 'tests/agent/unix_socket_cleanup_test.c',
    'tests/agent/unix_socket_stream_fixture_support.c',
    'tests/agent/unix_socket_stream_pending_control.c',
    'tests/core/resource_explicit_loop_test.c',
    'tests/agent/run_subprocess_stream_test.sh', 'tests/agent/subprocess_stream_test.c',
    'tests/agent/subprocess_stream_clock_test.c',
    'tests/agent/subprocess_stream_watch_failure_test.c',
    'lib/agent/c/agent_unix_socket_stream.c', 'lib/agent/c/agent_subprocess.c',
    'lib/core/resource_lease.c', 'lib/core/runtime_handle.cpp',
    'lib/core/resource_handle_internal.h', 'lib/core/event_loop.c',
    'lib/core/event_loop_' + backend + '.c', 'lib/core/event_loop_internal.h',
    'inc/eshkol/core/unix_socket_stream.h', 'inc/eshkol/core/resource_lease.h',
    'inc/eshkol/core/event_loop.h', 'inc/eshkol/core/handle.h',
    'inc/eshkol/core/subprocess_stream.h',
]
if system == 'Linux':
    sources += ['tests/agent/unix_socket_stream_regression_test.c']
source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
receipt = {
    'schema': 1, 'status': 'RUNNING',
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'platform': system + ' ' + platform.machine(), 'backend': backend,
    'sanitizers': ['address', 'undefined'] if sanitize else [],
    'leak_detection': sanitize and system == 'Linux',
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'working_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)),
    'source_sha256': source_hashes, 'commands': records,
    'scope': 'Actual hosted native socket, common lease/loop and PROCESS suites; no Eshkol JIT/AOT or package/release claim',
    'pending_control': 'Darwin private-path EAGAIN only in three pending cases and its fixture selftest; Linux uses the real backlog',
}

def save():
    (output / 'RESULT.json').write_text(json.dumps(receipt, indent=2) + '\n')

def run(name, argv, *, run_env=None, timeout=60):
    log = output / (name + '.log')
    started = time.monotonic()
    with log.open('wb') as stream:
        process = subprocess.Popen([str(x) for x in argv], cwd=root,
                                   env=run_env or env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
            code = 'TIMEOUT'
    records.append({'name': name, 'argv': [str(x) for x in argv], 'exit_code': code,
                    'seconds': round(time.monotonic() - started, 3), 'log': log.name,
                    'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()})
    save()
    print(name + ': ' + str(code), flush=True)
    if code != 0:
        print(log.read_text(errors='replace')[-12000:], flush=True)
        raise RuntimeError(name + ' did not pass')

try:
    for compiler in (cc, cxx):
        run(Path(compiler).name + '-version', [compiler, '--version'])
    runtime = ['lib/agent/c/agent_unix_socket_stream.c', 'lib/core/resource_lease.c',
               'lib/core/event_loop.c', 'lib/core/event_loop_' + backend + '.c']
    objects = []
    for number, source in enumerate(runtime):
        target = output / ('runtime-' + str(number) + '.o')
        run('compile-runtime-' + str(number), [cc, *cflags, '-c', root / source, '-o', target])
        objects.append(target)
    handle = output / 'handle.o'
    run('compile-handles', [cxx, *cxxflags, '-c', root / 'lib/core/runtime_handle.cpp', '-o', handle])
    objects.append(handle)
    extension = '.so' if system == 'Linux' else '.dylib'
    observation = output / ('libeshkol-socket-observation' + extension)
    shared = ['-shared'] if system == 'Linux' else ['-dynamiclib']
    run('build-observation', [cc, *cflags, '-fPIC', *shared,
                            root / 'tests/agent/unix_socket_stream_fixture_support.c', '-o', observation])
    control = None
    if system == 'Darwin':
        control = output / 'libeshkol-socket-pending-control.dylib'
        run('build-pending-control', [cc, *cflags, '-fPIC', '-dynamiclib',
                                     root / 'tests/agent/unix_socket_stream_pending_control.c', '-o', control])
    native = output / 'socket-native'
    native_object = output / 'socket-native.o'
    run('compile-native', [cc, *cflags, '-c', root / 'tests/agent/unix_socket_stream_test.c', '-o', native_object])
    run('link-native', [cxx, '-pthread', *sanflags, native_object, *objects,
                        observation, '-Wl,-rpath,' + str(output), '-o', native])
    cases = ('connect_accept partial_slow_peer peer_close_epipe deadline_accept deadline_connect '
             'cancellation nul_and_large_payload descriptor_count_check capability_denial path_lifecycle '
             'destination_errors connect_cancellation capability_revocation loop_lifetime borrowed_close zero_budget').split()
    for case in ['--peer-fixture-selftest', *cases]:
        case_env = env.copy()
        if control and case in ('--peer-fixture-selftest', 'deadline_connect', 'connect_cancellation', 'zero_budget'):
            case_env['DYLD_INSERT_LIBRARIES'] = str(control)
        run('socket-' + case.lstrip('-'), [native, case], run_env=case_env)
    cleanup_object, cleanup = output / 'cleanup.o', output / 'socket-cleanup'
    run('compile-cleanup', [cc, *cflags, '-c', root / 'tests/agent/unix_socket_cleanup_test.c', '-o', cleanup_object])
    run('link-cleanup', [cxx, '-pthread', *sanflags, cleanup_object, *objects, '-o', cleanup])
    run('socket-cleanup', [cleanup])
    common_objects = [handle, objects[3]]
    for number, source in enumerate(('tests/core/resource_explicit_loop_test.c',
                                     'lib/core/resource_lease.c', 'lib/core/event_loop.c')):
        target = output / ('common-' + str(number) + '.o')
        run('compile-common-' + str(number), [cc, *cflags, '-DESHKOL_RESOURCE_TESTING', '-c', root / source, '-o', target])
        common_objects.append(target)
    common = output / 'common-explicit-loop'
    run('link-common', [cxx, '-pthread', *sanflags, *common_objects, '-o', common])
    run('common-explicit-loop', [common])
    if system == 'Linux':
        target, executable = output / 'regression.o', output / 'socket-regressions'
        run('compile-regressions', [cc, *cflags, '-c', root / 'tests/agent/unix_socket_stream_regression_test.c', '-o', target])
        wraps = ['-Wl,--wrap=' + name for name in ('socket', 'accept4', 'recv', 'send', 'fchmodat', 'unlinkat', 'connect')]
        run('link-regressions', [cxx, '-pthread', *sanflags, target, *objects, *wraps, '-o', executable])
        run('socket-regressions', [executable])
    process_env = env.copy()
    process_env.update(CC=cc, CXX=cxx, ESHKOL_STREAM_SANITIZE='1' if sanitize else '0')
    run('process-stream-suite', ['bash', root / 'tests/agent/run_subprocess_stream_test.sh'],
        run_env=process_env, timeout=180)
    assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
               for name, digest in source_hashes.items()), 'qualification sources changed during execution'
    receipt['status'] = 'PASS'
except (OSError, RuntimeError, AssertionError, subprocess.SubprocessError) as error:
    receipt.update(status='FAIL', reason=str(error))
finally:
    receipt['artifacts_sha256'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir()) if path.is_file() and path.suffix not in ('.json', '.log')
    }
    save()
print(json.dumps({'status': receipt['status'], 'backend': backend, 'sanitized': sanitize,
                  'receipt': str(output / 'RESULT.json')}), flush=True)
raise SystemExit(0 if receipt['status'] == 'PASS' else 1)
PY
