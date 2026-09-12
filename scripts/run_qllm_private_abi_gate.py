#!/usr/bin/env python3
"""Link the actual private qLLM ABI and kill bridge mutations. No mock library."""
import argparse
import json
import hashlib
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--qllm-root', required=True, type=Path)
p.add_argument('--output', type=Path)
a = p.parse_args()
qroot = a.qllm_root.resolve()
results = []

def run(command, expected=0):
    proc = subprocess.run(list(map(str, command)), text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=180)
    if (proc.returncode == 0) != (expected == 0):
        raise RuntimeError(f'{command}: unexpected exit {proc.returncode}\n{proc.stdout}')
    return proc

with tempfile.TemporaryDirectory(prefix='eshkol-qllm-abi-') as temporary:
    work = Path(temporary)
    for enabled in ('OFF', 'ON'):
        build = work / enabled.lower()
        run(['cmake', '-S', root/'tests/bridge/qllm_private_abi', '-B', build,
             '-DESHKOL_QLLM_ENABLED='+enabled, '-DESHKOL_QLLM_ROOT='+str(qroot)])
        run(['cmake', '--build', build])
        tested = run(['ctest', '--test-dir', build, '--output-on-failure'])
        results.append({'name': 'real-abi-'+enabled, 'status': 'PASS', 'output': tested.stdout})
    cache = (work/'on/CMakeCache.txt').read_text()
    def setting(name):
        return next(line.split('=', 1)[1] for line in cache.splitlines()
                    if line.startswith(name+':'))
    library = setting('ESHKOL_QLLM_LIBRARY')
    compiler = setting('CMAKE_CXX_COMPILER')
    original = (root/'lib/bridge/qllm_interop.cpp').read_text()
    # The capacity mutant stays within the test's six-element backing array,
    # but violates the caller's declared five-element capacity.
    mutations = [
        ('dtype', 'tensor->options.dtype != QLLM_DTYPE_FLOAT32',
         'tensor->options.dtype == QLLM_DTYPE_FLOAT32', []),
        ('capacity', 'capacity < *size', 'false', []),
        ('eskb-version', 'version != ESKB_VERSION', 'version == ESKB_VERSION', []),
        ('registration', 'qllm_eshkol_register_qllm_natives() != QLLM_SUCCESS',
         'false', ['--jit']),
    ]
    for name, before, after, args in mutations:
        if original.count(before) != 1:
            raise RuntimeError('mutation anchor drift: '+name)
        changed = original.replace(before, after).replace(
            '#include "../backend/eskb_format.h"',
            '#include "'+str(root/'lib/backend/eskb_format.h')+'"')
        source = work/(name+'.cpp')
        source.write_text(changed)
        binary = work/name
        run([compiler, '-std=c++17', '-DESHKOL_HAS_QLLM=1', '-I'+str(root/'inc'),
             '-I'+str(qroot/'include'), source, root/'tests/bridge/qllm_private_abi_test.cpp',
             library, '-Wl,-rpath,'+str(Path(library).parent), '-o', binary])
        tested = run([binary, *args], expected=1)
        if 'FAIL line' not in tested.stdout and not (name == 'registration' and
                "undefined function 'qllm-tensor-create-zeros'" in tested.stdout):
            raise RuntimeError('mutant did not fail a contract assertion: '+name+'\n'+tested.stdout)
        results.append({'name': name, 'status': 'KILLED', 'exit_code': tested.returncode,
                        'output': tested.stdout})
    mismatch = work/'llvm-mismatch'
    mismatch.mkdir()
    (mismatch/'CMakeLists.txt').write_text(
        'cmake_minimum_required(VERSION 3.20)\nproject(QllmMismatch LANGUAGES CXX)\n'
        'set(ESHKOL_QLLM_ENABLED ON)\n'
        'set(ESHKOL_QLLM_ROOT "'+str(qroot)+'")\n'
        'include("'+str(root/'cmake/QllmBridge.cmake')+'")\n'
        'eshkol_qllm_check_host_llvm("999.0.0")\n')
    rejected = run(['cmake', '-S', mismatch, '-B', work/'mismatch-build'], expected=1)
    if 'conflicts with Eshkol LLVM 999.0.0' not in rejected.stdout:
        raise RuntimeError('LLVM compatibility was not proved: '+rejected.stdout)
    results.append({'name': 'mixed-llvm', 'status': 'REJECTED'})
    # Removing the actual registrar library must be observable at link time.
    link_source = work/'missing-symbol.cpp'
    link_source.write_text('#include <semiclassical_qllm/eshkol_bridge.h>\n'
        'int main() { return qllm_eshkol_register_qllm_natives(); }\n')
    missing = run([compiler, '-I'+str(qroot/'include'), link_source,
                   '-o', work/'missing-symbol'], expected=1)
    if 'qllm_eshkol_register_qllm_natives' not in missing.stdout:
        raise RuntimeError('missing qLLM was not rejected by the linker')
    results.append({'name': 'missing-real-library', 'status': 'REJECTED'})
receipt = {'qllm_root': str(qroot), 'qllm_head': run(['git', '-C', qroot, 'rev-parse', 'HEAD']).stdout.strip(),
           'eshkol_base_head': run(['git', '-C', root, 'rev-parse', 'HEAD']).stdout.strip(),
           'source_sha256': {str(path): hashlib.sha256((root/path).read_bytes()).hexdigest()
               for path in [Path('lib/bridge/qllm_interop.cpp'), Path('cmake/QllmBridge.cmake'),
                            Path('tests/bridge/qllm_private_abi_test.cpp')]},
           'results': results}
if a.output:
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2)+'\n')
for result in results:
    print(result['status']+': '+result['name'])
