# Linux production and staged-package subprocess streaming evidence

The [JSON record](subprocess-streaming-production-2026-11-10.json) preserves the
Linux runs performed on November 10, 2026 (America/New_York), at source revision
`a1f00d30596b77df7a526c2e4444d23f5075bde4`. It was assembled from retained logs
and artifacts on November 11. No new runtime run is claimed by this documentation
update. The [earlier minimal-archive record](subprocess-streaming-2026-11-04.json)
retains its original scope and result hashes; machine paths in summary commands are now placeholders.

Both the production build and staged-package driver completed 23 public checks
in each of JIT and AOT, a large-output fixture with 17,825,829 bytes per stream
and child exit 19, and expected closed-session re-entry rejection. The standalone
VM explicitly refused the unsupported bridge before bytecode emission. Original raw build and per-fixture logs are retained locally. Their original-byte hashes remain in the JSON; raw machine logs are not shipped in the repository. The driver checks child exit codes as well as success/error markers before
printing each PASS and its final summary. Expected rejection logs therefore
contain errors by design.

## Build and run

Use the recorded source revision in a separate checkout, with sibling build,
package, checks and toolchain directories. The compiler was Clang 22.1.8, LLVM
21.1.8, CMake 4.3.3, Linux x86_64 epoll, object ABI v1. The JSON records the toolchain and production dependency configuration. Configure arguments below
reproduce the retained CMake configuration; substitute your LLVM 21.1.8
`llvm-config` path. The raw original configure argv was not separately retained.

```sh
cmake -S . -B ../build-agent-production-20261110 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_CONFIG_EXECUTABLE="$PWD/../toolchain/usr/bin/llvm-config-21" \
  -DESHKOL_REQUIRED_LLVM_MAJOR=21 \
  -DESHKOL_BUILD_AGENT_FFI=ON -DESHKOL_BUILD_TESTS=ON \
  -DESHKOL_BUILD_EXAMPLES=OFF -DESHKOL_GPU_ENABLED=OFF \
  -DESHKOL_BLAS_ENABLED=OFF -DESHKOL_PYTHON_BINDINGS=OFF
cmake --build ../build-agent-production-20261110 \
  --target eshkol-run eshkol-repl stdlib eshkol-subprocess-stream-test eshkol-agent-ffi -j3
env -u LD_PRELOAD python3 scripts/test_subprocess_stream_public.py \
  --runner ../build-agent-production-20261110/eshkol-run \
  --build-dir ../build-agent-production-20261110 \
  --child ../build-agent-production-20261110/eshkol-subprocess-stream-test \
  --evidence-dir ../checks/pr914-production-public
```

The fresh production build had `ESHKOL_BUILD_AGENT_FFI=ON`, with bundled PCRE2,
SQLite, zlib, tree-sitter/grammars, Yoga and Linux curl archives and host platform
TLS dependencies. Its agent archive SHA-256 is
`771e02e5cac1bf1eafd4dddd3b2d27b7fb32c32ebe948437f744d0d631f13bc9`.
It does not replace or reuse the earlier minimal test archive.

## Staged layout and reproduction

The original temporary staging script is not retained. This equivalent recipe
reconstructs the verified runner/archive/module layout used by the package run;
it is a reproduction recipe, not a newly executed qualification. It follows the
runner/archive/module-copy portion of `.github/workflows/release.yml`, rather
than invoking the complete release workflow.

```sh
python3 - <<'PY'
from pathlib import Path
import shutil
build = Path('../build-agent-production-20261110')
package = Path('../package-agent-20261110')
fixtures = Path('../checks/pr914-package-fixtures')
archives = [
    'libeshkol-runtime.a', 'libeshkol-agent-ffi.a', 'eshkol-agent-pcre2.a',
    'eshkol-agent-sqlite3.a', 'eshkol-agent-zlib.a', 'eshkol-agent-tree-sitter.a',
    'eshkol-agent-tree-sitter-grammars.a', 'eshkol-agent-yoga.a', 'eshkol-agent-curl.a',
    'stdlib.o', 'stdlib.bc',
]
(package / 'bin').mkdir(parents=True, exist_ok=True)
for name in ('eshkol-run', 'eshkol-repl'):
    shutil.copy2(build / name, package / 'bin' / name)
for destination in (package / 'lib', package / 'lib/eshkol'):
    destination.mkdir(parents=True, exist_ok=True)
    for name in archives:
        shutil.copy2(build / name, destination / name)
shutil.copytree('lib', package / 'share/eshkol/lib', dirs_exist_ok=True)
for name in ('stdlib.esk', 'math.esk'):
    shutil.copy2(name, package / 'share/eshkol' / name)
(fixtures / 'tests/agent').mkdir(parents=True, exist_ok=True)
for name in ('public', 'large', 'reentry', 'vm_unsupported', 'vm_bridge'):
    name = 'subprocess_stream_' + name + '.esk'
    shutil.copy2(Path('tests/agent') / name, fixtures / 'tests/agent' / name)
link = fixtures / 'lib'
if link.is_symlink():
    link.unlink()
link.symlink_to((package / 'share/eshkol/lib').resolve(), target_is_directory=True)
PY
python3 scripts/stage_third_party_licenses.py \
  --build-dir ../build-agent-production-20261110 --package-dir ../package-agent-20261110
env -u LD_PRELOAD -u LD_LIBRARY_PATH python3 scripts/test_subprocess_stream_public.py \
  --source ../checks/pr914-package-fixtures \
  --runner ../package-agent-20261110/bin/eshkol-run \
  --build-dir ../package-agent-20261110/lib/eshkol \
  --child ../build-agent-production-20261110/eshkol-subprocess-stream-test \
  --evidence-dir ../checks/pr914-package-public
```

All nine archive copies and both standard-library artifacts match across the
production build and both package library directories. The runner and REPL
match the production binaries. The JSON includes their hashes, the child
fixture hash, staged module/license hashes, relevant source hashes and hashes
of the retained logs. Staged modules and copied fixtures also match the source
revision. The package fixture's `lib` points to staged modules, and the driver
sets `ESHKOL_LIB_DIR` to package `lib/eshkol`, disables JIT caching and passes no
support DSO. The package run unsets `LD_PRELOAD` and `LD_LIBRARY_PATH`.

## Qualification limits

This validates package-relative agent/runtime archives and Scheme modules on
this Linux host. Host LLVM/toolchain and platform shared libraries remain
required. The child fixture and Python driver are external test infrastructure.
The complete portable-release verifier and redistributed image-codec runtime
bundle were not exercised. No macOS public JIT/AOT or macOS packaged-runtime
result is claimed. VM streaming execution remains unsupported. Upstream CI
approval and final integration review remain separate maintainer gates.
