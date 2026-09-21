---
kind: project
status: current
owner-area: build
since: v1.3.5
sources:
  - CMakeLists.txt
  - cmake/LLVMToolchain.cmake
  - exe/eshkol-run.cpp
  - lib/repl/repl_jit.cpp
  - lib/core/runtime_math_compat.c
  - lib/ffi/eshkol_ffi.cpp
  - bindings/python/eshkol_module.cpp
  - tests/bindings/python_capsule_lifetime_test.py
  - scripts/generate_wasm_import_glue.py
  - scripts/check_wasm_imports.py
  - scripts/lib/evidence_paths.sh
  - scripts/check_changelog_completeness.py
  - docs/reference/runtime/environment-variables.md
  - docs/platform/BUILD_NOTES.md
  - docs/KNOWN_ISSUES.md
---
# Troubleshooting

Build and run problems the repository already knows the answer to. Each entry
is **symptom** (the message as the tool prints it), **cause**, **fix**.

For language-level limits see [KNOWN_ISSUES.md](KNOWN_ISSUES.md); for
per-platform build detail see [platform/BUILD_NOTES.md](platform/BUILD_NOTES.md);
for every variable named here see
[reference/runtime/environment-variables.md](reference/runtime/environment-variables.md).

Contents:
[LLVM](#llvm-version-discovery) ·
[Host compiler](#host-compiler) ·
[Standard library and modules](#standard-library-and-module-discovery) ·
[Caches](#stale-jit-and-module-caches) ·
[Stack](#stack-size) ·
[Links](#native-links) ·
[Python bindings](#python-bindings) ·
[Windows](#windows) ·
[WebAssembly](#webassembly) ·
[Type warnings](#type-warnings-after-upgrading) ·
[Gates](#release-and-gate-failures-met-locally)

## LLVM version discovery

The build pins **one** LLVM major and aborts on any other. The default pin is
LLVM 21, which is what the release packages are built with. The compiler
source builds against LLVM 18 through 24.

### `LLVM 21 llvm-config not found`

**Symptom**

```
LLVM 21 llvm-config not found. Install LLVM 21 and set LLVM_CONFIG_EXECUTABLE.
```

**Cause.** `cmake/LLVMToolchain.cmake` looks for `llvm-config-21` and then
`llvm-config`, first in the platform's conventional prefixes
(`/opt/homebrew/opt/llvm@21/bin` and `/usr/local/opt/llvm@21/bin` on macOS,
`/usr/lib/llvm-21/bin` and `/usr/local/lib/llvm-21/bin` on Linux) and then on
`PATH`. None was found.

**Fix.** Install LLVM 21, or point the build at the one you have:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_CONFIG_EXECUTABLE="$(brew --prefix llvm@21)/bin/llvm-config"   # macOS
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_CONFIG_EXECUTABLE=/usr/lib/llvm-21/bin/llvm-config             # Linux
```

### `Expected LLVM 21, got ...`

**Symptom**

```
Expected LLVM 21, got 18.1.8 from /usr/bin/llvm-config
```

**Cause.** An `llvm-config` was found, but its major version is not the
pinned one. The check exists so a mixed toolchain cannot be assembled by
accident.

**Fix.** Either select the LLVM 21 `llvm-config` as above, or build against
the version you have by changing the pin. Any major from 18 through 24 is
accepted by the source:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DESHKOL_REQUIRED_LLVM_MAJOR=18
```

Reconfigure in a fresh build directory when changing the pin, because the
resolved `LLVM_CONFIG_EXECUTABLE` is cached.

### Windows: `LLVM 21 CMake package not found`

See [Windows](#windows).

## Host compiler

> Eshkol v1.3.5-evolve is built and verified with **GCC 13** and with
> **Clang/LLVM 21**. Building the compiler itself with **GCC 15 is not
> supported in this release**.
> ([BUILD_NOTES.md](platform/BUILD_NOTES.md#supported-host-compilers);
> [KNOWN_ISSUES.md](KNOWN_ISSUES.md#supported-host-compilers) carries the same
> statement.)

### The system default is GCC 15

**Symptom.** The build uses GCC 15 because it is the distribution default.

**Cause.** GCC 15 is not a supported host compiler in this release.

**Fix.** Select a supported compiler for the whole build tree, C and C++
together, in a fresh build directory:

```sh
CC=gcc-13 CXX=g++-13 cmake -B build -DCMAKE_BUILD_TYPE=Release
# or
CC=clang-21 CXX=clang++-21 cmake -B build -DCMAKE_BUILD_TYPE=Release
```

The host compiler builds Eshkol. It does not generate code for Eshkol
programs, which are emitted through LLVM.

### `CUDA ... does not support GNU ...`

**Symptom**

```
CUDA 12.4 does not support GNU 14.2.0. Reconfigure the entire build with a
compatible compiler (for example CC=gcc-13 CXX=g++-13). Selecting only
CMAKE_CUDA_HOST_COMPILER is unsafe because it can mix libstdc++ ABI/library
search paths at the final link.
```

**Cause.** A CUDA toolkit older than 13.0 was found together with GCC 14 or
newer.

**Fix.** As the message says: reconfigure the whole tree with
`CC=gcc-13 CXX=g++-13`. To build without CUDA instead, pass
`-DESHKOL_GPU_ENABLED=OFF`.

### `A real GPU backend was required, but neither Metal nor CUDA was found`

**Cause.** `-DESHKOL_REQUIRE_GPU_BACKEND=ON` was set and no GPU SDK was found.
The build refuses to compile the CPU stub under a GPU label.

**Fix.** Install the GPU SDK, or drop `ESHKOL_REQUIRE_GPU_BACKEND`. A default
build (`ESHKOL_GPU_ENABLED=ON`) falls back to the CPU stub on its own.

## Standard library and module discovery

Every install artifact is resolved in one order, highest precedence first:
`$ESHKOL_LIB_DIR` (native artifacts) or `$ESHKOL_PATH` (module sources); `-L`
or `-I` directories; the install the running compiler belongs to; the working
directory and its `build/` trees; the system prefixes. The full rule is in
[environment-variables.md](reference/runtime/environment-variables.md#resolution-precedence).

### `Module '...' not found`

**Symptom**

```
ERROR: Module 'my.module' not found
ERROR:   Searched:
ERROR:     - /path/to/project/my.module.esk
ERROR:     - /path/to/install/lib/my/module.esk
ERROR:     - $ESHKOL_PATH entries
ERROR: Unresolved module dependency; refusing to compile an incomplete program
```

**Cause.** `(require my.module)` names a module that is in none of the searched
locations. A dotted name maps to a path: `my.module` is `my/module.esk`.

**Fix.** Put the module's root on the search path, with `-I` or with
`ESHKOL_PATH` (a path list: `:` separated, `;` on Windows):

```sh
eshkol-run -I /path/to/modules -r main.esk
ESHKOL_PATH=/path/to/modules eshkol-run -r main.esk
```

When running a compiler out of a build tree against a different source
checkout, name that checkout's library tree: `ESHKOL_PATH=/path/to/checkout/lib`.

### `Module '...' not found: its define-library form is at line N, below this import`

**Cause.** The library is defined in the same file, below the `import`. A
library is defined by its `define-library` form, so importing it earlier is a
forward reference, not a search-path problem (R7RS-small 5.6.1).

**Fix.** Move the `define-library` above the `import`, or put the library in
its own file.

### `Could not find libeshkol-runtime.a or legacy libeshkol-static.a`

**Symptom**

```
ERROR: Could not find libeshkol-runtime.a or legacy libeshkol-static.a
ERROR: Searched, in order: $ESHKOL_LIB_DIR, every -L directory, the directory
       holding this compiler (and its ../lib, ../lib/eshkol), the working
       directory's build trees, then the system prefixes
```

**Cause.** An ahead-of-time link needs the runtime archive and `stdlib.o`, and
the compiler binary was moved away from them, or only `eshkol-run` was built.

**Fix.** Build the artifacts, or say where they are:

```sh
cmake --build build --target stdlib          # in a source build
ESHKOL_LIB_DIR=/path/to/lib/eshkol eshkol-run program.esk -o program
eshkol-run -L /path/to/lib/eshkol program.esk -o program
```

`ESHKOL_LIB_DIR` has the highest precedence, and its `eshkol/` subdirectory is
searched with it. When an artifact comes from a system location, `eshkol-run`
says so on stderr with the path it used; an archive built from a different
Eshkol version produces a warning.

### `Warning: stdlib.bc not found — stdlib symbol discovery unavailable`

**Cause.** The JIT discovers standard-library symbols from `stdlib.bc`, and it
is not beside `stdlib.o` in any searched location.

**Fix.** Keep `stdlib.o` and `stdlib.bc` together; both are produced by the
`stdlib` target and both ship in `lib/eshkol/` of a package. Point
`ESHKOL_LIB_DIR` at the directory that holds the pair.

## Stale JIT and module caches

`eshkol-run -r` keeps a persistent run cache: the file is compiled once to a
native binary, and later runs re-execute it. The key covers the source bytes,
the Eshkol and LLVM versions, the `eshkol-run` binary fingerprint, the relevant
flags, the `stdlib.bc` and `stdlib.o` fingerprints, and the include, library
and linked-library flags, so ordinary edits and rebuilds miss the cache on
their own. See [JIT internals](reference/runtime/jit-internals.md).

| Cache | Location | Control |
|-------|----------|---------|
| Run cache (`-r`) | `$ESHKOL_JIT_CACHE_DIR`, else `$XDG_CACHE_HOME/eshkol/jit`, else `$HOME/.cache/eshkol/jit`; `%LOCALAPPDATA%\eshkol\jit` on Windows | `ESHKOL_JIT_CACHE=0` disables it; `ESHKOL_JIT_CACHE_TRACE=1` prints the decision |
| AOT module cache (`-c` objects) | `$ESHKOL_AOT_MODULE_CACHE_DIR`, else `$XDG_CACHE_HOME/eshkol/modules`, else `$HOME/.cache/eshkol/modules`; `%LOCALAPPDATA%\eshkol\modules` on Windows | `ESHKOL_AOT_MODULE_CACHE_TRACE=1` prints hits and misses |
| Stdlib JIT object | beside `stdlib.bc`, named `stdlib-jit-v4-<hash>-<triple>-<abi>.o` | keyed on the `stdlib.bc` content, the process triple and the object ABI |

### A run does not seem to pick up a change

**Diagnose.** Ask the cache what it did:

```sh
ESHKOL_JIT_CACHE_TRACE=1 eshkol-run -r program.esk
```

```
[jit-cache] hit cca516e3cf10964f830e002a3b0103e3b173d00771b66fde8b59dc6d887515ac
```

The trace prints `hit`, `miss`, `store` or `bypass <reason>`. The cache is
bypassed when it is disabled, when the program uses `eval` or `compile`, with
more than one input file or a debug or dump flag, with `-D` defines, and while
language-coverage tracing is on.

**Fix.** Run once without the cache, and if that differs, clear it:

```sh
ESHKOL_JIT_CACHE=0 eshkol-run -r program.esk
rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/eshkol/jit"
```

Entries older than 30 days are pruned and the cache is held to 1 GiB, so
clearing it is always safe. For isolated runs (tests, parallel jobs, a
read-only home), give each its own directory with `ESHKOL_JIT_CACHE_DIR`.

### `Ignoring ESHKOL_AOT_MODULE_CACHE_DIR under a forbidden temp root`

**Cause.** The AOT module cache refuses a directory under `/tmp` or
`/private/tmp`; the compiler does not create cache artifacts under the system
temporary directories.

**Fix.** Name a durable directory, or unset the variable to use the default.

### `-r: native link of '...' failed; refusing to fall back to a reduced in-process run`

**Cause.** The cache build links a standalone binary. A link failure under
`-r` is fatal by design: falling back to the in-process JIT would run a
program whose missing native symbols happen to resolve inside `eshkol-run`
and would certify a build that never linked.

**Fix.** Read the linker diagnostics printed above the message and see
[Native links](#native-links). `ESHKOL_JIT_CACHE=0` runs in-process and is a
way to keep working while the link is being fixed, not a fix.

## Stack size

### `eshkol: stack overflow: recursion depth exceeded ...`

**Symptom**

```
eshkol: stack overflow: recursion depth exceeded the 512 MiB stack (ESHKOL_STACK_SIZE); use tail recursion, or raise ESHKOL_STACK_SIZE and the OS stack limit to allow deeper recursion
```

Exit status 121. The size named is whatever `ESHKOL_STACK_SIZE` resolved to.

**Cause.** Non-tail recursion reached the native stack guard. Hosted JIT and
AOT entry code raise the soft `RLIMIT_STACK` when the operating system allows
it and check headroom at every generated function entry, so deep recursion
ends in this diagnostic rather than in a bare signal.

**Fix.** Make the recursion a tail call, which runs in constant stack:

```scheme
;; Non-tail: the addition happens after the recursive call returns.
(define (count-up n)
  (if (= n 0)
      0
      (+ 1 (count-up (- n 1)))))

;; Tail: the recursive call is the last thing the procedure does.
(define (count-up-tail n acc)
  (if (= n 0)
      acc
      (count-up-tail (- n 1) (+ acc 1))))

(display (count-up-tail 100000 0))
(newline)
(display (count-up 100000))
(newline)
```

With `ESHKOL_STACK_SIZE=1M` the tail form prints `100000` and the non-tail form
ends in the diagnostic above with exit status 121, under both `eshkol-run -r`
and an ahead-of-time binary; with the default ceiling both print `100000`.

Or raise the ceiling. `ESHKOL_STACK_SIZE` accepts `K`, `M` and `G` suffixes and
has a 1 MiB floor:

```sh
ESHKOL_STACK_SIZE=1G eshkol-run -r program.esk
```

On Linux the initial thread's reachable stack is fixed from the soft
`ulimit -s` at process launch, so raise that in the launching shell too:

```sh
ulimit -s 1048576 && ESHKOL_STACK_SIZE=1G ./program
```

Parallel workers have their own stack, sized by `ESHKOL_WORKER_STACK_BYTES`
(default 16 MiB). `ESHKOL_MAX_STACK` is a separate, optional software
recursion-depth ceiling. Which tail positions are proper tail calls is in
[tail-calls.md](reference/language/tail-calls.md).

## Native links

### `Linking timed out after 300 seconds and was aborted`

**Fix.** Raise or disable the limit: `ESHKOL_LINK_TIMEOUT_SECONDS=900`, or `0`
for no limit. `ESHKOL_OBJECT_EMIT_TIMEOUT_SECONDS` bounds object emission the
same way and is unbounded by default.

### The C++ driver used for links is not found, or is the wrong one

**Cause.** Ahead-of-time and cache links run a C++ driver: the one recorded at
build time when it still exists, otherwise a `clang++` or `c++` found on
`PATH`. A package moved to a machine whose LLVM lives elsewhere may find none.

**Fix.** `ESHKOL_CXX_COMPILER=/path/to/clang++ eshkol-run program.esk -o program`

### An undefined reference to `roundeven`

**Cause.** Scheme's `round` is ties-to-even, which the back end emits as the
LLVM `roundeven` intrinsic. On AArch64 that is one instruction. On x86-64
without SSE4.1 in the baseline it is a call to the C library's `roundeven`, a
C23 function that some platform C libraries do not provide.

**Fix.** None is needed with the shipped runtime: configuration checks for the
symbol and, where the platform lacks it, the runtime archive supplies it from
`lib/core/runtime_math_compat.c`. The configure log says which case applies:

```
-- roundeven: provided by the platform C library
-- roundeven: supplied by lib/core/runtime_math_compat.c
```

If the reference is undefined, the link is picking up a runtime archive from a
different build. Check which archive `eshkol-run` reports, and set
`ESHKOL_LIB_DIR` to the one that belongs to the compiler.

### AArch64 Linux links fail to find `ld.lld`

**Cause.** AArch64 links use `lld`, and a versioned package installs only
`ld.lld-21`.

**Fix.** `sudo ln -sf "$(command -v ld.lld-21)" /usr/local/bin/ld.lld`

## Python bindings

The bindings are a pybind11 extension, off by default. The API is documented
in [reference/bindings/python.md](reference/bindings/python.md) and the build
in [BUILD_NOTES.md](platform/BUILD_NOTES.md#python-bindings).

```sh
python3 -m venv .venv && .venv/bin/python -m pip install pybind11 numpy
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DESHKOL_PYTHON_BINDINGS=ON \
  -DESHKOL_PYTHON3_EXECUTABLE="$PWD/.venv/bin/python" \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build build --target eshkol_py
ctest --test-dir build -R python_bindings_capsule_lifetime --output-on-failure
```

### `Python bindings: pybind11 not found, skipping`

**Cause.** `ESHKOL_PYTHON_BINDINGS=ON` was requested but CMake found no
pybind11 package. This is a status line, not an error: the `eshkol_py` target
is simply absent.

**Fix.** Install pybind11 into the interpreter you build against and pass its
CMake directory: `-Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"`.

### `Could NOT find Python3 (missing: ... Development ...)`

**Cause.** The bindings request the interpreter **and** the development
component (headers and library) with
`find_package(Python3 REQUIRED COMPONENTS Interpreter Development)`. The
interpreter is installed without its development files.

**Fix.** Install them (`python3-dev` on Debian and Ubuntu, `python3-devel` on
Fedora; Homebrew and python.org builds include them), and make both
`Python3_EXECUTABLE` and `ESHKOL_PYTHON3_EXECUTABLE` name the same interpreter
so the module and its test agree.

### A link error asking to `recompile with -fPIC`

**Cause.** On ELF platforms the extension force-loads the static compiler,
REPL, runtime and agent dependency archives into one shared object, so every
object in that closure must be position-independent. The build turns on
`CMAKE_POSITION_INDEPENDENT_CODE` for the whole tree when
`ESHKOL_PYTHON_BINDINGS=ON`, before any target or fetched dependency is
created. The error appears when the option is switched on in a build directory
that already holds non-PIC objects.

**Fix.** Configure with `-DESHKOL_PYTHON_BINDINGS=ON` in a fresh build
directory.

### `FFI JIT runtime is not linked; link eshkol-repl-lib`

**Cause.** The FFI context found no JIT bridge. The bridge registers itself
from a static constructor in the REPL library, so a link that does not
force-load that archive drops it. The `eshkol_py` target force-loads both
`eshkol-static` and `eshkol-repl-lib`; the message appears in an embedding of
your own that links the FFI without them.

**Fix.** Force-load both archives: `-Wl,-force_load,<archive>` once per archive
on macOS, `-Wl,--whole-archive ... -Wl,--no-whole-archive` around both on ELF.

### `python_bindings_capsule_lifetime` fails with `SKIP:`

**Symptom.** The CTest fails and its output contains
`SKIP: eshkol Python module not built` or `SKIP: numpy not installed`.

**Cause.** The test is registered to fail rather than skip: a skipped lifetime
test is not evidence. It prints `SKIP:` when the module or NumPy cannot be
imported by the interpreter CMake recorded.

**Fix.** Build the `eshkol_py` target, and install NumPy into the interpreter
named by `ESHKOL_PYTHON3_EXECUTABLE`. The test finds the module through
`ESHKOL_PYTHON_MODULE_DIR`, which CTest sets to the target's output directory;
for a manual run set it yourself:

```sh
ESHKOL_PYTHON_MODULE_DIR=build python3 tests/bindings/python_capsule_lifetime_test.py
```

### Symbols resolve in `eshkol-run` but not when embedded

**Cause and behaviour.** A Python extension is normally loaded with local
symbol visibility, so the runtime symbols it exports are not visible to a
process-wide lookup. The REPL JIT therefore searches the image that contains
the runtime first, by that image's own loader handle, and then the process.
No `RTLD_GLOBAL` import flag and no preloading is needed.

## Windows

### `LLVM 21 CMake package not found`

**Symptom**

```
LLVM 21 CMake package not found. Install the official LLVM 21 Windows SDK and set LLVM_DIR or LLVM_HOME.
```

**Fix.** Install the LLVM 21 SDK and configure with its CMake package
directory. Visual Studio 2022 with the ClangCL toolset is the supported native
generator:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T ClangCL `
  -DCMAKE_BUILD_TYPE=Release `
  -DLLVM_DIR="C:/Program Files/LLVM/lib/cmake/llvm"
```

Use `-A ARM64` on Windows ARM64. Setting the `LLVM_HOME` environment variable
to the SDK root works as well.

### `... references missing DIA library ... Install the Visual Studio DIA SDK component`

**Cause.** The LLVM SDK's CMake package names the Debug Interface Access
library by an absolute path from the machine that built it.

**Fix.** Install the "DIA SDK" component of Visual Studio. The build then
substitutes the `diaguids.lib` of the active Visual Studio instance for the
architecture being built.

### `Generated Windows links require the compiler-rt builtins archive ...`

**Symptom**

```
Generated Windows links require the compiler-rt builtins archive from the
selected C++ driver '...', but no LLVM 21 clang_rt.builtins-{x86_64|aarch64}.lib
was found. Set ESHKOL_CXX_COMPILER or LLVM_HOME to a complete matching LLVM toolchain.
```

**Cause.** Programs compiled by Eshkol link against the compiler-rt builtins
of the consumer's LLVM installation, and the selected driver belongs to an
installation without them.

**Fix.** Point `ESHKOL_CXX_COMPILER` at the `clang++` of a complete LLVM 21
installation, or set `LLVM_HOME` to its root.

### Other Windows notes

- The MSYS2 / MinGW64 path and the MSVC path differ in how LLVM is found: MinGW
  uses `llvm-config`, the MSVC path uses the LLVM CMake package and the static
  runtime. Do not mix the two in one build directory.
- `ESHKOL_PATH` and other path lists are `;` separated.
- `-DESHKOL_FIXED_POINT_ENGINE=ON` needs the `__int128` ABI and is refused
  under MSVC and ClangCL.
- PowerShell scripts in the repository are ASCII, or start with a UTF-8 byte
  order mark, so Windows PowerShell 5.1 and PowerShell 7 decode them the same
  way. `python3 scripts/check_ps1_encoding.py` checks a change.

## WebAssembly

### `function import requires a callable` when the page instantiates the module

**Cause.** A runtime function that a WebAssembly build imports has no entry in
the JavaScript glue. The build picks up a new host helper as an `env` import on
its own; the browser then refuses to instantiate the module.

**Fix.** Add the import to **both** glue files, `web/eshkol-repl.js` and
`site/static/eshkol-runtime.js`, and to the core-key manifest. The procedure
is in
[WEB_PLATFORM.md](breakdown/WEB_PLATFORM.md#101-import-glue-is-generated).
Check with:

```sh
python3 scripts/check_wasm_imports.py --build-dir build
```

### `generated flat-AD block is stale; run generate_wasm_import_glue.py --write`

**Cause.** The block between `// BEGIN GENERATED FLAT-AD IMPORTS` and
`// END GENERATED FLAT-AD IMPORTS` in a glue file differs from
`scripts/wasm_flat_ad_imports.fragment.js`. The block is generated; it was
edited by hand, or the fragment changed without regenerating.

**Fix.**

```sh
python3 scripts/generate_wasm_import_glue.py --write
python3 scripts/generate_wasm_import_glue.py --check
```

### `core keys absent from contract` or `missing core contract keys`

**Cause.** The `eshkol_*` and `region_*` keys a glue file provides and the
list in `scripts/wasm_core_import_keys.json` disagree.

**Fix.** Add the new key to the manifest (a sorted JSON string array) when the
glue gained an import, or add the import to the glue when the manifest is
right. Both glue files must provide the same core set.

## Type warnings after upgrading

**Symptom.** A program that compiled quietly now prints type warnings.

**Cause.** Eshkol is gradually typed. Annotations are optional, a type finding
is a warning by default, and the program still compiles and runs. Each release
checks the relations its type system defines, so a newer compiler can report a
finding on unchanged code that an older one did not examine. The warning
describes the code, not a change in its behaviour. `--strict-types` makes type
errors fatal; `--unsafe` skips the checks.

**Fix.** Read [guide/GRADUAL_TYPING.md](guide/GRADUAL_TYPING.md): it explains
what each warning means, which are advisory, and how to annotate or restructure
the code it points at.

## Release and gate failures met locally

How a release is verified is in
[platform/RELEASE_PROCESS.md](platform/RELEASE_PROCESS.md).

### A gate reports that tests which passed did not execute

**Cause.** An evidence path was relative and two tools resolved it against
different directories. The repository fixes the meaning once: a relative
`TRACE_DIR` or `ICC_TRACE_DIR` is relative to the **repository root**, and the
gates make it absolute before any producer uses it
(`scripts/lib/evidence_paths.sh`).

**Fix.** Nothing, when you run the gates as shipped: `TRACE_DIR=scripts/icc_traces`
and `TRACE_DIR="$PWD/scripts/icc_traces"` mean the same thing. When you write
a new gate that reads an evidence location from its environment, source the
helper before first use:

```sh
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
```

`python3 tests/toolchain/test_v1_3_release_evidence_recipe.py` fails a script
that reads `TRACE_DIR` or `ICC_TRACE_DIR` without doing so.

### A harness tests the wrong compiler

**Cause.** The harnesses use `build/` unless told otherwise.

**Fix.** Name the tree: `BUILD_DIR=build-asan scripts/run_all_tests.sh`. A
relative `BUILD_DIR` is read against the repository root.

### The documentation-claims gate grades text you already changed

**Cause.** `scripts/check_doc_claims_residual.py` grades claims through the ICC
index, and the index predates your edit. An unregistered checkout is invisible
to ICC altogether.

**Fix.** Refresh the index for the checkout you are editing, then rerun:

```sh
icc reindex --repo <alias> --full
python3 scripts/check_doc_claims_residual.py --icc-bin icc --repo <alias>
```

### `check_changelog_completeness.py` exits `NO_DATA`

**Cause.** The gate walks `previous_tag..HEAD` and fails closed when it cannot:
the checkout is shallow, or the previous release tag is not present.

**Fix.**

```sh
git fetch --unshallow --tags     # in a shallow clone
git fetch --tags                 # otherwise
```

In GitHub Actions, check out with `fetch-depth: 0`. When the gate instead lists
an unaccounted pull request, give it one home: a `(#N)` reference in the
changelog section for the release, or an entry with a class and a specific
reason in `tests/coverage/changelog_no_user_facing_change.json`.

### `check_surface_counts.py` fails after a document edit

**Cause.** A registered document states a count, a date or a status that
disagrees with the manifests under `tests/coverage/` or with
`tests/coverage/release_record.json`, or a `<!-- release-record:KEY -->` span
was edited by hand.

**Fix.** Do not edit the number. Let the gate rewrite every record-owned claim:

```sh
python3 scripts/check_surface_counts.py --sync
python3 scripts/check_surface_counts.py --no-trace
```

If a page under `site/static/content/` is reported, regenerate it with
`scripts/build-site-content.sh`.

### Evidence is reported stale or the trace directory is empty

**Cause.** `scripts/check_evidence_staleness.py --require-trace-dir` refuses to
grade an empty trace root, and fails a high-severity criterion whose newest
evidence is older than the window (14 days by default;
`ESHKOL_EVIDENCE_MAX_AGE_DAYS` or `--max-age-days`).

**Fix.** Regenerate the evidence with the producer the criterion names; do not
widen the window to pass. Trace files are build products and are not committed.
