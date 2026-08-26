# Build-system integration (ESH-0215)

This document covers compiling `.esk` sources from a CMake project — the
supported path for embedding Eshkol as an AOT-compiled dependency of a
larger C/C++ project (e.g. Noesis) — and, specifically, how incremental
rebuilds stay correct when a source changes.

## The bug this fixes

Prior to ESH-0215, a build-system integration that compiled `main.esk` to
`main.o` via `eshkol-run --emit-object -o main.o main.esk` could only tell
its build graph that `main.o` depends on `main.esk`. If `main.esk` did
`(load "dep.esk")`, editing `dep.esk` left `main.o`'s only tracked
prerequisite unchanged — Ninja/Make saw no reason to rebuild, and the
downstream project silently linked a stale object. The workaround was to
`rm` the object and re-run the build 2-3 times, hoping a stale link
surfaced the mismatch. That is architecturally the same class of bug as
ESH-0183 (the `-r` persistent run-cache also used to hash only the entry
file — see `exe/eshkol-run.cpp`'s `transitiveSourceDigest`).

## `--emit-depfile PATH`

`eshkol-run`, when compiling to an object (`-c`/`--compile-only` or
`--emit-object`, with `-o`/`--output`), accepts `--emit-depfile PATH`. It
writes a Makefile-format depfile:

```
out.o: \
  main.esk \
  dep.esk \
  dep/of/dep.esk
```

The prerequisite list is exactly the entry file plus every file
transitively reachable from it via `(load "…")`, `(import "…")`, and
`(require module)`, resolved through the same search order the compiler
itself uses (the referring file's directory, the current directory, the
`-I` include paths / `$ESHKOL_PATH`, then the Eshkol `lib/` directory).
Paths with spaces, `#`, or `$` are escaped per Make depfile conventions.
The scan is deliberately generous — a dependency that turns out to be
unreachable at compile time only costs a redundant rebuild, whereas a
missed dependency reintroduces the stale-object bug.

Ninja's `DEPFILE` build-edge attribute (or `make`'s `-include *.d`) reads
this file and adds each listed path as an *implicit* dependency of the
edge's `OUTPUT`, so the build system reruns the compile whenever any of
them changes — without the project having to know the dependency graph
in advance at configure time.

## `cmake/EshkolCompile.cmake`

This repository ships the canonical CMake integration at
`cmake/EshkolCompile.cmake`. It provides:

```cmake
eshkol_compile_library(NAME
  SOURCES        <.esk files…>
  [INCLUDE_DIRS   <dirs…>]
  [LINK_LIBRARIES <targets…>]
  [DEFINES        <name=value…>]
  [DEPENDS        <extra files / targets…>])

eshkol_compile_executable(NAME
  SOURCES        <.esk files…>
  [INCLUDE_DIRS   <dirs…>]
  [LINK_LIBRARIES <targets…>]
  [DEFINES        <name=value…>]
  [DEPENDS        <extra files / targets…>])
```

Each `.esk` source becomes one `add_custom_command` that invokes
`eshkol-run --emit-object -o <obj> --emit-depfile <obj>.d ...`, attaches
`DEPFILE <obj>.d` to the command (Ninja, and Make on CMake >= 3.20 — see
below), and `DEPENDS`s on the source file itself as a floor guarantee even
where DEPFILE isn't supported. The resulting `.o` files are wired into a
normal `add_library(... SHARED ...)` / `add_executable(...)` target with
`LINKER_LANGUAGE CXX`, so the rest of a downstream project's CMake (linking
against the Eshkol runtime, packaging, install rules) is unaffected.

`Eshkol_COMPILER` must resolve to an `eshkol-run` binary — either set
explicitly, discovered via `find_package(Eshkol)`, or picked up
automatically from an in-tree `eshkol-run`/`eshkol::eshkol-run` CMake
target (as happens when this module is used from within the Eshkol build
itself — see `tests/build_integration/`).

## Discovering an INSTALLED Eshkol: `find_package(Eshkol)`

A consumer building against a packaged Eshkol (homebrew, or a GitHub release
tarball) rather than an in-tree checkout has no `eshkol-run`/`eshkol-runtime`
CMake targets to pick up automatically. Both packaging pipelines install
this repo's own `cmake/FindEshkol.cmake` and `cmake/EshkolCompile.cmake` to
`<prefix>/share/eshkol/cmake/`:

```cmake
list(APPEND CMAKE_MODULE_PATH "<prefix>/share/eshkol/cmake")
find_package(Eshkol REQUIRED)
include(EshkolCompile)

eshkol_compile_executable(my_program
  SOURCES my_program.esk
)
```

`find_package(Eshkol)` resolves `Eshkol_COMPILER` (the `eshkol-run` driver)
and produces the `Eshkol::eshkol` imported target, which
`eshkol_compile_executable()`/`eshkol_compile_library()` link into every
target they create automatically. Do not hand-roll a `find_library()` for
"the Eshkol library": the package carries TWO static archives that are easy
to confuse — `eshkol-runtime` (the lean hosted runtime a COMPILED PROGRAM
needs) and `eshkol-static` (the compiler/tool aggregate, for embedding the
compiler itself) — and every compiled Eshkol program also needs `stdlib.o`
linked in unconditionally (codegen calls `__eshkol_lib_init__` from a
program's synthesized entry point whenever the source has no explicit
`(define (main) ...)`, and only ever DEFINES that symbol inside `stdlib.o`).
`Eshkol::eshkol`'s `INTERFACE_LINK_LIBRARIES` already carries both the
correct archive and `stdlib.o` (plus, on Apple, the system frameworks the
runtime needs) as one complete link line — see `cmake/FindEshkol.cmake`'s
header comment for the full contract, and
`tests/integration/system_package/` for a working from-scratch consumer
project exercising it end to end (driven by
`scripts/run_system_package_integration_test.sh`; wired into the
`packaging-homebrew` job of `.github/workflows/adversarial-nightly.yml`
against a real `brew install --build-from-source` result, and gated by
`.icc/package-manifest.yaml`'s `integration_contract` entry when
`scripts/check_package_manifest.py` runs with `--verify-integration-contract`).

This module is scoped to macOS/Linux today; the link recipe has not yet
been established for the Windows/MSVC toolchain.

### Generator support for DEPFILE

`add_custom_command(... DEPFILE ...)` is a Ninja-only feature prior to
CMake 3.20; CMake 3.20+ also supports it for the Makefiles generator
family. Multi-config/IDE generators (Xcode, Visual Studio) do not support
it. `EshkolCompile.cmake` detects this at configure time
(`CMAKE_GENERATOR` + `CMAKE_VERSION`) and only passes `DEPFILE` when the
active generator honors it; on unsupported generators the custom command
still `DEPENDS`s on the entry `.esk` file (the pre-ESH-0215 behavior), so
nothing regresses — dependency tracking on `(load …)`ed files simply isn't
available there.

### `ESHKOL_OBJECT_MODE`

- `AUTO` (default): probes `eshkol-run --emit-object` at configure time
  and falls back to the legacy `--compile-only --output <stem>` path if
  the probe fails (e.g. a pre-`--emit-object` binary). When
  `Eshkol_COMPILER` is a generator expression (an in-tree, not-yet-built
  target), the probe can't run at configure time, so `AUTO` assumes
  `EMIT_OBJECT`.
- `EMIT_OBJECT` / `COMPILE_ONLY`: force one or the other.

`--emit-depfile` is emitted in both modes — both route through the same
object-emission code in `eshkol-run.cpp`.

## Verification fixture: `tests/build_integration/`

`tests/build_integration/` is a minimal end-to-end fixture built when
`ESHKOL_BUILD_TESTS` is on and the in-tree `eshkol-run`/`eshkol-runtime`
targets exist: `main.esk` `(load "dep.esk")`s a one-line helper and calls
it. It is wired into the top-level `CMakeLists.txt` via `add_subdirectory`
and links `eshkol-runtime` directly (the same target real generated
programs link against), so it exercises the production dependency graph
rather than a hand-rolled stand-in.

To reproduce the fix manually:

```sh
cmake -S . -B build -G Ninja
ninja -C build build_integration_demo
./build/tests/build_integration/build_integration_demo   # -> 42

# Edit the LOADED dependency, not the entry file:
$EDITOR tests/build_integration/dep.esk

ninja -C build build_integration_demo   # single invocation: recompiles + relinks
./build/tests/build_integration/build_integration_demo   # -> reflects the edit

ninja -C build build_integration_demo   # touch nothing again: no-op
```

The depfile itself can be inspected directly, or through Ninja's own deps
log:

```sh
cat build/tests/build_integration/CMakeFiles/build_integration_demo.esk.dir/main.esk.o.d
ninja -C build -t deps tests/build_integration/CMakeFiles/build_integration_demo.esk.dir/main.esk.o
```
