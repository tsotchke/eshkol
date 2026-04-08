# LLVM 21 Toolchain Unification Design

Date: 2026-04-04
Status: Approved for planning
Scope: Sub-project 1 of the LLVM migration effort

## Goal

Standardize Eshkol's lite/native build toolchain on LLVM 21 across Linux, macOS, and Windows, so local builds, packaging scripts, Docker parity environments, and later CI jobs all target one explicit compiler/runtime baseline.

## Why This Exists

The current tree mixes an older documented baseline (`LLVM 17`), recent compatibility code labeled as `LLVM 18+`, and local fixes that were needed to get Linux building again. That creates two problems:

1. Toolchain drift: different environments believe they are targeting different LLVM versions.
2. Debugging ambiguity: it is hard to tell whether a failure is a compiler bug, a platform bug, or a bad version-compatibility layer.

The intended outcome of this design is a single answer to "what LLVM does Eshkol use?": `LLVM 21`, unless a clearly documented bundled XLA/StableHLO exception applies.

## Non-Goals

This sub-project does not make the entire test suite green.

This sub-project does not fully update GitHub Actions, release workflows, or `act` automation. Those are follow-on changes after local/native tooling, Docker parity, and platform smoke paths are aligned on LLVM 21.

This sub-project standardizes Windows on the native Visual Studio 2022 + ClangCL + LLVM 21 SDK path.

## Recommended Approach

Use a hard pin to LLVM 21 across all supported local/native platforms.

This is preferred over dual-support because the existing drift appears to come from coding against a newer LLVM API while labeling broad compatibility ranges such as `18+`. A single pinned baseline reduces conditional logic, removes false compatibility claims, and makes later test triage attributable to one actual compiler version.

## Toolchain Contract

### CMake As Source of Truth

`CMakeLists.txt` becomes the authority for LLVM toolchain selection in lite/native builds.

For lite/native builds:

- resolve `llvm-config` by preferring explicit LLVM 21 locations first
- query the discovered toolchain for its major version
- fail early with a precise configuration error unless the major version is exactly `21`

For bundled or special-case builds:

- preserve the existing `STABLEHLO_ROOT` path as a separate mode
- make any different LLVM assumptions explicit rather than silently sharing the lite/native rules

### Script Behavior

Platform scripts must stop embedding their own independent LLVM policy. They should either:

- locate a platform-specific LLVM 21 installation and hand control to CMake, or
- rely on CMake to reject a misconfigured environment with a clear message

This applies to:

- `scripts/build-macos.sh`
- `scripts/test-ci-locally.sh`
- `scripts/test-homebrew.sh`
- `scripts/run_cpp_type_tests.sh`

### Backend Versioning Policy

Backend compatibility code should stop claiming support for vague ranges like `LLVM 18+` when the code was only validated against a newer API surface.

The backend should be simplified around LLVM 21 semantics in:

- `lib/backend/llvm_codegen.cpp`
- `lib/backend/arithmetic_codegen.cpp`
- `lib/backend/autodiff_codegen.cpp`
- `lib/backend/complex_codegen.cpp`
- `lib/backend/tensor_codegen.cpp`
- `lib/repl/repl_jit.cpp`

The goal is not cosmetic cleanup. The goal is to remove misleading compatibility branches that make root-cause analysis harder.

## Platform Rules

### Linux

Linux uses the official LLVM apt repository path and resolves `llvm-config-21`.

Expected shape:

- apt source points at the `llvm-toolchain-<distro>-21` repository
- installed packages use LLVM 21 package names
- `/usr/bin/llvm-config-21` is preferred over an unqualified `llvm-config`

The existing Linux build hygiene fixes remain in place unless LLVM 21 proves them incorrect:

- treat `llvm-config` flags as real argument lists instead of a quoted blob
- filter out LLVM-supplied flags that should not override project language mode or exception policy
- use the correct GNU ld spelling `-z stack-size=...`

### macOS

macOS uses Homebrew `llvm@21`.

Expected shape:

- Apple Silicon prefers `/opt/homebrew/opt/llvm@21`
- Intel prefers `/usr/local/opt/llvm@21`
- local scripts export PATH, include, and library paths from that keg before configuring

### Windows

Windows uses native Visual Studio 2022, the ClangCL toolset, and the official LLVM 21 SDK.

Expected shape:

- builds configure with the `Visual Studio 17 2022` generator and `-T ClangCL`
- `LLVM_DIR` points at the LLVM 21 SDK CMake package
- smoke validation can be driven from WSL2 into native Windows commands

## Files and Change Areas

### Core build authority

- `CMakeLists.txt`

### Backend/toolchain compatibility cleanup

- `lib/backend/llvm_codegen.cpp`
- `lib/backend/arithmetic_codegen.cpp`
- `lib/backend/autodiff_codegen.cpp`
- `lib/backend/complex_codegen.cpp`
- `lib/backend/tensor_codegen.cpp`
- `lib/repl/repl_jit.cpp`

### Local and packaging scripts

- `scripts/build-macos.sh`
- `scripts/test-ci-locally.sh`
- `scripts/test-homebrew.sh`
- `scripts/run_cpp_type_tests.sh`

### Docker parity environments

- `docker/debian/debug/Dockerfile`
- `docker/debian/release/Dockerfile`
- `docker/ubuntu/release/Dockerfile`
- `docker/xla/Dockerfile`
- `docker/cuda/Dockerfile`

### Follow-on CI/release work

These are acknowledged now but intentionally deferred:

- `.github/workflows/ci.yml`
- `.github/workflows/release.yml`

## Validation Plan

This sub-project is complete when the following validations have been run against the LLVM 21 baseline:

1. Linux native configure and build succeed.
2. Linux native test execution runs against the LLVM 21 build.
3. Linux Docker parity images build with LLVM 21.
4. macOS local/scripted build path resolves `llvm@21` correctly.
5. Windows build can be driven from WSL2 into a native Visual Studio 2022 + LLVM 21 environment and complete at least a configure/build/smoke cycle.

Minimum Windows smoke means:

- configure succeeds
- build succeeds
- one trivial program can be compiled
- the produced executable or equivalent smoke path runs successfully

This pass does not require every test suite to pass. It requires that remaining failures are now happening on a single pinned toolchain, which makes the next debugging phase coherent.

## Error Handling and Failure Modes

The preferred failure mode is early configuration failure, not a partial compile followed by obscure backend errors.

Required behavior:

- if `llvm-config` is missing, fail with an install hint for the active platform
- if LLVM is found but is not major version `21`, fail with a version mismatch message that shows the discovered executable and version
- if Windows is invoked without a usable Visual Studio 2022 + LLVM 21 environment, fail with a setup hint
- if `STABLEHLO_ROOT` implies a different LLVM world, make that branch explicit instead of silently reusing lite/native assumptions

## Risks

### XLA / StableHLO divergence

The bundled XLA path may not actually be on LLVM 21 yet. If so, it should remain an explicit exception during this sub-project rather than forcing fake uniformity.

### Existing Linux fixes may partially overlap with 21-only cleanup

Some local fixes were introduced to repair an invalid compatibility layer. Those changes should be retained if they are still correct on LLVM 21, and removed only if LLVM 21 makes them unnecessary.

### Windows environment discovery may be brittle

Windows smoke from WSL2 depends on installed and callable Visual Studio 2022 and LLVM 21 tooling. That path must be validated pragmatically rather than assumed from docs.

## Exit Criteria

This design is considered implemented when:

- LLVM 21 is the explicit lite/native baseline in CMake
- local scripts and Docker parity environments resolve LLVM 21 consistently
- Linux, macOS, and Windows smoke paths have been exercised against that baseline
- remaining failures can be attributed to compiler/runtime behavior on LLVM 21 instead of version ambiguity

## Next Sub-Project

After this toolchain unification lands, the next work item is:

1. update CI/CD and local `act` workflows to the same LLVM 21 baseline
2. drive the full test suite back to green on that baseline
