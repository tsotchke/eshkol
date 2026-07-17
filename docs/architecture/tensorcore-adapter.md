# TensorCore compiler-adapter ownership

Status: accepted for Eshkol 1.3.3-evolve (2026-07-14)

## Decision

Eshkol is the sole authority for Eshkol AST/IR lowering, LLVM declarations,
calling conventions, language bindings, diagnostics, and packaging for the
TensorCore integration. TensorCore remains the authority for its public C ABI,
status/dtype/backend values, buffers, kernels, transports, device discovery,
and execution semantics.

The canonical path is:

```text
(require tensorcore)
  -> lib/tensorcore.esk
  -> eshkol_tc_* flat FFI
  -> TensorCore installed public C ABI
```

The compiler registers the same 34 `eshkol_tc_*` declarations for LLVM/JIT,
AOT, REPL, SDK, and tool-driven compilation. Builds without TensorCore resolve
that surface to explicit-unavailable stubs. Compiler lowering is never changed
by an ambient environment variable.

Eshkol does not vendor TensorCore headers or reproduce TensorCore enum
ordinals. `lib/bridge/tensorcore_adapter.cpp` includes installed public headers,
constructs public descriptors, and exposes dtype accessors used by the Eshkol
module. Runtime capability and backend masks come only from
`tc_runtime_capabilities_get` ABI v1; Eshkol neither infers them from CMake
options nor publishes a competing bit registry.

## Build and discovery

TensorCore support is opt-in and consumes an installed CMake package:

```sh
cmake -S . -B build \
  -DESHKOL_TENSORCORE_ENABLED=ON \
  -DCMAKE_PREFIX_PATH=/path/to/tensorcore/prefix
cmake --build build
```

Eshkol requires the exported `tensorcore::tensorcore_shared` target and the
installed `tensorcore/capabilities.h` v1 contract. Generated
programs are linked by the compiler driver rather than CMake, so the shared
public-ABI target provides an explicit, portable link closure without trying to
flatten backend-specific static dependencies. The generated-program linker
receives the same installed artifact and runtime search path.

The supported public-ABI window is TensorCore 0.1.22 or newer within 0.1.x.
Older releases and a different major/minor line fail configuration. A patch
newer than `ESHKOL_TENSORCORE_MAX_TESTED_VERSION` is allowed by compatibility
policy but emits a release-gate warning. Runtime `tc_version()` is checked again
before initialization; malformed or incompatible versions return
`ESHKOL_TC_ERR_ABI_MISMATCH` (-1001).

Initialization then queries `tc_runtime_capabilities_get` using the installed
header's `TC_RUNTIME_CAPABILITIES_ABI_VERSION_1`. The returned struct version,
minimum size, reserved fields, exact v1 known-feature mask, available subset,
known backend bits, and available/compiled relationship are validated before
the context is exposed. An unknown query version fails with the TensorCore ABI
status; unknown feature/backend bits return
`ESHKOL_TC_ERR_CAPABILITY_MISMATCH` (-1002). This is a fail-closed dependency:
package version strings or compiled-target metadata never substitute for the
runtime query.

When integration is disabled, calls return `ESHKOL_TC_ERR_UNAVAILABLE`
(-1000), null, or zero as appropriate, and preserve the adapter-specific status
in `eshkol_tc_last_status()`.

## Compatibility window and removal handoff

TensorCore's historical `tc_eshkol_*` shim and
`eshkol/bridge/tensorcore_codegen.cpp` are compatibility-only. They are not a
source of compiler truth. The compatibility window covers Eshkol 1.3.x and
1.4.x; the earliest permitted removal is Eshkol 1.5.0.

Before TensorCore removes or demotes those files, all of these gates must pass
on the release candidate:

1. `tensorcore_codegen_test` passes LLVM `verifyModule` and conflicting ABI
   declarations fail deterministically.
2. `tensorcore_adapter_test` passes explicit-unavailable, installed portable
   CPU, capability-ABI negative cases, GEMM/attention descriptors, and
   available hardware-backed configurations.
3. `tensorcore_compatibility_test` produces identical GEMM results, statuses,
   version text, and status diagnostics through canonical and compatibility
   paths.
4. `(require tensorcore)` runs through an AOT link using the installed package.
5. Unsupported installed/runtime versions and unknown capability/backend bits
   fail with stable diagnostics.

After those gates ship, the TensorCore repository should:

- mark `tc_eshkol_*` deprecated for the remainder of the window;
- replace its compiler-adapter documentation with a link to this decision;
- remove `eshkol/bridge/tensorcore_codegen.cpp` and stop describing compiler
  integration steps; and
- retain only public ABI and compatibility-shim tests until Eshkol 1.5.0.

## Canonical files

- `lib/backend/tensorcore_codegen.cpp`: LLVM declarations and function-table
  registration
- `lib/bridge/tensorcore_adapter.cpp`: public-ABI FFI adaptation
- `inc/eshkol/tensorcore_adapter.h`: installed adapter contract
- `lib/tensorcore.esk`: installed Eshkol language module
- `tests/backend/tensorcore_*_test.cpp`: verification, runtime, and migration
  gates

TensorCore never needs Eshkol AST, IR, `CodegenContext`, LLVM, or compiler
headers on the canonical path.
