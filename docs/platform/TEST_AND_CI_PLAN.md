# Test and CI Plan

## Purpose

This document defines how the platform program is verified while protecting hosted Eshkol from regression.

## Testing Principles

- Hosted mode remains the baseline and must stay green.
- Freestanding support begins with smoke tests and becomes stricter over time.
- Platform work is not mergeable without tests appropriate to its stage.
- No hosted-only symbol leakage into freestanding artifacts is acceptable once runtime stratification begins landing.

## Test Categories

## 1. Profile tests

Purpose:

- verify profile parsing
- verify target/profile compatibility checks
- verify diagnostics for invalid combinations

Suggested location:

- `tests/freestanding/profile/`

Current guard:

- `tests/toolchain/execution_profile_test.cpp`
  - verifies profile-name resolution and valid/invalid profile combinations
  - verifies freestanding native profiles require explicit targets and imply compile-only/no-stdlib
- `tests/toolchain/eshkol_run_profile_cli_test.cpp`
  - verifies `eshkol-run` parses `--profile` and `--target` through the production driver
  - verifies hosted/freestanding incompatible flag combinations fail at the CLI boundary
  - verifies freestanding native profile selection can emit an exact requested object path
- `tests/v1_2_edge_cases/object_build_cli_contract_test.sh`
  - verifies `eshkol-run --profile freestanding-kernel-native` rejects a missing target
  - verifies `--profile freestanding-kernel-native --target <triple>` emits the exact requested object path
  - verifies unknown profile names produce a clear diagnostic

## 2. Low-level language tests

Purpose:

- fixed-width machine types
- pointer operations
- volatile lowering
- section and visibility attributes
- barrier and intrinsic lowering

Suggested location:

- `tests/freestanding/types/`
- `tests/freestanding/codegen/`

Current guard:

- `tests/toolchain/machine_integer_types_test.cpp`
  - verifies machine integer aliases in the HoTT environment
  - verifies builtin-name resolution for `u8`/`usize`-style annotations
  - verifies parser + type-checker integration for annotated machine integer signatures
- `tests/toolchain/pointer_type_surface_test.cpp`
  - verifies `ptr` / `pointer` aliases in the HoTT environment
  - verifies `(ptr T)` parsing, copying, and stringification
  - verifies parser + type-checker integration for pointer annotations
  - verifies `ptr-add` type-checking diagnostics and byte-offset LLVM IR lowering
- `tests/toolchain/fence_ops_test.cpp`
  - verifies parser, type-checker, and IR lowering for `compiler-fence` and `memory-fence`
- `tests/toolchain/volatile_ops_test.cpp`
  - verifies typed volatile load/store parsing, type-checking, and LLVM volatile IR lowering
- `tests/toolchain/atomic_ops_test.cpp`
  - verifies typed atomic load/store/exchange/fetch-add/fetch-sub parsing, ordering diagnostics, and LLVM atomic IR lowering
- `tests/toolchain/freestanding_elf_link_smoke_test.cpp`
  - verifies a freestanding object can link with a minimal ELF linker script without hosted runtime symbols
- `tests/toolchain/freestanding_reject_hosted_builtin_test.cpp`
  - verifies freestanding native codegen rejects hosted parallel builtins before emitting object artifacts
- `tests/toolchain/target_intrinsic_test.cpp`
  - verifies typed LLVM intrinsic parsing, type-checking diagnostics, and IR lowering
- `tests/toolchain/decl_attribute_test.cpp`
  - verifies declaration attribute parsing for `define`, `extern`, and `extern-var`
  - verifies IR lowering for section placement, alignment, weak imports/definitions, `llvm.used`, exported symbol names, and `noreturn`

## 3. Runtime isolation tests

Purpose:

- detect hosted-only symbol leakage
- ensure freestanding runtime links against only allowed surfaces

Suggested location:

- `tests/freestanding/runtime/`

## 4. Link and artifact tests

Purpose:

- verify object generation
- verify ELF generation
- verify linker-script handling
- later verify `.bin` and `.hex` conversions

Suggested location:

- `tests/freestanding/link/`

## 5. VM freestanding tests

Purpose:

- verify hook-driven VM build
- verify ESKB execution with explicit hooks
- verify no hidden OS assumptions

Suggested location:

- `tests/freestanding/vm/`

## 6. BSP and boot smoke tests

Purpose:

- verify reference-target images boot or execute
- verify serial/console output

Suggested location:

- `tests/bsp/`

## CI Lanes

## Phase 1 lanes

- existing hosted CI remains unchanged
- add non-blocking `freestanding-profile-smoke`

## Phase 2 lanes

- add non-blocking `freestanding-codegen-smoke`
- add non-blocking `freestanding-runtime-smoke`

## Phase 3 lanes

- promote `freestanding-profile-smoke` to blocking
- add blocking `freestanding-link-smoke`

## Phase 4 lanes

- add non-blocking `freestanding-vm-smoke`
- add non-blocking `bsp-qemu-smoke`

## Phase 5 lanes

- promote first reference-target QEMU smoke test to blocking
- promote `freestanding-vm-smoke` to blocking once the VM contract is stable

## Required Invariants

These invariants eventually become blocking:

- hosted suites still pass
- freestanding builds do not pull hosted-only symbols
- profile-incompatible modules fail early and clearly
- low-level operations lower correctly
- supported BSPs continue to boot their smoke tests

## Promotion Rules

A platform test may become blocking when:

- the feature is no longer marked experimental
- the relevant contract is documented in `docs/platform/`
- the test is stable across normal developer environments or CI containers

## Documentation Rule

Any new CI lane or test category must be documented here when it is introduced or promoted.
