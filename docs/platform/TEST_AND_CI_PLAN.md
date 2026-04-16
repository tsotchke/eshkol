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
