# Production surface retirement inventory

Source snapshot: `7b8fb16dc319340ebfc8339c289faedee36e909d` (detached).
This was the pre-migration source-only inventory; the implementation below records its resolution.

## Resolution

The generator now derives core IDs and arity only from the VM table, keeps LLVM
dispatch names as separate source evidence, and reads VM bootstrap independently
from the top-level definitions and export closure rooted at `lib/stdlib.esk`.
Historical compiler-only names and IDs are frozen in
`tests/coverage/retired_compiler_builtin_reference.json` with the original
compiler source SHA-256. The C source remains in place pending root review.

## Migrated consumers

- The generator now uses VM BUILTINS for core IDs/arity, preserves LLVM
  dispatch as `native_llvm` names only, and reads mirrors from VM annotations.
- The manifest was regenerated from the migrated production inputs; the
  classification logic now treats only VM entries, LLVM name evidence, syntax,
  bootstrap, stdlib closure exports, and module sources as live evidence.
- The vector-limit surface check now verifies the shared capacity boundary
  and the production VM make-vector condition in `vm_native.c`.
- Coverage and Moonlab docs describe the production source contract and label
  remaining references to the compiler as historical. A Python regression test
  mutates a temporary legacy source and proves generation is independent of it.

## Preserved evidence

Legacy compiler-only names and IDs remain in the frozen reference JSON with the
source SHA-256. Planned coverage policy and category requirements were not
removed or lowered during the migration.

The schema now distinguishes `vm` registrations, `native_llvm` dispatch-name evidence, and frozen retired references. It does not assign VM IDs or arity to LLVM-only names.
