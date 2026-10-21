# Production surface retirement inventory

Source snapshot: `7b8fb16dc319340ebfc8339c289faedee36e909d` (detached).
This is a source-only inventory. No tests or native builds were run, and the
committed `tests/coverage/language_surface.json` was treated as historical
generated output, not live evidence.

## Decision

Do not delete `lib/backend/eshkol_compiler.c` yet. Its source is outside the
CMake production source list, but it is still a live input to surface
intelligence and a source-contract test. Removing it now would break generation
and leave the manifest claiming obsolete native bindings and prelude names.

## Active consumers and migration requirements

- `scripts/gen_language_surface.py` parses `COMPILER_C` and `VM_C`; its
  `build_manifest()` unions both tables with names parsed from
  `llvm_codegen.cpp`. It also reads mirror annotations from both tables and
  extracts the prelude only from the legacy compiler's `scheme_prelude`.
  Manifest metadata currently labels the legacy table as `native_builtins`.
- `tests/coverage/language_surface.json` is generated output consumed by the
  classification/coverage gates. It cannot be edited as though the snapshot
  were live data; the generator and its consumer contract must change first.
- `CMakeLists.txt` still registers
  `eshkol_compiler_vector_limit_surface_test`; that test opens the retired C
  file and asserts it includes shared limits, uses the shared predicate, emits
  a particular diagnostic, and lacks an old clamp. This is a direct retirement
  blocker even though it does not build the C file.
- `tests/coverage/README.md`, `docs/design/MOONLAB_INTEGRATION.md`, and
  `docs/TESTING.md` describe the old source as active native/production
  authority. Other docs mention it as history, planned work, or a comparison;
  these need triage rather than global string replacement.

## Safe migration proposal

1. Make `lib/backend/eshkol_vm.c`'s `BUILTINS[]` the sole core binding/arity
   source. It is read by the VM and its shared arity helpers are used by LLVM
   codegen. Keep LLVM's parsed call-head set as a separate evidence dimension;
   it supplies names only, not stable IDs, binding metadata, or a full ABI.
2. Replace the legacy Scheme prelude scan with production prelude inputs:
   `lib/backend/vm_prelude_source.h` and `lib/stdlib.esk` (with an explicit
   distinction between VM bootstrap definitions and canonical stdlib names).
   Do not infer a native prelude from the legacy snapshot.
3. Rebase mirror handling on VM annotations and explicit production LLVM
   lowering relationships. Preserve names with distinct semantics and planned
   operations in separate historical/planned fields; do not silently discard
   features to make counts smaller.
4. Change the vector-limit surface test to assert the shared production
   contract through its actual header/VM binding or remove the obsolete
   legacy-only assertion after an equivalent production assertion exists.
5. Regenerate the manifest from live inputs, update classification and docs,
   then demonstrate that every claimed VM/native_llvm entry resolves in those
   sources. Only then remove the obsolete compiler and its exclusively
   historical references.

The exact unresolved blocker is step 1's output schema: LLVM dispatch parsing
currently returns names only, while the manifest has a single `native` label
and IDs/arity may be inherited from the legacy compiler. A small schema change
must distinguish VM binding, LLVM lowering, and historical/planned claims;
calling the LLVM name scan an ABI would overstate its evidence.
