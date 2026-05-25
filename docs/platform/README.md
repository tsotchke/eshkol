# Eshkol Freestanding / Platform Program

This directory documents the long-running platform program that makes Eshkol a profiled systems language capable of producing freestanding kernels, firmware, loaders, and embedded artifacts without disrupting the existing release schedule.

## Why This Program Exists

Eshkol already has the right high-level shape for systems work:

- a real compiler front end
- a production LLVM backend
- a production bytecode VM
- deterministic arena-based memory
- explicit runtime value layouts
- substantial standard library and module infrastructure

What it does not yet have is a clean freestanding architecture. Today the implementation is still hosted-first. This program documents the work required to add:

- execution profiles
- freestanding runtime families
- native freestanding LLVM output
- freestanding VM mode
- low-level machine primitives
- BSP and linker-script contracts
- a downstream kernel consumer path

## Current Program Model

- Active roadmap integration branch: `feature/v1.2-scale`
- Recommended long-lived platform branch: `feature/platform-freestanding`
- Primary repo for platform/toolchain work: `~/Desktop/eshkol`
- Downstream kernel repo: `~/Desktop/eshkol-kernel`

This program starts during `v1.2-scale`, runs in parallel with the roadmap, and converges publicly at `v1.8-platform`.

## How To Use These Docs

Read in this order:

1. [PROGRAM_OVERVIEW.md](PROGRAM_OVERVIEW.md)
2. [ARCHITECTURE.md](ARCHITECTURE.md)
3. [ROADMAP_ALIGNMENT.md](ROADMAP_ALIGNMENT.md)
4. [BRANCHING_AND_GOVERNANCE.md](BRANCHING_AND_GOVERNANCE.md)
5. [SYNC_POLICY.md](SYNC_POLICY.md)
6. [IMPLEMENTATION_WORKFLOW.md](IMPLEMENTATION_WORKFLOW.md)
7. [SLICE_CHECKLIST.md](SLICE_CHECKLIST.md)
8. [RUNTIME_INVENTORY.md](RUNTIME_INVENTORY.md)
9. [WORKSTREAMS.md](WORKSTREAMS.md)
10. [MILESTONES_AND_EXIT_CRITERIA.md](MILESTONES_AND_EXIT_CRITERIA.md)

Use the remaining docs as operational references while implementation proceeds.

## Document Index

- [PROGRAM_OVERVIEW.md](PROGRAM_OVERVIEW.md)
  - program vision, end-state architecture, phases, success definition
- [ROADMAP_ALIGNMENT.md](ROADMAP_ALIGNMENT.md)
  - how the platform program fits `v1.2` through `v1.8`
- [BRANCHING_AND_GOVERNANCE.md](BRANCHING_AND_GOVERNANCE.md)
  - branch roles, merge policy, founder-led decision process, delegation rules
- [SYNC_POLICY.md](SYNC_POLICY.md)
  - roadmap-to-platform sync cadence, hotspot files, merge-back criteria, full vs selective sync
- [IMPLEMENTATION_WORKFLOW.md](IMPLEMENTATION_WORKFLOW.md)
  - worktree layout, topic-branch lifecycle, merge flow, validation expectations
- [SLICE_CHECKLIST.md](SLICE_CHECKLIST.md)
  - release-quality checklist for every platform change slice
- [RUNTIME_INVENTORY.md](RUNTIME_INVENTORY.md)
  - concrete file, header, and ownership baseline for the runtime split
- [ARCHITECTURE.md](ARCHITECTURE.md)
  - technical architecture for profiles, runtimes, backends, BSPs, and kernel handoff
- [WORKSTREAMS.md](WORKSTREAMS.md)
  - subsystem-by-subsystem implementation tracks and dependencies
- [MILESTONES_AND_EXIT_CRITERIA.md](MILESTONES_AND_EXIT_CRITERIA.md)
  - milestone definitions, tests, exit criteria, and sign-off rules
- [RISKS_AND_NON_GOALS.md](RISKS_AND_NON_GOALS.md)
  - scope protection, risk register, deferred questions
- [TEST_AND_CI_PLAN.md](TEST_AND_CI_PLAN.md)
  - testing strategy, CI lanes, promotion rules, invariants
- [TARGET_SUPPORT_MATRIX.md](TARGET_SUPPORT_MATRIX.md)
  - support tiers, reference targets, target-class expectations
- [KERNEL_HANDOFF.md](KERNEL_HANDOFF.md)
  - contract between `eshkol` and `eshkol-kernel`
- [DECISIONS.md](DECISIONS.md)
  - append-only architecture and governance decision log

## Current Status

Program phase:

- documentation and planning complete enough to guide implementation
- execution profile architecture is merged on `master`
- `eshkol-run` exposes profile-aware CLI selection through `--profile NAME` and explicit target selection through `--target TRIPLE`
- runtime inventory baseline is documented on `master`
- runtime-core and runtime-hosted internal source sets are explicit in CMake
- hosted `runtime_exports.h` wrappers now live in a dedicated `runtime_exports_hosted.cpp` unit
- hosted runtime fatal/type-error handling now lives in a dedicated `runtime_errors_hosted.cpp` unit
- hosted dynamic parameter storage now lives in a dedicated `runtime_parameters_hosted.cpp` unit
- hosted in-flight operation tracking now lives in a dedicated `runtime_operations_hosted.cpp` unit
- hosted shutdown-hook registration and dispatch now lives in a dedicated `runtime_shutdown_hooks_hosted.cpp` unit
- hosted signal and exception handler installation now lives in a dedicated `runtime_signals_hosted.cpp` unit
- hosted runtime lifecycle and interrupt state now lives in a dedicated `runtime_lifecycle_hosted.cpp` unit
- hosted process stack-limit setup now lives in a dedicated `runtime_stack_hosted.cpp` unit
- hosted FILE-backed string-port helpers now live in a dedicated `runtime_string_ports_hosted.cpp` unit
- hosted arena/hash-table mutex and once primitives now live in a dedicated `runtime_arena_sync_hosted.cpp` unit
- hosted S-expression reader helpers now live in a dedicated `runtime_reader_hosted.cpp` unit
- hosted display/write and current-port helpers now live in a dedicated `runtime_display_hosted.cpp` unit
- hosted exception handling and forward-reference provider diagnostics now live in a dedicated `runtime_exceptions_hosted.cpp` unit
- hosted logger/resource-limit implementations are classified as `runtime-hosted`, and the AST pretty-printer is no longer treated as runtime substrate
- freestanding-safe bytevector helpers now live in a dedicated runtime-core translation unit instead of the hosted-heavy runtime state implementation
- closure environment and callable allocation helpers now live in a dedicated runtime-core translation unit instead of `arena_memory.cpp`
- closure reflection and lambda-registry helpers now live in a dedicated runtime-core translation unit instead of `arena_memory.cpp`
- continuation and dynamic-wind helpers now live in a dedicated runtime-core translation unit instead of `arena_memory.cpp`
- automatic-differentiation tape/node helpers now live in a dedicated runtime-core translation unit instead of `arena_memory.cpp`
- hash-table allocation, key hashing/equality, and mutation helpers now live in a dedicated runtime-core translation unit instead of `arena_memory.cpp`
- tensor allocation helpers now live in a dedicated runtime-core translation unit instead of `arena_memory.cpp`
- tensor linalg, broadcast, shape-conversion, concat, and batched-matmul helpers now live in a dedicated runtime-core translation unit instead of `arena_memory.cpp`
- generated-code list reverse, quasiquote append/splice, recursion-depth, and list/vector guard helpers now live in a dedicated runtime-core translation unit instead of `arena_memory.cpp`
- region stack/lifecycle/escape helpers and per-worker thread-local arena setup now live in a dedicated runtime-core translation unit instead of `arena_memory.cpp`
- deep structural equality for tagged values now lives in a dedicated runtime-core translation unit instead of `arena_memory.cpp`
- header-aware tagged-object allocation helpers now live in a dedicated runtime-core translation unit instead of `arena_memory.cpp`
- tagged cons allocation and accessor helpers now live in a dedicated runtime-core translation unit instead of `arena_memory.cpp`
- raw arena block allocation, scope reset/pop, statistics, and legacy list-node allocation now live in a dedicated runtime-core translation unit instead of `arena_memory.cpp`
- hosted arena poison diagnostics now live behind a profile hook instead of making runtime-core read process environment variables
- the C++ `Arena` RAII wrapper now lives outside runtime source families as a C++ adapter around the C arena ABI
- the `runtime-split-pending` source set has been retired; remaining runtime implementation files are classified as `runtime-core` or `runtime-hosted`
- the bytecode VM unity hub now builds through `eshkol-vm-unity-obj`, with explicit `vm-core`, `vm-hosted`, VM toolchain, and VM test component families guarded by a boundary test
- bytecode VM heap, stack, frame, constant-pool, and instruction capacities are now CMake cache profile knobs, with shared defaults in `inc/eshkol/backend/vm_limits.h`
- embedders can install a deterministic VM host-native table through the public C ABI, giving firmware/product profiles stable native-call slots before the desktop native table is physically split
- VM handles can switch to host-native-only native-call policy, so embedded/product runtimes can reject the broad desktop native table while still allowing fixed host-call slots
- loaded ESKB chunks now retain their function table, and embedders can query
  and run named VM entry points through the public C ABI
- public VM loading now materializes ESKB string constants as VM strings instead
  of silently degrading them to integer lengths
- VM load options can select host-native-only policy and reject ESKB string
  constants at load time, giving embedded profiles a checked path toward
  content-pack-only text, pre-run desktop-native rejection, and required named
  script entries
- `embedded-vm` emits no-desktop-preamble ESKB through the public VM header and
  rejects desktop-native bytecode during emission; native Windows stubs expose
  the same symbols while the full VM remains disabled there
- compiler-produced ESKB now includes closed top-level VM function entries, so
  product hooks such as `tick` can satisfy load-time required-entry checks
- `eshkol-run --profile embedded-vm --require-vm-entry NAME` now performs
  build-time ESKB admission and removes rejected bytecode outputs
- the public VM C ABI can enumerate decoded ESKB function entries, so product
  tooling can inspect emitted hooks without reaching into `EskbModule`
- VM function metadata is available through the public ABI for hook arity,
  closure, and bytecode-budget inspection
- hosted Windows x86_64 validation now has a remote SSH harness for Jack's
  Tailscale Windows PC: `scripts/remote_windows_verify.sh` can build the native
  Visual Studio 2022 + ClangCL tree, or use `--suite-only` against Jack's cached
  MSYS/UCRT build for bounded `windows-lite` validation without hosted runners
- shared allocation and weak-reference helpers now live in a dedicated runtime-core translation unit instead of `arena_memory.cpp`
- freestanding-safe tensor index helpers now live in a dedicated runtime-core translation unit instead of the hosted-heavy runtime state implementation
- freestanding-safe tensor fill helpers now live in a dedicated runtime-core translation unit instead of the hosted-heavy runtime state implementation
- machine integer and raw pointer annotation surfaces exist in the HoTT parser/type-checker path
- pointer conversion and byte-offset arithmetic builtins exist for the low-level surface: `null-ptr`, `ptr->usize`, `usize->ptr`, and `ptr-add`
- tracked `Ptr` bindings round-trip through variable storage and typed codegen without collapsing back into generic heap-object handling
- `addr-of` exists for storage-backed bindings, giving the low-level surface a direct address-taking primitive without widening the general procedure ABI
- `compiler-fence` and `memory-fence` exist with explicit ordering operands for compiler-only and system-scope barrier emission
- `volatile-load` and `volatile-store!` exist for typed MMIO-style load/store lowering over raw pointers
- `atomic-load`, `atomic-store!`, `atomic-exchange!`, `atomic-compare-exchange!`, `atomic-fetch-add!`, `atomic-fetch-sub!`, `atomic-fetch-and!`, `atomic-fetch-or!`, and `atomic-fetch-xor!` exist for typed raw-pointer atomic access with explicit memory orderings
- `target-intrinsic` exists as a typed LLVM intrinsic escape hatch for low-level codegen
- declaration attributes exist for section placement, alignment, symbol export/remapping, weak linkage, used retention, and no-return metadata
- deeper runtime/freestanding implementation remains staged on `feature/platform-freestanding`
- embedded targets remain a public `v1.8-platform` milestone
- early program work should remain merge-safe during roadmap releases before `v1.8`

Immediate priorities:

- physical VM source extraction behind the explicit VM source families
- richer product entry-point manifest diagnostics beyond explicit
  `--require-vm-entry` admission
- first `runtime-freestanding` hook definitions
- low-level machine-facing language surface
- freestanding LLVM object and ELF pipeline
- stdlib capability partitioning

## Program Rules

- Hosted behavior must not regress while platform infrastructure is added.
- Freestanding work lands in small mergeable slices.
- Kernel and bootloader work remain downstream until the toolchain can produce stable freestanding artifacts.
- Architectural decisions are recorded in [DECISIONS.md](DECISIONS.md).

## Related Docs

- [../architecture/README.md](../architecture/README.md)
- [../development/README.md](../development/README.md)
- [../future/MULTIMEDIA_SYSTEM_ARCHITECTURE.md](../future/MULTIMEDIA_SYSTEM_ARCHITECTURE.md)
- [../../ROADMAP.md](../../ROADMAP.md)
