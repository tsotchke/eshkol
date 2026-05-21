# Platform Program Decisions

This file is an append-only log of architecture and governance decisions for the freestanding / platform program.

## Status Values

- `Accepted`
- `Provisional`
- `Superseded`
- `Deferred`

---

## D-0001

- Date: 2026-04-15
- Status: Accepted
- Title: Start the platform program during `v1.2-scale`

### Context

The public roadmap places embedded cross-compilation in `v1.8-platform`, but the compiler, runtime, and language work needed for that milestone is too large to begin only at `v1.8`.

### Decision

Start the platform program during `v1.2-scale` as a parallel infrastructure effort that converges into the public roadmap by `v1.8-platform`.

### Consequences

- platform work can proceed now
- public roadmap wording does not need to change drastically
- only merge-safe infrastructure lands early

---

## D-0002

- Date: 2026-04-15
- Status: Accepted
- Title: Use a long-lived platform branch

### Context

The current active branch is `feature/v1.2-scale`, and there is already active work in central compiler/runtime files.

### Decision

Use:

- `feature/v1.2-scale` as the current release integration branch
- `feature/platform-freestanding` as the long-lived platform branch

### Consequences

- platform work can advance without destabilizing the release stream
- regular merge discipline is required

---

## D-0003

- Date: 2026-04-15
- Status: Accepted
- Title: The first deliverable is a freestanding toolchain, not a kernel

### Context

The temptation is to start immediately with a bootloader or kernel implementation.

### Decision

The first success metric is a stable freestanding Eshkol platform:

- profiles
- runtime split
- low-level language surface
- freestanding LLVM path
- BSP contract

Kernel work begins downstream only after that foundation exists.

### Consequences

- effort stays focused on reusable infrastructure
- `eshkol-kernel` remains downstream

---

## D-0004

- Date: 2026-04-15
- Status: Accepted
- Title: The VM is a first-class part of the platform strategy

### Context

Eshkol already has a production VM and ESKB format. It is useful beyond browser deployment.

### Decision

The platform program explicitly includes a `freestanding-vm` profile and a VM host-hook architecture.

### Consequences

- platform portability is not limited to LLVM-native targets
- the VM becomes a candidate runtime for monitors, recovery, and constrained systems

---

## D-0005

- Date: 2026-04-15
- Status: Accepted
- Title: First reference targets are QEMU-friendly general-purpose ISAs

### Context

The initial goal is to validate the platform architecture quickly and repeatably.

### Decision

Use these reference targets in order:

1. x86_64 QEMU PC
2. AArch64 QEMU virt
3. RISC-V QEMU virt

Real MCU support comes later once the freestanding contract is stable.

### Consequences

- first bring-up stays debuggable and automatable
- MCU-specific work does not block the core platform architecture

---

## D-0006

- Date: 2026-04-15
- Status: Accepted
- Title: Governance is founder-led with bounded delegation

### Context

The founder/maintainer is doing most of the work, but some bounded tasks may be delegated.

### Decision

Core architecture, syntax, runtime boundary, and repo-boundary decisions remain centralized. Delegation is implementation-focused and topic-branch bounded.

### Consequences

- design coherence is preserved
- helpers can still contribute effectively on decision-complete tasks

---

## D-0007

- Date: 2026-04-15
- Status: Accepted
- Title: Keep the platform integration branch in its own dedicated worktree

### Context

The roadmap branch, the platform integration branch, and short-lived topic branches need to move in parallel without stash churn or accidental branch contamination.

### Decision

Use:

- `~/Desktop/eshkol` for the active release/mainline branch
- `~/Desktop/eshkol-platform` for `feature/platform-freestanding`
- additional short-lived worktrees for active `topic/platform-*` branches when needed

### Consequences

- branch roles stay physically separated
- the platform integration branch remains continuously available for sync and validation
- topic branches remain disposable after merge

---

## D-0008

- Date: 2026-04-15
- Status: Accepted
- Title: Use explicit merge commits for platform syncs and topic-branch integration

### Context

The platform program is long-running and will repeatedly reconcile roadmap syncs, topic slices, and eventual merge-back into the release stream.

### Decision

Use explicit merge commits for:

- syncing mainline into `feature/platform-freestanding`
- merging accepted `topic/platform-*` branches into `feature/platform-freestanding`

### Consequences

- integration history remains legible
- platform slice boundaries stay visible
- later audit, bisect, and rollback are safer

---

## D-0009

- Date: 2026-04-15
- Status: Accepted
- Title: Runtime ownership is responsibility-based, not directory-based

### Context

The current implementation mixes allocator substrate, signal/process/runtime state, configuration, logging, image and ONNX helpers, knowledge-base persistence, and higher-level language services under `lib/core/`. Treating the entire directory as "the runtime" would produce a bad split and make freestanding support harder.

### Decision

The runtime split is responsibility-based:

- `runtime-core` owns the value ABI, allocator substrate, and profile-independent runtime contracts
- `runtime-hosted` owns process, filesystem, terminal, env, temp path, host compiler/linker, and hosted libc wrapper behavior
- higher-level language services remain outside the runtime family split even if they live under `lib/core/`

The concrete baseline for this classification is `docs/platform/RUNTIME_INVENTORY.md`.

### Consequences

- runtime extraction work is driven by documented ownership, not by directory moves
- `platform_runtime.h` and `runtime_exports.h` are treated as hosted-runtime surfaces from the start
- files such as `logic.cpp`, `workspace.cpp`, `inference.cpp`, and `introspection.cpp` are not silently folded into `runtime-core`

---

## D-0010

- Date: 2026-04-15
- Status: Accepted
- Title: Runtime decomposition begins with internal source sets, not immediate archive changes

### Context

The runtime is still delivered as part of `eshkol-static`, and too many core files still straddle freestanding and hosted concerns to make a clean archive split in one step.

### Decision

Introduce explicit internal build buckets first:

- `runtime-core`
- `runtime-hosted`
- `runtime-split-pending`

These are represented in CMake as internal object libraries while `eshkol-static` remains the delivered aggregate archive.

### Consequences

- build ownership becomes explicit without destabilizing downstream link behavior
- the remaining mixed files are visible instead of being silently misclassified
- later runtime archive extraction can proceed incrementally from an already-structured build graph

---

## D-0011

- Date: 2026-04-15
- Status: Accepted
- Title: Machine integer surface starts as a type-system and ABI slice, not a full arithmetic semantics split

### Context

The freestanding roadmap needs `u8`/`u16`/`u32`/`u64`/`usize` and signed counterparts, but the current Eshkol numeric pipeline still assumes generic integer arithmetic normalizes through the existing `Int64` / `Float64` paths. Forcing width-specific arithmetic semantics into this slice would spread through the parser, checker, runtime representation, arithmetic optimizer, and polymorphic runtime at once.

### Decision

Add the machine integer family to the HoTT builtin type environment, builtin-name resolution, closure type metadata, and typed-value extraction paths now, while keeping generic arithmetic promotion normalized to `Int64` for the integer family. Treat this first slice as the low-level type and ABI surface required for annotations and future freestanding lowering.

### Consequences

- Eshkol can now parse and resolve `i8`/`i16`/`i32`/`i64`/`isize` and `u8`/`u16`/`u32`/`u64`/`usize` in type annotations
- closure metadata and typed tagged-value extraction preserve those builtin type IDs instead of collapsing them immediately to `Value`
- generic arithmetic still widens machine integer annotations through the existing `Int64` path until dedicated width-aware lowering is introduced
- the next low-level language slices can build pointers, volatility, and native ABI lowering on top of a stable machine integer vocabulary

---

## D-0012

- Date: 2026-04-15
- Status: Accepted
- Title: `runtime_exports.h` wrappers live in dedicated hosted runtime glue

### Context

`lib/core/platform_runtime.cpp` previously mixed host/process/toolchain services from `platform_runtime.h` with generated-code and JIT ABI wrappers exported through `runtime_exports.h`. That hid the boundary between platform runtime services and hosted wrapper glue needed by generated code.

### Decision

Keep `platform_runtime.cpp` focused on `platform_runtime.h`, and move the `runtime_exports.h` implementation into a dedicated hosted runtime unit: `lib/core/runtime_exports_hosted.cpp`.

### Consequences

- the hosted wrapper surface now has an explicit build boundary
- later `runtime-core` and `runtime-freestanding` work can reason about `platform_runtime` and `runtime_exports` independently
- generated-code ABI glue is not implicitly bundled into future freestanding runtime layers

---

## D-0013

- Date: 2026-04-15
- Status: Accepted
- Title: Raw pointer types start as an annotation surface before pointer operations

### Context

The platform roadmap needs `ptr<T>`-style vocabulary for MMIO, foreign calls, buffer views, and later freestanding ABI lowering. Landing pointer conversions, address-of, volatile memory access, and pointer arithmetic all at once would couple parser syntax, type checking, codegen, memory safety policy, and runtime representation in one large change.

### Decision

Add a raw pointer type constructor to the HoTT type surface first. The accepted spelling is `(ptr T)`, with `pointer` as a parser alias and `ptr` / `pointer` as builtin type aliases. The initial runtime representation is the existing pointer representation, and type resolution preserves the `Ptr` builtin family while later slices add conversions and address-producing operations.

### Consequences

- Eshkol can now parse and resolve pointer annotations such as `(ptr u8)`
- pointer type expressions copy and print round-trip as `(ptr T)`
- the pointer surface depends on the machine integer vocabulary already present in `master`
- pointer conversions, address-of, volatile load/store, and stricter ABI semantics remain separate follow-up slices
