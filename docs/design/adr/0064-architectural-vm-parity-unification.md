---
kind: explanation
status: current
owner-area: vm
since: v1.4.0-connection
sources:
  - lib/backend/eshkol_vm.c
  - lib/backend/vm_char_string_ext.c
  - lib/backend/vm_curvature_ext.c
  - lib/backend/vm_prelude_source.h
  - scripts/vm_parity_audit.py
  - tests/vm_parity/PARITY.tsv
---
# ADR-0064: Architectural VM Parity Unification (Zero-Gap Single-Source Runtime)

- **Status:** Accepted (Target: `v1.4.0-connection`, supervised under Tsotchke-chan architectural gates)
- **Date:** 2026-11-26
- **Depends on:** ADR-0027 (recursive Taylor carrier), ADR-0029 (browser VM WebGPU dispatch), ADR-0030 (stackful coroutines), ADR-0058 (exact tensor element type), VM Parity Ratchet (`docs/VM_PARITY.md`, `tests/vm_parity/PARITY.tsv`)
- **Scope:** Unify the bytecode VM (`lib/backend/eshkol_vm.c`, `vm_compiler.c`, `vm_native.c`, `vm_wasm_repl.c`) with the native LLVM JIT/AOT engine across all non-native-only language surfaces, driving `PARITY.tsv` gaps from **322 → 0** and permanently enforcing `max_gaps = 0` in CI.

---

## 1. Context

Eshkol executes through two backends:
1. **Native LLVM backend (`lib/backend/llvm_codegen.cpp` + 38 sibling `*_codegen.cpp` modules)** — AOT binaries and OrcJIT (`eshkol-run -r`).
2. **Pure-C11 Bytecode VM (`lib/backend/eshkol_vm.c`, `vm_compiler.c`, `vm_native.c`, `vm_wasm_repl.c`)** — standalone interpreter (`eshkol-vm-standalone`), `--profile hosted-vm`, and the Emscripten-compiled browser playground (`site/static/eshkol-vm.wasm`, baseline `1,762,451` bytes / `1.76 MB`).

At `690b84532`, the parity manifest (`tests/vm_parity/PARITY.tsv`) tracks **997** surface entries:
- **630** `vm-supported`
- **45** `native-only-justified` (hardware lock-free atomics, raw pointer arithmetic, manual arena/region handles, memory/compiler fences, compile-time `extern`/`define-type`/`forall`/`include` forms)
- **322** `gap` (`316` `vm-unimplemented` + `6` `behavioral-divergence`)

### Root Architectural Cause of Drift
The VM evolved as a separate ~2.5 MB shadow compiler/runtime (`vm_compiler.c` 351 KB, `vm_native.c` 981 KB, `vm_parser.c` 118 KB, `vm_prelude_cache.h` 1.0 MB). Whenever a capability landed in LLVM codegen or `lib/core/*.esk`, the VM drifted unless someone manually re-implemented it a second time:
- **122 gaps** (69 ML/NN/optimizers + 35 tensor linalg/manipulation + 15 signal + 3 AD) exist solely because those kernels were emitted inside LLVM-only `tensor_*_codegen.cpp` rather than a shared pure-C11 runtime kernel library.
- **~110 gaps** (40 missing bindings, 16 string ops, 10 char ops, 25 `core.curvature` chart ops, R7RS/SRFI stdlib helpers, 3 leaked internal codegen names, plus 6 behavioral divergences) exist because the VM prelude (`vm_prelude_source.h`) diverged from `lib/core/*.esk` and lacked bindings for already-existing value tags (`VAL_CHAR`) and forms (`op:COMPOSE`, `op:LET_STAR_VALUES`, `op:MAKE_PARAMETER`, `op:PARAMETERIZE`).
- **~70 gaps** (36 port/file I/O, 6 time, 2 PRNG, 2 KB persistence, plus DNC/SDNC runtime hooks) lack a unified host-capability vtable separating hosted OS execution from fail-closed browser WASM execution.
- **~20 gaps** (9 stackful coroutines ADR-0030, 4 eval/environment reflection, remaining AD/DNC AST forms) require VM frame-stack / compiler lowering extensions.

---

## 2. Decision

We eliminate the shadow-runtime split via a **4-Wave Pure-C11 Unification Architecture** that preserves the VM's pure-C11 / zero-LLVM / Emscripten-WASM portability while establishing **one canonical implementation** for every non-native-only operation.

### Ordering Contract (Strict)
```
Wave 1 (Foundation: Core Semantics, VAL_CHAR ops, Scripted Prelude, Behavioral Divergences)
   │
   ├──▶ Wave 2 (Shared Pure-C11 Kernel Library: ML, Tensor Linalg, Signal, DNC/SDNC)
   │         ║   (parallel, strictly disjoint file ownership)
   └──▶ Wave 3 (Unified VmHostOps Capability VTable: Ports/IO, Time, PRNG, KB)
             │
             ▼
        Wave 4 (VM Stackful Coroutines ADR-0030, Eval/Env, Remaining AD/DNC AST, max_gaps=0 Ratchet)
```

Wave 1 MUST merge before Waves 2 and 3 begin integration (Wave 2 kernels and Wave 4 coroutines depend on corrected `eq?`/`eqv?`/`force`/`do`/`dynamic-wind`/`unquote-splicing` semantics and `VAL_CHAR` handling). Waves 2 and 3 run in parallel with strictly disjoint file ownership (`lib/core/kernels_*.c` + `inc/eshkol/runtime_kernels.h` vs `inc/eshkol/backend/vm_host_ops.h` + `lib/backend/vm_host_ops.c` + `lib/backend/vm_io.c`). Wave 4 integrates after Waves 1–3 and locks `max_gaps = 0` permanently in CI.

---

## 3. Architectural Components

### 3.1 Wave 1 — Core Semantics, `VAL_CHAR` Completeness, & Scripted Prelude Unification
1. **`VAL_CHAR` / `VAL_STRING` Invariant:**
   - `VAL_CHAR` is already a first-class tag in `vm_core.c` (`Value` payload stored in `as.i` as a Unicode scalar codepoint, matching `VmErrorValueSizeCheck`, `VmErrorValuePayloadOffsetCheck`, and `VmErrorValueTagSizeCheck` static assertions with zero struct-size or alignment change).
   - `VAL_STRING` remains a heap-indexed UTF-8 byte buffer (`VmString`) with unchanged alignment. All 10 character operations (`char-ci=?`, `char-ci<?`, `char-ci<=?`, `char-ci>?`, `char-ci>=?`, `char<=?`, `char>=?`, `char-foldcase`, `char-ready?`, `digit-value`) and all 16 extended string operations (`string`, `string-ci*`, `string-contains`, `string-contains?`, `string-copy!`, `string-foldcase`, `string-for-each`, `string-index`, `string-map`, `string-port?`, `string-set!`, `string<=?`, `string>=?`, `string>?`) operate directly on `VAL_CHAR` and `VAL_STRING` without collapsing characters to raw integers.
   - Verified under AddressSanitizer (`ASan`) + UBSan on boundary codepoints (`0`, ASCII, multi-byte UTF-8, invalid surrogates/indices).
2. **Behavioral Divergences (6 rows):**
   - Independently verify and resolve `eq?`, `eqv?`, `force`, `op:DO`, `op:DYNAMIC_WIND`, and `op:UNQUOTE_SPLICING` against `tests/vm_parity/resolved/*.esk` plus fresh nested/adversarial controls, then promote each to `vm-supported` in `PARITY.tsv`.
3. **Scripted Prelude Generation (Zero Manual Drift):**
   - `lib/backend/vm_prelude_cache.h` is **exclusively** generated by `scripts/regenerate_vm_prelude_cache.sh` via the CMake target `eshkol-vm-prelude-cache-gen` (enforced by CTest `vm_prelude_cache_is_current`). Never hand-edited.
   - Extend the compiled prelude/embedded-module closure (`lib/stdlib.esk` + `lib/core/list/search.esk` + `lib/core/curvature.esk` + R7RS/SRFI-1/SRFI-13 helpers) so pure-Eshkol stdlib definitions (`assq`, `assv`, `memv`, `partition`, `string-contains`, `acons`, `list*`, `list-copy`, `list-set!`, `vector-fill!`, `concatenate`, `last`, `inject-left`/`inject-right`/`left?`, exact/floor-division helpers, and all 25 `core.curvature` signed-curvature chart functions) are compiled once from canonical `.esk` sources rather than duplicated in C.
4. **Control / Binding Forms & Hygiene:**
   - Complete VM lowering for `op:COMPOSE`, `op:LET_STAR_VALUES`, `op:MAKE_PARAMETER`, `op:PARAMETERIZE`, and `hash-table-ref`.
   - Remove the 3 leaked internal codegen helper symbols (`%`, `%make-lazy-promise`, `%make-lazy-promise-force`) from the user-facing codegen dispatch manifest (or move them to internal compiler metadata) so internal lowering helpers never appear as user builtins.

### 3.2 Wave 2 — Shared Pure-C11 Portable Kernel Library (`inc/eshkol/runtime_kernels.h`)
1. **Single-Source C11 ABI:**
   - Factor portable implementations of all **69 ML/NN/optimizer ops**, **35 tensor linalg/manipulation ops**, **15 signal-processing ops**, and **DNC/SDNC numerical kernels** into pure C11 modules under `lib/core/kernels/` (`kernels_tensor_linalg.c`, `kernels_ml_ops.c`, `kernels_ml_optim.c`, `kernels_signal.c`, `kernels_dnc.c`), declared in `inc/eshkol/runtime_kernels.h`.
   - **Strict Purity Contract:**
     - **Pure C11 only** (compiles cleanly under `clang -std=c11`, `gcc -std=c11`, MSVC, and `emcc -std=c11`).
     - **Zero mutable global state** (all RNG, workspace, and state passed explicitly via context structs).
     - **Zero `malloc`/`calloc`/`realloc`/`free`** inside kernel bodies — every scratch or output allocation goes through an explicit caller-supplied `EshkolKernelAlloc` arena callback (backed by OALR arena on LLVM JIT/AOT and `VmArena` on the bytecode VM).
2. **Unified Backend Dispatch:**
   - LLVM codegen (`tensor_*_codegen.cpp`, `ml`, `signal`) calls the shared C11 kernels (preserving optional CUDA/Metal/XLA/BLAS fast paths where available and bit-identical or explicitly flagged).
   - `vm_native.c` calls the **exact same C11 functions** through thin value-unboxing adapters, eliminating duplicate math implementations permanently.
3. **Bit-Identical Differential Requirement:**
   - Every kernel family is checked by `scripts/run_vm_parity.sh` and `scripts/run_wasm_differential.sh` across JIT, AOT, hosted VM, and WASM (IEEE-754 deterministic paths, no undocumented ULP drift).

### 3.3 Wave 3 — Unified `VmHostOps` Capability VTable (`inc/eshkol/backend/vm_host_ops.h`)
1. **Explicit Host Boundary:**
   - Consolidate all **36 port/file I/O ops**, **6 time ops**, **2 PRNG ops**, **2 KB persistence/prefix-query ops**, and OS/filesystem hooks behind a single `const VmHostOps*` vtable registered at VM initialization.
   - **Hosted Implementation (`lib/backend/vm_host_ops_hosted.c`):** Implements full POSIX/Win32 I/O, directories, ports, high-resolution clocks, PRNG, and KB persistence, enforcing the **exact same narrow-only capability policy store** used by LLVM JIT and AOT.
   - **WASM Implementation (`lib/backend/vm_host_ops_wasm.c`):**
     - Compiles with `-fno-exceptions` (and `-fno-rtti` for any C++ TU).
     - **Strict fail-closed default:** any unimplemented or denied host operation returns a structured capability-refusal error without leaking host paths, pointers, or environment metadata. No silent no-op stubs.
     - **Immutable runtime policy:** the WASM capability policy table is `const`/frozen at instantiation and cannot be widened by guest code.
     - Verified in real WASM execution via `node` (`scripts/run_wasm_differential.sh` + dedicated capability-refusal tests) and headless browser test harness.
2. **Disjoint Ownership From Wave 2:**
   - Wave 3 owns `inc/eshkol/backend/vm_host_ops.h`, `lib/backend/vm_host_ops_*.c`, `lib/backend/vm_io.c`, and host-table hooks in `eshkol_vm.c` — it does **not** modify Wave 2's numerical kernel dispatch.

### 3.4 Wave 4 — VM Stackful Coroutines, Eval/Environment, AD/DNC AST Ops, & `max_gaps = 0` Ratchet
1. **Stackful Coroutines (ADR-0030, 9 ops):**
   - Implement `make-coroutine`, `coroutine?`, `coroutine-resume`, `coroutine-yield`, `coroutine-status`, `coroutine-raise`, `coroutine-release!`, `coroutine-stats`, and `current-coroutine` natively on the VM's frame/value stack (reusing the saved-frame machinery already proven for `VAL_CONTINUATION` / `call/cc`), with ASan/leak verification on abandoned coroutines.
2. **Eval & Environment Reflection (4 ops):**
   - Wire `eval`, `current-environment`, `interaction-environment`, and `null-environment` to invoke the VM compiler + evaluator cleanly against isolated lexical/global environment snapshots.
3. **Remaining AD & DNC AST Forms:**
   - Complete VM lowering for `op:DIFF`, `op:DIRECTIONAL_DERIV`, `op:LAPLACIAN` (aligned to ADR-0027 recursive Taylor carrier semantics), `op:DNC_*`, `op:SDNC_*`, and `op:KB_QUERY_PREFIX`.
4. **Permanent Zero-Gap Ratchet:**
   - Update `scripts/vm_parity_audit.py` with `--max-gaps 0` enforced in CI (`assurance-gates` / `surface-manifest`), so no future PR can merge a new native construct without either VM parity or an explicitly justified `native-only-justified` waiver.

---

## 4. Mandatory Verification Gates (Hard Blockers)

Every wave PR must pass ALL applicable gates before merge:

| Gate | Requirement | Enforcement |
|---|---|---|
| **WASM Size Budget** | `site/static/eshkol-vm.wasm` must not grow by `> 5%` over baseline (`1,762,451` B → hard ceiling `1,850,573` B) in Wave 2 without DCE / `-Os` / lazy-loading optimization. | `scripts/check_wasm_size_budget.py` (added in Wave 1/2) |
| **WASM Fail-Closed Security** | Unpermitted host ops in `VmHostOps` WASM build raise structured refusal; capability policy immutable at runtime; compiled `-fno-exceptions`; verified in real Node/browser WASM execution. | `scripts/run_wasm_differential.sh` + WASM capability negative controls |
| **Memory & Alignment Safety** | Zero ASan / UBSan findings across `VAL_CHAR`, `VAL_STRING`, shared C11 kernels (zero `malloc`, zero mutable globals), `VmHostOps`, and VM coroutines. | `linux-x64-asan-ubsan` + local ASan runs |
| **Bit-Identical Differential Parity** | Every promoted symbol has an execution-backed differential test passing across JIT, AOT, hosted VM, and WASM (`run_vm_parity.sh` + `run_wasm_differential.sh`). | CI + `tests/vm_parity/corpus/` |
| **Prelude Cache Currency** | `lib/backend/vm_prelude_cache.h` is always regenerated via `scripts/regenerate_vm_prelude_cache.sh` and verified by `vm_prelude_cache_is_current`. | CTest + CI |
| **Deterministic Cognitive Workloads** | Zero output drift on deterministic Eshkol workloads used by Noesis/qLLM (`sblc_relax_gate`, `run_stdlib_tests.sh`, cognitive fixtures). | Regression gate |
| **Zero-Gap Ratchet (Wave 4)** | `PARITY.tsv` reaches `gap = 0`; `vm_parity_audit.py --max-gaps 0` passes. | CI `surface-manifest` / `assurance-gates` |

---

## 5. Consequences

- **Positive:** True write-once, run-everywhere semantics across LLVM AOT, OrcJIT, Hosted Bytecode VM, and Browser WASM; eliminates the ~2.5 MB shadow-runtime maintenance tax; removes the largest public caveat from `v1.4.0-connection`; unblocks Stage 0 three-engine parity for reflective kernel judgments.
- **Trade-off:** Extracting 122 numeric kernels into pure C11 requires strict discipline around arena allocation (`EshkolKernelAlloc`) and WASM binary size (`<= +5%`), enforced automatically by CI gates.
