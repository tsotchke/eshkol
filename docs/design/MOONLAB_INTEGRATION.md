# Moonlab Quantum Simulator Integration

Status: SHIPPED — the integration described in this RFC is implemented and gated
behind `-DESHKOL_QUANTUM_ENABLED=ON` (`agent.quantum`, `agent.pqc`). As of
v1.3.4-evolve the pinned Moonlab revision is **v1.2.0**, which adds
`vqe_compute_qgt` (the quantum geometric tensor, enabling quantum-natural-gradient
optimization) and a smooth first-principles H2/LiH potential-energy surface. The
H2 equilibrium oracle at bond length 0.735 Å is `-1.142200155381` Ha (a `-2.95e-5`
Ha shift from the earlier PES). Differentiable quantum-chemistry examples and an
arbitrary-order-AD H2 vibrational-frequency example ship with the release. The RFC
text below is retained for the design rationale.

Target Eshkol version: v1.3.4-evolve (`inc/eshkol/eshkol.h`)
Scope: wire the Moonlab quantum simulator into Eshkol as quantum-computing
builtins, and replace Eshkol's placeholder `quantum-random` with a real,
Bell-verified entropy source.

This is a plan for the maintainer to review and decide go/no-go and scope. It
proposes no builtin wiring and touches no source. Every design claim below
cites the file and symbol it is grounded in, across four repositories:

- Eshkol — `/Users/tyr/Desktop/eshkol` (this repo).
- Moonlab — `github.com/tsotchke/moonlab` (MIT; pinned at v1.2.0 as of v1.3.4-evolve).
- quantum_rng — `/Users/tyr/Desktop/quantum_rng` (MIT, `github.com/tsotchke/quantum_rng`).

---

## 0. Context: the two projects already interlock one-way

Moonlab already **consumes** Eshkol. Moonlab's GPU backend bridges into
Eshkol's Metal/CUDA precision tiers for VQE/QAOA GEMM:
`src/optimization/gpu/backends/gpu_eshkol.h` in Moonlab exposes
`moonlab_eshkol_zgemm` (`gpu_eshkol.h:123`) dispatching through Eshkol, and
Eshkol carries the reciprocal note at `lib/backend/gpu/gpu_memory.mm:56`
(`// (Moonlab 2026-04-19 integration report). Per-call logging must be opt-in.`).

This document designs the **reverse** direction: **Eshkol consumes Moonlab**.
The two directions must share conventions (precision tiers, error codes, memory
ownership) so the pairing is symmetric; Section 3 pins those conventions from
the existing bridge.

---

## 1. Moonlab public C API surface an Eshkol binding would call

### 1.1 Library build facts (Moonlab `CMakeLists.txt`)

- Target/product: CMake target `quantumsim` -> `libquantumsim.{dylib,so}` /
  `quantumsim.dll` (`CMakeLists.txt:1051-1063`, `add_library(quantumsim ...)`).
- Static vs shared: selectable by `QSIM_BUILD_SHARED` (SHARED, versioned) or
  `QSIM_BUILD_STATIC` (STATIC); all three existing language bindings use the
  **shared** library.
- PIC: `set(CMAKE_POSITION_INDEPENDENT_CODE ON)` globally (`CMakeLists.txt:167`).
- Export macro: `MOONLAB_API` in `src/applications/moonlab_api.h:22-39`
  (GCC/Clang `__attribute__((visibility("default")))`, MSVC dllexport/dllimport
  gated on `MOONLAB_BUILDING_SHARED`/`MOONLAB_USING_SHARED`). Symbol hiding is
  opt-in (`QSIM_HIDDEN_VISIBILITY`, default OFF), so today every non-static
  `moonlab_*`/`quantum_*`/`gate_*`/`vqe_*` symbol is exported.
- Consumer link path: `-lquantumsim` at build time, or `dlopen`+`dlsym` the
  `moonlab_*` symbols at runtime feature-gated by `moonlab_abi_version()`
  (`moonlab_export.h:6-8, 72`).

There are **two ABI tiers**, and the binding should treat them differently:

1. **Committed stable ABI** — the `moonlab_*` symbols in
   `src/applications/moonlab_export.h` (ABI version 0.3.0). Name+signature
   stable across all 0.x. This is the frozen `dlsym` contract: CA-MPS tensor
   networks, TDVP, DMRG, topology, QRNG, ML-KEM.
2. **Lower-level `MOONLAB_API`-tagged surface** — `quantum_state_*`, `gate_*`,
   `measurement_*`, `vqe_*` in `src/quantum/` and `src/algorithms/`. Exported
   and used directly by the Python/Rust bindings, but not part of the frozen
   contract. The dense 32-qubit state-vector API lives here, **not** in
   `moonlab_export.h`.

Design consequence: **prefer the committed ABI where it covers a need** (QRNG,
PQC, CA-MPS, TDVP), and pin an exact Moonlab version when binding the
lower-level state-vector/gate/VQE surface. Feature-gate at load with
`moonlab_abi_version()`.

### 1.2 Opaque handles and structs

| Type | Header | Shape | Notes |
|---|---|---|---|
| `moonlab_ca_mps_t` | `moonlab_export.h:217` | opaque fwd-decl | Clifford-assisted MPS state |
| `moonlab_tdvp_engine_t` | `moonlab_export.h:521` | opaque fwd-decl | adaptive-bond TDVP engine |
| `quantum_state_t` | `src/quantum/state.h:131` | **public struct** | dense state vector (`num_qubits`, `state_dim`, `amplitudes`, `owns_memory`, `gpu_state`, `gpu_backend`) |
| `measurement_result_t` | returned by `quantum_measure` (`gates.h:293`) | struct-by-value | `{outcome, probability, entropy}` |
| `pauli_hamiltonian_t`, `vqe_ansatz_t`, `vqe_optimizer_t`, `vqe_solver_t`, `noise_model_t` | `src/algorithms/vqe.h` | opaque handles | VQE building blocks |
| `qrng_v3_ctx_t` | `src/applications/qrng.h:180` | public struct | lower-level QRNG context (not needed via stable ABI) |

Because `quantum_state_t` is a public struct, the Eshkol binding should hold it
as an opaque `ptr` handle (allocate via `quantum_state_create`) and never mirror
its layout in Scheme — the same discipline the Rust/JS bindings use for the
opaque handles, avoiding the ctypes-mirror fragility the Python binding accepts.

### 1.3 Error and ownership conventions

- Stable `moonlab_*` ABI: `int` return, `0` = success, negative = error.
  Scalar-energy entry points (DMRG) return `double` with `DBL_MAX` as the error
  sentinel (`moonlab_export.h:296-299`); topology one-shots use `INT_MIN`.
  `moonlab_status_string(int module, int status)` (`moonlab_export.h:700`)
  stringifies, with `moonlab_status_module_t` enumerating modules.
- State-vector/gate layer: `qs_error_t` enum (`state.h:79-88`): `QS_SUCCESS=0`,
  `QS_ERROR_INVALID_QUBIT=-1`, ... `QS_ERROR_DRIVER=-8`.
- QRNG layer: `qrng_v3_error_t` (`qrng.h:140-150`).
- Ownership: every `*_create`/`*_clone` has a matching `*_free`/`*_destroy`
  (`moonlab_ca_mps_free` `:222`, `quantum_state_destroy` `state.h:382`,
  `moonlab_tdvp_engine_free` `:678`, `vqe_*_free`). `moonlab_z2_lgt_1d_build`
  allocates `*out_paulis`/`*out_coeffs` the **caller must `free()`**
  (`moonlab_export.h:466`). `moonlab_qrng_bytes` is zero-setup: it lazily builds
  a process-lifetime context freed at `atexit` (`moonlab_qrng_export.c:33-75`).

### 1.4 The ~30 core functions worth exposing to Scheme, grouped

**Lifecycle / diagnostics**
- `moonlab_abi_version(int*,int*,int*)` — `moonlab_export.h:72` — version gate.
- `moonlab_status_string(int,int)` — `moonlab_export.h:700` — error strings.

**State-vector ops** (`src/quantum/state.h`)
- `quantum_state_create(int)` / `quantum_state_destroy` — `state.h:373/382`.
- `quantum_state_clone` / `_reset` / `_normalize` — `state.h:184/190/209`.
- `quantum_state_get_amplitude` / `_get_probability` — `state.h:238/246`.
- (GPU-resident) `quantum_state_create_gpu` / `_sync_to_host` / `_sync_from_host`
  — `state.h:414/425/439` — note this is where a state can ride Eshkol's own GPU.

**Gates** (`src/quantum/gates.h`, all `qs_error_t gate_*(quantum_state_t*, ...)`)
- `gate_hadamard` `:59`; `gate_pauli_x/_y/_z` `:37/44/51`;
  `gate_s/_s_dagger/_t/_t_dagger` `:66/73/80/87`.
- `gate_phase/_rx/_ry/_rz/_u3` `:94/101/108/115/124` (parametric — these carry
  the `double theta` that the AD bridge differentiates).
- `gate_cnot/_cz/_swap/_cphase` `:137/145/161/170`; `gate_crx/_cry/_crz`
  `:178/186/194`; `gate_toffoli/_fredkin` `:208/217`; `gate_qft/_iqft` `:248/255`.

**Measurement** (`src/quantum/measurement.h`, `quantum_measure` in `gates.h`)
- `quantum_measure` `gates.h:293`; `quantum_measure_all_fast` `gates.h:341`.
- `measurement_single_qubit` `:64`; `measurement_all_qubits` `:83`;
  `measurement_partial` `:94`.
- `measurement_probability_one/_zero` `:34/39`;
  `measurement_expectation_z/_x/_y` `:105/111/117`.

**VQE / algorithms** (`src/algorithms/vqe.h`)
- `pauli_hamiltonian_create/_free` `:135/144`; prebuilt molecular Hamiltonians
  `vqe_create_h2_hamiltonian/_lih_/_h2o_` `:174/185/195`;
  `vqe_exact_ground_state_energy` `:203` (FCI reference).
- Ansaetze `vqe_create_hardware_efficient_ansatz/_uccsd_/_symmetry_preserving_`
  `:246/261/272`; `vqe_ansatz_free`/`vqe_apply_ansatz` `:282/293`.
- `vqe_optimizer_create/_free` `:347/353`; `vqe_solver_create/_free`/`vqe_solve`
  `:407/418/496`.
- `vqe_compute_energy(solver, params)` `:476` — the scalar loss E(theta).
- `vqe_compute_gradient(solver, params, grad)` `:509` — exact gradient (see Sec 4).
- DMRG scalar drivers `moonlab_dmrg_tfim_energy` `moonlab_export.h:314`,
  `_heisenberg_energy` `:342`.

**QRNG** (the honest-random source)
- `moonlab_qrng_bytes(uint8_t* buf, size_t size)` — `moonlab_export.h:96`,
  impl `moonlab_qrng_export.c:57`. The stable, zero-setup, thread-safe entry.
  Doc (`moonlab_export.h:78-84`): "combines a hardware entropy pool (RDSEED /
  /dev/urandom / SecRandomCopyBytes) with a Bell-verified quantum simulation
  layer". Returns 0 / -1 (null buf) / -2 (engine init failed) / -3 (byte-gen
  failure, e.g. a rejected Bell-verification epoch).
- Lower-level (optional, for entropy diagnostics): `qrng_v3_verify_quantum`,
  `qrng_v3_get_entanglement_entropy` — `qrng.h:404/415`.

**PQC / ML-KEM (FIPS 203)** (`moonlab_export.h`, impl `src/crypto/mlkem/mlkem.c`)
- `moonlab_mlkem512_keygen_qrng/_encaps_qrng/_decaps` `:147/160/172`.
- `moonlab_mlkem768_keygen_qrng/_encaps_qrng/_decaps` `:183/184/185`
  (recommended default).
- `moonlab_mlkem1024_keygen_qrng/_encaps_qrng/_decaps` `:194/195/196`.
- Buffer-size macros `MOONLAB_MLKEM{512,768,1024}_{PUBLICKEY,SECRETKEY,CIPHERTEXT,SHAREDSECRET}BYTES`.

**CA-MPS / TDVP (large-system tensor-network engines, stable ABI)** — bind in a
later stage if there is demand; lifecycle `moonlab_ca_mps_create/_free`
`:220/222`, gates `:234-261`, observables `moonlab_ca_mps_expect_pauli_sum`
`:280`; TDVP `moonlab_tdvp_create_heisenberg/_tfim` `:553/574`, `_step`/`_evolve_to`
`:595/608`.

**Extension registries** (advanced; out of initial scope)
- `moonlab_register_decoder`/`_unregister_decoder` — `decoder_bench.h:151/158`.
- `moonlab_register_vendor_noise_backends` / `_profile` — `vendor_noise_backend.h:84/127`.

### 1.5 Existing embedding patterns (to mirror)

- **Python** (`bindings/python/moonlab/core.py`): `ctypes.CDLL` searching
  `MOONLAB_LIB_PATH`/`MOONLAB_LIB_DIR` then `build_release`/`build`/repo-root.
  Mirrors `quantum_state_t` as a `ctypes.Structure` and passes `byref` — the
  fragile pattern we avoid.
- **Rust** (`bindings/rust/moonlab-sys/build.rs:60-69`): `bindgen`; links
  `static=quantumsim` if `.a` present else `dylib`, rpath to `MOONLAB_LIB_DIR`,
  plus Accelerate/Metal/Foundation/Security/OpenMP/libm. Wraps opaque handles in
  RAII types calling `*_free` on `Drop`. **This is the model the Eshkol binding
  follows** (opaque handle + explicit free).
- **JavaScript** (`bindings/javascript`): Emscripten WASM of the
  `moonlab_export_lean.c` half — **deliberately excludes qrng/hardware-entropy**.
  Relevant to Section 5(a): the browser has no hardware entropy.

---

## 2. The established Moonlab<->Eshkol conventions to mirror

From `src/optimization/gpu/backends/gpu_eshkol.h` (the existing Moonlab->Eshkol
bridge), the reverse binding must reuse the same vocabulary:

- **Precision tiers** — `moonlab_eshkol_precision_t` (`gpu_eshkol.h:51-56`):
  `EXACT=0` (fp53 bit-exact fp64), `HIGH=1` (df64), `FAST=2` (f32), `ML=3`
  (fp24). These map to Eshkol's `ESHKOL_GPU_PRECISION` env semantics. Any
  Eshkol builtin that offers a Moonlab GPU-resident state (`quantum_state_create_gpu`)
  must expose tiers with **identical names/order**.
- **Status enum** — `moonlab_eshkol_status_t` (`gpu_eshkol.h:58-65`): `OK=0`,
  `NOT_BUILT=-1`, `NO_GPU=-2`, `INVALID_ARGS=-3`, `DISPATCH_FAILED=-4`, `OOM=-5`.
  The reverse binding's own status surface should follow the 0/negative pattern.
- **Memory ownership** — caller-owns-everything; host pointers passed by
  reference; GPU staging buffers stay internal (`gpu_eshkol.h:96-123`). The
  Eshkol shim must likewise never hand Moonlab a pointer it will free.
- **Complex element type** — `typedef double _Complex moonlab_cplx_t;`
  (`gpu_eshkol.h:49`), interleaved (re, im). Amplitude readback from Moonlab
  states uses the same interleaved-complex convention.
- **Compile-gate + graceful fallback** — the bridge is gated by
  `QSIM_ENABLE_ESHKOL`; when off, symbols still declare and `available()`
  returns 0. The Eshkol side mirrors this with an `ESHKOL_ENABLE_MOONLAB`
  gate (Section 6) so a build without libquantumsim degrades honestly rather
  than failing to link.

---

## 3. Eshkol's FFI / binding pattern (what the quantum binding mirrors)

Eshkol has **two** vehicles for a C-backed capability. Both are used here.

### 3.1 Vehicle A — the `agent.X` module pattern (for the rich quantum surface)

An external C library is surfaced as a Scheme module `lib/agent/<name>.esk`
that declares `extern` bindings to a C shim in `lib/agent/c/agent_<name>.c`.
Example, `lib/agent/sqlite.esk`:

```
(provide sqlite-open sqlite-close ...)
(extern i64  sqlite-open-raw   ptr     :real eshkol_sqlite_open)
(extern i32  sqlite-exec-raw   i64 ptr :real eshkol_sqlite_exec)
```

The `extern` form is `(extern <ret-type> <scheme-name> <arg-types...> :real
<c-symbol>)` (see `lib/agent/http.esk:20-62`, `lib/agent/sqlite.esk:26-45`).
`:real` maps the Scheme-visible name to the actual C symbol. Types are the FFI
primitives `i32`/`i64`/`double`/`ptr`/`void`. The `.esk` module wraps the raw
externs in ergonomic, memory-safe Scheme (e.g. `with-db`, `with-statement`).

The C shim (`lib/agent/c/agent_sqlite.c`, `agent_http_client.c`, etc.) is a thin
translation layer that includes the vendor header and exposes `eshkol_*` /
`qllm_*` symbols.

**Link path (AOT + JIT)** — `CMakeLists.txt:3117-3420`:
- All `lib/agent/c/*.c` shims compile into a STATIC lib `eshkol-agent-ffi`
  (`CMakeLists.txt:3195`). Optional vendor deps are discovered by
  `pkg_check_modules` (sqlite3 `:3181`, libcurl `:3190`, pcre2 `:3172`) and
  attached **PUBLIC** so they propagate to `eshkol-run`.
- **JIT/REPL**: `eshkol-run` force-loads `eshkol-agent-ffi`
  (`-force_load` / `--whole-archive`) so symbols stay live for `dlsym`
  (`CMakeLists.txt:3203-3210`).
- **AOT**: the link args are assembled into `ESHKOL_HOST_AGENT_FFI_LINK_ARGS`,
  spliced into `build_config.h`; `eshkol-run.cpp` appends them at the AOT link
  step **only when the user source contains `(require agent.…)`**
  (`CMakeLists.txt:3276-3296`). This gating is what lets programs that do not
  use the capability avoid pulling in the vendor library.

This is the right vehicle for the **rich quantum surface** (state vectors,
gates, measurement, VQE, PQC): many functions, opaque handles, ergonomic Scheme
wrappers, and it is only linked when `(require quantum.moonlab)` appears.

### 3.2 Vehicle B — a core builtin in the name->id tables (for `quantum-random`)

`quantum-random` already exists as a **core builtin**, not an agent module. Its
registration is in `lib/backend/eshkol_vm.c:642-644` (mirrored in
`lib/backend/vm_prelude_cache.h:1415-1417`):

```
{"quantum-random",       1860, 0},
{"quantum-random-int",   1861, 1},
{"quantum-random-range", 1862, 2},
```

There are **two current backends for this builtin, and they diverge** — the
honesty problem is present on both, at different severities:

1. **VM interpreter path** — `lib/backend/vm_native.c`. `vm_qrng_state`
   (`:4299`) is a `static uint64_t`; `vm_qrng_next_u64` (`:4304`) is a bare
   xorshift64* self-seeded from a stack address / `gettimeofday` / `getpid`
   (`:4305-4322`); the dispatch cases `1860/1861/1862` (`:12392-12420`, block
   header "Quantum-inspired RNG") call it. This is a **plain PRNG** — the
   in-tree comment even says so (`vm_native.c:4301`, "xorshift64* PRNG").
2. **LLVM AOT/JIT path** — `lib/backend/llvm_codegen.cpp` dispatches
   `quantum-random`/`-int`/`-range` by name (`:14405-14407`) to
   `codegenQuantumRandom` etc. (`:32446/32457/32468`), emitting calls to extern
   C functions `eshkol_qrng_double`/`_uint64`/`_range` declared at
   `:3612-3629`. Those are implemented in `lib/quantum/quantum_rng_wrapper.c`
   (`eshkol_qrng_double` `:31`) over `lib/quantum/quantum_rng.c`, compiled into
   the **core runtime archive** (`CMakeLists.txt:1608-1609`). But
   `lib/quantum/quantum_rng.c` is the **vendored v1 "quantum-inspired"
   simulation** — a software `qrng_ctx` mixing loop seeded from
   `get_system_entropy` (wall clock, pid, `clock()`, stack address, `rdtsc` —
   `quantum_rng.c:123-143`). It is better than xorshift but is **not** the
   Bell-verified, hardware-entropy, NIST-SP800-90B-health-tested `qrng_v3`
   engine that `moonlab_qrng_bytes` wraps.

So the same builtin can produce different numbers on the VM vs the AOT/JIT
backend, and **neither backend uses real quantum entropy**. Because
`quantum-random` is a registered core builtin on both the `vm` and `native_llvm`
backends (`tests/coverage/language_surface.json:7318-7353`), and because the
core-builtin path is the only one that works under WASM (the agent.X `require`
needs a filesystem and does not run in the browser), the honest fix **keeps it a
core builtin** and re-points *both* backends to the real engine:

- Re-point the LLVM path by upgrading `lib/quantum/quantum_rng.c` /
  `quantum_rng_wrapper.c` to source bytes from `moonlab_qrng_bytes` (or the
  `qrng_v3` engine from the quantum_rng repo), keeping the same `eshkol_qrng_*`
  extern seam so no codegen change is needed.
- Re-point the VM handler (`vm_native.c:12392-12420`) to call the same
  `eshkol_qrng_*` functions, **eliminating the VM-vs-LLVM divergence** (a
  C-backed builtin must behave identically on both paths).

Keeping it a core builtin (rather than moving it into an agent module) preserves
its require-free availability contract and its WASM reachability. See Section 6,
Stage S1, and Section 5.a for the WASM honesty label.

### 3.3 End-to-end wiring of a new C-backed builtin (reference trace)

To add a core builtin backed by C (the `quantum-random` re-point, and any core
`quantum-*` primitives that are not agent-gated):
1. Name->id registration: `lib/backend/eshkol_vm.c` `BUILTINS[]` (VM) and the
   corresponding native/LLVM table in `lib/backend/eshkol_compiler.c`
   (`language_surface.json._meta.sources`: `native_builtins` =
   `eshkol_compiler.c BUILTINS[]`, `vm_builtins` = `eshkol_vm.c BUILTINS[]`).
2. VM native handler: a `vm_*.c` (here `vm_native.c`) dispatch case.
3. AOT/JIT codegen: the `func_name` dispatch spine in the `*_codegen.cpp` family
   (`llvm-codegen` component in `.icc/architecture-model.yaml:39-57`).
4. Coverage: an entry in `tests/coverage/language_surface.json` (Section 5).

For the agent-module vehicle the touch set is instead: `lib/agent/quantum.esk`
(+ `provide`/`extern`), `lib/agent/c/agent_quantum.c` (shim),
`CMakeLists.txt` AGENT_FFI block (link libquantumsim), and coverage/ICC.

---

## 4. Flagship: the quantum-circuit -> Eshkol-AD bridge

**The single most important finding: Moonlab computes EXACT gradients, so this
bridge can honor Eshkol's AD honesty doctrine with no finite differences.**

### 4.1 What Moonlab provides

`vqe_compute_gradient(solver, params, grad)` (`vqe.h:509`, impl `vqe.c:1714`)
returns `dE/dtheta` for the whole parameter vector. Its dispatch is two exact
paths, no finite differences:

1. **Reverse-mode adjoint autograd (default fast path)** — `vqe_compute_gradient`
   tries `vqe_compute_gradient_adjoint` first (`vqe.c:1668`), conditioned on a
   hardware-efficient ansatz and noise-free simulation (`vqe.c:1672-1673`). It
   builds a `moonlab_diff_circuit_t` tape (`vqe_build_hea_diff_circuit`,
   `vqe.c:1628`), runs `moonlab_diff_forward`, then
   `moonlab_diff_backward_pauli_sum` (`vqe.c:1705`). Cost ~2 forward passes,
   independent of parameter count. The autograd engine is
   `src/algorithms/diff/differentiable.h` — the header states it is "the same
   algorithm used by PennyLane, Qiskit Aer, and JAX-Qsim for exact
   (non-stochastic) quantum gradients" (`differentiable.h:31-32`), with the VJP
   entry `moonlab_diff_backward_pauli_sum(c, forward_state, terms, num_terms,
   grad_out)` (`differentiable.h:216`) accumulating `d/dtheta` of
   `sum_k c_k <psi|P_k|psi>`.
2. **Parameter-shift rule (fallback)** — `vqe.c:1741-1768`,
   `grad[i] = (E(theta + (pi/2) e_i) - E(theta - (pi/2) e_i)) / 2` (documented at
   `vqe.h:501-502`). Also exact/analytic (comment `vqe.c:1762`: "Exact gradient
   via parameter shift"). Used for UCCSD / symmetry-preserving ansaetze and any
   noisy channel.

Both are exact. The loss itself is `vqe_compute_energy(solver, params)`
(`vqe.h:476`).

**Export gap to resolve:** neither `vqe_compute_gradient` nor any
`moonlab_diff_*` symbol is tagged `MOONLAB_API`, and `moonlab_export.h` does not
re-export them (grep of the stable header for grad/diff/backward yields nothing).
`vqe_compute_energy`/`vqe_solve`/`vqe_solver_create` **are** public. So the
gradient is reachable in a same-process static/shared link (symbols are not
hidden today) but is **not part of the frozen `dlsym` contract**. The clean fix
is a **maintainer-owned one-line change in Moonlab**: tag `vqe_compute_gradient`
(and ideally a thin `moonlab_vqe_gradient(...)` wrapper over
`moonlab_diff_backward_pauli_sum`) with `MOONLAB_API` and declare it in
`moonlab_export.h`. This is a cross-repo dependency the plan must call out
(Stage S3 depends on it).

### 4.2 Eshkol's AD machinery and the hook

Eshkol's AD is a compiler primitive: `lib/backend/autodiff_codegen.cpp` builds a
reverse-mode tape of `ad_node_t` nodes (`AutodiffCodegen::recordADNodeBinary`
`:1797`, `recordADNodeUnary` `:1917`, variable nodes `:2497`,
`recordADNodeTensor` `:2334`). The `ad_node_t` struct (`inc/eshkol/eshkol.h:971-1013`)
carries `saved_tensors`/`num_saved` fields (`:990-992`) for values the backward
pass needs, and `recordADNodeTensor` (`:2316-2334`) already records a node that
"does not compute a forward scalar" — the closest existing shape to a custom-VJP
node.

**But there is no hook today for an opaque external op to attach its own
backward, and the design must add one — this is the load-bearing gap.** The node
opcode space `ad_node_type_t` (`eshkol.h:853-956`) is a **closed enum**, and the
reverse pass dispatches through a **hard-coded switch over that enum** —
`get_tensor_backward_fn(int node_type)` in `lib/bridge/tensor_backward.cpp:638`
(`switch ((ad_node_type_t)node_type)`, `:639-673`) maps each fixed
`AD_NODE_TENSOR_*` opcode to a compiled-in `*_backward` function and its
`default:` case prints "no backward for AD_NODE type N — gradient signal lost"
and returns NULL. There is **no `AD_NODE_CUSTOM` opcode and no `backward_fn`/`vjp`
function-pointer field** on `ad_node_t` (the `params` union at `eshkol.h:995-1008`
holds only scalars). So Eshkol cannot, today, let an FFI op supply its own
gradient. The design therefore proposes a small, honest extension (below), not a
workaround.

### 4.3 Proposed architecture: Eshkol's reverse tape calls Moonlab's exact VJP

Architecture (cleanest, honesty-preserving):

- A `quantum.moonlab` builtin `qvqe-energy` evaluates the scalar loss
  `vqe_compute_energy(solver, params)` in the forward pass, where `params` is an
  Eshkol tensor/vector of the variational parameters.
- When that call occurs **inside an Eshkol AD context** (derivative/gradient
  codegen armed, `ESHKOL_DERIVATIVE_OP`), the forward emits a **custom AD tape
  node**. This requires the extension noted in 4.2: add an `AD_NODE_CUSTOM`
  opcode to `ad_node_type_t` and a `void (*backward)(ad_node_t*)` (plus an
  opaque saved-context pointer) field to `ad_node_t`, and route `AD_NODE_CUSTOM`
  through `get_tensor_backward_fn` (`tensor_backward.cpp:638`) to invoke that
  function pointer instead of the fixed table. The node is allocated through the
  same tape machinery as `recordADNodeTensor` and saves the `solver` handle and
  the forward `params` in `saved_tensors`/the new context field. Its backward,
  when the reverse sweep reaches it with incoming cotangent `g` (dLoss/dEnergy,
  a scalar), calls `vqe_compute_gradient(solver, params, grad_buf)` and scatters
  `g * grad_buf[i]` into the parameter gradient slots.
- This new `AD_NODE_CUSTOM` seam is **general** — it is the mechanism any future
  opaque-FFI differentiable op (not just Moonlab VQE) would use, so it is worth
  building cleanly rather than as a VQE special case.
- **Eshkol wraps Moonlab's own exact gradient** (adjoint or parameter-shift);
  Eshkol does not re-differentiate the circuit and does not fall back to finite
  differences. Moonlab returns exact; Eshkol multiplies by the upstream
  cotangent. This composes correctly with the rest of an Eshkol loss (a VQE
  energy feeding into a larger differentiable Eshkol expression differentiates
  end to end).

Why wrap rather than re-expose primitives: Moonlab already owns the physics
(generator structure `U_k = exp(-i theta_k G_k / 2)`, adjoint cotangent
propagation, Pauli-sum observables). Re-implementing that inside Eshkol's tape
would duplicate and risk drift. The custom-VJP node is the standard, minimal
seam.

### 4.4 Honesty constraint (hard requirement)

- The backward path calls **only** `vqe_compute_gradient` (exact adjoint /
  parameter-shift). It must **never** finite-difference `vqe_compute_energy` on
  the default path. This is Eshkol's stated doctrine, not a new rule:
  `docs/design/adr/0002-ad-alt-architect.md:24-25` — "The default SciML/PINN
  path must be exact AD or an explicit unsupported-op error. Hidden finite
  differences are disallowed."; `docs/design/adr/0002-ad-staged-dense-kernels.md:236-239`
  — derivative operators are "exact AD or an explicit unsupported error ...
  explicitly named numeric APIs (e.g. `finite-difference-gradient`), never as a
  hidden fallback", enforced by a `finite_difference_evals` counter
  (`lib/core/runtime_autodiff.cpp:40,47,52`) that tests assert is `0` on every
  path claimed exact. It is machine-checked by
  `.icc/architecture-model.yaml` `INV-ad-exact-no-finite-differences`
  (`:347-357`) and the generative AD-vs-FD oracle `ad_adversarial_fd_oracle`
  (`.icc/completion-oracles.yaml:113-116`).
- If a Moonlab configuration cannot yield an exact gradient (e.g. a gradient of a
  measurement-sampled quantity with no analytic form, or an ansatz Moonlab's
  gradient does not support), the builtin must raise an **explicit Eshkol error**
  ("quantum gradient unavailable for this configuration"), not silently
  substitute a numerical difference. Exact gradient or explicit error — the same
  contract Eshkol's own AD holds.
- Extend the AD-exact invariant so its site set includes the quantum custom-VJP
  node (Section 5c): the quantum backward must reference the Moonlab exact-gradient
  symbol and must **not** reference a finite-difference helper.

---

## 5. Honesty and coverage requirements

### 5.a Browser / WASM entropy must be labeled degraded, never silently "quantum"

The Moonlab JS binding is the `moonlab_export_lean.c` half compiled to WASM and
**deliberately excludes qrng/hardware-entropy** (file header of
`moonlab_qrng_export.c:5-9`; the lean export omits qrng). The browser has no
`RDSEED` / `/dev/urandom` / `SecRandomCopyBytes`, so `moonlab_qrng_bytes` cannot
run there. Eshkol already ships a WASM path (`.icc/architecture-model.yaml`
`wasm-path` component `:120-135`) with per-symbol JS glue and a known live drift
of `eshkol_qrng_*` glue between the two glue files (`INV-wasm-import-glue-equality`
note, `:300-311`).

Requirement: on the WASM/browser build, `quantum-random` must be **explicitly
labeled degraded** — e.g. resolve to a `crypto.getRandomValues`-seeded CSPRNG and
have the runtime/report state clearly that it is a classical CSPRNG fallback, not
Bell-verified quantum entropy. A predicate builtin (proposed `quantum-random-source`
returning a symbol like `'moonlab-qrng` / `'csprng-fallback` / `'insecure-prng`)
lets Scheme code and the acceptance gate detect the active source. Silent
"quantum" labeling in the browser is the exact honesty failure this integration
exists to remove; it must not be reintroduced at the WASM boundary.

### 5.b Coverage: the new builtins enter `language_surface.json` and get exercised

`tests/coverage/language_surface.json` is the ground-truth surface (958
builtins). The three existing quantum entries are at `:7318-7353`, each with
`name`, `ids`, `arity`, `backends` (`["vm","native_llvm"]`), `category`. Every
new `quantum-*` builtin (and any re-categorization of the existing three from
`ffi_system` to a new `quantum` category) must be added here with the same
schema, and the `counts`/`builtins_by_category` totals updated. The exposure
engines (`.icc` `exposure-engines` component `:107-118`;
`scripts/language_coverage.py`) diff what is exercised against this file and emit
a `language_surface_coverage` runtime_event; `INV-language-surface-exercise`
(`:280-298`) fails when the surface is not freshly exercised. So the new builtins
must be reachable by the generative corpus, not merely listed.

### 5.c New ICC architecture-model component + invariant

Add to `.icc/architecture-model.yaml`:

- A **`quantum-backend` component** (role `backend`) whose modules are the
  quantum shim + module: `lib/agent/c/agent_quantum.c`, `lib/agent/quantum.esk`,
  and the touched `vm_native.c` qrng handlers, with entry point the shim. This
  makes the quantum path a first-class, machine-checked part of the architecture
  model (the model is verified by `icc architecture-verify`, `:14-16`).
- A **`INV-quantum-random-routes-to-qrng` invariant** modeled on
  `INV-ffi-registration-completeness` (`:259-278`) and
  `INV-ad-exact-no-finite-differences` (`:347-357`) — a `dependency-presence`
  invariant, `fidelity: grep`, `severity: critical`. Its sites: the
  `quantum-random` native handler in `vm_native.c` must reference the Moonlab /
  quantum_rng entry (`moonlab_qrng_bytes` or `qrng_*`) and must **not** reference
  the bare `vm_qrng_next_u64` / xorshift path once the fix lands. This is the
  machine-checked encoding of "quantum-random must route to real quantum entropy,
  not a PRNG" and it makes any future regression to a placeholder fail ICC.
- A **`INV-quantum-ad-exact` invariant** (or an added site on the existing
  AD-exact invariant): the quantum custom-VJP backward references the Moonlab
  exact-gradient symbol, not a finite-difference helper (Section 4.4).
- Wire these into the release oracle so `icc readiness` counts them (per the
  standing rule that every new pillar is wired into the readiness oracle).

### 5.d Bell-test / NIST-health acceptance gate

Moonlab ships the primitives for a real gate:

- **Bell/CHSH**: `bell_test_run_chsh(config, results)`
  (`src/applications/bell_test.h:141`) fills `results.S`, the CHSH value
  `S = E(a1,b1)+E(a1,b2)+E(a2,b1)-E(a2,b2)` (`bell_test.h:77-78`);
  `bell_test_default_config()` `:132`. The QRNG's own verifier
  `qrng_v3_verify_quantum` (`qrng.h:404`) is a second source. Bell-inequality
  violation for the entangled path is `S > 2` (classical bound), with the
  documented device figure CHSH S = 2.83-2.90.
- **NIST SP800-90B health**: `health_tests_startup` (`health_tests.h:154`),
  continuous `health_test_rct` (repetition-count, `:197`) and `health_test_apt`
  (adaptive-proportion, `:212`).

Proposed gate `quantum_entropy_health_oracle` in
`.icc/completion-oracles.yaml`, modeled on the existing `ad_adversarial_fd_oracle`
criterion shape (`:113-116`, an `event_names` + `label` criterion backed by a
runtime_event): a test drives Eshkol's `quantum-random` through Moonlab, samples
a byte stream, and asserts (1) the source is `moonlab-qrng` (not fallback) on a
native build, (2) SP800-90B RCT/APT pass on the stream, and (3) a CHSH run
reports `S > 2` (entanglement present). Emit `quantum_entropy_health_oracle` as a
`kind: runtime_event`. On the WASM build the same gate asserts the source is
**explicitly** the labeled CSPRNG fallback (5.a) — so "degraded but honest"
passes and "silently quantum" fails.

---

## 6. Staged, shippable plan

Each stage is independently shippable, CI-gated, and small. Files listed are the
Eshkol files touched (Moonlab/quantum_rng are read/linked, not modified, except
the one flagged Moonlab export change in S3).

### S1 — Thin FFI binding + honest `quantum-random`
- **Scope**: link libquantumsim (or quantum_rng for the QRNG-only subset);
  bind core state-vector create/destroy, the common gates, measurement, and
  expectation; **re-point `quantum-random`/`-int`/`-range` to `moonlab_qrng_bytes`**.
  Add `quantum-random-source` predicate (5.a). Establish
  `ESHKOL_ENABLE_MOONLAB` build gate with graceful fallback (mirroring
  `QSIM_ENABLE_ESHKOL`).
- **Eshkol files** (honest `quantum-random`, core-builtin path — works under
  WASM): `lib/quantum/quantum_rng.c` + `quantum_rng_wrapper.c` (re-point the
  `eshkol_qrng_*` implementation to `moonlab_qrng_bytes` / `qrng_v3`;
  `wrapper.c:31-57`), `lib/backend/vm_native.c` (replace the xorshift
  `vm_qrng_*` bodies `:4299-4334, 12392-12420` with calls to the same
  `eshkol_qrng_*` — kills the VM-vs-LLVM divergence), `lib/backend/eshkol_vm.c`
  (add `quantum-random-source` predicate), `CMakeLists.txt` (link libquantumsim
  or vendored qrng_v3 into the core runtime archive at `:1608`),
  `tests/coverage/language_surface.json`.
- **Eshkol files** (rich quantum surface, agent.X path): `lib/agent/quantum.esk`
  (new), `lib/agent/c/agent_quantum.c` (new shim), `CMakeLists.txt` AGENT_FFI
  block (`:3117-3296`, add libquantumsim discovery + `ESHKOL_HOST_AGENT_FFI_LINK_ARGS`
  propagation), and the JIT weak-symbol table `lib/repl/repl_jit.cpp`
  (`ADD_OPTIONAL_AGENT_FFI_SYMBOL`, `:781-789`) so the REPL resolves the new
  `moonlab_*` symbols.
- **Acceptance**: `quantum-random` returns bytes sourced from `moonlab_qrng_bytes`
  on a native build (assert via `quantum-random-source` = `moonlab-qrng`), and
  the **VM and AOT/JIT backends agree** on the source; a unit test builds a Bell
  pair and measures the expected correlation; existing language-surface coverage
  stays green.
- **Dependencies**: libquantumsim (or vendored `qrng_v3`) present at build (else
  fallback path). No Moonlab changes.
- **Risk**: low-medium. Two sub-problems: (a) the core-builtin re-point must
  keep the `eshkol_qrng_*` extern seam so no codegen changes; (b) the agent.X
  surface must honor force-load discipline (`CMakeLists.txt:3286-3291`) and the
  JIT weak-symbol table (`repl_jit.cpp:761-789`) so `dlsym` finds symbols. The
  dense state-vector API is the non-frozen ABI tier — pin a Moonlab version.

### S2 — VQE / algorithms builtins (forward only)
- **Scope**: bind `pauli_hamiltonian_*`, the prebuilt molecular Hamiltonians,
  ansatz create/apply, `vqe_solver_*`, `vqe_solve`, `vqe_compute_energy`,
  `vqe_exact_ground_state_energy`, and DMRG scalar drivers. Forward evaluation
  only (no AD yet).
- **Eshkol files**: `lib/agent/quantum.esk` (+externs), `lib/agent/c/agent_quantum.c`
  (+VQE wrappers), `tests/coverage/language_surface.json`.
- **Acceptance**: an Eshkol VQE example reproduces `vqe_exact_ground_state_energy`
  for H2/LiH within tolerance; DMRG TFIM energy matches Moonlab's own value.
- **Dependencies**: S1.
- **Risk**: medium. Opaque handle lifetime management across Scheme GC — the
  `with-*` RAII wrapper idiom (`lib/agent/sqlite.esk` `with-db`) must own free.

### S3 — The AD bridge (flagship)
- **Scope**: custom-VJP tape node so `qvqe-energy` differentiates through
  Moonlab's exact gradient (Section 4). Explicit-error path when exact gradient
  is unavailable.
- **Eshkol files**: `inc/eshkol/eshkol.h` (add `AD_NODE_CUSTOM` to
  `ad_node_type_t` `:853-956` and a `backward`/saved-context field to
  `ad_node_t` `:971-1013`), `lib/bridge/tensor_backward.cpp` (route
  `AD_NODE_CUSTOM` through `get_tensor_backward_fn` `:638`),
  `lib/backend/autodiff_codegen.cpp` (emit the custom node alongside
  `recordADNodeTensor` `:2334`), the quantum shim (expose
  `vqe_compute_gradient`), `lib/agent/quantum.esk`,
  `tests/coverage/language_surface.json`, `.icc/architecture-model.yaml`
  (`INV-quantum-ad-exact`).
- **Cross-repo dependency (maintainer call)**: tag `vqe_compute_gradient` (and
  ideally add a `moonlab_vqe_gradient` wrapper) with `MOONLAB_API` and declare
  it in `moonlab_export.h`, so the gradient is part of the frozen contract, not
  an incidentally-visible symbol.
- **Acceptance**: `(gradient (lambda (theta) (qvqe-energy solver theta)) theta0)`
  matches Moonlab's `vqe_compute_gradient` exactly (bit-for-bit up to fp
  reassociation), and matches a central finite difference of `vqe_compute_energy`
  to tolerance in a **test-only** cross-check (never on the default path); a
  composed loss (VQE energy feeding a larger Eshkol expression) differentiates
  end to end; the unsupported-config case raises the explicit error.
- **Dependencies**: S2 + the Moonlab export tag.
- **Risk**: high. This is the novel seam. The AD-exact invariant must be
  extended (5.c) so the honesty contract is machine-checked.

### S4 — PQC / ML-KEM builtins
- **Scope**: bind `moonlab_mlkem{512,768,1024}_keygen_qrng/_encaps_qrng/_decaps`
  and size macros as `quantum-kem-*` builtins (bytevector in/out).
- **Eshkol files**: `lib/agent/quantum.esk` (or a sibling `quantum-crypto.esk`),
  the shim, `tests/coverage/language_surface.json`.
- **Acceptance**: FIPS 203 KAT round-trip (keygen -> encaps -> decaps yields the
  same shared secret); ML-KEM-768 default.
- **Dependencies**: S1 (link + QRNG entropy).
- **Risk**: low-medium. These are stable-ABI symbols; main care is buffer sizing.

### S5 — Coverage + arch-model + Bell/health gate
- **Scope**: land the `quantum-backend` component, `INV-quantum-random-routes-to-qrng`,
  the extended AD-exact invariant, the `quantum_entropy_health_oracle` gate, and
  full language_surface entries + exposure-engine reachability; wire all into the
  readiness oracle.
- **Eshkol files**: `.icc/architecture-model.yaml`, `.icc/completion-oracles.yaml`,
  `tests/coverage/language_surface.json`, `scripts/language_coverage.py` /
  generative corpus, a new acceptance test under `tests/`.
- **Acceptance**: `icc architecture-verify` passes with the new component and
  invariants; the Bell/health gate emits `quantum_entropy_health_oracle` (S>2,
  RCT/APT pass, source=`moonlab-qrng` native / labeled-fallback WASM); `icc
  readiness` counts the quantum pillar.
- **Dependencies**: S1-S4 (gate exercises whatever is bound).
- **Risk**: low. Mostly declarative; the value is that regressions to a
  placeholder now fail ICC.

### Licensing / vendoring — maintainer call
Both Moonlab and quantum_rng are MIT; Eshkol may **link** (dynamic, discovered
via pkg-config/`find_library` like sqlite3/libcurl) or **vendor** libmoonlab
sources. Recommendation for review, not decision:
- **Link (default)** matches the existing agent-FFI optional-dependency model
  (`CMakeLists.txt:3180-3193`), keeps Eshkol's build light, and lets Moonlab
  version independently. Risk: the non-frozen state-vector/gate ABI tier (1.1)
  requires a pinned Moonlab version.
- **Vendor** guarantees ABI stability and a hermetic build (helps WASM/S5 and
  CI), at the cost of carrying and updating Moonlab sources. MIT permits either;
  attribution must be preserved.
This is flagged as a maintainer decision, not resolved here.

---

## 7. Summary of go/no-go inputs

- Moonlab exposes a broad, mostly-stable C ABI (Section 1); the QRNG entry
  `moonlab_qrng_bytes` directly and honestly replaces the xorshift placeholder.
- The FFI vehicle is well-trodden (agent.X + core-builtin re-point, Section 3);
  the link path already supports optional vendor libraries gated by `(require)`.
- The flagship AD bridge is feasible **and honest**: Moonlab computes exact
  gradients (adjoint + parameter-shift, Section 4.1), so Eshkol's reverse tape
  can wrap them with no finite differences — pending one maintainer-owned Moonlab
  export tag.
- Honesty is enforceable, not aspirational: a machine-checked ICC invariant binds
  `quantum-random` to real entropy, the WASM path is explicitly labeled degraded,
  and a Bell/NIST gate backs the claim (Section 5).
- The work decomposes into five small, CI-gated stages (Section 6), each with a
  concrete acceptance test.

The open decisions for the maintainer: (1) link vs vendor libmoonlab; (2)
approve the one-line Moonlab export-tag change that S3 depends on; (3) scope —
how far down S2-S4 to go in the first release.
