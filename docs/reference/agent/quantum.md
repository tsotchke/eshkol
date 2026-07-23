# `agent.quantum` — Moonlab Quantum Simulation, CHSH, and VQE

Bindings over the Moonlab state-vector simulator: qubit states, gates,
projective measurement, a CHSH Bell experiment, molecular Hamiltonians, and a
Variational Quantum Eigensolver (VQE) whose energy is differentiable through
Eshkol's reverse-mode AD. Quantum states and Hamiltonians are opaque `i64`
handles (like `agent.sqlite` connections).

```scheme
(require agent.quantum)
```

Source: `lib/agent/quantum.esk`. C shim: `lib/agent/c/agent_quantum.c`
(symbols `eshkol_quantum_*`, `eshkol_vqe_*`), which links Moonlab's
`libquantumsim`.

## Availability — opt-in build flag

This module is only usable when Eshkol is built with
`-DESHKOL_QUANTUM_ENABLED=ON`, which FetchContents the Moonlab simulator
(see `docs/design/MOONLAB_INTEGRATION.md`). Without the flag the shim is not
compiled and `(require agent.quantum)` fails to resolve its externs — the same
graceful-unavailability contract `agent.http` uses for its optional libcurl
symbols. Nothing in this module pretends to work on a non-quantum build.

The `quantum-random` builtin family (below) is the exception: it exists in
**all** builds, with an honestly labeled source.

## Error contract

Every FFI failure raises a **catchable Eshkol error** carrying Moonlab's own
message (retrieved via `eshkol_quantum_last_error`), never a bogus value.
C APIs with only a `double` return channel signal failure with NaN; the
wrappers detect it and raise. State construction, `bell-chsh`, and Hamiltonian
construction are capability-gated on `ffi` (see [capabilities](capabilities.md)).

## State lifecycle

| Procedure | Signature | Notes |
|-----------|-----------|-------|
| `make-quantum-state` | `(make-quantum-state n)` | n-qubit state initialized to \|0…0⟩; returns an opaque handle; raises on invalid `n` or allocation failure |
| `quantum-state-destroy!` | `(quantum-state-destroy! st)` | Release Moonlab-side resources; double-destroy / invalid handle is a documented C-side no-op |
| `quantum-state-num-qubits` | `(quantum-state-num-qubits st)` | Number of qubits `st` was created with |
| `with-quantum-state` | `(with-quantum-state n fn)` | Allocates, calls `(fn st)`, destroys on every exit path via `dynamic-wind` (mirrors `with-db`) |

## Gates

Each gate applies in place and returns `st` for chaining. A failed gate
(bad qubit index, invalid handle) raises.

| Procedure | Signature |
|-----------|-----------|
| `apply-hadamard` | `(apply-hadamard st qubit)` |
| `apply-pauli-x` | `(apply-pauli-x st qubit)` — Pauli-X (NOT) |
| `apply-pauli-y` | `(apply-pauli-y st qubit)` |
| `apply-pauli-z` | `(apply-pauli-z st qubit)` |
| `apply-cnot` | `(apply-cnot st control target)` — flip `target` iff `control` is \|1⟩ |
| `apply-rx` | `(apply-rx st qubit theta)` — RX(theta) rotation |
| `apply-ry` | `(apply-ry st qubit theta)` |
| `apply-rz` | `(apply-rz st qubit theta)` |

## Measurement

| Procedure | Signature | Returns |
|-----------|-----------|---------|
| `measure` | `(measure st qubit)` | 0 or 1; projective Z-basis measurement, collapses the state. Sampled from Moonlab's own hardware-backed entropy context, not an Eshkol-controlled seed |
| `expectation-z` | `(expectation-z st qubit)` | ⟨psi\|Z\|psi⟩ in [-1, 1], without collapsing |
| `bell-chsh` | `(bell-chsh num-trials)` | Signed CHSH S value from Moonlab's Bell experiment at the canonical maximum-violation settings (Alice 0, pi/2; Bob pi/4, -pi/4). `num-trials` must be at least 4 |

### The CHSH acceptance gate

A local hidden-variable theory satisfies |S| <= 2; the ideal Bell state gives
S = 2 sqrt(2) ~= 2.828. The quantum test suite
(`tests/quantum/bell_chsh_test.esk`) gates on `2.4 < S <= 2.95` over 16000
shots — the lower bound proves a genuine quantum violation, the upper bound
catches a malformed normalization.

### Verified example (quantum-enabled build)

From `tests/quantum/quantum_smoke_test.esk` — a Bell pair's two qubits must
always measure equal (200/200 on the S1 acceptance run):

```scheme
(require agent.quantum)
(define (bell-correlated?)
  (with-quantum-state 2
    (lambda (st)
      (apply-hadamard st 0)
      (apply-cnot st 0 1)
      (= (measure st 0) (measure st 1)))))
```

## VQE — molecular Hamiltonians

Hamiltonians are opaque handles constructed from Moonlab's built-in molecular
models and released with `hamiltonian-destroy!` (or scoped with
`with-hamiltonian`).

| Procedure | Signature | Notes |
|-----------|-----------|-------|
| `make-h2-hamiltonian` | `(make-h2-hamiltonian bond-distance)` | H2/STO-3G Pauli Hamiltonian at `bond-distance` Angstroms |
| `make-lih-hamiltonian` | `(make-lih-hamiltonian bond-distance)` | LiH/6-31G |
| `make-h2o-hamiltonian` | `(make-h2o-hamiltonian)` | Moonlab's fixed-geometry H2O/STO-3G |
| `hamiltonian-destroy!` | `(hamiltonian-destroy! ham)` | Invalid/released handle is a C-side no-op |
| `with-hamiltonian` | `(with-hamiltonian ham fn)` | Calls `(fn ham)`, releases `ham` on every exit path |
| `hamiltonian-exact-ground-energy` | `(hamiltonian-exact-ground-energy ham)` | Direct-diagonalization ground-state reference energy in Hartree (including nuclear repulsion); intended for small Hamiltonians |
| `vqe-optimize` | `(vqe-optimize ham [iterations])` | Runs ideal Moonlab VQE, returns the optimized variational energy in Hartree. Default: one-layer hardware-efficient ansatz, ADAM, 500 iterations; `iterations` must be positive |
| `vqe-energy` | `(vqe-energy ham params)` | Exact ideal-state energy E(params) of the default one-layer hardware-efficient ansatz. **Differentiable** (see below) |
| `vqe-gradient` | `(vqe-gradient ham params)` | Moonlab's exact dE/dtheta vector. `params` must be a vector with exactly `2*nqubits` entries; wrong length raises. Reverse-mode adjoint differentiation, no finite differences |

> **Moonlab v1.2.0 (pinned as of v1.3.4-evolve).** The pinned backend adds the
> quantum geometric tensor (`vqe_compute_qgt` in the Moonlab API), which supplies
> the Fubini-Study metric behind **quantum-natural-gradient** optimization, and a
> **smooth first-principles H2/LiH potential-energy surface** in place of the
> earlier tabulated surface. With the smooth PES, the H2 equilibrium reference at
> bond distance 0.735 Å is `-1.142200155381` Ha (a `-2.95e-5` Ha shift from the
> earlier surface). See [MOONLAB_INTEGRATION](../../design/MOONLAB_INTEGRATION.md).

## `vqe-energy` is differentiable through Eshkol AD

`vqe-energy` delegates to the `vqe-energy-primitive` compiler builtin rather
than an ordinary FFI extern. In normal execution it calls Moonlab's
fixed-parameter energy evaluator. Inside a reverse-mode AD context
(`gradient`), the primitive records an `AD_NODE_CUSTOM` tape node carrying
Moonlab's **exact adjoint** vector-Jacobian product, so the opaque quantum
circuit composes with surrounding differentiable Scheme arithmetic through the
ordinary chain rule (see [AD architecture](../ad/architecture.md)).

Verified by `tests/quantum/vqe_ad_test.esk`: the Eshkol `gradient` of
`vqe-energy` matches `vqe-gradient` (Moonlab's native adjoint) exactly, matches
a central finite-difference oracle to ~1e-11, and composes through the chain
rule (differentiating `E(p)^2` yields `2E * dE/dp`).

```scheme
(require agent.quantum)
(with-hamiltonian
 (make-h2-hamiltonian 0.735)
 (lambda (ham)
   (let* ((params (vector 0.1 0.2 0.3 0.4))
          ;; Eshkol reverse-mode AD through the quantum circuit:
          (g (gradient (lambda (p) (vqe-energy ham p)) params))
          ;; ...agrees with Moonlab's native adjoint gradient:
          (g-moonlab (vqe-gradient ham params))
          ;; ...and composes: d/dp [E(p)^2] = 2 E(p) dE/dp
          (g-sq (gradient (lambda (p)
                            (let ((e (vqe-energy ham p))) (* e e)))
                          params)))
     (display g) (newline))))
```

## `quantum-random` — honest quantum randomness (all builds)

These are core compiler builtins, not part of the `agent.quantum` module, and
they work in **every** build. Both backends (LLVM AOT/JIT and the bytecode VM)
route through the single `eshkol_qrng_*` wrapper
(`lib/quantum/quantum_rng_wrapper.c`), so they agree on both the numbers and
the entropy source.

| Builtin | Signature | Returns |
|---------|-----------|---------|
| `quantum-random` | `(quantum-random)` | double in [0, 1) |
| `quantum-random-int` | `(quantum-random-int bound)` | integer in [0, bound) |
| `quantum-random-range` | `(quantum-random-range min max)` | integer in [min, max] (inclusive) for integer arguments |

The source is honestly labeled per build configuration:

- **Quantum-enabled build** (`-DESHKOL_QUANTUM_ENABLED=ON`): draws from
  Moonlab's Bell-verified QRNG (`moonlab_qrng_bytes`), which combines a
  hardware entropy pool with a Bell-verified quantum simulation layer. On the
  rare documented failure of that call, the wrapper falls back to the classical
  generator for that one draw and prints a one-time stderr diagnostic — never a
  silent source switch.
- **Default build**: a classical software PRNG — not quantum hardware, not
  Bell-verified, and not claimed to be.

The C accessor `eshkol_qrng_source_label()` reports which configuration is
active: `"moonlab-qrng"` or `"classical-fallback"`.

```scheme
(display (quantum-random)) (newline)
(display (quantum-random-int 100)) (newline)
(display (quantum-random-range 10 20)) (newline)
```
```
0.459341
95
10
```

## See also

- [`agent.pqc`](pqc.md) — ML-KEM post-quantum KEM over the same Moonlab link
  target.
- [Capabilities](capabilities.md) — the `ffi` capability gates state,
  Hamiltonian, and CHSH construction.
- [FFI & AOT linking](ffi.md) — how the externs resolve under `-r` vs AOT.
