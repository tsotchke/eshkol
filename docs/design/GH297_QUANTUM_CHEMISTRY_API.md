# GH #297: Differentiable Quantum-Chemistry API Decision

Status: implemented and runtime-gated

## Decision

All five examples proposed in `RichardHoekstra/eshkol@b76cc4f` belong in
`examples/`. The dependency-free H2 Taylor-tower example is available in every
build. The four Moonlab examples remain opt-in under
`ESHKOL_QUANTUM_ENABLED=ON`.

The supported Eshkol API is:

```scheme
(make-pauli-hamiltonian coefficients operators
                        [nuclear-repulsion [hf-reference]])
(vqe-qgt hamiltonian parameters)
```

The public constructor is deliberately not `make-pauli5-hamiltonian`. It takes
equal-length vectors of coefficients and Pauli strings, so it works beyond the
five-term H2 reduction. Optional molecular metadata supplies the scalar nuclear
repulsion and the Hartree-Fock occupation bitstring used by the ansatz.

## Ownership boundary

Moonlab owns:

- Pauli Hamiltonian storage and term validation;
- the smooth H2 potential-energy surface;
- `vqe_compute_qgt` and its row-major Fubini-Study metric;
- the production QNG optimizer and optimizer policy.

Eshkol owns:

- opaque integer handles and deterministic cleanup;
- validation and conversion between Scheme vectors and the Moonlab ABI;
- the thin expert-level `vqe-qgt` matrix accessor;
- differentiable composition around Moonlab calls;
- examples that expose the underlying mathematics.

Eshkol does not reimplement QGT and does not claim the Scheme linear solve in
`qng_vqe.esk` as the production optimizer. That solve is an inspectable example;
Moonlab's QNG implementation remains the operational owner.

## ABI and dependency pin

Eshkol pins `tsotchke/moonlab` v1.2.0 at
`e441957b22698ce93ad4868d585ac7ca3baa281f`. ICC verification established:

- `MOONLAB_API int vqe_compute_qgt(vqe_solver_t*, const double*, double*)`;
- return value `0` on success and `-1` on error;
- a symmetric row-major `n x n` matrix, where `n` is the ansatz parameter count;
- the QGT source is part of `libquantumsim` and the macOS build links cleanly;
- the smooth H2 PES is in the same v1.2.0 lineage.

The previous `d2503460` pin did not export `vqe_compute_qgt` and therefore could
not support the accepted accessor.

## Runtime design

The Pauli constructor crosses the FFI in two phases: allocate a Hamiltonian with
known qubit/term counts, then add each validated term. If a term fails, Scheme
destroys the partial handle before raising.

QGT uses a scoped native context parallel to the existing gradient context. Each
context owns its solver, ansatz, optimizer, parameter buffer, and matrix. This
avoids the contributor prototype's process-global `double[64]`, permits nested
or concurrent callers, and discovers the actual parameter count instead of
hard-coding four parameters.

Default-off builds compile matching stubs for every symbol. They return the
existing explicit unavailability error rather than leaving unresolved externs
or fabricating zero matrices.

## Numerical gate

`scripts/run_quantum_chemistry_examples_gate.sh` verifies:

1. dependency-free H2 Taylor-tower output in both JIT and AOT lanes;
2. generic Pauli construction against a known one-qubit ground energy;
3. QGT dimensions, symmetry, and nonnegative diagonal;
4. differentiable H2 VQE energy;
5. smooth-PES and full-response vibrational frequencies;
6. QNG convergence against the Moonlab exact H2 energy.

The full response Hessian contains one gauge-redundant HEA direction. Its mixed
response is also exactly zero, so the example applies a `1e-10` Tikhonov shift
to select the zero solution in that null direction. The physical response and
the reported 5003.19 cm^-1 frequency are stable at displayed precision.
