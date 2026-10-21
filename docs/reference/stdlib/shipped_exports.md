# Additional shipped Eshkol exports

This page closes the DD-11 reference gap for shipped modules whose exports were
not present in the standard-library index. Signatures and module locations are
taken from the source files named in each section. The anchors are checked by
`scripts/check_public_api_docs.py`.

## `core.ad.algebraic_qp`

Source: `lib/core/ad/algebraic_qp.esk`. The exact convexity, certificate,
active-set and pullback contracts are documented in
[ad_algebraic_qp.md](ad_algebraic_qp.md).

### `algebraic-qp-solve`

Solves a bounded exact strictly convex quadratic program by deterministic
inequality active-set enumeration and KKT certification.

### `algebraic-qp-solution`

Returns owned primal values, equality multipliers, inequality multipliers and
objective value from a certified result.

### `algebraic-qp-info`

Returns active-set attempt count, copied active inequality indices, strict
pullback-admission status and total factorization attempts.

### `algebraic-qp-verify`

Independently checks a returned certificate against the caller's QP.

### `algebraic-qp-pullback`

Computes exact cotangents for the Hessian, linear term and equality and
inequality data on an admitted strict active branch.
## `core.ad.algebraic_spectral`

Source: `lib/core/ad/algebraic_spectral.esk`. Exact simple-branch admission,
projector residuals, pullback convention, and refusal boundaries are in
[ad_algebraic_spectral.md](ad_algebraic_spectral.md).

### `algebraic-characteristic`

Returns ascending coefficients of the exact characteristic polynomial after checking the Cayley–Hamilton residual.

### `algebraic-eigenvalue-with-pullback`

Certifies a supplied simple eigenvalue of a symmetric matrix and returns its rank-one projector with the unconstrained matrix pullback.

### `algebraic-eigenvalue-lift`

Lifts a supplied enclosing root through an AD context while rechecking the exact simple-branch certificate.

## `core.ad.algebraic_sparse`

Source: `lib/core/ad/algebraic_sparse.esk`. The exact sparse-curvature contract,
structural-pattern obligation, and probe summary are in
[ad_algebraic_sparse.md](ad_algebraic_sparse.md).

### `algebraic-sparse-hessian`

Recovers an exact sparse Hessian from an owner domain, objective, point, and explicit structural pattern, optionally using a whole-column HVP callback.

### `algebraic-sparse-hessian-ref`

Returns an owned exact Hessian coefficient at the requested indices.

### `algebraic-sparse-hessian-dense`

Returns an owned dense view of an exact sparse Hessian.

### `algebraic-sparse-hessian-summary`

Returns dimension, color, actual probe-call, upper-entry, and recovery-mode counts.

## `core.ad.algebraic_subspace`

Source: `lib/core/ad/algebraic_subspace.esk`. Exact admission, spectral separation,
JVP/VJP conventions, ownership and refusal boundaries are documented in
[ad_algebraic_subspace.md](ad_algebraic_subspace.md).

### `make-algebraic-subspace`

Admits an exact symmetric matrix and nontrivial invariant orthogonal projector
whose selected and complementary spectra are disjoint.

### `algebraic-subspace-projector`

Returns an owned copy of the admitted projector.

### `algebraic-subspace-jvp`

Returns the exact derivative of the invariant projector in a symmetric matrix direction.

### `algebraic-subspace-vjp`

Returns the symmetric-parameterization cotangent for a projector seed.

### `algebraic-subspace-info`

Returns matrix size, selected rank, linear-system size and coefficient depth.

### `verify-algebraic-subspace-jvp`

Re-admits the source problem and checks the defining linearized tangent equations.

### `make-algebraic-subspace-from-factors`

Builds the invariant projector from a caller-supplied coprime monic rational split of the full characteristic polynomial, then reuses exact projector admission and separation checks.

## `core.ad.sparsity`

Source: `lib/core/ad/sparsity.esk`. The coloring contract and numeric/exact
shared ownership are in [ad_sparsity.md](ad_sparsity.md).

### `hessian-color-graph`

Returns deterministic distance-two colors and their count for a validated undirected adjacency vector.

## `core.ad.algebraic_implicit`

Source: `lib/core/ad/algebraic_implicit.esk`. The full contract and bounded
polynomial syntax are in [ad_algebraic_implicit.md](ad_algebraic_implicit.md).

### `algebraic-ad-system-root`

Lifts a supplied simple branch of a bounded coupled polynomial system through
an exact AD context using its formally derived primal Jacobian.

### `algebraic-ad-system-residual`

Returns the full enclosing-domain residual vector for a bounded polynomial
system at supplied values.

## `agent.http`

Source: `lib/agent/http.esk`. The complete transport reference is in
`docs/reference/agent/http.md`; these entries are also reachable from the
standard-library index.

### `http-stream-error`

Returns the last transport or parser error for a stream, or an empty value when
no error has been recorded.

### `sse-event-id`

Returns the event identifier from an SSE event.

### `sse-event-retry-ms`

Returns the server-requested SSE retry delay in milliseconds.

## `agent.quantum`

Source: `lib/agent/quantum.esk`. These procedures are available only in a
quantum-enabled build; the lifecycle and error contract are in
`docs/reference/agent/quantum.md`.

### `make-pauli-hamiltonian`

Creates an opaque Hamiltonian handle from a Pauli-term count, qubit count, and
reference energy. Add terms with the documented Hamiltonian operations before
passing the handle to VQE procedures.

## `core.ad.interval`

Source: `lib/core/ad/interval.esk`. An interval is represented as a `(lo . hi)`
pair. Primitive operations widen ordinary floating-point results by the
module's conservative padding constants; division and logarithm reject invalid
domains.

### `interval-add`

Adds two intervals and returns an outward-widened enclosure.

### `interval-cos`

Returns an outward-widened enclosure of cosine over an interval.

### `interval-div`

Divides two intervals; raises an error when the denominator contains zero.

### `interval-exp`

Returns an outward-widened enclosure of exponential over an interval.

### `interval-log`

Returns an outward-widened enclosure of logarithm; raises when the interval
reaches zero or a negative value.

### `interval-mid`

Returns the midpoint of an interval.

### `interval-mul`

Multiplies two intervals and returns an outward-widened enclosure.

### `interval-neg`

Negates an interval by exchanging and negating its endpoints.

### `interval-pow`

Raises an interval to an integer power using the module's interval arithmetic.

### `interval-sin`

Returns an outward-widened enclosure of sine over an interval.

### `interval-sub`

Subtracts the second interval from the first and widens the enclosure.

### `interval-union`

Returns the smallest interval containing both input intervals.

### `interval-widen`

Expands an interval by an explicit epsilon on both sides.

### `interval?`

Reports whether a value has the interval pair representation.

### `iv-abs-pad`

Returns the absolute padding floor used by interval construction.

### `iv-eps`

Returns the machine-epsilon constant used by the interval padding policy.

### `iv-rel-pad`

Returns the relative padding factor used by algebraic interval operations.

## `core.ad.taylor_models`

Source: `lib/core/ad/taylor_models.esk`. These exports expose tunables for the
dense-sampling remainder estimate used by the Taylor-model reference.

### `taylor-model?`

Reports whether a value has the Taylor-model vector representation.

### `tm-nsamp`

Returns the number of sample points used to estimate the derivative bound.

### `tm-safety`

Returns the safety factor applied to the sampled derivative maximum.

## `core.ad.tensor_tower`

Source: `lib/core/ad/tensor_tower.esk`. Tensor towers are vectors of tensor
coefficients ordered from the primal coefficient through the requested order.

### `tt-add`

Adds two tensor towers coefficient by coefficient.

### `tt-const`

Creates a tower with a tensor primal coefficient and zero higher coefficients.

### `tt-div`

Computes the coefficient recurrence for division of two tensor towers.

### `tt-exp`

Computes the coefficient recurrence for elementwise exponential of a tensor
tower.

### `tt-hadamard-cauchy`

Multiplies tensor towers with Cauchy convolution and elementwise multiplication
as the inner operation.

### `tt-neg`

Negates each tensor coefficient in a tower.

### `tt-order`

Returns the highest coefficient order of a tensor tower.

### `tt-scale-const`

Scales every tensor coefficient by one ordinary scalar constant.

### `tt-sub`

Subtracts tensor towers coefficient by coefficient.

### `tt-value`

Returns the primal tensor, the order-zero coefficient of a tower.

## `tensor.utils`

<a id="tensor.utils"></a>

Source: `lib/tensor/utils.esk`. This module provides high-level tensor shape
helpers and is loaded with `(require tensor.utils)`.

## `tensorcore`

Source: `lib/tensorcore.esk`. These wrappers expose the Eshkol-owned flat
TensorCore adapter ABI. They return unavailable status values when the optional
backend is not present; see `docs/architecture/tensorcore-adapter.md`.

### `tc-adapter-available?`

Reports whether the TensorCore adapter is available.

### `tc-adapter-status`

Returns the adapter availability or ABI status code.

### `tc-attention-forward`

Runs scaled dot-product attention over TensorCore buffers.

### `tc-buffer-alloc`

Allocates a TensorCore buffer in the supplied context.

### `tc-buffer-free`

Releases a TensorCore buffer.

### `tc-buffer-map`

Maps a TensorCore buffer for host access.

### `tc-buffer-size`

Returns the allocated byte size of a TensorCore buffer.

### `tc-device-info`

Returns device family, name, memory, and dtype capability information.

### `tc-device-name`

Returns the device name for a TensorCore context.

### `tc-gemm`

Runs typed GEMM with optional transposes, alpha, and beta values.

### `tc-gemm-bf16`

Runs the default BF16 TensorCore GEMM operation.

### `tc-gemm-fp16`

Runs the default FP16 TensorCore GEMM operation.

### `tc-gemm-fp32`

Runs the default FP32 TensorCore GEMM operation.

### `tc-init`

Initializes TensorCore and returns an opaque context or null on failure.

### `tc-last-backend`

Returns the numeric code of the last selected TensorCore backend.

### `tc-last-backend-name`

Returns the name of the last selected TensorCore backend.

### `tc-last-status`

Returns the last TensorCore adapter status code.

### `tc-runtime-capabilities`

Returns the authoritative ABI, capability-mask, and backend-mask values for a
TensorCore context, or an empty list when the ABI is unsupported.

### `tc-runtime-capabilities-abi-version`

Returns the TensorCore runtime-capabilities ABI version.

### `tc-runtime-capabilities-status`

Validates a requested runtime-capabilities ABI version for a context.

### `tc-shutdown`

Releases a TensorCore context and returns its status code.

### `tc-status-string`

Returns the human-readable string for a TensorCore status code.

### `tc-version`

Returns the TensorCore version string.

## `web.web`

<a id="web.web"></a>

Source: `lib/web/web.esk`. This module is the browser-facing DOM, event, timer,
storage, fetch, and canvas surface. Its exports are registered as a module
surface even though the procedure list is implemented by the browser bridge.


## `core.algebraic_batch`

Source: `lib/core/algebraic_batch.esk`. The packed exact multiplication contract,
preflight bounds and receipt fields are documented in
[algebraic_batch.md](algebraic_batch.md).

### `make-algebraic-multiplication-plan`

Compiles an admitted algebraic owner into a sparse exact multiplication table.

### `algebraic-batch-mul`

Preflights and multiplies equally sized vectors of exact algebraic values.


## `core.algebraic_batch`

Source: `lib/core/algebraic_batch.esk`. Signatures and semantic constraints are documented in [algebraic_batch.md](algebraic_batch.md).

### `make-algebraic-multiplication-plan`

See [algebraic_batch.md](algebraic_batch.md) for this exported procedure and its source-backed contract.

### `algebraic-batch-mul`

See [algebraic_batch.md](algebraic_batch.md) for this exported procedure and its source-backed contract.


## `core.algebraic_domain`

Source: `lib/core/algebraic_domain.esk`. Signatures and semantic constraints are documented in [algebraic_linalg.md](algebraic_linalg.md).

### `algebraic-domain`

See [algebraic_linalg.md](algebraic_linalg.md) for this exported procedure and its source-backed contract.

### `algebraic-domain-ref`

See [algebraic_linalg.md](algebraic_linalg.md) for this exported procedure and its source-backed contract.


## `core.algebraic_field`

Source: `lib/core/algebraic_field.esk`. Signatures and semantic constraints are documented in [algebraic_field.md](algebraic_field.md).

### `make-algebraic-field`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `algebraic-field-degree`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `algebraic-field-polynomial`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `algebraic-generator`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `algebraic-element`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `algebraic-coefficients`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `algebraic-add`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `algebraic-sub`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `algebraic-scale`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `algebraic-mul`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `algebraic-inverse`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `algebraic-zero?`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `algebraic-equal?`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `algebraic-rational`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `algebraic-enclosure`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `algebraic-sign`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `make-algebraic-embedding`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `make-algebraic-compositum`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `algebraic-field-certificate`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.

### `make-multiquadratic-field`

See [algebraic_field.md](algebraic_field.md) for this exported procedure and its source-backed contract.


## `core.algebraic_linalg`

Source: `lib/core/algebraic_linalg.esk`. Signatures and semantic constraints are documented in [algebraic_linalg.md](algebraic_linalg.md).

### `algebraic-factor`

See [algebraic_linalg.md](algebraic_linalg.md) for this exported procedure and its source-backed contract.

### `algebraic-factor-solve`

See [algebraic_linalg.md](algebraic_linalg.md) for this exported procedure and its source-backed contract.

### `algebraic-factor-solve-transpose`

See [algebraic_linalg.md](algebraic_linalg.md) for this exported procedure and its source-backed contract.

### `algebraic-factor-summary`

See [algebraic_linalg.md](algebraic_linalg.md) for this exported procedure and its source-backed contract.

### `algebraic-matrix-mul`

See [algebraic_linalg.md](algebraic_linalg.md) for this exported procedure and its source-backed contract.

### `algebraic-solve`

See [algebraic_linalg.md](algebraic_linalg.md) for this exported procedure and its source-backed contract.


## `core.algebraic_update`

Source: `lib/core/algebraic_update.esk`. Exactness, update policy, residual,
positivity and ownership constraints are documented in
[algebraic_update.md](algebraic_update.md).

### `algebraic-update-factor`

Creates an owned exact factor for a square matrix, optionally requiring positive definiteness.

### `algebraic-update`

Returns a bounded exact low-rank update for `A + U V^T`.

### `algebraic-update-symmetric`

Applies a signed weighted symmetric update to an admitted positive-definite factor.

### `algebraic-update-solve`

Solves the updated exact matrix system and checks the full-domain residual.

### `algebraic-update-solve-transpose`

Solves the transpose of the updated exact matrix and checks the full-domain residual.

### `algebraic-update-pullback`

Returns the exact solution and an adjoint pullback for matrix and right-hand-side cotangents.

### `algebraic-update-info`

Returns the update route, rank, factorization-attempt counters and correction depth.


## `core.polynomial_exact`

Source: `lib/core/polynomial_exact.esk`. Signatures and semantic constraints are documented in [polynomial_exact.md](polynomial_exact.md).

### `exact-poly`

See [polynomial_exact.md](polynomial_exact.md) for this exported procedure and its source-backed contract.

### `exact-poly-zero?`

See [polynomial_exact.md](polynomial_exact.md) for this exported procedure and its source-backed contract.

### `exact-poly-sub`

See [polynomial_exact.md](polynomial_exact.md) for this exported procedure and its source-backed contract.

### `exact-poly-scale`

See [polynomial_exact.md](polynomial_exact.md) for this exported procedure and its source-backed contract.

### `exact-poly-mul`

See [polynomial_exact.md](polynomial_exact.md) for this exported procedure and its source-backed contract.

### `exact-poly-divmod`

See [polynomial_exact.md](polynomial_exact.md) for this exported procedure and its source-backed contract.

### `exact-poly-inverse-mod`

See [polynomial_exact.md](polynomial_exact.md) for this exported procedure and its source-backed contract.

### `exact-poly-eval`

See [polynomial_exact.md](polynomial_exact.md) for this exported procedure and its source-backed contract.

### `exact-poly-real-root-count`

See [polynomial_exact.md](polynomial_exact.md) for this exported procedure and its source-backed contract.

### `exact-poly-isolate`

See [polynomial_exact.md](polynomial_exact.md) for this exported procedure and its source-backed contract.


## `core.ad.algebraic`

Source: `lib/core/ad/algebraic.esk`. Signatures and semantic constraints are documented in [ad_algebraic.md](ad_algebraic.md).

### `make-algebraic-taylor`

See [ad_algebraic.md](ad_algebraic.md) for this exported procedure and its source-backed contract.

### `make-algebraic-hyperdual`

See [ad_algebraic.md](ad_algebraic.md) for this exported procedure and its source-backed contract.

### `algebraic-ad-value`

See [ad_algebraic.md](ad_algebraic.md) for this exported procedure and its source-backed contract.

### `algebraic-ad-constant`

See [ad_algebraic.md](ad_algebraic.md) for this exported procedure and its source-backed contract.

### `algebraic-ad-seed`

See [ad_algebraic.md](ad_algebraic.md) for this exported procedure and its source-backed contract.

### `algebraic-ad-coefficients`

See [ad_algebraic.md](ad_algebraic.md) for this exported procedure and its source-backed contract.

### `algebraic-ad-add`

See [ad_algebraic.md](ad_algebraic.md) for this exported procedure and its source-backed contract.

### `algebraic-ad-sub`

See [ad_algebraic.md](ad_algebraic.md) for this exported procedure and its source-backed contract.

### `algebraic-ad-mul`

See [ad_algebraic.md](ad_algebraic.md) for this exported procedure and its source-backed contract.

### `algebraic-ad-inverse`

See [ad_algebraic.md](ad_algebraic.md) for this exported procedure and its source-backed contract.

### `algebraic-ad-equal?`

See [ad_algebraic.md](ad_algebraic.md) for this exported procedure and its source-backed contract.

### `algebraic-ad-derivative`

See [ad_algebraic.md](ad_algebraic.md) for this exported procedure and its source-backed contract.

### `algebraic-ad-root`

See [ad_algebraic.md](ad_algebraic.md) for this exported procedure and its source-backed contract.

### `algebraic-ad-context`

See [ad_algebraic.md](ad_algebraic.md) for this exported procedure and its source-backed contract.

### `call-with-algebraic-derivative`

See [ad_algebraic.md](ad_algebraic.md) for this exported procedure and its source-backed contract.


## `core.ad.algebraic_implicit`

Source: `lib/core/ad/algebraic_implicit.esk`. Signatures and semantic constraints are documented in [ad_algebraic_implicit.md](ad_algebraic_implicit.md).

### `algebraic-ad-system-root`

See [ad_algebraic_implicit.md](ad_algebraic_implicit.md) for this exported procedure and its source-backed contract.

### `algebraic-ad-system-residual`

See [ad_algebraic_implicit.md](ad_algebraic_implicit.md) for this exported procedure and its source-backed contract.

### `algebraic-system-linearization`

See [ad_algebraic_implicit.md](ad_algebraic_implicit.md) for this exported procedure and its source-backed contract.


## `core.ad.algebraic_incremental`

Source: `lib/core/ad/algebraic_incremental.esk`. Signatures and semantic constraints are documented in [ad_algebraic_incremental.md](ad_algebraic_incremental.md).

### `make-algebraic-incremental-plan`

See [ad_algebraic_incremental.md](ad_algebraic_incremental.md) for this exported procedure and its source-backed contract.

### `algebraic-incremental-new`

See [ad_algebraic_incremental.md](ad_algebraic_incremental.md) for this exported procedure and its source-backed contract.

### `algebraic-incremental-step`

See [ad_algebraic_incremental.md](ad_algebraic_incremental.md) for this exported procedure and its source-backed contract.

### `algebraic-incremental-output`

See [ad_algebraic_incremental.md](ad_algebraic_incremental.md) for this exported procedure and its source-backed contract.

### `algebraic-incremental-info`

See [ad_algebraic_incremental.md](ad_algebraic_incremental.md) for this exported procedure and its source-backed contract.


## `core.ad.algebraic_pullback`

Source: `lib/core/ad/algebraic_pullback.esk`. Signatures and semantic constraints are documented in [ad_algebraic_pullback.md](ad_algebraic_pullback.md).

### `algebraic-solve-with-pullback`

See [ad_algebraic_pullback.md](ad_algebraic_pullback.md) for this exported procedure and its source-backed contract.

### `algebraic-factor-with-pullback`

See [ad_algebraic_pullback.md](ad_algebraic_pullback.md) for this exported procedure and its source-backed contract.

### `algebraic-system-with-pullback`

See [ad_algebraic_pullback.md](ad_algebraic_pullback.md) for this exported procedure and its source-backed contract.

### `algebraic-reverse-solve`

See [ad_algebraic_pullback.md](ad_algebraic_pullback.md) for this exported procedure and its source-backed contract.

### `algebraic-reverse-system`

See [ad_algebraic_pullback.md](ad_algebraic_pullback.md) for this exported procedure and its source-backed contract.


## `core.ad.algebraic_reverse`

Source: `lib/core/ad/algebraic_reverse.esk`. Signatures and semantic constraints are documented in [ad_algebraic_reverse.md](ad_algebraic_reverse.md).

### `algebraic-value-and-pullback`

See [ad_algebraic_reverse.md](ad_algebraic_reverse.md) for this exported procedure and its source-backed contract.

### `algebraic-reverse-value`

See [ad_algebraic_reverse.md](ad_algebraic_reverse.md) for this exported procedure and its source-backed contract.

### `algebraic-reverse-constant`

See [ad_algebraic_reverse.md](ad_algebraic_reverse.md) for this exported procedure and its source-backed contract.

### `algebraic-reverse-add`

See [ad_algebraic_reverse.md](ad_algebraic_reverse.md) for this exported procedure and its source-backed contract.

### `algebraic-reverse-sub`

See [ad_algebraic_reverse.md](ad_algebraic_reverse.md) for this exported procedure and its source-backed contract.

### `algebraic-reverse-mul`

See [ad_algebraic_reverse.md](ad_algebraic_reverse.md) for this exported procedure and its source-backed contract.

### `algebraic-reverse-inverse`

See [ad_algebraic_reverse.md](ad_algebraic_reverse.md) for this exported procedure and its source-backed contract.

### `algebraic-reverse-pow`

See [ad_algebraic_reverse.md](ad_algebraic_reverse.md) for this exported procedure and its source-backed contract.

### `algebraic-hvp`

See [ad_algebraic_reverse.md](ad_algebraic_reverse.md) for this exported procedure and its source-backed contract.

### `algebraic-ridge-new`

Declared by [core.ad.algebraic_ridge](../../../lib/core/ad/algebraic_ridge.esk#L4); see [ad_algebraic_ridge.md](ad_algebraic_ridge.md) for its arithmetic, shape, ownership and refusal contract.

### `algebraic-ridge-step`

Declared by [core.ad.algebraic_ridge](../../../lib/core/ad/algebraic_ridge.esk#L4); see [ad_algebraic_ridge.md](ad_algebraic_ridge.md) for its arithmetic, shape, ownership and refusal contract.

### `algebraic-ridge-output`

Declared by [core.ad.algebraic_ridge](../../../lib/core/ad/algebraic_ridge.esk#L4); see [ad_algebraic_ridge.md](ad_algebraic_ridge.md) for its arithmetic, shape, ownership and refusal contract.

### `algebraic-ridge-info`

Declared by [core.ad.algebraic_ridge](../../../lib/core/ad/algebraic_ridge.esk#L4); see [ad_algebraic_ridge.md](ad_algebraic_ridge.md) for its arithmetic, shape, ownership and refusal contract.

### `algebraic-ridge-pullback`

Declared by [core.ad.algebraic_ridge](../../../lib/core/ad/algebraic_ridge.esk#L4); see [ad_algebraic_ridge.md](ad_algebraic_ridge.md) for its arithmetic, shape, ownership and refusal contract.

### `algebraic-factor-positive-definite?`

Declared by [core.algebraic_linalg](../../../lib/core/algebraic_linalg.esk#L3); see [algebraic_linalg.md](algebraic_linalg.md) for its arithmetic, shape, ownership and refusal contract.

### `algebraic-incremental-moment-ref`

Declared by [core.ad.algebraic_incremental](../../../lib/core/ad/algebraic_incremental.esk#L4); see [ad_algebraic_incremental.md](ad_algebraic_incremental.md) for its arithmetic, shape, ownership and refusal contract.

### `make-tensor-circuit`

Declared by [core.dbsp.tensor_circuit](../../../lib/core/dbsp/tensor_circuit.esk#L3); see [dbsp_tensor_circuit.md](dbsp_tensor_circuit.md) for shape, arithmetic, signed-update and ownership constraints.

### `tensor-circuit-step`

Declared by [core.dbsp.tensor_circuit](../../../lib/core/dbsp/tensor_circuit.esk#L3); see [dbsp_tensor_circuit.md](dbsp_tensor_circuit.md) for shape, arithmetic, signed-update and ownership constraints.

### `tensor-circuit-output`

Declared by [core.dbsp.tensor_circuit](../../../lib/core/dbsp/tensor_circuit.esk#L3); see [dbsp_tensor_circuit.md](dbsp_tensor_circuit.md) for shape, arithmetic, signed-update and ownership constraints.

### `tensor-circuit-ref`

Declared by [core.dbsp.tensor_circuit](../../../lib/core/dbsp/tensor_circuit.esk#L3); see [dbsp_tensor_circuit.md](dbsp_tensor_circuit.md) for shape, arithmetic, signed-update and ownership constraints.

### `tensor-circuit-work`

Declared by [core.dbsp.tensor_circuit](../../../lib/core/dbsp/tensor_circuit.esk#L3); see [dbsp_tensor_circuit.md](dbsp_tensor_circuit.md) for shape, arithmetic, signed-update and ownership constraints.

## `core.dbsp.circuit`

Source: [lib/core/dbsp/circuit.esk](../../../lib/core/dbsp/circuit.esk). The logical state view includes hidden distinct counts and join arrangements for checkpoint verification.

### `dbsp-circuit-checkpoint-state`

Returns a pointer-free observation of all logical operator state for integrity verification during checkpoint restore.

## `core.dbsp.resident`

Source: [lib/core/dbsp/resident.esk](../../../lib/core/dbsp/resident.esk). Owner lifetime, generation-checked snapshots, compaction, and native-only boundaries are documented in [dbsp_resident.md](dbsp_resident.md).

### `call-with-dbsp-resident`

Runs a callback with a scoped resident owner and its initial snapshot.

### `dbsp-resident-step`

Imports changes and returns a retained successor snapshot after one circuit tick.

### `dbsp-resident-release!`

Releases a retained snapshot handle.

### `dbsp-resident-output`

Returns an owned observation of a snapshot's circuit output.

### `dbsp-resident-delta`

Returns an owned observation of a snapshot's output delta.

### `dbsp-resident-work`

Returns an owned observation of a snapshot's circuit work count.

### `dbsp-resident-stats`

Returns root, arena, compaction and copy counters for the owner.

### `dbsp-resident-compact!`

Requests compaction at an idle owner boundary and returns updated statistics.

## `core.dbsp.checkpoint`

Source: [lib/core/dbsp/checkpoint.esk](../../../lib/core/dbsp/checkpoint.esk). The bounded exact-data codec, trusted callback registry and caller/filesystem authorization boundary are described in [dbsp_checkpoint.md](dbsp_checkpoint.md).

### `make-dbsp-resumable`

Builds a versioned declarative wrapper for an immutable standing DBSP circuit.

### `dbsp-resumable-step`

Applies the next exact logical tick and returns a successor snapshot with its output delta.

### `dbsp-resumable-output`

Returns an owned observation of the current output Z-set.

### `dbsp-resumable-tick`

Returns the current exact logical tick.

### `dbsp-resumable-checkpoint`

Encodes the template contract, input state, tick and digest of full logical operator state in canonical bounded bytes.

### `dbsp-resumable-restore`

Verifies canonical integrity and template compatibility, rebuilds from current inputs and checks the full operator-state digest before returning a snapshot.

### `dbsp-checkpoint-save`

Writes a checkpoint through the existing atomic file-replacement primitive; caller authorization and filesystem permissions govern access.

### `dbsp-checkpoint-load`

Reads a bounded checkpoint file and returns it only after verified restoration.

## `core.certified_solve_pullback`

Source: [lib/core/certified_solve_pullback.esk](../../../lib/core/certified_solve_pullback.esk). The two exact solve certificates, uncertain-model premises and pullback interval conventions are documented in [certified_solve_pullback.md](certified_solve_pullback.md).

### `solve-certified-native-pullback`

Certifies primal and transpose solves independently, then encloses `-lambda*x^T` and the RHS cotangent from the original matrix, RHS, seed and supplied uncertainties.

### `verify-certified-native-pullback`

Recomputes both original-model solve certificates and their interval products; it does not trust stored status or derivative boxes.

## `core.certified_native_solve`

Source: `lib/core/certified_native_solve.esk`; see [certified_native_solve.md](certified_native_solve.md) for the candidate-generation and exact verification contract.

### `solve-certified-native`

Generates candidates with the native dense solver, then admits a result only through exact verification against the original rational model.

### `verify-certified-native-solve`

Rechecks a proved result against caller-supplied source data and tolerance.

## `core.certified_solve`

Source: `lib/core/certified_solve.esk`; see [certified_solve.md](certified_solve.md) for the exact rational admission contract, uncertainty bounds, and theorem limits.

### `certify-linear-solve`

Computes an exact a-posteriori componentwise error enclosure for a received candidate when the weighted contraction bound is strictly below one.

### `verify-linear-solve-certificate`

Recomputes the receipt from caller-supplied sources; true means the record matches, even when its status is `unknown`. Callers must separately require `proved` before using boxes as an enclosure.
## `core.ad.derived_operator`

Source: `lib/core/ad/derived_operator.esk`. The restricted syntax, exact finite-change identity, derivative contract and refusal limits are documented in [ad_derived_operator.md](ad_derived_operator.md).

### `make-derived-operator`

Compiles one bounded pure scalar polynomial and its symbolic partials and shifted finite-change polynomial.

### `derived-operator-value`

Evaluates the admitted polynomial at a fixed-width point.

### `derived-operator-jvp`

Computes the gradient paired with a direction at the point.

### `derived-operator-vjp`

Computes the formal polynomial transpose of the gradient scaled by a seed.

### `derived-operator-delta`

Computes the exact polynomial difference between the point plus an increment and the point.

### `derived-operator-info`

Returns immutable arity, degree and conservative term-work metadata.

### `derived-operator-enclosure`

Evaluates the source polynomial and its symbolic derivatives over exact rational boxes with rigorous interval bounds.

## `core.certified_nonlinear`

Source: `lib/core/certified_nonlinear.esk`; the root and implicit-sensitivity certificate contract is documented in [certified_nonlinear.md](certified_nonlinear.md).

### `make-certified-polynomial-system`

Compiles a bounded coupled polynomial system with exact rational coefficients.

### `certify-polynomial-root`

Validates a supplied approximate root using whole-box Jacobian bounds and a strict contraction certificate.

### `verify-polynomial-root-certificate`

Recomputes the certificate from the source system and caller supplied trial data.

## `core.ad.root_branch`

Source: [lib/core/ad/root_branch.esk](../../../lib/core/ad/root_branch.esk). The exact whole-step continuation contract and certificate limits are documented in [ad_root_branch.md](ad_root_branch.md).

### `make-root-branch`

Admits an exact polynomial family and a selected simple real root with a certified isolating interval.

### `root-branch-step`

Certifies continuation across a whole parameter step using interval Newton and a root-count certificate.

### `root-branch-enclosure`

Returns the certified root enclosure at the current parameter.

### `root-branch-time`

Returns the exact current parameter value.

### `root-branch-sensitivity`

Returns an exact interval enclosure for the implicit derivative on the admitted branch.

### `verify-root-branch-step`

Recomputes the recorded step certificate and checks it against the supplied branch state.

## `core.net`

Native numeric-address sockets; see [net.md](net.md) for platform, capability, lifetime and deadline limits.

### `tcp-listen`

Binds a numeric IPv4/IPv6 TCP listener; port zero selects an ephemeral port.

### `tcp-connect`

Connects a numeric TCP endpoint within the supplied millisecond deadline.

### `tcp-accept`

Returns an accepted stream handle, or a network error when its bounded wait fails.

### `tcp-read`

Reads up to the byte capacity; empty string is EOF and error -2 is timeout.

### `tcp-write`

Sends string bytes, including NUL; returns the possibly partial byte count.

### `tcp-close`

Invalidates a handle once; stale and repeated closes return error -4.

### `udp-bind`

Binds a numeric datagram endpoint, including an ephemeral local port.

### `udp-sendto`

Sends one datagram, including an empty payload, to a numeric endpoint.

### `udp-recvfrom`

Returns #(payload numeric-peer-endpoint); oversized datagrams refuse with error -4.

### `udp-close`

Closes a datagram handle once and rejects a stale handle.

### `net-local-address`

Returns the numeric local address, or a network error.

### `net-peer-address`

Returns the numeric TCP peer address, or a network error.

### `net-local-port`

Returns the exact local port, or a network error.

### `net-peer-port`

Returns the exact TCP peer port, or a network error.

### `net-nonblocking!`

Selects immediate polling; unavailable operations return timeout error -2.

### `network-error?`

Recognizes the public three-element network-error vector.

### `network-error-code`

Returns the error code from a recognized network-error value.

### `network-error-operation`

Returns the operation symbol from a recognized network-error value.

## `core.ad.incremental_reverse`

Source: `lib/core/ad/incremental_reverse.esk`. Exact scalar-DAG pullbacks across sparse input and cotangent edits are described in [ad_incremental_reverse.md](ad_incremental_reverse.md).

### `make-incremental-pullback`

Creates a persistent pullback state from an exact owner, named inputs, a fixed DAG, outputs, input values and output cotangent seeds.

### `incremental-pullback-step`

Applies sparse additive input and seed changes and returns the successor state, output and input-cotangent deltas, and work counters.

### `incremental-pullback-output`

Returns owned output values and input cotangents for a pullback state.

### `incremental-pullback-work`

Returns forward refresh, reverse refresh, changed-edge, copied-slot and scanned-node work counts for a step.

### `incremental-pullback-info`

Returns the declared input, node and output counts for a state.
