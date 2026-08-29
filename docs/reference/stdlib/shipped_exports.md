# Additional shipped Eshkol exports

This page closes the DD-11 reference gap for shipped modules whose exports were
not present in the standard-library index. Signatures and module locations are
taken from the source files named in each section. The anchors are checked by
`scripts/check_public_api_docs.py`.

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
helpers and is loaded with `(require tensor-utils)`.

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
