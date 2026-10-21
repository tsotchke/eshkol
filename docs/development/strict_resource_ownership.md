# Native strict resource ownership (experimental)

`eshkol-run --strict-resources` adds an opt-in static check for owned network
resources created by the stock `core.net` module. The check is available for
native compilation and `--run`; it does not change the default compiler path.
It is not available to the VM backend or dynamic `--eval` input.

The checker recognizes TCP listeners, TCP streams, and UDP sockets returned by
the pinned stock SDK `core.net` definitions. It tracks successful acquisitions,
error-result refinement through the authentic `network-error?` predicate,
observation, consuming close, lexical borrows, and structured branch joins.
Factories and resource-consuming calls are accepted only through verified
module bindings; a same-spelled local function, unknown builtin, raw extern,
or mutated alias does not gain a network effect contract.

Required module initializers are checked along with user top-level forms,
because the compiler and JIT loader execute those forms. Function bodies are
not executed during module initialization, so their declaration expressions
are not treated as resource acquisition. Calls whose effects cannot be proved
in the supported fragment are rejected when they could affect resource safety.
Bound callable values require explicit effect provenance; computed callable
results and recursive callable initializers are not assumed resource-free.

The checked fragment excludes resource-sensitive loops, continuations, dynamic
evaluation, quasiquotation, and unsupported control forms. Diagnostics identify
these boundaries rather than treating them as safe. The `tests/v1_4_resource_ownership`
corpus covers compile-time positives and falsifiers; its runtime loopback test
allocates a fresh local TCP port for each JIT and AOT run.

Opaque `.o` inputs, explicit `-l` libraries, and WebAssembly output are refused
with this profile because their initializers or runtime effects are outside the
checked source AST.
