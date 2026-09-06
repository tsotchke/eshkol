# Eshkol-S: the device-eligible fragment

Status: contract, stage S2a of the XLA-to-TPU program. This document defines
what "device-eligible" means for a piece of Eshkol code, how every compiler
builtin is classified against that definition, and the parity rule a
device-classified builtin must satisfy before it is trusted. It does not
implement a lowering (that is S2b, which reads the classification this
document's companion table produces as its input) and it does not implement
region formation (a later stage; see "Region formation" below for how the
two connect). The oracle criteria this document exists to earn are
`stablehlo_fragment_contract_present` and `stablehlo_builtin_classification_complete`
in `.icc/completion-oracles.yaml` (target `stablehlo-fragment-coverage`).

## Why a fragment, not the whole language

Eshkol is a dynamically-typed Scheme with cons cells, strings, ports,
call/cc, mutation, and an untyped numeric tower alongside typed tensors.
StableHLO is a fixed-shape (or boundedly dynamic-shape), statically-typed,
side-effect-free tensor IR with structured control flow. Most of Eshkol
cannot be compiled to StableHLO, and pretending otherwise produces either a
compiler that silently falls back to something else while claiming device
execution, or a compiler that refuses to build almost every real program.
The fragment approach instead names, precisely, the subset of Eshkol
expressions that map onto StableHLO honestly, and treats everything else as
a first-class citizen that runs on the host. A program is not required to be
written entirely in the fragment for any of it to benefit — see "Region
formation" below.

## The Eshkol-S contract

An Eshkol expression is in Eshkol-S — the device-eligible fragment — when
every one of the following holds. All five conditions are conjunctive; an
expression that fails any one of them is not in Eshkol-S and stays on the
host.

### 1. Value domain

Every value the expression produces or consumes is one of:

- a numeric scalar of a StableHLO-representable element type (the signed and
  unsigned fixed-width integers, IEEE binary floating point, and bf16 that
  StableHLO's element-type set defines), or
- a tensor of such scalars with a StableHLO-representable shape (see
  "Shape discipline" below), or
- a tuple composed recursively of the above.

Excluded, unconditionally: cons cells and lists, strings and characters,
ports, closures and continuations (values, not the call sites — see
"Structured control flow" for lambda arguments passed to host
higher-order functions), symbols, hash tables, the arbitrary-precision
exact/rational numeric tower, complex numbers as Eshkol represents them
(tagged heap values, not a fixed-width complex element type StableHLO
commits to), and every consciousness-engine value (knowledge bases, factor
graphs, workspaces, substitutions). None of these has a StableHLO
representation the compiler is willing to claim today; if one becomes
representable later, its builtins move from `host` to `device` in the
classification table and the criterion that gates this document is
re-earned, not silently reinterpreted.

### 2. Shape discipline

Shapes are static, or dynamic only along dimensions StableHLO's bounded
dynamism can express (a declared upper bound with a runtime-tracked actual
size, `stablehlo`'s `get_dimension_size` / bounded-dynamic-shape machinery).
An expression whose shape depends on a value that is not itself a bound —
for example a shape that depends on how many elements of a list satisfy a
predicate — is not in Eshkol-S at that point, even if every operation
involved is otherwise device-eligible.

### 3. Structured control flow

Control flow lowers to one of StableHLO's structured control-flow ops:

- `if` with both arms in the fragment lowers to `stablehlo.case` (a
  two-branch case).
- A tail-recursive loop over fragment-typed state (accumulator values with
  fixed shape and type across iterations) lowers to `stablehlo.while`.

Excluded: `call/cc` and any other non-local exit, `set!` on any binding that
is not itself loop-carried state inside a `stablehlo.while` region (i.e. no
mutation of host state from inside a fragment), and exception handling
(`with-exception-handler`, `error`) — these are host control-flow features
by construction; StableHLO has no notion of an escaping exception.

### 4. Purity

The expression performs no I/O and mutates nothing outside its own result.
This rules out file and port operations, RNG functions that draw from
host-managed seed state, in-place tensor mutation (`tensor-set!` as opposed
to the value-returning `tensor-set`), and any function whose contract is
"update this host-side record" (optimizer step functions, gradient-tape
bookkeeping). A pure StableHLO computation can still *use* random bits —
`stablehlo.rng` and `stablehlo.rng_bit_generator` are pure, seed-in
seed-out — but Eshkol's current RNG builtins are specified against host PRNG
state, so they are classified `host` until a device-seeded variant exists.

### 5. Builtin closure

Every builtin the expression calls has a StableHLO lowering, or a
decomposition into a sequence of ops that do. "Decomposition" is the CHLO
pattern already established in the ecosystem this program targets: `asinh`
decomposes to `log` and `sqrt` primitives, `gcd` decomposes to a bounded
`stablehlo.while` computing repeated remainder, and so on. A builtin with no
known lowering or decomposition is not callable from Eshkol-S, full stop —
it does not get partial credit, and an expression that calls it is host, not
"host except for that one call."

## What is host-only, and why

Three kinds of thing are excluded from Eshkol-S by nature, not by
temporary limitation:

- **Values with no fixed device representation**: cons/lists, strings,
  symbols, ports, hash tables, the exact/rational numeric tower, the
  consciousness-engine's knowledge/factor-graph/workspace objects, and
  Eshkol's own automatic-differentiation tape. The tape in particular is
  worth naming explicitly: `gradient`, `jacobian`, `hessian`, and the
  `ad-*`/`dual-*` families all read and write a host-resident tape data
  structure to do reverse- or forward-mode differentiation. The *tensor
  arithmetic* those functions differentiate through can be, and often is,
  device-eligible; the differentiation machinery itself is not, until a
  device-native AD strategy (e.g. tracing through `stablehlo.while` with an
  XLA-native VJP) replaces the host tape. Until then, tape-facing builtins
  are `host`.
- **Non-local and side-effecting control**: `call/cc`, exception handling,
  file/network/process I/O, mutation of host state, RNG against host seed
  state. StableHLO has no representation for any of these; they do not
  become device-eligible by writing a bigger decomposition, because the
  thing being asked for (an escaping continuation, a side effect visible
  outside the computation) is not a property StableHLO computations can
  have.
- **Host-side object lifecycle and metadata**: constructing or destroying an
  opaque handle (a manifold handle, a dataloader, an optimizer's Adam
  state), and querying static metadata about a tensor (its declared shape,
  dtype, or the raw data pointer) rather than computing over its
  *contents*. These are compiler/runtime bookkeeping operations, not tensor
  computations, even when the object they manage wraps device data.

## Three labels for every builtin

Every builtin the compiler registers gets exactly one of three labels. This
is what makes "the whole language has been considered" a checkable claim
rather than a slogan: an unclassified builtin is a gap, and the coverage
gate treats it as one.

- **`device`**: the builtin has a StableHLO lowering, or a decomposition
  into ops that do, per condition 5 above. Graded by the parity rule below.
  A `device` label is a claim that the builtin *can* be represented purely
  in the fragment when all of its arguments are; it is not itself a claim
  that today's emitter has implemented that lowering (that gap is exactly
  what `stablehlo_device_builtin_parity`, the S2b criterion, exists to
  close).
- **`host`**: the builtin is host-only by nature, per one of the three
  reasons above (no device value representation, non-local/side-effecting
  control, or object lifecycle/metadata). Listed explicitly in the
  classification table so that an omission reads as a gap, not as an
  implicit "obviously host" that nobody had to write down.
- **`host-with-device-inner`**: a host builtin whose argument evaluation can
  itself contain a device region. The canonical case is `map` over a list
  of tensors: `map` itself is host (it walks a list, a host-only value),
  but the procedure it applies at each step can be a pure tensor
  computation that region formation is free to outline into a device
  function. Every entry with this label names the specific inner
  evaluation that is eligible — usually "the argument procedure's body,
  when that body is itself in Eshkol-S" for a true higher-order function,
  but occasionally something narrower.

Coverage is defined as the count of builtins carrying one of these three
labels, over the total number of builtins the compiler registers. A `host`
classification counts as full credit toward coverage: the goal is that
every construct in the language has been considered and placed, not that
every construct executes on an accelerator. An unclassified builtin — one
present in the compiler's registry but absent from the classification table,
or vice versa — is a hard failure of `stablehlo_builtin_classification_complete`,
checked mechanically by `scripts/check_builtin_classification.py` against
`lib/backend/xla/builtin_classification.yaml`.

The compiler's builtin registry is not a single table; `scripts/gen_language_surface.py`
already reconciles the three dispatch surfaces that exist (the native/AOT
closure table and LLVM dispatch in `lib/backend/eshkol_compiler.c`, the
bytecode VM's table in `lib/backend/eshkol_vm.c`, plus the small quantum/PQC
agent-module surface) into one deduplicated list, published as
`tests/coverage/language_surface.json`. `builtin_classification.yaml` is
built from that list's `builtins` array — not retyped from memory — and the
checker re-derives the registry the same way (by loading the same manifest)
every time it runs, so a builtin added to the compiler and never classified
fails the gate the next time it runs, with no need to remember to update a
second, independent list by hand.

## The parity rule

A builtin classified `device` is not trusted at parity until its StableHLO
lowering is executed and its result is compared, element-by-element, against
the same computation run through the existing host runtime (the native/LLVM
or VM execution path already used everywhere else in the compiler), on the
same inputs. The comparison tolerance is stated per dtype, because a
bit-exact requirement across float pipelines that do not commit to the same
reduction order or fused-multiply-add behavior is not an honest bar:

| dtype   | tolerance                                             |
|---------|--------------------------------------------------------|
| f64     | \|device - host\| <= 1e-9 absolute, or 1e-9 relative, whichever is looser |
| f32     | \|device - host\| <= 1e-5 absolute, or 1e-5 relative, whichever is looser |
| bf16    | \|device - host\| <= 4e-2 absolute, or 4e-2 relative, whichever is looser (bf16 carries roughly 3 significant decimal digits; this bound is the numerics campaign's existing bf16 sweep tolerance, not a new number invented for this document; measured against it on TPU in "bf16 across the dimension sweep" below) |
| integer types (signed/unsigned, all widths) | exact equality |
| boolean | exact equality |

### Operation classes: the dtype bound is not the whole rule

The table above holds a matmul and a logarithm to the same bar, and they do
not deserve the same bar. An add, a multiply, a dot product, a reduction and
a transpose are *exact* operations: performed in f32 their only error is the
rounding of the inputs and of the result. `exp`, `log` and `tanh` are not
operations at all on a TPU — they are approximations, evaluated by a
reduced-precision elementwise unit whose accuracy is a documented property of
the hardware and not a defect in any lowering.

Measured by `tests/xla/op_parity_test` on TPU hardware through its PJRT
plugin, f32 device arithmetic against the f64 host reference, over the shapes
that harness uses:

| op | max abs error | max rel error | class |
|----|---------------|---------------|-------|
| add, subtract, multiply (rank 2 and broadcast) | 0 | 0 | arithmetic |
| divide | 6.8e-8 | 7.9e-8 | arithmetic |
| matmul (`[4,6] x [6,3]`) | 0 | 0 | arithmetic |
| transpose (`[4,6]->[6,4]` and `[2,3,4]` perm `{2,0,1}`) | 0 | 0 | arithmetic |
| broadcast (`[6] -> [4,6]`) | 0 | 0 | arithmetic |
| reduce sum / mean / max / min (full and per-axis) | 0 | 0 | arithmetic |
| sin | 2.8e-8 | 3.4e-8 | transcendental |
| cos | 6.1e-8 | 1.6e-7 | transcendental |
| exp | 1.2e-5 | 3.2e-6 | transcendental |
| tanh | 4.4e-5 | 5.2e-5 | transcendental |
| log | 6.7e-5 | **2.2e-4** | transcendental |

So every row of the arithmetic class landed at or below 7.9e-8 relative —
two orders inside its 1e-5 bound — while the transcendental class reached
2.2e-4, twenty-two times outside it. Holding the second group to 1e-5 does
not make it more accurate; it makes the gate report a hardware property as a
defect, which is how a gate stops being read.

Each `device` builtin therefore belongs to exactly one **tolerance class**,
and the class scales the dtype bound above:

| class | membership | bound |
|-------|-----------|-------|
| arithmetic | exact operations: `+ - * /`, dot/matmul, every reduction, and every pure data movement (transpose, reshape, broadcast, concatenate, slice, pad, gather, scatter) | the dtype bound above, unchanged |
| transcendental | approximated elementary functions evaluated by the device's elementwise unit: `exp`, `log`, `sin`, `cos`, `tanh`, and anything later added beside them (`sigmoid`, `sqrt`, `pow`, `erf`, `rsqrt`) | **100x** the dtype bound: 1e-3 at f32, 1e-7 at f64. bf16 is the exception and keeps 4e-2 unscaled, because at bf16 the dtype's own quantization already dominates any approximation error the elementwise unit adds |

The factor is 100 rather than the measured 22x because a bound sitting just
above the worst observation is a bound that fails the next time the input
range, the tensor shape or the TPU generation changes, and a flaky gate gets
switched off. 1e-3 at f32 is still roughly three and a half correct decimal
digits, and it is three orders of magnitude tighter than what a genuinely
wrong lowering produces: the transpose defect S2b found and fixed reported
2.14 relative, and a wrong op or a wrong shape is an O(1) error, never a
1e-4 one. The bound is loose enough not to be flaky and tight enough that no
real defect can hide behind it.

Two things this rule deliberately does NOT do. It does not widen the
arithmetic class — every exact operation is still held to the dtype bound,
and any exact op that misses it is a defect to be found, not a tolerance to
be raised. And membership is decided by what an operation IS, not by what it
measured today: `sin` and `cos` currently come back at f32 rounding level on
this hardware and are still classified transcendental, because classifying by
measurement would mean re-deciding the contract every time a number moved.

The f64 transcendental figure (1e-7) is derived from the factor, not
measured: no PJRT plugin exercised in this program computes transcendentals
in f64 yet. When one does, the measurement replaces the derivation here.

Should a specific device builtin need a looser bound than its class allows
(a reduction over a very large tensor accumulating more rounding error, for
instance), that exception is recorded next to the measurement that justified
it; a looser tolerance is never assumed in advance.

### Measured: the dot is computed in the precision it is asked for

An f32 `stablehlo.dot_general` at StableHLO's DEFAULT precision is not an f32
dot on a TPU. The matrix unit takes bf16 operands, so both operands are
rounded to bf16 and only the accumulation is f32 — about three decimal digits
short of what the program asked for, with the operand types, the result type
and the shapes all still f32 and nothing reporting the demotion.

Every matmul row in this program measured 0.000e+00 error against that, for
one reason: their inputs were multiples of 0.25 and 0.125 at small magnitudes,
all exactly representable in bf16's 8 mantissa bits, so the rounding was the
identity. The demotion first appeared in the gradient harness's two-layer
composite, whose dot operands are `tanh` outputs and therefore not dyadic: its
gradients came back at 1.7e-3 and 1.9e-3 relative, which is 2^-9.

Measured on TPU with a deliberately non-dyadic matmul row (operands stepping
by 0.17 and 0.13):

| dot precision_config | forward matmul row, max rel | gradient matmul row, max rel | composite dL/dW1 | dL/dW2 |
|---|---|---|---|---|
| DEFAULT (what a null config means) | 4.829e-3 — **FAIL** | 3.063e-3 — **FAIL** | 1.706e-3 | 1.850e-3 |
| HIGHEST (what the emitter now emits) | 9.018e-8 | 7.776e-8 | 4.203e-6 | 1.865e-5 |

Both FAIL entries are against the 1e-5 arithmetic bound. The dyadic matmul
rows read 0.000e+00 in every cell of that table, under both settings.

The arithmetic class is not widened for this. Computing in a narrower type
than the program asked for is a defect to be found, which is what the rule
above says, so `stablehlo_emitter.cpp` emits an explicit `HIGHEST`
precision_config on every `dot_general`. That costs real time on TPU — it is
the multi-pass bf16 decomposition rather than a single pass — and
`ESHKOL_XLA_DOT_PRECISION=default|high|highest` overrides it without a rebuild
for anyone who has measured that the loss is acceptable for their model. Both
harnesses now carry a non-dyadic matmul row, so a return to the silent
demotion cannot pass either gate.

### Measured: bf16 across the dimension sweep, and at the Poincare boundary

Stage S7 of the XLA-to-TPU program (criterion `xla_bf16_numerics_bounded`,
gate `scripts/run_xla_gate.sh --numerics`, harness
`tests/xla/bf16_numerics_test.cpp`). `ESHKOL_XLA_DEVICE_DTYPE=bf16` makes
bf16 the device element type end to end: parameters, buffers and results
are `bf16` in the StableHLO module, PJRT stages `kBf16` buffers, and the
host's f64 tensors are rounded to bf16 (round-to-nearest-even, in
`device_lowering.cpp`) on the way in and widened exactly on the way back.
Every number below is from that path on TPU hardware, against a direct f64
evaluation of the same formula, graded by the rule above: `|device - host|
<= 4e-2` absolutely OR relatively, whichever is looser.

**The 4e-2 bf16 bound holds, and no third class was needed.** 258 graded
rows — every S2 op at d in {2, 4, 16, 64, 256, 1024}, every S4 geometric
primitive at the same six dimensions under the mixed-precision policy
below, and the explicit hyperbolic-boundary rows — landed inside it. The
largest graded error anywhere was `cos` at 4.99e-2 relative / 1.25e-2
absolute (inside on the absolute bound). A third class would only have
been added if a row had been measured outside 4e-2 and still been a
correct lowering; none was.

**S2 ops, raw bf16 (d does not change the domain; it changes the tensor
size and, for reductions, the accumulation length).** Worst of the six
dimensions per op:

| op | max abs | max rel | note |
|----|---------|---------|------|
| add / subtract / multiply / divide | 2.5e-2 / 1.06e-2 / 3.75e-2 / 2.79e-3 | 4.46e-3 / 9.49e-3 / 6.16e-3 / 4.88e-3 | flat in d |
| maximum / minimum / abs / negate | 6.88e-3 / 5.0e-3 / 3.75e-3 / 3.75e-3 | 3.0e-3 / 2.8e-3 / 3.23e-3 / 3.23e-3 | flat in d; the error is the bf16 rounding of the input itself |
| exp / log / sin / cos / tanh | 1.37 / 5.1e-3 / 4.66e-3 / 1.25e-2 / 1.9e-3 | 1.45e-2 / 1.88e-2 / 6.59e-3 / **4.99e-2** / 4.24e-3 | flat in d; exp's absolute is at exp(4.5) |
| sqrt / rsqrt / sigmoid / atanh / pow | 8.36e-3 / 2.75e-3 / 1.62e-3 / 1.13e-2 / 1.35e-2 | 4.1e-3 / 3.13e-3 / 2.59e-3 / 8.16e-3 / 5.62e-3 | flat in d |
| reduce_sum | 1.25e-3 (d=2) ... **3.2 (d=1024)** | 2.66e-3 at every d | absolute error grows linearly with d; the row passes on the RELATIVE bound only |
| reduce_mean / max / min | 3.13e-3 / 6.25e-3 / 1.95e-4 | 2.66e-3 / 2.91e-3 / 9.77e-4 | flat in d |
| reduce_prod (factors 1 +/- 2^-6, bf16-exact) | 1.95e-3 | 1.96e-3 | flat in d |

`reduce_sum` is the one row where d shows: a bf16 sum over 1024 terms of
order 1 is off by 3.2 absolutely at a constant 2.7e-3 relatively. That is
the accumulation being carried in bf16 and it is why the reductions are the
first thing the mixed-precision policy widens.

**S4 geometric primitives, both tables.** Worst of the six dimensions per
primitive; "raw" is the device computing entirely in bf16, "mixed" is the
policy below (bf16 in and out, f32 inside). Interior points (|x| = 0.5 on
the ball, unit vectors on the sphere), c = 1, guard eps = 1e-6:

| primitive | raw max abs / rel | mixed max abs / rel |
|-----------|------------------|---------------------|
| mobius_add | 2.98e-3 / 2.39e-2 | 1.07e-3 / 1.76e-2 |
| poincare_exp_map_origin | 2.07e-3 / 7.94e-3 | 8.64e-4 / 6.59e-3 |
| poincare_log_map_origin | 1.69e-3 / 1.13e-2 | 8.90e-4 / 6.19e-3 |
| poincare_exp_map | 2.81e-3 / 2.08e-2 | 2.12e-3 / 2.17e-2 |
| poincare_log_map | 4.98e-3 / 2.03e-2 | 1.07e-3 / 1.59e-2 |
| hyperbolic_distance | 5.01e-3 / 2.97e-3 | 5.90e-3 / 3.03e-3 |
| poincare_project | 2.14e-4 / 6.57e-3 | 2.14e-4 / 6.71e-3 |
| poincare_retract | 1.23e-3 / 26.3 (abs bound) | 1.23e-3 / 26.3 (abs bound) |
| sphere_project | 1.57e-3 / 6.08e-3 | 3.37e-3 / 6.08e-3 |
| sphere_retract | 2.85e-3 / 26.3 (abs bound) | 2.18e-3 / 26.3 (abs bound) |
| sphere_exp_map | 3.95e-3 / 1.56e-2 | 1.89e-3 / 1.56e-2 |
| sphere_log_map | 9.83e-3 / 9.66e-3 | 4.84e-3 / 7.56e-3 |
| spherical_distance | 5.16e-3 / 2.53e-3 | 5.16e-3 / 2.53e-3 |
| euclidean_exp_map / log_map | 1.23e-3 / 26.3 ; 1.34e-3 / 84.9 (abs bound) | same |
| euclidean_distance | 3.45e-3 / 4.31e-3 | 1.77e-3 / 2.50e-3 |

The large relative figures on the retract and Euclidean rows are
components that are near zero (a step that cancels a coordinate), graded on
the absolute bound as the rule says; their absolute errors are the bf16
rounding of the inputs. Interior, the raw and mixed tables are within a
factor of two of each other: away from the boundary bf16 compute is not
the problem, bf16 storage is.

**The hyperbolic boundary, d = 64, c = 1, points at |x| sqrt(c) = r.** This
is the row the criterion label names. Each cell is max abs / max rel:

| primitive | r = 0.9 raw | 0.9 mixed | 0.99 raw | 0.99 mixed | 0.999 raw | 0.999 mixed | 1-2^-8 raw | 1-2^-8 mixed |
|---|---|---|---|---|---|---|---|---|
| poincare_exp_map_origin | 6.6e-4 / 5.2e-3 | 6.6e-4 / 5.2e-3 | 7.1e-4 / 5.8e-3 | 8.1e-4 / 5.5e-3 | 7.7e-4 / 6.1e-3 | 6.6e-4 / 5.1e-3 | 1.0e-3 / 9.2e-3 | 6.3e-4 / 5.4e-3 |
| poincare_log_map_origin | 3.6e-3 / 1.3e-2 | 2.9e-3 / 1.1e-2 | 1.8e-2 / 3.5e-2 | 2.5e-3 / 4.9e-3 | **inf / inf FAIL** | 5.4e-3 / 8.9e-3 | 3.5e-3 / 6.9e-3 | 4.9e-3 / 9.1e-3 |
| poincare_exp_map | 6.7e-4 / 5.9e-3 | 8.7e-4 / 6.3e-3 | 8.7e-4 / 6.9e-3 | 8.7e-4 / 6.2e-3 | 5.6e-4 / 4.4e-3 | 5.6e-4 / 4.4e-3 | 1.1e-3 / 6.7e-3 | 8.0e-4 / 5.8e-3 |
| poincare_log_map | 5.2e-4 / 5.2e-2 | 4.2e-4 / 2.3e-2 | 5.4e-4 / 6.3e-2 | 4.4e-5 / 6.3e-3 | 1.4e-3 / 1.0 | 3.4e-5 / 2.8e-2 | 2.8e-5 / 7.7e-3 | 1.0e-4 / 2.6e-2 |
| hyperbolic_distance | 2.9e-2 / 1.6e-2 | 2.2e-2 / 1.2e-2 | 2.0e-2 / 4.8e-3 | 1.1e-2 / 2.6e-3 | **inf / inf FAIL** | 2.6e-2 / 4.0e-3 | 1.5e-2 / 2.9e-3 | 1.7e-2 / 3.2e-3 |
| mobius_add | 1.6e-3 / 1.1e-2 | 7.3e-4 / 4.9e-3 | 7.7e-4 / 5.4e-3 | 7.7e-4 / 4.7e-3 | 5.1e-4 / 4.0e-3 | 5.1e-4 / 4.0e-3 | 9.2e-4 / 5.4e-3 | 9.2e-4 / 6.9e-3 |

Two raw rows are FAIL and are recorded as such, not exempted. The
mechanism is exactly the one the criterion label anticipates, and it is
worth being precise about which value bf16 cannot represent:

- **bf16 cannot represent the conformal factor 1 - c|x|^2 near the
  boundary, and the norm reduction saturates to 1.** At r = 0.999, |x|^2
  is 0.998. bf16 has 8 significand bits, so its spacing just below 1.0 is
  2^-8 = 0.0039 (and 2^-9 just above 0.5): 0.998 is not representable and
  the raw-bf16 dot product `sum(x_i^2)` accumulates to exactly 1.0. From
  there `sqrt(c)|x| = 1.0`, and the log map's artanh argument is 1.0.
- **The artanh clamp cannot engage in bf16.** `qllmArtanh` clamps its
  argument at `kArtanhClamp = 1.0 - 1e-7`
  (`lib/backend/xla/geometric_lowering.cpp:54`), emitted as a constant in
  the module's element type. In bf16 that constant IS 1.0, so `select(arg
  >= clamp, clamp, arg)` selects 1.0 and `artanh(1.0) = inf`. In f32 the
  same constant is 0.99999994, which is the value the host computes with.
  The same saturation makes `(1 - c|x|^2)` exactly 0 in the distance's
  denominator, hence its inf.
- **1 - 2^-8 passes in raw bf16 while 0.999 does not** because 1 - 2^-8 =
  0.99609375 is a bf16 grid point: its square, 0.9922, sits between grid
  points but rounds to 0.9921875, not to 1.0, so the conformal factor
  survives as 2^-7. The failure is not "too close to the boundary" in a
  continuous sense; it is whether |x|^2 rounds to a value below 1.0 or to
  1.0 itself.
- **The host's own floors are unreachable in bf16.** The Mobius quotient
  is floored on the host at `kMobiusDenFloor = 1e-15`
  (`lib/bridge/qllm_bridge.cpp:221`), and the projection and retract
  guards take `eps = 1e-6` as a runtime operand (the convention in
  `tests/xla/geometric_parity_test.cpp:639`). None of those values is
  distinguishable from 0 next to a value of order 1 in bf16 (2^-8 spacing),
  so a lowering that relies on them to keep a denominator away from zero
  gets no help from them under bf16 compute. The mixed-precision policy is
  what restores them: with the body in f32, `1 - 1e-7`, `1e-6` and the
  conformal factor at r = 0.999 all take their intended values, which is
  why every boundary row is bounded in the mixed column.

**Near-coincident points are bounded absolutely, not relatively.** Rows at
y = x + 1e-3 u (unit u), d = 64, for sphere_log_map, spherical_distance,
hyperbolic_distance and poincare_log_map measured max rel of 1.0 in raw
mode (the device returns 0: `1 - <x,y>` is below the 2^-8 grid the inputs
were rounded to on transfer, so `acos(1) = 0` and `acosh(1) = 0`) and 1.0
to 35.6 in mixed mode, with max abs between 7.5e-5 and 1.3e-3. Every one
of those rows PASSES under the "absolute or relative, whichever is looser"
rule, because the true answer is itself of order 1e-3. That is the honest
reading: for two points closer than bf16 can tell apart, bf16 says "same
point" to within 1e-3 absolutely and carries no relative information at
all. A model that needs the direction between near-coincident points has
to keep those points in f32 storage; no compute policy recovers what the
transfer rounded away. The harness reports these under their own
`storage_limited` counter so the regime stays visible.

### Mixed precision under bf16: what stays f32, and why

The policy, implemented in `buildGeometricModule()`
(`lib/backend/xla/geometric_lowering.cpp`, `mixed_precision=true`) and
applied by `runGeometric` / `runGeometricGradient` when the device dtype is
bf16:

- **Storage and transfer are bf16.** Every module parameter and every
  result is `bf16`; PJRT stages 2-byte buffers; nothing about the memory
  footprint or the transfer volume changes.
- **The primitive body computes in f32.** Each parameter is widened
  `bf16 -> f32` immediately on entry, the whole decomposition — every
  reduction (dot products and norms), the conformal factor, the artanh
  argument and its clamp, the Mobius quotient, the acosh — is emitted in
  f32, and the result(s) are narrowed `f32 -> bf16` once, at the return. A
  VJP under the policy differentiates the f32 body and narrows its
  cotangents the same way.

This is deliberately the whole body and not a per-op selection. The
measurements above identify three things that MUST be f32 — the norm and
dot-product reductions (the raw reduce_sum row grows to 3.2 absolute at
d=1024; the raw norm at r=0.999 saturates to exactly 1), the artanh
argument and its clamp (1 - 1e-7 does not exist in bf16), and the
conformal factor / distance denominator near the boundary (0 in bf16 at
r=0.999) — and everything else in these primitives is a handful of
elementwise ops on the same values, which cost nothing to keep in f32 next
to them. The TPU's elementwise unit computes in f32 anyway; a body that
narrowed to bf16 between ops would be paying rounding for no throughput.
Widening exactly those three and narrowing around them would produce a
module with more converts than arithmetic and no measurable gain over this
one.

What the policy does NOT do, and is measured not to do: it does not change
the interior numbers materially (the two S4 tables above agree to within a
factor of two on every interior row), it does not recover information the
bf16 storage of the inputs already lost (the near-coincident rows), and it
does not exist for f32 or f64 device dtypes (`mixed_precision` is a no-op
unless the device element type is BF16).

The gate grades the policy-active rows, because the policy is what the
lowering ships. The raw table is measured on every run and its failures
are printed with their numbers, so a future change that makes raw bf16
bounded — or breaks the policy — shows up as a changed table, not as a
silent pass. `ESHKOL_XLA_BF16_FORCE_FAIL=1` disables the policy for the
graded rows, and the gate then reports FAIL on the two 0.999 rows above:
that is the recorded proof that `xla_bf16_numerics_bounded` can fail.

### Gradients take the class of the operation they differentiate

A reverse-mode VJP is graded by the same rule and against the same two
bounds, under the tolerance class of the FORWARD operation it differentiates,
not of the ops it happens to be built from. The VJP of `tanh` is
`g * (1 - tanh(x)^2)`: it evaluates the same approximated elementary
function the forward pass did, so it is transcendental even though the
multiply and the subtract in it are exact. The VJP of a matmul is two more
matmuls and is arithmetic. Classifying a gradient by its own op list would
put every transcendental backward in the arithmetic class purely because the
chain rule multiplies, which is how a hardware property would come to be
reported as a defect.

Two properties of a gradient are NOT tolerance questions and are graded
exactly, because they are choices rather than approximations:

- **Un-broadcasting.** A `[3]` bias broadcast against a `[2,3]` activation
  receives a `[3]` cotangent that is the sum over the axis it did not span.
  A rule that omits that sum produces the RIGHT SHAPE and the wrong numbers,
  which no shape check catches, so a broadcast row is part of the gradient
  contract rather than an optional case.
- **Ties in `max`/`min` reductions.** A reduction with repeated extrema has
  no derivative, only a convention, and the device's convention must be the
  host's or a program's gradient would depend on where it ran. Eshkol's host
  AD defines `max`/`min` for scalars only (`AD_NODE_MAX`/`AD_NODE_MIN`): the
  whole gradient goes to the first operand when it is strictly greater and
  otherwise to the second, so a reduction — which is a fold of that rule —
  gives the whole cotangent to the LAST tied element in row-major order and
  nothing to the others. `tests/xla/host_max_tie_convention.esk` measures
  that through the host language AD and the gradient gate refuses to pass if
  the two paths disagree.

## Region formation: how the whole language gets it

A program does not have to be written in Eshkol-S for any of it to benefit.
Region formation (a later stage than this document covers) walks the AST,
marks each node eligible or not against the contract above, and outlines
each *maximal* eligible connected subgraph into a device function; the host
program calls that function at the boundary and continues. This mirrors how
tracing works in the production systems this program is modeled on: the
graph break is the unit of failure, and it is reported, not hidden. A
program that produces one giant region runs (almost) entirely on the
device; a program that produces none runs exactly as it does today. Both are
correct outcomes of the same pass.

The boundary between a region and the host is an explicit host-to-device
transfer on the way in and a device-to-host transfer on the way out — the
same boundary `PjrtClient::bufferFromHost` and `bufferToHost` already
implement. Builtins that materialize a tensor from host data (`tensor`,
`make-tensor`) or that convert between a host-domain value and a tensor
(`tensor->vector`, `vector->tensor`) sit exactly at this boundary; they are
classified `host` in the table below not because tensors are host values,
but because the act of *transferring* is a host-side operation, distinct
from computing over an already-resident device tensor.

The fragment is expected to grow over time, and each growth is its own
oracle criterion, not a silent reclassification: non-escaping closures
inlined into their call sites, fixed-length lists reinterpreted as tensors,
host-side loops over tensor batches hoisted into the fragment. None of that
is implemented by this document; it is named here so that a future
classification change is judged against a written contract rather than
against whatever the emitter happens to accept that week.
