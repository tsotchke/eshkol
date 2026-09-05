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
| bf16    | \|device - host\| <= 4e-2 absolute, or 4e-2 relative, whichever is looser (bf16 carries roughly 3 significant decimal digits; this bound is the numerics campaign's existing bf16 sweep tolerance, not a new number invented for this document) |
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
