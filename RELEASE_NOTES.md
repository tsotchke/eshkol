# Eshkol v1.3.5-evolve — Release Notes

**Candidate date:** September 11, 2026.
**Status:** release candidate; final verification and publication are pending.

Feed the compiler a source file sixteen thousand parentheses deep, on a thread
with an eight-megabyte stack, and it compiles it. That is not a metaphor for
robustness. The recursive-descent parser was replaced with an explicit
continuation stack: a child parse suspends into a heap-allocated coroutine
frame and is resumed through a linked list, so `await_suspend` only records a
child and `final_suspend` only suspends — neither one calls the other. Native
stack consumption is therefore independent of grammar nesting, and stays
independent in an unoptimized build. Two downstream passes that used to
re-introduce the dependency, the type checker's `synthesize` and the code
generator's `codegenAST → codegenOperation → codegenCall → codegenArithmetic`
chain, run on the same driver. **16,000.**

The rest of the release is the same discipline applied to answers rather than
to depth. Dense tensor autodifferentiation now executes end to end, where the
reverse pass previously fell into dead code on the path a live program could
reach; the two lowerings are gated against each other for byte-identical
gradients. `tensor-apply` used to
resolve its second operand through a table of builtin *names*, so shadowing a
name changed nothing; it now calls the callable you actually passed, through
the same dispatcher an ordinary lambda application uses. The constant-curvature
geometry surface used to exist twice; it exists once. And the bytecode VM,
which had no heap reclamation of any kind, reclaims.

## Highlights

### The parser has no recursion budget

`stack space exhausted during parsing — expression nesting too deep` was a real
answer the compiler could give to a legal program. Expressions, list special
forms, quote and quasiquote, vectors, string interpolation, types, `match`
patterns, `syntax-rules` patterns, import sets and feature requirements now all
suspend into explicit continuations instead of into the native stack. Lambda
capture collection became an explicit pending-node vector, and type parsing
attaches arena-owned child trees rather than copying subtrees, which removes a
quadratic type allocation along the way. Two gates hold the line: one runs
16,000 levels of nesting on an actual 8 MiB pthread stack under an 8 MiB
process resource limit, the other fixes both the soft and hard process stack
limits at 8 MiB and executes 16,000 nested additions through JIT and AOT. Both
run on Linux x64 and macOS ARM64 in CI. The 65,536-byte safety margin is
unchanged, and no production stack limit moved.

The Linux stack guard was also inverted: it measured consumed space where it
meant remaining space, and now measures from the low stack bound.

This is a claim about the parse-and-lower chain, not about every pass in the
compiler. Specialized backend helpers outside that chain keep their synchronous
entry points.

### Dense tensor autodiff executes

`matmul`, `tensor-sum` and `tensor-mean` each record exactly **one** dense AD
node under differentiation, and that node is what `matmul` returns. Before, the
dense path was unreachable: the reverse pass discriminated on `tensor_gradient`,
which is null at record time, so a tensor node fell into the scalar dispatch and
dereferenced the `input1`/`input2` pointers a tensor node legitimately leaves
null. The reverse pass now discriminates on `tensor_value` *or*
`tensor_gradient`; under AD, `matmul` returns the node tagged as a callable with
the AD-node subtype, and outside AD mode returns the plain tensor unchanged. A
new registry row bridges scalarized operands by identity scatter — it performs
no arithmetic, which is why the scalar and dense lowerings agree exactly rather
than closely. Dense elementwise arithmetic goes through the same shape-aware
dispatcher and accumulates repeated indices for broadcast operands.

The gate compares the two lowerings for byte-identical gradients, ratchets the
recorded node count, and asserts that the per-op cost of a 6×6 matmul equals
that of a 2×2 once the gradient driver's per-input variable nodes are
subtracted. Ledger entry SW-48.

### `tensor-apply` calls the callable, not the name

`(tensor-apply tensor callable)` evaluates both operands once and invokes the
resolved callable on each scalar in row-major order. No function-name table and
no identity substitution participates any more, so a lexical binding or a user
definition that shadows a historical builtin name determines what runs. A
non-callable raises, even for an empty tensor. Native code generation routes it
through the same `codegenClosureCall` an ordinary lambda application uses, and
the VM routes it through the single `vm_enter_call` entry shared by threaded
`OP_CALL`, switch `OP_CALL` and native higher-order invocation. The scalar load
is the one tensor indexing already uses, so reverse-mode node pointers and
complete forward jets survive the call — gradients, differentiable captures and
Hessians work through user procedure forms.

Gated across four engines — LLVM JIT at O0, LLVM AOT at O2, VM source execution
and emitted ESKB — each of which must report all **31** ordered assertions, with
JIT caching disabled.

### The bytecode VM reclaims memory

`(with-region ...)` used to lower to `begin` on the VM: the body ran, the value
came back, and not one byte was returned — peak RSS was the same to within a
tenth of a percent whether the wrapper was there or not. It now reclaims, and
the claim is measured rather than asserted. Swept by iteration count on macOS
ARM64, peak RSS is **33 MB at 1,000 iterations, 34 MB at 4,000, and 34 MB at
16,000** — sixteen times the work for one megabyte — against **304 MB** for the
identical program on the identical binary with the evacuator disabled, and
125 MB for the unwrapped control. The gate requires the flat curve, the on/off
separation, *and* the printed answer to be identical either way.

The port matches native semantics, not native implementation. Native copies the
escaping subgraph; the VM marks from its root set and sweeps at arena-block
granularity, because a VM value addresses the heap by a small integer index and
a copying evacuator would have to rewrite those indices — a missed rewrite
aliases a live object and surfaces as a wrong *value*, never as invalid memory
access. Marking
moves nothing, so `eq?`, shared structure and cycles need no special handling.
The subtype table classifies the full 33-wide heap tag space with a compile-time
span check, a fatal startup check that every row is filled, and a `default:` arm
that pins rather than guesses. Every uncertain case — an unclassified subtype, a
continuation captured inside the region, a failed bookkeeping allocation — pins
the region and degrades toward a leak, never toward a dangling index.

Outside a region the VM still does not reclaim, and the heap-growth watchdog
stays for exactly that case. The user-reachable `region-open` / `region-close`
handle surface remains bookkeeping-only on the VM.

### Continuations are multi-shot on all three engines

A captured continuation can be invoked any number of times, from any dynamic
extent, including after the procedure that captured it has returned. Generators,
coroutines and `amb`-style backtracking now run correctly on first attempt,
where they previously terminated abnormally on native and hung on the VM.
Native gives a capture that may outlive
its frame a durable copy of the live C stack, restored to the same addresses
before the `longjmp`, so every interior pointer stays valid with no relocation;
escape-only captures keep the original zero-overhead path. The VM snapshots its
operand stack and call-frame array and excludes top-level bindings — the
*store* — from the *control* snapshot, so `set!` and `define` effects at top
level survive re-entry. `dynamic-wind` reroots on both engines per R7RS 6.10.

A continuation captured inside `with-region` now pins that region on native, as
it already did on the VM. The failure direction on both engines is a leak, never
a dangle.

### One implementation of the constant-curvature geometry

The qLLM bridge called its own copies of distance, the exponential and
logarithmic maps and geodesic attention; it now calls the shared
`riemannian_core.h` primitives, so there is one implementation and one set of
reverse rules. Those rules are derived from the shared scaled forward rather
than written independently: the log coincidence limit is exact, spherical
forward and reverse share a domain, attention is overflow-safe, and a derivative
request beyond `f64` is refused explicitly instead of answered. Curvature series
use a degree-10 branch in `q = K r² / 4` with cancellation-free derivatives,
witnessed against binary128 across a 1,067-binade sweep, and adjoints are
exponent-scaled at subnormal curvature.

### A mathematical construction, written down as a build plan

Eshkol now documents, step by step, how a published finite-time Navier-Stokes
blowup construction would be obtained inside the language. The new design note
walks the paper's own structure — similarity coordinates and the leading field,
the cumulative radial moments, the admissible stress cone, the heat exterior and
the analytic axis profiles, the order-by-order background correction, the
auxiliary torus, the two-family stress solve, the residual-improvement ladder,
and the localization to a compactly supported force — across 84 numbered proof
steps, and for each step names the Eshkol primitive that performs it or the
build item that will, together with the gate that certifies it. What makes this
tractable is the combination the language already ships: exact rational and
bignum arithmetic, Taylor towers whose coefficients stay exact, forward and
reverse differentiation, and validated enclosures — so an identity that is
supposed to cancel closes to exact zero rather than to a tolerance. The four
steps that are executable today — the viscosity-scaling identity through the AD
residual operator, the similarity exponents as an exactly solved rational
system, the leading-order profile balance by Taylor-coefficient collection with
a negative control, and the pulse momentum-flux averages with the two-family
stress solve — run as the companion example programs the note calls for
(`examples/mathematics_navier_stokes_{viscosity_scaling,similarity_scales,pulse_stress,first_principles}.esk`),
discovered by the existing examples suite. A residual oracle built on
automatic differentiation (`core.pde.ns-residual`, over a new
`core.symbolic` layer of polynomials and truncated power series on the exact
rationals) mechanizes the construction's residual ladder, verifying the residual to
order N, and a ledger records, per step, whether the result is exact, validated by
enclosure, or analytic-only — built up honestly rather than claimed ahead of
what runs.

What *is* in this cut is programs that verify published finite witnesses in
pure Eshkol, and they run: the 2026 Jacobian-conjecture counterexample and its
fiber geometry (11 checks), AlphaTensor rank-23 and rank-47
matrix-multiplication factorizations over F2 (256 basis pairs), the
FunSearch 512-cap in AG(8,3) (130,816 exact pair checks), and a further set of
exact-mathematics example programs spanning finite group cohomology,
Dijkgraaf-Witten and Yetter invariants, sheaf cohomology on finite spaces,
homotopy colimits, and Hodge classes on Fermat hypersurfaces — each with a
closed-form or exactly-computed verdict, negative controls, and independent
cross-checks. All pass.

### Model I/O: ESKM v1 is the validated default

Public tensor and model saves go through validated ESKM v1 readers and writers
on both engines; a single tensor is stored as one record with an empty name.
Loading validates its payload and refuses corrupt or missing input rather than
materializing whatever the bytes happened to say. Publication writes a complete
temporary checkpoint in the destination directory and commits it by
same-directory rename, so a handled failure before the commit preserves the old
destination. A deterministic, resource-bounded fuzz gate runs malformed
checkpoints, a compatibility corpus pins the historical format, and a
cross-reader matrix exercises the four producer/consumer engine combinations.

This is an atomic-replacement contract, not a power-loss durability guarantee.
Checkpoint save does not `fsync` both the file and its parent directory, and
abrupt machine loss or `SIGKILL` cleanup is not promised.

### Assurance gates that are measured against deliberate mutations

A gate that has never rejected anything is not evidence. The compiler-assurance
runner executes the production closed-enum gate **twelve** times in an isolated
source projection — four repeats each with zero, one and two handwritten AD
dispatcher case labels removed — and requires that every baseline pass, every
treatment be rejected, and the finding counts increase with dose by more than
the measured within-dose spread. It has already caught a false green: a registry
include had been exempting all manual inline cases from missing-member checks,
and only disposition macros that emit case labels now earn generated coverage
credit. Alongside it, a public-API linkage gate generates a volatile
function-pointer relocation per exported prototype from the umbrella header and
requires the result to compile, link and execute; it covers **104** prototypes
today. Every run writes an evidence receipt — argv, streams, exit status,
timing, source hash, gate hash, git revision.

That framework is also what produced the routing work above. When it was
introduced it reported 32 AST operation switches carrying omissions or
`default:` arms and seven direct callable consumers outside the canonical
dispatcher; `docs/platform/COMPILER_ASSURANCE.md` still records those findings
as blocking, and the routing change that answers them landed afterwards. Which
of the two the release ships is a question for the final battery, not for this
paragraph.

Elsewhere in the assurance layer: every completion-oracle criterion is bound to
a registered test and gated, so a pillar cannot quietly stop being covered; a
self-verdict scanner, build fingerprints and adversarial scenarios feed the
readiness artifact; the failure-attribution parser has a self-test and the
aggregate runner rejects stale bare-`FAIL` attribution; an external Rosette
oracle runs as an advisory P7 differential lane; and the release readiness gate
is bound to the exact checkout it graded, so evidence from an earlier branch run
cannot certify a later cut.

## Also in this release

- **Exact and nested differentiation.** Foreign-epoch perturbations are opaque
  with respect to the current value recurrence rather than flattened to a
  constant, so a closure-captured outer tower is preserved rather than
  erased unreported by an inner pass, and depth and per-level order are
  now unbounded — every pass
  owns its own level and a foreign level is a coefficient of the active one.
  An exact rational point now reaches the derivative carrier at all three
  scalar operators, a vanishing tangent keeps the seed's exactness instead of
  demoting an exact sum to a double, and a comparison or branch inside a
  differentiand acts on the carrier's primal rather than its tagged bits. A
  named-let or `do`-loop variable captured by an inline `derivative`/
  `derivative-n`/`taylor` is read as the value it is, not as a re-wrapped
  pointer marker. Exact tangent sidecars have a defined layout and survive
  evacuation. `abs` and `relu` apply one whole-series rule on both the native
  and VM Taylor dispatchers. Curried gradient-of-gradient is exact: with
  `(define g (gradient f))`, `(jacobian g point)` answers the Hessian
  entry-for-entry.
- **The exact tower closes several remaining gaps.** `(sqrt 4/9)` answers
  `2/3`, and it is `exact?`: `sqrt` of a perfect square and `expt` with an
  exact rational exponent take the root over the rationals when one exists and
  fall back to the inexact path only when one does not, so exactness follows
  the value rather than the operator. `expt` is exact for a rational base and
  for a negative exponent — `(expt 2/3 -3)` is `27/8`, with the reciprocal
  taken exactly rather than rebuilt from a float. A flat numeric vector literal
  keeps an exact rational or bignum element as the value it is instead of
  reinterpreting it as the bit pattern of a double, and a tensor built from an
  exact non-integer element converts it once, explicitly, at construction. A
  quoted bignum or bignum-rational literal — native, quasiquoted, or on the
  bytecode VM — is the same value as its evaluated form.
  `core.exact_linalg` adds exact rational linear algebra and torus averaging
  (matrix multiply, transpose, fraction-free determinant, solve, inverse,
  rank, nullspace) over the scalar exact tower, staying exact under R7RS
  numeric contagion whenever every input does.
- **Exactness under differentiation is a property of the runtime value, not of
  the shape of the source.** The exact tier used to decide from a static
  whitelist over the differentiand's body, so identical arithmetic demoted to a
  double when the constant arrived through a top-level `define` rather than an
  inline literal, when the body was a several-deep composed call, or when the
  point argument was an expression such as `(car ts)` whose runtime value was
  exact all along. A differentiand that branches on a numeric comparison of the
  differentiation variable now differentiates on the carrier's primal, and a
  call expression may be passed directly as the differentiand. The tier reads
  the carrier's runtime exactness, so `(derivative f 1/3)` is exact whenever the
  arithmetic it performs is. One shape is deliberately held back: two enclosing
  differentiation levels over an order-2 inner pass raise a diagnostic rather
  than answer, because the carrier holds exactly one first-order companion. The
  carrier rewrite that lifts the restriction is v1.4 work, and this release does
  not claim nested differentiation at arbitrary depth.
- **Exact-rational arithmetic reclaims memory like integer arithmetic.**
  A running exact-rational loop now grows resident memory with the values it
  produces rather than with the work an operation does. GCD reduction now
  runs before multiplication rather than after; a numeric primitive commits
  a reclamation boundary of its own instead of retaining whatever scratch
  the arena had until the enclosing scope ended; a loop's own scope now
  promotes only the live set across a back edge instead of retaining the
  whole iteration by default; `cond` clauses are analyzed structurally
  instead of falling through the scope-admission test as an unrecognized
  callee; and exact-tower allocation sites route through the same
  arena-selection call as every other loop temporary. A crossed heap
  ceiling is now a fail-closed contract: the runtime reports the breach once,
  in bytes, and exits nonzero rather than printing per-block and exiting 0.
- **Every callable builtin is a first-class value on both engines.** A
  builtin usable in call position but not as a value — `(map vector-copy
  ...)` raised `Undefined variable: vector-copy` though `(vector-copy v)`
  worked directly — is now resolvable as a value across 586 more names,
  audited mechanically against the full builtin surface manifest.
- **Element-wise vector/tensor arithmetic dispatches on both operands, and
  says where.** A scalar against a Scheme vector in the *left* operand
  position (`(* (vector 1 2) 2)`) raises a catchable type error naming the
  actual source line; both operand positions are classified before either is
  dereferenced, independent of whichever call site first emitted the shared
  arithmetic dispatcher. Mismatched element-wise operands are refused at the
  shape check rather than read past the shorter one (ledger LE-22), an
  arithmetic runtime error carries the line of the form that raised it rather
  than the line of the dispatcher (ledger LE-19), and a raw floating-point
  operand reaching the tagged-value path is boxed before use, so the emitted
  IR is well-formed for every operand shape.
- **Certified enclosures.** A proof-backed layer under the existing validated
  interval arithmetic and Taylor models: outward-rounded interval arithmetic
  and Makino-Berz Taylor models whose remainders are always derived from a
  proven bound, never sampled, available through an explicit `rigorous?`
  flag with the validated modules' default behavior unchanged. Two runtime
  primitives, `fl-next-up` and `fl-next-down`, carry `nextafter` into both the
  native backend and the bytecode VM and are what the outward rounding is
  built on — one nudge per endpoint, with exact operands staying exact. The
  new leaf modules are `core.ad.rigorous_interval` (`ia+`, `ia-`, `ia*`,
  `ia/`, `ia-sqrt`, `ia-exp`, `ia-log`, `ia-sin`, `ia-cos`, `ia-atan`,
  `ia-pi`) and `core.ad.rigorous_taylor_models` (`tm+`, `tm*`, `tm-compose`,
  `tm-integrate`, `tm-deriv`, `tm-bound`, `tm-enclose`, `tm-prove-nonzero`,
  `tm-prove-bound`, and the transcendentals), re-exported from
  `core.ad.taylor_models` beside a `tm-rigorous?` predicate.
  `docs/reference/stdlib/certified-enclosures.md` writes down every remainder
  derivation.
- **The browser REPL answers, and a gate now watches that it does.** The
  REPL's echo of a form's value is emitted by the session that owns the
  transcript rather than by the `display` opcode, so the opcode keeps exactly
  one meaning and the echo carries its own line terminator — which matters
  through Emscripten, where stdout reaches the embedder one complete line at a
  time. `(display "hi")` in the REPL is a fragment awaiting a `(newline)`,
  exactly as it is under `eshkol-run -r`. The bundle is now built by
  `scripts/build-wasm-repl.sh` from a source list shared with the WebAssembly
  execute-and-diff lane, so the module the site loads and the module CI
  executes are the same link, and the lane drives `repl_eval` through a
  `print` callback shaped like the site's, checking the transcript a
  line-oriented host actually receives against
  `tests/wasm_diff/REPL_TRANSCRIPT.tsv`.
- **Forward-mode duals survive the neural primitives** — layer norm, scaled-dot
  attention and `tensor-get` — on both engines, and runtime, statically typed
  and densified tensor activations all route to the same rules.
- **The VM's own semantics.** `letrec` and `letrec*` patch each upvalue from the
  sibling slot it actually captured, including forward references. A wrong-arity
  call to a raw builtin or a conversion intrinsic is refused, and the refusal
  reads the builtin's real minimum rather than its opcode's operand count. A
  handler returning from a non-continuable `raise` raises a secondary exception
  on every engine. Character predicates return canonical booleans and classify
  Unicode identically to native. Exception handlers are released during
  teardown, and the implicit Riemannian-Adam state pool grows instead of
  clobbering past sixteen shapes.
- **The VM loads the canonical standard library** on the source, REPL and
  bytecode paths, and its prelude cache is gated against the dispatch table.
- **One reader grammar.** Tokenizer, VM parser, VM datum reader and hosted datum
  reader share one string-escape grammar, and the runtime reader handles
  quasiquote, unquote and unquote-splicing — all four readers agree. R7RS 7.1.1
  vertical-line symbols read and write on both engines.
- **One PRNG sequence per seed** across native JIT, native AOT and the VM.
- **Shared limits, decided before allocation.** A strict vector limit of 2^28 is
  rejected before allocation on every engine, and the hosted compiler's
  `make-vector` uses that same shared limit and diagnostic rather than its own
  256-element clamp.
- **Closure capture at scale.** Capture counts are encoded losslessly in a
  versioned bytecode, dynamic capture storage traverses every capture in
  evacuation and parallel paths, and parallel dispatch uses the dynamic
  environment ABI for all capture counts across `map`, `for-each`, `execute` and
  `fold`.
- **Tail transfer without an arity bound.** The tail-transfer dispatcher removes
  the differing-arity and non-AArch64 restrictions.
- **Object ABI v2, stages 1 and 2.** Cache keys carry an ABI fingerprint,
  `--abi-fingerprint` reports the object ABI tag directly instead of a `strings`
  heuristic, the WASM lane guards its geometry, and a machine-generated header
  inventory pins the layout with a mixed-link guard and a ratchet over 1,303
  scanned sites.
- **Compiles against LLVM 18 through 24.** The intrinsic-signature check, block
  terminator queries and the loop-vectorize hint each go through the
  compatibility layer, and AArch64 instruction selection at O0 is no longer
  quadratic.
- **Toolchain and packaging.** A canonical `FindEshkol.cmake` and a complete
  packaged link contract; `compile_commands.json` exported whenever tests are
  built, so the ABI ratchet always has its input; a `linux-x64-debug` required
  lane that builds with assertions and switch warnings; ephemeral, non-root,
  least-privilege container runners for the self-hosted mesh; and a shared
  FetchContent source cache.
- **A machine-consumable REPL protocol.** `eshkol-repl --machine` keeps its
  original `EREPL READY` / `EREPL DONE` / `EREPL FAIL` framing and now also
  speaks EREPL v1: JSON requests on stdin (`eval`, `complete`, `is_complete`,
  `reset`, `shutdown`) answered with `EREPL/1 {...}` response lines on stderr.
  A driver can evaluate code, get identifier completions, ask whether an input
  form is complete, and interrupt a running evaluation without a PTY, without
  matching prompts by regular expression, and without classifying failures by
  this project's error wording — every failure carries a structured
  `error.kind` from a small closed set. An `eval` response reports the form's
  own value separately from whatever it printed. `tools/erepl_client.py` is a
  stdlib-only Python reference driver with a `--self-test` covering every
  request type, wired into CTest.
- **The same binary64 arithmetic on every engine.** Floating-point contraction
  is a per-target liberty, so leaving it at the compiler default made
  cross-engine agreement depend on which instructions a back end happens to
  have: a forward-mode dual quotient rule shared by the native layer-norm
  kernel and the VM's dual division became one fused multiply-add on AArch64
  and x86-64-with-FMA and a separate multiply and subtract on WebAssembly,
  differing in the last printed digit. Every translation unit, and the WASM
  differential's Emscripten invocation, now compile with contraction off, so
  both engines evaluate binary64 arithmetic exactly as written; a kernel that
  wants a fused, singly-rounded product asks for it with an explicit `fma()`.
  `docs/VM_PARITY.md` records the rule as part of the parity contract.
- **A documented optional argument is a legal call on both engines, from a
  derived fact.** `(substring "hello" 1)`, `(append)`, `(gcd)`,
  `(make-vector 3)`, `(make-string 3)`, `(read-line)`, `(bytevector-append)`
  and `(hash-ref table key)` answer the same thing under the bytecode VM as
  under native code. The VM's minimum arities are now generated from code that
  runs — the fixed-arity macros the native dispatch expands — rather than
  transcribed by hand across 739 rows.
- **A variadic builtin used as a value answers what the name answers in
  operator position.** `(map list xs)`, `(map vector xs ys)`,
  `(apply vector (list 1 2 3))` and `(define f string-append) (f "a" "b" "c")`
  all agree with the call-position lowering. The first-class builtin table
  declares which rows are variadic and how each computes its answer from a rest
  list, so a variadic name has one implementation rather than one per call site.
- **`(exit <computed integer>)` is accepted on every engine.** A computed exit
  argument is unpacked by runtime type tag the way every other polymorphic
  numeric builtin is: a double is clamped to `[0, 255]` and truncated, an int64
  is truncated, a boolean follows R7RS 6.11, and any other runtime type raises a
  catchable runtime error rather than feeding an arbitrary bit pattern to the
  process exit status. A CTest case asserts the exact process status on native
  JIT, native AOT and the standalone VM.
- **Built-in R7RS libraries resolve from one table on both engines.**
  `(import (scheme base) …)` names a library Eshkol provides itself; the set of
  such libraries now lives in a single header consulted by the native front end
  and the VM, so adding one is a single row and neither engine can drift from
  the other's idea of which libraries exist without a source file. Every R7RS
  import modifier — `only`, `except`, `prefix`, `rename` — reaches the VM, and
  parallel workers keep their captured values.
- **The bytecode VM reports execution coverage for the constructs it lowers
  inline.** Under the coverage trace, a compiled form emits a marker carrying
  the same stable head-symbol hash the call marker uses, so reaching it at run
  time is the construct's execution evidence, and the marker survives bytecode
  serialization so the standalone VM and the hosted VM profile report
  identically. Instrumentation is opt-in and behaviour-neutral: an unarmed run
  emits no extra instruction, and all corpus programs produce byte-identical VM
  output armed and unarmed. Both engine-parity floors are now measured ratchets
  rather than aspirations — each is written from the run's own measured
  fraction, and a baseline whose floor exceeds the corpus ceiling it was
  measured against is rejected as malformed rather than graded. On this cut the
  differential covers **321 of 1,139 constructs (28.18%)** and **155 of 473
  high-risk constructs (32.77%)**, with **five dispositioned divergences and no
  new one**. The high-risk surface the corpus does not yet reach is filed as a
  v1.4 corpus-growth item rather than left implicit; `docs/VM_PARITY.md` carries
  both measurements and that inventory.
- **Binding scope and recursive call resolution.** A binding form's names now
  shadow only inside that form — the free-variable walk carries a per-scope
  bound set instead of subtracting a form's names from the whole vector
  afterwards, so a capture recorded by a preceding initializer survives an
  inner rebinding of the same name. And a local recursive binding called from a
  lambda created in its own body calls the binding rather than the enclosing
  lambda, which is the shape a depth-first walk takes when its inner iteration
  is `for-each` or `map`.

## Contributors

Gabriel Kahen led the ESKM model I/O work this cycle: the v1 compatibility
corpus, the deterministic fuzz gate and malformed-checkpoint rejection, the
four-engine cross-reader matrix, the subsystem handoff record, and the
engine-parity realignment against the merged dispatch, with further ESKM
hardening carried into this cut.

## Not claimed by this release

- **No StableHLO, PJRT or TPU execution.** The XLA backend provides a
  JAX/StableHLO-*style* API surface and dispatch hierarchy while executing
  through direct LLVM code generation into eleven C runtime entry points and
  calibrated BLAS/GPU libraries. The current execution path does not compile
  through MLIR or StableHLO. Nothing in this release changes that, and no TPU
  training, multi-device production readiness, or new v1.4 synchronization or
  networking capability is claimed.
- **ESKM v2 is a design decision, not a writer.** ESKM v1 remains the default
  and the compatibility baseline; no v2 reader or writer is authorized, and the
  public save APIs continue to emit v1. The proposed VM materialization of
  rank-0 and empty ESKM tensors is likewise unimplemented: native model loading
  can materialize them and VM model loading cannot.
- **The compiler-architecture routing gate's verdict is not asserted here.** See
  the assurance section above; the number belongs to the final battery.
- **VM reclamation is Stage 1.** An escaping object with an out-of-line payload
  (a vector's element array, a bignum's limbs) keeps the arena block that
  payload occupies; escaping cons and closure structure is copied out exactly. A
  continuation captured inside a region pins that region. Objects promoted out
  of a region live in the enclosing arena for its lifetime, which is OALR's
  semantics and equally true natively.
- **Two continuation cases stay out of scope this release**, each with its own
  diagnostic path: a binding established after capture on the VM's
  operand-stack store is refused rather than left ambiguous (SW-61), and a
  non-boxed `set!`-assigned local is rolled back on re-entry on both engines
  pending assignment conversion (SW-62).

## Migration and persistence contracts

**Riemannian Adam requires explicit state on the VM.** The legacy
`riemannian-adam-step` form refuses instead of sharing moments between
unrelated parameters of the same shape. Allocate a separate state for each
parameter and pass the latest returned point into its next update:

```scheme
(define state (make-riemannian-adam-state point))
(set! point
  (riemannian-adam-step! state point gradient learning-rate beta1 beta2 curvature))
```

Points and gradients must satisfy their manifold/tangent contracts. A refused
geometric update leaves optimizer state unchanged. See the
[geometry reference](docs/reference/stdlib/geometry.md).

**Public tensor and model saves use validated ESKM v1.** A single tensor is
stored as one record with an empty name. Both native and VM paths use the
validated readers/writers; this release does not restore the older unchecked
ESKT dispatch. Publication writes a complete temporary checkpoint in the
destination directory and commits it by rename. Handled failures before that
commit preserve the old destination. This is an atomic-replacement contract,
not a power-loss durability guarantee: checkpoint save does not synchronize
both file and parent directory with `fsync`. Abrupt machine loss or `SIGKILL`
cleanup is not promised. See the
[checkpoint contract](docs/design/ATOMIC_CHECKPOINT_SAVES.md).

**`tensor-apply` resolves its callable lexically.** A program that relied on the
old builtin-name whitelist — passing a symbol whose spelling matched a builtin
while a local binding of the same name was in scope — will now call the local
binding. Pass the procedure you mean.

The full change record is in [CHANGELOG.md](CHANGELOG.md). The AD support matrix
records native and VM carrier limits explicitly
([docs/reference/ad/support-matrix.md](docs/reference/ad/support-matrix.md)), and
known limitations are in [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).

## Final verification — pending

<!-- RELEASE_EVIDENCE_PENDING -->
<!-- readiness: fill from final battery -->

The final source commit, platform results, CTest and engine-parity counts,
ICC `v1.3.5-evolve` verdict, and release-package checks have **not yet been
recorded for this cut**. Earlier branch runs do not certify this release.
Replace this section with the actual final battery and artifact receipts
before tagging or publishing. No readiness or test-pass total is asserted here.

---

# Eshkol v1.3.4-evolve — Release Notes

A resident-correctness release. Every correctness gap surfaced by long-duration
resident workloads, and by downstream users, is closed at the architectural root:
automatic memory reclamation matches explicit regions, `parallel-map` is
race-free, gradients are exact through every callable form and every
differentiation point, printed floats round-trip, and the strict type checker
accepts idiomatic dynamic-but-validated code. `gradient` now runs at full parity
on the bytecode VM as well as native codegen, so exact automatic differentiation
is available on every substrate. The high-precision numerics wave — Ozaki-II
exact and reduced-precision GEMM tiers on Metal and CUDA, a mixed-precision
linear solver, and a native 128-bit integer type — lands alongside a Moonlab
v1.2.0 quantum pin, full hosted-VM tensor-matmul parity, and a round-tripping
numeric printer, over a hardened toolchain and a broadened assurance surface.

The second half of the cycle is a consumer-hardening correctness wave, and it
has one organising principle: a wrong answer must not be able to look like a
right one. The change that made the rest possible is that an emitted
compile-time error now prevents artifact emission and execution, where the
compiler previously diagnosed a program and then built and ran it anyway. That
turned answers that had passed unnoticed into build failures, and the wave
that followed corrected them at the root. Exactness is now decided from an operand's runtime tag
rather than from a result's value shape, both on the native flonum
integer-division family and across the bytecode VM's numeric surface.
Automatic differentiation answers exactly at exact points, survives
per-iteration nursery reclamation, and no longer returns zeros above gradient
arity 16. `define-library` and `import` resolve same-unit libraries on all
three back ends, including a VM lane that previously knew none of those forms.
`--shared-lib` links a real, C-ABI-correct shared library instead of exiting
zero with no artifact. Alongside those corrections the release adds a portable
event loop, a fixed-point and `i128` exact-accumulation engine, the qLLM bridge
implementation its documented backward rules had been waiting for, embedding
and Fréchet-mean backward passes, and a release gate that finally reads CTest
results as evidence.

**Release Date**: July 31, 2026

**Release gates** (all measured on the release cut): aggregate suite 45/45
suites and 770 individual tests; CTest 183/183, which as of this release is
itself completion-oracle evidence rather than advice; executable language
coverage 1,091/1,091 (100.0%, floor PASS); SICP full-book gate 88/88 probes
across all five chapters under both `-r` and AOT; reference-Scheme differential
oracle 34/34 AGREE against chibi-scheme 0.12.0; VM parity differential 184/184;
qLLM oracle gate 10/10; ICC readiness 100, verdict `ready`. The VM parity
manifest is 956 rows — 581 `vm-supported`, 44 `native-only-justified`, 331
`gap`, of which 17 are verified behavioral divergences with reproducible
programs under `tests/vm_parity/found/`.

## Highlights

### Automatic memory reclamation matches explicit regions

- **Iter-scope partial reclamation (ESH-0214e) closes the resident-memory
  series.** A resident tick loop that mutates persistent state on every
  iteration — a knowledge base, workspace, or growing list — used to get no
  per-iteration reclamation at all and leaked one iteration's transient garbage
  forever (about 3,366 bytes per tick; roughly 355 MB over 100,000 ticks).
  Such a loop is now lowered with a per-loop nursery region: each iteration's
  allocations land in the nursery, the existing structural write barriers
  promote any persistent-mutation escapee out of the nursery at the store, each
  tail-call back edge promotes the loop-carried out-values and resets the
  nursery, and the loop exit escapes the result and tears the nursery down. The
  same tick loop is now flat at 34 MB — identical to its explicit `with-region`
  twin — with every stored value reading back correct on JIT and AOT. Automatic
  scoping now matches `with-region` in resident loops, so `with-region` is no
  longer required to get flat RSS. A new RSS gate pins the behavior.

  Since v1.3.5, the conditions on that 34 MB figure are stated exactly rather
  than left to inference. It is a 100 000-tick measurement of a fixture that
  *publishes* five freshly consed cells per tick, so what it is flat in is the
  ~3,366 bytes/tick of transient garbage the nursery removed — not in the
  240 bytes/tick the program asks to keep, which grow with tick count on any
  no-GC allocator. A loop that publishes immediates or pre-allocated objects
  instead retains exactly zero bytes per tick, indefinitely. See
  [docs/reference/runtime/memory-model.md](docs/reference/runtime/memory-model.md#what-is-flat-and-what-is-not)
  for the full matrix and the open build item.

### Race-free parallelism

- **`parallel-map` is safe for collection-valued closures.** A closure mapped in
  parallel whose body used per-iteration scope reclamation — an internal
  named-let loop, or a builtin such as `memv` — could return dangling or
  overlapping structure once the input crossed the parallel threshold. On a pool
  worker operating on the shared thread-safe arena, scope reclamation now
  degrades to commit-only (allocations are retained; the shared scope stack is
  never rewound), which is the established correctness-over-throughput fallback.
  Results are now identical to serial `map`. ThreadSanitizer: 73 arena data
  races to 0. New deterministic regression and an ICC gate.

### Exact gradients through every callable form

- **`gradient` recovers callable arity through a function parameter or wrapper.**
  Indirect `(gradient f point)` and curried `((gradient f) point)` forms are now
  byte-identical to the direct call for scalar multi-argument, vector, and
  non-polynomial losses, on both JIT and AOT. The operator recovers the
  callable's arity from its closure metadata instead of assuming a single tensor
  argument. There is no finite-difference fallback anywhere in the gradient path
  — every form is exact reverse-mode AD. A 25-check suite pins the equivalence.
- **Custom-VJP transitive captures now contribute their full sensitivity.** A
  vector-Jacobian-product whose backward closure reaches a captured value
  through an intermediate closure no longer drops that term unreported.
- **`gradient` now runs on the bytecode VM at full parity.** Forward/reverse-mode
  `gradient` — direct, through a callable parameter, and curried — is
  byte-identical to native codegen across the VM's source and bytecode axes, so
  Eshkol is self-differentiating on every substrate. `op:GRADIENT` and
  `op:DERIVATIVE` become `vm-supported` in the parity manifest; higher-order
  nesting (gradient-of-derivative / Taylor tower) stays native-only. The public
  low-level reverse-mode AD tape surface (`ad-pow`, `ad-gradient-of`,
  `ad-value-of`, `ad-tape-length`) is completed on JIT and AOT at the same time.
- **Whole-point tensor-loss gradients are exact.** An arity-1 loss whose body
  applies elementwise arithmetic to its whole vector/tensor argument now
  backpropagates from the sole element for a scalar-valued output and raises a
  clean `jacobian` diagnostic for a vector-valued one, instead of dropping the
  tangent or dereferencing a non-AD value.
- **Differentiation points are classified by runtime value, not syntax.** A point
  that is a variable bound to a vector, a general expression, or a `(the …)`
  wrapper is now routed exactly like the identical literal — closing an
  externally reported `hessian` failure and an externally reported unreported-
  wrong `gradient` at cons-routed vector/list points, and the residual
  `hessian`/`laplacian` variable-bound case behind them.
- **Reverse-mode training loops stay flat under `with-region`.** The AD tape's
  node-pointer array now grows from the tape's owning arena, so it is reclaimed
  with the region at `region_pop` instead of accreting residual memory each step.

### A diagnosed program no longer builds and runs

- **An emitted error diagnostic prevents artifact emission and execution.** A
  compile-time `ERROR: …` now stops emission, linking and execution outright,
  where the compiler previously printed the diagnostic and built and ran the
  binary anyway. Reporting an error is one call at any of 805 sites; propagating
  one is a return path through every enclosing frame, and the codegen frames
  recover by substituting a placeholder and carrying on. All 805 sites funnel
  through four logging primitives, so an authoritative error state now lives in
  one place and every path contributes to it. This is the mechanism that
  surfaced the rest of this release's corrections: each had already been
  reported at compile time, and honoring every report converted answers that
  had gone unremarked into build failures — several of them fixed in this
  release, which is how they were found.

### Differentiation is exact at exact points, and survives automatic reclamation

- **`derivative`, `gradient` and `hessian` at exact points.** At a rational or
  bignum point these three used to return garbage — an exact number is
  HEAP-tagged, so its data field holds a pointer, and every AD entry point
  reinterpreted that field as a number, differentiating at the object's
  address. One authority per question now dispatches on the runtime tag, and
  refuses a non-numeric point with a catchable type error rather than inventing
  a number for it. With that resolved, the exactness gap behind it closed too:
  at an exact point all five operators now run the same Taylor-tower pass
  `derivative-n` runs, so `(derivative f x)` equals `(derivative-n f x 1)` and
  `(hessian f x)` equals `(derivative-n f x 2)` in value **and** in exactness.
  `(derivative (lambda (x) (* x x x)) 1/3)` is `1/3` with `exact?` true, where
  the operators previously either terminated abnormally or, once that was resolved,
  returned only the nearest `double`. The tier keeps `+ - * /` and
  non-negative-integer `expt` exact and demotes to f64 at the first
  transcendental, per R7RS exactness contagion.
- **Gradients through loop-filled vectors are correct again.** `(gradient f x)`
  now computes the right gradient when `f` fills a vector with `vector-set!`
  inside a loop and then selects a component; before this, the derivative was
  attributed to the last element written, for every element read. Primal values
  stayed exact and stderr was empty, so a row-by-row Jacobian — the standard
  idiom — came out uniform garbage while looking plausible. The per-iteration
  nursery introduced by the reclamation work above resets its arena on each
  back edge, and the write barrier that must promote escapees did not recognise
  a dual number as pointer-carrying, so the tangent was recycled underneath the
  computation. The barrier, the `with-region` escape path and the nursery
  recycle now share one predicate, and tape-retained AD nodes allocate from the
  tape's owning arena.
- **Gradient arity above 16 works, rather than returning zeros.** A gradient of
  a closure reached as a value had its argument spread clamped at 16, so a
  declared arity of 17 to 32 either terminated abnormally, produced an
  unreported all-zero gradient, or raised a type error. The spread is now
  computed correctly and emitted once per module out of
  line, with the arity dispatch done by the closure dispatcher's own runtime
  argument-count switch, so the ceiling of 32 is real. The out-of-line form is
  also smaller than what preceded the widening, which restores compile times on
  Windows, where the linkage the inline spread used cannot be discarded.
- **Embedding and Fréchet-mean backward passes.** Two rules that previously
  refused outright now compute. The embedding adjoint is a scatter-add;
  duplicate indices accumulate and unselected rows are bitwise zero, both
  asserted directly because both have failure modes that stay the right shape
  and magnitude. The weighted Fréchet (Karcher) mean on the Poincaré ball is
  differentiated by implicit differentiation of its optimality condition rather
  than by unrolling the solver.

### Exactness across the numeric tower

- **The bytecode VM kept inexactness.** The VM decided a numeric result's
  domain from its *value* rather than from its operands' tags, so
  `(/ (- 2.0 1.0) (+ 2.0 1.0))` answered the exact `1/3` where native answered
  `0.3333333333333333`: any integral-valued double collapsed to an exact
  integer, in both the runtime constructor and the constant folder.
  Independently, the rational natives truncated a flonum operand into an int64
  numerator, so `(* 0.5493061443340549 1/3)` evaluated `(* 0 1/3)` and answered
  the exact 0. Exactness is now decided from operand tags throughout, and the
  folder folds by the parser's literal exactness flags. Divergent native-vs-VM
  numeric combinations drop from 79 to 20, and the remainder — bignum-over-
  bignum rationals the VM's rational type cannot represent — are answered as a
  correctly-rounded inexact value and recorded as a justified parity row
  rather than left unmarked.
- **Native flonum integer division follows R7RS 6.2.6.** `modulo` and
  `remainder` had no flonum path, so both operands went through the int64
  unpack and the answer was the remainder of two IEEE-754 *bit patterns*
  (`(modulo 5.5 2.0)` returned `6192449487634432`); `remainder` also called the
  C library's round-to-nearest `remainder()`, a different function from
  Scheme's truncated one. `quotient`'s double path narrowed to int64 and packed
  the result exact, so it broke contagion by construction and saturated to a
  single plausible-looking constant past 2^63 — the same constant it returned
  for division by `0.0`. The floor-division family was int64-only, so
  `floor-remainder` disagreed with the `modulo` that R7RS defines it to equal.
  All of these are fixed, a zero divisor now raises uniformly across the
  family including the mixed-bignum route, and `(/ (expt 2 100) 0.0)` is
  `+inf.0` rather than `0`.

### Modules

- **`define-library` and `import` resolve libraries defined in the same file,
  on all three back ends.** `define-library` validated its library name and
  discarded it, so `import` had nothing to consult but a filesystem search that
  can only find a library living in some *other* file — and reported
  `Module 'smoke.v1_3' not found` about a library written one line above.
  R7RS-small 5.6.1 defines a library by its form and lets the forms that follow
  import it, so a unit's own libraries now come first in the resolution order,
  ahead of precompiled stdlib modules and the search path. A library becomes
  resolvable by being processed rather than by living in a particular file, so
  an import placed *above* its `define-library` still fails, now with a
  diagnostic naming the line the library is defined on. The bytecode VM was the
  real gap: it knew none of `define-library`, `import` or `export`, and
  compiled such a program into bytecode that warned about undefined variables
  and then died at run time, while the same file ran on JIT and AOT. Three
  latent VM gaps closed with it, including a `provide` that emitted
  nothing and left a stray `OP_POP` discarding a live value — shifting every
  later binding down a slot.

### New runtime capability

- **A portable event loop.** `make-event-loop`, `event-loop-add-fd!`,
  `event-loop-remove-fd!`, `event-loop-poll`, `event-loop-close` and
  `event-loop-backend` wrap kqueue on macOS and BSD, epoll on Linux and
  Android, and IOCP on Windows, with a fail-closed stub on WebAssembly. Exactly
  one backend is compiled in, and the portable half — handle registry, argument
  validation, generation-tagged handles, result coalescing — is shared by every
  backend and by the bytecode VM, so there is no native/VM parity surface to
  maintain separately. Verified with a pipe round-trip inside its timeout, an
  idle poll that waits its budget and returns rather than hanging, and 1,000
  sequential open/close cycles that never exhaust the descriptor table.
- **A fixed-point and `i128` exact-accumulation engine.** `esk_i128`, a
  parametric `fixed<W,F>` with explicit per-operation rounding, and a
  block-scaled `dot_exact` reduction over an i128 accumulator, for reductions
  that must be bit-exact regardless of summation order. 200 shuffles of 4,096
  elements and 50 matmul contraction orders are byte-identical under the exact
  path, while an f64 control drifts about 1.5e-06 between orderings on the same
  inputs — and the exact path measured *faster* than an f64 double-double
  baseline (13.6 GB/s against 4.3 GB/s), so exactness is not paid for in
  throughput here. `ESHKOL_FIXED_POINT_ENGINE` is ON by default and forced OFF
  on MSVC and ClangCL, which lack the `__int128` ABI it relies on. The
  `eshkol-fixedpoint` shared library is self-contained C11 over libc and libm,
  so it is consumable without the rest of the toolchain.
- **The qLLM bridge is implemented.** Its header declared 17 functions and the
  generated API documentation reported them all documented, but none was
  defined anywhere in the tree — while the bridge's *backward* half had already
  shipped, with exact gradient rules for 11 of 13 tensor AD node types
  compiled into the runtime and unreachable, because nothing ever created a
  node of those types. All 17 are now implemented and record canonical AD tape
  nodes, so those rules run. Where a backward rule cannot differentiate a shape
  exactly the forward refuses rather than recording a node whose gradient would
  be wrong. `ESHKOL_QLLM_ENABLED` is opt-in and OFF by default, and turning it
  on without a discoverable library is a configure-time error rather than a
  unreported no-op.

### High-precision numerics

- **Ozaki-II exact DGEMM** recovers full-f64 `C = A*B` from reduced-precision
  tensor cores, certified against an independent CPU f64 reference and against
  the native 128-bit integer path. A CUDA INT8 (IMMA) tier and a Metal
  reduced-precision fully-GPU fast tier are both opt-in and default off; on CUDA
  the INT8 path is engaged only when a measured crossover and an accuracy-budget
  selector show it beating native `cublasDgemm` (externally contributed).
- **`linear-solve`** is a full-f64 dense solver with a mixed-precision
  iterative-refinement fast path that certifies a full-f64 residual and falls
  back to a plain-f64 LU when it cannot — correctness is guaranteed, the speedup
  is opportunistic.
- **Native 128-bit integer type `i128`** — a first-class fixed-width wrapping
  signed integer off the numeric tower, with a full builtin surface on both the
  native and VM paths that compute bit-identical results.

### Types

- **Checked `(the <type> expr)` ascription** asserts a type to the checker as a
  trusted assertion and is a pure runtime no-op (byte-identical IR).
- **Predicate-guarded narrowing** across `if` and `and` for eight type
  predicates, cancelled at `set!`; **sum-type annotations honored on named-let
  parameters**; a **numeric-tower join** gives recursive accumulators their
  least-upper-bound numeric type. Nine new type-system tests including negative
  soundness cases.
- **Linear `Qubit` type** with use-exactly-once enforcement on `define`d linear
  parameters, giving quantum registers a no-cloning guarantee at the type level.
  The linear checker's scope handling and conditional-branch counting are
  hardened so that leaving a scope clears its linear bindings and a linear value
  used in exactly one branch of an `if`/`cond` is accepted (externally
  contributed).

### Printer

- **Shortest-round-trip numeric printing (R7RS 6.2.6).** `display`, `write`, and
  `number->string` emit the shortest decimal that reads back as the identical
  `double`; integral doubles keep their no-`.0` form; native and VM output is
  byte-identical through one shared portable-C routine. `(sqrt 2.0)` prints
  `1.4142135623730951`.

### VM parity and quantum

- **Reverse/forward-mode `gradient` reaches full VM parity** (see *Exact
  gradients through every callable form* above), so the parity manifest gains
  `op:GRADIENT` and `op:DERIVATIVE` as `vm-supported` and the hosted VM is
  self-differentiating; higher-order nesting stays native-only.
- **Hosted-VM tensor matmul parity** is complete: `arange` in 1/2/3-argument
  forms, nested-literal tensor operands, and multi-dimensional `tensor-ref` /
  `tensor-set!`.
- **Moonlab pinned to v1.2.0** adds `vqe_compute_qgt` (quantum geometric tensor
  / quantum natural gradient) and a smooth first-principles H2/LiH
  potential-energy surface; the H2 equilibrium oracle at 0.735 Å is updated to
  `-1.142200155381` Ha. Differentiable quantum-chemistry examples and an
  arbitrary-order-AD H2 vibrational-frequency example ship with the release.

### Toolchain and packaging

- **`--shared-lib` links a real, C-ABI-correct shared library.** The documented
  flag raised `--compile-only`, wrote an object and a bitcode file, and exited
  zero with no library anywhere. Making it link exposed why it could never have
  worked as documented: LLVM's calling convention for a first-class-struct
  return is not the platform C calling convention for the same struct, so a C
  caller compiled against the public header read the tagged value's flags byte
  as the payload. Library-mode codegen now emits a platform-C-ABI thunk per
  exported top-level function — `[2 x i64]` on AArch64, x86-64 SysV, riscv64,
  ppc64le and loongarch64, `sret` plus by-pointer on Windows x64, and a
  diagnostic refusal on 32-bit targets, which neither shape models. The
  relocatable `--shared-lib -c` object, which other Eshkol modules call with
  the internal convention, stays unwrapped. Separately, the runtime archives
  are now position-independent: a `thread_local` compiled non-PIC took the
  local-exec TLS model, so on ELF hosts the library could not be produced at
  all.
- **The browser WASM glue is complete, and the checked-in artifacts are
  current.** `eshkol_write_value`, `eshkol_write_value_to_port` and
  `eshkol_builtin_arena_used` had no `env` import stubs in the browser glue,
  so programs reaching them failed to instantiate; the gap was verified by
  compiling a program that reaches those symbols and diffing the module's real
  imports against the glue, rather than by trusting the import scanner. Both
  checked-in artifacts (`eshkol-site.wasm`, `eshkol-vm.wasm`) are regenerated
  from current source through their canonical recipes, closing WASM
  differential failures against the release tree.
- **Transitive FFI-link discovery, with fatal link failures under `-r`.** Native
  agent-FFI link requirements now propagate through the full transitive
  `(load …)` / `(import …)` / `(require …)` source closure, so a dependency
  reached only indirectly is linked instead of failing the native link with
  unresolved symbols; and a generated-program link failure under `-r` is now
  fatal (nonzero exit) instead of being masked, unreported, by a reduced
  in-process fallback that exited zero.
- **Homebrew-compatible builds.** Every bundled agent-FFI dependency resolves
  without a live download, so `brew install` works, while the default developer
  and release builds stay byte-for-byte unchanged. Pre-existing packaging gaps
  that left the keg non-functional (missing runtime and agent-FFI archives,
  misplaced module sources) are closed, and the release auto-bump anchors its
  substitutions so the new dependency pins survive a version bump.

### Assurance

- **Dynamic edge coverage for the v1.3.4 surface.** A seeded, bounded,
  depth-parametric generator and runner reconcile the generative/adversarial
  harnesses with every new-feature family — nursery iter-scope loops, capturing
  `parallel-map`, exact gradient through callable/curried forms, `i128`
  boundaries, native tensor/`matmul`, the round-trip printer, and the low-level
  AD tape — each probe self-checking against a computed ground truth across JIT,
  AOT-O0, AOT-O2, and the VM, and gated in ICC.
- **Coverage-floor hardening.** The executable language-surface manifest is
  regenerated for the new VM special forms and the coverage floor is enforced
  across the new-feature families, so a newly public surface cannot ship without
  executable evidence. The surface and the floor are both **1,091**, at
  **1,091/1,091 (100.0%) execution-backed** coverage: a construct earns its row
  only by dispatching or executing in a passing run, lexical name-presence is a
  diagnostic that earns no release credit, and the monotonic deficit ledger
  refuses to record a larger deficit without an explicit override.
- **WASM execute-and-diff differential lane.** A new lane builds the VM
  WebAssembly module from current source, executes the VM-supported corpus under
  Node, and byte-diffs its output against native `eshkol-run -r`, so WASM output
  is now gated against native rather than only checked for a valid binary;
  divergences are tracked per file (EXCLUDED / XFAIL, with an unexpected match
  failing the gate).
- **CTest results are release-gate evidence.** No completion-oracle criterion
  consumed a CTest result at all — the test-evidence criteria were index-level
  ("the tests exist and are runnable"), so a red CTest run could not turn the
  release gate red, and a pillar could ship with a perfectly good CTest gate
  that the target judging the cut never looked at. `scripts/run_ctest_gate.sh`
  now emits per-test, per-group and whole-suite trace events, and a group whose
  regex matches no configured test is reported ABSENT and **fails** the gate,
  so a pillar cannot quietly stop being covered because its tests were renamed
  or configured out. Eight criteria are wired, covering the fixed-point engine,
  the exact-input AD identity tier, the runtime-closure arity spread, same-unit
  `define-library`, the self-checking VM surface suite, VM parity and the event
  loop.
- **Display-only tests assert.** A recurring test shape — `display` the
  computed values, print an unconditional pass banner, and leave the expected
  values in a comment — meant the harness saw exit 0 and the comparison the
  comment described was never written. The first wave of that sweep is in this
  release; each test it turns red exposes a correctness gap that had gone
  unmeasured.
- **Two harness issues that produced false verdicts are corrected.** The
  toolchain-fingerprint guard tried the BSD `stat -f` format before the GNU
  `stat -c` one; on GNU coreutils `-f` means `--file-system`, so the
  "fingerprint" was free-block and inode counters and every green Linux run was
  declared `INVALID RUN` (exit 3). And the stale-directory prune globbed an
  unmatched pattern into `du`, which under `set -euo pipefail` terminated the
  calling suite unreported — which is what made the language-coverage floor read
  as a false red when it was in fact green.
- **The five-way surface baseline is re-anchored.** The P8 axis-6 ratchet is
  shrink-only, and current master produces the same disagreement count with a
  different set — four AD entries resolved, four region-handle entries appeared
  — which a shrink-only ratchet cannot absorb. The four new entries record a
  genuine, pre-existing cross-backend naming asymmetry between the VM's
  `_region-open` and the native `region-open`; it stays open as a tracked build
  item rather than being resolved by editing the baseline.

---

# Eshkol v1.3.3-evolve — Release Notes

Windows ARM64 packages now use a single platform-correct JIT target contract
for both live LLJIT compilation and the persistent stdlib object cache. This
avoids LLVM 21's invalid AArch64-COFF Large-model SEH metadata while preserving
external call reach through RuntimeDyld stubs and full host-data reach through
nearby COFF import-address cells. A co-located per-object arena keeps JIT-owned
code, read-only data, and writable data inside Small-model relocation reach;
explicit Branch26 and ADRP span guards fail safely if a future object outgrows
that contract. This prevents layout-sensitive `PAGEBASE_REL21` truncation
without weakening stack probing, exceptions, or cacheability.
Windows packages also publish and explicitly register the Taylor-tower AD state
globals required by relocated cache-disabled JIT modules.
Generic release stdlib generation now uses the common 128-bit x86-64/AArch64
tensor-vector baseline regardless of the hosted runner's AVX width; the release
validator rejects wider fixed vectors as well as scalable or optional-ISA IR.
CUDA-labeled artifacts now fail closed unless the real NVIDIA backend is
compiled. Linux x64/ARM64-SBSA and Windows x64 jobs install pinned CUDA 12.4
toolchains and verify that `nvcc`, cuBLAS, the CUDA runtime, and the CUDA source
graph are present. The unsupported Windows ARM64 CUDA label has been removed
rather than shipping its CPU fallback under a GPU name. Explicit
`sm_72/75/80/86/89/90` code preserves Xavier through current RTX coverage.
CUDA 12 builds on newer GNU hosts now require a compatible compiler for the
whole build, avoiding unsafe nvcc-only host overrides that mix libstdc++ ABI
and library search paths. Unix workflow configuration uses a scalar toolkit
hint so non-CUDA macOS lanes remain compatible with Bash 3.2 under `set -u`.
Generated AOT and persistent-cache links now resolve CUDA runtime/cuBLAS names
from the consumer's explicit toolkit roots, `nvcc`, and standard multiarch
layouts instead of replaying hosted-runner absolute paths. Linux links require
the configured CUDA ABI-major sonames, so CUDA 12 artifacts fail closed rather
than substituting CUDA 13 unreported. Windows uses native shell-free driver paths
instead of MSVC STL generic-path conversion, keeping generated links compatible
with consumer Visual C++ import libraries that predate `__std_replace_copy_2`.
Its CUDA 12.4 setup also requests only documented Windows subpackages; `nvcc`
provides its compiler internals without invalid standalone `crt`/`nvvm` names.
The Windows CUDA lane uses Ninja Multi-Config so Eshkol C/C++ remains LLVM 21
ClangCL while `nvcc` uses the CUDA-supported v142 MSVC host, avoiding the
Visual Studio CUDA-target `MSB4023` metadata failure without changing the
release package layout. Backend validation reads the complete generated
multi-config Ninja graph before any CUDA-labeled build proceeds. The v142
host compiler cache entry uses CMake's forward-slash path form so nvcc receives
an intact `-ccbin` value even when Visual Studio is installed below a path with
spaces.

**Release Date**: July 16, 2026

Eshkol v1.3.3-evolve combines the completed Moonlab quantum stack with a
language-wide correctness and evidence campaign. Every declared language
surface now has deterministic executable evidence, native and VM behavior is
cross-checked, and the release is green across the aggregate,
full-book SICP, external-reference, generative, WebAssembly, CTest, and ICC
architecture gates. Full technical detail lives in
[CHANGELOG.md](CHANGELOG.md).

**Release gates**: 44/44 suites and 716/716 tests; CTest 76/76; SICP 88/88
JIT+AOT probes; Chibi Scheme 34/34 AGREE; five-oracle generative differential
127 programs with zero divergences; VM parity 68/68; VM extended surface
53/53; executable language coverage 1057/1057 (100%); WebAssembly imports
101/101 provided; Taylor monomorphization equivalence 441/441 under both JIT
and AOT; ICC architecture model 8/8; ICC readiness 100/100.

## Highlights

### Quantum computing, differentiation, and post-quantum cryptography

- **S1 — honest quantum execution.** `agent.quantum` exposes state creation,
  gates, measurement, and randomness routed through Moonlab's Bell-verified
  QRNG. The Bell smoke is perfectly correlated at 200/200 shots. (#261)
- **S2 — VQE.** Hamiltonian construction, exact energy, variational energy,
  optimization, and native gradients are public builtins. H2 converges to its
  exact ground-state energy within `4.4e-16`. (#268)
- **S3 — differentiate through circuits.** `AD_NODE_CUSTOM` adds a general
  custom vector-Jacobian-product node to Eshkol's reverse tape. VQE energy is
  composably differentiable inside ordinary Eshkol functions and agrees with
  Moonlab's adjoint and finite differences. (#270)
- **S4 — ML-KEM.** FIPS 203 ML-KEM 512/768/1024 key generation,
  encapsulation, and decapsulation are available through `agent.pqc`, with
  Bell-verified QRNG seeding and NIST KAT/round-trip checks. (#272)
- **S5 — permanent evidence.** A hosted macOS quantum lane runs the complete
  quantum suite. A 16K-shot CHSH gate measures `S ~= 2.86`, beyond the
  classical bound, and quantum behavior is wired into the AD adversarial
  oracle, executable coverage manifest, and ICC architecture model. (#273)

Quantum remains build-gated with `-DESHKOL_QUANTUM_ENABLED=ON`; the default
build remains dependency-light. The pinned Moonlab revision includes the
upstream macOS weak-import linker fix and builds without a local override.

### Complete executable language coverage

- The coverage manifest now has **1,057/1,057 executable rows**. Credit
  requires a deterministic reachable execution trace; tokens, declarations,
  and dead code do not count.
- Native and bytecode backends agree on **68/68 parity probes**, including
  multiple values, vectors, closures, parameters, include forms, numeric
  boundaries, full datum read/write round trips, and ESKB execution.
  Unsupported features fail explicitly rather than fabricating values.
- The formerly dormant hosted VM surface is now executed by **53/53 tests**,
  covering I/O, system, image, polling, process, concurrency, crypto, format,
  networking, and layout operations.
- `make-parameter`/`parameterize` now use real dynamic parameter objects on
  native, VM, and WebAssembly-hosted paths, including converter-once and
  unwind-safe dynamic extent. (#271)

### Correctness and production hardening

- Poincare-ball exp/log behavior now uses the correct Riemannian tangent norm
  at the base point, with off-origin analytic length and round-trip checks.
- Complete R7RS multiple-value semantics now agree across JIT, AOT, and VM.
- Hosted port rebinding/lifecycle, exact 64-bit `current-jiffy`, proper-list
  `directory-walk`, and image-buffer ownership now behave correctly.
- Tail-call library exports can no longer be stripped at O2; rational/bignum
  region evacuation no longer leaves dangling interior pointers; the Windows
  lite harness reports compile timeouts honestly. (#265)
- Large-list sort is now a stable bottom-up vector merge sort: the two-million
  element stress case drops from roughly 32 GB peak RSS to about 362 MiB.
  (#266)
- The GPU correctness gate now runs on Windows Git Bash/MSYS hosts, where it
  previously skipped them unreported. It uses the supported official-SDK ClangCL + Ninja
  compiler path with MSVC as nvcc's host compiler; a real RTX 3060 dispatched
  through CUDA cuBLAS and matched 10 CPU-reference probes with maximum relative
  difference `0`. PE/COFF hosted-runtime linkage no longer relies on ELF weak
  functions.
- Release and verification shell entry points now reject unsafe build roots,
  symlink escapes, and destructive cleanup targets before operating. The same
  path contract is exercised across Bash, Git Bash, and constrained ARM64
  shells. (#278)
- Generated Linux AOT executables now preserve the runtime search paths of
  every linked dependency, including indirect LLVM, C++ runtime, curl,
  SQLite, ncurses, OpenSSL, and Nix-store libraries. The linker derives these
  paths from actual `-L` and shared-library inputs and from the selected host
  compiler instead of requiring a custom `LD_LIBRARY_PATH`. (#279)
- Linux binary archives now carry a hashed, licensed libpng/libjpeg/libwebp/
  zlib runtime closure under `lib/eshkol/runtime-deps`. Both packaged tools and
  generated run-cache/AOT executables resolve that relocatable closure, so
  image I/O remains enabled without requiring matching codec development
  packages—or the release builder's absolute library paths—on the target host.
- Precompiled `stdlib.o` and `stdlib.bc` release artifacts are optimized at O2
  for LLVM's generic architecture baseline instead of the transient runner's
  CPU. User and JIT compilation still specialize for the target host, while a
  release-time LLVM-disassembly gate rejects SVE/SVE2 and other builder-only
  wide-vector IR before packaging. This keeps ARM64 archives runnable on
  baseline ARMv8 systems such as Cortex-A72 rather than only on SVE builders.
- Exact numeric and automatic-differentiation corrections cover bignums,
  rationals, tensors, forward-over-reverse composition, Hessians, and
  explicit unsupported-op errors instead of unreported zero gradients.

### Release integrity

The full suite uses an isolated requested build directory; reference and
generative oracles execute portably on macOS; freestanding object checks allow
only target intrinsics (never undeclared hosted ABI calls); every required
WebAssembly environment import is present in both JavaScript runtimes; and the
tag workflow supports a non-publishing manual dry run of the complete release
matrix before any immutable release tag is created. Installed package module
resolution is also exercised with the persistent cache disabled: agent modules
must resolve from the executable-relative source tree, and a missing explicit
`require` is a hard failure rather than a diagnostic followed by execution.
Native Windows hosts retain that bounded JIT ABI through the generated PE
export table without mixing static LLVM target archives with `LLVM-C.dll`.
Generated Windows AOT/cache links also resolve the selected consumer LLVM 21
toolchain's architecture-matched compiler-rt builtins archive at runtime, so
128-bit bignum/rational division links on both x64 and ARM64 without embedding
the release runner's absolute SDK path.

---

# Eshkol v1.3.2-evolve — Release Notes

**Release Date**: July 9, 2026

Eshkol v1.3.2-evolve is an evolve point release over v1.3.1-evolve. It closes
the last resident-memory correctness gap for long-running loops, makes region
scoping safe under parallelism, completes the automatic-differentiation
`input2` gradient path, and adds developer tooling and Binary Lambda Calculus
depth. Full technical detail lives in [CHANGELOG.md](CHANGELOG.md); this page is
the user-facing summary.

**Release gates**: builds on the v1.3.1-evolve gates with a new poison-hardened
region-evacuator coverage gate
(`tests/memory/region_evac_subtype_coverage_test.sh`) that promotes and reads
back logic/workspace state over 1,000,000 region-wrapped mutations under
`ESHKOL_ARENA_POISON=1` at flat RSS.

## Highlights

### Resident-memory correctness

- **Forever-flat loops that mutate persistent logic/workspace state.** The
  region escape evacuator now deep-walks the `SUBSTITUTION`, `FACT`,
  `KNOWLEDGE_BASE`, `FACTOR_GRAPH`, and `WORKSPACE` subtypes (ESH-0214d). A
  resident tick loop can wrap its body in `(with-region ...)` to reclaim
  per-iteration transient garbage while its escaping knowledge-base/workspace
  state is promoted intact — previously those subtypes were shallow-copied and
  dangled into the freed arena. (#226)
- **Region scoping is thread-safe.** `parallel-map` combined with `with-region`
  no longer races on the shared current-arena slot. (#217)

### Automatic differentiation

- **`input2` gradients complete for `conv2d`/`batchnorm`/`layernorm`/
  `attention`** — the second operand (kernel / gamma / K / V) now receives
  gradients. (#212)

### Tooling and language

- **`eshkol-doc`** generates an API reference from Doxygen comments. (#213)
- **Binary Lambda Calculus universal machine**: `(blc-U)` decodes and runs
  Tromp's 232-bit self-interpreter; BLC8 byte I/O and ASCII lambda diagrams
  round out `core.blc`. (#218)

### Robustness

- Three deferred latent issues triaged and closed: ESH-0223, ESH-0227, ESH-0228. (#215)

---

# Eshkol v1.3.1-evolve — Release Notes

**Release Date**: July 9, 2026

Eshkol v1.3.1-evolve is a resident-robustness point release over
v1.3.0-evolve: two fixes aimed squarely at long-running/daemon processes and
large-persisted-state workloads, plus a comprehensive documentation pass.
Full technical detail lives in [CHANGELOG.md](CHANGELOG.md); this page is the
user-facing summary.

**Release gates**: builds on the v1.3.0-evolve release gates (see below) with
a new AOT flat-RSS regression gate
(`tests/memory/define_loop_flat_rss_aot_test.sh`) that compiles the
guard-wrapped self-tail-recursive `define`-loop shape ahead-of-time and fails
if peak RSS exceeds a generous flat threshold, so a regression in the ESH-0214b
fix cannot pass unreported.

## Highlights

### Resident-robustness fixes

- **Long persisted-state files now read safely.** The reader's list parser
  (`read_list`) was rewritten from per-element native recursion to an
  iterative loop, so reading a long flat list — e.g. a 46K-entry persisted
  state file — no longer overflows the native stack. Verified: the pre-fix
  reader raised SIGBUS at 20M elements; post-fix, the same input reads
  cleanly. (#191)
- **`define`-loop daemons now hold flat memory.** Automatic per-iteration
  arena-scope reclamation — previously limited to named-let loops — now also
  covers self-tail-recursive top-level `define` loops, and the escape
  analysis that gates it accepts a catch-all `guard` clause instead of
  rejecting any guarded body outright. This is exactly the shape of a
  production daemon/resident loop: a top-level `define` loop wrapped in an
  error boundary. Verified in AOT mode: a 1,000,000-iteration allocating
  guard-wrapped `define` loop holds peak RSS at 27MB with the fix on, versus
  2608MB with the fix off. (#192)

### Comprehensive documentation pass

- Doxygen doc-comments added across all 64 public headers (`inc/eshkol/**`)
  and most implementation files (`lib/**`).
- A new navigable documentation index (`docs/README.md`); orphaned
  (unindexed) docs reduced from 73 to 3.
- Press materials and website content updated to reflect the shipped v1.3
  state; roadmap views aligned with what has actually shipped.

### Known issues

See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) and the CHANGELOG's Known
Issues section for the current, itemized list (none block ordinary use).

---

## Previous Releases

### Eshkol v1.3.0-evolve — Arbitrary-Order Automatic Differentiation

**Release Date**: July 7, 2026

Eshkol v1.3.0-evolve is the "evolve" release: arbitrary-order automatic
differentiation, full R7RS conformance on the portable differential
corpus, and a hardening pass across closures, tail calls, and long-running
processes.

**Release gates** (all green on the release SHA): ICC readiness oracle
`v1.3-evolve` ready (100/100, trace-verified); CI 14/14 lanes including
windows-arm64 lite/CUDA/XLA; SICP full-book gate 88/88 probes across all 5
chapters under both `-r` and AOT; reference-Scheme differential oracle 34/34
AGREE vs. chibi-scheme.

#### Highlights

##### Automatic differentiation, best-in-class and beyond

Eshkol's AD system already did exact forward-mode, reverse-mode, and
symbolic differentiation. v1.3.0-evolve adds a second axis on top: **order**.
A new Taylor-tower engine (13 phases, P0 through P12 — see the
[Automatic Differentiation guide](docs/guide/AUTOMATIC_DIFFERENTIATION.md)
and the [CHANGELOG](CHANGELOG.md#added--automatic-differentiation-taylor-tower-campaign-p0-p12)
for the full phase-by-phase breakdown) computes *every* derivative up to an
arbitrary order `k` in a single pass:

- **Arbitrary order** — `(taylor f x k)` and `(derivative-n f x k)` return
  the full coefficient series or the `k`-th derivative for any `k`, not just
  first/second order.
- **Exact, not approximate** — when the input is an exact number and the
  function only uses exact-preserving arithmetic, the coefficients come back
  as exact arbitrary-precision (bignum/rational) values instead of `double`
  approximations. Most autodiff systems, JAX included, only ever produce
  floating-point derivatives; Eshkol can hand you the exact rational
  derivative when the math supports it.
- **Validated** — Taylor models (`taylor-model`, `tm-range`, `tm-eval`) pair
  the polynomial with a rigorous interval-remainder bound, giving a
  *provable* enclosure of a function's range or value, not just a point
  estimate. This is a step beyond what mainstream AD/ML frameworks expose to
  user code at all.
- **Multivariate and sparse** — `mixed-partial`/`gradient-n` recover
  arbitrary-order mixed partials via a Griewank-Utke-Walther (GUW)
  propagation layer; `sparse-hessian`/`sparse-mixed-partials` exploit sparsity
  with graph-coloring so the cost scales with variable-interaction bandwidth,
  not dimension.
- **Composes with everything else** — tensor-valued towers differentiate
  through `matmul`/`conv2d`/`sigmoid`/`tanh`; reverse-over-Taylor lets
  `gradient` differentiate through a `derivative-n` call; checkpointed
  reverse-mode keeps the memory cost of high-order reverse AD sub-linear;
  towers work correctly through `if`/`cond`/named-let/recursion; and
  tower-based numerics (`taylor-ode-solve`, `taylor-root`,
  `taylor-inverse-series`) put all of this to work solving ODEs, root-finding,
  and series inversion.

See the [Automatic Differentiation guide](docs/guide/AUTOMATIC_DIFFERENTIATION.md)
for worked examples and the full API reference.

##### Full R7RS conformance on the portable corpus

A new reference-Scheme differential oracle diffs Eshkol's behavior against
chibi-scheme 0.12.0 across a 34-program portable R7RS corpus (numeric, list,
vector, string, char, binding, control-flow, equality, and I/O). It started
this release cycle at 27/34 (79.4%) and every divergence is now fixed:
**34/34 (100%) AGREE** with chibi-scheme on that corpus. Fixed along the way:
`apply` with leading arguments, multi-vector `vector-map`/`vector-for-each`,
quasiquoted vector literals, `cond`/`case` `=>` arrow clauses, an allocating
`vector-copy` (including on `#(...)` tensor-backed literals), the
`error-object?`/`error-object-message`/`error-object-irritants` condition
family, R7RS string-escaping in `write`, nested ellipsis (`x ... ...`) in
`syntax-rules`, and the 2-argument form of `substring`. See the
[CHANGELOG](CHANGELOG.md#fixed--r7rs-conformance) for the itemized list.

##### Robustness: closures, tail calls, and long-running processes

A cluster of fixes targets programs that run for a long time or recurse
deeply — the kind of correctness property that only shows up in production,
not in a quick test:

- Mutual tail calls (`even?`/`odd?`-style cross-function recursion) are now
  proper O(1)-stack R7RS tail calls on AArch64.
- Named-let loops are tail-call-optimized in every legal tail position, not
  just the immediate loop body — including tail calls made through a `guard`
  error-boundary wrapper.
- Curried closures can now capture up to 64 variables (up from a ceiling of
  16 that corrupted memory unreported past that point).
- A production-triggered class of unbounded RSS growth in long-running loops
  is closed, and loops that are provably safe now get automatic,
  zero-annotation per-iteration memory reclamation.
- A graceful-shutdown race that could SIGSEGV after `SIGTERM` is closed, deep
  recursion overflow now fails with a diagnostic instead of an unexplained
  `SIGILL`, and `eshkol-run -r`/AOT caching now correctly invalidates when an
  indirectly loaded/required dependency changes.

See [CHANGELOG.md](CHANGELOG.md#fixed--compiler--runtime-robustness) for the
full list with root causes.

##### Also in this release

- A new build integration surface: `--emit-depfile` plus a canonical
  `cmake/EshkolCompile.cmake` for consumers embedding the Eshkol compiler in
  their own CMake build.
- Browser REPL / WASM fixes so every example on [eshkol.ai](https://eshkol.ai)
  runs, including the tensor computing examples.
- A permanent, ICC-wired adversarial-testing infrastructure — differential,
  edge-matrix, AD-oracle, stress, VM-parity, depth-parametric, and external
  (reference-Scheme / sanitizer-fuzz / metamorphic) test pillars — so these
  classes of correctness issue keep getting caught going forward. See
  [docs/TESTING.md](docs/TESTING.md).

##### Known issues

See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) and the CHANGELOG's Known
Issues section for the current, itemized list (none block ordinary use).

### Eshkol v1.2.3-scale — Platform Artifact Closeout

**Base Release Date**: May 1, 2026
**Closeout Date**: May 20, 2026
**Platform Artifact Date**: May 25, 2026

Eshkol v1.2.3-scale is the platform-artifact closeout point release for
v1.2.0-scale, the *production-readiness* release. The v1.1
line proved the math (autodiff, tensors, the consciousness engine);
v1.2 makes it shippable: trained models save and load, error messages
point at the actual line, the Python FFI is stable and zero-copy,
deep recursion doesn't blow the stack on Darwin, and a long tail of
correctness/security issues that surfaced under real workloads is now
closed.

The headline addition isn't a feature — it's the edge-case regression
suite that catches every fix in this release going forward. The
v1.2.0 release shipped with 62 tests; the current v1.2.x Noesis M0
closeout build carries **87 passing edge/security tests**, a clean
37-suite aggregate gate, and a full Noesis aggregate smoke pass.

## v1.2.3 Platform Artifact Addendum (May 25, 2026)

`v1.2.3-scale` is a packaging and release-integrity patch over the v1.2.1
language/runtime surface. It supersedes the unpublished `v1.2.2-scale` tag
attempt by adding the hosted Windows x64 COFF linker fix needed for the full
artifact matrix:

- the release workflow now treats the 16-package platform set as a checked
  contract before publishing:
  - Linux x64/ARM64 lite/XLA/CUDA tarballs
  - macOS arm64/x64 lite/XLA tarballs
  - Windows x64/ARM64 lite/XLA/CUDA zips
- `SHA256SUMS.txt` is generated from the final merged `dist/` directory.
- the publish job refuses to append to or overwrite an existing GitHub release.
- `release_workflow_surface_test` pins this behavior in CTest so future
  release-workflow edits cannot drop platform artifacts unreported.
- generated parallel worker initializer symbols are module-local on native
  Windows so hosted x64 release packages link cleanly against `stdlib.o`.
- the Homebrew formula template now targets the public `v1.2.3-scale` archive;
  the tap formula still needs its computed SHA256 after the release tarball is
  published.

## v1.2.1 Noesis M0 Closeout Addendum (May 20, 2026)

`v1.2.1-scale` closed the Noesis M0 audit path:

- `tests/v1_2_edge_cases` passes **87/87**, including shared
  hash-table mutation under `parallel-map`, late variadic REPL forward
  refs, binary I/O, match predicate binding, tensor pixel fill,
  first-class builtins, channels, threads, object-build CLI contract
  coverage, bounded HTTP server smoke coverage, and shell-level CLI/linker
  probes.
- `scripts/run_all_tests.sh` passes **37/37 suites** and **528/528
  self-reported individual tests**. The aggregate now counts suites that
  report `Results: N passed, M failed`, so the logic and v1.2 edge/security
  runners are included in the release total.
- Noesis `tests/smoke/all.esk` passes with `NOESIS_ALL_RC=0`.
- VM C API checks pass **81/81**, CTest passes **15/15**, and stress tests
  pass **3/3** on the final release-gate build.
- The previously intermittent dual-neural failure is resolved by
  serializing runtime hash-table access; the focused Noesis
  `dual_neural` smoke passed 8/8 stress repeats on the corrected build.
- LL's underlying CLI behavior is corrected: `--emit-object` accepts
  compatibility flags, writes the requested `-o path`, and no longer
  creates the stale `.o.o` output.
- The Homebrew formula template points at the public `v1.2.1-scale` release
  archive; the public tap formula carries the computed SHA256 after tagging.
- The release workflow's platform asset matrix is now guarded by
  `release_workflow_surface_test`: every `v*` tag must publish the 16 expected
  Linux/macOS/Windows lite/XLA/CUDA archives, generate `SHA256SUMS.txt`, and
  refuse to append assets to an existing release.

## What's New in v1.2.0-scale

### Production Deployment

- **Model serialization** — `.eshkol-model` is an ESKB-extended
  binary format that round-trips trained networks (architecture +
  weights + metadata) so you can save a model on the training box
  and load it on the inference box.
- **Stable C ABI + Python bindings** — `inc/eshkol/c_abi.h` is the
  versioned public header; `pip install eshkol` gives you pybind11
  bindings with NumPy zero-copy interop.  Gradient computation,
  structured returns, and error recovery all crossed the FFI cleanly
  after the v1.2 hardening pass.
- **Per-thread arenas** — concurrent code paths
  (parallel-map workers, thread-pool tasks) now use thread-local
  arena slots so allocation in one worker never stomps another's.
- **Image I/O** — PNG/JPEG/WebP/BMP read/write/resize for vision
  pipelines, backed by native platform/system codec APIs
  (ImageIO/CoreGraphics on macOS, system libpng/libjpeg/libwebp on
  Linux, GDI+ on Windows).
- **Plotting stdlib** — inline matplotlib-style charts via PNG output
  for notebook-style workflows.

### Compiler Diagnostics

- **Actionable error messages** — compile errors now report the
  exact source line and column with a caret underline:
  ```
  /path/to/file.esk:6:4: error: Unknown function: undefined-fn
      6 |   (undefined-fn 1 2 3))
        |    ^
  ```
  Previously every error pointed at line 1 because the parser's
  comment-stripping reader consumed newlines.  The fix preserves
  newlines from comments and threads a cumulative file-line counter
  across `eshkol_parse_next_ast_from_stream` calls.

### Stdlib + Language

- **JSON Schema validation** (Draft 7 subset) — `json-schema-valid?`
  and `json-schema-validate` for experiment-manifest /
  preregistration enforcement.  Supports type, properties, required,
  additionalProperties, items, min/max length/items, minimum/maximum
  (with exclusive variants), enum, const, pattern (substring),
  oneOf / anyOf / allOf / not.  Auto-loaded via stdlib.
- **R7RS-compliant scoping for stdlib redefines** — user `(define
  (foo …))` after `(require stdlib)` cleanly shadows stdlib's `foo`
  at link time (LinkOnceODR linkage on stdlib functions) and at
  call-site lowering (variadic-info hygiene clears stale entries
  on redefine).  Previously a user redefine of a variadic stdlib
  function with a fixed-arity signature compiled with an
  arity-mismatch warning and terminated abnormally at runtime.
- **AD scalar derivative on inline lambdas** — `(derivative
  (lambda (x) …) point)` inside a wrapper function now correctly
  flows through the runtime closure dispatch.  Previously it
  returned -inf / wrong values because the new-style derivative
  codegen path bailed out without calling the closure for
  function-parameter operands.
- **Reflection** — `procedure-arity`, `record-fields`, `describe`
  for runtime inspection of user-defined procedures and records.
- **Memoization / LRU cache stdlib** — `(memoize fn :lru 256)`.
- **PRNG seeding + deterministic replay** — `(seed-prng! …)` and
  per-stream isolation for reproducible experiments.
- **Lazy sequences / streams** (SRFI 41).
- **Time API** — ISO-8601 parse/format + duration types.
- **Regex capture groups** — `regex-group` / `match-groups`.
- **CLI argument parser** — `(parse-args)` for noesis CLI entry
  points.
- **call-with-values + URL/base64url encoding** finalised.

### Build, Link, and Platform

- **macOS deep recursion** — every binary now ships with
  `LC_MAIN.stacksize = 512 MB` (`-Wl,-stack_size,0x20000000`) on
  Darwin.  The flag had only been wired into one of the two link
  paths in `eshkol-run`; the common compile-and-link path inherited the
  8 MB default unreported, and any non-tail-recursive Scheme code
  hit `eshkol_check_recursion_depth + 4` with a SIGSEGV on its own
  frame push.
- **`--wasm` is self-contained** — the WASM emit path no longer
  falls through to native clang++ link.  `eshkol-run file.esk
  --wasm -o foo.wasm` produces the .wasm via LLVM in-memory codegen
  and exits cleanly; no spurious "_main referenced from
  initial-undefines" link errors.
- **Stdlib functions are weak-linked** — user code can override a
  stdlib symbol with their own definition without a "duplicate
  symbol" link error.  The fix mirrors the Windows
  `WeakAnyLinkage` path onto macOS/Linux LinkOnceODR.
- **AD value-typed captures** — derivatives that close over
  function-parameter `tagged_value` Arguments (e.g. `loss-fn`
  capturing `input`/`target`/`b` in `compute-loss-gradient`) no
  longer fail LLVM IR verification with "PtrToInt source must be
  pointer".
- **CI**: new `linux-x64-asan-ubsan` lane runs the v1.2 edge-case
  suite under `-DESHKOL_ENABLE_ASAN=ON -DESHKOL_ENABLE_UBSAN=ON`.
  `ESHKOL_ENABLE_TSAN=ON` and `ESHKOL_ENABLE_MSAN=ON` are scaffolded;
  TSan/MSan-built libstdc++ on apt.llvm.org is a v1.3 prerequisite.
- **Tagged release assets**: the release workflow treats the complete
  16-package platform set as a checked contract before publishing:
  Linux x64/ARM64 lite/XLA/CUDA tarballs, macOS arm64/x64 lite/XLA tarballs,
  Windows x64/ARM64 lite/XLA/CUDA zips, plus `SHA256SUMS.txt`.
- **`stdlib.o` rebuild correctness** — `file(GLOB_RECURSE …
  CONFIGURE_DEPENDS)` now watches every `lib/{core,math,signal,
  random,web,tensor,quantum,ml}/*.esk` so editing a transitive
  required module triggers a stdlib rebuild.  Previously only edits
  to `lib/stdlib.esk` itself did.

### Hardening

- **CRITICAL**: shell-string injection in `agent_subprocess.c` —
  fixed by switching to `posix_spawn` with `argv` arrays; the
  `popen("sh -c …")` path is gone.
- **CRITICAL**: Python FFI derivative-method AST injection in
  `eshkol_module.cpp` — fixed by canonicalising via the parser
  rather than string-substituting into source.
- **HIGH** (3 items): integer-overflow guards on arena, KB-load,
  and image-IO size computations (`__builtin_mul_overflow`).
- **HIGH** (4 items): path-traversal defence with percent-decode +
  component-check, TOCTOU race fixes on `stat → open`, and a
  Windows-subprocess buffer-size off-by-one.
- **HIGH**: 36 previously error-swallowing sites across the runtime now either
  surface the error or are documented as intentional.
- **MEDIUM**: ReDoS-resistant regex engine (counted-quantifier
  backtracker with bounded-state ceiling), SQL-injection guards on
  the persistence path, URL validator that rejects scheme
  smuggling.

### Testing

- **87-test v1.2 edge/security suite** at `tests/v1_2_edge_cases/`
  covering symbol consistency under gensym, AD tape state across
  worker threads, parser line tracking, stdlib symbol resolution,
  the JSON Schema validator, HTTP server smoke behavior, and every real
  correction in this release.
  Runs under `bash scripts/run_v1_2_edge_cases_tests.sh` (also
  invoked by `run_all_tests.sh`). Includes shell-style tests for compile-time
  diagnostics, CLI/linker probes, REPL protocol checks, and server smoke paths
  that don't fit the `.esk → run → check exit` shape.
- **Master suite EXIT=0** end-to-end across 37 sub-suites and 528
  self-reported tests:
  features, stdlib, list, memory, modules, types, typesystem,
  autodiff, ml, neural, json, system, complex, cpp_type, vm, parser,
  control_flow, logic, bignum, rational, parallel, signal,
  optimization, examples, xla, gpu, error_handling, macros, repl,
  web, tco, io, benchmark, migration, codegen, numeric,
  v1_2_edge_cases.

## Carry-forward to v1.3

- **Native media stack** — use ImageIO/CoreGraphics, system
  libpng/libjpeg/libwebp, and GDI+ on the three host platforms so the
  active image backend does not rely on vendored third-party media
  code.
- **AD `input2` plumbing for non-matmul tensor ops** — the
  backward kernels for conv2d / batchnorm / layernorm / attention
  / multi-head-attention exist; the forward implementations need
  to be rewritten to multi-channel / per-feature shape so the
  Wengert tape can consume them.  Matmul (the only AD-supported
  tensor op exercised by the suite today) is correctly wired.
- **TSan / MSan CI lanes** — pending TSan/MSan-built libstdc++.
- **Spec-doc generator (`eshkol-doc`)** — extract type signatures
  + docstrings from the indexed module graph.
- **True module-private internals** — `(provide …)` is currently
  informational under both AOT and JIT (item Z); v1.3 reintroduces
  a proper rename pass while keeping cross-file calls to provided
  symbols working.
- **AD-1 follow-up** — re-extract `codegenDerivativeMonolith` into
  the new code path; the v1.2 fix delegates from `derivative()`
  to the monolith as a stop-gap.

---

# Eshkol v1.1.13-accelerate — Windows ARM64 + Release Workflow + VM Closure Fixes

**Release Date**: April 9, 2026

Eshkol v1.1.13-accelerate adds native Windows ARM64 support, rewrites the release workflow into a 16-lane build matrix that produces lite/XLA/CUDA variants for every supported platform, corrects two critical bytecode-VM closure-handling gaps that had affected the browser REPL and gradient descent demos, hardens setjmp/longjmp on Windows for both x64 and ARM64, and overhauls the website for full mobile responsiveness.

## What's New in v1.1.13-accelerate

### Windows ARM64 Native Support

- Full build path for Windows ARM64 via VS 2022 + ClangCL + LLVM 21 aarch64 SDK
- CMake auto-detects `clang_rt.builtins-{x86_64|aarch64}.lib` based on `CMAKE_VS_PLATFORM_NAME`
- Multi-arch DIA SDK lookup: scans both `Program Files` and `Program Files (x86)` for both `amd64` and `arm64`
- REPL JIT links the architecture-appropriate LLVM target libraries (`LLVMAArch64*` on ARM64, `LLVMX86*` on x64)
- 16 release artifacts per tag: 6 Windows (x64/arm64 × lite/xla/cuda), 6 Linux, 4 macOS

### setjmp/longjmp Cross-Platform Hardening

- Windows ARM64: uses `Intrinsic::sponentry` as the hidden `_setjmpex` context (matches Clang lowering)
- Windows x64: switched from `Intrinsic::localaddress` to `Intrinsic::frameaddress(0)` for the hidden context — produces stable, correctly-aligned frames
- Removed all compile-time `#ifdef _WIN32` branches in favor of runtime `Triple::isOSWindows()` — proper cross-compilation
- Dynamic `jmp_buf` sizing via `eshkol_jmp_buf_size()` runtime helper (no more hard-coded 256-byte buffers)

### Runtime Symbol Renames

POSIX shim functions are now renamed with an `eshkol_` prefix to disambiguate from MSVC's deprecated POSIX shims:

`fopen → eshkol_fopen`, `access → eshkol_access`, `remove → eshkol_remove`, `rename → eshkol_rename`, `mkdir → eshkol_mkdir`, `rmdir → eshkol_rmdir`, `chdir → eshkol_chdir`, `stat → eshkol_stat`, `opendir → eshkol_opendir`

Generated programs call `eshkol_runtime_init()` at start of `main()` (non-REPL mode).

### Codegen Error Handling

- New `fatal_codegen_error_` flag — codegen now fails hard on undefined-function/undefined-variable/private-symbol errors instead of emitting `printf`/`exit` runtime stubs unreported
- New `declared_functions_by_ast` map keyed by AST node identity — fixes function resolution when multiple `define`s share a name within the same module

### VM Closure Corrections (browser REPL + bytecode VM)

Two critical closure-handling gaps in the bytecode VM had disabled correct autodiff results in demos involving captured upvalues; both are now corrected:

- **Named-let nested closure PC offset**: When a lambda is created inside a `(let loop ...)` body, the loop's bytecode is inlined into the parent function with PC adjustments — but the inner lambda's `OP_CLOSURE` constant (its `func_pc`) was *not* offset by the loop's start position. The inner closure ended up jumping to a stale location with the wrong upvalue count, manifesting as "UPVALUE INDEX OUT OF BOUNDS" plus gradient always equal to 1 in named-let gradient descent.
- **Native 252 upvalue relay**: When a lambda inside a function captures a variable via the parent's upvalue (`is_local=false`), native 252 was reading `vm->stack[vm->fp + slot]` — treating the upvalue index as a stack-frame offset, reading whichever local happened to be at that slot. Fix: read from `vm->stack[vm->fp - 1]` (the parent closure per the calling convention), then index into `parent_cl->closure.upvalues[slot]`.

Together these restore correct gradients for **every** autodiff demo on the website. The "Train a Neural Network" front-page card now converges to ~0.891 over 3 data points, and the named-let gradient descent in `/learn` chapter 5 converges to `y/x`.

### CI / Release Workflow Overhaul

- Release workflow rewritten as two matrices: `unix-release-matrix` (10 jobs) + `windows-release-matrix` (6 jobs)
- New `publish-release` job downloads all artifacts, generates `SHA256SUMS.txt`, and publishes the GitHub release
- Per-architecture LLVM SDK caching on Windows runners (cache key includes `${arch}` and SDK version)
- CI workflow updated: `windows-2022` → `windows-latest`, `max-parallel: 2` Windows throttling
- Removed Docker-based XLA/CUDA build paths in favor of native CMake builds

### Website — Mobile Responsiveness

- Hamburger nav menu collapses the 7 top-level nav links on screens ≤720px; opens as a full-width dropdown; auto-closes when a link is clicked
- `html, body { overflow-x: hidden }` plus `min-width: 0` on flex/grid children — no more horizontal page scroll on any viewport (verified across 5 viewport sizes × 6 routes)
- Code blocks (`runnable-code` wrappers) now scroll horizontally *inside* the block instead of pushing the page wider
- `.docs-layout` switched from `1fr` to `minmax(0, 1fr)` — fixes the docs page reporting 972px wide on a 375px viewport
- `.comparison-table` becomes scrollable on ≤720px so the comparison table on `/downloads` doesn't push the page

### Browser REPL Error Display

- REPL now captures stderr (compile warnings, parse errors) into `_vmStderr` and displays them as `error: undefined variable 'foo'` instead of re-prompting with no indication anything failed
- Suppresses the trailing `()` NIL fallback when a compile error fired
- Shows `error: could not parse expression` when nothing parses
- Same fix applied to runnable code blocks (Run ▶ buttons across the site)

### Test Results

- 35/35 test suites, 100% pass rate on macOS ARM64, Linux x64, Linux ARM64, Windows x64, Windows ARM64
- 32/32 runnable site examples verified end-to-end in headless Chromium across mobile/tablet/desktop viewports
- All 16 release-build lanes green on the v1.1.13-accelerate tag

---

# Eshkol v1.1.12-accelerate — Toolchain Unification + Platform Hardening

**Release Date**: April 7, 2026

Eshkol v1.1.12-accelerate unifies the toolchain on LLVM 21 across all platforms, adds a native Windows build path via Visual Studio 2022 + ClangCL, fixes ARM64 and Windows x64 ABI issues in the runtime, adds clean URL routing to the website, and expands CI/CD coverage.

## What's New in v1.1.12-accelerate

### LLVM 21 Toolchain Unification

- Standardized entire build on LLVM 21 across Linux, macOS, and Windows
- New `cmake/LLVMToolchain.cmake`: authoritative LLVM version discovery and enforcement at configure time
- New `scripts/lib/llvm21-env.sh`: platform-aware LLVM 21 activation for all shell scripts
- Hard version check: configure fails with a clear error if LLVM major version is not exactly 21
- Removed misleading `LLVM 18+` compatibility branches from backend codegen

### Native Windows Build (Visual Studio 2022)

- Full native build via Visual Studio 2022 + ClangCL + LLVM 21 SDK
- Configures with `Visual Studio 17 2022` generator and `-T ClangCL`
- `region_escape_tagged_value_into` ABI fix: passes `eshkol_tagged_value_t` by pointer to satisfy Windows x64 calling convention for 16-byte aggregates

### ARM64 ABI Fix

- Fixed `call_thunk_closure` in `arena_memory.cpp:3908`: ARM64 returns 16-byte structs in register pairs (x0:x1), not via hidden return buffer
- Resolves dynamic-wind + call/cc thunk invocation on Apple Silicon and Linux ARM64

### Mutual TCO Fix

- `llvm_codegen.cpp`: version-gated tail call kind — `TCK_MustTail` on LLVM < 18, `TCK_Tail` on LLVM ≥ 18
- Fixes "LLVM ERROR: cannot use musttail" on Linux with LLVM 21

### Website — Clean URL Routing

- Navigation now uses `/downloads`, `/learn`, `/docs` etc. instead of `/#/downloads`
- GitHub Pages 404-redirect SPA routing for direct URL access
- History API (`pushState`/`popstate`) replaces `hashchange`

### CI/CD Expansion

- New GitLab CI matrix: Linux x64/arm64 × lite/XLA/CUDA + macOS × lite/XLA + Windows
- GitHub CI updated to LLVM 21 baseline across all runners
- Docker parity images (`docker/debian/`, `docker/ubuntu/`) updated to LLVM 21

### Test Results

- 35/35 test suites, 438/438 tests, 100% pass rate (macOS ARM64, Linux x64)

---

# Eshkol v1.1.11-accelerate - Performance Acceleration Release

**Release Date**: March 27, 2026

Eshkol v1.1-accelerate builds on the v1.0-foundation with comprehensive performance acceleration. Every v1.1 roadmap item is now complete: XLA backend (5/5), SIMD vectorization (4/4), concurrency (5/5), extended math (5/5), bignum/rational (6/6), consciousness engine (4/4), R7RS extensions (6/6), dual backend (7/7), and Windows platform (5/5) -- totaling 47/47 items.

## What's New in v1.1-accelerate

### Web Platform

Eshkol compiles to WebAssembly and runs in the browser. The project website ([eshkol.ai](https://eshkol.ai)) is itself written in Eshkol — 1,500+ lines compiled to a 502KB WASM binary.

- **Browser REPL**: A 63-opcode bytecode interpreter compiled via Emscripten runs in the browser with 555+ built-in functions. Users can evaluate Eshkol expressions without installing anything.
- **Automatic Differentiation in Browser**: Forward-mode AD via dual numbers works through the bytecode VM. Arithmetic opcodes detect dual number operands and dispatch to dual arithmetic (product rule, quotient rule, chain rule). `(derivative (lambda (x) (* x x)) 3.0)` returns `6` in the browser.
- **Interactive Examples**: Every code example on the website has a Run button with inline output. Examples span AD, neural network training, ODE solving, knowledge base queries, and exact arithmetic.
- **59 DOM Bindings**: Create elements, manipulate styles, handle events, draw on canvas, manage routing, access local storage — all from Eshkol compiled to WASM.
- **8-Chapter Interactive Textbook**: Progressive tutorial from basics through AD, tensors, scientific computing, and the consciousness engine — every example runnable.

### Bytecode VM — Production Complete

The bytecode VM is a fully production-grade execution engine:

- **555+ built-in functions** including character operations, bitwise logic, type predicates, string processing, list utilities, math extensions, complex numbers, and port I/O
- **Automatic differentiation in the VM**: Forward-mode AD via dual number propagation through all arithmetic and transcendental operations
- **R7RS control flow**: `call/cc` with continuation capture/restore, `guard`/`raise`, `dynamic-wind`, `values`/`call-with-values`
- **Exact arithmetic**: Rational literals (`1/3`), bignums, complex numbers, `+nan.0`/`+inf.0`/`-inf.0`
- **Consciousness engine**: Knowledge base queries with pattern matching, factor graphs, global workspace
- **Mutual recursion**: Top-level function defines can reference each other
- **System integration**: `directory-entries`, `command-line`, thread pool
- **176/176 tests passing**

### XLA Backend (Dual-Mode Architecture)

Tensor operations now dispatch through a multi-tier acceleration hierarchy:
- **StableHLO/MLIR path**: When MLIR is available, emits StableHLO ops for HW-optimized execution
- **LLVM-direct path**: Default mode with hand-tuned LLVM IR generation
- **Threshold dispatch**: XLA (>=100K elements) -> cBLAS (>=64) -> SIMD (>=64) -> scalar
- 6 core operations fully wired: matmul, elementwise, reduce, transpose, broadcast, slice

### SIMD Vectorization

Tensor loops are now explicitly vectorized with LLVM loop metadata and 64-byte aligned allocation:
- CPU feature detection for SSE2, SSE4.1, AVX, AVX2, AVX-512, and NEON
- SIMD micro-kernels for all tensor arithmetic and activation functions
- Loop vectorization metadata attached to all tensor operation back-edges
- Platform-specific tuning via cache-blocked matrix multiplication

### Signal Processing Library

New `signal.filters` module with 13 DSP functions:
- **Window functions**: Hamming, Hann, Blackman, Kaiser (with inline Bessel I0)
- **Convolution**: Direct O(N*M) and FFT-based O(N log N)
- **Filters**: FIR filter application, IIR Direct Form I
- **Butterworth design**: Lowpass, highpass, bandpass via bilinear transform
- **Analysis**: Frequency response (magnitude + phase)

### Optimization Algorithms

New `ml.optimization` module with 4 gradient-based optimizers:
- **Gradient descent** with configurable learning rate and convergence tolerance
- **Adam** (Adaptive Moment Estimation) with bias correction
- **L-BFGS** with two-loop recursion and backtracking Armijo line search
- **Conjugate gradient** (Fletcher-Reeves) with automatic restarts

All optimizers use the builtin `gradient` function (forward-mode AD with dual numbers).

### Parallelism & Concurrency

- `parallel-map`, `parallel-fold`, `parallel-filter`, `parallel-for-each`
- `future`/`force` for asynchronous computation
- Work-stealing thread pool with hardware-aware sizing
- Thread-safe arena memory management

### Arbitrary-Precision Arithmetic

- Bignum integers with full R7RS compliance (35 codegen gaps fixed)
- Rational numbers (exact fractions)
- Automatic overflow promotion (int64 -> bignum) and demotion
- All arithmetic, comparison, and I/O operations for both types

### Consciousness Engine

Novel AI primitives integrated at the compiler level:
- Logic programming (unification, substitutions, knowledge bases)
- Active inference (factor graphs, belief propagation, free energy minimization)
- Global workspace theory (modules, softmax competition, content broadcasting)
- 22 builtin operations spanning logic, inference, and workspace

### Dual Backend Architecture

Eshkol now ships with a complete bytecode VM alongside the LLVM native compiler:
- **Bytecode VM**: 64 opcodes, 250+ native calls, ESKB binary format, invoked via `-B` flag
- **Weight Matrix Transformer**: 126/126 inline programs and 123/123 traced programs passing, 3-way verified, 12.22M analytical parameters
- **qLLM Bridge**: Eshkol-to-qLLM tensor conversion for semiclassical inference

### Windows Platform Support

Native Windows builds are now supported:
- **MSYS2/MinGW64 native build** (PR #9 by mattneel)
- UTF-8-safe REPL with proper console code page handling
- Runtime DLL bundling for standalone distribution
- Path normalization for Windows-style backslash paths

### R7RS Compliance

- `call/cc` and `dynamic-wind`
- `guard`/`raise` exception handling
- Bytevectors, `let-syntax`/`syntax-rules`, symbol operations
- Tail call optimization validation
- `(load "path")` R7RS file loading support

### GPU Backends

- Metal backend for Apple Silicon (SF64 software float64 emulation)
- CUDA backend with cuBLAS integration
- 5 GPU operations: elementwise, matmul, reduce, softmax, transpose

## Test Results

35 test suites passing with 438 test files covering all subsystems.

---

# Eshkol v1.0.0-foundation - Production Release

**Release Date**: December 12, 2025

v1.0-foundation marks the first production release of Eshkol — a programming language that integrates compiler-level automatic differentiation, deterministic arena memory management, and homoiconic native-code execution as first-class language features rather than library overlays.

## What is Eshkol?

Eshkol is a production-grade Scheme dialect built on LLVM infrastructure, designed for gradient-based optimization, neural network development, and scientific computing. It combines functional programming elegance with native performance while eliminating garbage collection entirely.

## v1.0-foundation Achievements

### Complete Production Compiler

Eshkol v1.0-foundation delivers a fully functional compiler with:

- **Modular LLVM backend** with 21 specialized code generation modules
- **HoTT-inspired gradual type system** with bidirectional type checking
- **Comprehensive parser** supporting S-expressions, type annotations, pattern matching, and macros
- **Ownership and escape analysis** for automatic allocation strategy optimization
- **Module system** with dependency resolution and circular dependency detection
- **Interactive REPL** with LLVM ORC JIT compilation
- **170+ test files** providing comprehensive verification

### Compiler-Integrated Automatic Differentiation

First-class AD system operating at compiler, runtime, and LLVM IR levels:

- **Forward-mode AD** using dual number arithmetic
- **Reverse-mode AD** with computational graph and tape stack
- **Symbolic AD** through AST transformation
- **Nested gradients** up to 32 levels deep
- **8 vector calculus operators**: derivative, gradient, jacobian, hessian, divergence, curl, laplacian, directional-derivative
- **Polymorphic implementation** supporting int64, double, dual numbers, AD nodes, and tensors

### Deterministic Memory Management (OALR)

Zero garbage collection with ownership-aware lexical regions:

- **Arena allocation** with O(1) bump-pointer allocation
- **Escape analysis** automatically determining stack/region/shared allocation
- **with-region syntax** for lexical memory scopes  
- **Ownership tracking** preventing use-after-move at compile time
- **Fully deterministic** - zero GC pauses for real-time applications

### Comprehensive Language Features

**300+ language elements including:**
- 39 special forms (define, lambda, let/let*/letrec, if/cond/case/match, etc.)
- 60+ list operations with full Scheme compatibility
- 30+ string utilities
- 25+ tensor operations
- 10 hash table operations
- Complete I/O system with ports and exception handling
- Hygienic macros (syntax-rules)
- Pattern matching with 7 pattern types
- Multiple return values (values, call-with-values, let-values)

### Rich Standard Library

Modular library organization with pure Eshkol implementations:

- **stdlib.esk** - Central module re-exporting core functionality
- **math.esk** - Linear algebra (det, inv, solve), numerical integration, root finding, statistics
- **core.functional** - compose, curry, flip combinators
- **core.list** - higher-order functions, transformations, queries, sorting
- **core.strings** - extended string manipulation
- **core.json** - JSON parsing and serialization
- **core.data** - CSV processing, Base64 encoding

### Production-Ready Infrastructure

- **Cross-platform**: macOS (Intel/Apple Silicon), Linux (x86_64/ARM64), Windows (MSYS2/MinGW64)
- **Docker containers**: Debian and Ubuntu images
- **CMake build system**: Modern, maintainable build infrastructure
- **Comprehensive documentation**: Language specification, user reference, API docs
- **Package generation**: Homebrew formula, Debian packages

## Installation

### Quick Start

```bash
git clone https://github.com/tsotchke/eshkol.git
cd eshkol
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run a program
build/eshkol-run tests/neural/nn_working.esk

# Start interactive REPL
build/eshkol-repl
```

### System Requirements

- **LLVM** 10.0+ (14+ recommended)
- **CMake** 3.14+
- **C17/C++20 compiler** (GCC 8+, Clang 6+)
- **readline** (optional, for REPL enhancements)

## Example: Neural Network Training

```scheme
(require stdlib)

;; Sigmoid activation
(define (sigmoid x)
  (/ 1.0 (+ 1.0 (exp (- 0.0 x)))))

;; Mean squared error loss
(define (mse-loss pred target)
  (let ((diff (- pred target)))
    (* 0.5 (* diff diff))))

;; Forward pass
(define (forward weights bias input)
  (sigmoid (+ (tensor-dot weights input) bias)))

;; Compute loss gradient for backpropagation
(define (loss-gradient weights bias input target)
  (gradient 
    (lambda (params)
      (mse-loss 
        (forward (vref params 0) (vref params 1) input)
        target))
    (vector weights bias)))

;; Training works - automatic differentiation handles the calculus
```

## What Makes v1.0-foundation Special

### 1. Compiler-Integrated AD - Not a Library

Unlike JAX, PyTorch, or TensorFlow, Eshkol's automatic differentiation is built into the **compiler itself**, operating on AST, runtime values, and LLVM IR simultaneously. This enables differentiation of **any** Eshkol function without framework constraints or graph tracing overhead.

### 2. Homoiconic Native Code

Lambdas compile to LLVM-native code but retain their source S-expressions in closure structures, enabling both **runtime introspection** and **native performance** - a combination no other compiled language achieves.

### 3. Zero Garbage Collection

Arena-based memory management provides **fully deterministic** performance without GC pauses, making Eshkol suitable for real-time systems, trading algorithms, and control systems where predictable timing is critical.

### 4. Production-Quality Implementation

This isn't a research prototype - it's a complete compiler with comprehensive testing, thorough documentation, and a clear architectural foundation for future expansion.

## Documentation

- **[Language Specification](docs/COMPLETE_LANGUAGE_SPECIFICATION.md)** - Complete technical specification
- **[Language Reference](docs/reference/language/INDEX.md)** - User-focused reference with examples
- **[Vision Documents](docs/vision/)** - Purpose, competitive analysis, roadmap
- **[Architecture Guide](docs/ESHKOL_V1_ARCHITECTURE.md)** - Technical architecture overview
- **[API Reference](docs/API_REFERENCE.md)** - Comprehensive function documentation
- **[Quickstart](docs/QUICKSTART.md)** - Hands-on tutorial

## Known Limitations

v1.1-accelerate builds on v1.0-foundation. Remaining planned features:

- **Distributed computing** - Planned v1.2 (Q2 2026)

See [ROADMAP.md](ROADMAP.md) and [docs/vision/FUTURE_ROADMAP.md](docs/vision/FUTURE_ROADMAP.md) for detailed development plans.

## Next Steps

### For Users

1. **Explore the REPL**: `build/eshkol-repl`
2. **Try the examples**: `build/eshkol-run tests/autodiff/*.esk`
3. **Read the docs**: Start with [docs/ESHKOL_LANGUAGE_GUIDE.md](docs/ESHKOL_LANGUAGE_GUIDE.md)
4. **Experiment with AD**: The automatic differentiation system is production-ready

### For Contributors

1. **Review architecture**: [docs/ESHKOL_V1_ARCHITECTURE.md](docs/ESHKOL_V1_ARCHITECTURE.md)
2. **Check the roadmap**: [ROADMAP.md](ROADMAP.md) for v1.1/v1.2 plans
3. **See contribution guidelines**: [CONTRIBUTING.md](CONTRIBUTING.md)
4. **Join development**: See open issues on GitHub for contribution areas

### For Researchers

1. **Study the AD implementation**: [docs/vision/ADDENDUM_TECHNICAL_WHITE_PAPER_V1.md](docs/vision/ADDENDUM_TECHNICAL_WHITE_PAPER_V1.md)
2. **Examine memory architecture**: [docs/breakdown/MEMORY_MANAGEMENT.md](docs/breakdown/MEMORY_MANAGEMENT.md)
3. **Analyze type system**: [docs/breakdown/TYPE_SYSTEM.md](docs/breakdown/TYPE_SYSTEM.md)
4. **Explore homoiconic closures**: [docs/vision/AI_FOCUS.md](docs/vision/AI_FOCUS.md)

## Acknowledgments

Eshkol v1.0-foundation represents years of research and implementation, synthesizing ideas from:
- **Scheme** for elegant functional programming
- **LLVM** for world-class code generation
- **Homotopy Type Theory** for rigorous type foundations
- **Region-based memory** research for deterministic allocation

We thank early testers and contributors who provided valuable feedback during development.

## License

Eshkol is released under the **MIT License** - see [LICENSE](LICENSE) for details.

## Contact

- **GitHub Repository**: https://github.com/tsotchke/eshkol
- **Issues**: Reports and feature requests
- **Discussions**: Technical questions and community engagement

---

**Eshkol v1.0-foundation** establishes a new standard for programming languages combining automatic differentiation, deterministic memory, and homoiconic native code. This is not a preview - this is only the beginning. Eshkol has a production-grade compiler ready for gradient-based computing, neural network development, and scientific applications where mathematical correctness and performance are non-negotiable.

*Where mathematical elegance meets uncompromising performance.*
