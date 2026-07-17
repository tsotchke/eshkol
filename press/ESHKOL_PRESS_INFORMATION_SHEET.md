# Eshkol v1.3.3-evolve — Press Information Sheet

For trade-press and academic readers preparing coverage of the v1.3 release
line — arbitrary-order automatic differentiation, full R7RS conformance,
resident/daemon-workload robustness, and, as of v1.3.3-evolve, opt-in
differentiable quantum computing with post-quantum cryptography — and the
SDNC paper artefact it carries.

---

## Identity

| | |
|:---|:---|
| Project | Eshkol |
| Version | v1.3.3-evolve |
| Builds on | v1.3.2-evolve (9 July 2026), v1.3.1-evolve, v1.3.0-evolve (7 July 2026) |
| Release date | 16 July 2026 |
| Licence | MIT |
| Source | https://github.com/tsotchke/eshkol |
| Website | https://eshkol.ai |
| Maintainer | tsotchke / Tsotchke Corporation |
| Contact | team@tsotchke.org |

---

## Headline

A six-layer transformer with 12.22 million analytically-constructed parameters
executes a bounded 83-opcode bytecode VM bit-identically. The result is a
constructive — not statistical — proof that a fixed-weight transformer can
*be* an interpreter when its weights are derived from the instruction-set
specification rather than fit by gradient descent. The reproducibility
artefact, including the weight tensor, traces, and a three-way agreement
report, ships in the same repository as the compiler that hosts it, and
regenerates in one command on a developer laptop.

The host language has moved in this release line too: v1.3.0-evolve gave
Eshkol's automatic differentiation a second axis, arbitrary order, with
exact bignum/rational derivatives and full R7RS conformance on a portable
differential corpus; v1.3.1-evolve added the robustness — flat memory in
long-running loops, a reader that doesn't overflow the stack on large
persisted state — that makes an Eshkol program safe to run unattended as a
daemon, plus a comprehensive documentation pass across the public embedding
API. v1.3.2-evolve extended that memory-safety work to the logic/workspace
heap subtypes and made region scoping thread-safe under `parallel-map`.
v1.3.3-evolve is the largest step in the line: an opt-in quantum computing
stack — Moonlab state-vector simulation, a variational quantum eigensolver
whose gradients flow through Eshkol's own automatic differentiation via new
custom-VJP tape nodes, a CHSH Bell-inequality gate, Bell-verified quantum
randomness, and ML-KEM (FIPS 203) post-quantum cryptography — alongside real
`make-parameter`/`parameterize` dynamic parameters, the `core.dbsp`
incremental-dataflow module, bignum-capable exact rationals, and one-pass
reverse-mode gradients. The same release ran a silent-wrong-answer
correctness campaign driven by two new generative exposure engines, closing
several bytecode-VM silent-miscompile classes, tail-call gaps in six special
forms, and a 26x `--wasm` size regression; it also closes out the
region-evacuator series with the last heap subtype, `PROMISE`, and replaces
an overstated v1.3.2-evolve automatic-differentiation claim with the real
fix: exact gradients through the `input2` path for first-class losses and
vector/learnable gamma.

---

## Why this is news

The dominant pattern in machine learning today is *parametric*: a generic
architecture is trained to approximate a target behaviour over a data
distribution. SDNC is *constructive*: an architecture is given a closed-form
weight assignment that realises a target behaviour exactly. The result is
narrow — the target is a 256-dimensional, 83-opcode VM — but it is exact,
auditable, and reproducible. For a transformer with weights tied to a
specification rather than to a checkpoint, every numerical claim is decidable
by inspecting the weights.

Eshkol is the host language for that result. The language is a compiled R7RS
Scheme dialect with automatic differentiation and a small neuro-symbolic stack
as compiler primitives. The SDNC artefact uses Eshkol's differentiation
infrastructure (the same that powers `gradient`, `jacobian`, `hessian`) as
the substrate in which the transformer's own forward and backward passes are
expressed. The relationship is reflexive: a language with first-class AD
hosts a transformer that has AD as part of its instruction set.

The paper is *The Self-Differentiating Neural Computer: Computable
Transformers via Analytical Weight Construction* (tsotchke, 2026). The
companion repository is `noesis`. The artefact directory is
`artifacts/paper/`; reference documentation lives at *docs/SDNC.md*,
*docs/breakdown/COMPUTABLE_TRANSFORMER.md*, and
*docs/breakdown/VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md*.

---

## The language

### Identity

Eshkol is an R7RS-compatible Scheme dialect. The implementation passes
roughly 232 of 244 R7RS-small procedures (~95%), includes hygienic
`syntax-rules` macros, first-class single-shot continuations (`call/cc`,
`dynamic-wind`, `guard`/`raise`, `delay`/`force`), bytevectors, records via
`define-record-type`, and `eval` with all three R7RS environment
constructors (*docs/breakdown/OVERVIEW.md §Eshkol vs. Scheme*). A separate,
newer measure — a reference-Scheme differential oracle that diffs Eshkol
against chibi-scheme 0.12.0 on a 34-program portable corpus — reports
34 of 34 AGREE (100%), up from 27/34 at the start of the v1.3.0-evolve cycle.

The parser handles ninety-four operation types over an S-expression syntax
with line/column tracking and an R7RS-compliant internal-defines transform
to `letrec*`. The macro expander supports ellipsis patterns, nested
patterns, and hygienic renaming. As of v1.3.1-evolve, the S-expression reader
(`read_list`) is iterative rather than per-element recursive, so reading
back a very large persisted list no longer risks a native-stack overflow.
See *lib/frontend/parser.cpp*, *lib/frontend/macro_expander.cpp*, and
*lib/core/runtime_reader_hosted.cpp*.

### Implementation

| Component | Lines | File / directory |
|:---|---:|:---|
| Main LLVM codegen | 33,962 | *lib/backend/llvm_codegen.cpp* |
| Total LLVM backend (main + ~30 modules) | ≈85,500 | *lib/backend/* |
| Autodiff codegen (order ≤ 2, incl. custom-VJP) | ≈12,600 | *lib/backend/autodiff_codegen.cpp* |
| Taylor-tower runtime (arbitrary order) | ≈1,460 | *lib/core/runtime_taylor.c*, *lib/core/taylor_recurrences.def* |
| Taylor-tower stdlib modules (GUW, tensor towers, models, checkpointing, numerics, sparse) | ≈2,270 | *lib/core/ad/{guw,tensor_tower,taylor_models,checkpoint,taylor_numerics,sparse_guw,interval}.esk* |
| Backward-mode kernels | ≈1,400 | *lib/backend/tensor_backward.cpp* |
| String / I/O / JSON / CSV | 3,293 | *lib/backend/string_io_codegen.cpp* |
| Work-stealing parallel codegen | 2,601 | *lib/backend/parallel_llvm_codegen.cpp* |
| Arithmetic (incl. bignum/rational/complex) | 2,491 | *lib/backend/arithmetic_codegen.cpp* |
| Collection ops | 2,348 | *lib/backend/collection_codegen.cpp* |
| Parser | 8,354 | *lib/frontend/parser.cpp* |
| Macro expander | 861 | *lib/frontend/macro_expander.cpp* |
| Type checker | 1,999 | *lib/types/type_checker.cpp* |
| Arena memory | ≈1,535 | *lib/core/arena_memory.h* and *lib/core/runtime_arena_\*.cpp* |
| S-expression reader (iterative as of v1.3.1-evolve) | 555 | *lib/core/runtime_reader_hosted.cpp* |
| Logic engine | ≈1,180 | *lib/core/logic.cpp* |
| Active-inference engine | ≈1,200 | *lib/core/inference.cpp* |
| Global workspace | 354 | *lib/core/workspace.cpp* |
| Quantum FFI (opt-in, v1.3.3-evolve) | ≈1,450 | *lib/agent/quantum.esk*, *lib/agent/c/agent_quantum.c* |
| ML-KEM post-quantum KEM (opt-in, v1.3.3-evolve) | ≈520 | *lib/agent/pqc.esk*, *lib/agent/c/agent_pqc.c* |
| Incremental dataflow (v1.3.3-evolve) | 448 | *lib/core/dbsp.esk* |
| Weight-matrix transformer | ≈7,500 | *lib/backend/weight_matrices.c* |
| Bytecode VM + runtime libs | ≈41,000 | *lib/backend/eshkol_vm.c* and runtime |

Total compiler infrastructure is approximately 286,000 lines of C17 and C++20
across more than 130 files (*docs/DESIGN.md §Implementation Scale*). v1.3.1-evolve
added roughly 12,600 lines of Doxygen-format documentation across 116 files —
50 of the 64 public headers under `inc/eshkol/` (≈4,650 lines) and 56
previously-undocumented implementation files under `lib/` (≈7,478 lines) —
comments only, no behaviour change; v1.3.2-evolve added a further
`eshkol-doc` reference generator (`docs/api/`) that harvests those comments
automatically.

### Target backend

LLVM 21 is the version-enforced target on every platform; the build aborts
with a clear error message if `llvm-config` reports any other major version
(*cmake/LLVMToolchain.cmake §`eshkol_validate_llvm_major`*).
Targets currently supported:

- macOS, Intel and Apple Silicon
- Linux, x86-64 and ARM64
- Windows, x86-64 and ARM64, via Visual Studio 2022 + ClangCL + the LLVM 21 SDK
- WebAssembly, via `eshkol-run --wasm` (self-contained module, does not fall
  through to a native link step)

CI is green across all 14 lanes: Linux and macOS (x64/ARM64) each cover
lite/XLA/CUDA-capable variants, Windows x64 ships lite, XLA, and CUDA, and
Windows ARM64 ships lite and XLA. Windows ARM64 CUDA is not offered because
NVIDIA does not provide the required toolkit for that target
(*README.md §Platform*).

### Positioning

The language occupies a region not covered by the established alternatives.

- **Compared with other Scheme implementations** (Racket, Chez, Chicken): Eshkol
  compiles to native code through LLVM rather than to bytecode or via a
  source-to-C transform; it integrates automatic differentiation, GPU
  dispatch, and a neuro-symbolic stack at the compiler level rather than as
  optional libraries. It provides single-shot continuations via setjmp/longjmp
  rather than the full multi-shot continuations available in Racket or Chez.
- **Compared with AD-first systems** (Julia + Zygote, Python + JAX or PyTorch):
  Eshkol's AD is integrated into the compiler at the IR level rather than
  obtained through tracing, source-to-source rewriting, or operator
  overloading, and — since v1.3.0-evolve — computes exact derivatives at any
  order via Taylor towers, which JAX's `jax.experimental.jet` approaches only
  numerically; as of v1.3.3-evolve those exact gradients also flow correctly
  through the `input2` path (kernel/gamma/K/V) for first-class losses, not
  just literal-loss compile-time-known functions. The host runtime has no
  garbage collector; allocation is bounded by the arena reset boundary, and
  as of v1.3.1-evolve self-tail-recursive loops reclaim their arena scope
  automatically per iteration — a region-escape evacuator, completed in
  v1.3.3-evolve, promotes any heap object that escapes that scope (including
  logic/workspace state and, as of this release, a `delay`d promise) rather
  than leaving it dangling — which makes Eshkol viable in contexts (real-time
  control loops, embedded inference, long-running daemons) where Python or
  Julia's GC pauses, or unbounded process RSS growth, are disqualifying.

Each comparison is grounded in *docs/breakdown/OVERVIEW.md §Comparison with
other languages*; the document explicitly enumerates where Eshkol gives less
than the alternative (multi-shot continuations, mature library ecosystem,
GUI toolkits).

---

## Memory model: OALR

Ownership-Aware Lexical Regions replace garbage collection. The model
consists of:

- A single global arena allocator with an 8 KB minimum block size,
  bump-pointer allocation, and batch deallocation via arena reset.
- An 8-byte object header prepended to every heap object
  `{subtype:u8, flags:u8, ref_count:u16, size:u32}`. The header is at
  offset −8 from the data pointer returned by allocators.
- Twenty-four heap-subtype slots assigned through v1.3.1-evolve (slot 14
  remains reserved for a future `RULE` backward-chaining type; slot 23 is
  `HEAP_SUBTYPE_TAYLOR`, added in v1.3.0-evolve for the arbitrary-order
  Taylor-tower AD engine) and five callable subtypes, consolidating eight
  historical pointer types into two supertypes (`HEAP_PTR`, `CALLABLE`).
  As of v1.3.3-evolve every subtype that carries interior pointers —
  including `PROMISE`, the last one — is deep-walked by the region-escape
  evacuator rather than shallow-copied on escape from a `with-region` scope.
  See *inc/eshkol/eshkol.h §heap subtypes*.
- 16-byte tagged values laid out `{type:u8, flags:u8, reserved:u16,
  padding:u32, data:u64}`. When the compiler can prove the type at
  compile time it emits untagged LLVM IR, eliminating the tagging
  overhead entirely.
- A 512 MB main-thread stack via linker flags (`-Wl,-stack_size,0x20000000`
  on Darwin; `-Wl,-z,stacksize=0x20000000` on Linux), runtime
  configurable via `ESHKOL_STACK_SIZE`. The default maximum
  recursion depth is 100,000 frames.
- Per-thread arenas (1 MB, lazily allocated through `thread_local`) for
  parallel workers; the global arena is used only to construct result
  lists after parallel tasks have completed (*docs/breakdown/PARALLEL_COMPUTING.md §2.1*).
- **Per-iteration arena-scope reclamation for self-tail-recursive loops**
  (extended in v1.3.1-evolve). A conservative static escape analysis
  (`namedLetIterScopeSafe`) proves a loop body's arena allocations don't
  escape across the tail-call back-edge; when it does, the loop's arena
  scope is reclaimed every iteration with zero source annotation. v1.3.0-evolve
  covered named-let loops; v1.3.1-evolve extended the same analysis to
  self-tail-recursive `define` loops and accepts a catch-all guard clause in
  the loop body. Verified on a 1,000,000-iteration loop: RSS goes from
  1,369 MB (unbounded growth) to 224 MB (flat). See
  *lib/backend/llvm_codegen.cpp*.
- **Region-escape evacuator** (ESH-0214a through e, completed v1.3.3-evolve).
  When a value allocated inside a `with-region` block escapes that block
  (is returned, stored into an outer binding, or captured by a closure), a
  companion evacuator deep-walks it and promotes its interior pointers into
  the surviving arena before the region is popped, rather than leaving them
  dangling. Coverage was built out subtype by subtype: `CONS`/`VECTOR`/
  `HASH`/`TENSOR`/`EXCEPTION`/`CLOSURE` first, then the logic/workspace
  subtypes (`SUBSTITUTION`/`FACT`/`KNOWLEDGE_BASE`/`FACTOR_GRAPH`/
  `WORKSPACE`) in v1.3.2-evolve, and `PROMISE` in v1.3.3-evolve, which closes
  the series. `ESHKOL_ARENA_POISON=1` poisons `arena_destroy`'d memory so any
  remaining gap in that coverage crashes loudly instead of corrupting
  silently. See *lib/core/runtime_regions.cpp* and *lib/backend/llvm_codegen.cpp*.
- Optional linear types via `(owned ...)`, `(borrow value body)`, and
  `(shared ...)`; the third activates reference counting against the
  header's 16-bit `ref_count` field (*README.md §Memory architecture*).

The model is what makes the system bit-reproducible. Two back-to-back release
builds produce byte-identical `build/stdlib.bc` and `build/eshkol-run`;
hash-map iteration order is keyed on stable strings from the AST and does
not reach the emitted IR (*docs/HARDENING.md §`#184` deterministic execution*).

---

## Numeric tower

The R7RS numeric tower is complete, with exactness tracked through a flags
byte on each tagged value.

- **int64**: 64-bit signed integers stored inline in the tagged value (immediate).
- **bignum**: Arbitrary-precision integers as `HEAP_PTR` with subtype 11.
  Automatic promotion on int64 overflow and demotion when the result fits
  in 64 bits again.
- **rational**: Exact fractions (subtype 19), always reduced via GCD. As of
  v1.3.3-evolve the representation is a canonical discriminated union: a
  zero-allocation int64 numerator/denominator fast path, with a bignum
  numerator/denominator pair taken only on overflow — so exact rationals no
  longer degrade to double at bignum magnitudes, and bignum-magnitude
  rational literals parse. Verified byte-identical against Python
  `Fraction`.
- **double**: IEEE 754 64-bit floats (inexact).
- **complex**: Heap-allocated `{real:f64, imag:f64}` with Smith's-formula
  division.

R7RS semantics hold for mixed arithmetic: exact + exact = exact, exact + inexact
= inexact. As of v1.3.0-evolve, exactness propagates through arbitrary-order
differentiation as well: `derivative-n`/`taylor` return exact bignum/rational
coefficients when the seed point is exact and the function only uses
exact-preserving operators, demoting to `double` on overflow or at the first
transcendental call. See *docs/DESIGN.md §Exact arithmetic* and
*lib/backend/arithmetic_codegen.cpp*.

---

## Automatic differentiation

Two orthogonal axes: **mode** (symbolic, forward, reverse) and, since
v1.3.0-evolve, **order** (arbitrary, via Taylor towers).

**Symbolic mode.** AST rewriting at compile time using twelve differentiation
rules. Zero runtime overhead when the function is syntactically known.

**Forward mode.** 16-byte dual numbers `{value:f64, derivative:f64}`
propagated through arithmetic, transcendentals, and activations. Suitable
for functions R → Rⁿ. The dual-number type has its own tag
(`ESHKOL_VALUE_DUAL_NUMBER = 6`) and is dispatched at the LLVM IR level.

**Reverse mode.** A computational graph with more than twenty AD node types
(elementary arithmetic, transcendentals, scalar utilities, neural-network
activations) recorded onto a Wengert tape during the forward pass. The tape
is topologically sorted and walked in reverse during gradient construction.
A 32-level tape stack enables nested gradients for Hessians, natural
gradient, and meta-learning constructions. See
*lib/backend/autodiff_codegen.cpp* (≈12,600 lines) and
*lib/backend/tensor_backward.cpp* (≈1,400 lines).

**Custom-VJP tape nodes (new in v1.3.3-evolve).** A new reverse-mode tape
node (`AD_NODE_CUSTOM`) carries an externally supplied vector-Jacobian
product, so a foreign/FFI computation with a known adjoint participates
exactly in reverse-mode AD rather than through a finite-difference
approximation. First user: Moonlab's VQE gradient is bridged into the tape,
making `(vqe-energy ...)` differentiable through ordinary `gradient` code —
the release gate requires the bridged adjoint to match Moonlab's native
adjoint to within `1e-8` and a central finite difference to within `1e-4`,
and a VQE custom-VJP probe now sits permanently in the AD adversarial
oracle.

**One-pass reverse gradients and AD introspection (staged-kernel Phase A,
new in v1.3.3-evolve).** The per-component gradient replay is collapsed into
one primal plus one reverse pass reading every input gradient from the tape
(N primal calls become one, verified at N=4 and N=64), with a runtime
mixed-record guard keeping mixed-mode reverse-over-forward on the proven
per-component path. New `(ad-counters)`/`(ad-primal-calls)` builtins expose
primal-call/reverse-pass/tape counters to Scheme.

**Arbitrary-order mode (Taylor towers, new in v1.3.0-evolve).** A closed-recurrence
engine (`lib/core/taylor_recurrences.def`, `lib/core/runtime_taylor.c`)
computes every derivative up to an arbitrary order `k` in one pass: `k+1`
coefficients and O(k²) work, versus the 2^k blow-up of nested dual numbers.
Delivered across thirteen gated phases, P0 through P12 (see
*docs/design/AD_TAYLOR_TOWER.md* and *docs/AD_CAMPAIGN.md*):

- `(taylor f x k)` / `(derivative-n f x k)` — the coefficient series or the
  scalar `k`-th derivative, for any `k`.
- Exact bignum/rational coefficients when the seed point is exact and the
  function uses only exact-preserving arithmetic (verified with 68
  exact-coefficient checks); automatic demotion to `double` on overflow or
  the first transcendental call.
- `taylor-model` / `tm-range` / `tm-eval` — a Taylor polynomial paired with
  a rigorous interval-remainder bound, for a provable range/value enclosure.
- `mixed-partial` / `gradient-n` — arbitrary-order mixed partials via a
  Griewank-Utke-Walther (GUW) directional-propagation layer.
- `sparse-hessian` / `sparse-mixed-partials` — sparse high-order recovery
  via greedy star-coloring graph recovery.
- `checkpointed-gradient` — a Griewank/binomial √N checkpoint schedule for
  high-order reverse-mode AD, holding at most one block's tape live at a
  time (measured peak-node ratio ≈1.8 at N=200 vs. ≈4.0 dense).
- `taylor-ode-solve`, `taylor-root`, `taylor-inverse-series` — numerical
  methods built directly on the tower (fixed-step order-`k` IVP solving,
  Householder-family root refinement, Lagrange-inversion series reversion).
- Towers are tensor-valued (`core.ad.tensor_tower`) and compose with
  `matmul`/`conv2d`/activations; they work correctly through
  `if`/`cond`/named-let/recursion and `map`/`fold`.
- Perturbation confusion is handled structurally: every differentiation
  context carries its own epoch tag in the tower's header.
- Zero heap allocation on the common path: when the order `k` is a
  compile-time literal, the whole tower unrolls into stack-allocated,
  branch-free SSA IR.

Eight vector-calculus operators (order ≤ 2) are language primitives:

```
derivative              (lambda (x) ...) -> R -> R
gradient                (lambda (v) ...) -> R^n -> R^n
jacobian                (lambda (v) ...) -> R^n -> R^{m x n}
hessian                 (lambda (v) ...) -> R^n -> R^{n x n}
divergence              vector field   F: R^n -> R^n      => R^n -> R
curl                    vector field   F: R^3 -> R^3      => R^3 -> R^3
laplacian               scalar field   f: R^n -> R        => R^n -> R
directional-derivative  f, point, direction               => R
```

The `vref` operator is AD-aware: during gradient computation it creates AD
nodes; outside that context it is a simple pointer dereference. This
context-sensitivity is achieved through runtime type inspection on closure
arguments (*docs/breakdown/OVERVIEW.md §Automatic differentiation*).

A note on costs as currently measured: forward mode incurs roughly a 2–3×
slowdown, reverse mode 3–5× with O(n) memory, symbolic mode zero runtime
overhead because the rewrite is at compile time (*README.md §Autodiff
overhead*). The Taylor-tower engine is O(k²) in the requested order `k` and
zero-heap when `k` is a compile-time literal.

**Reverse-mode tensor-op gradients (`input2`), corrected in v1.3.3-evolve.**
`gradient` on `conv2d`, `batchnorm`, `layernorm`, and `attention` propagates
an exact gradient to the second operand (kernel / gamma-beta / K-V) in both
the literal-loss form (a compile-time-known `Function*`) and the first-class
form (a loss value with no compile-time `Function*`, e.g. one selected or
constructed at runtime); a v1.3.2-evolve CHANGELOG entry had claimed this
path was already complete, but an adversarial finite-difference audit found
the change shipped a test and a roadmap update with no gradient code behind
it — the first-class path still silently returned a zero gradient. The
v1.3.3-evolve fix adds a reverse-mode tensor path to the closure branch of
`gradient`'s codegen, wires batch-norm/layer-norm gamma/beta as per-feature
AD nodes instead of one scalar, and turns any remaining unsupported
tensor-op backward path into an explicit compile error instead of a silent
zero. Finite-difference-verified across matmul/conv2d/attention-K-V/
vector-gamma in both forms.

See the [Automatic Differentiation guide](../docs/guide/AUTOMATIC_DIFFERENTIATION.md)
for a worked, example-verified walkthrough of all thirteen phases, and
[`docs/reference/ad/INDEX.md`](../docs/reference/ad/INDEX.md) for the API
reference.

---

## Neuro-symbolic stack

Twenty-two compiler builtins implement three theoretical frameworks as
first-class language operations rather than library calls.

**Logic programming (Robinson's resolution, 1965)**

```
unify  walk  make-substitution  make-fact  make-kb
kb-assert!  kb-query
logic-var?  substitution?  kb?  fact?
```

Logic variables use the `?x` syntax, which the parser transforms into
`ESHKOL_LOGIC_VAR_OP` AST nodes. The leading `?` is a valid R7RS identifier
start character, so the syntax requires no grammar change. Implementation:
*lib/core/logic.cpp* (≈1,180 lines).

**Active inference (Friston's free-energy principle, 2010)**

```
make-factor-graph  fg-add-factor!  fg-infer!  fg-update-cpt!
free-energy  expected-free-energy
factor-graph?
```

The runtime supports belief propagation, CPT updates (which enable real
learning by mutating the CPT and resetting messages so beliefs reconverge),
and both variational free energy and expected free energy. Implementation:
*lib/core/inference.cpp* (≈1,200 lines).

**Global workspace theory (Baars 1988; Bengio 2017 computational formulation)**

```
make-workspace  ws-register!  ws-step!  workspace?
```

`ws-step!` is fully implemented end-to-end: the LLVM codegen loop calls
registered closures via the closure dispatcher, and C runtime helpers
(`eshkol_ws_make_content_tensor`, `eshkol_ws_step_finalize`) handle
content-tensor wrapping and softmax broadcast.
Implementation: *lib/core/workspace.cpp* (354 lines).

Heap subtypes assigned to these objects:
`HEAP_SUBTYPE_SUBSTITUTION = 12`, `HEAP_SUBTYPE_FACT = 13`,
`HEAP_SUBTYPE_KNOWLEDGE_BASE = 15`, `HEAP_SUBTYPE_FACTOR_GRAPH = 16`,
`HEAP_SUBTYPE_WORKSPACE = 17`. Type tag `ESHKOL_VALUE_LOGIC_VAR = 10`.
See *inc/eshkol/eshkol.h §heap subtypes*.

---

## The SDNC artefact

The repository ships a three-way verification harness for the SDNC paper:

1. **Reference C interpreter** — a direct switch over the eighty-three opcodes;
   the ground truth.
2. **Simulated transformer** — C functions that mirror the six layers
   (Gaussian-attention instruction fetch, polarisation-identity product,
   address-resolution preprocessing, gated opcode dispatch, tape write +
   parent load, backward gradient dispatch + write-back).
3. **Matrix-based forward pass** — explicit weight matrices generated by
   `generate_weights`, applied via the gated FFN formula at each layer:
   y = W_down · (σ(W_g · x + b_g) ⊙ (W_u · x + b_u)) + b_d.

Agreement across all three modes constitutes the verification chain.
Coverage of the 256-dimensional state vector is field-wise (PC, SP, TOS, SOS,
registers, arena cells, tape, flags); the current artefact reports
123 of 123 traced programs agreeing on the final output and 123 of 123
agreeing at every intermediate step. The strict weight artefact covers
82 of 83 canonical opcodes; the one remaining opcode is `OP_NATIVE_CALL`,
the deliberate external boundary for host-runtime services.

Reproduction is one command:

```bash
scripts/paper/run_paper_suite.sh
```

Expected wall time on a 2023 M2 Max is under five minutes. Outputs land in
`artifacts/paper/outputs/` and the harness prints stable SHA-256 hashes:

```
SHA-256  weights.qlmw              381599e7a5607b4047ede0d6c8e6d270cb81dbdebfdb0bf0c0eba38758aa3f0c
SHA-256  vm-traces.jsonl           4239cbb91dc9abb9abe80528c5b4ac4c2121a85db5a50dbf43c634a77e304801
SHA-256  transformer-traces.jsonl  4239cbb91dc9abb9abe80528c5b4ac4c2121a85db5a50dbf43c634a77e304801
SHA-256  comparison-report.json    80aa6fed4db40bca521217ae8777677173fe7eeb239baa69847111e7ac674105
SHA-256  opcode-coverage.json      152a4bacc483d8985abeb08bc0d44112144f536ed663274bc7b1eeccbdd2dfe4
```

The transformer trace and the VM trace share a hash because they agree
bitwise. Platform divergence is treated as a bug; issue reports should
include the CPU, libc, and floating-point environment.

Implementation note: the paper proves the gated indicator function is exact
in float32 for any scale S > 33.2; the artefact ships with S = 300 (rather
than the working constant S = 100) because at S = 100 the softmax score gap
between the peak position and its neighbours is ≈ 35.4, leaving a residue
e⁻³⁵·⁴ ≈ 4.6 × 10⁻¹⁶ that accumulates as `tos = 4.4e-16` at step 1206 of
`tail sum(100)` versus exactly zero in the reference. Raising S to 300
pushes the gap above 87 so e⁻ᵍᵃᵖ underflows to literal float32 zero. See
*lib/backend/weight_matrices.c:59-84* and *docs/SDNC.md §float32 saturation margin*.

---

## Parallelism

The scheduler is a per-worker Chase-Lev work-stealing deque
(*Dynamic Circular Work-Stealing Deque*, Chase and Lev, 2005) with
epoch-based reclamation, three-stage idle backoff (spin / yield / sleep),
and hardware-aware sizing (`std::thread::hardware_concurrency()`,
override via `ESHKOL_NUM_THREADS`). See
*inc/eshkol/backend/work_stealing_deque.h* (documented in v1.3.1-evolve) and
*lib/backend/thread_pool.cpp*.

Primitives: `parallel-map`, `parallel-fold`, `parallel-filter`,
`parallel-for-each`, `future`, `force`, `future-ready?`.

A void-pointer ABI boundary keeps tagged values from being passed by value
across the C/LLVM boundary, eliminating the optimisation-level-dependent
struct-by-value corruption that surfaced on ARM64. All tagged-value
construction and destruction occurs within LLVM IR; only `void*` crosses
into C.

Measured 4–12× speed-up of `parallel-map` on 24 cores
(*docs/breakdown/ROADMAP.md §1.1-accelerate completed*; the underlying
root-cause fix is recorded in project memory as the parallel-map
flags-byte fix: worker tagged-value flags were hardcoded to zero, which
mis-dispatched into the bignum path; packing `{type, flags}` into
`item_type:i64` with the default flipped on restored real parallelism
across both AOT and JIT paths). Shutdown safety was hardened in
v1.3.0-evolve: `eshkol_runtime_shutdown()` now stops and joins the global
parallel thread pool before running shutdown hooks, closing a use-after-free
race that could `SIGSEGV` after a graceful `SIGTERM` was already logged.

---

## GPU acceleration

Adaptive dispatch through *lib/backend/blas_backend.cpp*. Calibration
constants (measured on Apple Silicon):

| Backend | Peak | Overhead | Dispatch range |
|:---|---:|---:|:---|
| SIMD (vectorised) | 25 GFLOPS | ~0 | ≤ 16 elements |
| cBLAS (Apple Accelerate / AMX) | 1,100 GFLOPS | 5 µs | 17 to ~10⁹ elements |
| Metal GPU (SF64 software float64) | 200 GFLOPS | 200 µs | > 10⁹ elements |

SF64 (Software Float64) emulates double precision using double-double
arithmetic — two 32-bit mantissas combined for an effective precision of
roughly 100 bits — because Metal GPUs lack native float64. Implementation:
*lib/backend/gpu/metal_softfloat.h* and *lib/backend/gpu/gpu_memory.mm*.
CUDA dispatches through cuBLAS on NVIDIA.

The cost model selects the backend per operation. The defaults are
empirically calibrated and configurable through `ESHKOL_GPU_PRECISION`,
`ESHKOL_BLAS_PEAK_GFLOPS`, `ESHKOL_GPU_PEAK_GFLOPS`.

---

## Tensor and ML framework

Compiler-level tensor operations span more than a dozen domain-specific
codegen modules plus the dispatcher in *lib/backend/tensor_codegen.cpp*.
Coverage:

- 16 activations (relu, relu6, sigmoid, tanh, gelu, swish, mish,
  softmax, log-softmax, softplus, softsign, leaky-relu, prelu, elu,
  selu, celu)
- 14 loss functions (mse-loss, mae-loss, cross-entropy-loss, bce-loss,
  huber-loss, kl-div-loss, hinge-loss, smooth-l1-loss, focal-loss,
  triplet-loss, contrastive-loss, label-smoothing-loss,
  cosine-embedding-loss)
- 5 optimisers + 3 gradient utilities (sgd-step, adam-step, adamw-step,
  rmsprop-step, adagrad-step; zero-grad!, clip-grad-norm!, check-grad-health)
- 5 weight initialisers (xavier-uniform!, xavier-normal!, kaiming-uniform!,
  kaiming-normal!, lecun-normal!)
- 4 learning-rate schedulers (linear-warmup-lr, step-decay-lr,
  exponential-decay-lr, cosine-annealing-lr)
- 7 CNN layers (conv1d, conv2d, conv3d, max-pool2d, avg-pool2d,
  batch-norm, layer-norm)
- 8 transformer ops (scaled-dot-attention, multi-head-attention,
  positional-encoding, rotary-embedding, causal-mask, padding-mask,
  feed-forward, embedding)
- 6 data-loading ops (make-dataloader, dataloader-next, dataloader-reset!,
  dataloader-length, dataloader-has-next?, train-test-split)
- `agent.quantum` builtins (opt-in, built with `-DESHKOL_QUANTUM_ENABLED=ON`):
  state creation/teardown, Hadamard/Pauli/CNOT/rotation gates,
  `measure`/`expectation-z`, H2/LiH/H2O molecular Hamiltonians, VQE
  energy/gradient/optimise, the `bell-chsh` CHSH experiment, and
  `with-quantum-state`/`with-hamiltonian` helpers — plus `agent.pqc`'s
  ML-KEM `mlkem-keygen`/`mlkem-encaps`/`mlkem-decaps` builtins

All ML builtins integrate with reverse-mode AD: calling `gradient` on any
composition produces an exact gradient, and — since v1.3.0-evolve —
tensor-valued Taylor towers (`core.ad.tensor_tower`) extend arbitrary-order
differentiation through `matmul`/`conv2d`/activations as well. As of
v1.3.3-evolve that includes the quantum VQE energy: Moonlab's exact adjoint
gradient enters the reverse tape through a custom-VJP node, so a variational
quantum circuit differentiates like any other composition. The conv2d
backward pass uses stride-based scatter/gather indexing that doesn't map
cleanly to GEMM, and LayerNorm/BatchNorm backward are inherently sequential
reductions; see *docs/KNOWN_ISSUES.md* for the current, itemized state of
these ML-kernel performance characteristics.

---

## Agent FFI

A native FFI surface for systems programming. Each backend is implemented in
C and exposed to Eshkol through tagged-value calling conventions.

- **HTTP client** — libcurl-backed, per-thread easy interface, TLS feature
  check. Implementation: *lib/agent/c/agent_http_client.c*.
- **SQLite** — `sqlite3_open_v2`, prepared-statement bindings, dynamic-size
  column-text retrieval. Implementation: *lib/agent/c/agent_sqlite.c*.
- **Subprocess** — `posix_spawn` with argv arrays. The `popen("sh -c …")`
  path was removed in v1.2 (security advisory `#190`); the `-argv`
  variants (`process-spawn-argv`, `run-argv`, `run-argv-capture`) are
  the recommended interface for any command built from external input.
  Implementation: *lib/agent/c/agent_subprocess.c*.
- **Filesystem watch** — kqueue on macOS, inotify on Linux. Implementation:
  *lib/agent/c/agent_watch.c*.
- **Crypto, regex, terminal, compression, etc.** — additional native
  backends in *lib/agent/c/*, each with a corresponding `.esk` wrapper
  in *lib/agent/*.

AOT linking is automatic: `ESHKOL_HOST_AGENT_FFI_LINK_ARGS` in the build
config is consulted, and the AST is scanned pre-process for require
declarations so AOT binaries link the HTTP, SQLite, and subprocess
backends without the user having to specify library flags. v1.3.1-evolve
added Doxygen documentation across every agent-FFI implementation file in
*lib/agent/c/*.

---

## Tooling

- **eshkol-run** — production AOT compiler with executable, object file
  (`-c -o`), shared library (`--shared-lib`), and WebAssembly (`--wasm`)
  output modes; supports JIT execution (`-r`).
- **eshkol-repl** — interactive REPL via LLVM OrcJIT, documented as part of
  the v1.3.1-evolve implementation doc-comment pass. Preloads stdlib functions
  and globals from precompiled `.o` and `.bc` metadata. The `--machine` mode
  emits `EREPL READY` / `DONE` / `FAIL` framing on stderr for warm-worker
  IPC.
- **eshkol-doc** — API reference generator, added in v1.3.2-evolve. Harvests
  Doxygen `/** @brief */` comments from `inc/` and `lib/` and generates
  `docs/api/` (Markdown pages plus an HTML index).
- **eshkol-pkg** — package manager. TOML manifests, git-based
  registry, recursive submodule discovery. Commands: `init`, `build`,
  `run`, `add`, `clean`.
- **eshkol-lsp** — Language Server Protocol. Completions,
  hover, go-to-definition, diagnostics, formatting.
- **VS Code extension** — syntax highlighting, LSP integration, build tasks.
  Source: *tools/vscode-eshkol/*.
- **Inter-Component Communication (ICC)** — agent-FFI-ready oracle and
  pytest-format smoke harness under `.icc/`; native Eshkol-aware (accepts
  Eshkol VM step / halt records as `eshkol_vm_step` / `eshkol_vm_halt`
  events; recognises `runtime_event` compact dict form and explicit
  `kind` JSON records). Hardened in v1.3.3-evolve (#232) to also check
  region-evacuator poison coverage and the corrected `input2` gradient
  gate, and extended with the release's two generative exposure engines
  (the multi-oracle differential harness and the AD-vs-finite-difference
  adversarial oracle) as permanent release gates. ICC readiness oracle
  reports 100/100, trace-verified.

---

## Documentation

v1.3.1-evolve's principal addition, alongside its memory/reader robustness
fixes, was documentation coverage; v1.3.2-evolve turned that coverage into a
generated, navigable artefact:

- **Public C-API headers.** Doxygen-format documentation across 50 of the
  64 public headers under `inc/eshkol/` — backend codegen, runtime core,
  the type system, the XLA backend, subprocess/macro-expander/qLLM-bridge
  surfaces, the thread pool and work-stealing deque, the logger, model I/O,
  platform runtime, and runtime exports — across six commits, roughly
  4,650 lines of new documentation.
- **Implementation doc-comments.** 56 previously-undocumented implementation
  files under `lib/` — agent FFI, the type checker, the parser, the REPL,
  core non-runtime modules, the quantum RNG, FFI bridges — across three
  commits, roughly 7,478 lines.
- **Navigable reference index.** A per-subsystem documentation index at
  [`docs/reference/language/`](../docs/reference/language/INDEX.md),
  [`ad/`](../docs/reference/ad/INDEX.md),
  [`runtime/`](../docs/reference/runtime/INDEX.md),
  [`tensors/`](../docs/reference/tensors/INDEX.md),
  [`stdlib/`](../docs/reference/stdlib/INDEX.md), and
  [`agent/`](../docs/reference/agent/INDEX.md), each an example-verified
  index into the corresponding function and syntax reference, linked from
  *README.md §Documentation*.
- **Generated API reference (`eshkol-doc`, v1.3.2-evolve).** The Doxygen
  comments above are now harvested automatically into `docs/api/` rather
  than requiring a hand-maintained index.

Combined, the v1.3.1-evolve pass alone is 116 files and roughly 12,600 lines
of new documentation — comments and reference pages only, no behaviour
change.

---

## Hardening and robustness posture

**v1.3.3-evolve (current release).** Alongside the quantum, post-quantum,
dynamic-parameter, incremental-dataflow, and exact-rational features
described elsewhere in this sheet, the release ran a silent-wrong-answer
correctness campaign driven by two new generative exposure engines, both
wired permanently into the ICC release oracle:

- **Generative multi-oracle differential harness**: deterministically grown
  R7RS-subset programs cross-checked against chibi-scheme, JIT, AOT at
  O0/O2, and the bytecode VM, plus metamorphic invariants. It exposed
  several VM silent-miscompile classes — all fixed, including real
  bignum-aware VM arithmetic and comparisons and exact large-integer
  literals.
- **Generative AD-vs-finite-difference adversarial oracle**: 147 probes /
  436 component checks across 21 generated files under JIT and AOT, where a
  zero AD gradient at a point where finite differences are nonzero is a
  hard failure. It drove root fixes for every known silent-zero
  gradient/Hessian path found by the harness.
- **TCO for `cond`/`case`/`when`/`unless`/`and`/`or`** (#254):
  self-tail-recursion in all six forms now compiles to real loop back-edges
  (previously SIGBUS around 2M iterations unless written with `if`);
  verified to 2,000,000 iterations, JIT and AOT.
- **Scalable, stable `sort`/`filter`** (#244, #266): both are now
  accumulator tail loops, and the list merge sort was replaced with a
  stable bottom-up vector merge sort — 2M elements at ~362 MiB peak instead
  of ~32 GB retained.
- **Exact `input2` gradients for first-class losses and vector/learnable
  gamma** (#229): the real fix for a v1.3.2-evolve CHANGELOG claim that an
  adversarial audit found to be a no-op. See *Automatic differentiation*
  above for the full description.
- **Region-escape evacuator covers `PROMISE`** (ESH-0214e, #230) and the
  bignum-rational and parameter-object write-barrier follow-ups (#265,
  #267): closes the ESH-0214 series (a through e). Verified flat at
  ~116 MB under `ESHKOL_ARENA_POISON=1` over escape-then-force for both
  `delay` and `make-promise`.
- **ICC oracle hardening** (#232): the `input2` and evacuator fixes above
  are now load-bearing release gates, not one-off verifications — and, with
  quantum enabled, Bell-pair (200/200), CHSH (S ≈ 2.86, gate 2.4 < S ≤ 2.95),
  VQE-vs-exact-energy, and ML-KEM NIST-KAT gates join the matrix.

**v1.3.2-evolve.** Deepened the memory-safety and concurrency work:

- **Region-escape evacuator covers logic/workspace subtypes** (ESH-0214d,
  #226): `SUBSTITUTION`/`FACT`/`KNOWLEDGE_BASE`/`FACTOR_GRAPH`/`WORKSPACE`
  are deep-walked on escape instead of shallow-copied; `arena_destroy` is
  poisoned under `ESHKOL_ARENA_POISON` so region use-after-free crashes
  loudly.
- **Thread-safe region scope stack** (#217): `parallel-map`/future
  callbacks that opened a `with-region` no longer race on the shared
  current-arena slot.
- **Three deferred latent bugs triaged** (#215): ESH-0223 (named-let stack
  overflow at high iteration counts), ESH-0227 (apply-loop SIGBUS),
  ESH-0228 (`sleep-ms` argument type check).

**v1.3.1-evolve.** Two fixes closed the gap between "correct" and "safe to
run unattended" that the region-evacuator series above continues:

- **Flat memory in long-running loops** (ESH-0214b): per-iteration
  arena-scope reclamation, previously limited to named-let TCO loops,
  now also covers self-tail-recursive `define` loops, including a
  catch-all guard body in the loop. Verified on a 1,000,000-iteration
  loop: RSS 1,369 MB (unbounded growth) → 224 MB (flat).
- **Iterative S-expression reader** (ESH-0191): `read_list` no longer
  recurses one native stack frame per list element. Verified: the prior
  implementation crashed with SIGBUS reading a 20-million-element list;
  the rewritten reader completes cleanly at the same size.

**v1.3.0-evolve release gates** (green on the release SHA, and the base
this release line builds on): ICC readiness oracle 100/100, trace-verified;
CI 14/14 lanes including windows-arm64 lite/CUDA/XLA; SICP full-book gate
88/88 probes across all five chapters under both `-r` and AOT
(`scripts/run_sicp_smoke.sh`); reference-Scheme differential oracle 34/34
AGREE vs. chibi-scheme 0.12.0 on the P7a portable corpus
(`scripts/run_reference_differential.sh`).

**Permanent adversarial-testing infrastructure**, shipped in v1.3.0-evolve
and wired into the ICC release oracle rather than run once and discarded:
a multi-path differential harness with a seeded fuzzer, a feature-pair
edge matrix, an AD finite-difference oracle, a stress harness with
RSS/time budgets, a VM-parity ratchet, depth-parametric sweeps, and the
external reference-Scheme differential oracle. This is the same
infrastructure whose adversarial audit found the v1.3.2-evolve `input2`
overstatement above. See *docs/TESTING.md*.

**v1.2-line hardening** (carried forward, unchanged in this release):
fourteen audit blockers and seven critical/high security fixes closed
(subprocess shell-injection, Python-FFI AST-injection, integer-overflow
guards on arena/KB/image, path-traversal, TOCTOU, ReDoS, SQL-injection
guards, URL/header CRLF injection). See *docs/HARDENING.md* for the
itemized table with severities and resolutions.

---

## Web platform

Eshkol compiles to WebAssembly via `eshkol-run --wasm`, producing a
self-contained module that does not fall through to a native link step.
As of v1.3.3-evolve, `--wasm` output dead-strips unused stdlib, closing a
26x size regression on a small program that genuinely uses the stdlib:
5.57 MB → 60 KB (635 → 21 functions), with first-class functions and native
homoiconic display preserved, and a CI size gate now guards the bound.

The project website at https://eshkol.ai is itself an Eshkol program,
compiled to WebAssembly and served by GitHub Pages. The site embeds a
browser REPL where forward-mode automatic differentiation
(`(derivative (lambda (x) (* x x)) 3.0)` returning 6.0) runs through the
bytecode-VM dual-number propagation path without native code. The
interactive textbook has every example runnable in-browser.

The browser REPL uses the bytecode VM rather than LLVM JIT: an
opcode-dispatch register-plus-stack interpreter with 250+ native call IDs,
ESKB binary format with LEB128 encoding and CRC32 checksums
(*docs/DESIGN.md §Dual backend architecture*).

---

## Dual backend architecture

Two production execution backends share the same language semantics with
independent value representations:

- **LLVM native** (primary). 16-byte tagged values, ~30 codegen modules,
  ~85,500 lines, the default for `eshkol-run`.
- **Bytecode VM** (*lib/backend/eshkol_vm.c* plus its 32 *vm_\*.c* modules). A
  register-plus-stack interpreter with 250+ native call IDs, ESKB binary file
  format (section-based, LEB128, CRC32). Invoked via `eshkol-run input.esk -B
  output.eskb`. Coverage: arithmetic, closures, continuations, exception
  handling, tensors, complex / rational / bignum, logic / inference /
  workspace, hash tables, bytevectors, parameters, I/O. As of v1.3.3-evolve,
  `make-parameter`/`parameterize` are genuine runtime dynamic parameter
  objects — converters, a dynamic binding stack, correct unwinding — on both
  the native and VM execution paths, replacing a parser-level closure
  rewrite.

The weight-matrix transformer (*lib/backend/weight_matrices.c*, ~7,500
lines) is a third execution surface that exists for the SDNC paper and
the qLLM/transformer weight-loading pipeline. The strict-artefact contract
is 123 of 123 traced programs verified three ways (reference interpreter =
simulated transformer = matrix-based forward pass). Exports use the QLMW
binary format for qLLM consumption.

---

## Standard library

Auto-loaded stdlib modules compiled to `build/stdlib.o` via `--shared-lib`.
Namespaces:

- `core.functional.*` — composition, currying, combinators
- `core.list.*` — higher-order list functions
- `core.data.*` — JSON, CSV, Base64
- `core.strings.*` — thirty-plus string utilities
- `core.ad.*` — Taylor-tower AD stdlib layer added in v1.3.0-evolve: `guw`
  (multivariate mixed partials), `tensor_tower` (tensor-valued towers),
  `taylor_models` (validated enclosures), `checkpoint` (checkpointed
  reverse-over-Taylor), `taylor_numerics` (ODE/root/series-inversion
  solvers), `sparse_guw` (sparse Hessian recovery), `interval` (interval
  arithmetic support)
- `math.*` — special functions (Bessel, Gamma, Beta), ODE solvers
  (Euler, RK4), root finding, statistics
- `signal.*` — Cooley-Tukey radix-2 DIT FFT, IFFT, Hamming / Hann /
  Blackman / Kaiser windows, FIR and IIR direct-form, Butterworth
  low-pass / high-pass / band-pass via the bilinear transform
- `ml.*` — Adam, AdamW, L-BFGS, conjugate gradient, learning-rate
  schedulers
- `random.*` — PRNG with explicit seeding (`seed-prng!`), per-stream
  isolation; as of v1.3.3-evolve `quantum-random` and friends draw from
  Moonlab's Bell-verified QRNG when the opt-in quantum build is enabled,
  and are honestly labeled as a classical fallback when it is not
  (`eshkol_qrng_source_label()` reports which source is active)
- `web.*` — WASM / DOM API, HTTP fetch
- `tensor.*` — shape manipulation, stacking, broadcasting helpers
- `core.blc` — Binary Lambda Calculus, added in v1.3.2-evolve. A pure-Eshkol
  implementation of John Tromp's BLC: De Bruijn-indexed terms as
  homoiconic s-expressions, `blc-encode`/`blc-decode` for Tromp's
  self-delimiting bit encoding, normal-order `blc-eval`, a decoded
  232-bit universal machine (`blc-U`), BLC8 byte I/O, and ASCII lambda
  diagrams. Loaded on demand via `(require core.blc)`. See
  *docs/guide/BINARY_LAMBDA_CALCULUS.md*.
- `core.dbsp` — incremental dataflow, added in v1.3.3-evolve: Z-sets
  (weighted multisets) as a commutative group, the `z^-1`/`D`/`I` stream
  operators, incremental relational operators (including the three-term
  incremental join and multiplicity-correct `distinct`), and the generic
  incrementalizer `Q^Δ = D ∘ lift(Q) ∘ I`, in pure Eshkol. Acceptance gate
  27/27 under JIT and AOT. See *lib/core/dbsp.esk*.
- `agent.quantum` — Moonlab quantum state-vector simulation, added in
  v1.3.3-evolve: state creation, gates, measurement, expectation values,
  VQE with molecular Hamiltonians, and the `bell-chsh` CHSH experiment.
  Opt-in: built when the compiler is configured with
  `-DESHKOL_QUANTUM_ENABLED=ON`. See *lib/agent/quantum.esk*.
- `agent.pqc` — ML-KEM (FIPS 203) post-quantum key encapsulation, added in
  v1.3.3-evolve: `mlkem-keygen`/`mlkem-encaps`/`mlkem-decaps` at the
  512/768/1024 security levels over R7RS bytevectors, NIST-KAT-verified.
  Opt-in via `ESHKOL_QUANTUM_ENABLED`, with honest not-enabled stubs
  otherwise. See *lib/agent/pqc.esk*.

The stdlib uses `LinkOnceODR` linkage so user redefinitions cleanly
shadow stdlib functions without the historical "duplicate symbol"
link errors.

---

## Availability

- **Repository**: https://github.com/tsotchke/eshkol
- **Website**: https://eshkol.ai
- **Browser REPL**: https://eshkol.ai/learn
- **Licence**: MIT
- **Build prerequisites**: CMake 3.14+, LLVM 21, a C17 + C++20 compiler
  (GCC 11+, Clang 14+), Ninja recommended.
- **Build**: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`.
- **Homebrew tap**: `brew tap tsotchke/eshkol && brew install eshkol`; the
  tap formula carries the computed release SHA-256 after tagging.

---

## Citation

```bibtex
@software{eshkol2026,
  title    = {Eshkol: A Programming Language for Mathematical Computing},
  author   = {tsotchke},
  version  = {1.3.3-evolve},
  year     = {2026},
  url      = {https://github.com/tsotchke/eshkol}
}
```

The SDNC paper is *The Self-Differentiating Neural Computer: Computable
Transformers via Analytical Weight Construction* (tsotchke, 2026); the
companion repository is `noesis` and the artefact is `artifacts/paper/`.

---

## Contact

Press, programme committees, and academic correspondence:
**team@tsotchke.org**.
Security disclosures: **security@eshkol.ai** (see *SECURITY.md* for the
disclosure process and the supported-version table; initial response within
3 business days, fix-or-mitigation plan within 14 days for HIGH and CRITICAL).
