# Eshkol Arbitrary-Order Automatic Differentiation — Taylor-Tower Design

**Status:** proposal (design + Phase-0 POC landed; P1–P12 tracked in the ledger)
**Supersedes ceiling:** ESH-0118 (forward jet fixed at 2 perturbation levels → `derivative^n`/`gradient^n` = 0 for n ≥ 3)
**Builds on / subsumes:** PR #138 (8-component jet for reverse-over-nested-forward, ESH-0117)
**Author:** tsotchke
**Target:** post-v1.3.0-evolve (not a tag blocker — #138 already handles the reported order-≤2 nested case)
**Ledger:** ESH-0185 (P0, done) · ESH-0186..0197 (P1..P12, open)

---

## 0. Decision (TL;DR)

Adopt **univariate truncated-Taylor arithmetic** (a coefficient array to order K, computed via closed recurrences) as the single computational kernel for all high-order AD. Layer the following on top of that one kernel — and, per the "build in full" mandate, ship every layer as a real, gated phase rather than deferring any of them:

1. **Compile-time-K monomorphization as the primary path** — when the requested order is a literal at the call site (the overwhelmingly common case in a compiler), emit the entire tower as unrolled, stack-allocated, branch-free SSA IR. **Zero heap allocation in AD hot loops.** Runtime-K heap towers are the correctness fallback for dynamic order.
2. **Perturbation-confusion safety** — each active differentiation context gets a distinct epoch **tag** carried in the tower's reserved `flags` word, so nested `derivative`/`gradient` cannot silently cross-contaminate (§5a).
3. **Exact-coefficient towers** — a coefficient-type flag (`double` vs `rational`/`bignum`) is designed into the layout now, so high-order derivatives of polynomial/rational functions can be *exact*, reusing Eshkol's rational dispatch (§4, Phase P6).
4. **One FP-contraction policy across both tiers** — the unrolled-IR tier and the runtime-loop tier use the *same* contraction rule so `mono ≡ runtime` is bit-exact-achievable (§6a).
5. **Tensor-valued towers** — tower coefficients generalize from scalars to tensors so high-order AD covers the ML path (conv2d / attention / batchnorm / layernorm gradients). Built, not deferred (Phase P7).
6. **Griewank–Utke–Walther (GUW) directional interpolation** for arbitrary-order *multivariate* mixed partials (Phase P4), and **Taylor models** (validated AD with rigorous interval remainder) as a later phase (P8).
7. **Self-verifying recurrence table** — the shared X-macro table `taylor_recurrences.def` generates the emitter, the runtime kernel, *and* a per-primitive analytic d=8 correctness test, so adding a primitive auto-adds its gate (§5b).
8. **Structural back-compat** — order ≤ 2 keeps the existing 4-component jet byte-for-byte; the tower is a separate tagged representation selected only when order ≥ 3 is requested.
9. **Frontier differentiable programming** — after the numeric kernel is complete, P9–P12 lift it through Eshkol control flow, memory-efficient high-order reverse, user-facing series numerics, and sparse high-order tensors. That makes AD a property of the language, not just a property of scalar kernels.

This is the JAX-`jet` / Griewank & Walther approach, specialized to exploit that Eshkol is an AOT compiler rather than a runtime tracer, and hardened with perturbation tags, exact coefficients, tensor coefficients, validated (Taylor-model) arithmetic, and full differentiable-programming integration.

---

## 1. Problem

Eshkol forward-mode AD uses a fixed **4-component jet**:

```
{ value, d1·e1, d2·e2, d12·e1·e2 }   // exactly TWO perturbation levels
```

So `derivative^n` / `gradient^n` return an exact **0** at order ≥ 3 (ESH-0118, found by the P6a depth oracle, PR #133). PR #138 extended the live jet to **8 components** (adding a third ε level) to fix reverse-over-nested-forward (ESH-0117), but 8 = 2³ is the **hyper-dual wall**: each additional order *doubles* the struct. It cannot scale to arbitrary order.

We need exact derivatives to arbitrary order n, without exponential blowup, without regressing the order-≤2 hot path or the mixed-mode/reverse work in PRs #37/#39/#48/#84/#95/#113/#117/#138, and without the silent-wrong-nested-gradient trap (§5a).

---

## 2. Goals & non-goals

**Goals**
- Exact (to machine precision, or *bit-exact* on the rational/bignum path) `derivative^n` for arbitrary n.
- Exact `gradient`, `hessian`, `laplacian`, `directional` (order ≤ 2 unchanged; higher via GUW).
- **Correct nested differentiation** — no perturbation confusion, ever (§5a).
- No heap allocation on the common (literal-order) path.
- Order ≤ 2 semantics and performance strictly unchanged.
- Coverage of the ML AD path (tensor-valued towers) and rigorous enclosures (Taylor models).
- Every phase independently verifiable against the P6a oracle (`scripts/run_ad_depth.sh`, to d=8) and wired into the ICC release oracle.

**Non-goals (explicit)**
- Complex-domain Taylor / multicomplex — out of scope.
- Dense high-order *mixed* tensor *storage* API beyond GUW recovery remains a non-goal for the dense case; P12 adds sparse storage/recovery for structurally sparse tensors.

---

## 3. Alternatives considered

| Approach | Order-K cost | Verdict |
|---|---|---|
| **Nested dual / hyper-dual** | **2^K** components | Rejected — exponential; PR #138's 8-jet is already the 2³ wall |
| **Multicomplex** | ~2^K | Rejected — exponential, analytic-only |
| **Symbolic differentiation** | blows up | Rejected — not viable in a numeric compiler |
| **Finite differences** | cheap | Rejected — catastrophic cancellation at high order, inexact |
| **Truncated Taylor-mode** | **K+1** coeffs, O(K²) ops | **CHOSEN** — closed recurrences, exact, standard (Griewank & Walther Ch. 13; JAX `jax.experimental.jet`) |

Taylor-mode is the only exact method that is polynomial (not exponential) in the order. This is settled AD theory; the design work is the *Eshkol integration*, below.

---

## 4. Representation

Selected by requested order and whether that order is a compile-time literal:

| Tier | When | Storage | Recurrences |
|---|---|---|---|
| `AD_JET4` (existing) | order ≤ 2 | 4-field struct — **hot path, unchanged** | existing |
| `AD_JET8` (PR #138) | reverse-over-nested-forward, transitional | 8-field struct | existing; **subsumed in Phase 3** |
| `AD_TAYLOR` **stack** (primary new) | order ≥ 3, **K literal at call site** | `[K+1 x double]` alloca / SSA values — **no heap** | **unrolled** straight-line IR |
| `AD_TAYLOR` **heap** (fallback) | order dynamic (K not statically known) | `HEAP_SUBTYPE_TAYLOR` arena block | runtime loops |

**Heap layout** (mirrors `HEAP_SUBTYPE_TENSOR`, which stores homogeneous doubles as int64 bitpatterns). The coefficient-type flag (enhancement #2) and the perturbation tag (enhancement #1) are designed into the header **now**, so no re-layout is needed when P6/P7 land:

```c
// arena block behind a tagged HEAP_PTR with subtype HEAP_SUBTYPE_TAYLOR
typedef struct {
    uint32_t order_k;      // K: highest coefficient index (series has K+1 entries)
    uint32_t flags;        // packed: see bitfield below
    // coefficient storage follows, laid out per (flags & COEFF_MASK):
    //   COEFF_F64      : double        c[K+1]        (default, fast)
    //   COEFF_RATIONAL : esh_value_t   c[K+1]        (tagged rational/bignum, EXACT — P6)
    //   COEFF_TENSOR   : esh_value_t   c[K+1]        (each a HEAP_SUBTYPE_TENSOR ptr — P7)
    // (accessed through helpers; never raw-indexed across coeff types)
    union { double f64[/*K+1*/]; uint64_t bits[/*K+1*/]; } c;
} esh_taylor_t;
```

`flags` bitfield (the reserved word is now fully specified — see §5a for the tag):

```
bits [0..7]   COEFF_MASK    coefficient type: 0=F64, 1=RATIONAL, 2=TENSOR   (enh #2, #4)
bits [8..15]  RESERVED0     (dtype/rank hint for TENSOR coeffs; 0 otherwise)
bits [16..31] EPOCH_TAG     perturbation-confusion tag of the owning
                            differentiation context (enh #1, §5a)
```

Series semantics: `f(x₀ + t) = Σ_{k=0..K} c_k · t^k`, hence `f^(n)(x₀) = n! · c_n`.

**Tagged-value integration:** a Taylor value is a tagged `HEAP_PTR` with subtype `HEAP_SUBTYPE_TAYLOR` (new heap subtype constant). The tag byte / heap-subtype dispatch already exists for tensors, vectors, complex, bignum, rational — this adds one more. The 16-byte tagged value's data field `{4}` holds the arena pointer (per [[reference-architecture]]).

**Why runtime-K in the tag, but compile-time-K preferred:** K lives in the struct so dynamic-order calls work; but the *primary* path never allocates this struct — see §6.

---

## 5. The recurrence kernel

For input series `u`, `w` → result series `s`, all O(K) or O(K²):

```
mul   s = u*w        : s_k = Σ_{j=0..k} u_j · w_{k-j}                       (Cauchy convolution)
div   s = u/w        : s_k = ( u_k − Σ_{j=1..k} w_j · s_{k-j} ) / w_0
exp   s = exp(u)     : s_k = (1/k) Σ_{j=1..k} j · u_j · s_{k-j}
log   s = log(u)     : s_k = ( u_k − (1/k) Σ_{j=1..k-1} j · s_j · u_{k-j} ) / u_0
sin/cos (coupled)    : s_k = (1/k) Σ_{j=1..k} j · u_j · c_{k-j}
                       c_k = −(1/k) Σ_{j=1..k} j · u_j · s_{k-j}
pow   s = u^r        : s_k = (1/(k·u_0)) Σ_{j=1..k} (j·r − (k−j)) · u_j · s_{k-j}
sqrt, tan, atan, tanh, exp2, expm1, log1p : derived from the above / their own linear recurrences
```

Add/sub/scale are elementwise. Constants seed `c = {value, 0, 0, …}`; the differentiation variable seeds `c = {x₀, 1, 0, …}`.

All recurrences above are **verified numerically in the Phase-0 POC** (`tests/ad_taylor_poc/taylor_poc.c`) to d=8 at rel-err < 1e-12 (see §15).

### 5a. Perturbation-confusion safety (enhancement #1)

Nested differentiation such as

```scheme
(derivative (lambda (x) ((derivative g) x)))
```

is the classic **perturbation-confusion** trap: if the inner and outer differentiations share a perturbation, the outer derivative silently absorbs the inner one and you get a *wrong* gradient with no error. This is the Siskind–Pearlmutter problem; JAX solves it with per-trace *levels*. Eshkol adopts the same discipline, mechanized through the tower header:

- **Epoch tags.** Every dynamically-active differentiation context is assigned a distinct 16-bit **epoch tag** (a monotonically increasing counter per nesting entry, wrapping is a hard error). The tag is written into `flags[16..31]` (§4) of every tower seeded within that context. Compile-time-monomorphized towers carry the tag as an immediate constant in the emitted IR (a `constexpr` level id per lexical `derivative` site), so there is no runtime counter on the hot path.
- **Tag-gated combination.** Binary ops (`mul`, `div`, …) only *combine perturbations* of towers whose epoch tag equals the current context's tag. A tower carrying a **foreign** tag (an inner or outer level) is treated as a **constant** with respect to the current level: its order-≥1 coefficients are not differentiated at this level — the op uses only its `c[0]` value, exactly as a plain scalar would be. This is precisely JAX's "lift a value from an outer trace as a constant."
- **Seeding & extraction.** `seedDerivativeInput` stamps the current tag; `extractDerivativeResult` asserts the extracted tower's tag matches the requesting context and refuses (compile error for literal order; runtime trap for dynamic) on mismatch, so a leaked inner tower can never be silently read as an outer result.
- **Interaction with JET4/JET8.** The existing e1/e2 (and #138 e3) perturbation levels are the tag mechanism at orders ≤ 2 already; the tower generalizes the same idea to n levels. During P3, JET8's implicit levels are re-expressed as explicit tags so there is one confusion model across all tiers.

**Gate:** a dedicated nested-AD confusion suite — `(derivative (λ(x) (* x ((derivative (λ(y) (* x y))) 2))))` and friends — whose analytic answers are known, plus the existing P6a nested cells to d=8. A confusion bug shows up as an off-by-a-term derivative and is caught by the analytic reference.

### 5b. Self-verifying recurrence table (enhancement #5)

The recurrences live in a single X-macro table, **`taylor_recurrences.def`**, which is the *only* source of truth. It is consumed three ways:

```c
// taylor_recurrences.def  — one line per primitive
//   TAYLOR_OP(name, arity, RUNTIME_BODY, IR_EMIT_BODY, TEST_FN, TEST_X0, TEST_ORDER)
TAYLOR_OP(exp, 1, TR_EXP_LOOP, TR_EXP_EMIT, exp,        0.5, 8)
TAYLOR_OP(log, 1, TR_LOG_LOOP, TR_LOG_EMIT, log,        1.3, 8)
TAYLOR_OP(sin, 1, TR_SIN_LOOP, TR_SIN_EMIT, sin,        0.7, 8)
// ... etc
```

Three expansions of the same table:
1. **Runtime kernel** (`runtime_taylor.c`) — expands `RUNTIME_BODY` into the heap-loop implementation.
2. **IR emitter** (`autodiff_codegen.cpp`) — expands `IR_EMIT_BODY` into the unrolled LLVM builder calls.
3. **Auto-generated test** (`taylor_recurrences_test.c`, generated at build) — for each row, seeds `x` at `TEST_X0`, runs the tower to `TEST_ORDER`, and compares `k!·c_k` against `k`-th finite-order derivative of the C-library `TEST_FN` (obtained via a high-precision reference: exact for polynomials/rationals, and via the *coupled* analytic derivative pattern for transcendentals) at rel-err < 1e-12.

Consequence: **adding a primitive is a one-line table edit that automatically adds its d=8 correctness gate.** You cannot merge a recurrence without its test.

---

## 6. Compile-time-K monomorphization (primary path)

The Eshkol-specific win. Because we compile ahead-of-time and `derivative^n` / `((derivative-n k) f)` almost always has a **literal** k at the call site:

- Codegen detects a compile-time-constant order K.
- It allocates the tower as an LLVM `[K+1 x double]` on the stack (or as SSA values) — **no arena, no heap**.
- Each recurrence is **fully unrolled** into straight-line IR (the sums become explicit FMA chains — see §6a). No loops, no branches, no dispatch inside the AD computation.
- The epoch tag (§5a) is an IR immediate, so perturbation safety costs nothing at runtime here.
- Result: for constant K the entire n-th-derivative computation is branch-free numeric IR — competitive with or faster than a runtime tracer like JAX.

**Fallback:** when K is genuinely dynamic (e.g. `((derivative-n user-input) f)`), emit a call into the runtime heap-tower kernels (§4 heap tier, §5 loops).

**Correctness invariant (a gate):** for every K, the monomorphized result must equal the runtime result **bit-for-bit** — see the contraction policy below, which is what makes bit-exactness achievable rather than merely tight-ULP.

### 6a. FP-contraction policy (enhancement #3)

The convolution sums in §5 are dot products, and whether the compiler fuses `a*b + c` into a single `fma` changes the last bit. If the unrolled IR tier fused while the runtime-loop tier did not (or vice-versa), `mono(K)` and `runtime(K)` would differ by a few ULP and the metamorphic gate could only be a tolerance, not equality. Policy, applied to **both** tiers identically:

- **Emit explicit `llvm.fma.f64` for every multiply-accumulate in the convolution recurrences, in both the unrolled IR and the runtime kernel** (the runtime kernel is compiled with the same explicit `fma` calls — via `std::fma` / `fma()` — not left to the C compiler's `-ffp-contract` default). Reduction order is fixed to ascending `j` in both tiers.
- Equivalently stated as the single knob: **contraction ON and identical** across tiers, associativity fixed. (The alternative — `-ffp-contract=off` in both — is documented as the fallback if a target lacks fast `fma`; it also yields bit-exactness because neither tier then fuses.)
- Because both tiers now perform the *same* operations in the *same* order, `mono(K) ≡ runtime(K)` is **bit-exact**, and that is the P2 gate. Should a target ever be unable to honor identical contraction (e.g. a soft-float lib without `fma`), the gate degrades to a documented `≤ 2 ULP` bound and CI records which mode is in force.

**Files:** `lib/backend/autodiff_codegen.cpp` (the emitter + dispatch; extends `makeDual4`/`dualAdd`/`dualMul`/`dualField`/`seedDerivativeInput`/`extractDerivativeResult`), `lib/core/runtime_autodiff.cpp` + new `lib/core/runtime_taylor.c` (runtime kernels), `taylor_recurrences.def` (shared table, §5b).

---

## 7. Multivariate / mixed partials (Phase P4)

Full dense order-n derivative tensors in m variables are O(mⁿ) — inherently large. We use the **pragmatic split**, with the *complete* method built as a real phase:

- **`derivative^n` (univariate):** the tower directly. `f^(n) = n!·c_n`.
- **`gradient` / first-order multivariate:** existing forward/reverse (unchanged).
- **`hessian`, `laplacian`, order-2 mixed:** existing e1/e2 `AD_JET4` nesting (unchanged, fast).
- **Order ≥ 3 mixed partials:** **Griewank–Utke–Walther (2000)** — recover the full symmetric derivative tensor by propagating **univariate Taylor series along a set of directions** `f(x + t·v_i)` and interpolating. This adds *no new AD primitives*: it is orchestration + a linear solve over the same kernel.

**The hook that makes GUW additive (in the kernel from day one):** expose

```
taylor_propagate(f, x, direction v, K)  ->  series of f along v
```

as the public primitive. Given it, the GUW interpolation layer (P4) drops in without touching the kernel. Directional derivatives `∂^n f(x + t·v)` are a plain univariate tower along v.

---

## 8. Reverse-over-Taylor (high-order gradients, Phase P5)

`gradient` of a high-order scalar quantity (e.g. `∇` of a Laplacian) = the existing reverse tape with **tower-valued primals and adjoints**. The tape's forward pass records tower primals; the backward pass propagates tower adjoints. Perturbation tags (§5a) keep the reverse level distinct from the forward tower levels.

This is the **highest-risk** piece (reverse mode has crash history — [[project_ad_multivar_operators]] — and the pure-Scheme tape `lib/core/ad/tape.esk` has AOT constraints: no `for-each`/`map` in core modules — [[project_stateful_ad_tape]]). It lands behind its own oracle, AOT-verified.

---

## 9. Exact-coefficient towers (enhancement #2, Phase P6)

Eshkol already carries a full exact numeric tower — rationals and bignums with contagion rules ([[project_bignum_dispatch_pattern]]). For polynomial and rational functions the Taylor coefficients are themselves *exact rationals*, so high-order derivatives can be returned with **zero floating error** — a differentiator that no double-only AD framework offers.

- The `COEFF_MASK` field (§4) selects the coefficient type. `COEFF_RATIONAL` stores each `c_k` as a tagged `esh_value_t` (rational or bignum) rather than a raw double.
- The recurrences of §5 are reused verbatim, but the multiply-accumulate dispatches through `arith_->{mul,add,sub,div}` (the exact numeric tower) instead of `fma`. `div`/`pow` stay exact when the divisor is rational; `exp`/`log`/`sin`/`cos` fall back to `COEFF_F64` (they are transcendental — no exact rational series), with a clear predicate `taylor-exact?` telling the user which regime they are in.
- **Contagion:** seeding an exact input (`(taylor f (exact x0) k)`) yields exact coefficients through the whole polynomial/rational subgraph; the first transcendental op demotes to `COEFF_F64` with a recorded flag, matching Eshkol's existing exactness-contagion semantics.
- **Gate:** exact reference derivatives of `x^p`, `p(x)/q(x)` computed with the rational tower must match a symbolic/bignum oracle **bit-for-bit** (not just 1e-12), plus the `numeric_depth` exactness-contagion suite extended with a `taylor` column.

---

## 10. Tensor-valued towers (enhancement #4, Phase P7)

The AD smoke tests already exercise `conv2d` / attention / batchnorm / layernorm **gradients**; high-order AD must cover this ML path, not just scalars. We therefore generalize tower coefficients from scalars to **tensors**.

- `COEFF_TENSOR` (§4) makes each `c_k` a `HEAP_SUBTYPE_TENSOR` value; `RESERVED0` records the shared dtype/rank hint. All coefficients of one tower share a shape.
- **Elementwise ops** (`add`, `sub`, `scale`, and `exp`/`log`/`sin`/`cos` applied elementwise) lift directly: the recurrence structure is unchanged, only the per-coefficient scalar op becomes the corresponding tensor op.
- **`mul` → the appropriate bilinear tensor contraction.** Where the underlying primal op is a matmul, the Cauchy convolution's `u_j · w_{k-j}` becomes a matmul; where it is a **conv2d**, it becomes conv2d (`mul→conv where appropriate`); attention/batchnorm/layernorm decompose into these plus elementwise ops. The convolution *over series index k* is orthogonal to the tensor contraction *over spatial/feature axes* — they compose cleanly, reusing `lib/backend/tensor_conv_codegen.cpp` / `tensor_extras_codegen.cpp` for the inner op ([[project_tensor_op_codegen_path.md]]).
- **GPU:** tensor-coefficient towers ride the existing Metal/CUDA tensor path; the series loop stays on host, the per-index tensor op dispatches to device.
- **Gate:** high-order derivatives of `conv2d`/attention/batchnorm/layernorm vs. a finite-difference-checked reference and vs. the existing first-order tensor-AD results at order 1 (must agree). Wired into the AD smoke harness.

---

## 11. Taylor models — validated AD (enhancement #6, Phase P8)

A **Taylor model** is a Taylor polynomial plus a rigorous **interval remainder** `[lo, hi]` that provably encloses the truncation error over a domain box, giving *guaranteed* enclosure bounds — the basis of validated numerics, rigorous ODE integration, and verified global optimization.

- Representation extends the tower with an interval remainder term: `TM = (Σ c_k t^k) ⊕ R`, `R` an interval. Interval endpoints use directed rounding (round-down for `lo`, round-up for `hi`) — Eshkol's rounding-mode control is a prerequisite this phase adds.
- Each recurrence in §5 gets an **interval-remainder rule**: `mul` composes the polynomial parts by Cauchy convolution and bounds the discarded high-order tail into `R` with outward rounding; `div`/`exp`/`log`/`sin`/`cos` use the Lagrange/Cauchy remainder bounded over the domain box. This is standard Makino–Berz Taylor-model arithmetic, layered on the existing kernel.
- API: `(taylor-model f x domain k) -> (coeffs . remainder-interval)`; `taylor-model-enclosure` returns a guaranteed `[lo,hi]` for `f` over the box.
- **Gate:** for functions with known exact ranges, the returned enclosure must **contain** the true range (soundness), and its width must shrink as k grows (tightness) — checked against analytic bounds and against a brute-force fine-grid sampling that must always fall inside the enclosure.

---

## 12. Integration & back-compat

- **Dispatch order (perf-preserving):** scalar / `AD_JET4` first, `AD_JET8` next, `AD_TAYLOR` last. A tower only appears when order ≥ 3 is explicitly requested, so the hot path is never burdened.
- **New-numeric-type checklist** (per [[reference-bugs-fixed]]): add the `HEAP_SUBTYPE_TAYLOR` case to `arith_->{add,sub,mul,div,pow, exp,log,sin,cos,…}`, `compare`/`abs`, `convertTo*`, `findFreeVariablesImpl`, `number->string`, and the type checker. Missing any one silently corrupts.
- **JET4 unchanged; JET8 subsumed in P3:** keep #138's 8-jet behind its tests until reverse-over-nested-forward parity holds on the tower, then remove the special case. No churn on freshly-merged code.
- **Arena/tape interop:** the stateful tape must accept tower values; verified **AOT**, not just `-r`.

---

## 13. Eshkol API surface

```scheme
(derivative f)                 ; 1st derivative — unchanged
((derivative-n k) f)           ; k-th derivative via tower (k literal → monomorphized)
(derivative f #:order k)       ; equivalent keyword form
(taylor f x k)                 ; returns the K+1 Taylor coefficients as a tensor/list
(taylor-exact? series)         ; #t if coeffs are exact rational/bignum (P6)
(gradient f) (hessian f)       ; unchanged (order ≤ 2)
(laplacian f) (directional f v); unchanged
(hessian-n f order) ...        ; arbitrary mixed partials — P4 (GUW)
(taylor-model f x domain k)    ; validated enclosure (poly + remainder interval) — P8
```

All double-path results are exact to machine precision; the rational/bignum path (P6) is bit-exact; the Taylor-model path (P8) returns rigorous enclosures. `taylor` exposes the tower as a first-class value (useful for series methods, ODE integrators, etc.).

---

## Frontier: differentiable programming (Phases P9-P12)

P0-P8 make the Taylor tower the single arbitrary-order AD kernel. P9-P12 make that kernel a language facility: derivative operators can cross ordinary Scheme control flow, the reverse tape can scale to deep high-order graphs, users can program directly with series values, and multivariate recovery can exploit sparsity instead of materializing dense tensors by default. The target is a true differentiable-programming language, where loops, branches, closures, list combinators, ODE solvers, and sparse tensor workflows are differentiable when their semantics admit it.

### P9. Differentiable control flow

**Motivation:** Real Eshkol programs do not look like straight-line scalar kernels. They use `if`/`cond`/`case`, named-let loops, recursion, closures, higher-order calls, `map`, and `fold`. AD must be robust through those constructs, including nested derivative operators, or the tower remains only a numeric-kernel differentiator.

**Representation / algorithm sketch:** Tower values flow through SSA phis, loop-carried variables, closure captures, and higher-order call boundaries as ordinary tagged values. The epoch tag already reserved in `flags[16..31]` (§4, §5a) is the control-flow invariant: a tower seeded by the active derivative context keeps its tag across branches, tail calls, closure environments, and list combinators; a foreign-tag tower is lifted as a constant in the current context. This prevents perturbation confusion when an inner `derivative` is invoked inside a loop body, branch arm, recursive call, or mapped closure.

- **Tail calls / loops:** self tail calls and mutual-TCO lower to jumps/trampolines that carry tower values and the current epoch as part of the normal value environment. No new epoch is allocated for an iteration; only entering a new derivative operator creates a tag. Loop-carried towers become ordinary phi values for the stack-tier path, or heap tower pointers for dynamic K.
- **Branch selection:** AD differentiates the branch actually selected by the primal predicate, giving the derivative of the corresponding piece of a piecewise function. At a kink where branch predicates switch, the default policy is a deterministic differentiability error unless both arms agree through the requested order. A future opt-in keyword can request a documented subgradient policy for convex primitives, but silent arbitrary branch derivatives are forbidden.
- **Recursion and closures:** closure environments may capture tower values; extraction asserts the requesting epoch. Recursive calls pass tagged towers through the same ABI as scalars/heap values. Mutual recursion inherits the mutual-TCO discipline so differentiating structurally recursive programs does not regress into stack growth when the original program is tail-recursive.
- **Named-let and `map`/`fold`:** named-let lowers to the same loop/TCO path. `map` differentiates each closure application independently while preserving the caller epoch; `fold` differentiates the recurrence over its accumulator, so the accumulator may itself be a tower or a container of towers.

**Integration points:** `lib/backend/control_flow_codegen.cpp` for branch and loop phis; `lib/backend/tail_call_codegen.cpp` for self/mutual TCO; `lib/backend/call_apply_codegen.cpp` and `lib/backend/function_codegen.cpp` for closure calls/captures; `lib/backend/map_codegen.cpp` and `lib/core/list/*.esk` for list combinators; `lib/backend/autodiff_codegen.cpp`, `lib/core/runtime_autodiff.cpp`, and `lib/core/runtime_taylor.c` for seeding/extraction and tower dispatch.

**ICC/oracle gate:** extend `scripts/run_ad_depth.sh` with a control-flow oracle differentiating programs with data-dependent loops and branches to d=8 vs analytic references. The suite must include branch-selected polynomials, loop trip counts depending on primal data but fixed around the evaluation point, named-let recurrence examples, recursive/tail-recursive functions, nested derivatives inside branches, and `map`/`fold` closures. `scripts/run_control_flow_tests.sh`, `scripts/run_recursion_depth.sh`, and `scripts/run_tco_tests.sh` remain green; ICC records the new `ad_control_flow_depth` criterion.

**Risk note:** Piecewise functions are not differentiable at every point. The dangerous failure mode is silently returning the derivative of one arm at a kink, so the default kink policy is explicit error unless arm derivatives agree through K or an opt-in subgradient policy is declared. The other major risk is epoch leakage through closures; extraction assertions and foreign-tag-as-constant semantics catch it.

### P10. Checkpointed high-order reverse

**Motivation:** P5 reverse-over-Taylor makes high-order gradients possible, but a naive tower-valued tape stores O(tape*K) primal coefficient data. That is unacceptable for deep graphs, long loops, and ML workloads. High-order reverse needs checkpointing/rematerialization so memory scales with a controlled budget.

**Representation / algorithm sketch:** Adapt Griewank binomial checkpointing to a tower-valued tape. Instead of retaining every tower primal for every operation, the tape stores checkpoints at scheduled cut points: the primal inputs, epoch tag, tower order K, coefficient type, and enough AOT-stable closure/function identity to replay the segment. During the reverse sweep, missing tower primals are rematerialized by replaying the forward segment under the same epoch and FP-contraction policy, then tower adjoints propagate normally. The schedule is deterministic and part of the tape metadata, so replayed tower coefficients match the original forward pass bit-exactly when P2's contraction policy applies.

For the pure-Scheme tape (`lib/core/ad/tape.esk`), the checkpoint schedule must respect the existing AOT constraints: no reliance on host tracing, no unsupported `for-each`/`map` in core modules, and no runtime-only closures that cannot be regenerated by AOT. The checkpoint planner is therefore represented as explicit tape records and counted loops, not as higher-order Scheme iteration.

**Integration points:** `lib/core/ad/tape.esk` for checkpoint records and rematerialization schedule; `lib/core/runtime_autodiff.cpp` for reverse entry/exit and memory accounting; `lib/core/runtime_taylor.c` for tower replay kernels; `lib/backend/autodiff_codegen.cpp` for AOT emission of replayable segments; benchmark/oracle wiring in `scripts/run_ad_depth.sh` and ICC memory criteria.

**ICC/oracle gate:** a high-order reverse oracle runs a deep scalar graph and a loop-expanded graph at d=8, compares gradients against the dense-tape baseline and analytic references, and enforces a fixed memory budget that the naive O(tape*K) tape would exceed. The gate records max tape bytes, rematerialization count, and gradient agreement.

**Risk note:** Rematerialization is only valid for pure, deterministic segments. Effects, nondeterministic primitives, mutation not represented in the tape, or divergent FP contraction can make replay differ from the forward pass. The phase must conservatively disable checkpointing for impure segments and fall back to the dense tape with an explicit budget failure when no safe schedule exists.

### P11. Tower-based user numerics

**Motivation:** Once the tower is first-class, users should be able to program with Taylor series directly: Taylor-series ODE integration, series-based root finding and function inversion, and analytic continuation. This exposes the same kernel that powers AD as a general numerical method, not an internal-only derivative representation.

**Representation / algorithm sketch:** `taylor` returns an opaque series value backed by the same tower layout and recurrences (§4-§5). User methods consume and produce those series through public accessors rather than raw coefficient storage. ODE integration seeds a tower for time and recursively solves for the coefficients of `y(t+h)` from `y' = f(t,y)`. Root finding and function inversion use series Newton/Lagrange-style updates over tower values. Analytic continuation advances by repeated local Taylor expansions, optionally using P8 Taylor-model remainders to choose step sizes and validate enclosures.

**API sketch:**

```scheme
(define s (taylor (lambda (x) (/ 1 (- 1 x))) 0.0 8))
(taylor-coeff s 4)                         ; coefficient c_4
(taylor-eval s 0.25)                       ; evaluate the truncated series

(taylor-ode (lambda (t y) (* -1.0 y))
            1.0 #:t0 0.0 #:h 0.1 #:order 8 #:steps 10)

(taylor-root (lambda (x) (- (* x x) 2.0))
             #:x0 1.0 #:order 8 #:tol 1e-12)

(taylor-invert (lambda (x) (+ x (* x x)))
               #:around 0.0 #:order 8)

(analytic-continue (lambda (x) (/ 1 (- 1 x)))
                   #:from 0.0 #:to 0.5 #:order 8)
```

**Integration points:** `lib/math/ode.esk` for ODE-facing APIs; `lib/math/special.esk` and `lib/math.esk` for root/inversion/continuation exports; `lib/backend/autodiff_codegen.cpp`, `lib/core/runtime_taylor.c`, and `taylor_recurrences.def` for series operations; `docs/API_REFERENCE.md` once the APIs move from design to implementation.

**ICC/oracle gate:** an ODE/root-find oracle verifies that `y'=-y, y(0)=1` converges to `exp(-t)` as order/step refinement increases, that `x^2-2` converges to `sqrt(2)`, and that inversion examples compose back to identity through order d=8. ICC records the new `ad_user_numerics` criterion.

**Risk note:** A public series API can freeze internal representation too early. The API must expose opaque series operations and documented coefficient access, not raw tower headers. Analytic continuation can cross singularities; Taylor-model validation or explicit radius/domain guards must reject unsafe steps rather than returning plausible nonsense.

### P12. Sparse high-order tensors

**Motivation:** P4 GUW recovery can produce arbitrary-order multivariate mixed partials, but dense storage is O(m^n). Many Eshkol workloads are partially separable: sparse Hessians, banded Jacobian structure, local stencil functions, and ML blocks where only a small subset of mixed partials is nonzero. The GUW layer should exploit that structure instead of forcing dense tensors.

**Representation / algorithm sketch:** Add a sparse GUW recovery mode over the same `taylor_propagate(f, x, v, K)` primitive (§7). A conservative structural pass builds a variable-interaction hypergraph from codegen/type information and optional runtime probes. Seed directions are then selected using star-coloring / Walther-style sparse derivative compression so one directional Taylor propagation recovers multiple structurally independent mixed partials. Recovered entries are stored in a CSR/CSF-like format keyed by sorted multi-index:

```
order_offsets[n]        ; range of sparse entries for derivative order n
multi_index_ids[]       ; canonical sorted variable multi-index per entry
values[]                ; coefficient / tensor value
```

For Hessians this degenerates to standard sparse symmetric CSR. For order > 2, the same idea becomes compressed sparse fiber storage over canonical multi-indices. The algorithm must be conservative: an uncertain dependency is treated as possibly nonzero, which may reduce compression but cannot drop a true derivative.

**Integration points:** P4's GUW orchestration in `lib/backend/autodiff_codegen.cpp` and `lib/core/runtime_taylor.c`; dependency/free-variable analysis in `findFreeVariablesImpl` and the type/codegen environment; sparse tensor interop where available in tensor codegen; tests in the AD depth and metamorphic harnesses.

**ICC/oracle gate:** a sparse Hessian oracle for partially separable functions compares sparse recovery to dense GUW/Hessian recovery through order d, verifies identical nonzero values and conservative structural zeros, and records propagation count savings. Higher-order sparse tensor examples compare against dense recovery for small m,n where dense is still tractable.

**Risk note:** The main correctness risk is a false structural zero. Sparsity detection must be conservative, with dense fallback and metamorphic dense-vs-sparse comparisons on small cases. The main performance risk is coloring overhead dominating small problems; the sparse path should require either an explicit user request or an estimated win.

---

## 14. Phased implementation plan

Each phase gated by `scripts/run_ad_depth.sh` (derivative^d / gradient^d to d=8) and wired into the ICC oracle (`ad_depth` criterion). Ledger ids in the last column.

| Phase | Deliverable | Gate | ESH |
|---|---|---|---|
| **P0** | Standalone C POC (§15): univariate recurrences `t_mul/t_div/t_exp/t_log/t_sincos/t_pow`, x⁵ & sin & exp & 1/(1-x) & log(1+x) to d=8 | rel-err < 1e-12 vs analytic; exits 0 | **ESH-0185** (done) |
| **P1** | `HEAP_SUBTYPE_TAYLOR` + runtime heap tower + `derivative^n` (univariate) + epoch-tag scaffold (§5a) in `flags` | `derivative^d` PASS to d=8 → **closes ESH-0118**; nested-AD confusion suite PASS; nothing deprecated | **ESH-0186** |
| **P2** | Compile-time-K monomorphization (unrolled stack towers) + one FP-contraction policy (§6a) | correctness PASS + no heap in AD loop + metamorphic `mono(K) ≡ runtime(K)` **bit-exact** | **ESH-0187** |
| **P3** | Subsume `AD_JET8`: route reverse-over-nested-forward through the tower; re-express levels as explicit tags; remove 8-comp special case | #138's tests still PASS; JET4 retained; confusion model unified | **ESH-0188** |
| **P4** | GUW multivariate layer: `gradient^n` / mixed partials to arbitrary order via `taylor_propagate` + interpolation | `gradient^d` PASS to d=8 | **ESH-0189** |
| **P5** | Reverse-over-Taylor + tower-aware tape (high-order gradients) | reverse high-order oracle PASS, AOT-verified | **ESH-0190** |
| **P6** | Exact-coefficient towers (`COEFF_RATIONAL`, §9) | rational/bignum derivatives **bit-exact** vs symbolic oracle; exactness-contagion suite PASS | **ESH-0191** |
| **P7** | Tensor-valued towers (`COEFF_TENSOR`, §10): conv2d/attention/batchnorm/layernorm high-order AD | ML AD smoke PASS; order-1 agrees with existing tensor-AD; FD-checked | **ESH-0192** |
| **P8** | Taylor models (§11): validated AD with rigorous interval remainder | enclosure soundness (contains true range) + tightness (width ↓ in k) | **ESH-0193** |
| **P9** | Differentiable control flow: towers through loops, conditionals, recursion, closures, named-let, `map`/`fold`; perturbation-safe nested derivative contexts | data-dependent loop/branch oracle PASS to d=8 vs analytic; kink policy enforced; TCO/control-flow suites PASS | **ESH-0194** |
| **P10** | Checkpointed high-order reverse: Griewank binomial checkpointing/rematerialization for tower-valued tapes | high-order reverse on deep graph PASS to d=8 within memory budget; agrees with dense tape/analytic oracle | **ESH-0195** |
| **P11** | Tower-based user numerics: Taylor-series ODE integration, series root finding/inversion, analytic continuation APIs | ODE/root/inversion examples converge to analytic solution; ICC `ad_user_numerics` PASS | **ESH-0196** |
| **P12** | Sparse high-order tensors: sparse GUW mixed-partial recovery via star-coloring/Walther seed selection and CSR/CSF-like storage | sparse Hessian/partial tensor matches dense recovery through d; propagation-count savings recorded | **ESH-0197** |

Also: the self-verifying recurrence table (§5b) is introduced in P1 and every subsequent phase adds its primitives *through* the table, so each new primitive auto-adds a d=8 gate.

---

## 15. Testing & verification

- **P6a depth oracle** (`run_ad_depth.sh`) to d=8 is the primary gate at every phase; regenerate `scripts/icc_traces` then confirm via `icc readiness`.
- **Self-generated per-primitive tests** (§5b): every row of `taylor_recurrences.def` yields a d=8 analytic test automatically.
- **Metamorphic laws:** `mono(K) ≡ runtime(K)` (bit-exact, §6a); `taylor(f,x,k)[n]·n! == derivative^n(f,x)`; linearity/product/chain identities across orders. Wired into `.icc/…/metamorphic.jsonl`.
- **Nested-AD confusion suite** (§5a): analytic-answer nested `derivative`/`gradient` cases.
- **Analytic references:** x^p, exp, log, sin/cos, rational functions — closed-form derivatives to d=8.
- **Exactness (P6):** bit-exact vs bignum/symbolic oracle.
- **Tensor (P7):** FD-checked conv2d/attention/batchnorm/layernorm; order-1 agreement with existing tensor-AD.
- **Enclosure (P8):** soundness + tightness vs analytic bounds and grid sampling.
- **Differentiable control flow (P9):** data-dependent loops/branches, named-let, recursion, nested derivative-in-branch cases, and `map`/`fold` closures to d=8 vs analytic references.
- **Checkpointed reverse (P10):** deep high-order reverse graph within a fixed memory budget; dense-vs-checkpointed gradient agreement and rematerialization accounting.
- **User numerics (P11):** ODE/root/inversion examples converge to analytic answers as order/step refinement increases.
- **Sparse tensors (P12):** sparse Hessian/higher tensor recovery matches dense GUW on tractable cases; structural-zero detection is conservative.
- **Regression:** AD oracle 56/56, SICP 88/88, no perturbation-confusion regressions from the existing suites.
- **No AD-hot-loop allocation:** benchmark asserting zero arena growth for literal-K derivatives (P2).

### Phase-0 POC (delivered in this PR)

`tests/ad_taylor_poc/taylor_poc.c` — self-contained C, no repo deps; `tests/ad_taylor_poc/run.sh` compiles it with `cc` and runs it (exit 0 iff all checks pass). It implements `t_mul` (Cauchy convolution), `t_div`, `t_exp`, `t_log`, `t_sincos` (coupled), and `t_pow`, seeds `x = {x₀, 1, 0, …}`, and computes:

- `x⁵` (via `t_pow`, cross-checked via repeated `t_mul`) at x₀ = 2.0
- `sin(x)` / `cos(x)` (coupled) at x₀ = 0.0
- `exp(x)` at x₀ = 0.5
- `1/(1-x)` (via `t_sub` + `t_div`) at x₀ = 0.5
- `log(1+x)` (via `t_add` + `t_log`) at x₀ = 0.0

comparing `k!·c_k` against closed-form analytic derivatives for k = 0..8 at rel-err < 1e-12 (63/63 checks pass).

**Expected — x⁵ at x₀ = 2.0** (`f^(k) = k!·c_k`):

| k | c_k | f^(k) |
|---|---|---|
| 0 | 32 | 32 |
| 1 | 80 | 80 |
| 2 | 80 | 160 |
| 3 | 40 | 240 |
| 4 | 10 | 240 |
| 5 | 1 | 120 |
| 6 | 0 | 0 |
| 7 | 0 | 0 |
| 8 | 0 | 0 |

**Expected — sin(x) at x₀ = 0:** derivatives cycle 0, 1, 0, −1, 0, 1, 0, −1, 0 for k = 0..8.

Both confirmed by the POC run.

---

## 16. Risk ledger

| Risk | Mitigation |
|---|---|
| **Perturbation confusion (silent-wrong nested gradients)** | epoch tags in `flags`, tag-gated combination, extraction assert (§5a); dedicated confusion suite |
| Heap churn in AD hot loops | P2 monomorphization is the primary path; heap only for dynamic K |
| Two tiers drift | shared `taylor_recurrences.def` (self-verifying, §5b); metamorphic `mono≡runtime` gate |
| `mono ≢ runtime` at the bit level | one FP-contraction policy across both tiers (§6a); explicit `fma`, fixed reduction order |
| Reverse-over-tower bugs | isolated to P5, own oracle, AOT-verified, extra review |
| Dispatch-surface growth | strict ordering; tower absent unless order ≥ 3 requested |
| JET8 churn | subsumed only in P3 after parity, not preemptively |
| Exact-coeff blowup (bignum size at high K) | opt-in `COEFF_RATIONAL`; predicate `taylor-exact?`; transcendental demotion to F64 |
| Tensor-tower memory/compute cost | shares tensor path + GPU dispatch; series loop on host |
| Taylor-model unsoundness (remainder too tight) | outward-rounded interval arithmetic; grid-sampling containment gate |
| Control-flow AD at kinks | default differentiability error unless branch derivatives agree through K or an explicit subgradient policy is requested |
| Epoch leakage through closures/HOFs | epoch tag travels in tower flags; foreign-tag towers are constants; extraction asserts the requesting epoch |
| Checkpoint replay drift | checkpoint only pure deterministic segments; replay uses the same contraction policy; dense-tape fallback for impure segments |
| User numeric APIs ossify internals | expose opaque series operations, not raw tower headers |
| Sparse tensor false zeros | conservative dependency analysis; dense fallback; dense-vs-sparse oracle on small cases |
| Numerical stability of div/log/pow at high K near singularities | documented domain guards; NaN/Inf propagation tested (numeric_depth oracle) |

---

## 17. Open questions / future work

- Dense high-order mixed tensor *storage* API remains future; P12 covers sparse storage/recovery only.
- Optional monomorphization of small dynamic K∈{3,4} if profiles justify.
- Complex-domain towers / multicomplex — out of scope for now.
- Taylor-model-based rigorous ODE integrator and verified global optimizer built on P8 — future application layer.

---

*Phase-0 POC committed at `tests/ad_taylor_poc/`; P1–P12 filed in the ledger as ESH-0186..0197 (post-v1.3). Each phase carries its own ICC/oracle gate per §14.*
