# ADR 0002 — Dense tensor AD nodes and a staged value-and-grad kernel

- Status: Proposed
- Date: 2026-07-09
- Deciders: AD/compiler cluster
- Supersedes: none
- Related: `docs/design/AD_STAGED_KERNEL_HANDOFF.md` (maintainer handoff, the
  consuming SciML/PINN side), `docs/design/AD_TAYLOR_TOWER.md`
- Scope: `lib/backend/autodiff_codegen.cpp`, `lib/backend/tensor_backward.cpp`,
  `lib/backend/llvm_codegen.cpp`, `lib/backend/tensor_arith_codegen.cpp`,
  `lib/backend/tensor_reduce_codegen.cpp`, `lib/core/runtime_autodiff.cpp`,
  `inc/eshkol/eshkol.h`, `inc/eshkol/backend/tensor_backward.h`

---

## 1. Context

Eshkol is a compiled R7RS Scheme on LLVM 21 in which automatic differentiation
is a compiler primitive, not a library. The AD substrate is already
semantically rich: a scalar reverse tape (`ad_node_t` / `ad_tape_t`), forward
jets, Taylor towers with arbitrary-order univariate derivatives, reverse-over-
forward and reverse-over-Taylor, and GUW higher-order helpers. Several dense
tensor backward kernels already exist. The consuming workload is JAX/Phydrax-
style SciML/PINN training: compile a tensor loss once, then run
`value + grad + update` many times over resident parameter buffers.

The maintainer handoff (`docs/design/AD_STAGED_KERNEL_HANDOFF.md`) claims the
semantic pieces are present but the *execution model* is wrong for training:
dense tensor ops scalarize, `gradient`/`jacobian` replay the user function per
component, hidden finite differences sit in higher-order paths, and there is no
staged `value_and_grad` kernel. This ADR independently verifies those claims
against current `master`, sets them against how JAX, Enzyme, PyTorch, Zygote and
Tapenade architect dense reverse-mode, and commits to a concrete architecture
and build sequence.

A separate correctness pass has already confirmed that the *scalarized* tensor
gradients are numerically correct — the ops decompose into many scalar AD nodes
but the adjoints flow and match closed form and finite-difference oracles
(`tests/integration/autodiff_tensor_test.esk:40-157`). The problem this ADR
addresses is therefore **not correctness**; it is **the scalarization and
staging architecture** — cost, tape size, memory lifetime, and the missing
compile-once/run-many boundary.

### 1.1 Verified current-state analysis

Every blocker in the handoff was checked against the source on this branch.
Verdicts, with file:line evidence:

| # | Handoff claim | Verdict | Evidence |
|---|---------------|---------|----------|
| 1 | `gradient` rebuilds a tape and replays `f` once **per input component** for tensor/vector inputs (not one-pass) | **TRUE** | `autodiff_codegen.cpp:5261-5278` component loop `i=0..n-1`; per-iteration tape alloc `5283-5286`, function call `5566`, `backpropagate` `5840`, tape reset `5880` |
| 2 | `jacobian` uses `(i,j)` double-loop replay (replays `f` per Jacobian entry), not row-sweep | **TRUE** | `autodiff_codegen.cpp:6588-6603` nested `i_out<m` / `j_in<n`; per-entry tape `6609-6610`, call `6732`, `backpropagate` `6832` |
| 3 | Dense tensor ops scalarize in AD mode (matmul, elementwise, reductions build scalar nodes) | **TRUE** | matmul nested M/N/K scalar `recordADNodeBinary(4,…)` mul + `(2,…)` add `llvm_codegen.cpp:27679-27726`; elementwise per-element `tensor_arith_codegen.cpp:517`; sum/mean add-chains `tensor_reduce_codegen.cpp:1948,2226,2236` |
| 4 | A dense `recordADNodeTensor` matmul callsite exists but is **bypassed** when the scalarizing AD branch runs | **TRUE** | dense record `llvm_codegen.cpp:27824`; guard `27783` `if (autodiff_ && ad_mode && !after_matmul_compute)` where `after_matmul_compute` is set exactly by the scalarizing branch `27623` |
| 5 | `eshkol_tensor_backward_dispatch` has default/skipped cases that **silently drop** gradients | **TRUE** | `tensor_backward.cpp:1362-1365` `default:` → bare `break;` ("gradient silently dropped"); explicit skip group `1348-1360` (ATTENTION/TRANSPOSE/SUM/BROADCAST_ADD/BROADCAST_MUL/EMBEDDING/FRECHET_MEAN). Also a known partial: attention backward "gradients flow only into V, never Q or K" `lib/bridge/tensor_backward.cpp:646` |
| 6 | Hidden finite differences in `gradientHigherOrder`, vector/tensor Hessian fallback, and Scheme tape | **TRUE** | central diff `autodiff_codegen.cpp:3464,3522,3558-3559`; Hessian epsilon fallback `7686-7689,8168-8169,8365`; `record-fd-op!` central diff `lib/core/ad/tape.esk:37,130,148-159` (`tape-fd-eps 1e-6`) |
| 7 | `ad_node_t` has the listed tensor fields in the stated order | **TRUE** | `inc/eshkol/eshkol.h:971-1013`: `tensor_value`983, `tensor_gradient`984, `input3`987, `input4`988, `saved_tensors`991, `num_saved`992, `params` union 995-1008, `shape`1011, `ndim`1012 |
| 8 | `recordADNodeTensor` exists; `propagateGradient` has a tensor fast path calling `eshkol_tensor_backward_dispatch` when `tensor_gradient != null` | **TRUE** | `recordADNodeTensor` def `autodiff_codegen.cpp:2334`; fast path `10134-10164`, dispatch call `10157-10159` |
| 9 | 13 dense backward kernels present | **TRUE** | `tensor_backward.cpp`: transpose100, reshape117, positional128, sum141, mean154, maxpool172, avgpool204, conv2d251, batchnorm428, layernorm485, matmul543/550, attention568, multihead684, embedding932 |
| 10 | No existing `value_and_grad` / `EshkolStagedKernel` / `eshkol_compile_staged_value_grad` | **CONFIRMED ABSENT** | symbols appear only in the handoff doc; no `.c/.cpp/.h/.esk` definition |
| 11 | No existing AD counters (`EshkolADCounters`, `primal_calls`) | **CONFIRMED ABSENT** | only in handoff doc; no implementation in `lib/`, `inc/`, `src/` |
| 12 | `ad_tape_t.variables`/`num_variables` exist and variables are appended | **PARTIAL / DEAD** | fields declared `inc/eshkol/eshkol.h:1028-1029`, only ever zero-initialized `runtime_autodiff.cpp:399-400`; never populated. "Variables appended to tape" is **FALSE** — the fields are dead |
| 13 | Taylor tower + reverse-over-Taylor hooks present | **TRUE** | `runtime_taylor.c` + `taylor_recurrences.def`; `eshkol_taylor_lift_ad_node`1377, `eshkol_taylor_extract_tangent`1359; `towerCtxPush`/`Pop` `autodiff_codegen.cpp:332,355` |
| 14 | Scalarized tensor gradients are numerically correct (`input2` done — correctness not the blocker) | **CORROBORATED** | `tests/integration/autodiff_tensor_test.esk:40-157` asserts reverse-mode tensor grads against closed form and finite-diff (sum→ones, `dot(x,x)`→`2x`, sum-of-squares→`2x`) |

Two verified facts sharpen the plan beyond the handoff:

- **`ad_tape_t::variables` is dead code (claim 12).** The handoff's one-pass
  gradient (Phase A) reads "all variable gradients" after a single backprop, but
  there is currently no list of variables to read. Phase A must *first* make the
  tape actually track its input variable nodes (populate `variables`/
  `num_variables`), or thread an explicit `ad_node_t** vars` bundle. This is a
  concrete prerequisite the handoff understates.
- **Silent gradient drop has a known-wrong instance already shipping** — the
  attention backward stub drops Q/K adjoints (`lib/bridge/tensor_backward.cpp:646`).
  This is not merely a future risk from recording more tensor nodes; it is a
  latent correctness bug behind an op that reports success. Strictness (blocker
  2.5 / Phase C.5) is therefore also a bug-containment measure, not only a
  performance guardrail.

---

## 2. SOTA context

How mature systems architect dense reverse-mode and staged value-and-grad, and
what Eshkol should borrow.

**JAX — linearize + transpose (VJP from JVP).** JAX does not hand-write reverse
rules per primitive. It computes a JVP, `linearize` splits primal from the
linear tangent computation, and `transpose` inverts that linear map to yield the
VJP; transpose rules exist only for *linear* primitives, so the transpose rule
table is far smaller than the JVP table
([JAX autodiff cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html),
[custom derivatives JEP](https://docs.jax.dev/en/latest/jep/2026-custom-derivatives.html)).
The staged form is explicit:

```
y, f_lin = jax.linearize(fun, *primals)   # stage once, cache the linear map
grads    = transpose(f_lin)(cotangent)    # reuse across cotangents
```

The linearized graph *is* the cache; evaluating it repeatedly is much cheaper
than re-tracing
([Linearization is all you need](https://antixk.netlify.app/blog/linearization_ad/)).
`value_and_grad` is exactly "run primal once, keep the value, run the transposed
linear map once." Because it operates on whole arrays, it never scalarizes.

**Enzyme — LLVM-IR-level, not a tape.** Enzyme differentiates the *optimized*
SSA instruction stream: it analyzes data dependencies and synthesizes an adjoint
routine that propagates adjoints in reverse execution order, handling loads/
stores and in-place updates as they appear after lowering. It is neither
source-text transformation nor a runtime tape; it is a compiler pass, so the
reverse pass is itself optimized
([Enzyme NeurIPS 2020](https://arxiv.org/pdf/2010.01709),
[Enzyme repo](https://github.com/EnzymeAD/Enzyme)). This is the endgame for a
fully static-shape straight-line kernel: emit the reverse pass as straight-line
LLVM, no runtime tape at all.

**PyTorch — per-op dense `grad_fn` nodes.** Each output tensor carries a
`grad_fn` node recording which op produced it and how to differentiate it; a
backward node saves exactly the forward tensors its VJP needs, edges connect it
to input nodes, and leaves get `AccumulateGrad`
([PyTorch autograd overview](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/)).
Crucially, one dense op = **one** node with a dense backward kernel — never one
node per scalar element. This is the model Eshkol's `recordADNodeTensor` +
`tensor_backward.cpp` were built for but that codegen bypasses (blocker 2.4).

**Zygote / Tapenade — source-to-source pullbacks.** Zygote generates the
backward pass as in-memory IR "as if written by hand"; each function has a
`pullback` (VJP) rule, and rules compose via ChainRules
([Zygote](https://github.com/FluxML/Zygote.jl),
[ChainRulesCore](https://juliadiff.org/ChainRulesCore.jl/stable/)). Tapenade
does the same on C source text. The shared abstraction across all of them: **a
per-primitive VJP/pullback rule that maps output cotangents to input
cotangents over whole arrays.**

**Synthesis for Eshkol.** The universal core abstraction is the **VJP /
pullback: a linear map from output adjoint to input adjoints, per dense
primitive, over whole tensors.** Eshkol's existing per-node dense backward
kernel (`eshkol_backward_matmul` etc.) *is* a VJP rule — the dispatch in
`propagateGradient` (`autodiff_codegen.cpp:10157`) is already a pullback
executor. What is missing is (a) codegen that records **one dense node per op**
instead of scalarizing, (b) **totality** (every recorded node has a real VJP or
the compiler errors), and (c) a **staged boundary** that stops rebuilding the
tape per call. Eshkol should not, in this ADR's horizon, build JAX's full
linearize/transpose partial-evaluation machinery; it already has an explicit
tape and dense VJP kernels, which is the PyTorch model and a far shorter path.
The linearize/transpose framing is retained as the *v2.0 aspiration* (Section 4)
because it is what unifies the forward-jet/Taylor machinery with reverse.

---

## 3. Decision

We adopt the handoff's direction but refine it into five concrete architectural
commitments.

### 3.1 Dense tensor AD-node model (adopt, PyTorch `grad_fn` shape)

A dense tensor op in AD mode records **exactly one** `ad_node_t` of an explicit
dense op type (`AD_NODE_MATMUL`, `AD_NODE_TENSOR_{ADD,SUB,MUL,DIV}_DENSE`,
`AD_NODE_SUM`, `AD_NODE_MEAN`, broadcast variants, activations). The node stores
dense parent node pointers (`input1..input4`), the dense forward result
(`tensor_value`), the minimal saved forward tensors for its VJP
(`saved_tensors`/`num_saved`), op params, and `shape`/`ndim`. Backward is a
single dense kernel dispatch through the existing
`eshkol_tensor_backward_dispatch` fast path (`autodiff_codegen.cpp:10134-10164`).
Tape size then scales with the count of *tensor* ops, not scalar element ops.
The forward numeric compute stays on the existing optimized dense path; AD mode
only changes what node is recorded, never the numerics.

Decision detail: **dense op IDs are distinct from scalar op IDs.** Never reuse a
scalar op ID for a tensor node (handoff C.3). Dispatch must key on a dense enum
so an unimplemented dense backward cannot alias onto a scalar rule.

### 3.2 VJP abstraction vs per-node backward (keep per-node, make it total and first-class)

We **keep the per-node dense backward dispatch** as the VJP executor — it is the
PyTorch/ChainRules model and matches the existing substrate — and we do **not**
build JAX-style linearize/transpose in this horizon. Two refinements make it a
proper VJP registry rather than an ad-hoc switch:

1. **Totality is enforced.** Every dense op ID that codegen can record MUST have
   a registered dense backward. The `default:` and skip cases at
   `tensor_backward.cpp:1348-1365` become: strict mode → hard error
   (`ESHKOL_KERNEL_UNSUPPORTED_AD`); release fallback → warn-once, and only when
   the compiler *explicitly* selected a numeric fallback. **Silent zero/missing
   gradient is prohibited.** Gate with `ESHKOL_AD_STRICT_TENSOR=1` (default in
   tests). Rule: *do not record a tensor node whose backward defaults to no-op.*
2. **Record-then-error is inverted to check-then-record.** Codegen queries a
   `tensorBackwardSupported(op_id, shapes)` predicate *before* recording; if
   false in strict mode it emits the unsupported-op error at the op site (better
   diagnostics than a reverse-pass failure).

This makes the per-node backward table the single source of truth for "what is
exactly differentiable," which is precisely the guarantee the SciML/PINN side
requires (exact AD or explicit unsupported error).

### 3.3 Staged `value_and_grad` kernel + ABI (adopt Phase G/H/I with an arena refinement)

Adopt the pointer/out-param C ABI (`EshkolStagedKernel`, `EshkolTensorDesc`,
`EshkolKernelCall`, `eshkol_compile_staged_value_grad` /
`eshkol_staged_kernel_run`) from handoff Phase G. Refinements:

- **Where intermediates live.** Not the global tape arena that resets per scope.
  A staged kernel owns a **dedicated resident arena** created at compile time,
  sized by a static memory plan (Phase I) keyed on the shape spec (Phase H).
  Parameter and gradient buffers are **caller-owned resident** buffers passed by
  pointer; forward intermediates and saved-for-backward tensors live in the
  kernel's scratch/resident arena; nothing in the hot path calls
  `arena_allocate_tensor_full`. Internal tensors are **views over planned
  scratch offsets**, not fresh allocations.
- **The tape is resident, not rebuilt.** For static-shape straight-line code the
  node graph structure is identical every call. Allocate the tape once
  (`tape_allocations = O(1)` across steps); per step, clear gradients without
  resetting `num_nodes` (new runtime helper `arena_tape_zero_gradients`), run one
  primal, run one reverse. This is the surgical realization of "no per-parameter
  tape rebuild" and is reachable without the Enzyme-style emitted-reverse.
- **Shape specialization is a guarded cache key.** Key = source identity +
  compiler/LLVM version + target triple + opt level + dtype/rank/dims/strides for
  every param/input/output + supported-op-set version + AD/staging flags. Runtime
  guard: shape mismatch → `ESHKOL_KERNEL_BAD_SHAPE`, never silent reinterpret.
- **The emitted-straight-line reverse (Enzyme-style) is explicitly deferred to
  v2.0.** It is the correct long-run design (no runtime tape, fully optimized
  reverse) but is a large surface; the resident-tape staged kernel captures most
  of the training-loop win first.

Start narrow (handoff G.2): f64 only, static contiguous row-major shapes,
straight-line + shape-static loops, ops = elementwise ±*/ , full sum/mean,
matmul, tanh/sigmoid/Stan; scalar loss; gradients w.r.t. param buffers;
coordinate JVP/Taylor for the few coordinate variables.

### 3.4 Making finite differences explicit (numeric API + counter + strict gate)

Policy, adopted: `(gradient f x)`, `(hessian f x)`, `(laplacian f x)`, and all
SciML/PINN derivative operators are **exact AD or an explicit unsupported
error** on the default path. Finite differences survive **only** behind
explicitly named numeric APIs (e.g. `finite-difference-gradient`,
`fd-hessian`), never as a hidden fallback inside the exact operators.

Non-breaking migration path (this matters — silent FD currently "works" for
users):

1. Add a `finite_difference_evals` counter (Section 3.5) incremented at every FD
   site (`autodiff_codegen.cpp:3464`, `7686`, `8365`; `tape.esk:148-159`).
2. Rename/relocate the FD implementations behind the explicit numeric API;
   `gradientHigherOrder`'s central-difference closure becomes
   `finite-difference-gradient` and is no longer what `(gradient f)` lowers to.
3. Where the exact path is not yet implemented (e.g. vector/tensor Hessian), the
   default operator returns the unsupported error under
   `ESHKOL_AD_STRICT`; a one-time deprecation warning names the explicit numeric
   API for users who genuinely want FD.
4. Tests assert `finite_difference_evals == 0` on every path claimed exact.

### 3.5 Composition with Taylor towers and higher-order AD (towers stay exact)

Dense tensor nodes join the **same** `ad_node_t` graph, so reverse-over-forward
and reverse-over-Taylor keep working through the existing hooks
(`towerCtxPush`/`Pop` `autodiff_codegen.cpp:332,355`;
`eshkol_taylor_lift_ad_node` `runtime_taylor.c:1377`). The decision for
higher-order in the SciML/PINN regime is the **mixed-mode split**, which is also
how JAX/PyTorch keep Hessian-vector products efficient:

- **Few coordinate variables** (x, y, z per collocation point): exact
  forward/Taylor jets for `u, ux, uy, uz, uxx, uyy, uzz` and the Laplacian. Reuse
  the existing exact Taylor tower — no dense-tensor Hessian.
- **Many trainable parameters**: one reverse sweep for the scalar loss.
- **Explicitly do not** attempt a full dense-tensor Hessian in this horizon; the
  epsilon Hessian fallback (`autodiff_codegen.cpp:7686-8365`) is replaced by the
  exact coordinate-Hessian-diagonal path for PINNs, or an unsupported error
  elsewhere. This keeps the "tower must stay exact" invariant and avoids the
  worst combinatorial and accuracy hazards.

Constraint this imposes on 3.1: for higher-order to compose, a dense backward
kernel must itself be expressible as differentiable tensor ops (backward-of-
backward). Until a given kernel is, its node is first-order-only and must be
strict-errored when lifted into a tower — not silently linearized.

---

## 4. Implementation trajectory

Sequenced by dependency and risk. Phase letters map to the handoff.

**v1.3.2 — shippable, surgical, low risk (no ABI surface).**
The scalar oracle is already correct (claim 14), so dense routing is validated
*against it* on tiny shapes — a differential test, not a leap of faith.

- **Phase A — counters + one-pass gradient.** Highest leverage. See 4.1.
- **Phase B — row-sweep Jacobian.** Depends on A's helpers +
  `arena_tape_zero_gradients`. Turns `m*n` replays into `m` (or `1+m`).
- **Phase C.1 / C.2 / C.5 — matmul dense routing, sum/mean dense nodes, strict
  unsupported.** C.1 removes the largest scalarization source (matmul inner
  loop); C.5 (strictness) must land *with or before* any new dense recording and
  also contains the shipping attention-Q/K silent-drop bug.
- **Phase F (scaffolding) — FD counter + strict gate.** The counter and
  `ESHKOL_AD_STRICT` flag land now even before every FD site is removed, so
  tests can assert exactness incrementally.

**v1.4 — feature-complete exact AD (moderate risk).**

- **Phase C.3 / C.4 — dense elementwise + broadcast backward** (needs reduce-
  over-broadcast-axes metadata).
- **Phase D — first-class `valueAndGradient`** primitive (flat parameter leaves,
  not PyTree ergonomics). Depends on A.
- **Phase E — exact PINN coordinate derivatives** (the mixed-mode split of 3.5).
- **Phase F (completion) — remove all hidden FD** from default operators.

**v2.0 — the training substrate (high risk, large surface).**

- **Phase G — staged kernel ABI** (`eshkol_compile_staged_value_grad`).
- **Phase H — static shape specialization** + guarded cache.
- **Phase I — kernel memory plan** (scratch offsets, views, no hot-loop alloc).
- **Phase J — staged optimizer epilogue** (SGD → Adam/Rprop).
- **Stretch — Enzyme-style emitted straight-line reverse** replacing the resident
  runtime tape for static-shape kernels.

### 4.1 First PR (concrete scope) — Phase A

Rationale: counters *define* "are we done" (one-pass vs component-loop is
measurable, not asserted), and the one-pass gradient is the single change that
most cuts primal calls. It touches no ABI and reuses the verified scalar oracle.

Deliverables:

1. **AD counters.** Add `EshkolADCounters` (`primal_calls`, `reverse_passes`,
   `tape_allocations`, `tape_nodes`, `scalar_ad_nodes`, `tensor_ad_nodes`,
   `tensor_backward_dispatches`, `tensor_backward_unsupported`,
   `finite_difference_evals`, …) + `eshkol_ad_counters_reset` /
   `eshkol_ad_counters_get` in `lib/core/runtime_autodiff.cpp` and a header. Wire
   increments at: tape allocation, `backpropagate` entry, the user-function call
   in `gradient`, `recordADNodeBinary`/`recordADNodeTensor`,
   `eshkol_tensor_backward_dispatch`, and every FD site
   (`autodiff_codegen.cpp:3464,7686,8365`, `tape.esk:148`). Expose
   `(ad-reset-counters!)` / `(ad-counters)` builtins.
2. **Make the tape track its variables.** Populate the currently-dead
   `ad_tape_t::variables` / `num_variables` (`inc/eshkol/eshkol.h:1028-1029`,
   zeroed at `runtime_autodiff.cpp:399-400`) when AD input variable nodes are
   created — this is the prerequisite the handoff omits.
3. **One-pass `gradient` refactor.** Replace the component loop
   (`autodiff_codegen.cpp:5261-5887`) for the tensor/reverse path with: build the
   AD input bundle once, one `callFunctionWithADInputs`, one `backpropagate`,
   then `emitReadAllVariableGradients`. Extract helpers
   `buildADTensorInputFromTensor`, `extractScalarOutputADNode`,
   `emitReadAllVariableGradients` (declared in `autodiff_codegen.h`). Preserve
   the scalar exact fast paths and nested-AD/capture handling; move captures
   behind helpers. No pointer-range heuristics for AD-node detection.
4. **`arena_tape_zero_gradients`** runtime helper (clear gradients without
   resetting `num_nodes`) — lands here so Phase B can consume it.
5. **Tests.** `tests/ad/value_grad_one_pass_test.esk` (`f(w)=Σ w_i²`, grad `2w`)
   and `tests/ad/tensor_gradient_one_pass_test.esk`
   (`loss(W,x)=Σ(Wx)²` vs analytic), each asserting **both** numeric correctness
   **and** counters: `primal_calls == 1`, `tape_allocations == 1`,
   `reverse_passes == 1`, `finite_difference_evals == 0`. Keep
   `tests/integration/autodiff_tensor_test.esk` green as the oracle.

Out of scope for PR 1: matmul dense routing (C.1), the ABI, any FD removal
(only the counter). Deliberately small.

---

## 5. Consequences

**Positive.**

- `gradient` cost drops from ~`n` primal calls / `n` tapes / `n` backprops to
  1/1/1 for a scalar loss; `jacobian` from `m*n` to `m`.
- Tape size scales with tensor-op count, not scalar element count — the matmul
  inner-loop scalar explosion (`llvm_codegen.cpp:27679-27726`) disappears, so
  dense backward kernels (`tensor_backward.cpp:543`) finally carry the work.
- The exactness guarantee becomes machine-checkable: `finite_difference_evals ==
  0` on the SciML/PINN path, enforced by tests and the strict flag.
- Silent gradient drops (incl. the shipping attention Q/K bug,
  `lib/bridge/tensor_backward.cpp:646`) become loud errors.

**Negative / costs.**

- Strict mode will turn some programs that *appear* to work (silent zero/partial
  grads) into hard errors. Mitigated by warn-once release fallback and an env
  override, but it is a real behavior change that must be release-noted.
- Two AD lowering regimes coexist during the migration (scalarized legacy +
  dense) until Phase C is complete — more surface to keep green.
- The staged ABI (v2.0) introduces a compile cache and resident arenas — new
  lifetime and invalidation logic.

**Neutral.**

- Forward numerics are unchanged; only the recorded node changes. Existing Taylor
  and reverse-over-Taylor tests (`tests/ad/*taylor*`, `reverse_over_taylor_test`)
  must stay green as a hard gate at every phase.

---

## 6. Risks and adversarial notes

- **Dense vs scalar divergence.** A dense backward kernel bug yields wrong
  gradients where the scalar path was right. Mitigation: every dense op is
  differential-tested against the verified scalar oracle on tiny shapes before
  the scalar path is retired; never retire the scalar path for an op until its
  dense kernel matches. The attention backward is already known-wrong — do not
  route it dense until Q/K adjoints exist.
- **Arena lifetime under staging.** `saved_tensors` needed for backward must
  outlive the forward pass. In a per-scope-resetting arena they can be freed
  before reverse runs. Mitigation: saved-for-backward tensors are owned by the
  kernel-resident arena for the kernel lifetime, and the memory plan (Phase I)
  marks each saved tensor live from its producer to its backward consumer;
  scratch reuse must never alias a still-live saved tensor. This is the single
  highest-risk correctness hazard of the whole program.
- **Dead `variables` field (verified claim 12).** One-pass gradient reads "all
  variable gradients" but the tape does not track variables today. If Phase A
  ships the one-pass loop without populating `variables`/`num_variables`, it will
  read garbage. This is called out as an explicit PR-1 prerequisite.
- **Shape-specialization blowup.** One compiled kernel per
  (dtype,rank,dims,strides,…) key is combinatorial across batch sizes.
  Mitigation: cache with a bounded size; bucket/pad batch dimensions; document
  that an unbucketed dynamic batch triggers recompile until a dynamic-shape path
  exists. Do not let the key omit strides (silent aliasing) or over-include
  (cache thrash).
- **Higher-order composition.** A dense first-order-only kernel silently lowered
  inside a Taylor tower would break the "tower stays exact" invariant.
  Mitigation: lifting a non-second-differentiable dense node into a tower is a
  strict error, not a silent linearization.
- **Strict-mode regressions in the existing suite.** Turning skip cases
  (`tensor_backward.cpp:1348-1360`) into errors may surface tests that were
  quietly relying on zero gradients. Triage each before flipping the default.

---

## 7. Alternatives considered

- **JAX-style linearize/transpose partial evaluation now.** Rejected for this
  horizon. It is the most elegant unification (JVP table + small transpose table,
  automatic VJP) and remains the v2.0 north star, but Eshkol already has an
  explicit tape and dense VJP kernels — the PyTorch per-node model is a far
  shorter path to the same training win. Revisit when the emitted straight-line
  reverse (Enzyme-style) is on the table.
- **Enzyme-style emitted reverse as the first step.** Rejected as first step,
  adopted as v2.0 stretch. Differentiating optimized LLVM IR gives the fastest
  reverse pass and no runtime tape, but it is a large, high-risk surface; the
  resident-tape staged kernel captures most of the loop win far sooner.
- **Keep scalarization, optimize the scalar tape.** Rejected. Scalarized matmul
  is the bottleneck (handoff §7.7); no amount of scalar-tape tuning reaches dense
  kernel throughput, and tape size stays proportional to element count.
- **Leave finite-difference fallbacks as a silent safety net.** Rejected —
  violates the hard SciML/PINN constraint (exact AD or explicit error) and hides
  accuracy bugs. FD stays only as an explicitly named numeric API.
- **Do nothing / more Scheme wrapper parity.** Rejected (handoff §7.1). The
  wrapper layer is not the bottleneck; the execution model is.

---

## 8. Sources

Code (this branch): `lib/backend/autodiff_codegen.cpp`,
`lib/backend/tensor_backward.cpp`, `lib/backend/llvm_codegen.cpp`,
`lib/backend/tensor_arith_codegen.cpp`, `lib/backend/tensor_reduce_codegen.cpp`,
`lib/core/runtime_autodiff.cpp`, `lib/core/runtime_taylor.c`,
`lib/core/ad/tape.esk`, `lib/bridge/tensor_backward.cpp`, `inc/eshkol/eshkol.h`,
`tests/integration/autodiff_tensor_test.esk` (line citations inline above).

Web:
- JAX autodiff cookbook — https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html
- JAX custom-derivatives JEP (JVP/VJP, transpose rules) — https://docs.jax.dev/en/latest/jep/2026-custom-derivatives.html
- "Linearization is all you need for an autodiff library" — https://antixk.netlify.app/blog/linearization_ad/
- Enzyme (Moses & Churavy, NeurIPS 2020) — https://arxiv.org/pdf/2010.01709
- Enzyme repository — https://github.com/EnzymeAD/Enzyme
- PyTorch autograd engine overview — https://pytorch.org/blog/overview-of-pytorch-autograd-engine/
- Zygote.jl — https://github.com/FluxML/Zygote.jl
- ChainRulesCore.jl (pullback/VJP rules) — https://juliadiff.org/ChainRulesCore.jl/stable/
