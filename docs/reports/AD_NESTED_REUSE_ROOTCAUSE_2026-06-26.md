# Root cause: nested/higher-order AD corruption (ESH-0070) — pinpointed 2026-06-26

This supersedes the earlier "+/- jet is broken" hypothesis. The +/- symptom was a
red herring; the real bug is narrower and deeper. Investigated by exhaustive bisection
on a fresh `eshkol-run` built from the PR #75 branch (`fix/nested-ad-perturbation-esh0067`).
**The bug lives in PR #75's code** — #75 must NOT merge until this is fixed.

## Exact characterization (all are `(gradient (lambda(y)(gradient (lambda(x) BODY) y)) 7.0)`)
| BODY | expected f'' | got | verdict |
|---|---|---|---|
| `(* x x)` | 2 | 2 | OK |
| `(* (* x 2.0)(* x 2.0))` (inline) | 8 | 8 | OK |
| `(let ((t x)) (* t t))` | 2 | 2 | OK (t = the param, not computed) |
| `(let ((t (* x x))) t)` (bind, **use once**) | 2 | 2 | OK |
| `(let ((t (* x x))) (* t 1.0))` (use once) | 2 | 2 | OK |
| `(let ((t (* x x))) (* t t))` (**use twice**) | 588 | 196 | **BROKEN** |
| `(let ((a (* x 1.0))) (* a a))` (use twice) | 2 | 0 | **BROKEN** |
| `(* (+ x 0.0)(+ x 0.0))` | 2 | 0 | **BROKEN** |

## The invariant
- **First-order AD is always correct** (e.g. the 1st derivative of every case above is exact).
- 2nd order is corrupted **iff a *computed* dual (the result of any op, not a raw
  parameter) is bound to a local/let/loop variable and then *read more than once*.**
- Single use is fine; inline duplication (recomputing the subexpression) is fine. Only
  *reuse of one stored computed-dual* corrupts.
- The corruption is asymmetric across the two reads: `(let ((t (* x x))) (* t t))`
  yields 196 = exactly **one** of the four product-rule cross terms (a1·b2 = 14·14),
  the other three vanish. So the two reads of `t` return **different** dual contents.

## Why this is the Noesis named-let blow-up (Issue 1) AND the +/- symptom
- A named-let loop variable is a bound value re-read every iteration → same bug,
  compounded per iteration → `3.48e36`.
- `(+ x 0.0)` / `(abs x)` produce a *computed* dual; when that intermediate is
  materialized and reused (or bound), the cross term dies → looked like "+/- is broken".
- The mathematically-identical *straight-line* code (no rebinding/reuse) is exact —
  exactly what Noesis observed.

## Mechanism (from `-O0 --dump-ir` of `(let ((t (* x x))) (* t t))`)
- `t` is one alloca holding a tagged value; `t.load` and `t.load91` are two loads of
  the **same** tagged pointer → both point at the same heap dual struct. So the
  corruption is **not** in the load/store of the binding.
- The `*` operator path emits, in addition to the forward 4-jet arithmetic, a stream of
  **reverse-mode tape nodes**: `arena_allocate_ad_node_with_header` +
  `arena_tape_add_node` (many, interleaved with the two `load %dual_number` of `t`'s
  heap struct at IR lines ~14425 and ~14465). The forward 4-jet and the reverse tape
  are both live during a `gradient` that should be pure forward-over-forward.
- The two heap reads of `t` return different dual contents ⇒ the heap dual backing `t`
  is being **aliased/overwritten by the tape machinery** (an `ad_node`/dual arena
  allocation reusing or partially overwriting `t`'s 32-byte dual between the two reads),
  collapsing the cross/second-perturbation components on the second read.

## Two coupled problems to fix
1. **Correctness:** a `gradient` lowered to forward-mode 4-jets must not let the
   reverse-mode tape allocations alias the forward dual heap structs. Either (a) don't
   build a reverse tape at all on the pure-forward nested path, or (b) give the forward
   dual structs storage that the tape can never reuse/overwrite, or (c) represent a
   bound dual by value (carry the 4-jet in the binding) rather than via a shared heap
   pointer that the tape can clobber. (c) is the cleanest and also fixes reuse-aliasing
   generally.
2. **Performance (deal-breaker per user):** the per-op `arena_allocate_ad_node` +
   `arena_tape_add_node` on every arithmetic op inside `gradient` is the source of the
   per-call leak (a 20 000-iteration hoisted 2nd-order gradient SIGILLs; 5 000 is fine)
   and the µs-scale cost. The forward nested path should allocate O(1), not O(ops), tape
   memory — ideally zero for pure forward-mode. Fixing (1) via "no tape on forward path"
   addresses both at once.

## Verification gates for the fix (must all pass, `-r` AND AOT)
- The table above: 2, 8, 2, 2, 2, **588**, **2**, **2**.
- Noesis named-let: `post-loop'(0.05,k=1)` = -307.2; k=2..5 match finite-difference; no blow-up.
- 50 000-iteration loop of a 2nd-order gradient: completes, no SIGILL/SIGBUS, bounded memory.
- `tests/ad/nested_ad_test.esk` (10/10 from #75) still green; ad_input2/hessian/jacobian gates green.
- Report ns/call before/after (target: µs-scale, no per-call growth).

## Repro one-liners
```
./build/eshkol-run -r <(echo '(display (gradient (lambda(y)(gradient (lambda(x)(let ((t (* x x))) (* t t))) y)) 7.0))(newline)')   # want 588, get 196
./build/eshkol-run -r <(echo '(display (gradient (lambda(y)(gradient (lambda(x)(* (+ x 0.0)(+ x 0.0))) y)) 7.0))(newline)')          # want 2, get 0
```
