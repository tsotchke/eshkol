# Language-surface coverage gap — exposure engines

Ground truth: `tests/coverage/language_surface.json`
(regenerate with `python3 scripts/gen_language_surface.py`).
Live measurement: `tests/coverage/coverage_run.json`
(regenerate with `python3 scripts/language_coverage.py`).

This report answers one question: of everything an Eshkol program can invoke,
how much do the two generative exposure engines actually exercise today, and
which untested constructs matter most?

## The complete language surface

| Surface | Count | Source of truth |
|---|---|---|
| Builtins (all backends) | 958 | `eshkol_compiler.c` + `eshkol_vm.c` id-tables + `llvm_codegen.cpp` AOT dispatch |
| - in native first-class closure table | 427 | `eshkol_compiler.c BUILTINS[]` |
| - in bytecode VM table | 637 | `eshkol_vm.c BUILTINS[]` |
| - in LLVM AOT `func_name==` dispatch | 733 | `llvm_codegen.cpp` |
| - AOT-only (not in either id-table) | 299 | R7RS IO/mutation, NN/optimizer/linalg, atomics |
| Special forms (surface keywords) | 116 | `parser.cpp get_operator_type` + direct dispatch |
| AST op kinds (`eshkol_op_t`) | 112 | `inc/eshkol/eshkol.h` |
| Prelude higher-order fns | 16 | `eshkol_compiler.c scheme_prelude` |
| **Deduped user-facing surface** | **1033** | union of the above, minus `_`-internal helpers |

Note on the third backend: the task brief estimated ~639 builtins from the two
id-tables. That is the VM/native-closure surface only. The LLVM AOT backend
dispatches a further **299 builtins by call-head name** that are absent from the
id-tables — the entire R7RS file-IO / mutation surface (`set-car!`,
`call-with-output-file`, `with-exception-handler`, `eval`, `string-map`,
`vector-fill!`, ...), the neural-net / optimizer / linear-algebra surface
(`adam-step`, `xavier-normal!`, `tensor-svd`, `conv1d`, `multi-head-attention`,
`fft`, ...), atomics, and the extended numeric tower (`exp2`, `cbrt`,
`floor/`, `exact-integer?`). These are real, callable, and were being missed;
they are folded into the manifest with backend tag `native_llvm`.

### Builtins by risk category

| Category | Count | Category | Count |
|---|---|---|---|
| ffi_system | 274 | consciousness | 37 |
| tensor_ad | 218 | vector | 23 |
| numeric | 86 | hash | 22 |
| predicate | 70 | higher_order | 14 |
| geometry | 61 | misc_core | 13 |
| io_port | 48 | control_flow | 5 |
| string_char | 41 | misc (internal) | 8 |
| list_pair | 38 | | |

`geometry` (differential-geometry / Riemannian manifolds) is broken out of
`tensor_ad` because it is an independent, silent-wrong-prone numeric surface
(`manifold-exp-map`, `riemann-curvature`, `parallel-transport`, `se3-log`, ...).

## Current coverage: 102 / 1033 = 9.9%

Both engines together exercise 102 distinct surface constructs. The two engines
are complementary and narrow:

- `gen_generative_corpus.py` exercises the **R7RS-small core**: integer/real/
  ratio arithmetic, booleans, `list`/`cons`/`append`/`reverse`/`map`, a handful
  of string/char/vector ops, and the control forms `if`/`cond`/`let`/`let*`/
  named-`let`/`and`/`or`/`lambda`/`quote`. It is deliberately restricted to the
  intersection of Eshkol and chibi-scheme so it can differential-test.
- `gen_ad_adversarial.py` exercises the **AD + tensor** slice: `derivative`,
  `gradient`, `hessian`, `laplacian`, `tensor`, `reshape`, `tensor-sum/mul/
  dot/mean/matmul`, `conv2d`, `softmax`, `batch-norm`, `layer-norm`,
  `scaled-dot-attention`, plus scaffolding (`vector-set!`, `set!`, `begin`).

Covered per category (covered / total):

| Category | Covered / Total | Category | Covered / Total |
|---|---|---|---|
| numeric | 29 / 85 | vector | 8 / 23 |
| tensor_ad | 17 / 233 | string_char | 9 / 40 |
| list_pair | 10 / 38 | predicate | 10 / 70 |
| control_flow | 6 / 28 | higher_order | 3 / 20 |
| binding_form | 6 / 14 | io_port | 2 / 48 |
| macro_syntax | 1 / 8 | module | 1 / 8 |

Every other category — geometry, consciousness, hash, ffi_system,
memory_region, misc_core — is at **0% coverage**.

## Uncovered constructs, ranked by silent-wrong risk

The ranking puts the categories where a wrong answer is *silent* (a plausible
number instead of a crash) first. These are the highest-value gaps to close.

### Tier 1 — silent-wrong numeric / differentiable (close first)

1. **numeric — 56 / 85 uncovered.** The numeric tower is the classic
   silent-miscompile surface. Untested: `exact`/`inexact`/`exact->inexact`,
   `numerator`/`denominator`/`rationalize`, the transcendentals `asin`/`acos`/
   `atan`/`atan2`/`sinh`/`cosh`, `exp2`/`log2`/`log10`/`cbrt`, the R7RS
   division family `floor/`/`floor-quotient`/`truncate/`/`truncate-quotient`,
   `arithmetic-shift` and all `bitwise-*`, complex (`make-rectangular`,
   `real-part`, `magnitude`, `angle`, `conjugate`), and `add2/sub2/mul2/div2`.
2. **tensor_ad — 216 / 233 uncovered.** The engine exercises ~17 tensor/AD ops;
   the manifest holds 233. Untested: the entire reverse-mode AD-tape API
   (`ad-tape-new`, `ad-backward`, `ad-gradient`, `ad-mul`, ...), higher AD
   (`jacobian`, `taylor`, `derivative-n`, `divergence`, `curl`,
   `directional-derivative`), tensor linalg (`tensor-svd`, `tensor-qr`,
   `tensor-cholesky`, `tensor-solve`, `tensor-inverse`, `tensor-det`), the
   full optimizer/init surface (`adam-step`, `sgd-step`, `xavier-normal!`,
   `kaiming-uniform!`, `clip-grad-norm!`), activations/losses (`gelu`, `elu`,
   `silu`, `huber-loss`, `focal-loss`), pooling/conv (`conv1d`, `conv3d`,
   `max-pool2d`), `einsum`, `fft`/`ifft`, `multi-head-attention`, dataloaders.
3. **geometry — 61 / 61 uncovered (0%).** No engine touches differential
   geometry. `manifold-exp-map`, `manifold-log-map`, `parallel-transport`,
   `riemann-curvature`, `ricci-scalar`, `geodesic-distance`, `se3-log`,
   `so3-exp`, `mobius-add`, `poincare-distance`, `slerp`, ... All numeric,
   all silent-wrong-prone, all invisible today.
4. **control_flow — 22 / 28 uncovered.** `case`, `when`, `unless`, `do`,
   `guard`/`raise`/`with-exception-handler`, `call/cc`/
   `call-with-current-continuation`, `dynamic-wind`, `values`/
   `call-with-values`/`let-values`/`let*-values`, `delay`/`delay-force`/`force`,
   `match`. Control-flow miscompiles (wrong branch, dropped exception) are
   silent.
5. **consciousness — 38 / 38 uncovered (0%).** The neuro-symbolic engine
   (`unify`, `kb-assert!`, `kb-query`, `fg-infer!`, `free-energy`,
   `ws-step!`, and the DNC/SDNC memory ops) is entirely unexercised by the
   exposure engines.

### Tier 2 — structural / data (crashy but still important)

6. **higher_order — 17 / 20 uncovered.** `filter`, `fold-left`, `fold-right`,
   `for-each`, `reduce`, `compose`, `any`, `every`, `find`, `sort`, and the
   `parallel-*` variants.
7. **list_pair — 28 / 38 uncovered.** The full `c[ad]+r` family beyond
   `cadr`/`caddr`, `member`/`assoc`/`memq`/`assq`, `list-tail`, `iota`,
   `last`, `set-car!`/`set-cdr!`, `list-copy`.
8. **string_char — 31 / 40, vector — 15 / 23, predicate — 60 / 70, hash — 22 /
   22.** Bytevectors (0%), the `string-ci*`/`char-ci*` comparison families,
   `string-map`/`string-for-each`, `vector-fill!`/`vector-copy!`, all
   `char-*?` classifiers, and the entire hash-table API are untested.

### Tier 3 — breadth (mostly crash-visible)

9. **io_port — 46 / 48, ffi_system — 276 / 276 (0%).** File IO, ports,
   sockets, HTTP/WebSocket, tree-sitter, regex, process/signal, atomics,
   FFI pointers, time, compression. Mostly crash-visible, lower silent-risk,
   but large surface.
10. **binding_form / module / memory_region / macro_syntax.**
    `define-record-type`, `case-lambda`, `parameterize`, `define-values`,
    `letrec*`; `import`/`define-library`/`cond-expand`/`include`; the OALR
    ownership operators (`with-region`, `owned`, `move`, `borrow`, `shared`,
    `weak-ref`) at 0%; and the macro system (below).

## Claimed-but-unverified: macro hygiene

`macro_syntax` is 1/8 covered (only `quote`, via the `'` reader macro).
`define-syntax`, `syntax-rules`, `let-syntax`, `letrec-syntax`, `quasiquote`,
`unquote`, `unquote-splicing`, and `syntax-error` are **entirely unexercised**.

This category is doubly high-risk: the docs
(`docs/COMPLETE_LANGUAGE_SPECIFICATION.md` §3.10.1,
`docs/ESHKOL_QUICK_REFERENCE.md`) advertise *hygienic* `syntax-rules` macros,
but hygiene is exactly the property a differential/metamorphic engine would
stress (capture-avoiding renames, nested ellipsis). The project audit flagged
macro hygiene as claimed-but-not-fully-verified. Because no exposure engine
generates a single macro definition today, any hygiene defect is invisible.
Quasiquote/unquote templating is likewise untested end-to-end.

## What this means for the next phase

The manifest is the roadmap. To move the needle fastest, extend the generators
category by category in the Tier-1 order above:

1. Grow `gen_generative_corpus.py`'s typed grammar to emit the rest of the
   numeric tower, the `c[ad]+r`/list/assoc surface, `case`/`when`/`unless`/`do`,
   the higher-order library (`filter`/`fold-*`/`for-each`/`sort`), the
   string-ci/char-ci/`hash-*`/bytevector surfaces, and `define-record-type` /
   `let-values` / `guard`. All are differential-testable against chibi.
2. Grow `gen_ad_adversarial.py` (or a new AD engine) to cover the reverse-mode
   AD-tape API, `jacobian`/`taylor`/`derivative-n`, tensor linalg, the
   optimizer/activation/loss surface, and — as a distinct new engine — the
   differential-geometry and consciousness surfaces, each with an in-language
   oracle (FD for geometry, algebraic laws for `unify`/`kb-query`).
3. Add a dedicated macro/hygiene engine that generates `syntax-rules`
   definitions and asserts hygiene properties.

Re-run `scripts/language_coverage.py` after each change; the covered fraction
is the objective progress metric and is wired as an ICC completion-oracle
criterion (see `README.md`).
