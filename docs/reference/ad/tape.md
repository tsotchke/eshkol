# The reverse-mode tape builtins — `ad-*`

**Native name/id table**: [`lib/backend/eshkol_vm.c`](../../../lib/backend/eshkol_vm.c) `BUILTINS[]`, ids 390-409, 1841-1844, 2082-2088.
**LLVM dispatch**: [`lib/backend/llvm_codegen.cpp`](../../../lib/backend/llvm_codegen.cpp) (`ad-*` return types and `system_->ad*` handlers).
**Native runtime**: [`lib/core/ad_tape_builtins.c`](../../../lib/core/ad_tape_builtins.c) (sret wrappers), [`lib/backend/vm_autodiff.c`](../../../lib/backend/vm_autodiff.c) (the tape itself), [`lib/core/system_builtins.c`](../../../lib/core/system_builtins.c) (the counters).
**VM dispatch**: [`lib/backend/vm_native.c`](../../../lib/backend/vm_native.c).
**Counters**: [`lib/core/runtime_autodiff.cpp`](../../../lib/core/runtime_autodiff.cpp).
**Regression**: `tests/vm/ad_tape_lowlevel_regression.esk` (JIT, AOT and VM), `tests/ad/fd_counter_negative_test.esk`.

The `ad-*` builtins are the **explicit Wengert-tape API**: you allocate a tape,
record variables and operations on it by hand, run one reverse sweep, and read the
adjoints back. They are the low-level counterpart to the `gradient` / `jacobian` /
`hessian` **operators** documented in [operators.md](operators.md), which build and
walk their tape for you.

There are **33 names** across **30 native ids** — three of the names are aliases.
Every one of them is registered on both engines.

Three surfaces share the word "tape" and are easy to confuse:

| Surface | What it is | Reference |
|---|---|---|
| `ad-*` builtins | Compiler/VM builtins over the C `AdTape`. Scalars only. Fixed op set. | **this page** |
| `core.ad.tape` Scheme layer (`with-tape`, `tape-mul`, `record-op!`, …) | A separate, pure-Scheme tape that records an explicit backward **closure** per op, so custom ops and vector-valued nodes work. | [`../stdlib/ad_tape.md`](../stdlib/ad_tape.md) |
| The operator tape | The tape `gradient` and friends build internally. Never exposed as a handle. | [architecture.md](architecture.md) |

`core.ad.tape` re-exports the first row's names through its own `provide` block, so
requiring that module and calling `ad-var` reaches the builtin documented here.

## Engine availability

All 33 names are available on **native AOT**, **native REPL/JIT (`-r`)** and the
**bytecode VM**. `tests/coverage/language_surface.json` records
`"backends": ["vm", "native_llvm"]` for every one.

The two engines differ in three places, all of them in failure and instrumentation
behaviour rather than in results:

1. **Bad-tape return.** Given an argument that is not a live tape, the native path
   returns `-1` from every node-producing op (`ad_tagged_t` int, `ad_tape_builtins.c`)
   while the VM pushes the empty list `()` (`vm_native.c` cases 391, 392, 1844).
   `ad-tape-length` returns `0` on both.
2. **Tape ownership.** `(ad-tape-new)` on native calls `ad_tape_new_owned`, which
   pushes a scope on the global arena so `(ad-tape-release …)` can roll the whole
   tape back. On the VM the tape is allocated from the VM region stack and
   `ad-tape-release` is a *logical* release only — it nulls the heap object's payload
   pointer, and the arena memory returns at region pop.
3. **Counter scope.** The counters below are a process-global struct on native
   (`__eshkol_ad_counters`) and per-`VM` fields on the VM. A program that runs on
   one engine sees only that engine's counts.

## The 33 names

| Name | Id | Arity | Returns |
|---|---:|---:|---|
| `ad-tape-new` | 390 | 0 | tape handle |
| `ad-tape-release` | 1841 | 1 | `()` |
| `ad-tape-length` | 1843 | 1 | integer node count |
| `ad-const` | 391 | 2 | node index |
| `ad-var` | 392 | 2 | node index |
| `ad-add` | 394 | 3 | node index |
| `ad-sub` | 395 | 3 | node index |
| `ad-mul` | 396 | 3 | node index |
| `ad-div` | 397 | 3 | node index |
| `ad-pow` | 1844 | 3 | node index |
| `ad-sin` | 398 | 2 | node index |
| `ad-cos` | 399 | 2 | node index |
| `ad-exp` | 400 | 2 | node index |
| `ad-log` | 401 | 2 | node index |
| `ad-sqrt` | 402 | 2 | node index |
| `ad-neg` | 403 | 2 | node index |
| `ad-abs` | 404 | 2 | node index |
| `ad-relu` | 405 | 2 | node index |
| `ad-sigmoid` | 406 | 2 | node index |
| `ad-tanh` | 407 | 2 | node index |
| `ad-backward` | 408 | 2 | `()` (side-effecting) |
| `ad-gradient` / `ad-gradient-of` | 409 | 2 | real |
| `ad-node-value` / `ad-value` / `ad-value-of` | 1842 | 2 | real |
| `ad-reset-counters!` | 2082 | 0 | `()` |
| `ad-primal-calls` | 2083 | 0 | integer |
| `ad-reverse-passes` | 2084 | 0 | integer |
| `ad-tape-allocations` | 2085 | 0 | integer |
| `ad-finite-difference-evals` | 2086 | 0 | integer |
| `ad-counters` | 2087 | 0 | association list |
| `ad-note-finite-difference!` | 2088 | 0 | `()` |

**Every tape op takes the tape as its first argument.** `(ad-node-value node)` with
one argument does not read a node — it is an arity error on native and returns `()`
on the VM.

## Lifecycle

### `(ad-tape-new)` — id 390

Allocates a fresh tape and returns an opaque handle. No arguments.

```scheme
(define tp (ad-tape-new))
```

On native the tape owns a scope on the global arena; on the VM it is allocated from
the current VM region.

### `(ad-tape-release tape)` — id 1841

Releases the tape. **Idempotent** — a second release is safe, and so is releasing a
tape that was never owned: the native path is guarded by a magic sentinel so a
double release and a legacy non-owned tape both no-op. Returns `()`.

After release the handle is dead. Any further `ad-*` op on it is undefined from the
Scheme side; do not keep node indices across a release either.

```scheme
(define tp (ad-tape-new))
(ad-tape-release tp)
(ad-tape-release tp)     ; safe
```

Releasing matters in iterative fitting: without it, a loop that allocates a tape per
iteration grows the arena monotonically. That was the defect the owned-sub-arena
tape was introduced to fix.

### `(ad-tape-length tape)` — id 1843

Number of nodes recorded so far, `0` for a released or invalid tape. Useful as a
cheap assertion that a forward pass recorded what you expected.

```scheme
(define tp (ad-tape-new))
(define x (ad-var tp 2.0))
(define y (ad-mul tp x x))
(display (ad-tape-length tp)) (newline)
```

## Leaves

### `(ad-var tape value)` — id 392

Records a **differentiable variable** leaf holding `value` (an exact integer or a
real; native converts an int tagged value to double) and returns its node index.
Variables are the nodes whose adjoints you read after `ad-backward`.

```scheme
(define tp (ad-tape-new))
(define x (ad-var tp 3.0))
```

### `(ad-const tape value)` — id 391

Records a **constant** leaf. Structurally the same as `ad-var`; the distinction is
intent — a constant is not a gradient target. It is what you pass as `ad-pow`'s
exponent, which must itself be a node.

```scheme
(define tp (ad-tape-new))
(define three (ad-const tp 3.0))
```

## Recording operations

All of these take node **indices**, not values, and return a new node index.

> The recorders read `tape->nodes[i]` directly and are **not** bounds-checked
> (`lib/backend/vm_autodiff.c`). Pass only indices returned by an earlier op on the
> *same* tape. The accessors (`ad-gradient`, `ad-node-value`) *are* bounds-checked
> and return `0.0` for an out-of-range index.

### Binary — `(ad-add tape a b)` `(ad-sub tape a b)` `(ad-mul tape a b)` `(ad-div tape a b)` — ids 394-397

Record the arithmetic op between nodes `a` and `b`.

```scheme
(define tp (ad-tape-new))
(define x (ad-var tp 3.0))
(define y (ad-var tp 4.0))
(define s (ad-add tp x y))
(define p (ad-mul tp x y))
```

### `(ad-pow tape base-node exponent-node)` — id 1844

Both arguments are **nodes** — this is the distinction from `core.ad.tape`'s
`tape-pow`, whose exponent is a plain number. The forward value is `pow(base, exp)`;
the reverse derivatives are `∂/∂base = exp · base^(exp−1)` and
`∂/∂exp = value · ln(base)`.

Because the exponent derivative contains `ln(base)`, an exponent node that is a
gradient target over a non-positive base yields a non-finite adjoint. Record the
exponent with `ad-const` when you do not intend to differentiate through it.

```scheme
(define tp (ad-tape-new))
(define x (ad-var tp 2.0))
(define p (ad-pow tp x (ad-const tp 3.0)))
```

### Unary — `(ad-sin tape a)` `(ad-cos tape a)` `(ad-exp tape a)` `(ad-log tape a)` `(ad-sqrt tape a)` `(ad-neg tape a)` `(ad-abs tape a)` `(ad-relu tape a)` `(ad-sigmoid tape a)` `(ad-tanh tape a)` — ids 398-407

Record the named function of node `a`.

```scheme
(define tp (ad-tape-new))
(define x (ad-var tp 0.5))
(define h (ad-tanh tp (ad-mul tp x x)))
```

## Reverse pass and read-back

### `(ad-backward tape output-node)` — id 408

Runs the reverse sweep. It **zeroes every node's gradient first**, seeds
`output-node`'s adjoint to `1.0`, then walks the tape backwards from `output-node`
to node 0. Returns `()` — the result is the gradients left on the tape.

Two consequences follow from the zeroing:

- `ad-backward` is **re-runnable**. Calling it again with a different output node
  gives that output's gradients, not a sum of the two. There is no accumulation
  across sweeps.
- Nodes recorded *after* `output-node` are not visited, and their adjoints are left
  at zero.

An `output-node` outside `0 .. length−1` is a no-op.

```scheme
(define tp (ad-tape-new))
(define x (ad-var tp 3.0))
(define y (ad-mul tp x x))
(ad-backward tp y)
```

### `(ad-gradient tape node)` / `(ad-gradient-of tape node)` — id 409

The accumulated adjoint at `node` after `ad-backward`, as a real. Returns `0.0` for
an invalid tape or an out-of-range node — so a `0.0` here means *either* a true zero
gradient *or* a mistake. Check `ad-tape-length` if you need to tell them apart.

```scheme
(define tp (ad-tape-new))
(define x (ad-var tp 3.0))
(define y (ad-mul tp x x))
(ad-backward tp y)
(display (ad-gradient tp x)) (newline)      ; d(x*x)/dx at x=3
```

### `(ad-node-value tape node)` / `(ad-value tape node)` / `(ad-value-of tape node)` — id 1842

The forward (primal) value stored at `node`, as a real. Same `0.0`-on-failure
convention as `ad-gradient`.

```scheme
(define tp (ad-tape-new))
(define x (ad-var tp 3.0))
(define y (ad-mul tp x x))
(display (ad-node-value tp y)) (newline)
```

## Instrumentation counters

Seven builtins expose five counters. They exist so that a program can **prove** a
property of its own gradient path rather than assert it.

| Builtin | Counter | Incremented by |
|---|---|---|
| `ad-primal-calls` | `primal-calls` | each user-function evaluation the AD machinery performs |
| `ad-reverse-passes` | `reverse-passes` | each reverse sweep the operator path runs |
| `ad-tape-allocations` | `tape-allocations` | each operator tape allocated |
| — (read via `ad-counters` only) | `tape-nodes` | each node the operator path records |
| `ad-finite-difference-evals` | `finite-difference-evals` | `ad-note-finite-difference!` |

### `(ad-reset-counters!)` — id 2082

Zeroes all five. Returns `()`. Call it immediately before the region you want to
measure.

### `(ad-counters)` — id 2087

Returns all five as an association list, in this order:

```scheme
((primal-calls . N) (reverse-passes . N) (tape-allocations . N)
 (tape-nodes . N) (finite-difference-evals . N))
```

Both engines build the list by prepending in the reverse of that order, so the
ordering above is stable and may be relied on. `tape-nodes` is the only counter with
no standalone reader.

### `(ad-note-finite-difference!)` — id 2088

Reports **one** finite-difference perturbation evaluation and returns `()`. This is
the **write end** of `finite-difference-evals`, and it is the reason the read end is
an instrument at all.

The guarantee "no finite-difference fallback anywhere in the gradient path" has an
executable form: `(= (ad-finite-difference-evals) 0)`. Before this builtin existed
the increment function `eshkol_ad_count_fd()` had **zero callers on the native back
end**, so that equality was true by construction and would have stayed green if an FD
fallback had been introduced the next day. Every finite-difference site — compiler,
runtime or stdlib Scheme — now reports through here (`lib/core/ad/tape.esk`'s
`record-fd-op!` calls it once per perturbation, twice per input for a central
difference), so the assertion is a measurement and not a tautology.

`tests/ad/fd_counter_negative_test.esk` is the negative control: it deliberately
routes a backward pass through `record-fd-op!` and asserts the counter *rises*. A
gate that only ever checks for zero cannot distinguish "no FD happened" from "the
counter is broken"; the negative control is what separates them.

### Scope, and what the counters do not measure

The tape counters instrument the **operator** AD path (`gradient`,
`reverse-gradient`, and the machinery in `autodiff_codegen.cpp` /
`runtime_autodiff.cpp`), not the explicit `ad-*` tape documented above. Neither
`(ad-tape-new)` nor any `ad-*` recorder increments `tape-allocations` or
`tape-nodes` on either engine — the explicit tape is a different C type
(`AdTape` in `vm_autodiff.c`) from the operator tape (`ad_tape_t` in
`runtime_autodiff.cpp`), and only the latter is counted. A hand-built tape therefore
reads `0` allocations, which is correct rather than a defect: nothing on the operator
path was used.

`finite-difference-evals` is the exception — it is engine- and path-independent,
because it is written explicitly by whoever performs a perturbation.

```scheme
(ad-reset-counters!)
;; ... run the gradient computation under test ...
(display (ad-finite-difference-evals)) (newline)   ; 0 is the exactness assertion
(display (ad-counters)) (newline)
```

## See also

- [operators.md](operators.md) — `derivative`, `gradient`, `jacobian`, `hessian`,
  `laplacian`, `directional-derivative`, `divergence`, `curl`, `diff`.
- [architecture.md](architecture.md) — the forward jet, the operator tape, the AD
  node registry and the exact geometric backwards.
- [support-matrix.md](support-matrix.md) — the AD-oracle matrix and how to run it.
- [`../stdlib/ad_tape.md`](../stdlib/ad_tape.md) — the pure-Scheme `core.ad.tape`
  module, custom ops, vector-valued nodes, `with-tape`.
- [`../../guide/AUTOMATIC_DIFFERENTIATION.md`](../../guide/AUTOMATIC_DIFFERENTIATION.md)
  — conceptual guide.
