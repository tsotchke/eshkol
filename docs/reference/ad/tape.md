# Reverse-mode tape lifetime

The compiler's reverse-mode operators allocate an internal tape for each
forward/backward pass. The tape is not a user-visible value: it records AD nodes,
their parent links, and the saved forward state needed by the reverse sweep.

## Mark and release

The native runtime exposes the tape lifetime operations
`arena_allocate_tape` and `arena_tape_release`. Allocation creates a dedicated
tape sub-arena for the tape header, node-pointer array, and recorded nodes; the
requested arena is only the sub-arena's parent for region teardown. Release
destroys that tape interval after the caller has copied every result out of the
tape, without rewinding or poisoning parent-arena allocations. Release is
rejected during an active reverse pass. This is the lightweight lifetime
boundary for a resident loop:

```text
for each training step:
    tape = arena_allocate_tape(arena, capacity)
    run forward pass and reverse pass
    copy gradients to the result
    arena_tape_release(tape)
```

Generated native `gradient`, `jacobian`, and runtime-closure gradient paths use
this boundary. A caller does not need to wrap each step in `(with-region ...)`,
and the result tensors and user values allocated by the differentiated function
remain valid after the tape sub-arena is released. If a tape is created inside a
region and is not explicitly released, region teardown destroys its child tape
arena with the parent.

The bytecode VM has a separate arena implementation. Its internal operator tape
is owned by the active `VmRegionStack`; VM programs use an enclosing region for
the same reclamation boundary. The VM's explicit low-level `ad-*` tape has its
own `(ad-tape-new)` / `(ad-tape-release tape)` lifecycle and is documented in
[`../stdlib/ad_tape.md`](../stdlib/ad_tape.md).

## Explicit low-level tape

The low-level `ad-*` builtins are a separate Wengert tape API. They allocate a
tape handle, append scalar nodes, run one reverse sweep, and read node values or
gradients:

```scheme
(define tape (ad-tape-new))
(define x (ad-var tape 3.0))
(define y (ad-mul tape x x))
(ad-backward tape y)
(display (ad-gradient tape x)) (newline) ; 6
(ad-tape-release tape)
```

`ad-tape-release` is idempotent. After release, the handle and its node indices
must not be used. The explicit API is available on native JIT/AOT and the
bytecode VM. This reference page is the AD tape documentation introduced by PR
#513; the complete builtin table and operation details remain in
[`../stdlib/ad_tape.md`](../stdlib/ad_tape.md).

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
