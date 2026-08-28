# Reverse-mode tape lifetime

The compiler's reverse-mode operators allocate an internal tape for each
forward/backward pass. The tape is not a user-visible value: it records AD nodes,
their parent links, and the saved forward state needed by the reverse sweep.

## Mark and release

The native runtime exposes the tape lifetime operations
`arena_allocate_tape` and `arena_tape_release`. Allocation marks the owning arena
before the tape header, node-pointer array, and recorded nodes are created.
Release rewinds that marked interval after the caller has copied every result out
of the tape. Release is LIFO; an out-of-order release is rejected. This is the
lightweight lifetime boundary for a resident loop:

```text
for each training step:
    tape = arena_allocate_tape(arena, capacity)
    run forward pass and reverse pass
    copy gradients to the result
    arena_tape_release(tape)
```

Generated native `gradient`, `jacobian`, and runtime-closure gradient paths use
this boundary. A caller does not need to wrap each step in `(with-region ...)`,
and the result tensors are allocated before the tape interval is released.

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
