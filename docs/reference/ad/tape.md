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
