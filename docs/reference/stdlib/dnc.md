# `core.dnc` — differentiable external memory (NTM/DNC head)

**Source**: [`lib/core/dnc.esk`](../../../lib/core/dnc.esk)
**Require**: `(require core.dnc)` — **NOT** auto-loaded via `(require stdlib)`. Note: the functions are **native codegen builtins** (heap subtype `HEAP_SUBTYPE_DNC`), so they actually resolve even without the require; require it anyway for clarity and portability. The `.esk` file only carries the `provide` list.

A learnable external memory bank of `N` rows × `W` columns that can be read and written **with gradients**. Addressing is by content (cosine similarity → softmax) or by location, controlled by a temperature `beta`: at high `beta` it behaves as a **bit-exact addressable store** (verified round-trip), at low `beta` it is smooth and differentiable (learnable). All vectors are Eshkol `#(...)` float vectors. The math mirrors `lib/backend/diff_memory_prototype.c` byte-for-byte (the C oracle / source of truth).

## Functions

### `(make-dnc-memory N W)`
Create an `N`×`W` memory bank, zeroed, usage counters at 0. Returns an opaque DNC handle.

### `(dnc-memory? x)`
Predicate: `#t` iff `x` is a DNC memory handle.

```scheme
;; dnc-basic.esk
(require core.dnc)
(define mem (make-dnc-memory 64 8))
(display (dnc-memory? mem))(newline)
(display (dnc-memory? 42))(newline)
```
```
#t
#f
```

### `(dnc-content-address mem key beta)`
Content-based addressing. `key` is a length-`W` `#(...)` vector; returns a length-`N` weight vector `softmax(beta * cosine(key, row_i))`. Higher `beta` sharpens toward the nearest row.

### `(dnc-loc-address addr beta N)`
Location-based addressing. Returns a length-`N` weight vector that is an indicator "bump" at integer address `addr` (sharp at high `beta`). Note `N` is passed explicitly here (it is not read from a memory handle).

### `(dnc-read mem wvec)`
Read: given a length-`N` weight vector `wvec`, returns the length-`W` weighted combination of memory rows.

### `(dnc-write! mem wvec erase add)`
NTM-style erase/add write (mutates `mem`). `wvec` is length-`N` (where to write), `erase` and `add` are length-`W`. A full-ones `erase` makes the write a pure overwrite (`write == add`). Returns `mem`.

```scheme
;; dnc-roundtrip.esk — write to a location, read it back (bit-exact at high beta)
(require core.dnc)
(define N 64)(define W 4)
(define mem (make-dnc-memory N W))
(define beta 10000.0)
(define w (dnc-loc-address 3 beta N))       ; address row 3
(dnc-write! mem w (make-vector W 1.0) (vector 1.0 2.0 3.0 4.0))
(display (dnc-read mem (dnc-loc-address 3 beta N)))(newline)
```
```
#(1 2 3 4)
```

### `(dnc-alloc-weights mem beta)`
Return a length-`N` allocation weighting over the **least-used** rows (dynamic memory allocation à la DNC). Sharper at higher `beta`.

### `(dnc-read-grad mem key target beta)`
Exact gradient of the content-addressed read loss. Returns a **pair** `(dkey . dmem)` where `dkey` is length-`W` (gradient w.r.t. the query key) and `dmem` is length-`N*W` (gradient w.r.t. the memory contents), for the loss `0.5 * ||read(content-address(key)) - target||^2`. Agrees with central finite differences to ~1e-6 (the C oracle reaches ~9.6e-9).

```scheme
;; dnc-grad.esk
(require core.dnc)
(define N 8)(define W 4)
(define mem (make-dnc-memory N W))
(dnc-write! mem (dnc-loc-address 0 10000.0 N) (make-vector W 1.0) (vector 0.5 -0.5 0.2 0.1))
(define key (vector 0.3 -0.5 0.2 0.8))
(define target (vector 1.0 -1.0 0.5 0.0))
(define gp (dnc-read-grad mem key target 2.0))
(display "pair? ")(display (pair? gp))(newline)
(display "dkey len ")(display (vector-length (car gp)))(newline)
(display "dmem len ")(display (vector-length (cdr gp)))(newline)
```
```
pair? #t
dkey len 4
dmem len 32
```

## Typical use — SGD learning on the key
The acceptance test `tests/dnc/dnc_test.esk` uses `dnc-read-grad` in a gradient-descent loop; `key -= lr * dkey` strictly decreases the read loss (observed 20068.5 → 5709.45 over 100 steps), and at high `beta` write→read is bit-exact (round-trip error < 1e-5).

## Edge cases
- Length mismatches (e.g. a `key` not of length `W`) raise inside the native op; the acceptance test deliberately does not exercise the raising path inline to avoid aborting the suite.
- `dnc-memory?` on any non-DNC value returns `#f` (does not error).

## Verification note
`tests/dnc/dnc_test.esk` passes 11/11 under `eshkol-run -r` (bit-exact round-trip, gradient-vs-FD < 1e-3, SGD loss decrease). No `.swarm` ledger issues reference the DNC builtins.
