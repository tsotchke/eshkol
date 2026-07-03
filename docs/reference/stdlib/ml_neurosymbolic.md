# `core.ml.neurosymbolic` — differentiable symbol embeddings + soft unification

**Source**: [`lib/core/ml/neurosymbolic.esk`](../../../lib/core/ml/neurosymbolic.esk)
**Require**: `(require core.ml.neurosymbolic)` — **NOT** auto-loaded via `(require stdlib)`. Pure-Scheme module; calling its functions without the require fails with *"called undefined function 'make-embedding-table'"*. It internally `(require core.ad.tape)`s, so the tape is pulled in transitively.

The neuro-symbolic bridge. The consciousness engine's `unify` is an exact boolean match; **soft unification** replaces that with a **differentiable** similarity in `[0,1]` over learnable per-symbol embedding vectors, so a system can *learn* which symbols should unify (cat≈feline, cat≉rock) rather than only test syntactic identity. Built on [`core.ad.tape`](ad_tape.md): embeddings are parameters, each soft-unify is a tape forward (dot → sigmoid), and `tape-gradient` + SGD update the embeddings.

An **embedding table** is `#(dim entries)` where `entries` is a mutable alist `(sym . vector)`. Embeddings are initialised to small, **symbol-deterministic** values (a hash of the printed symbol seeds a small LCG), so distinct symbols start distinct with no RNG dependency.

## Functions

### `(make-embedding-table dim)`
Create an empty embedding table with `dim`-dimensional embeddings.

### `(emb-dim tbl)`
Return the embedding dimension.

### `(embed! tbl sym)`
Return `sym`'s (mutable) embedding vector, **creating** it (deterministically seeded) on first access. The returned vector is the live table entry — mutating it mutates the table (this is how training updates embeddings in place).

### `(vdot a b)`
Dot product of two equal-length numeric vectors. (Exported helper.)

```scheme
;; ns-basic.esk
(require core.ml.neurosymbolic)
(define tbl (make-embedding-table 8))
(display "dim ")(display (emb-dim tbl))(newline)
(define e (embed! tbl 'cat))
(display "emb len ")(display (vector-length e))(newline)
(display "vdot(e,e) ")(display (vdot e e))(newline)
```
```
dim 8
emb len 8
vdot(e,e) 2.41432
```

### `(soft-unify tbl a b)`
Differentiable unification confidence `sigmoid(<emb a, emb b> / sqrt(dim))` in `[0,1]`. Creates embeddings for `a`/`b` if new.

### `(soft-unify-loss tbl a b target)`
Squared error `(soft-unify(a,b) - target)^2`.

### `(soft-unify-train! tbl a b target lr)`
One differentiable SGD step that pushes `soft-unify(a,b)` toward `target` by updating **both** embeddings in place (via the tape). Returns the loss **before** the update.

```scheme
;; ns-train.esk — learn that cat unifies with feline
(require core.ml.neurosymbolic)
(define tbl (make-embedding-table 8))
(display "before ")(display (soft-unify tbl 'cat 'feline))(newline)
(let loop ((i 0))
  (if (< i 200) (begin (soft-unify-train! tbl 'cat 'feline 1.0 0.5) (loop (+ i 1)))))
(display "after  ")(display (soft-unify tbl 'cat 'feline))(newline)
```
```
before 0.443577
after  0.960393
```

### `(kb-attention tbl keys query)`
Differentiable retrieval: the `query` symbol attends over the `keys` (a list of symbols) by scaled-dot-product similarity, softmaxed. Returns an **alist** `(key . attention-weight)` whose weights sum to 1. As the embeddings are trained, the query routes to related keys — a neural query over a symbolic store.

### `(kb-retrieve tbl keys query)`
Argmax over `kb-attention`: return the single key the query attends to most.

```scheme
;; ns-kb.esk
(require core.ml.neurosymbolic)
(define tbl (make-embedding-table 8))
;; train cat<->feline so a 'feline query routes to 'cat
(let loop ((i 0))
  (if (< i 200) (begin (soft-unify-train! tbl 'cat 'feline 1.0 0.5) (loop (+ i 1)))))
(display (kb-attention tbl (list 'cat 'dog 'rock) 'feline))(newline)
(display (kb-retrieve tbl (list 'cat 'dog 'rock) 'feline))(newline)
```
```
((cat . 0.87199) (dog . 0.00842293) (rock . 0.119587))
cat
```

## Edge cases
- `embed!` returns the **live** table vector, not a copy — this is intentional (training mutates it). Copy it if you need a snapshot.
- Embedding init is deterministic per symbol name, so `soft-unify` of fresh symbols is reproducible across runs (the "before" value above is stable).
- Not exported (internal helpers, not for direct use): `ns-make-vec`, `ns-reverse`, `ns-scale`, `ns-sub!`, `ns-assq`, `ns-init`.

## Verification note
All functions verified under `eshkol-run -r` (outputs above are verbatim); `soft-unify` rises from 0.443577 toward 0.960393 after 200 training steps toward target 1.0, and `kb-retrieve` correctly routes a `feline` query to the trained `cat` key. No `.swarm` ledger issues reference these functions.
