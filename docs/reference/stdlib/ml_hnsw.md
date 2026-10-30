# `core.ml.hnsw` — hierarchical navigable small-world index

**Source**: [`lib/core/ml/hnsw.esk`](../../../lib/core/ml/hnsw.esk)
**Require**: `(require core.ml.hnsw)` — not loaded by `(require stdlib)`.
**Design**: [ADR-0049](../../design/adr/0049-neuro-symbolic-bridge.md)

Approximate k-nearest-neighbour search over vectors (Malkov & Yashunin, 2018). Each point gets a level `floor(-ln(U) * mL)`, `mL = 1/ln(M)`, and is linked on every layer up to it to its `M` nearest already-inserted neighbours (`2M` on layer 0). A query descends greedily from the top-layer entry point and runs a beam search of width `ef` on layer 0. Levels come from an LCG seeded at creation, so the same points inserted in the same order give the same graph on every engine.

## Functions

- `(make-hnsw dim metric [M ef-construction])` — an empty index; `metric` is `'l2` (squared Euclidean distance) or `'cosine` (`1 - cos`); defaults `M = 8`, `ef-construction = 64`.
- `(hnsw-insert! idx vec payload)` — add a point; returns its id. A vector of the wrong dimension is an error.
- `(hnsw-search idx q k ef)` — the `k` nearest points as `(payload . distance)`, nearest first; `ef >= k` trades time for recall.
- `(hnsw-brute-force idx q k)` — the exact answer by a linear scan, the reference recall is measured against.
- `(hnsw-count idx)`, `(hnsw-dim idx)`, `(hnsw-vector idx id)`, `(hnsw-payload idx id)`, `(hnsw-distance idx a b)`.
- `(hnsw-index-kb! idx kb pred arity embed)` — index every fact `(pred a1 ... ak)` of a knowledge base under `(embed fact)`; returns the number indexed.
- `(kb-query-nearest idx q k [ef])` — the `k` facts whose embeddings are nearest `q`.

```scheme
;; hnsw-ex.esk
(require core.ml.hnsw)
(define idx (make-hnsw 2 'l2))
(hnsw-insert! idx (vector 0.0 0.0) 'origin)
(hnsw-insert! idx (vector 1.0 0.0) 'east)
(hnsw-insert! idx (vector 0.0 1.0) 'north)
(display (hnsw-search idx (vector 0.2 0.1) 2 16)) (newline)
(define kb (make-kb))
(kb-assert! kb (make-fact 'at 'paris 48.86 2.35))
(kb-assert! kb (make-fact 'at 'london 51.51 -0.13))
(kb-assert! kb (make-fact 'at 'rome 41.90 12.50))
(define geo (make-hnsw 2 'l2))
(display (hnsw-index-kb! geo kb 'at 3 (lambda (f) (vector (caddr f) (cadddr f))))) (newline)
(display (kb-query-nearest geo (vector 50.85 4.35) 2)) (newline)
```
```
((origin . 0.05000000000000001) (east . 0.6500000000000001))
3
((at paris 48.86 2.35) (at london 51.51 -0.13))
```

## Verification
[`tests/ml/hnsw_test.esk`](../../../tests/ml/hnsw_test.esk) checks exact answers on small sets, mean recall@10 of at least 0.95 against the linear scan over 400 random points, the cosine metric, the dimension guard and nearest-fact KB queries.
