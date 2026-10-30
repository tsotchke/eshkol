# `core.dbsp.ordered`

`(require core.dbsp.ordered)` maintains exact weighted ranks, nearest-rank quantiles
and top-k views under signed row updates.

```scheme
(define empty (make-dbsp-ordered car 3)) ; row score, fixed optional top-k view
(define step (dbsp-ordered-step empty
  (zset-from-weighted '((2 (10 alice)) (1 (20 bob))))))
(define state (vector-ref step 0))
(dbsp-ordered-quantile state 1/2)
(dbsp-ordered-top state 3)
```

The scorer is pure and stable and returns an exact rational. Rows obey the same
immutable-row contract as ordinary DBSP circuits. Ordering is by score, then the
existing canonical row encoding for deterministic ties. A tick accepts at most
65536 changed rows, consolidates signed weights, and requires every integrated
row multiplicity nonnegative. Absent/excess retractions and scoring failures
publish no successor.

A private persistent AVL tree retains each distinct row once, with its positive
exact integer multiplicity, height and subtree mass. Path copying and rotations
preserve old snapshots and forks. Counts can be bignums; selection and top-k split
weights without expanding occurrences. Exact arithmetic loss refuses.

`dbsp-ordered-select state rank` uses a zero-based weighted rank and returns
`#(row score first-rank multiplicity)`. `dbsp-ordered-rank state row` returns the
row's half-open rank interval or #f. `dbsp-ordered-quantile state q` accepts an
exact rational q in [0,1], selecting rank `max(0,ceil(q*mass)-1)`. Empty/out-of-range
selection refuses. Quantile arithmetic must remain exact.

`dbsp-ordered-top state k` returns a Z-set of the greatest k occurrences, splitting
a boundary row's weight when needed. k is an exact integer in 0–65536; a request
larger than the mass returns all rows. The returned Z-set's own canonical order
is not a presentation of ranking. Its outer storage is owned; row payloads remain
immutable shared values.

Step returns `#(successor top-view-delta work)`. If a fixed k was configured at
construction, its signed output delta integrates to the successor's top-k view.
Without configured k the delta is #f. `dbsp-ordered-info` returns
`#(distinct-rows total-mass tree-height)`. `dbsp-ordered-work` returns
`#(consolidated-changed-rows visited-index-nodes allocated-nodes rotations)`.
The latter records actual index update work, including temporary rotation nodes.

Point ranks/selection use tree-height work. Updates additionally pay for delta
consolidation, canonical key construction and scorer calls. Top-k materialization
visits the selected suffix and constructs a Z-set; configured top-view differencing
materializes both old and new views, so that cost depends on k and is not included
in index-node counters. Retaining history or growing logical data grows memory.
There is no flat-RSS, changefeed delivery, event-time window, approximate quantile,
checkpoint format or SQL/parser claim. A circuit's signed output can feed this
view explicitly through `dbsp-ordered-step`.

Tests compare every signed prefix with independent expanded full sorting, check
all weighted ranks and several quantiles, integrate top deltas, split ties, retain
history, balance ascending/descending construction, delete a complete index,
exercise bignum weights without expansion and bound local index work.
