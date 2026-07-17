# `core.dbsp` — DBSP incremental dataflow (Z-sets)

**Source**: [`lib/core/dbsp.esk`](../../../lib/core/dbsp.esk)
**Require**: `(require core.dbsp)` — not auto-loaded; depends on `stdlib`.

A pure-Eshkol reference implementation of the DBSP calculus (Budiu, Chajed,
McSherry, Ryzhyk, Tannen, VLDB 2023): Z-sets, the stream operators
z^-1 / D / I, and incremental (delta-to-delta) relational operators. This is
the first slice of the incremental-dataflow spine adopted by
[ADR 0009](../../design/adr/0009-incremental-dataflow-dbsp.md) — zero compiler
changes, ordinary R7RS-shaped Eshkol Scheme, in the precedent of `core.blc`.

Acceptance gate: `tests/stdlib/dbsp_test.esk`, 27/27 under both JIT and AOT.

## Concepts

**Z-set** (Z[A]): a finite integer-weighted multiset — a finite map from rows
to **exact** integer weights (bignums included). Weight +1 inserts one
occurrence, -1 retracts one; negative weights make retraction a group
operation, so Z-sets form a commutative group under addition. A Z-set is
always **consolidated** at an operator boundary: `equal?` rows carry one
summed weight and zero-weight rows are absent.

The external representation is a pointer-free Eshkol value:

```
#(dbsp-zset-v1 ((weight row) ...))
```

Entries are ordered deterministically by a canonical injective rendering of
`row`, so iteration, equality, and serialization are reproducible across
native, JIT, bytecode, and WASM. Row equality remains `equal?`; the canonical
string is only an ordering/indexing aid. Supported row domains: symbols,
exact numbers, strings, chars, booleans, and lists/vectors of these.

**Stream**: a finite sequence s : {0..n-1} -> Z[A], represented as an ordinary
list of Z-sets indexed by logical tick t (the finite "stream runner" of
ADR 0009's v1.5.0 slice; a standing runtime with an unbounded clock is a later
slice).

The three DBSP stream operators:

```
z^-1 (delay):      (z^-1 s)[0] = 0,  (z^-1 s)[t] = s[t-1]
D (differentiate): D(s) = s - z^-1(s)          snapshots -> changes
I (integrate):     I(s)[t] = sum(i=0..t) s[i]  changes  -> snapshots
```

D and I are mutual inverses: **D . I = I . D = identity**. That identity is
what makes incremental view maintenance a calculus: for **any** batch query
Q : Z[A] -> Z[B], the incremental form

```
Q^D = D . lift(Q) . I
```

maps an input delta stream to the corresponding output delta stream.
**Linear** operators (map, filter, project, union) are their own incremental
form — apply them directly to each delta. **Bilinear** join incrementalizes
via the discrete product rule (below).

## Z-set commutative-group core

| Procedure | Signature | Notes |
|-----------|-----------|-------|
| `zset-empty` | `(zset-empty)` | The additive identity |
| `zset?` | `(zset? x)` | Recognizer |
| `zset-singleton` | `(zset-singleton row weight)` | Weight 0 yields the identity; non-integer weight raises |
| `zset-from-weighted` | `(zset-from-weighted entries)` | Build from a `((weight row) ...)` list: validates exact integer weights, combines duplicates, drops zeros, canonicalizes |
| `zset-weight` | `(zset-weight zs row)` | Consolidated weight of `row`, default 0 |
| `zset-add` | `(zset-add a b)` | Group addition |
| `zset-negate` | `(zset-negate a)` | Group negation |
| `zset-sub` | `(zset-sub a b)` | `a + (-b)` |
| `zset-scale` | `(zset-scale k zs)` | Multiply every weight by exact integer `k` |
| `zset-consolidate` | `(zset-consolidate zs)` | Defensive renormalization to canonical form |
| `zset-empty?` | `(zset-empty? zs)` | |
| `zset-positive?` | `(zset-positive? zs)` | Every weight > 0 |
| `zset=?` | `(zset=? a b)` | Group equality (canonical entries coincide) |
| `zset-entries` | `(zset-entries zs)` | Canonically ordered `(weight row)` entries, nonzero weights only |
| `zset-rows` | `(zset-rows zs)` | Rows with nonzero weight, canonical order |
| `zset-size` | `(zset-size zs)` | Number of distinct rows with nonzero weight (support size) |

## Z-set relational operators

| Procedure | Signature | Notes |
|-----------|-----------|-------|
| `zset-map` | `(zset-map f zs)` | Weight-preserving row map, consolidating collisions. Linear |
| `zset-filter` | `(zset-filter pred zs)` | Weight-preserving selection. Linear |
| `zset-project` | `(zset-project f zs)` | Map plus consolidation (same as `zset-map`). Linear |
| `zset-union` | `(zset-union zs ...)` | Variadic Z-set addition; the empty union is the identity. Linear |
| `zset-join` | `(zset-join a b a-key b-key emit)` | Indexed equijoin: for each pair of rows whose keys are `equal?`, emit `(emit a-row b-row)` with the product of the two weights. Bilinear |
| `zset-distinct` | `(zset-distinct zs)` | Batch set semantics: weight > 0 becomes 1, weight <= 0 dropped. The incremental form is `incremental-distinct` |
| `zset-count` | `(zset-count zs)` | Sum of weights (a group homomorphism into Z) |
| `zset-sum` | `(zset-sum zs)` | Sum of `value * weight` over numeric rows |

## Streams and the DBSP stream operators

| Procedure | Signature | Notes |
|-----------|-----------|-------|
| `stream-zero` | `(stream-zero n)` | n copies of the empty Z-set |
| `stream-delay` | `(stream-delay s)` | The z^-1 operator: one-tick delay initialized to the group zero; length preserved |
| `dbsp-delay` | `(dbsp-delay s)` | Alias for `stream-delay` under its DBSP name |
| `stream-D` | `(stream-D s)` | `D(s) = s - z^-1(s)`, pointwise |
| `stream-I` | `(stream-I s)` | Running prefix sum |
| `stream-add` | `(stream-add sa sb)` | Pointwise addition (equal lengths) |
| `stream-negate` | `(stream-negate s)` | Pointwise negation |
| `stream-lift` | `(stream-lift q s)` | Lift a batch query to a stream operator applied at every tick: `(lift Q s)[t] = Q(s[t])` |
| `dbsp-compose` | `(dbsp-compose f g)` | `f . g` — realizes the DBSP chain rule `(Q1 . Q2)^D = Q1^D . Q2^D` |

## Incremental operators (delta stream -> delta stream)

| Procedure | Signature | Notes |
|-----------|-----------|-------|
| `dbsp-incrementalize` | `(dbsp-incrementalize q)` | The generic incrementalizer `Q^D = D . lift(Q) . I` for **any** batch query — the executable specification every specialized operator below is proven equal to (batch equivalence, ADR 0009 §9) |
| `incremental-map` | `(incremental-map f)` | Returns a delta-stream transformer (linear: applies to each delta directly) |
| `incremental-filter` | `(incremental-filter pred)` | Same |
| `incremental-project` | `(incremental-project f)` | Same |
| `incremental-union` | `(incremental-union dsa dsb)` | Pointwise Z-set addition of two delta streams |
| `incremental-join` | `(incremental-join dsa dsb a-key b-key emit)` | Bilinear join via the discrete product rule (below) |
| `incremental-distinct` | `(incremental-distinct ds)` | Multiplicity-correct set-semantics view (below) |

### The join product rule

At each tick t, with prior snapshots A0, B0 and current changes dA, dB:

```
d(A join B) = (dA join B0) + (A0 join dB) + (dA join dB)
```

The cross term `(dA join dB)` is essential: it makes a same-tick insert/delete
interleaving exact and prevents spurious rows. Integrating the output delta
stream equals the batch join of the integrated inputs.

### Multiplicity-correct distinct

`incremental-distinct` maintains one exact count per row and emits only
boundary changes: +1 on a 0-to-positive transition and -1 on a positive-to-0
transition. It never rescans the accumulated relation, and same-tick
cancellation produces no boundary event.

## Verified example

```scheme
(require core.dbsp)

;; Batch query: keep even rows.
(define (evens zs) (zset-filter even? zs))

;; Delta stream: tick 0 inserts 1,2,3; tick 1 inserts 4 and retracts 2.
(define deltas
  (list (zset-from-weighted '((1 1) (1 2) (1 3)))
        (zset-from-weighted '((1 4) (-1 2)))))

;; Generic incrementalizer: Q^D = D . lift(Q) . I
(define evens-inc (dbsp-incrementalize evens))
(for-each (lambda (d) (display (zset-entries d)) (newline))
          (evens-inc deltas))
```
```
((1 2))
((-1 2) (1 4))
```

Tick 0 emits the insertion of the even row 2; tick 1 emits its retraction and
the insertion of 4 — exactly the change in the query's output, never a
recomputation of the whole relation.
