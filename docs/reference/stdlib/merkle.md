# `core.merkle` — FNV-1a hashing, Merkle trees, content-addressable storage

**Source**: [`lib/core/merkle.esk`](../../../lib/core/merkle.esk)
**Require**: `(require core.merkle)` — must be required individually (not auto-loaded by `(require stdlib)`).

Pure-Eshkol Merkle trees and a content-addressable store (CAS). The default hash is
**FNV-1a 64-bit** (the inner loop is in `lib/core/merkle.c` for correct `uint64_t`
overflow semantics) — fast and well-distributed for in-process content addressing
but **not cryptographically secure**. For tamper resistance, pass a crypto hash
(e.g. `agent.crypto`'s `sha256`) via the optional `hash-fn` argument to
`merkle-leaf`, `merkle-tree-with-hash`, `merkle-verify`, or `make-cas-with-hash`.

Tree representation:
- leaf  → `(vector 'merkle-leaf hash data)`
- inode → `(vector 'merkle-inode hash left right)`

Odd levels duplicate the last node (Bitcoin convention) so every inode has two
children.

## Hash

### `(fnv1a-64 str)`
FNV-1a 64-bit hash of the UTF-8 bytes of `str`. Returns the uint64 bit pattern as an
int64 — it **may be negative**; only equality and hex rendering matter for content
addressing. Deterministic for a given input.

```scheme
;; merkle.esk
(require core.merkle)
(display (hash->hex (fnv1a-64 "hello"))) (newline)
(display (hash->hex (fnv1a-64 "a")))     (newline)   ; spec vector 0xaf63dc4c8601ec8c
```
```
a430d84680aabd0b
af63dc4c8601ec8c
```

Edge case: `(hash->hex (fnv1a-64 ""))` = `"cbf29ce484222325"` (the FNV offset basis).

### `(hash->hex h)`
Render a 64-bit hash `h` as a 16-char lowercase hex string.

```scheme
(require core.merkle)
(display (string-length (hash->hex (fnv1a-64 "x")))) (newline)
```
```
16
```

## Merkle tree

### `(merkle-leaf data [hash-fn])`
Wrap a string `data` as a leaf node. Optional second arg is the hash function
(defaults to `fnv1a-64`). `data` must be a string.

### `(merkle-leaf? n)` / `(merkle-inode? n)`
Type predicates. `merkle-leaf?` is true for a leaf vector, `merkle-inode?` for an
internal node.

```scheme
;; merkle.esk
(require core.merkle)
(define lf (merkle-leaf "data"))
(display (merkle-leaf? lf))  (newline)
(display (merkle-inode? lf)) (newline)
(display (merkle-data lf))   (newline)
```
```
#t
#f
data
```

### `(merkle-data n)`
Original data string of a leaf node. **Errors** (`merkle-data: not a leaf node`) if
`n` is an inode.

### `(merkle-root n)`
Hash at the top of the tree, or the leaf's own hash if `n` is a leaf. Errors if `n`
is not a merkle node.

### `(merkle-tree data-list)` / `(merkle-tree-with-hash data-list hash-fn)`
Build a Merkle tree from a list of strings. `merkle-tree` uses FNV-1a;
`merkle-tree-with-hash` takes an explicit hash function. Returns the root node.
An **empty list errors** (`cannot build tree from empty list`).

```scheme
;; merkle.esk
(require core.merkle)
(define t (merkle-tree (list "alice" "bob" "carol" "dave")))
(display (hash->hex (merkle-root t))) (newline)
(display (merkle-leaves t))           (newline)
```
```
115bb5e7e82ae101
(alice bob carol dave)
```

Determinism: same data → same root; different **order** → different root.

### `(merkle-leaves tree)`
Walk the tree left-to-right, returning the list of leaf data strings. An
odd-one-out duplicated leaf is counted once.

### `(merkle-proof tree index)`
Inclusion proof for the leaf at `index`: a list of `(sibling-hash . side)` pairs
from leaf up to root (`side` is `'left` or `'right`). Do not reverse it — this is
the order `merkle-verify` consumes.

### `(merkle-verify root-hash leaf-data proof [hash-fn])`
Verify that `leaf-data` is included under `root-hash` given `proof`. Returns `#t`
on a valid proof, `#f` otherwise. `hash-fn` defaults to `fnv1a-64` and must match
the one the tree was built with.

```scheme
;; merkle.esk
(require core.merkle)
(define t  (merkle-tree (list "alice" "bob" "carol" "dave")))
(define p0 (merkle-proof t 0))
(display p0)                                        (newline)
(display (merkle-verify (merkle-root t) "alice" p0)) (newline)
(display (merkle-verify (merkle-root t) "MALLORY" p0)) (newline)
```
```
((21748447695211092 . right) (5625647597590589385 . right))
#t
#f
```

Edge case: an empty proof `'()` verifies only if `(hash-fn leaf-data)` equals
`root-hash` directly (i.e. a single-leaf tree).

## Content-addressable store (CAS)

A CAS is `(vector 'eshkol-cas hash-fn hash-table)`. Keys are 16-char hex content
hashes; storing identical data yields the same key (dedup).

### `(make-cas)` / `(make-cas-with-hash hash-fn)`
Create a CAS using the default FNV-1a hash, or an explicit hash function.

### `(cas? c)`
Predicate: is `c` a CAS handle?

### `(cas-put! c data)`
Store `data`, returning its content hash as a 16-char hex string. Mutates the store.

### `(cas-get c hash)`
Retrieve the data stored under `hash`, or `#f` if absent.

### `(cas-has? c hash)`
Predicate: does `c` contain an entry for `hash`?

### `(cas-size c)`
Number of distinct objects stored.

### `(cas-keys c)`
List of all stored content hashes.

```scheme
;; merkle.esk
(require core.merkle)
(define c (make-cas))
(display (cas? c))       (newline)
(display (cas-size c))   (newline)
(define h (cas-put! c "payload"))
(display h)              (newline)
(display (cas-get c h))  (newline)
(display (cas-has? c h)) (newline)
(display (cas-keys c))   (newline)
(display (cas-get c "0000000000000000")) (newline)   ; absent
```
```
#t
0
cfb8a9d063b5e9e5
payload
#t
(cfb8a9d063b5e9e5)
#f
```

Edge case: `cas-get` of an absent hash returns `#f`; a duplicate `cas-put!` of the
same data returns the same hash and does not grow the store.
