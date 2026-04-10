# Tutorial 16: Hash Tables

Eshkol provides mutable hash tables with FNV-1a hashing and open
addressing — O(1) average lookup and insertion.

---

## Creating and Using Hash Tables

```scheme
;; Create an empty hash table
(define ht (make-hash-table))

;; Insert key-value pairs
(hash-table-set! ht "name" "Alice")
(hash-table-set! ht "age" 30)
(hash-table-set! ht "city" "NYC")

;; Lookup
(display (hash-table-ref ht "name"))    ;; => Alice
(display (hash-table-ref ht "age"))     ;; => 30

;; Check existence
(display (hash-table-exists? ht "name"))  ;; => #t
(display (hash-table-exists? ht "foo"))   ;; => #f

;; Delete
(hash-table-delete! ht "city")
(display (hash-table-exists? ht "city"))  ;; => #f

;; Size
(display (hash-table-size ht))  ;; => 2
```

---

## Iterating

```scheme
;; Get all keys
(display (hash-table-keys ht))
;; => ("name" "age")

;; Get all values
(display (hash-table-values ht))
;; => ("Alice" 30)
```

---

## Use Cases

Hash tables are ideal for:
- Frequency counting: `(hash-table-set! ht word (+ 1 (or (hash-table-ref ht word) 0)))`
- Memoisation: cache expensive computation results by input
- Symbol tables: fast lookup by name
- Graphs: adjacency lists keyed by node ID
