# Tutorial 9: The Module System

Eshkol uses `require` and `provide` for module loading with dependency
resolution and cycle detection.

---

## Part 1: Using Modules

```scheme
;; Load the standard library
(require stdlib)

;; Load specific sub-modules
(require core.list.higher_order)   ;; map, fold, filter, etc.
(require core.list.transform)      ;; take, drop, append, reverse
(require core.list.sort)           ;; sort
(require core.strings)             ;; string operations
(require core.json)                ;; JSON parsing
(require core.data.csv)            ;; CSV parsing
(require core.data.base64)         ;; Base64 encoding/decoding
(require core.functional.compose)  ;; compose, pipe
(require core.functional.curry)    ;; curry, flip
```

---

## Part 2: Creating Modules

A module is an `.esk` file with a `provide` declaration listing its exports:

```scheme
;;; mylib.esk — a custom library

(provide square cube hypotenuse)

(define (square x) (* x x))
(define (cube x) (* x x x))

;; Internal helper — NOT exported
(define (sum-of-squares a b) (+ (square a) (square b)))

(define (hypotenuse a b) (sqrt (sum-of-squares a b)))
```

Use it:

```scheme
(require mylib)

(display (square 5))      ;; => 25
(display (hypotenuse 3 4))  ;; => 5.0
;; (sum-of-squares 3 4) would error — it's not exported
```

---

## Part 3: Module Paths

Module names map to file paths with dots as separators:

| Module name | File path |
|---|---|
| `stdlib` | `lib/core/**/*.esk` (all sub-modules) |
| `core.list.higher_order` | `lib/core/list/higher_order.esk` |
| `core.json` | `lib/core/json.esk` |
| `mylib` | `./mylib.esk` (relative to current file) |

---

## Part 4: Available Standard Library Modules

### Lists
- `core.list.higher_order` — map, fold, fold-left, fold-right, filter, for-each, any, every
- `core.list.transform` — take, drop, append, reverse, partition, list-copy
- `core.list.sort` — sort (merge sort)
- `core.list.search` — member, assoc, list-ref, list-tail
- `core.list.query` — count-if, find, length
- `core.list.generate` — iota, range, make-list, repeat, zip
- `core.list.convert` — list->vector, vector->list
- `core.list.compound` — caar, cadr, cdar, cddr, etc.

### Functional
- `core.functional.compose` — compose, pipe
- `core.functional.curry` — curry, flip

### Data
- `core.json` — JSON parsing and generation
- `core.data.csv` — CSV parsing and writing
- `core.data.base64` — Base64 encode/decode
- `core.strings` — string operations

### I/O
- `core.io` — display, write, read, newline, ports

### Operators
- `core.operators.arithmetic` — +, -, *, /
- `core.operators.compare` — <, >, <=, >=, =
- `core.logic.predicates` — null?, pair?, number?, etc.
- `core.logic.boolean` — and, or, not
- `core.logic.types` — type predicates

---

*Next: Tutorial 10 — Macros and Metaprogramming*
