# Tutorial 20: Bitwise Operations and System Interface

---

## Bitwise Operations

```scheme
(display (bitwise-and 12 10))     ;; => 8   (1100 & 1010 = 1000)
(display (bitwise-or 12 10))      ;; => 14  (1100 | 1010 = 1110)
(display (bitwise-xor 12 10))     ;; => 6   (1100 ^ 1010 = 0110)
(display (bitwise-not 0))         ;; => -1  (all bits flipped)
(display (bit-shift-left 1 8))    ;; => 256
(display (bit-shift-right 256 4)) ;; => 16
(display (popcount 255))          ;; => 8   (8 bits set)
```

---

## Environment Variables

```scheme
(display (getenv "HOME"))       ;; => /Users/alice
(display (getenv "PATH"))       ;; => /usr/local/bin:...
```

---

## Time

```scheme
(display (current-time))        ;; Unix timestamp (seconds)
```

---

## File System

```scheme
;; Check if file/directory exists
(display (file-exists? "hello.esk"))  ;; => #t or #f

;; Remove file
(remove "tempfile.txt")

;; Rename
(rename "old.txt" "new.txt")
```

---

## Command Line Arguments

Programs compiled with `eshkol-run` receive `argc` and `argv` through
the standard main wrapper:

```scheme
;; Access is via the runtime — argc/argv are passed to main
;; and available in the entry function context
```

---

## Random Numbers

```scheme
;; Pseudo-random double in [0, 1)
(display (random))

;; Quantum-seeded RNG (uses hardware entropy when available)
(display (quantum-random))
(display (quantum-random-range 1 100))
```

---

## Process Exit

```scheme
(exit 0)    ;; exit with code 0
(exit 1)    ;; exit with code 1
```
