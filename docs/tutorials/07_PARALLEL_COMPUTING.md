# Tutorial 7: Parallel Computing

Eshkol provides three parallel primitives backed by a work-stealing thread
pool. All three compose with the rest of the language — closures, autodiff,
exact arithmetic.

---

## Part 1: parallel-map

Apply a function to each element of a list in parallel:

```scheme
;; Sequential map (single-threaded)
(define result (map (lambda (x) (* x x)) '(1 2 3 4 5 6 7 8)))
;; => (1 4 9 16 25 36 49 64)

;; Parallel map (work-stealing thread pool)
(define result (parallel-map (lambda (x) (* x x)) '(1 2 3 4 5 6 7 8)))
;; => (1 4 9 16 25 36 49 64) — same result, uses all cores
```

`parallel-map` splits the list across available cores, maps the function
in parallel, and reassembles results in order. The function must be pure
(no side effects) for correctness.

### When to Use

Parallel map pays off when the per-element computation is expensive:

```scheme
;; Expensive computation — benefits from parallelism
(define (heavy-compute x)
  (let loop ((i 0) (acc 0.0))
    (if (= i 1000000)
        (+ acc x)
        (loop (+ i 1) (+ acc (sin (* i 0.001)))))))

(define results (parallel-map heavy-compute '(1 2 3 4 5 6 7 8)))
```

For cheap operations (addition, multiplication), the overhead of thread
scheduling exceeds the benefit. Use regular `map` for those.

---

## Part 2: parallel-fold

Parallel reduction with an associative binary operator:

```scheme
;; Sum a large list in parallel
(define big-list (iota 1000000))  ;; (0 1 2 ... 999999)
(define total (parallel-fold + 0 big-list))
;; => 499999500000

;; Product (careful with overflow — will promote to bignum)
(define product (parallel-fold * 1 '(1 2 3 4 5 6 7 8 9 10)))
;; => 3628800
```

The operator MUST be associative: `(op (op a b) c) = (op a (op b c))`.
Addition, multiplication, min, max are all associative. Subtraction is NOT
— `parallel-fold` with `-` will give wrong results.

---

## Part 3: Futures

Futures represent asynchronous computations:

```scheme
;; Launch a computation in the background
(define f (future (lambda () (fib 40))))

;; Do other work while f computes...
(display "Computing in background...")
(newline)

;; Block until the result is ready
(define result (force f))
(display result)
(newline)
```

Futures are useful when you have independent computations that can
overlap:

```scheme
;; Two expensive computations in parallel
(define f1 (future (lambda () (heavy-compute 1))))
(define f2 (future (lambda () (heavy-compute 2))))

;; Both are running concurrently
;; Collect results (blocks until each is ready)
(define r1 (force f1))
(define r2 (force f2))
(display (+ r1 r2))
```

---

## Thread Pool Architecture

All three primitives use the same work-stealing thread pool:

- **Thread count** = number of CPU cores (auto-detected)
- **Work stealing** — idle threads steal tasks from busy threads' queues
- **Arena memory** — each thread has its own arena region, merged on join
- **No GC pressure** — parallel computation is deterministic, zero GC

---

*Next: Tutorial 8 — Control Flow (call/cc, continuations, exceptions)*
