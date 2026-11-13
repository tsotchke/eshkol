# Structural types: records, higher-rank polymorphism, session types

Eshkol's type checker has a structural layer (ADR-0046) beside the gradual
checker of [GRADUAL_TYPING.md](../../guide/GRADUAL_TYPING.md). It types three
families of programs that need type variables and structure rather than named
types:

- **extensible records** with row polymorphism (`core.records`);
- **higher-rank polymorphism**: functions whose parameters are themselves
  polymorphic;
- **session types**: communication protocols on channel endpoints
  (`core.sessions`), checked at compile time.

Errors in this layer are compile-time errors on every engine: the native
compiler (JIT and AOT) refuses to generate code, and the bytecode VM refuses to
run the program. `--unsafe` is the only way to proceed past them. Set
`ESHKOL_DUMP_STRUCTURAL_TYPES=1` to print the type the checker gives every
structurally typed top-level definition.

## Which code is checked

A top-level form is checked structurally when

- one of its annotations uses a structural type former: `record`, `chan`, a
  protocol (`!`, `?`, `choose`, `offer`, `rec`, `dual`, `end`), `-o`,
  `forall`, `values`, or a user type constructor such as `(st-ref s a)`;
- it mentions an operation of a required structural library (`core.records`,
  `core.sessions`); or
- it mentions, or is mentioned by, a definition that is checked structurally.

Everything else is left to the gradual checker. A value coming from unchecked
code has the type `any`, which is consistent with every type. Branches whose
types have no common type join to `any`, as in the gradual checker; a
mismatch against a known type (an argument, an annotation, a field) is an
error.

## Type syntax

| Type | Meaning |
|---|---|
| `number`, `string`, `boolean`, `char`, `symbol`, `null`, `any` | base types (`integer` and `real` are `number` here) |
| `(-> A ... R)` | function |
| `(-o A ... R)` | one-shot function: called (or handed on) exactly once |
| `(forall (a b) T)` | polymorphic type; may appear in argument position |
| `(record (l T) ...)` | closed record with exactly these fields |
| `(record (l T) ... r)` | open record: these fields and the row `r` |
| `(chan P)` | a channel endpoint following protocol `P` |
| `end` | protocol: finished; the endpoint must be closed |
| `(! T P)` / `(? T P)` | protocol: send / receive a `T`, then continue as `P` |
| `(choose (l P) ...)` | protocol: this side picks one label |
| `(offer (l P) ...)` | protocol: the other side picks; this side handles every label |
| `(rec x P)` | recursive protocol; `x` inside `P` stands for the whole |
| `(dual P)` | the other side's view of `P` |
| `(values A B ...)` | multiple values |
| `(name T ...)` | an abstract type constructor, e.g. `(st-ref s a)` |

Type variables that are not bound by `forall` and are free in a top-level
signature are quantified over that signature. `define-type` names any of
these, including protocols: `(define-type Echo (rec loop (offer ...)))`.

## Records and row polymorphism

```scheme
(require core.records)

(define (get-x p) (record-ref p 'x))
(define (norm1 p) (+ (abs (record-ref p 'x)) (abs (record-ref p 'y))))

(define p (record 'x 3 'y -4))
(display (norm1 p))                               ; 7
(display (norm1 (record 'x 1 'y 2 'name "pt")))   ; 3
```

With `ESHKOL_DUMP_STRUCTURAL_TYPES=1` the checker reports the principal types:

```text
[types] get-x : (forall (a r) (-> (record (x a) r) a))
[types] norm1 : (forall (r) (-> (record (x number) (y number) r) number))
```

`r` is a row variable: whatever other fields the record has. The operations,
whose label argument must be a quoted symbol literal:

| Operation | Type rule |
|---|---|
| `(record 'l1 v1 'l2 v2 ...)` | a closed record with those fields (labels distinct) |
| `(record-ref r 'l)` | `r : (record (l a) r')` gives `a` |
| `(record-extend r 'l v)` | `r : (record r')` where `r'` has no `l`, gives `(record (l T) r')` |
| `(record-remove r 'l)` | `r : (record (l a) r')` gives `(record r')` |
| `(record-update r 'l v)` | `r : (record (l a) r')` gives `(record (l T) r')` |
| `record?`, `record-has?`, `record-labels`, `record->alist` | ordinary procedures |

Rows never contain a label twice, so `(record-extend (record 'x 1) 'x 2)` is a
compile-time error. Selecting an absent field is a compile-time error.

**Structural subtyping.** Where a *closed* record type is expected — a
parameter annotated `(record (x number))`, say — a record with more fields is
accepted (width subtyping); fields are compared covariantly. Inference never
adds a subtyping step on its own; row variables are what let an unannotated
function accept wider records.

```scheme
(define (only-x (p : (record (x number)))) : number (record-ref p 'x))
(only-x (record 'x 1 'y 2))        ; accepted: extra field y
(define (keep-rest (p : (record (x number) r))) : (record r)
  (record-remove p 'x))             ; r is rigid: everything but x survives
```

Records built with `record` are structural. Types made by
`define-record-type` stay nominal and are not records in this sense. Records
may not hold endpoints or one-shot functions.

## Higher-rank polymorphism

A parameter can be annotated with a polymorphic type and then used at several
types inside the body. Checking is bidirectional and predicative (Peyton
Jones, Vytiniotis, Weirich and Shields, 2007): polymorphic types come from
annotations, and a type variable is never instantiated with a polymorphic
type.

```scheme
(define (apply-both (f : (forall (a) (-> a a))) (n : number) (s : string)) : number
  (+ (f n) (string-length (f s))))

(apply-both (lambda (x) x) 4 "abc")          ; 7
(apply-both (lambda (x) (+ x 1)) 1 "s")      ; compile-time error: not polymorphic
```

Church numerals are rank-2 combinators:

```scheme
(define (twice f x) (f (f x)))     ; (forall (a) (-> (-> a a) a a))
(define (to-int (n : (forall (a) (-> (-> a a) a a)))) : number
  (n (lambda (k) (+ k 1)) 0))
(to-int twice)                      ; 2
```

**Encapsulation (runST).** A rank-2 type confines a resource to the dynamic
extent of a call:

```scheme
(define (st-new (tok : (st s)) (v : a)) : (st-ref s a) (vector v))
(define (st-get (tok : (st s)) (r : (st-ref s a))) : a (vector-ref r 0))
(define (run-st (action : (forall (s) (-> (st s) a)))) : a
  (action (vector 'st-token)))

(run-st (lambda (tok) (st-get tok (st-new tok 55))))   ; 55
(run-st (lambda (tok) (st-new tok 0)))                 ; compile-time error:
;; the type variable `s` would escape its scope
```

Local `let` bindings are not generalised; a lambda-bound function without an
annotation is monomorphic inside its body.

## Session types

```scheme
(require core.sessions)

(define-type Adder (offer (add (? number (? number (! number end))))
                          (quit end)))

(define (adder-server (c : (chan Adder))) : null
  (session-offer c
    'add (lambda (c)
           (let-values (((x c) (session-recv c)))
             (let-values (((y c) (session-recv c)))
               (session-close (session-send c (+ x y))))))
    'quit (lambda (c) (session-close c))))

(define (add-remote (c : (chan (dual Adder))) (a : number) (b : number)) : number
  (let-values (((sum c) (session-recv (session-send (session-send (session-select c 'add) a) b))))
    (session-close c)
    sum))

(display (add-remote (session-fork adder-server) 20 22))   ; 42
```

Every operation consumes the endpoint it is given and returns the endpoint for
the rest of the protocol:

| Operation | Type |
|---|---|
| `(session-send c v)` | `(chan (! T P))`, `T` → `(chan P)` |
| `(session-recv c)` | `(chan (? T P))` → `(values T (chan P))` |
| `(session-select c 'l)` | `(chan (choose (l P) ...))` → `(chan P)` |
| `(session-offer c 'l1 h1 ...)` | `(chan (offer (l1 P1) ...))`, each `hi : (-o (chan Pi) A)` → `A` |
| `(session-close c)` | `(chan end)` → `null` |
| `(make-session-pair)` | → `(values (chan P) (chan (dual P)))` |
| `(session-fork proc)` | `proc : (-o (chan P) any)` runs on a worker thread → `(chan (dual P))` |
| `(tcp-session handle)` | an endpoint over a connected `core.net` TCP handle |

The checker proves:

- **the protocol is followed**: each operation is the one the protocol permits
  next; an offer handles exactly the offered labels; a select names an offered
  label;
- **endpoints are linear**: each endpoint is used exactly once on every path —
  not twice, not dropped on some branch, not discarded as the value of a
  non-final expression, not passed to code whose type is unknown, not stored
  in a record, and captured only by one-shot (`-o`) closures;
- **the two ends agree**: the ends made by `make-session-pair` or
  `session-fork` have dual protocols.

Recursive protocols are written with `rec` and driven by recursion or a named
`let` whose loop variable carries the endpoint:

```scheme
(define-type Summer (rec loop (offer (more (? number loop))
                                     (done (! number end)))))
(define (summer (c : (chan Summer))) : null
  (let loop ((c c) (acc 0))
    (session-offer c
      'more (lambda (c) (let-values (((x c) (session-recv c))) (loop c (+ acc x))))
      'done (lambda (c) (session-close (session-send c acc))))))
```

**Over sockets.** `tcp-session` wraps a connected TCP handle; messages travel
one per line in `write` syntax, so the same protocol types and operations work
across processes. The remote peer is not trusted: `session-offer` checks every
received label and reports a protocol violation for an unexpected one. The
types prove that the local program follows its protocol, not what the remote
program sends.

Sending an endpoint over an endpoint (delegation) and runtime validation of
received payload types at a TCP boundary are planned for v1.9.1. Type
variables range over unrestricted types, so a polymorphic function cannot be
instantiated at an endpoint type; quantification over linear types is planned
for v2.0.

## Engines

| | JIT / AOT | Bytecode VM |
|---|---|---|
| Records, higher-rank programs | run | run |
| Session programs | run (threads, `core.net`) | refused programs are refused identically; running them needs the native engine |
| Every compile-time error above | refuses code generation | refuses to run |
