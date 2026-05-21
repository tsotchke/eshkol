# Tutorial 4: The Consciousness Engine

Eshkol's consciousness engine integrates three computational frameworks —
logic programming, active inference, and global workspace theory — into a
small set of compiler builtins. This tutorial shows you how to use each
one. Every form below matches the actual argument shape the compiler
expects (see `lib/backend/logic_workspace_codegen.cpp` and
`lib/core/{logic,inference,workspace}.cpp`).

---

## Part 1: Logic Programming

### Knowledge Bases and Facts

```scheme
;; Create a knowledge base
(define kb (make-kb))

;; Assert facts — make-fact takes a predicate and individual arguments
(kb-assert! kb (make-fact 'parent 'alice 'bob))
(kb-assert! kb (make-fact 'parent 'bob 'charlie))
(kb-assert! kb (make-fact 'parent 'alice 'diana))

;; Query: who are alice's children?
;; Logic variables use ?name syntax (no quote — the ? prefix is
;; recognised by the parser as a logic variable, not a symbol)
(display (kb-query kb (make-fact 'parent 'alice ?child)))
(newline)
;; => ((parent alice bob) (parent alice diana))

;; Query: who is charlie's parent?
(display (kb-query kb (make-fact 'parent ?p 'charlie)))
(newline)
;; => ((parent bob charlie))
```

The `?variable` syntax creates logic variables. Queries find all facts in
the KB that unify with the pattern.

### Unification

Unification is the core operation — it finds a substitution that makes two
terms equal:

```scheme
;; Unify a logic variable with a value
(define s1 (unify ?x 42 (make-substitution)))
(display (walk ?x s1))
(newline)
;; => 42

;; Chain variables: ?x -> ?y -> 99
(define s2 (unify ?x ?y (make-substitution)))
(define s3 (unify ?y 99 s2))
(display (walk ?x s3))
(newline)
;; => 99

;; Structural unification of facts
(define f1 (make-fact 'parent ?a 'bob))
(define f2 (make-fact 'parent 'alice ?b))
(define s4 (unify f1 f2 (make-substitution)))
(display (walk ?a s4))  ;; => alice
(display (walk ?b s4))  ;; => bob
```

### Type Predicates

```scheme
(logic-var? ?x)           ;; => #t  (bare ?x, no quote)
(logic-var? 'hello)       ;; => #f
(substitution? s1)        ;; => #t
(kb? kb)                  ;; => #t
(fact? (make-fact 'a 'b)) ;; => #t
```

---

## Part 2: Active Inference

Active inference models the world as a factor graph and uses belief
propagation to infer hidden states from observations.

### Factor Graphs

A factor graph has a fixed number of variables; each variable has a fixed
number of discrete states; factors connect groups of variables and carry
a conditional probability table (CPT). The signature is
`(make-factor-graph num-vars dims-tensor)` — two arguments — where
`dims-tensor` gives the number of states for each variable.

```scheme
;; Three variables, each with two discrete states.
(define fg (make-factor-graph 3 #(2 2 2)))

;; Add a binary factor connecting variables 0 and 1.
;; - var-indices is a tensor of variable ids the factor depends on.
;; - cpt is a tensor of conditional probabilities, flattened in row-major
;;   order over the (states of var 0) x (states of var 1) grid.
;;   CPTs are stored in log space internally; you supply natural-scale
;;   probabilities and the runtime takes the log.
(fg-add-factor! fg #(0 1) #(0.9 0.1 0.2 0.8))

;; Binary factor connecting variables 1 and 2.
(fg-add-factor! fg #(1 2) #(0.7 0.3 0.4 0.6))

;; Run loopy belief propagation for 10 iterations.
(fg-infer! fg 10)
```

After `fg-infer!`, each variable's belief vector is converged (modulo the
iteration bound). The runtime stores beliefs internally; you read them out
indirectly through `free-energy` and `expected-free-energy`, or directly
through `fg-marginal` (a system builtin, not in the core 22).

### Free Energy

Free energy quantifies surprise — how much the model's predictions diverge
from the observed evidence:

```scheme
;; Observations are encoded as #(var-index observed-state) pairs.
;; Passing a single 2-element tensor #(0 0) is read as: variable 0 was
;; observed in state 0.
(define fe (free-energy fg #(0 0)))
(display fe)
(newline)

;; For multiple observations, pass a tensor of pair tensors.
;;   #(#(0 0) #(2 1))  =>  var 0 in state 0  AND  var 2 in state 1.

;; Expected free energy for action selection takes the variable that the
;; agent intends to act on, plus the state it would push that variable to:
;;   (expected-free-energy fg action-var action-state)
(define efe (expected-free-energy fg 0 0))
(display efe)
(newline)
```

The pair format keeps observation specifications terse: a state vector
over every variable would force you to encode "unobserved" with a
sentinel, whereas a list of pairs only mentions the variables that were
actually clamped.

### Learning with CPT Updates

`fg-update-cpt!` rewrites the CPT of an existing factor (referenced by
its insertion index — the same order in which `fg-add-factor!` was
called). After updating a factor, the message cache is reset so the next
`fg-infer!` reconverges from the new CPT.

```scheme
;; Update factor 0's CPT in light of new evidence.
(fg-update-cpt! fg 0 #(0.95 0.05 0.15 0.85))

;; Re-run inference — beliefs reconverge with the updated factor.
(fg-infer! fg 10)
```

`fg-observe!` is the related primitive for clamping a variable to a
specific observed state (closer in spirit to "evidence" than "learning");
it also resets the message cache so subsequent `fg-infer!` calls see the
clamp.

---

## Part 3: Global Workspace Theory

The workspace implements a cognitive architecture where specialised
modules compete for global broadcast via softmax competition. The
signature is `(make-workspace dim max-modules)` — `dim` is the
dimensionality of each module's content vector; `max-modules` is the
maximum number of modules the workspace can hold (the runtime hard-caps
at 16, regardless of what you pass).

### Creating a Workspace

```scheme
;; A workspace with 3-element content vectors, capacity for up to 4 modules.
(define ws (make-workspace 3 4))

;; Register processing modules. Each is a triple of (workspace, name, fn).
;; The closure takes the current workspace content and must return a
;; `(cons salience proposal-tensor)` pair: the salience is a scalar
;; (double) used in softmax competition, and the proposal-tensor is a
;; tensor of length `dim` that becomes the broadcast content if this
;; module wins. The salience is read from `car`; the tensor from `cdr`.
(ws-register! ws "visual"
  (lambda (content) (cons 0.8 #(0.8 0.5 0.3))))
(ws-register! ws "auditory"
  (lambda (content) (cons 0.6 #(0.2 0.9 0.1))))
(ws-register! ws "memory"
  (lambda (content) (cons 0.3 #(0.5 0.5 0.5))))

;; Step the workspace — modules compete via softmax, the winner broadcasts.
;; ws-step! takes exactly one argument (the workspace) and reads the
;; current content from the workspace itself; you do not pass a stimulus
;; vector — instead, drive the content via subsequent ws-register! /
;; ws-step! cycles, or by calling the runtime helper
;; eshkol_ws_make_content_tensor before the cycle.
(ws-step! ws)
```

### Type Predicates

```scheme
(factor-graph? fg)   ;; => #t
(workspace? ws)      ;; => #t
```

---

## Part 4: Combining the Systems

The real power is combining logic + inference + workspace:

```scheme
;; 1. Logic layer: store structured symptom knowledge.
(define kb (make-kb))
(kb-assert! kb (make-fact 'symptom 'fever))
(kb-assert! kb (make-fact 'symptom 'cough))

;; 2. Inference layer: model causal relationships.
;;    Three binary variables: symptom -> illness -> consequence.
(define fg (make-factor-graph 3 #(2 2 2)))
(fg-add-factor! fg #(0 1) #(0.8 0.2 0.1 0.9))  ;; fever -> flu
(fg-add-factor! fg #(1 2) #(0.7 0.3 0.3 0.7))  ;; flu -> fatigue
(fg-infer! fg 10)

;; 3. Workspace layer: compete for attention.
(define ws (make-workspace 1 4))
(ws-register! ws "logic"
  (lambda (content)
    ;; Closure returns (cons salience proposal-tensor).
    ;; Salience is high when the KB has at least one symptom asserted.
    (if (null? (kb-query kb (make-fact 'symptom ?s)))
        (cons 0.1 #(0.1))
        (cons 0.9 #(0.9)))))
(ws-register! ws "inference"
  (lambda (content)
    ;; Salience could be derived from the model's free energy on a
    ;; per-cycle observation; placeholder constant for now.
    (cons 0.5 #(0.5))))
(ws-step! ws)
```

---

## The Consciousness Engine Builtins

The user-visible 22-builtin surface (logic + inference + workspace, plus
predicates). Two further primitives — `fg-observe!` and
`kb-query-prefix` — are emitted by the parser as dedicated ops too,
and `fg-marginal`, `fg-entropy`, `kb-retract!` reach the runtime through
`SystemCodegen`; refer to
`docs/breakdown/CONSCIOUSNESS_ENGINE.md` for the full surface.

| Builtin | Arity | Description |
|---|---|---|
| `make-substitution` | 0 | Create an empty substitution |
| `unify` | 3 | Unify two terms under a substitution |
| `walk` | 2 | Resolve a variable through a substitution |
| `make-fact` | 1 + | Create a fact from a predicate and arguments |
| `make-kb` | 0 | Create an empty knowledge base |
| `kb-assert!` | 2 | Add a fact to a KB |
| `kb-query` | 2 | Query a KB with a fact pattern; returns matching facts |
| `logic-var?` | 1 | Test if value is a logic variable |
| `substitution?` | 1 | Test if value is a substitution |
| `kb?` | 1 | Test if value is a knowledge base |
| `fact?` | 1 | Test if value is a fact |
| `make-factor-graph` | 2 | `(make-factor-graph num-vars dims-tensor)` |
| `fg-add-factor!` | 3 | `(fg-add-factor! fg var-indices-tensor cpt-tensor)` |
| `fg-infer!` | 2 | `(fg-infer! fg max-iters)` |
| `fg-update-cpt!` | 3 | `(fg-update-cpt! fg factor-idx new-cpt-tensor)` |
| `free-energy` | 2 | `(free-energy fg observations-tensor)` |
| `expected-free-energy` | 3 | `(expected-free-energy fg action-var action-state)` |
| `factor-graph?` | 1 | Test if value is a factor graph |
| `make-workspace` | 2 | `(make-workspace dim max-modules)` |
| `ws-register!` | 3 | `(ws-register! ws name process-fn)` |
| `ws-step!` | 1 | Step the workspace — one softmax cycle, winner broadcasts |
| `workspace?` | 1 | Test if value is a workspace |

---

*Next: Tutorial 5 — Signal Processing*
