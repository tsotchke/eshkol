# Tutorial 4: The Consciousness Engine

Eshkol's consciousness engine integrates three computational frameworks —
logic programming, active inference, and global workspace theory — into 22
builtins. This tutorial shows you how to use each one.

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
(logic-var? ?x)           ;; => #t (bare ?x, no quote)
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

```scheme
;; Create a factor graph with 3 variables, each with 2 states
(define fg (make-factor-graph 3))

;; Add factors (conditional probability tables)
;; Factor connecting variables 0 and 1
(fg-add-factor! fg 0 1 #(0.9 0.1 0.2 0.8))

;; Factor connecting variables 1 and 2
(fg-add-factor! fg 1 2 #(0.7 0.3 0.4 0.6))

;; Run belief propagation (10 iterations)
(fg-infer! fg 10)

;; Read the inferred beliefs
(display "Beliefs after inference:")
(newline)
```

### Free Energy

Free energy quantifies surprise — how much the model's predictions diverge
from observations:

```scheme
;; Compute variational free energy
;; Low free energy = good model of the data
(define fe (free-energy fg #(0 0)))  ;; observe var 0 in state 0
(display fe)
(newline)

;; Expected free energy for action selection
(define efe (expected-free-energy fg #(0 0)))
(display efe)
(newline)
```

### Learning with CPT Updates

```scheme
;; Update conditional probability table based on new evidence
(fg-update-cpt! fg 0 #(0.95 0.05 0.15 0.85))

;; Re-run inference — beliefs reconverge with new evidence
(fg-infer! fg 10)
```

---

## Part 3: Global Workspace Theory

The workspace implements a cognitive architecture where specialised
modules compete for global broadcast via softmax competition.

### Creating a Workspace

```scheme
;; Create a workspace
(define ws (make-workspace))

;; Register processing modules (each is a closure that takes
;; input and returns a content tensor with a salience score)
(ws-register! ws
  (lambda (input) #(0.8 0.5 0.3)))  ;; module 1: visual
(ws-register! ws
  (lambda (input) #(0.2 0.9 0.1)))  ;; module 2: auditory
(ws-register! ws
  (lambda (input) #(0.5 0.5 0.5)))  ;; module 3: memory

;; Step the workspace — modules compete via softmax,
;; winner broadcasts to all others
(ws-step! ws #(1.0 0.0 0.0))  ;; input stimulus
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
;; 1. Logic layer: store structured knowledge
(define kb (make-kb))
(kb-assert! kb (make-fact 'symptom 'fever))
(kb-assert! kb (make-fact 'symptom 'cough))

;; 2. Inference layer: model causal relationships
(define fg (make-factor-graph 3))
(fg-add-factor! fg 0 1 #(0.8 0.2 0.1 0.9))  ;; fever → flu
(fg-add-factor! fg 1 2 #(0.7 0.3 0.3 0.7))  ;; flu → fatigue
(fg-infer! fg 10)

;; 3. Workspace layer: compete for attention
(define ws (make-workspace))
;; Each module queries a different aspect of the problem
(ws-register! ws (lambda (in)
  ;; Logic module: query KB
  (if (null? (kb-query kb (make-fact 'symptom ?s)))
      #(0.1) #(0.9))))
(ws-register! ws (lambda (in)
  ;; Inference module: check free energy
  #(0.5)))
```

---

## All 22 Builtins

| Builtin | Description |
|---|---|
| `make-substitution` | Create empty substitution |
| `unify` | Unify two terms under a substitution |
| `walk` | Resolve a variable through substitutions |
| `make-fact` | Create a fact from a list pattern |
| `make-kb` | Create an empty knowledge base |
| `kb-assert!` | Add a fact to a KB |
| `kb-query` | Query a KB with a pattern (returns matching facts) |
| `logic-var?` | Test if value is a logic variable |
| `substitution?` | Test if value is a substitution |
| `kb?` | Test if value is a knowledge base |
| `fact?` | Test if value is a fact |
| `make-factor-graph` | Create a factor graph with N variables |
| `fg-add-factor!` | Add a factor (CPT) between two variables |
| `fg-infer!` | Run belief propagation for N iterations |
| `fg-update-cpt!` | Update a factor's conditional probability table |
| `free-energy` | Compute variational free energy |
| `expected-free-energy` | Compute expected free energy for action selection |
| `factor-graph?` | Test if value is a factor graph |
| `make-workspace` | Create a global workspace |
| `ws-register!` | Register a processing module (closure) |
| `ws-step!` | Step the workspace (softmax competition + broadcast) |
| `workspace?` | Test if value is a workspace |

---

*Next: Tutorial 5 — Signal Processing*
