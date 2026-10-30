# `core.ml.diff_logic` — differentiable logic programs

**Source**: [`lib/core/ml/diff_logic.esk`](../../../lib/core/ml/diff_logic.esk)
**Require**: `(require core.ml.diff_logic)` — not loaded by `(require stdlib)`. It requires [`core.ml.neurosymbolic`](ml_neurosymbolic.md) for `kb-facts`.
**Design**: [ADR-0049](../../design/adr/0049-neuro-symbolic-bridge.md)

Forward chaining with soft truth values, so gradients flow through rule
application. A *valuation* assigns every ground atom `p(a, b)` of a binary
predicate over a finite domain a truth value in `[0, 1]`, stored as a flat
`n x n` vector (`R(x, y)` at index `x*n + y`). Extensional predicates are read
from the consciousness engine's knowledge base; rules derive intensional
predicates round by round with

| Connective | Soft form |
|---|---|
| `a AND b` | `a * b` (product t-norm) |
| `a OR b` | `a + b - a * b` (probabilistic sum) |
| `NOT a` | `1 - a` |
| `EXISTS z` | `1 - prod_z (1 - .)` (noisy-or over `z`) |

On 0/1 relations every operator is the exact relational one, so a soft
program with one-hot rule weights computes the same atoms as the crisp
program.

A rule **template** fixes the shape of a clause and leaves its body predicates
to be learned (Evans & Grefenstette, *Learning Explanatory Rules from Noisy
Data*, 2018). Each body slot holds a softmax over the candidate predicates and
uses the softmax-weighted mixture of their relations (the Neural LP / TensorLog
relaxation of choosing one predicate):

| Kind | Clause |
|---|---|
| `chain` | `head(x, y) <- P(x, z), Q(z, y)` |
| `conj` | `head(x, y) <- P(x, y), Q(x, y)` |
| `inverse` | `head(x, y) <- P(y, x)` |
| `copy` | `head(x, y) <- P(x, y)` |

A head may itself be a candidate, so recursive programs are written directly;
`steps` bounds the derivation depth. Every function is written over generic
arithmetic, so a loss built from `dl-forward-chain` is differentiated exactly
by `(gradient loss theta)` over the rule logits.

## Domains and relations

### `(dl-domain constants)`, `(dl-domain-from-kb kb preds)`
A domain from a list of constants, or from every constant of the facts of the binary predicates `preds`, first-seen order. `(dl-domain-size dom)`, `(dl-domain-constants dom)` and `(dl-constant-index dom c)` read it.

### `(dl-relation-from-kb dom kb pred)`, `(dl-relation-zero dom)`
The 0/1 relation of the facts `pred(a, b)` (constants outside the domain are skipped); the all-false relation.

### `(dl-relation-ref dom r x y)`, `(dl-relation-set! dom r x y v)`
Read and write `R(x, y)` by constant index.

## Connectives and relation operators

`(dl-and a b)`, `(dl-or a b)`, `(dl-not a)` on truth values;
`(dl-compose dom r s)` is `EXISTS z. R(x, z) AND S(z, y)`;
`(dl-conjoin r s)` and `(dl-disjoin r s)` are pointwise;
`(dl-inverse dom r)` is `R(y, x)`;
`(dl-mix weights rels)` is `sum_i w_i R_i`;
`(dl-softmax theta off k)` is the softmax of `theta[off .. off+k-1]` as a list.

## Programs

### `(make-dl-program candidates rules)`
`candidates` lists the predicates a body slot may choose from; `rules` lists `(head kind)`. The program's parameters are one block of `|candidates|` logits per body slot, in rule order: `(dl-program-param-count prog)`. `(dl-program-init-params prog)` is the all-zero vector (every candidate equally likely).

### `(dl-forward-chain theta prog dom base steps)`
`base` is an alist `(pred . relation)` of the extensional predicates. Every rule head starts at its base relation (or false); each of the `steps` rounds ORs into every head what each of its rules derives from the previous round's valuation. Returns the final valuation as an alist; `(dl-valuation-ref val pred)` reads one relation.

### `(dl-cross-entropy r target)`
Binary cross-entropy of a relation against a 0/1 target, entries clamped into `[1e-7, 1 - 1e-7]`.

### `(dl-rule-weights theta prog)`, `(dl-extract-rules theta prog)`
The learned softmax weights per rule and slot; and each rule as a symbolic clause with its most probable body, paired with a confidence (the product of the chosen weights):
`(((grandparent ?x ?y) <- (parent ?x ?z) (parent ?z ?y)) . 0.99...)`.

### `(dl-true-atoms dom r pred threshold)`, `(dl-assert-derived! kb dom r pred threshold)`
The ground atoms whose truth exceeds `threshold`; and assert each of them the KB does not already hold, returning how many were added. The exact engine then answers `kb-query` from them.

```scheme
;; dl-grandparent.esk
(require core.ml.diff_logic)
(define kb (make-kb))
(for-each (lambda (f) (kb-assert! kb (make-fact (car f) (cadr f) (caddr f))))
          '((parent ann bob) (parent bob cid) (parent bob eli) (parent cid dan)
            (spouse ann fox) (spouse fox ann)))
(define dom (dl-domain-from-kb kb '(parent spouse)))
(define base (list (cons 'parent (dl-relation-from-kb dom kb 'parent))
                   (cons 'spouse (dl-relation-from-kb dom kb 'spouse))))
(define prog (make-dl-program '(parent spouse) '((grandparent chain))))
(define target (dl-compose dom (cdr (assq 'parent base)) (cdr (assq 'parent base))))
(define (loss theta)
  (dl-cross-entropy (dl-valuation-ref (dl-forward-chain theta prog dom base 1) 'grandparent) target))
(define theta (dl-program-init-params prog))
(let train ((k 0))
  (if (< k 100)
      (let ((g (gradient loss theta)))
        (let upd ((i 0))
          (if (< i (vector-length theta))
              (begin (vector-set! theta i (- (vector-ref theta i) (* 0.5 (vector-ref g i)))) (upd (+ i 1)))))
        (train (+ k 1)))))
(display (car (car (dl-extract-rules theta prog)))) (newline)
(define gp (dl-valuation-ref (dl-forward-chain theta prog dom base 1) 'grandparent))
(display (dl-true-atoms dom gp 'grandparent 0.5)) (newline)
(display (dl-assert-derived! kb dom gp 'grandparent 0.5)) (newline)
(display (map (lambda (s) (walk '?g s)) (kb-query kb '(grandparent ann ?g)))) (newline)
```
```
DL_GRANDPARENT_OUTPUT
```

## A complete program

[`examples/neurosymbolic_kinship.esk`](../../../examples/neurosymbolic_kinship.esk)
learns `grandparent` and the recursive `ancestor` from one family's facts,
prints the learned clauses, runs them on a family the training never saw, and
asserts the conclusions into that family's KB, where `kb-query` answers from
them.

## Verification
[`tests/ml/diff_logic_test.esk`](../../../tests/ml/diff_logic_test.esk) checks
the crisp semantics of every operator, reverse mode against forward mode
through rule application (1e-10), that training recovers
`grandparent <- parent o parent` and a recursive `ancestor` program, that the
learned programs decide every atom of a held-out family correctly, and that
derived atoms asserted into the KB are returned by `kb-query`.
