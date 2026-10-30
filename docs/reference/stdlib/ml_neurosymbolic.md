# `core.ml.neurosymbolic` — symbol embeddings, soft unification, attention over the KB

**Source**: [`lib/core/ml/neurosymbolic.esk`](../../../lib/core/ml/neurosymbolic.esk)
**Require**: `(require core.ml.neurosymbolic)` — not loaded by `(require stdlib)`. It requires [`core.ad.tape`](ad_tape.md) itself.
**Design**: [ADR-0049](../../design/adr/0049-neuro-symbolic-bridge.md)

The consciousness engine's `unify` is exact: two ground symbols match only
when they are the same symbol. This module gives every symbol of a knowledge
base a learnable embedding vector and scores the match of two distinct symbols
with a differentiable similarity, so a program learns which symbols should
unify (`father` with `parent`, `cat` with `feline`). Variables, substitutions
and the occurs check stay the engine's own (`walk`, `unify`, `kb-query`).

An embedding table has two views:

- the **table view**: `embed!` returns a symbol's live vector; the symbol-pair
  functions `soft-unify`, `soft-unify-train!` and `kb-attention` read it.
- the **parameter view**: `embedding-params` packs every embedding into one flat
  vector `theta` (symbol *i* occupies `theta[i*dim .. i*dim+dim-1]`). Every
  function that takes `theta` is written over generic arithmetic, so a loss
  `(lambda (theta) ...)` built from them is differentiated exactly by
  `(gradient loss theta)` (reverse mode) or by `derivative` (forward mode).
  `embedding-load!` writes trained values back into the table.

## Embedding table

### `(make-embedding-table dim)`, `(emb-dim tbl)`, `(embedding-count tbl)`
A new empty table of `dim`-dimensional embeddings; its dimension; the number of symbols it holds.

### `(embed! tbl sym)`
The live vector of `sym`, created on first use. Initial values are symbol-deterministic (a hash of the printed symbol seeds an LCG), so runs are reproducible with no RNG state. Mutating the vector mutates the table.

### `(embedding-index tbl sym)`, `(embedding-symbols tbl)`
The position of `sym` in the parameter view (registering it when new); every symbol in index order.

### `(embedding-params tbl)`, `(param-embedding theta tbl sym)`, `(embedding-load! tbl theta)`
Pack the table into a fresh flat vector; read `sym`'s embedding out of `theta` (works on AD carriers; `sym` must already be registered, since a parameter vector has a fixed layout); copy `theta` back into the table (its length must equal `count * dim`).

### `(embedding-mask tbl syms)`
A 0/1 vector over the parameter view marking the blocks of `syms`. Multiply a gradient by it to train those embeddings and hold the others fixed.

## The knowledge-base bridge

### `(kb-facts kb pred arity)`
The facts `(pred a1 ... ak)` of a KB, read with `kb-query` through fresh logic variables, as ground lists in insertion order.

### `(embedding-attach-kb! tbl kb schema)`
Attach a KB. `schema` lists `(predicate . arity)`; every predicate and every symbol occurring in those facts gets an embedding. Returns the symbol count. `(embedding-kb tbl)` and `(embedding-schema tbl)` read the attachment back, and `(table-facts tbl)` lists every fact of the schema.

```scheme
;; ns-kb.esk
(require core.ml.neurosymbolic)
(define kb (make-kb))
(kb-assert! kb (make-fact 'parent 'ann 'bob))
(kb-assert! kb (make-fact 'parent 'bob 'cid))
(kb-assert! kb (make-fact 'likes 'bob 'tea))
(define tbl (make-embedding-table 4))
(display (embedding-attach-kb! tbl kb '((parent . 2) (likes . 2)))) (newline)
(display (embedding-symbols tbl)) (newline)
(display (kb-facts kb 'parent 2)) (newline)
(define theta (embedding-params tbl))
(display (vector-length theta)) (newline)
(display (param-embedding theta tbl 'bob)) (newline)
```
```
6
(parent ann bob cid likes tea)
((parent ann bob) (parent bob cid))
24
#(0.4730000000000001 -0.402 0.14300000000000002 0.7)
```

## Term embeddings

### `(embed-term theta tbl term)`
A symbol is its embedding; a logic variable is the zero vector; a number `x` is `x` in component 0; a compound `(f a1 ... ak)` is the holographic reduced representation `emb(f) + sum_i role_i (*) embed-term(a_i)` (Plate 1995), with fixed unit-norm role vectors per argument position and `(*)` circular convolution.

### `(circular-convolution a b)`
`c[k] = sum_j a[j] b[(k - j) mod n]`.

### `(vdot a b)`
Dot product of two equal-length vectors.

## Similarities

Each maps two embedding vectors to `[0, 1]` and is differentiable.

| Function | Value |
|---|---|
| `(rbf-similarity a b mu)` | `exp(-\|a-b\|^2 / (2 mu^2))`; exactly 1 when `a = b` |
| `(cosine-similarity a b)` | `(1 + cos(a, b)) / 2` |
| `(sigmoid-similarity a b)` | `sigmoid(<a, b> / sqrt(dim))` |

`(make-rbf-similarity mu)` and `(make-cosine-similarity)` return the two-argument procedures the soft-unification functions take.

## Soft unification

### `(soft-unify-terms theta tbl t1 t2 sigma sim)`
Returns `(substitution . score)` or `#f`. Both terms are walked through `sigma` with `walk`; a logic variable is bound with the engine's `unify` (score 1, occurs check included); pairs unify componentwise and their scores multiply (product t-norm); two equal atoms score exactly 1; two distinct ground atoms score `(sim emb1 emb2)`; a structural mismatch (different arity, atom against pair) fails. Whenever `unify` succeeds, this returns the same bindings with score 1.

### `(soft-kb-query theta tbl pattern sim)`
Soft-unify `pattern` against every fact of the attached KB, across every predicate of the schema (so the predicate symbol matches softly too). Returns `(fact substitution score)` triples for the facts that unify structurally.

### `(soft-prove theta tbl goal sim)`, `(soft-prove-score theta tbl goal sim)`
The best proof `(score fact substitution)` of `goal` from the facts, or `#f`; and its score (0 when nothing unifies). The maximum is taken on the primal, so the gradient flows through the selected proof.

```scheme
;; ns-unify.esk
(require core.ml.neurosymbolic)
(define kb (make-kb))
(kb-assert! kb (make-fact 'parent 'ann 'bob))
(kb-assert! kb (make-fact 'likes 'bob 'tea))
(define tbl (make-embedding-table 4))
(embedding-attach-kb! tbl kb '((parent . 2) (likes . 2)))
(embedding-index tbl 'father)
(define theta (embedding-params tbl))
(define rbf (make-rbf-similarity 1.0))
(define s (make-substitution))
(define hard (soft-unify-terms theta tbl '(parent ?x bob) '(parent ann bob) s rbf))
(display (cdr hard)) (display " ") (display (walk '?x (car hard))) (newline)
(define soft (soft-unify-terms theta tbl '(father ?x bob) '(parent ann bob) s rbf))
(display (cdr soft)) (display " ") (display (walk '?x (car soft))) (newline)
(display (soft-unify-terms theta tbl '(parent ?x) '(parent ann bob) s rbf)) (newline)
(display (map (lambda (r) (list (car r) (caddr r))) (soft-kb-query theta tbl '(father ann ?y) rbf))) (newline)
(define best (soft-prove theta tbl '(father ann ?y) rbf))
(display (cadr best)) (display " ") (display (walk '?y (caddr best))) (newline)
(display (gradient (lambda (th) (soft-prove-score th tbl '(father ann bob) rbf)) theta)) (newline)
```
```
1 ann
0.5736269539233155 ann
#f
(((parent ann bob) 0.5736269539233155) ((likes bob tea) 0.056180587494525105))
(parent ann bob) bob
#(0.3487651879853759 -0.11931440641604961 0.04589015631386528 0.47725762566419855 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -0.3487651879853759 0.11931440641604961 -0.04589015631386528 -0.47725762566419855)
```

The gradient touches only the `parent` block (index 0) and the `father` block (index 5): the other symbols of the proof matched exactly.

## Attention over the knowledge base

### `(neural-query theta tbl pattern sim temperature)`
Answer a pattern with exactly one logic variable. Every fact that unifies structurally with the pattern is a key with logit `log(score) / temperature`, `score` its soft-unification score under `sim`; the answer of a fact is the value the variable is bound to. Returns an alist `(answer . probability)`, answers in first-occurrence order, probabilities summing to 1. Arguments equal in the pattern and a fact contribute exactly 1, so a predicate similarity learned on some entities carries over to entities never seen in training (Neural Theorem Provers, Rocktaschel & Riedel 2017). With `rbf-similarity` the logits are Gaussian-kernel attention over the predicate and argument embeddings.

### `(neural-query-probability theta tbl pattern answer sim temperature)`, `(neural-query-answer theta tbl pattern sim temperature)`
The probability of one answer (0 when absent); the most probable answer.

### `(attend q keys values temperature)`, `(attend-rbf q keys values mu)`, `(attend-facts theta tbl query facts temperature)`
Key-value attention returning `(weights . context)`, `weights` a list summing to 1 and `context = sum_i w_i v_i`. `attend` uses scaled dot-product logits `<q, k> / (temperature sqrt(dim))`; `attend-rbf` uses `-|q - k|^2 / (2 mu^2)`; `attend-facts` attends from a query term over fact terms with their term embeddings as keys and values.

### `(kb-attention tbl keys query)`, `(kb-retrieve tbl keys query)`
Symbol-key attention on the table view: an alist `(key . weight)` by scaled dot product, and its argmax.

```scheme
;; ns-attend.esk
(require core.ml.neurosymbolic)
(define kb (make-kb))
(kb-assert! kb (make-fact 'parent 'ann 'bob))
(kb-assert! kb (make-fact 'parent 'bob 'cid))
(kb-assert! kb (make-fact 'likes 'bob 'tea))
(define tbl (make-embedding-table 4))
(embedding-attach-kb! tbl kb '((parent . 2) (likes . 2)))
(define theta (embedding-params tbl))
(define rbf (make-rbf-similarity 1.0))
(display (neural-query theta tbl '(parent bob ?w) rbf 1.0)) (newline)
(display (neural-query-answer theta tbl '(parent bob ?w) rbf 1.0)) (newline)
(define r (attend-facts theta tbl '(parent bob ?w) (table-facts tbl) 1.0))
(display (car r)) (newline)
(display (circular-convolution (vector 1.0 0.0 0.0 0.0) (vector 0.5 0.25 0.0 -1.0))) (newline)
(display (cosine-similarity (vector 1.0 0.0) (vector 0.0 1.0))) (display " ") (display (rbf-similarity (vector 1.0 0.0) (vector 1.0 0.0) 0.5)) (newline)
```
```
((bob . 0.2795880438586839) (cid . 0.6490237091484821) (tea . 0.07138824699283398))
cid
(0.3642416851090726 0.405086787460393 0.23067152743053448)
#(0.5 0.25 0 -1)
0.5 1
```

## Symbol-pair training on the table view

### `(soft-unify tbl a b)`, `(soft-unify-loss tbl a b target)`
`sigmoid(<emb a, emb b> / sqrt(dim))` and its squared error against `target`.

### `(soft-unify-train! tbl a b target lr)`
One SGD step on both embeddings through the [`core.ad.tape`](ad_tape.md) stateful tape. Returns the loss before the update.

```scheme
;; ns-train.esk — learn that cat unifies with feline
(require core.ml.neurosymbolic)
(define tbl (make-embedding-table 8))
(display "before ")(display (soft-unify tbl 'cat 'feline))(newline)
(let loop ((i 0))
  (if (< i 200) (begin (soft-unify-train! tbl 'cat 'feline 1.0 0.5) (loop (+ i 1)))))
(display "after  ")(display (soft-unify tbl 'cat 'feline))(newline)
```
```
before 0.44357664326978336
after  0.96039327982842
```

## Learning with the parameter view

A complete program that learns the meaning of a word no fact uses is
[`examples/neurosymbolic_relation_paraphrase.esk`](../../../examples/neurosymbolic_relation_paraphrase.esk):
three queries about one family train `father` toward `parent`, and the same
queries about a second family are then answered from its `parent` facts with
probability above 0.96.

## Verification
[`tests/ml/neurosymbolic_bridge_test.esk`](../../../tests/ml/neurosymbolic_bridge_test.esk)
checks soft unification against `unify`, reverse-mode gradients against
forward mode (1e-12) and closed forms, attention normalisation, held-out
generalisation, and that no finite differences run;
[`tests/ml/neurosymbolic_test.esk`](../../../tests/ml/neurosymbolic_test.esk)
covers the table view.
