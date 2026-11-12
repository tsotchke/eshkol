# `core.ml.bpe` — byte-pair-encoding tokenizer

**Source**: [`lib/core/ml/bpe.esk`](../../../lib/core/ml/bpe.esk)
**Require**: `(require core.ml.bpe)` — not loaded by `(require stdlib)`.
**Design**: [ADR-0049](../../design/adr/0049-neuro-symbolic-bridge.md)

Subword tokenization by learned merges (Sennrich, Haddow & Birch, 2016). Training splits each whitespace-separated word into characters, the last carrying the end-of-word marker `"</w>"`, and repeatedly merges the most frequent adjacent pair, counted over word frequencies; ties go to the lexicographically smallest pair, so training is deterministic. Encoding applies the learned merges to each word lowest rank first, which reproduces training's segmentation. A model is plain data: its ranked merges and its vocabulary, whose positions are the token ids.

## Functions

- `(bpe-train corpus num-merges)` — learn up to `num-merges` merges from a list of strings.
- `(bpe-encode m text)` — the token strings of a text.
- `(bpe-encode-ids m text)` — token ids; a character never seen in training is an error.
- `(bpe-decode m ids)` — the text back, end-of-word markers becoming single spaces.
- `(bpe-merges m)`, `(bpe-vocab m)`, `(bpe-token-id m token)`, `(make-bpe-model merges vocab)` — read a model or rebuild one from its data; `bpe-end-of-word` is the marker.

```scheme
;; bpe-ex.esk
(require core.ml.bpe)
(define m (bpe-train '("low low low lower newest newest widest") 6))
(display (bpe-merges m)) (newline)
(display (bpe-encode m "lowest newer")) (newline)
(define ids (bpe-encode-ids m "lowest newer"))
(display ids) (newline)
(display (bpe-decode m ids)) (newline)
```
```
((l . o) (e . s) (es . t</w>) (lo . w</w>) (e . w) (ew . est</w>))
(lo w est</w> n ew e r</w>)
(8 15 3 10 4 1 12)
lowest newer
```

## Verification
[`tests/ml/bpe_test.esk`](../../../tests/ml/bpe_test.esk) checks the first merges of the paper's example, determinism, segmentation of seen and unseen words, id round trips and a model rebuilt from its merges and vocabulary.
