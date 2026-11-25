# Complete finite-class Busy Beaver maximality

Standalone exact replay of all labeled total two/three-state binary Turing
tables. This follow-up extends the single-machine halt/cost examples with
complete source coverage and nonhalting proofs. It reconstructs known values:
`S(2,2)=6`, `Sigma(2,2)=4`, `S(3,2)=21`, `Sigma(3,2)=6`.

Run from any directory:

```sh
python3 repro/busy-beaver-maximality/run_gate.py --eshkol-bin /path/to/eshkol-run
```

The gate uses the tested checkout's stdlib, compiles/runs JIT and AOT-O2,
compares exact complete receipts and independently decides every two-state
table with a sparse-tape Python implementation. The two finite representations
share the stated repeat/ray mathematical lemma, not simulation code. A separate
readiness control rejects equal but incomplete output.

## Exact source class

For n states there are `(4*(n+1))^(2*n)` labeled total tables, including every
halting write/direction. Initial state is 0, blank symbol is 0, and the tape
is infinite in both directions. Row `2*q+read` has digit
`write + 2*(direction + 2*next)`, with direction 0=left, 1=right, next=n=halt.
No symmetry quotient or omitted transition is counted as coverage.

## Portable proof data

`bb2-cover.sexp` and `bb3-cover.sexp` are Scheme datums, read as data and never
evaluated. The header `(bb-cover-v1 n node-count)` is followed by a preorder
tree. `(split slot)` has exactly `4*(n+1)` following child subtrees, one per
possible digit, and must split an unset row. Leaves are `(unknown)` or one
of the following checked proofs:

- `(halt steps)`: replay the actual source, halt at exactly that step and
  count the complete tape's nonblank cells.
- `(repeat start steps side)`: replay, then verify a full configuration cycle
  or an inductively repeatable left/right ray. The complete infinite suffix
  or prefix is compared; discarded tape stays behind all future translated
  reads. A finite frame contains every possibly written cell of the prefix.
- `(closed-control)`: all control-reachable rows are specified and avoid halt.
- `(ngram width modulus words)`: the complete initial tape's windows are
  admitted and every compatible local join preserves the supplied invariant.
  This is no cropping assumption. Width/phase and join budgets are explicit.
- `(regular left-DFA right-DFA relation)`: a finite relation between control,
  scanned symbol and the two tape-stack summaries. Blank padding is stationary;
  the initial tuple belongs; **every** DFA preimage of a pop stays in the
  relation; no related transition reaches halt or reads an unspecified row.

Every proof dependency must name an assigned source row. Therefore a leaf
proves all its cylinder's completions, rather than one convenient filling.
Split accounting proves exact disjoint coverage. Unknown cylinders preserve
their full mass and withhold both upper bounds. Missing children, repeated
splits, extra data, malformed invariants and exceeded budgets refuse.

The 445-node two-state partition covers 20,736 tables. The 115,409-node
three-state partition covers 16,777,216: 7,571,840 halt, 9,205,376 do not.
Its 512 regular certificates close the previous remaining obligations;
216 other leaves use local-word invariants. No proof search runs in the
recipient. Runtime regions release per-node proof/replay temporaries.

The gate replaces the first regular proof with `unknown`, retaining the
partition. Exactly one table becomes unresolved: upper time `21 -> unknown`
and upper productivity `6 -> unknown`, with lower bounds unchanged. Positive
rule controls and malformed-proof refusals ensure this does not merely test
matching output or a missing helper.

The checker profile admits n=2 or 3, at most 200,000 tree nodes, 64 replay
steps, DFA sizes up to eight per side, local widths/phases up to eight,
1024 words and 100,000 joins per local proof. The input reader allocates each
datum before shape admission; these limits do not bound every possible
malformed input's reader allocation. Proof files shipped here are finite and
hashed by the qualification receipt.

The trusted base is source encoding, the simulator and repeat/local/regular
induction rules, partition accounting, integer arithmetic, input reader and
compiler. This is executable checking, not a proof-assistant formalization,
new champion, larger-class maximality or a computable universal halting bound.
The earlier five-state example remains an individual halt proof; its maximality
does not follow from this packet. All selected source and proof data are MIT.

The native gate explicitly sets `ESHKOL_JIT_CACHE=0` for every child, so its
`-r` leg runs in-process JIT rather than the persistent executable-cache route.
The readiness self-test intercepts the real first child boundary with the
variable unset and set to `1` in the caller. Both must pass `0` to the child
without changing the caller environment; this control launches no native work.
The gate separately requires complete native receipts and independent oracles.
