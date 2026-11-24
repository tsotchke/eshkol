# Measure the moved CNOT's fault after Steane recovery

Both orders of this seven-wire preparation have the same ideal operator on all 128 input columns. The replay asks a different question. If a Y acts on each operand immediately after the same CNOT, do the original and reordered preparations give the same logical X readout under the published recovery table?

| Schedule | Position of source gate 12 | Fault | Measured syndrome | Recovery | Logical X before recovery | Logical X after recovery |
|---|---:|---|---|---|---:|---:|
| Depth 9 | 12 | None | 000000 | Identity | +1 | +1 |
| Depth 8 | 20 | None | 000000 | Identity | +1 | +1 |
| Depth 9 | 12 | YY | 000000 | Identity | +1 | +1 |
| Depth 8 | 20 | YY | 110110 | Y on wire 5 | +1 | −1 |

Indices are zero-based. The host asks the existing Eshkol scheduler for the reordered gate IDs and checks the permutation, both depths and the signed ideal operators. It inserts two physical Y gates immediately after source gate 12, CNOT(0,6), wherever that gate appears. The decoder and checks are fixed before either execution.

## Reproduce

Use Python 3.10+ and the native compiler for the quantum feature branch. The recorded build used Linux x86-64 and the v1.3.6-evolve feature compiler at O1. Run from the checkout root with `eshkol-run` on PATH, or pass its executable path with `--eshkol`.

```sh
python3 verification/quantum-rewrites/fault_readout.py
```

For each row, the driver uses seeds 1 and 73. Each seed starts three fresh seven-wire regions, one per executable invocation. First it measures in the X basis and the Z basis on separate preparations. The three X-check parities and three Z-check parities form the syndrome. It looks that syndrome up in the published table, re-prepares the tape, applies the indicated correction, and measures transversal X. Each native run checks the full region gate log against its requested tape after measurement. Region optimization is disabled so the scheduled and inserted gates reach the configured state-vector backend in the specified order. [Recorded observations](fault-readout.receipt.json)

The 23 gates, six checks and all 22 decoder entries come from the first witness in the upstream fixture. The decoder file contains only the published entries. Both measured syndromes have an entry. The tape and decoder digests are checked before use. [Upstream source fixture](https://github.com/tsotchke/eshkol/blob/3f933148c73fee234cb142a714df2a6854837faa/repro/quantum-qec/results.json) · [Fixture license](steane-tape.LICENSE)

The result concerns one gate-local error and ideal recovery on replicated preparations in the built-in simulator. It supplies a counterexample to preserving the recovered logical X for every such fault. It does not estimate hardware error rates or model noisy syndrome extraction. The existing exact operator check covers the ideal circuits. This packet adds native observations and introduces no Lean theorem.
