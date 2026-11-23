# Context-bound quantum rewrites

A rewrite may preserve a one-wire channel while changing its controlled two-wire operator. Applying Y then X gives iZ. The replacement under control emits CZ followed by S on the control. Reversing the direction needs the opposite phase. The test changes only the emitted S to S†, keeps the accepted forward phase evidence, and checks the actual instructions against the source operator.

```sh
python3 verification/quantum-rewrites/replay.py
```

Use the feature branch’s `lib/core/quantum/circuit.esk`, Python 3.10 or newer, Lean 4.35.0, and the Linux x86-64 Eshkol v1.3.6-evolve x64 lite compiler with matching LLVM 21 libraries. Put lean and eshkol-run on PATH or pass their executable paths with `--lean` and `--eshkol`.

The native circuit library computes the ordered scalar from the exact input matrices. Lean checks the correction against an independently specified context, including wire order and authorized premises. The standalone consumer reads the emitted instructions. The replay also lowers those observed instructions to the branch’s circuit representation and checks exact operator equality there.

For the H/CNOT Steane example, the scheduler preserves all 23 gates and reduces ideal depth from 9 to 8. The replay validates wire occupancy, checks overlapping reordered pairs, and compares signed amplitudes on all 128 input columns. Equal Hadamard counts give the same denominator on both tapes. This models unrestricted connectivity and unit gate times. The MIT-licensed source tape is the first preparation in the previously published fixture. [Source fixture](https://github.com/tsotchke/eshkol/blob/3f933148c73fee234cb142a714df2a6854837faa/repro/quantum-qec/results.json) · [Fixture license](steane-tape.LICENSE)

This replay exercises the exact circuit library and the standalone emitter. The production region/backend handoff is a separate consumer.
