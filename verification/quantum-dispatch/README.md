# Check two-wire rewrites at state-vector dispatch

In the four-gate encoding of controlled Y followed by controlled X, the
replacement is CZ followed by S on the control wire. Substituting S† leaves
the optimizer’s calculation of the scalar intact. The region rejects the
changed operator before the first backend call.

From the checkout root, with Python 3.10+, Lean 4.35.0 and a native
`eshkol-run` built for the quantum-types feature branch with its matching
runtime, run:

```sh
python3 verification/quantum-dispatch/replay.py
```

Set `ESHKOL_RUN` and `LEAN` to select installed executables. The replay uses
a temporary copy of this checkout’s library. It checks the Lean files, then
runs each native fixture in file-mode JIT and as an O1 AOT executable.
The normal region regression is also registered in CTest as
`quantum_authority_test_jit` and `quantum_authority_test_aot`.

The region snapshots the remapped pending tape, gives the optimizer a
separate copy and dispatches the checked copy. The replay changes the
phase, wire, global sign and exactness at emission. It also mutates an
instruction retained by the producer after the check. The backend must
receive the checked correction. An ordinary optimizer rewrite that
preserves only projective equality falls back to the original tape.

A second control makes native arena promotion fail after the executed log
has been prepared and before the amplitudes can be published. It checks the
identities of the pending queue, log, state box and amplitudes, the layout,
amplitude values and register liveness. Retrying the same register in a fresh
child arena must measure (1 0), with two backend attempts and one logged X.
A failed preparation can leave promoted storage in the destination arena
until its teardown.

`Q1.dispatch_sound` connects the modeled dispatch to the original pending
operator under the checker's soundness hypothesis.
`Q1.PhaseHandoff.wrong_never_dispatched` discharges that claim for the
incorrect correction. Its axiom dependencies are `propext`, `Classical.choice`
and `Quot.sound`. The region model treats successful publication as one
transition. The native retry result covers synchronous allocation failure
with one writer on the unrouted, built-in two-wire simulator. Compiler
refinement, concurrent observers, asynchronous interruption and external
device effects remain open.
