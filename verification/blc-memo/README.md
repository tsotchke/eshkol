# Checking the destination of a memoized BLC result

Two distinct suspensions both evaluate to identity. A faulty evaluation stores the result of cell 1 in cell 0, then returns identity. Checking every returned term still accepts this run. Replaying the writes against the pending update frame rejects the first write to cell 0.

The memoizing evaluator lives under `fixtures/core/`. The replay builds it in a private module directory alongside the checkout’s `core.blc`. The user-facing evaluator remains the one shipped by the standard library.

## Replay

Use Linux x86-64, Eshkol v1.3.6-evolve x64 lite and matching LLVM 21 libraries, a C++ compiler, GNU binutils, Valgrind with Callgrind, Python with llvmlite 0.46, and Lean 4.35.0 on PATH.

```sh
python3 verification/blc-memo/replay.py --linked-refinement
```

The uninstrumented AOT and JIT runs establish the comparison results. An instrumented run records allocation, update and use. A separate native interposer counts linked setter returns. The replay also checks the valid-pointer paths in the inspected x86-64 pair primitives. Changing the destination, omitting a whole invocation or duplicating a linked setter gives distinct failures.

## What the results cover

| Question | Evidence |
|---|---|
| Which normal forms can the machine return? | `register_reference_correspondence` relates successful normalization with some finite fuel in each model. The cost units differ. [Correspondence](Correspondence.lean) |
| Which suspension does this frame own? | Heap invariants, lexical lookup and the two publication stores describe the logical cell and its payload. [Heap](Heap.lean) · [Publication](Publication.lean) · [Lookup](Lookup.lean) |
| Are the store and invocation records complete? | The replay compares the instrumented records with the control run and tests missing and duplicate events. [Observation](Observation.lean) · [Dispatch](Dispatch.lean) |
| How does strong readback reach syntax? | The register model reconstructs lambdas and neutral applications. The one-based bridge checks the codec and substitution step. [Readback](Readback.lean) · [Syntax](Syntax.lean) |
| What did the compiled fixture actually do? | The linked replay interprets a limited set of pair instructions and samples this binary. [Linked-pair model](LinkedPair.lean) · [ELF reader](linked_pair.py) |

The register theorem uses `propext`, `Classical.choice` and `Quot.sound`. The source inventory binds parsed forms in the fixture and the imported helpers. Compilation, arbitrary native instructions, malformed inputs, and preservation of the public beta budget under sharing remain outside the claim.
