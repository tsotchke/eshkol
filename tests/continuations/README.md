# Continuation re-entry measurement fixtures

These three programs measure exactly what happens when a captured
`call/cc` continuation is invoked more than once — specifically, after the
dynamic extent that captured it has already exited (the classic
generator/`amb` shape). They were written to settle a documentation
dispute: `docs/reference/language/continuations.md` claimed Eshkol has
"full re-entrant continuations, not merely a one-shot escape," while three
other docs (`docs/breakdown/CONTINUATIONS.md`,
`docs/COMPLETE_LANGUAGE_SPECIFICATION.md`,
`docs/internal/ESHKOL_V1_LANGUAGE_REFERENCE.md`,
`docs/breakdown/SCHEME_COMPATIBILITY.md`) state call/cc is single-shot via
setjmp/longjmp. Neither side had a re-entry test to point to.

See `docs/reference/language/continuations.md` for the resolved, precise,
per-engine account of current behavior, and `.icc/silent-wrong-ledger.yaml`
entries SW-51 (native, LOUD-ERROR) and SW-52 (bytecode VM, SILENT-WRONG,
waived) for the ledgered defects these fixtures reproduce.

## Why these are not wired into any pass/fail CI harness

Every file here reproduces either a crash (native JIT/AOT) or an infinite
loop / silently wrong transcript (bytecode VM) by design — that is the
finding. There is no well-formed PASS/FAIL marker convention in this
repository's test harnesses for "must crash the same way it did last time"
or "must hang," and building one was judged not worth the risk of an
unstable CI lane for a measurement fixture. Run them manually:

```
# doc_example_multishot.esk — correct on native, hangs on the VM (kill it):
./build/eshkol-run -r tests/continuations/doc_example_multishot.esk
ESHKOL_VM_NO_DISASM=1 timeout 5 ./build/eshkol-vm-standalone-test tests/continuations/doc_example_multishot.esk

# reentry_after_function_return.esk — SIGILLs on native, correct on the VM:
./build/eshkol-run -r tests/continuations/reentry_after_function_return.esk
ESHKOL_VM_NO_DISASM=1 ./build/eshkol-vm-standalone-test tests/continuations/reentry_after_function_return.esk

# generator_coroutine.esk — SIGSEGVs on native, silently wrong on the VM:
./build/eshkol-run -r tests/continuations/generator_coroutine.esk
ESHKOL_VM_NO_DISASM=1 ./build/eshkol-vm-standalone-test tests/continuations/generator_coroutine.esk
```

(macOS has no `timeout(1)` by default; wrap with a manual background +
`kill` if it is not installed.)

## What this settles

- `call/cc` being escape-only is not, by itself, a defect — most real uses
  of `call/cc` (early return, exception-style unwinding) are escape-only
  and native handles those correctly and efficiently.
- The defect is that `continuations.md` claimed capability the
  implementation does not have, and that the three other docs disagreed
  with it instead of all three being reconciled to the measured truth.
- Full multi-shot re-entrant continuations remain a tracked goal — see the
  "Build item: full multi-shot re-entrant continuations" section of
  `docs/reference/language/continuations.md` for scope and target.
