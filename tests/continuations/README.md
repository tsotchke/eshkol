# Continuation re-entry fixtures

These programs pin what happens when a captured `call/cc` continuation is
invoked more than once — in particular after the dynamic extent that captured
it has already exited, which is the shape every generator, coroutine and
`amb`-style backtracking search needs.

They are **gated in CI** by `scripts/run_continuation_tests.sh`, which runs
each fixture on all three engines — native JIT (`-r`), native AOT, and the
bytecode VM — and compares the transcript against the committed expected file
in `expected/`. Comparing the exact transcript on all three engines, rather
than looking for a `PASS` marker, is deliberate: these programs measure WHERE
control resumes, and the way to get that wrong is to produce plausible output
in the wrong order or from the wrong extent. Output is normalised the same way
`scripts/run_vm_parity.sh` normalises it (banner lines stripped, all newlines
removed), because the VM emits a newline after every `display` where native
emits none.

```
scripts/run_continuation_tests.sh              # all fixtures, all three engines
BUILD_DIR=build scripts/run_continuation_tests.sh
```

## The fixtures

| fixture | what it pins |
| --- | --- |
| `doc_example_multishot.esk` | the documented top-level multi-shot example; regression test for SW-61 |
| `reentry_after_function_return.esk` | re-entry after the capturing frame returned; regression test for SW-60 |
| `generator_coroutine.esk` | a generator that captures its return continuation once, inside the producer |
| `generator_multishot.esk` | a correctly structured generator, re-capturing per request |
| `amb_backtracking.esk` | McCarthy `amb`: each choice point re-entered once per alternative |
| `region_capture_resume.esk` | capture inside `with-region`, resumed after the region exits |
| `assignment_conversion.esk` | a non-captured `set!`-assigned local survives continuation re-entry (SW-62) |
| `assignment_binding_forms.esk` | adversarial coverage for parameters, named-let, do, let-values, internal define, and letrec assignment conversion on native and VM |
| `assignment_scan_depth.esk` | mutation after 70 body expressions remains visible to continuation re-entry (no fixed scan window) |

## History

These fixtures were written to settle a documentation dispute, and originally
sat outside CI because every one of them either crashed (native SIGILL/SIGSEGV,
ledger SW-60) or hung / produced a wrong transcript (bytecode VM, SW-61) by
design — that was the finding. Both defects are fixed, so they are gates now.

Two expectations recorded during that investigation were themselves wrong and
have been corrected here:

- `generator_coroutine.esk` was said to owe
  `gen1: 1 / gen2: 2 / gen3: 3 / gen4: done`. It does not: the program captures
  `return-k` once, inside `producer`, so every `yield` returns into the extent
  of the FIRST consumer that entered the producer. Native and the VM — two
  independent implementations — now agree byte for byte on the transcript that
  actually follows. `generator_multishot.esk` is the correctly structured
  generator and does owe `gen1: 1 / gen2: 2 / gen3: 3 / gen4: done`.
- The second `about to re-invoke` line in
  `reentry_after_function_return.esk` is correct, not a replay defect: invoking
  `k` returns 11 into the `(display (f))` of the first line, and execution then
  continues forward through the remaining top-level forms.

See `docs/reference/language/continuations.md` for the per-engine account of
how re-entry is implemented and the ownership rule for regions. The VM-only
representation limit remains documented there; assignment conversion closes
SW-62 on both engines.
