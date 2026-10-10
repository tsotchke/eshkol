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

The script is part of `scripts/run_all_tests.sh`. At v1.3.6-evolve it reports
`continuations: 48 passed, 0 failed` (16 fixtures × three engines).

## The fixtures

| fixture | what it pins |
| --- | --- |
| `doc_example_multishot.esk` | the documented top-level multi-shot example (bytecode VM re-entry) |
| `reentry_after_function_return.esk` | re-entry after the capturing frame returned (native re-entry) |
| `generator_coroutine.esk` | a generator that captures its return continuation once, inside the producer |
| `generator_multishot.esk` | a correctly structured generator, re-capturing per request |
| `amb_backtracking.esk` | McCarthy `amb`: each choice point re-entered once per alternative |
| `region_capture_resume.esk` | capture inside `with-region`, resumed after the region exits |
| `assignment_conversion.esk` | a non-captured `set!`-assigned local survives continuation re-entry |
| `assignment_binding_forms.esk` | adversarial coverage for parameters, named-let, do, let-values, internal define, and letrec assignment conversion on native and VM |
| `assignment_guard_binding_forms.esk` | guard-handler mutation matrix for let, let*, let-values, letrec, internal define, parameters, and do on native and VM |
| `assignment_initializer_forms.esk` | continuation re-entry from let* and letrec initializers preserves mutable binding locations on native and VM |
| `assignment_scan_depth.esk` | mutation after 70 body expressions remains visible to continuation re-entry (no fixed scan window) |
| `assignment_guard_handler_capture.esk` | the shared native/VM observed-after-mutation analysis keeps a location captured by a guard handler live across re-entry |
| `guard_handler_snapshot.esk` | a native multi-shot continuation captured inside `guard` restores that guard's exception-handler chain on every invocation |
| `region_capture_resume_nested.esk` | capture two `with-region`s deep, resumed after both exit: every open region is pinned, not only the innermost |
| `region_escape_only_no_pin.esk` | an escape-only `call/cc` inside `with-region` (recognised by `callCCContinuationStaysLocal()`) takes no region pin |
| `region_handle_close_inside_callcc.esk` | the one shape where an escape-only `call/cc` still pins: a region handle closed inside the continuation's usable extent |

## History

These fixtures were written to settle a documentation question, and originally
sat outside CI because re-entry after the capturing extent had exited was not
yet supported: native stopped on a fatal signal and the bytecode VM did not
reproduce the transcript — that was the finding. Both engines now support it,
so the fixtures are gates.

Two expectations recorded during that investigation did not match R7RS
semantics, and the committed expected files carry the R7RS answer:

- `generator_coroutine.esk` was said to owe
  `gen1: 1 / gen2: 2 / gen3: 3 / gen4: done`. It does not: the program captures
  `return-k` once, inside `producer`, so every `yield` returns into the extent
  of the FIRST consumer that entered the producer. Native and the VM — two
  independent implementations — now agree byte for byte on the transcript that
  actually follows. `generator_multishot.esk` is the correctly structured
  generator and does owe `gen1: 1 / gen2: 2 / gen3: 3 / gen4: done`.
- The second `about to re-invoke` line in
  `reentry_after_function_return.esk` is the R7RS answer, not a replay: invoking
  `k` returns 11 into the `(display (f))` of the first line, and execution then
  continues forward through the remaining top-level forms.

See `docs/reference/language/continuations.md` for the per-engine account of
how re-entry is implemented and the ownership rule for regions. The VM-only
representation limit remains documented there; assignment conversion gives
`set!`-assigned locals the same re-entry behaviour on both engines.
