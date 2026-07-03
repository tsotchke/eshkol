# `eshkol-repl` — Interactive REPL & Warm-Worker Protocol

`eshkol-repl` is the interactive read-eval-print loop built on the same in-process
LLVM JIT that powers `eshkol-run -r`. It also exposes a **machine-driven mode**
(`--machine`) for use as a long-running, JIT-warm worker by sister projects.

## Options

```
Usage: eshkol-repl [OPTIONS]

  --stdlib, -s    Load standard library on startup
  --machine, -m   Machine-driven mode: emits EREPL READY/DONE/FAIL framing on
                  stderr; suppresses banner / prompts; implies --stdlib.
  --help, -h      Show this help message
```

Interactively, forms are read (newline-terminated with balanced parens),
evaluated, and results printed. See `repl_utils.h` for the built-in help topics
(e.g. `with-region [name] [size] body ...`).

## Machine mode (EREPL protocol)

`--machine` turns the REPL into a warm worker: it loads the stdlib and JIT once,
then evaluates forms sent on stdin without ever paying the cold-start cost again.
Framing sentinels go to **stderr** so the program's own output on **stdout**
stays clean.

### Protocol

1. Spawn `eshkol-repl --machine`.
2. Read **stderr** until `EREPL READY` — the JIT and stdlib are warm. This is
   emitted exactly **once** after initialization.
3. Send one top-level form on **stdin** (newline-terminated, balanced parens).
4. Read the form's output on **stdout** until you see `EREPL DONE` (success) or
   `EREPL FAIL` (error) on **stderr**. One of these is emitted **once per
   top-level form**.
5. Repeat step 3-4 for each subsequent form.

Each sentinel ends with `\n` and an explicit flush. `EREPL FAIL` is emitted when
a form raises or fails to parse; the `DONE`/`FAIL` sentinel is written *after*
cleanup so any exception text has already flushed to stdout.

### Example session (conceptual)

```
spawn:   eshkol-repl --machine
stderr:  EREPL READY
stdin:   (display (* 6 7))(newline)
stdout:  42
stderr:  EREPL DONE
stdin:   (car '())
stdout:  <error text>
stderr:  EREPL FAIL
```

This is the mechanism that answers the JIT cold-start cost for embedding
projects: cold `eshkol-run -r` re-pays JIT setup each invocation, whereas a
persistent `eshkol-repl --machine` worker has a marginal per-form cost of
roughly zero.

## Related

- [JIT internals](jit-internals.md) — the stdlib object cache and code-model
  behavior that make warm evaluation fast.
- [`eshkol-run`](eshkol-run.md) — `-r` / `-e` one-shot JIT execution and the
  persistent run cache.
