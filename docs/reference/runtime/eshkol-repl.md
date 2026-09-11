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
Framing goes to **stderr** so the program's own output on **stdout** stays
clean. As of protocol version 1 ("EREPL v1"), a `--machine` session also
accepts a versioned JSON request/response protocol on top of that framing,
so a driver — a Jupyter-style kernel, a language-server backend, a sister
project's test harness — can talk to `eshkol-repl` without a PTY, without
regexing prompts, and without classifying errors by matching this project's
error-message wording. `tools/erepl_client.py` is a complete, stdlib-only
Python reference implementation of everything in this section; treat it as
the executable form of this contract, and keep the two in sync.

### Backward compatibility

The original bare-line framing is unchanged and always emitted, whether or
not a session ever sends a JSON request:

- `EREPL READY` — once, after the JIT and stdlib have warmed up.
- `EREPL DONE` / `EREPL FAIL` — once per evaluated top-level form (bare
  legacy input *or* a `"op":"eval"` JSON request), before that form's
  structured response frame (see below).

A client that only ever watches those three lines and reads stdout between
them keeps working exactly as before. EREPL v1 is additive: everything
below is new frames and a new stdin input shape, layered on top of the
original protocol rather than replacing it.

### Frame grammar

Two kinds of line appear on **stdin** (requests) and **stderr** (framing +
responses); **stdout** carries only bytes an evaluated program itself wrote
via `display`/`write`/`print`/etc. — never protocol framing, and, for a
JSON `"op":"eval"` request, never an implicit echo of the form's own return
value either (see `value` under eval, below). This is the one property the
whole protocol exists to guarantee: a driver never has to guess which
stdout bytes are "the answer" and which are the program's own output,
because the answer is never on stdout at all — it is always the response
frame's `value` field.

- **Legacy bare form** (stdin): a Scheme form, newline-terminated with
  balanced parens (multi-line input accumulates exactly as it does
  interactively). Evaluated with the original auto-display behavior: a
  non-definition form's result is printed to stdout via an implicit
  `display`, indistinguishable there from anything the form printed itself.
  Kept only for compatibility with the original, pre-v1 protocol; new
  clients should use the JSON `"op":"eval"` request instead, which reports
  the value separately from stdout and gives it a structured type.
- **EREPL v1 JSON request** (stdin): one line whose first non-whitespace
  character is `{`. No Eshkol source form can start with `{`, so this can
  never collide with a legacy bare form. A request is exactly one JSON
  object with string-valued fields (`{"id": "...", "op": "...", ...}`); it
  is always exactly one stdin line — embedded newlines in a `code` field
  travel JSON-escaped, not raw, and the request is dispatched as soon as
  the line is read (it does not go through the legacy multi-line
  accumulator).
- **EREPL v1 response frame** (stderr): one line, `EREPL/1 ` followed by a
  single-line JSON object, `\n`-terminated with an explicit flush — as with
  every stderr line in this protocol. Every response frame that answers a
  request whose JSON carried an `"id"` echoes that id back verbatim in its
  own `"id"` field; a request with no `"id"` (or an empty one) gets `"id":
  null` back. This is what lets a driver pair requests with responses on a
  single stderr stream even when, in principle, more than one is ever
  in flight (v1 itself is a strict one-at-a-time request/response cycle —
  see "Concurrency", below — but the id is there so a driver never has to
  assume that stays true forever).

### `ready` — protocol handshake

Emitted once, immediately after `EREPL READY`, before the session reads its
first line of input:

```json
{"type":"ready","protocol_version":1,"pid":12345,"eshkol_version":"1.3.5-evolve"}
```

- `protocol_version` (integer) — the EREPL protocol version this build
  speaks. A driver should refuse to proceed with the JSON protocol (falling
  back to the legacy bare-form protocol, or failing outright) if this is
  not a version it understands.
- `pid` (integer) — the child process's own process id, for `interrupt()`
  (see below) and for a driver's own process bookkeeping. Never assume this
  equals the pid the driver's own spawn call returned; on some platforms —
  and always if the driver has gone through a shell — they can differ.
- `eshkol_version` (string) — `eshkol --version`'s version string, for
  diagnostics; not part of the compatibility contract (see below).

### `"op":"eval"` — evaluate one top-level form

Request:

```json
{"id":"1","op":"eval","code":"(+ 1 2)"}
```

`code` must parse as exactly one top-level Eshkol form. Evaluation does
**not** auto-display a result to stdout the way the legacy bare-form
protocol does: stdout receives only what the form's own code explicitly
writes, and the form's own value is reported separately, in `write` form
(quoted strings, `#t`/`#f`, etc. — never `display` form, which is
ambiguous for strings vs. symbols).

Before the response frame, the legacy `EREPL DONE` / `EREPL FAIL` bare line
is still emitted (see "Backward compatibility"), so a driver watching only
the old sentinel still sees a form complete.

Success response:

```json
{"type":"result","id":"1","ok":true,"stdout":"","value":"3","value_type":"integer"}
```

- `stdout` (string) — exactly the bytes this evaluation wrote to stdout,
  embedded directly in the frame rather than left for the driver to read
  off the raw stdout pipe. This is deliberate, not a convenience: stdout
  and stderr are two independent OS pipes, and nothing guarantees a reader
  observes "the stdout bytes are available" and "the stderr response frame
  is available" in the order this process produced them, even though it
  always finishes writing an evaluation's stdout before flushing that
  evaluation's response frame. Relying on frame order removes the race
  instead of asking every driver, on every platform, to get a two-pipe race
  right. (The same bytes are also replayed to the real stdout pipe once
  capture ends, so a plain pipe-tailing consumer — a human, a log —
  still sees them; just after the form finishes rather than incrementally
  while it runs.)
- `value` / `value_type` — the form's own value, and its coarse runtime
  type name (`integer`, `real`, `boolean`, `string`, `pair`, `symbol`,
  `procedure`, `vector`, `null` for an unspecified/no-value result, etc. —
  the same classification the language exposes as `type-of`). A definition
  form (`define`, ...) evaluates to an unspecified value, reported as
  `value_type: "null"`.

Failure response:

```json
{"type":"result","id":"2","ok":false,"stdout":"","error":{
  "kind":"type-error",
  "message":"car: argument is not a pair",
  "line":null,
  "column":null,
  "filename":null,
  "printed":"type-error: car: argument is not a pair",
  "irritants":[]
}}
```

`error.kind` is a member of a small, closed, stable set — this is the field
a driver should classify on, never `error.message`, which is prose and can
change wording across releases without notice:

| kind | meaning |
| --- | --- |
| `error` | generic R7RS `error`/condition |
| `type-error` | type mismatch |
| `file-error` | file operation failed |
| `read-error` | read/parse error raised by running code (e.g. `read`) |
| `syntax-error` | syntax error raised by running code |
| `range-error` | index/value out of range |
| `arity-error` | wrong number of arguments |
| `divide-by-zero` | division by zero |
| `user-exception` | a user-defined condition type (`raise`/`error` with a custom type) |
| `parse-error` | `code` itself failed to parse as one top-level form (never reached evaluation) |
| `interrupted` | the evaluation was aborted by `interrupt()` — see below |
| `crash` | a native crash (segfault, floating-point exception, ...) during evaluation, recovered the same way the interactive REPL recovers from one |
| `internal-error` | an unexpected internal (C++-level) failure outside normal Eshkol condition handling |

`error.line` / `error.column` / `error.filename` are `null` when the
underlying condition carries no source location (most runtime errors, all
of `parse-error`/`interrupted`/`crash`/`internal-error`); `error.printed`
is the same human-readable one-line summary the interactive REPL would
print, for logging; `error.irritants` is the condition's R7RS irritants
(additional data attached via `error`/`raise`), each in `write` form —
empty for conditions that carry none.

### `"op":"complete"` — identifier completion

```json
{"id":"3","op":"complete","prefix":"str-"}
```
```json
{"type":"completion","id":"3","matches":["string-append","string-length",...]}
```

`matches` is every builtin and session-defined identifier starting with
`prefix`, sorted and deduplicated — the same candidate set interactive tab
completion draws from.

### `"op":"is_complete"` — does this text form a complete expression?

```json
{"id":"4","op":"is_complete","code":"(display 1"}
```
```json
{"type":"is_complete","id":"4","status":"incomplete"}
```

`status` is one of `"complete"`, `"incomplete"` (more input needed — e.g. an
editor should keep accepting lines rather than submit), or `"invalid"` (an
unmatched closing paren; no amount of further input fixes it). This uses
exactly the same paren/string/comment scan the interactive multi-line
editor uses to decide when to submit, so it can never drift from what
submitting the same text interactively would actually do.

### `"op":"reset"` — clear session bookkeeping

```json
{"id":"5","op":"reset"}
```
```json
{"type":"reset","id":"5","ok":true}
```

Clears this session's tracked-definitions list (used for completion and
duplicate-definition detection). Per-session JIT symbols are **not**
undone — the same caveat the interactive `:reset` command documents; there
is no way to unload a JIT-compiled definition.

### `"op":"shutdown"` — end the session cleanly

```json
{"id":"6","op":"shutdown"}
```
```json
{"type":"shutdown","id":"6","ok":true}
```

The response frame is flushed, and the process then exits through the same
ordered teardown as `:quit` / EOF (joins JIT worker threads, runs runtime
shutdown hooks) rather than a bare `exit()`. Prefer this over closing stdin
or killing the process: it avoids a spurious abort on some platforms if
JIT worker threads are still holding runtime locks.

### Interrupting a running evaluation

There is deliberately no `"op":"interrupt"` request: while a form is
evaluating, the session's stdin reader is blocked inside that evaluation
and cannot see a new request frame arrive. Interrupt is instead delivered
out-of-band, as a signal — `SIGINT` on POSIX, `CTRL_BREAK_EVENT` (via
`GenerateConsoleCtrlEvent`) on Windows, sent to the child's `pid` (from the
`ready` frame). This aborts the in-flight evaluation and returns its
`"op":"eval"` response with `error.kind: "interrupted"`; the session
remains fully usable for the next request. Sending it while the session is
idle (no evaluation in flight) is a no-op with no response owed, matching
the semantics a Jupyter-style kernel's own interrupt already has.

### Malformed requests and unrecognized operations

A stdin line that starts with `{` but is not valid EREPL v1 JSON, or whose
`"op"` is not one of the operations above, never reaches evaluation at all.
It gets a distinct frame type instead of `"result"`, so a driver never has
to infer "nothing was evaluated" from context:

```json
{"type":"error","id":null,"error":{"kind":"protocol-error","message":"unknown op: bogus", ...}}
```

### Concurrency

EREPL v1 is a strict request/response cycle: send one request, wait for its
response, then send the next. There is no request pipelining, and (aside
from `interrupt()`, which is out-of-band by design — see above) no way to
cancel or overlap two in-flight requests. A driver that wants concurrent
evaluation should run multiple `eshkol-repl --machine` child processes
rather than multiplex one.

### Compatibility promise

- Within protocol version 1, fields are **never removed or repurposed**
  from a response frame's JSON shape once shipped; new fields may be
  added, and existing clients must ignore fields they don't recognize.
  `error.kind`'s enumerated set gains new members only — an existing value
  keeps its exact meaning.
- A breaking change (removing/repurposing a field, changing a request's
  required shape, changing `error.kind`'s meaning for an existing value)
  requires a new protocol version, announced as a new `protocol_version` in
  the `ready` frame and a new frame prefix (`EREPL/2 ` alongside, not
  instead of, `EREPL/1 ` for as long as v1 clients are expected to keep
  working) — mirroring how this version was introduced additively on top of
  the original bare-line framing rather than replacing it.
- `eshkol_version` is diagnostic only and never part of this contract;
  don't branch on it.

### Example transcripts

Success:

```
spawn:   eshkol-repl --machine
stderr:  EREPL READY
stderr:  EREPL/1 {"type":"ready","protocol_version":1,"pid":12345,"eshkol_version":"1.3.5-evolve"}
stdin:   {"id":"1","op":"eval","code":"(display (* 6 7))"}
stdout:  42
stderr:  EREPL DONE
stderr:  EREPL/1 {"type":"result","id":"1","ok":true,"stdout":"42","value":"()","value_type":"null"}
```

Runtime error:

```
stdin:   {"id":"2","op":"eval","code":"(car (quote ()))"}
stderr:  EREPL FAIL
stderr:  EREPL/1 {"type":"result","id":"2","ok":false,"stdout":"","error":{"kind":"type-error","message":"car: argument is not a pair","line":null,"column":null,"filename":null,"printed":"type-error: car: argument is not a pair","irritants":[]}}
```

Interrupt (a runaway evaluation, aborted mid-flight, session still usable
afterward):

```
stdin:      {"id":"3","op":"eval","code":"(let loop () (loop))"}
  (out-of-band: driver sends SIGINT to the child's pid)
stderr:     EREPL FAIL
stderr:     EREPL/1 {"type":"result","id":"3","ok":false,"stdout":"","error":{"kind":"interrupted","message":"evaluation interrupted","line":null,"column":null,"filename":null,"printed":"evaluation interrupted","irritants":[]}}
stdin:      {"id":"4","op":"eval","code":"(+ 40 2)"}
stderr:     EREPL DONE
stderr:     EREPL/1 {"type":"result","id":"4","ok":true,"stdout":"","value":"42","value_type":"integer"}
```

This is the mechanism that answers the JIT cold-start cost for embedding
projects: cold `eshkol-run -r` re-pays JIT setup each invocation, whereas a
persistent `eshkol-repl --machine` worker has a marginal per-form cost of
roughly zero — and, as of EREPL v1, a driver can consume that worker
directly, on any platform pipes work on, without a PTY.

### Reference client

`tools/erepl_client.py` is a stdlib-only Python reference driver
(`EReplClient`: `execute`, `interrupt`, `complete`, `is_complete`,
`reset`, `shutdown`) implementing everything on this page, plus a
`--self-test` that spawns a built `eshkol-repl` and exercises every request
type, including an interrupted infinite loop and a structured runtime
error:

```sh
python3 tools/erepl_client.py --self-test --binary build/eshkol-repl
```

## Related

- [JIT internals](jit-internals.md) — the stdlib object cache and code-model
  behavior that make warm evaluation fast.
- [`eshkol-run`](eshkol-run.md) — `-r` / `-e` one-shot JIT execution and the
  persistent run cache.
