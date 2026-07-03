# `core.logging` — structured JSON-Lines logging

**Source**: [`lib/core/logging.esk`](../../../lib/core/logging.esk)
**Require**: `(require core.logging)` — **NOT** auto-loaded via `stdlib`; require it explicitly. (It internally pulls in `core.json`.)

Each log call emits a single JSON object on one line (JSON-Lines), so `jq`/Datadog/Elastic can parse without multi-line reassembly. Every record carries `ts`, `level`, `msg`, and (when scoped) `trace_id`; extra `fields` are appended. A level threshold and output sink are held in module-global state.

`fields` is a list of key/value entries. Because Eshkol's reader parses `'("k" . "v")` as a proper 3-element list rather than a dotted pair, use the **2-list form** `(list "k" v)` — the portable form. Values may be strings, numbers, booleans, symbols, or `null`; anything else is rendered via `write` into a quoted string. Reserved keys (`ts`, `level`, `msg`, `trace_id`) in `fields` raise an error.

## Functions

### `(log-debug msg [fields])` / `(log-info msg [fields])` / `(log-warn msg [fields])` / `(log-error msg [fields])`
Emit a record at that level if the current level threshold permits. `fields` is optional.

### `(log-set-level! level)`
Set the minimum level to emit: `'debug` < `'info` < `'warn` < `'error`. Invalid level → error.

### `(log-level)`
Return the current threshold symbol.

### `(log-set-output! sink)`
Route output. `sink` may be an output port, a path **string** (opened for writing, owned/closed by the logger on the next `log-set-output!`), or `#f` to reset to `current-output-port`.

### `(log-output-port)`
Return the current effective output port (the configured sink, or `current-output-port` when unset).

### `(log-with-trace! trace-id thunk)`
Run `thunk` with `trace_id` bound to `trace-id` for every log call inside it; the previous trace id is restored afterward (even on raise, via `dynamic-wind`).

```scheme
;; logging.esk
(require core.logging)
(display (log-level)) (newline)         ; info (default)
(log-set-level! 'debug)
(define p (open-output-string))          ; capture deterministically
(log-set-output! p)
(log-info "hello" (list (list "k" 1) (list "flag" #t)))
(log-warn "careful")
(log-with-trace! "trace-xyz"
  (lambda () (log-error "boom" (list (list "code" 500)))))
(display (get-output-string p))
```
```
info
{"ts":"2026-07-03T21:03:42.097Z","level":"info","msg":"hello","k":1,"flag":true}
{"ts":"2026-07-03T21:03:42.098Z","level":"warn","msg":"careful"}
{"ts":"2026-07-03T21:03:42.098Z","level":"error","msg":"boom","trace_id":"trace-xyz","code":500}
```
(The `ts` values are wall-clock and will differ per run; the rest is exact.)

Edge cases: `(log-set-level! 'trace)` → `Unhandled exception: log-set-level!: invalid level`. A `fields` entry using a reserved key (e.g. `(list "msg" "x")`) raises `log: field key is reserved`. `log-output-port` returns a port object, which `display`s as `#<unknown>` — it is meant for passing to port operations, not printing.
