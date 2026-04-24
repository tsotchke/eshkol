# Known bug: `parameterize` + `current-output-port` doesn't redirect `display`

**Status:** OPEN — discovered 2026-04-24 while writing the Quirk 11
regression test. Predates the Quirk 11 fix; the first-class wrapper
inherits the same limitation as the call-site `display`.

## Symptom

```scheme
(define p (open-output-string))
(parameterize ((current-output-port p))
  (display "hello"))
(get-output-string p)   ;; => "" (expected "hello")
```

`"hello"` appears on stdout, not in the string port. R7RS §6.13.3
requires `display` (zero-argument port form) to write to
`(current-output-port)`; the parameterize binding should make the
string port the current one.

## Root cause (not yet fixed)

The runtime `eshkol_display_value` always writes to stdout (`stdout`
FILE*). It doesn't consult the `current-output-port` parameter's
current binding. `eshkol_display_value_to_port` exists but is only
called when the user passes an explicit port argument at the call
site: `(display x p)` works; `(display x)` under parameterize does
not.

## Why it didn't block Quirk 11

Quirk 11 is about the bare name `display` resolving to a callable
value. That contract is verified by
`first_class_io_test.esk` via side-effect counters — it does not
require output capture.

## Fix sketch (for v1.2.1 or later)

- Expose `current-output-port` as a real parameter object in the
  runtime, backed by a per-thread FILE* / port-pointer slot.
- `eshkol_display_value` (zero-argument port form) reads that slot.
- `parameterize` already rebinds parameter objects; it just needs
  to rebind THIS one.

## Related

- Port plumbing for `write`, `newline`, `read` likely has the same gap.
- None of the current Noesis code paths rely on parameterize +
  display (they pass ports explicitly), so this is not a hot-path
  blocker, just a correctness hole.
