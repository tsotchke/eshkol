# `agent.subprocess` — Process Spawning

Spawn, drive, and reap child processes. Two families exist: **shell** spawns
(`process-spawn-shell`, `run-command`) which pass a command string to `/bin/sh
-c`, and **argv** spawns (`process-spawn-argv`, `run-argv`) which exec a program
directly with no shell — safe against shell injection (#190).

```scheme
(require agent.subprocess)
```

Source: `lib/agent/subprocess.esk`. C symbols: `qllm_process_*`.

## Ownership & cleanup contract (#94)

A process handle is an opaque affine capability. **You own it until you call
`process-destroy` exactly once.** The native handle becomes a tombstone after
destroy, so a repeated destroy or any operation through a closed handle emits
a diagnostic and returns its documented failure sentinel instead of accessing
freed memory. The lifecycle is:

1. spawn → handle
2. drive (`process-write-stdin`, `process-read-*`, `process-wait`)
3. `process-destroy` — closes resources, reaps the child, and leaves a
   diagnostic tombstone for any later accidental use

The `process-read-all-*` calls return an owned C buffer that the binding copies
into an Eshkol string with the native byte length (so embedded NUL bytes are
preserved) and then frees via `qllm_process_free_buffer`; you do not free it
yourself. `process-read-stdout-bytes` and `process-read-stderr-bytes` return
`(bytes . native-byte-length)` when callers need the exact count. The high-level
`run-command`/`run-argv` wrappers always
`process-destroy` on every exit path (success, timeout-kill, spawn failure), so
they never leak. If you use the low-level API directly, you must call
`process-destroy` yourself — including after a timeout, where the pattern is
`process-kill` → `process-wait 5000` → `process-destroy`.

## Low-level API

| Procedure | Signature |
|-----------|-----------|
| `process-spawn` | `(process-spawn command cwd)` → handle or `#f` |
| `process-spawn-shell` | `(process-spawn-shell command cwd)` |
| `process-spawn-argv` | `(process-spawn-argv argv cwd)` — `argv` is `(prog arg…)` |
| `process-spawn-argv-env` | `(process-spawn-argv-env argv cwd env)` — direct argv with an environment overlay |
| `process-spawn-argv-options` / `process-spawn-argv-with-options` | `(process-spawn-argv-options argv options)` — direct argv with `cwd`, `env`, `stdin`, and `process-group` options |
| `process-write-stdin` | `(process-write-stdin proc data)` |
| `process-close-stdin` | `(process-close-stdin proc)` |
| `process-read-stdout` | `(process-read-stdout proc max-bytes)` |
| `process-read-stderr` | `(process-read-stderr proc max-bytes)` |
| `process-read-all-stdout` | `(process-read-all-stdout proc max-bytes)` |
| `process-read-all-stderr` | `(process-read-all-stderr proc max-bytes)` |
| `process-read-stdout-bytes` / `process-read-stderr-bytes` | `(process-read-*-bytes proc max-bytes)` → `(bytes . native-byte-length)` |
| `process-wait` | `(process-wait proc timeout-ms)` → `0` exited, `1` timed out |
| `process-running?` | `(process-running? proc)` |
| `process-exit-code` | `(process-exit-code proc)` |
| `process-pid` | `(process-pid proc)` — for trace IDs / external observability |
| `process-kill` | `(process-kill proc [signal])` |
| `process-destroy` | `(process-destroy proc)` — **required** |

## Convenience wrappers

| Procedure | Signature | Returns |
|-----------|-----------|---------|
| `run-command` | `(run-command command [cwd] [timeout-ms])` | exit code (int); `-1` spawn fail |
| `run-command-capture` | `(run-command-capture command [cwd] [timeout-ms] [max-output])` | alist `((exit-code . N)(stdout . s)(stderr . s))` |
| `run-argv` | `(run-argv argv [cwd] [timeout-ms])` | exit code |
| `run-argv-capture` | `(run-argv-capture argv [cwd] [timeout-ms] [max-output])` | alist |

Defaults: `cwd` = `"."`, `timeout-ms` = `30000`, `max-output` = `4194304`
(4 MiB). On timeout the child is killed and `run-*-capture` reports exit code
`124` with a `[Process timed out …]` note appended to stderr. The `-capture`
wrappers spawn with stdin wired to `/dev/null` (no unused pipe).

## Argv safety

`run-argv`/`run-argv-capture` and `process-spawn-argv` never invoke a shell, so
metacharacters in arguments are inert. Prefer them over the shell variants for
untrusted input. `process-argv-check-args` rejects format-string-shaped args.

`process-spawn-argv-env` overlays the named string pairs in `env` on the
scrubbed inherited environment; variables not named by the overlay retain
their inherited values. `#f` or an empty environment leaves the inherited
environment unchanged. The options form accepts these keys:

- `cwd`: a directory string or `#f` (default `"."`)
- `env`: the string alist overlay (default `#f`)
- `stdin`: `pipe` or `null` (default `pipe`)
- `process-group`: boolean (default `#f`)

Environment keys and values cannot contain `=`, TAB, or NUL because the
length-preserving FFI wire format uses TAB-separated entries.

## Concurrent spawn bound

All shell and argv spawns share a bounded concurrent-process limit. The
default is 64 process handles; set `ESHKOL_SUBPROC_MAX_CONCURRENT` to a value
from 1 through 4096 to change it. When the bound is reached, spawn returns
`#f` and emits a diagnostic naming the limit. Destroy handles that are no
longer needed so capacity is released. This bounds Eshkol-owned concurrent
children; shell grammar can still ask the shell itself to create pipeline
children, which remains subject to the operating system's process limits.

## Sandbox limits (env)

`process-spawn` honors resource caps from the environment when set:
`ESHKOL_SUBPROC_CPU_SEC`, `ESHKOL_SUBPROC_MEM_MB`, `ESHKOL_SUBPROC_NOFILE`,
`ESHKOL_SUBPROC_NPROC` (see [environment variables](../runtime/environment-variables.md)).

## Capabilities

Subprocess spawning is gated by the `subprocess` / `shell` capabilities when a
policy is active — see [capabilities](capabilities.md).
