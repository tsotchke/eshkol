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

A process handle is an opaque pointer. **You own it until you call
`process-destroy`.** The lifecycle is:

1. spawn → handle
2. drive (`process-write-stdin`, `process-read-*`, `process-wait`)
3. `process-destroy` — frees the handle and reaps the child

The `process-read-all-*` calls return an owned C buffer that the binding copies
into an Eshkol string and then frees via `qllm_process_free_buffer`; you do not
free it yourself. The high-level `run-command`/`run-argv` wrappers always
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
| `process-write-stdin` | `(process-write-stdin proc data)` |
| `process-close-stdin` | `(process-close-stdin proc)` |
| `process-read-stdout` | `(process-read-stdout proc max-bytes)` |
| `process-read-stderr` | `(process-read-stderr proc max-bytes)` |
| `process-read-all-stdout` | `(process-read-all-stdout proc max-bytes)` |
| `process-read-all-stderr` | `(process-read-all-stderr proc max-bytes)` |
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

## Sandbox limits (env)

`process-spawn` honors resource caps from the environment when set:
`ESHKOL_SUBPROC_CPU_SEC`, `ESHKOL_SUBPROC_MEM_MB`, `ESHKOL_SUBPROC_NOFILE`,
`ESHKOL_SUBPROC_NPROC` (see [environment variables](../runtime/environment-variables.md)).

## Capabilities

Subprocess spawning is gated by the `subprocess` / `shell` capabilities when a
policy is active — see [capabilities](capabilities.md).
