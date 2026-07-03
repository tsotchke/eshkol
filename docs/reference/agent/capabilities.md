# `core.capabilities` — Capability Policy

A process-local allow-list that gates hosted/agent side effects. **Off by
default**: with no policy installed, every operation is allowed and existing
v1.2 programs run unchanged. Once you install a policy, it is *deny-by-default* —
only the capabilities you list are permitted.

```scheme
(require core.capabilities)
```

Source: `lib/core/capabilities.esk`. Runtime symbols:
`eshkol_capability_runtime_clear`, `eshkol_capability_runtime_begin_install`,
`eshkol_capability_runtime_allow`.

## API

| Procedure | Signature | Description |
|-----------|-----------|-------------|
| `capability-install-policy!` | `(capability-install-policy! allow-list)` | Install an allow-list of capability **symbols**; activates deny-by-default. `error`s unless given a list of symbols |
| `capability-clear-policy!` | `(capability-clear-policy!)` | Remove the policy; back to allow-all |
| `capability-policy` | `(capability-policy)` | The active allow-list, or `#f` if none |
| `capability-policy-active?` | `(capability-policy-active?)` | `#t`/`#f` |
| `capability-allowed?` | `(capability-allowed? cap . context)` | `#t` if `cap` permitted under current policy |
| `capability-require!` | `(capability-require! cap . context)` | Assert `cap`; raises if denied |

Installing a policy mirrors the allow-list into the C runtime
(`capability-sync-runtime!` → `…begin-install` then one `…allow` per symbol) so
that native-side checks see the same policy.

## Core capability symbols

`file-read`, `file-write`, `env-read`, `env-write`, `subprocess`, `shell`,
`network`.

## Which agent operations check which capability

| Operation | Capability required (when a policy is active) |
|-----------|-----------------------------------------------|
| `agent.http` requests (`http-get`/`http-post`/`http-request`/`http-stream-open`) | `network` |
| `agent.subprocess` argv spawns (`process-spawn-argv`, `run-argv…`) | `subprocess` |
| `agent.subprocess` shell spawns (`process-spawn-shell`, `run-command…`) | `shell` |
| `agent.git-ffi` (shells out to `git`) | `shell` / `subprocess` |
| File I/O builtins | `file-read` / `file-write` |
| Environment access | `env-read` / `env-write` |

The HTTP client calls `(capability-require! 'network url)` on every request, so
denying `network` blocks outbound HTTP even if the module is loaded.

## Example

```scheme
(require core.capabilities)
(require agent.http)

;; Allow only network; deny subprocess/shell/file-write.
(capability-install-policy! '(network))

(capability-allowed? 'network)     ;; => #t
(capability-allowed? 'shell)       ;; => #f
;; (run-command "rm -rf /")        ;; would raise: shell capability denied

(capability-clear-policy!)         ;; back to allow-all
```
