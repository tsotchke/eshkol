# `core.capabilities` — Capability Policy

A process-local allow-list that gates hosted/agent side effects. **Off by
default**: with no policy installed, every operation is allowed and existing
v1.2 programs run unchanged. Once you install a policy, it is *deny-by-default* —
only the capabilities you list are permitted.

```scheme
(require core.capabilities)
```

Source: `lib/core/capabilities.esk`, over the runtime policy store in
`lib/core/capability_policy.cpp` — one store, linked by the native runtime, the
bytecode VM and the VM's WebAssembly bundle, so a policy a program installs
holds whichever engine runs it. Runtime symbols:
`eshkol_capability_runtime_clear`, `eshkol_capability_runtime_begin_install`,
`eshkol_capability_runtime_allow`, `eshkol_capability_require` (the
allow-or-record guard) and `eshkol_capability_enforce` (the guard that raises).

## API

| Procedure | Signature | Description |
|-----------|-----------|-------------|
| `capability-install-policy!` | `(capability-install-policy! allow-list)` | Install an allow-list of capability **symbols**; activates deny-by-default. While a policy is active this narrows it to the names both hold. `error`s unless given a list of symbols |
| `capability-clear-policy!` | `(capability-clear-policy!)` | Remove the policy. Refused while one is active — raises; a no-op returning `#t` when there is none |
| `capability-policy` | `(capability-policy)` | The active allow-list, or `#f` if none |
| `capability-policy-active?` | `(capability-policy-active?)` | `#t`/`#f` |
| `capability-allowed?` | `(capability-allowed? cap . context)` | `#t` if `cap` permitted under current policy |
| `capability-require!` | `(capability-require! cap . context)` | Assert `cap`; raises if denied |

Installing a policy mirrors the allow-list into the C runtime
(`capability-sync-runtime!` → `…begin-install` then one `…allow` per symbol) so
that native-side checks see the same policy.

## Core capability symbols

`file-read`, `file-write`, `env-read`, `env-write`, `subprocess`, `shell`,
`network`, `ffi`, `keychain`, `display`, `audio`, `midi`.

See [the language reference page](../language/capabilities.md) for what each one
gates, operation by operation.

## Which agent operations check which capability

| Operation | Capability required (when a policy is active) |
|-----------|-----------------------------------------------|
| `agent.http` requests (`http-get`/`http-post`/`http-request`/`http-stream-open`) | `network` |
| `agent.subprocess` argv spawns (`process-spawn-argv`, `run-argv…`) | `subprocess` |
| `agent.subprocess` shell spawns (`process-spawn-shell`, `run-command…`) | `shell` |
| `agent.git-ffi` (shells out to `git`) | `shell` / `subprocess` |
| File I/O builtins | `file-read` / `file-write` |
| Environment access | `env-read` / `env-write` |
| `(dlopen path)`, `(dlsym handle name)`, optional platform libraries | `ffi` |
| `agent.keychain` | `keychain` |

The HTTP client calls `(capability-require! 'network url)` on every request, so
denying `network` blocks outbound HTTP even if the module is loaded. The gate is
also in the runtime, at the point where each client opens its connection and
where a socket comes into existence, so a caller that reaches a runtime entry
point without going through the Scheme wrapper meets the same policy.

A refused process, shell or foreign-code operation raises a catchable error
naming the capability and the operation, e.g.
`capability denied: subprocess (process-spawn)`. The platform's library loading
and external-program launch, and the socket layer under `core.net`/`core.tls`,
report instead and return their own "unavailable" value — see
[the language reference page](../language/capabilities.md). Either way the
effect does not happen.

A policy only narrows: installing one while another is active keeps the names
both hold, and clearing is refused while a policy is active. Note that granting
`subprocess` grants the machine, since a child process starts with no policy of
its own.

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
