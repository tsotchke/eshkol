# Capability Policy (`core.capabilities`)

Eshkol has a process-local capability policy that gates host-facing operations
(file, environment, subprocess, shell, network). It is provided by the
`core.capabilities` module.

```scheme
(require core.capabilities)
```

## Default behaviour

The default policy is **inactive** (`#f`): no checks are performed and existing
programs run unchanged. Installing an allow-list makes the policy **active**, and
from then on every capability *not* in the list is denied by default.

Core hosted capability symbols:

| Symbol | Governs |
|--------|---------|
| `file-read` | reading files |
| `file-write` | writing/creating/deleting files |
| `env-read` | reading environment variables (`get-environment-variable`) |
| `env-write` | setting environment variables |
| `subprocess` | spawning subprocesses |
| `shell` | shell execution |
| `network` | network access |

## API

| Procedure | Effect |
|-----------|--------|
| `(capability-install-policy! allow-list)` | Activate a policy from a list of capability **symbols**. Errors if the argument is not a list of symbols. |
| `(capability-clear-policy!)` | Deactivate the policy; checks allow by default again. |
| `(capability-policy)` | Return the current allow-list, or `#f` if inactive. |
| `(capability-policy-active?)` | `#t` iff a policy is currently active. |
| `(capability-allowed? cap)` | `#t` if `cap` is permitted under the current policy (always `#t` when inactive). |
| `(capability-require! cap)` | Return `#t` if allowed; otherwise `raise` an error. |

## Example

```scheme
(require core.capabilities)
(display (capability-policy-active?)) (newline)   ; #f initially
(capability-install-policy! '(file-read env-read))
(display (capability-policy-active?)) (newline)   ; now active
(display (capability-allowed? 'file-read)) (newline)
(display (capability-allowed? 'network)) (newline)
(capability-clear-policy!)
(display (capability-allowed? 'network)) (newline)  ; allowed again
```
```
#f
#t
#t
#f
#t
```

## The `getenv` / `env-read` story

`get-environment-variable` (alias `getenv`) is gated on the `env-read` capability.
The gate is enforced in the runtime for both the JIT and AOT paths.

With no policy, it reads normally:
```scheme
(display (get-environment-variable "HOME")) (newline)   ; => the value
```

Under an **active** policy that does not grant `env-read`, the read is denied:
the runtime prints a diagnostic to stderr and the call returns `#f`; execution
continues (it is not a catchable exception).
```scheme
(require core.capabilities)
(capability-install-policy! '(file-read))   ; env-read NOT granted
(display (get-environment-variable "HOME")) (newline)
(display "after") (newline)
```
```
capability denied: env-read
#f
after
```

Granting `env-read` restores access:
```scheme
(require core.capabilities)
(capability-install-policy! '(env-read))
(display (get-environment-variable "HOME")) (newline)   ; => the value
```

File operations (`file-read`/`file-write`), subprocess, shell, and network entry
points follow the same pattern: when a policy is active and the relevant
capability is absent, the operation is denied and returns a benign value rather
than performing the effect.
