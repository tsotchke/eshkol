# `core.capabilities` — process-local capability policy

**Source**: [`lib/core/capabilities.esk`](../../../lib/core/capabilities.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.capabilities)`

A process-local allow-list gate for hosted/agent surfaces. By default there is **no policy** (`#f`), so all capability checks pass and existing programs run unchanged. Installing an allow-list makes the policy *active*: any capability not in the list is denied. Core hosted capabilities include `file-read`, `file-write`, `env-read`, `env-write`, `subprocess`, `shell`, and `network`. The Scheme layer mirrors the allow-list into the C runtime via `extern` hooks so native operations honor the same policy. Related tasks: `.swarm/tasks/ESH-0076` ("Capability-denied getenv/file ops return #f silently", done) and `.swarm/tasks/ESH-0077` ("CTest for the --emit-object + libeshkol-static.a manual-link AOT path (getenv + capability)").

> Note: the provided symbols use the full `capability-` prefix (`capability-install-policy!`, `capability-clear-policy!`, `capability-policy`, `capability-policy-active?`, `capability-allowed?`, `capability-require!`).

## Functions

### `(capability-install-policy! allow-list)`
Installs `allow-list`, a list of capability symbols, and activates the policy (deny-by-default for anything not listed). Also syncs the list into the C runtime. Raises if `allow-list` is not a list of symbols. Mutating (`!`).

### `(capability-clear-policy!)`
Removes the active policy, restoring the default allow-all behavior, and clears the runtime mirror. Mutating (`!`).

### `(capability-policy)`
Returns the current allow-list when a policy is active, or `#f` when no policy is installed.

### `(capability-policy-active?)`
Returns `#t` if a policy is currently installed, else `#f`.

### `(capability-allowed? capability . context)`
Returns `#t` if `capability` (a symbol) is permitted under the current policy. With no active policy, always `#t`. Extra `context` arguments are accepted and ignored.

### `(capability-require! capability . context)`
Returns `#t` if `capability` is allowed; otherwise raises `"capability denied"`. Use it to gate an operation.

```scheme
;; capabilities.esk
(require stdlib)
(display (capability-policy-active?)) (newline)      ; no policy yet
(display (capability-policy)) (newline)
(display (capability-allowed? 'file-read)) (newline)  ; allow-all by default
(capability-install-policy! '(file-read network))
(display (capability-policy-active?)) (newline)
(display (capability-policy)) (newline)
(display (capability-allowed? 'file-read)) (newline)  ; listed
(display (capability-allowed? 'file-write)) (newline) ; not listed -> denied
(display (capability-require! 'network)) (newline)    ; allowed -> #t
(capability-clear-policy!)
(display (capability-policy-active?)) (newline)
(display (capability-allowed? 'file-write)) (newline) ; allow-all again
```
```
#f
#f
#t
#t
(file-read network)
#t
#f
#t
#f
#t
```

Requiring a denied capability raises:

```scheme
(require stdlib)
(capability-install-policy! '(file-read))
(display (capability-require! 'file-write)) (newline)
```
```
Unhandled exception: capability denied
```

Edge cases: installing an empty allow-list `'()` still activates the policy (and, since the list is empty, denies everything). `capability-install-policy!` requires every element to be a symbol; passing a non-symbol or non-list raises `"capability-install-policy!: expected a list of symbols"`. The policy is process-local state and persists across calls until cleared.
