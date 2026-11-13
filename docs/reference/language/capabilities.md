# Capability Policy (`core.capabilities`)

Eshkol has a process-local capability policy that gates host-facing operations:
the file surface, the environment, process creation, shell execution, sockets
and foreign-code loading. It is provided by the `core.capabilities` module.

```scheme
(require core.capabilities)
```

## Default behaviour

The default policy is **inactive** (`#f`): no checks are performed and existing
programs run unchanged. Installing an allow-list makes the policy **active**, and
from then on every capability *not* in the list is denied by default.

This is the rule the whole system turns on: **deny inside a policy**. A program
that installs no policy reaches every host operation exactly as it always has.
Once a policy is active, every capability it does not name is refused, and that
holds on all three engines — the LLVM JIT, an ahead-of-time executable, and the
bytecode VM — because they consult one policy store.

**A policy only ever narrows.** Installing one while another is active keeps the
capabilities both name and drops the rest, so a program cannot widen what it was
given; and `(capability-clear-policy!)` is refused while a policy is active,
raising a catchable capability error, because a policy the governed code can
switch off is not a policy. With no policy installed, clearing is a no-op that
succeeds. A host embedding Eshkol that needs to return to the unrestricted
state — after running a guest computation, or between test cases — uses the C
entry point `eshkol_capability_runtime_host_reset()`, which no Eshkol program
can name: declaring any of the policy's own entry points in an `(extern …)` is
refused when the program compiles.

```scheme
(capability-install-policy! '(file-read file-write))
(capability-install-policy! '(file-read network))   ; narrows
(capability-allowed? 'file-read)                    ; => #t
(capability-allowed? 'file-write)                   ; => #f, dropped
(capability-allowed? 'network)                      ; => #f, never held
```

**Granting `subprocess` grants everything the host can do.** The policy lives in
this process's memory, so a child process started with `subprocess` begins with
no policy at all; only a `(fork)` child inherits one. Granting `subprocess` to
code you want to confine is granting it the whole machine by way of a child.

Core hosted capability symbols:

| Symbol | Governs |
|--------|---------|
| `file-read` | reading files |
| `file-write` | writing/creating/deleting files |
| `env-read` | reading environment variables (`get-environment-variable`) |
| `env-write` | setting environment variables |
| `subprocess` | creating a child process |
| `shell` | handing a command line to the host command interpreter |
| `network` | creating a socket: connecting, listening, or sending a request |
| `ffi` | loading foreign code into the process |
| `keychain` | reading and writing host keychain entries |
| `display` | opening a window (`platform.window`) |
| `audio` | opening an audio device (`platform.audio`) |
| `midi` | opening a MIDI port (`platform.midi`) |

### What each capability gates

`(process-spawn-pty cmd)` returns the pair `(pid . master-fd)` on every engine,
or `#f` when it cannot start the child or the host has no pseudo-terminal. The
descriptor belongs to the program: read the child's output through it and close
it with `(socket-close …)`.

`subprocess` — `(fork)`, `(execv path argv)`, `(process-spawn cmd env)` and
every `process-spawn-*` variant including `process-spawn-pty`,
`process-spawn-argv-env` and `process-spawn-argv-options`; the process-control
operations `(process-kill pid sig)`, `(process-kill-tree pid sig)` and
`(process-setpgid pid pgid)`; and the `core.subprocess` handle layer, which is
gated where the child is created rather than only in its Scheme wrappers.

`shell` — `(system cmd)`, and the spawns that hand their command line to the
platform shell: `(process-spawn cmd env)`, `(process-spawn-pty cmd)` and
`core.subprocess`'s shell entry points. A spawn that execs an argv directly
needs `subprocess` alone.

`network` — the socket layer under `core.net` and `core.tls`, so a TCP or UDP
socket, a TLS client and a TLS listener all reach the gate where the socket
comes into existence; `(unix-socket-connect path)`; the HTTP clients, including
the core `(http-request method url headers body timeout)` builtin and the
`core.http` client on every platform backend; `(websocket-connect url headers)`;
and the server listen path, `(http-server-create port)` and the accept that
follows it.

`ffi` — a program's own `(extern … :real symbol)`, which is the construct that
reaches outside the language: on the native engines the declared name is bound
to a gate that asks for `ffi` and then calls the symbol. Modules shipped with
Eshkol are exempt, since their declarations are how the standard library reaches
the runtime entry points it is built on and each of those carries its own gate.
Also `(dlopen path)`, `(dlsym handle name)`, the runtime's own loading of an
optional platform library, and the library loaders whose path an environment
variable can choose (Vulkan, PJRT, cuBLAS).

`keychain` — the host keychain entry points.

## API

| Procedure | Effect |
|-----------|--------|
| `(capability-install-policy! allow-list)` | Activate a policy from a list of capability **symbols**. Errors if the argument is not a list of symbols. While a policy is active this narrows it: the result is the names both hold. |
| `(capability-clear-policy!)` | Deactivate the policy. Refused while one is active (raises); a no-op that returns `#t` when there is none. |
| `(capability-policy)` | Return the current allow-list, or `#f` if inactive. |
| `(capability-policy-active?)` | `#t` iff a policy is currently active. |
| `(capability-allowed? cap)` | `#t` if `cap` is permitted under the current policy (always `#t` when inactive). |
| `(capability-require! cap)` | Return `#t` if allowed; otherwise `raise` an error. |

A refusal at a sink is an ordinary Eshkol error whose message names the
capability and the operation — `capability denied: shell (system)` — so `guard`
and `with-exception-handler` catch it like any other condition, and an
unhandled refusal ends the program the same way an unhandled `(error ...)`
does. A refusal never performs the effect it refused.

```scheme
(require core.capabilities)
(capability-install-policy! '(file-read))
(display (guard (e (#t (error-object-message e)))
           (system "echo reached the shell")))
(newline)
```
```
capability denied: shell (system)
```

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

## Two shapes of refusal

The environment and file surfaces keep the shape they have always had: the
runtime reports the denial on stderr and the call returns a benign value
(`#f` or the empty result), and execution continues, so a program written
against those return values keeps working under a policy. Three more entry
points answer the same way, because their callers already handle the value they
return: the platform's library loading, which answers "nothing loadable"; the
platform's external-program launch, which answers `-1`; and the socket layer
under `core.net` and `core.tls`, which answers the lease outcome
`ESHKOL_IO_DENIED` (`-12`, the symbolic reason `denied`).

The process, shell, socket and foreign-code sinks refuse by raising instead,
because every value they could return — a pid, an exit status, a descriptor, a
handle — is also a legitimate one, and a sentinel would be indistinguishable
from success. Those refusals are catchable, as shown above.

Either way the effect does not happen.
