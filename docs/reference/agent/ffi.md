# Foreign Function Interface (FFI)

Eshkol calls C functions through the `extern` declaration. This is the mechanism
every `agent.*` module uses to bind to its C runtime (`qllm_*`, `eshkol_*`).

## `extern` syntax

```
(extern <ret-type> <eshkol-name> <arg-type>* [<modifier>]*)
```

- **Return type** (required symbol) comes first.
- **Eshkol-visible name** (required symbol) comes second — this is the name your
  code calls.
- Zero or more **parameter type** symbols follow, until `)` or a `:`-modifier.
- Trailing `:`-modifiers must come last.

Parsed at `lib/frontend/parser.cpp` (extern form ~8480-8565; modifier tail
~1590-1645).

### Modifiers

| Modifier | Effect |
|----------|--------|
| `:real <sym>` (alias `:extern-symbol <sym>`) | The actual C symbol to link. If omitted, the Eshkol name is used verbatim as the C symbol. |
| `:weak` | Weak linkage. |
| `:no-return` | Function does not return. |

There is a sibling form `(extern-var …)` for external global variables (accepts
only `:real` / `:extern-symbol`).

### Example

```scheme
(extern i32  sha256-raw  ptr i64 ptr i64 :real eshkol_sha256)
(extern ptr  qllm-http-get  ptr i32     :real qllm_http_get)
(extern void process-destroy-raw  ptr   :real qllm_process_destroy)
```

## Type keywords → C ABI

`mapStringToType` (`lib/backend/llvm_codegen.cpp` ~23089-23108), matched
case-sensitively:

| Keyword | C / LLVM type |
|---------|---------------|
| `void` | void |
| `i32` / `int` | i32 |
| `i64` / `long` | i64 |
| `f32` / `float` | float (f32) |
| `f64` / `double` | double (f64) |
| `ptr` / `void*` / `char*` / `string` | pointer |
| `...` | variadic marker |
| *(anything else)* | defaults to i64 (with a warning) |

## Tagged-value boundary conversion

Eshkol values are 16-byte tagged values (see the [memory model](../runtime/memory-model.md)).
At the FFI boundary the compiler unboxes/reboxes automatically:

**Arguments (Eshkol → C):**

- integer params ← `unpackInt64FromTaggedValue`
- `double`/`f64` params ← `unpackDoubleFromTaggedValue`
- `ptr` params ← int payload → `IntToPtr` (a **string** is passed as the raw
  pointer to its NUL-terminated payload), **after** the pointer-argument type
  guard below
- boolean ← truncated to `i1`

### Pointer arguments are type-checked at the boundary

The `ptr` conversion above reinterprets the tagged value's 64-bit payload as an
address. On its own that means a **number** passed where a `ptr` belongs becomes
a pointer literally equal to that number, and the C callee faults on it — a
misplaced `5000` produced `SIGSEGV at address 0x1388` with no diagnostic. As of
v1.3.4, codegen emits a check ahead of that conversion for every parameter an
`extern` declared `ptr` / `string` / `char*`, and a value that provably cannot be
an address raises a **catchable** type error naming the extern, the argument
position and the declared type:

```
FFI type error in process-spawn-argv-flags-raw (C symbol qllm_process_spawn_argv_flags):
argument 2 is declared `ptr` and requires a string or pointer handle,
but got the integer 5000
```

It is an `ESHKOL_EXCEPTION_TYPE_ERROR`, so `guard` / `with-exception-handler`
can intercept it; uncaught, it exits nonzero instead of faulting.

What is **rejected**: numbers, characters, symbols, `#t`, dual numbers, complex
numbers, logic variables — the immediate tags that cannot denote memory.

What is **accepted**: strings, bytevectors and every other heap object; opaque
handles returned by a `ptr`-returning extern; callables; `'()`; and `#f`, which
is how Eshkol spells a NULL pointer argument (`(process-spawn-raw command cwd #f
0)` passes a NULL `envp`).

The check is a denylist of immediate tags rather than an allowlist of pointer
tags on purpose: Eshkol's type byte also carries the multimedia resource tags,
the deprecated pointer aliases and the port flag bits, so an allowlist that
missed one would reject a legitimate pointer. The tradeoff is that an exotic tag
encoding may slip through unchecked — never that a working call breaks.

Native link-time fingerprint — **COMPLETE**. The fingerprint covers the active
object ABI's header size, subtype offset, and payload alignment. `--freestanding`
objects remain outside that link check because their contract has no external
runtime dependency.

WASM geometry guard — **COMPLETE**. A generated `wasm32` entry point calls the
`eshkol_wasm_abi_check` import before user code with the pointer width, active
object-header size/alignment and field offsets, plus the 16-byte tagged-value
size/alignment and type/flags/reserved/data/padding offsets. The two published
JS glue files (`web/eshkol-repl.js` and `site/static/eshkol-runtime.js`) verify
the complete tuple and throw a diagnostic on any mismatch; the WASM lite lane
therefore fails loudly instead of interpreting a changed layout. The current
glue implements the active v1 geometry and rejects a v2 module until its JS
layout is deliberately migrated.

Guards emitted at the boundary do not replace validating your own wrapper's
parameters: the boundary error can only name the `-raw` extern and an argument
*position*, whereas a check in the wrapper names the parameter the caller
actually got wrong. `lib/agent/subprocess.esk` does both — see
`process-check-cwd!` there.

**Return values (C → Eshkol):**

| C return | Repacked as |
|----------|-------------|
| `i32` | `SIToFP` → tagged **double** (so error codes compare with `= < >`) |
| `i64` | tagged **int** (used for opaque handles) |
| `f32`/`f64` | tagged **double** |
| `ptr` | tagged `HEAP_PTR` |

Because a returned `ptr` is an opaque handle, to turn an FFI-returned `char*` into
an Eshkol string you call `(ptr->string p)`, which allocates a header'd Eshkol
string and copies the bytes. Modules that return C-owned buffers wrap this
(e.g. `process-owned-c-string->string` in `agent.subprocess`, which copies then
frees the C buffer).

> Because `i32` returns are repacked as doubles, an FFI status code of `-1`
> compares correctly with numeric predicates; do not assume it stays an exact
> integer.

## Requiring agent FFI and AOT linking

Two questions decide whether a compiled binary links the agent C runtime:

1. **Does the program use agent FFI?** `requires_agent_ffi` (in
   `exe/eshkol-run.cpp`) scans top-level `(require …)` forms. Any module whose
   name starts with `agent.` → yes. Otherwise it resolves each required module to
   a file and **recurses transitively**: a program that only does
   `(require core.memory)` still links agent FFI because `core.memory` internally
   `(require agent.crypto)`. (The scan is text-based and conservative — false
   positives only cost a few extra linked libs.)

2. **What gets linked?** When agent FFI is needed, the compiler splices
   `ESHKOL_HOST_AGENT_FFI_LINK_ARGS` into the link command. That macro
   (`cmake/build_config.h.in`, assembled in `CMakeLists.txt`) force-loads
   `libeshkol-agent-ffi.a` (`-Wl,-force_load,…` on Apple; `--whole-archive`
   elsewhere) and appends the C dependencies that were found at configure time:
   **libcurl** (HTTP), **sqlite3**, **pcre2** (regex), **ncurses** (terminal).
   If those libraries were not available at build time, the macro is empty and
   calls fall through to runtime "unavailable" stubs.

   **Crypto is the one exception** (2026-07 update): `hmac-sha256` / `sha256` /
   `random-bytes` / `random-hex` (`lib/agent/crypto.esk`) are backed by
   `lib/core/crypto_primitives.c`, which lives in the *core* `eshkol-runtime`
   archive, not `eshkol-agent-ffi`. They are linked into **every** AOT/`-r`
   binary unconditionally — independent of `ESHKOL_BUILD_AGENT_FFI` and
   independent of whether `requires_agent_ffi`'s `(require agent.…)` scan
   fires. This was a deliberate fix (Noesis bug report #2, 2026-07-04): these
   four symbols previously lived in the much larger, more frequently rebuilt
   `eshkol-agent-ffi` archive and were only linked in when the scan detected
   `agent.crypto` usage (directly or transitively, e.g. via `core.memory`). A
   standalone `eshkol-run -r script.esk` AOT link that raced a *concurrent*
   rebuild of `libeshkol-agent-ffi.a` (e.g. another terminal running
   `cmake --build`) could see a partially-written archive and fail with
   `Undefined symbols: _eshkol_hmac_sha256` etc — indistinguishable from a
   real missing-symbol bug from the caller's side, even though the finished
   archive had the symbols all along. Moving crypto to the always-linked core
   runtime removes the race entirely rather than papering over it.
   See `lib/core/crypto_primitives.c` and the "Crypto system libs" comments in
   `CMakeLists.txt` / `exe/eshkol-run.cpp` / `lib/backend/llvm_codegen.cpp`.

   As a secondary mitigation, if an AOT link still fails (any archive, any
   symbol), `eshkol-run` now checks whether a `.a` on the link line was
   modified in the last 30 seconds and — in addition to the linker's own
   error — prints a note suggesting a concurrent rebuild may be in progress
   and to retry once it finishes.

### JIT (`-r`) vs AOT resolution

- Under **`-r` / `-e`**, the JIT host (`eshkol-run` / `eshkol-repl`) is *itself*
  linked against `libeshkol-agent-ffi.a` and force-references every agent symbol
  via weak declarations (`ESHKOL_AGENT_FFI_SYMBOL`, `lib/repl/repl_jit.cpp`).
  The ORC JIT resolves `(extern …)` calls to these in-process symbols with
  `dlsym`. A stripped host simply leaves them null.
- Under **AOT**, the produced binary must contain the archive + deps, which is
  exactly what the `ESHKOL_HOST_AGENT_FFI_LINK_ARGS` splice provides.

Practical implication: always verify agent-FFI code under **AOT**, not just
`-r` — the JIT can resolve a symbol the AOT link would miss if the link-args
scan didn't fire.

## FFI marshalling gotchas

Established behaviors worth knowing before you hit them the hard way. None of
these are bugs in the sense of "will be changed" — they're documented here so
they're an informed choice instead of a debugging session.

### `fwrite`/byte-oriented C calls want BYTES, not codepoints

`string-length` counts **codepoints**, not bytes. A string containing
multibyte UTF-8 (e.g. an em-dash `—`, 3 bytes / 1 codepoint) will silently
truncate if you pass `string-length` to a byte-counted C call:

```scheme
;; WRONG on multibyte input — writes only (string-length s) BYTES, which for
;; a string containing multibyte UTF-8 is fewer bytes than the string has,
;; silently dropping the tail:
(fwrite s 1 (string-length s) f)

;; RIGHT — string-byte-length reads the string header's byte-size field
;; directly (no UTF-8 decoding), matching what `fwrite` expects:
(fwrite s 1 (string-byte-length s) f)
```

`(string-byte-length "café")` is `5` (the `é` is 2 bytes in UTF-8);
`(string-length "café")` is `4`. This bit a real deployment: `core.memory`'s
durable append-only log (`lib/core/memory_store.esk`) used `string-length`
with `fwrite` and lost the tail of any line containing multibyte UTF-8 — see
`ms-append-fsync!` there for the fixed call site.

### C `NULL` arrives in Scheme as `()`, not `0` or `#f`

An FFI function returning `ptr`/`char*` that yields `NULL` (e.g. `fopen` on a
missing path, `getenv` on an unset variable) marshals to Eshkol's empty list
`'()`, not the integer `0` and not `#f`. Test with `(null? x)`, not `(= x 0)`
or `(if x …)` (the latter also happens to work since `'()` is truthy in
Eshkol's `if`, unlike `#f` — one more reason `(null? x)` is the only
unambiguous check). This is the official convention; every FFI wrapper in
`lib/agent/*.esk` and `lib/core/memory_store.esk` (`null-ptr?`) follows it.

### Every raw `ptr`/`char*` FFI return is an opaque handle — convert explicitly

Per the "Return values" table above, an `extern` declared to return `ptr` (or
`char*`/`string` — all three map to the same LLVM pointer type, see "Type
keywords → C ABI") comes back as a tagged `HEAP_PTR` handle, not an
auto-marshalled Eshkol string. `(display p)` on it prints an opaque
`#<heap:N>`, not the C string's contents — you must call `(ptr->string p)`
explicitly to get a real Eshkol string. Verified consistent across both
`-r`/JIT and AOT-compiled execution as of 2026-07 (a field report from a
Noesis build flagged this as differing between the interpreter and the
compiled runtime for `getenv`; that specific divergence did not reproduce in
isolation on the current codebase, but since the two execution paths
*can* differ in principle, don't rely on either behavior implicitly — always
convert explicitly with `(ptr->string p)` and check `(null? p)` first):

```scheme
(extern ptr c-getenv ptr :real getenv)
(define p (c-getenv "PATH"))
(if (null? p)
    (display "not set")
    (display (ptr->string p)))   ; never rely on (display p) alone
```

When the native API supplies an explicit byte length, use
`(ptr->string-n p length)` instead. It copies exactly `length` bytes and
therefore preserves embedded NUL bytes; `ptr->string` intentionally uses
`strlen` and is only correct for NUL-terminated C text. A negative length
raises an Eshkol error, while a null pointer or zero length produces the empty
string.

### Variadic `extern`s corrupt arguments on arm64 — fixed-arity only

Declaring a variadic C function directly, e.g. `(extern void printf char*
...)`, is unsafe on arm64: the AAPCS64 calling convention passes/spills
named vs. variadic arguments differently (variadic args go through the
stack-based register save area), but Eshkol's codegen emits a call as if the
callee were an ordinary fixed-arity function. Arguments after the fixed
prefix arrive corrupted. This was hit for real: `open(path, mode)`'s libc
signature is actually variadic (`open(const char*, int, ...)` for the
optional file-creation `mode`), and declaring it directly made `mode` arrive
as garbage, creating unreadable files — see the header comment in
`lib/core/memory_store.esk`, which works around it by declaring only
`fopen`/`fwrite`/`fflush`/`fileno`/`fsync`/`fclose` (all genuinely
fixed-arity in libc) and avoiding `open` entirely.

**Only fixed-arity `extern`s are portable.** If you need a variadic C
function, wrap it behind a small fixed-arity C shim (a real function taking
concrete arguments that internally calls the variadic one) and declare
*that* as the `extern` instead. As of 2026-07, declaring a variadic `extern`
(any parameter list ending in `...`) emits a compile-time WARNING pointing
at this section, so the risk is surfaced at the declaration site rather than
discovered via corrupted data at runtime:

```
WARNING: extern 'my-fn' (real: my-fn) is declared variadic ('...'): variadic
externs are unsafe on arm64 (argument corruption past the fixed prefix — see
lib/core/memory_store.esk header comment and docs/reference/agent/ffi.md).
Only fixed-arity externs are portable; wrap this function behind a
fixed-arity C shim instead.
```

## `agent.subprocess` arities — `cwd` is positional and required

The spawn family's signatures are fixed-arity with `cwd` in the **second**
position. There is no optional tail and no default: omitting `cwd` is an arity
error, and passing a timeout in its place is a type error. Both are reported now;
before v1.3.4 the first bound the result to null and kept running, and the second
dereferenced the timeout as an address.

| Procedure | Arity | Positions |
|-----------|-------|-----------|
| `process-spawn` | exactly 2 | `command` `cwd` |
| `process-spawn-shell` | exactly 2 | `command` `cwd` |
| `process-spawn-argv` | exactly 2 | `argv` `cwd` |
| `process-wait` | exactly 2 | `handle` `timeout-ms` |
| `process-write-stdin` | exactly 2 | `handle` `data` |
| `process-read-stdout` / `process-read-stderr` | exactly 2 | `handle` `max-bytes` |
| `process-read-all-stdout` / `process-read-all-stderr` | exactly 2 | `handle` `max-bytes` |
| `process-running?` / `process-exit-code` / `process-pid` / `process-close-stdin` / `process-destroy` | exactly 1 | `handle` |
| `process-kill` | 1 or 2 | `handle` `[signal]` (default 15) |
| `run-command` | 1 to 3 | `command` `[cwd]` `[timeout-ms]` |
| `run-command-capture` | 1 to 4 | `command` `[cwd]` `[timeout-ms]` `[max-output]` |
| `run-argv` | 1 to 3 | `argv` `[cwd]` `[timeout-ms]` |
| `run-argv-capture` | 1 to 4 | `argv` `[cwd]` `[timeout-ms]` `[max-output]` |

`cwd` accepts a directory path string, or `#f` to inherit the caller's working
directory. Defaults for the bracketed optionals: `cwd` `"."`, `timeout-ms`
`30000`, `max-output` `4194304` bytes per stream.

The `run-*` convenience wrappers take their optionals **positionally**, so the
timeout is always the THIRD argument:

```scheme
;; WRONG — 5000 lands in cwd. Raises:
;;   run-argv-capture: `cwd` must be a string path or #f, but got the number 5000.
(run-argv-capture (list "/bin/echo" "hi") 5000)

;; RIGHT — cwd explicit, timeout third
(run-argv-capture (list "/bin/echo" "hi") "." 5000)
```

And the fixed-arity spawns require `cwd`:

```scheme
;; WRONG — arity error at compile time:
;;   Arity mismatch: process-spawn-argv expects 2 arguments but got 1
(define h (process-spawn-argv (list "echo" "hi")))

;; RIGHT
(define h (process-spawn-argv (list "echo" "hi") "."))
```

An arity mismatch is a **compile-time** failure: no binary is written under
`-o`, and `-r` exits nonzero without running the program.
