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
| `ptr` / `char*` / `string` | pointer |
| `...` | variadic marker |
| *(anything else)* | defaults to i64 (with a warning) |

## Tagged-value boundary conversion

Eshkol values are 16-byte tagged values (see the [memory model](../runtime/memory-model.md)).
At the FFI boundary the compiler unboxes/reboxes automatically:

**Arguments (Eshkol → C):**

- integer params ← `unpackInt64FromTaggedValue`
- `double`/`f64` params ← `unpackDoubleFromTaggedValue`
- `ptr` params ← int payload → `IntToPtr` (a **string** is passed as the raw
  pointer to its NUL-terminated payload)
- boolean ← truncated to `i1`

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
   **libcurl** (HTTP), **sqlite3**, **pcre2** (regex), **ncurses** (terminal),
   and OpenSSL/Security frameworks (crypto). If those libraries were not
   available at build time, the macro is empty and calls fall through to
   runtime "unavailable" stubs.

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
