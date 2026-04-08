# Eshkol Developer Tools Reference

**Status:** Production (v1.1.12)
**Applies to:** Eshkol compiler v1.1-accelerate and later

---

## Overview

Eshkol ships two primary developer tooling components: a Language Server Protocol (LSP) server (`eshkol-lsp`) and a Visual Studio Code extension. Together they provide syntax highlighting, live diagnostics, code completion, hover documentation, and go-to-definition navigation without any external runtime dependencies. The compiler itself exposes debug flags for inspecting internal representations at each compilation stage.

This document covers the full architecture, feature set, configuration options, and debugging workflows for each component.

---

## 1. LSP Server

### 1.1 Source and Build

| Property | Value |
|----------|-------|
| Source file | `tools/lsp/eshkol_lsp.cpp` (1018 lines) |
| Binary name | `eshkol-lsp` |
| Version | `1.1.12` |
| License | MIT |
| External dependencies | None (JSON implemented inline) |

The server is a self-contained C++ binary. It depends only on the Eshkol parser library (via `eshkol/eshkol.h`) for real parse-error extraction and links against no third-party JSON library.

### 1.2 Transport Layer

The server uses **JSON-RPC 2.0 over stdin/stdout** with the standard `Content-Length:` header framing defined by the LSP specification:

```
Content-Length: <byte_count>\r\n
\r\n
<JSON body>
```

The `JsonRpcTransport` class (line ~297) handles all framing. Reading blocks on `std::cin.read()` after parsing the `Content-Length` header. Responses and notifications are written to `std::cout` with an immediate `flush()`. There is no buffering of outbound messages.

The internal JSON implementation (`JsonValue` / `JsonParser`, lines 36–291) is a minimal recursive-descent parser supporting all JSON types including nested objects, arrays, strings with Unicode escapes (`\uXXXX`), and numbers. It has no external dependency and is intentionally lightweight — its sole purpose is to parse and serialize LSP protocol messages.

### 1.3 Server Lifecycle

The `EshkolLanguageServer` class runs a single-threaded event loop in `run()`:

```cpp
void run() {
    while (!shutdown_requested_) {
        std::string body = transport_.read_message();
        if (body.empty()) {
            if (std::cin.eof()) break;
            continue;
        }
        handle_message(body);
    }
}
```

Message dispatch (lines 521–539) is a plain string-match chain:

| Incoming Method | Handler |
|-----------------|---------|
| `initialize` | `handle_initialize` — sends capabilities, sets `initialized_ = true` |
| `initialized` | no-op |
| `shutdown` | `handle_shutdown` — sends null response, sets `shutdown_requested_ = true` |
| `exit` | `handle_exit` — calls `std::exit(0)` if shutdown was clean, `exit(1)` otherwise |
| `textDocument/didOpen` | `handle_did_open` — stores document, runs diagnostics |
| `textDocument/didChange` | `handle_did_change` — updates document content, runs diagnostics |
| `textDocument/didClose` | `handle_did_close` — removes document, clears diagnostics |
| `textDocument/completion` | `handle_completion` |
| `textDocument/hover` | `handle_hover` |
| `textDocument/definition` | `handle_definition` |
| any other method with `id` | sends JSON-RPC error -32601 (Method not found) |

### 1.4 Server Capabilities

Advertised in the `initialize` response:

```json
{
  "capabilities": {
    "textDocumentSync": 1,
    "completionProvider": {
      "triggerCharacters": ["(", " "]
    },
    "hoverProvider": true,
    "definitionProvider": true
  },
  "serverInfo": {
    "name": "eshkol-lsp",
    "version": "1.1.12"
  }
}
```

`textDocumentSync: 1` means **Full sync** — on every `didChange` the client sends the entire document text, not incremental deltas. The server stores documents in a `DocumentStore` (an `unordered_map<string, TextDocument>`) keyed by URI.

### 1.5 Document Store

```cpp
struct TextDocument {
    std::string uri;
    std::string content;
    int version;
};
```

The `DocumentStore` class (lines 369–393) provides `open`, `update`, `close`, and `get` operations. Documents are stored by URI string. The version field mirrors the client's version counter but is not used for any conflict detection — the last received content on `didChange` is taken as authoritative (full sync semantics).

---

## 2. LSP Feature Details

### 2.1 Diagnostics

**Handler:** `publish_diagnostics` (line 621)
**Trigger:** `didOpen`, `didChange`, `didClose`

Diagnostics are published as a `textDocument/publishDiagnostics` notification after every document change. The diagnostic pipeline has two stages:

**Stage 1 — Parser integration.** The document content is streamed into `eshkol_parse_next_ast_from_stream()` in a loop. Any C++ exception thrown by the parser is caught and silently discarded; the paren-balance check below provides the user-visible error in that case.

**Stage 2 — Parenthesis balance check.** A single-pass scanner over the document tracks:
- `in_comment` — set by `;`, cleared by `\n`
- `in_string` — set/cleared by `"`, respects `\\` escapes
- `paren_depth` — incremented by `(`, decremented by `)`

Two error conditions are reported:

| Condition | Severity | Message |
|-----------|----------|---------|
| `paren_depth < 0` at any `)` | Error (severity 1) | `"Unmatched closing parenthesis"` |
| `paren_depth > 0` at end of file | Error (severity 1) | `"N unclosed parenthesis(es)"` |

The diagnostic `source` field is always `"eshkol"`. When a file is closed (`didClose`), an empty diagnostics array is sent to clear any previously reported errors from the client's UI.

### 2.2 Completion

**Handler:** `handle_completion` (line 721)
**Trigger:** typing `(` or space (configured in `triggerCharacters`)

The completion handler extracts the word being typed at the cursor using `get_word_at()` (line 879). This function uses Scheme-aware identifier boundaries — all characters are valid except `(`, `)`, `"`, `'`, whitespace, `;`, `,`, `` ` ``, and `#`.

Three completion item sources are merged:

**Keywords (LSP kind 14 = Keyword)** — 33 special forms and syntax keywords:

```
define   lambda    if       cond       else      case
let      let*      letrec   begin      do
and      or        not      when       unless
set!     quote     quasiquote  unquote  unquote-splicing
define-syntax  syntax-rules  let-syntax  letrec-syntax
import   export    library    require    provide
define-record-type  define-type
with-region  owned  move  borrow  shared  weak-ref
guard    raise  with-exception-handler
values   call-with-values  call-with-current-continuation
dynamic-wind  parameterize  make-parameter
```

**Builtins (LSP kind 3 = Function)** — approximately 80 built-in functions spanning arithmetic, predicates, pairs/lists, strings, vectors, I/O, tensor operations, autodiff, parallel primitives, and system calls. The complete list is defined in the `builtins()` static method (lines 437–488).

**Document-local defines (LSP kind 3 or 6)** — extracted from the open document by `extract_defines()` (line 912). Function definitions (`(define (name ...)`) get kind 3 (Function); variable definitions (`(define name ...)`) get kind 6 (Variable), unless the variable's value starts with `(lambda`, in which case it is also classified as kind 3.

Filtering is prefix-substring: if the typed prefix is non-empty, only items where the label contains the prefix as a substring are included. The response always sets `isIncomplete: false`.

For items with inline documentation, the `documentation` field is populated from `get_doc()` (see hover section below). All other items omit the field.

### 2.3 Hover

**Handler:** `handle_hover` (line 783)

The server extracts the word under the cursor using the same `get_word_at()` logic as completion, then looks up documentation in a three-tier cascade:

1. **`get_doc()` map** — 16 entries with full markdown documentation strings for: `define`, `lambda`, `if`, `let`, `cond`, `begin`, `cons`, `car`, `cdr`, `map`, `filter`, `fold`, `tensor`, `tensor-dot`, `gradient`, `parallel-map`, `owned`, `move`, `borrow`, `shared`, `weak-ref`, `with-region`, `eval`.

2. **Keyword/builtin fallback** — if the word is in the `keywords()` or `builtins()` lists but has no doc entry, the hover content is generated as:
   - `**word** — Eshkol special form`
   - `**word** — Eshkol built-in function`

3. **Document-local define fallback** — if the word is defined in the current document:
   - `**word** — function (defined in this file)`
   - `**word** — variable (defined in this file)`

Hover content is returned as a `MarkupContent` object with `kind: "markdown"`. If none of the three tiers match, the server returns `null` (no hover).

Example hover content for `gradient`:

```
(gradient f) -> f'
Compute gradient of f via autodiff.
```

### 2.4 Go-to-Definition

**Handler:** `handle_definition` (line 835)

The server searches the open document for two patterns using `find_definition()` (line 967):

- `(define (name` — function definition (searched first; offset 9 to position past `(define (`)
- `(define name` — variable definition (searched second; offset 8 to position past `(define `)

The search walks the raw document string and counts line/column numbers by scanning characters. The returned `Location` object contains the document URI and a zero-width range at the start of the name token.

Limitation: go-to-definition is document-local only. Cross-file navigation (e.g. jumping to a `provide`d symbol in another module) is not supported in v1.1.12.

---

## 3. VSCode Extension

### 3.1 Overview

| Property | Value |
|----------|-------|
| Source directory | `tools/vscode-eshkol/` |
| Entry point | `src/extension.ts` |
| Publisher | `tsotchke` |
| Extension ID | `tsotchke.eshkol` |
| Version | `1.1.12` |
| VSCode engine requirement | `^1.75.0` |
| Runtime dependency | `vscode-languageclient ^9.0.1` |
| Language | TypeScript (compiled to `out/extension.js`) |

### 3.2 Language Registration

The extension registers the `eshkol` language ID for files with the `.esk` extension. Three configuration files define the language behaviour:

**`language-configuration.json`** — Editor mechanics:
- Line comment: `;`
- Block comment: `#| ... |#`
- Auto-closing pairs: `()`, `[]`, `{}`, `"..."`, `#| ... |#`
- Word pattern: `[a-zA-Z_][a-zA-Z0-9_\-!?*+/<>=.]*` (standard Scheme identifier rules)
- Indentation increase on: `define`, `lambda`, `let`, `let*`, `letrec`, `begin`, `if`, `cond`, `case`, `when`, `unless`, `do`, `guard`, `with-region`, `borrow`, `match`
- Indentation decrease on: lines starting with `)`

**`syntaxes/eshkol.tmLanguage.json`** — TextMate grammar with named scopes:
- `comment.line.semicolon.eshkol` — `;` to end of line
- `comment.block.eshkol` — `#| ... |#`
- `string.quoted.double.eshkol` — double-quoted strings with escape sequences
- `constant.character.eshkol` — character literals (`#\space`, `#\newline`, etc.)
- `constant.language.boolean.eshkol` — `#t`, `#f`
- `constant.numeric.*` — integers, floats, hex (`#x`), binary (`#b`), octal (`#o`), complex numbers
- `keyword.control.eshkol` — special forms (`define`, `lambda`, `if`, etc.)
- `keyword.operator.eshkol` — `quote`, `quasiquote`, `unquote`
- `support.function.builtin.eshkol` — arithmetic operators and predicates
- `entity.name.function.eshkol` — function definitions
- `variable.other.eshkol` — identifiers

**`snippets/eshkol.json`** — 16 code snippets:

| Prefix | Expansion | Description |
|--------|-----------|-------------|
| `defn` | `(define (name params) body)` | Define a named function |
| `def` | `(define name value)` | Define a variable |
| `lambda` | `(lambda (params) body)` | Anonymous function |
| `let` | `(let ((var val)) body)` | Local variable binding |
| `let*` | `(let* ((v1 val1) (v2 val2)) body)` | Sequential bindings |
| `if` | `(if test consequent alternate)` | Conditional |
| `cond` | `(cond (test expr) ... (else default))` | Multi-way conditional |
| `map` | `(map (lambda (x) body) list)` | Map over list |
| `tensor` | `(tensor (dims) values)` | Create a tensor |
| `grad` | `(gradient (lambda (x) body) point)` | Autodiff gradient |
| `pmap` | `(parallel-map (lambda (x) body) list)` | Parallel map |
| `region` | `(with-region body)` | Lexical memory region |
| `guard` | `(guard (exn ((test) handler)) body)` | Exception handler |
| `match` | `(match expr (pattern1 body1) ...)` | Pattern matching |
| `begin` | `(begin expr1 expr2)` | Sequential block |
| `do` | `(do ((var init step)) ((test) result) body)` | Iteration construct |

### 3.3 LSP Client Integration

The extension activates on `onLanguage:eshkol`. If `eshkol.lsp.enabled` is `true` (default), `startLspClient()` launches the `eshkol-lsp` binary as a child process using `TransportKind.stdio`. The `vscode-languageclient` library handles all JSON-RPC framing, capability negotiation, and message routing.

```typescript
const serverOptions: ServerOptions = {
    run:   { command: lspPath, transport: TransportKind.stdio },
    debug: { command: lspPath, transport: TransportKind.stdio },
};

const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: 'file', language: 'eshkol' }],
    synchronize: {
        fileEvents: vscode.workspace.createFileSystemWatcher('**/*.esk'),
    },
};
```

The client watches all `*.esk` files in the workspace for file system events (creation, deletion, renaming), though the server itself does not act on file watch notifications — those are handled by the LSP client library for cache invalidation.

### 3.4 Commands

Three commands are registered in the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`):

**`Eshkol: Compile Current File`** (`eshkol.compile`)

Compiles the active `.esk` file to a native binary. The terminal command issued is:

```
eshkol-run "<file.esk>" -o "<file>"
```

The output path strips the `.esk` extension. If the file is not an Eshkol file, an error message is shown.

**`Eshkol: Compile and Run Current File`** (`eshkol.run`)

Compiles and immediately runs the binary in a single terminal invocation:

```
eshkol-run "<file.esk>" -o "<file>" && "<file>"
```

Both compile commands reuse an existing `Eshkol` terminal if one is open and still running (checked via `terminal.exitStatus === undefined`).

**`Eshkol: Restart Language Server`** (`eshkol.restartLsp`)

Stops the current LSP client, discards it, and starts a new one. Useful when `eshkol-lsp` has crashed or been rebuilt. Shows an information message on success.

### 3.5 Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `eshkol.lsp.path` | string | `""` | Absolute path to `eshkol-lsp` executable. Empty = search `PATH`. |
| `eshkol.lsp.enabled` | boolean | `true` | Enable or disable the language server. |
| `eshkol.compiler.path` | string | `""` | Absolute path to `eshkol-run` executable. Empty = search `PATH`. |

When `eshkol.lsp.path` is empty, `getLspPath()` returns the bare string `"eshkol-lsp"`, relying on the system `PATH`. The same logic applies to `eshkol.compiler.path` → `"eshkol-run"`.

### 3.6 Task Provider

The extension defines a custom task type `eshkol` for use in `.vscode/tasks.json`:

```json
{
  "type": "eshkol",
  "task": "build",
  "file": "src/main.esk"
}
```

Required property: `task` (string). Optional: `file` (string path to source file).

### 3.7 Installation

**From source:**

```bash
cd tools/vscode-eshkol
npm install
npm run compile          # produces out/extension.js
```

To install into VS Code directly from the directory, use the `Extensions: Install from VSIX...` command after packaging with `vsce package`, or use the `code --extensionDevelopmentPath` flag during development.

**Prerequisites:** `eshkol-lsp` must be on the system `PATH` (or configured via `eshkol.lsp.path`) for IDE features to function. Build it with:

```bash
cd build && make -j8
# eshkol-lsp binary is placed in build/bin/
```

---

## 4. Compiler Debug Flags

The `eshkol-run` binary (source: `exe/eshkol-run.cpp`) exposes several flags for inspecting compiler internals. These are distinct from runtime debugging — they affect the compilation pipeline itself.

### 4.1 Command-Line Flags

| Flag | Short | Type | Effect |
|------|-------|------|--------|
| `--dump-ast` | `-a` | flag | Dump the parsed AST to `<output>.ast` (or `<module>.ast` if no `-o` given) |
| `--dump-ir` | `-i` | flag | Dump generated LLVM IR to `<output>.ll` (or `<module>.ll`) |
| `--debug` | `-d` | flag | Set logger to `ESHKOL_DEBUG` level; also prints IR to stdout after generation |
| `--debug-info` | `-g` | flag | Emit DWARF debug info (enables `lldb`/`gdb` source-level debugging) |
| `--optimize` | `-O` | int | LLVM optimization level: 0=none, 1=basic, 2=full, 3=aggressive |
| `--strict-types` | — | flag | Type errors are fatal instead of warnings |
| `--unsafe` | — | flag | Skip all type checks |
| `--compile-only` | `-c` | flag | Emit `.o` object file only, do not link to binary |
| `--shared-lib` | `-s` | flag | Build a shared library (no `main`; uses `LinkOnceODRLinkage`) |
| `--no-stdlib` | `-n` | flag | Do not auto-load the standard library (`build/stdlib.o`) |
| `--eval` | `-e` | string | JIT-evaluate a single expression and print result |
| `--run` | `-r` | flag | JIT-run a file without compiling to disk |

### 4.2 AST Dump (`--dump-ast`)

When `-a` is passed, after parsing and before code generation the compiler calls `eshkol_ast_pretty_print()` for each top-level form and writes the result to a `.ast` file:

```bash
eshkol-run program.esk -o program --dump-ast
# writes: program.ast
```

Without `-o`, the AST file is named from the first source file's basename. The file contains one block per top-level AST node separated by `=== AST Node ===` / `=================` markers.

### 4.3 LLVM IR Dump (`--dump-ir`)

When `-i` is passed, LLVM IR is generated normally and then written to a `.ll` text file via `eshkol_dump_llvm_ir_to_file()`:

```bash
eshkol-run program.esk -o program --dump-ir
# writes: program.ll
```

The `.ll` file is human-readable LLVM IR. This is the primary tool for diagnosing codegen issues — see Section 6 for usage patterns.

When both `--dump-ir` and `--compile-only` are used together, the IR is dumped and then compiled to a `.o` object file (plus a `.bc` bitcode file for REPL JIT use).

### 4.4 Debug Mode (`--debug`)

Sets the internal logger level to `ESHKOL_DEBUG` (via `eshkol_set_logger_level`), which enables verbose `eshkol_info()` output throughout the compilation pipeline:

- Ownership analysis pass results
- Escape analysis results
- LLVM module name selection
- Optimization level applied
- DWARF path resolved
- All `eshkol_info()` calls in the code generator

Additionally, debug mode calls `eshkol_print_llvm_ir(llvm_module)` to print the IR directly to stdout after generation, before the linker runs. This is useful when you cannot easily read a `.ll` file (e.g. during CI).

### 4.5 Environment Variables

| Variable | Effect |
|----------|--------|
| `ESHKOL_DUMP_REPL_IR=1` | In the REPL/JIT path (`lib/repl/repl_jit.cpp`), prints each JIT-compiled module's LLVM IR to `stderr` before execution |
| `ESHKOL_DEBUG_DL=1` | Prints the module's DataLayout string, target triple, and the LLJIT's DataLayout to `stderr` — useful for diagnosing ABI mismatches between the JIT and precompiled `.o` files |

These environment variables are checked with `getenv()` in `repl_jit.cpp` (lines 573–579) and have no effect during normal ahead-of-time compilation.

---

## 5. Debugging Workflow

### 5.1 Inspecting LLVM IR

The most powerful diagnostic technique for codegen issues is to dump the LLVM IR and read it directly. The IR reveals the actual execution order, PHI node structure, and whether types are being dispatched correctly.

**Basic IR dump:**
```bash
eshkol-run program.esk -o /tmp/prog --dump-ir
cat /tmp/prog.ll
```

**With optimization disabled for readability:**
```bash
eshkol-run program.esk -o /tmp/prog --dump-ir -O 0
```

**What to look for in the IR:**

1. **Type dispatch chains** — tagged value operations typically produce a series of `icmp` + `br` instructions testing the type byte (index 0 of the struct). If a bignum, rational, or complex number is falling through to the wrong branch, the IR will show the missing `icmp` check.

2. **PHI node predecessors** — if code calls `codegenClosureCall` inside a loop with PHI nodes for loop variables, the PHI predecessors will be wrong (pointing at the entry block of the closure call instead of the loop back-edge). This is the canonical cause of infinite-loop bugs in `vector-for-each`, `string-map`, etc. The fix is always `alloca + store/load`.

3. **Struct field indices** — tagged values are `{i8, i8, i16, i32, i64}` (type, flags, reserved, padding, data). The data field is at index 4. If an `extractvalue` uses index 3, it is reading the padding field.

4. **Execution order in letrec** — if a side-effecting form (e.g. `vector-set!`) appears after a later expression in the IR despite appearing first in source, check `transformInternalDefinesToLetrec` in the parser. Only consecutive defines from the body start should be lifted.

### 5.2 REPL IR Debugging

When using the JIT REPL (interactive or `--run`), set `ESHKOL_DUMP_REPL_IR=1` to see each submitted expression's IR:

```bash
ESHKOL_DUMP_REPL_IR=1 eshkol-run --run test.esk 2>ir_dump.txt
```

This is especially useful when the compiled-to-binary path works but the REPL gives wrong results — ABI differences (opt level, DataLayout) are the usual cause.

To diagnose DataLayout/ABI mismatches:

```bash
ESHKOL_DEBUG_DL=1 eshkol-run --run test.esk 2>&1 | head -6
# [REPL] Module DataLayout: e-m:o-i64:64-i128:128-n32:64-S128
# [REPL] Module Triple: arm64-apple-macosx14.0.0
# [REPL] LLJIT DataLayout: e-m:o-i64:64-i128:128-n32:64-S128
```

If the LLJIT DataLayout differs from the module's DataLayout, or if the JIT's optimization level differs from the stdlib's (see the MEMORY note about `CodeGenOptLevel::None`), struct arguments will be passed incorrectly on ARM64 for 3+ argument calls.

### 5.3 Source-Level Debugging with lldb

Compile with DWARF debug info:

```bash
eshkol-run program.esk -o program -g
lldb ./program
```

The `-g` flag calls `eshkol_enable_debug_info()` with the resolved absolute path of the source file before IR generation. DWARF info is embedded in the object file and linked into the final binary.

### 5.4 Diagnosing Parser Issues

The common symptom pattern "works at top level but not inside a function" indicates a parser transformation issue. Check `transformInternalDefinesToLetrec` in `lib/frontend/parser.cpp`. The rule is: only consecutive `define` forms at the start of a body are grouped into `letrec*`; once a non-`define` expression appears, no further defines are hoisted.

For parenthesis or reader issues, the LSP server's diagnostic output (visible in VSCode's Problems panel) reflects what the Eshkol parser actually sees — if the LSP shows no errors but the compiler fails, the issue is in semantic analysis or code generation, not the parser.

### 5.5 Test Scripts

Tests are run via shell scripts in `scripts/`:

```bash
cd /path/to/eshkol
bash scripts/run_parallel_tests.sh
bash scripts/run_macros_tests.sh
```

When grepping test output for failures, use `grep -q "^FAIL:"` or `grep -qE "Failed:[[:space:]]+[1-9]"`. The pattern `grep -qi "FAIL"` produces false positives because it matches "Failed: 0" in summary lines.

---

## 6. Implementation References

The following source locations are the primary entry points for each tool component:

| Component | File | Key Lines |
|-----------|------|-----------|
| LSP server main loop | `tools/lsp/eshkol_lsp.cpp` | 401–410 (`run()`) |
| LSP message dispatch | `tools/lsp/eshkol_lsp.cpp` | 521–539 (`handle_message()`) |
| LSP capabilities | `tools/lsp/eshkol_lsp.cpp` | 542–574 (`handle_initialize()`) |
| Diagnostics | `tools/lsp/eshkol_lsp.cpp` | 621–718 (`publish_diagnostics()`) |
| Completion | `tools/lsp/eshkol_lsp.cpp` | 721–780 (`handle_completion()`) |
| Keyword list | `tools/lsp/eshkol_lsp.cpp` | 419–434 (`keywords()`) |
| Builtin list | `tools/lsp/eshkol_lsp.cpp` | 437–488 (`builtins()`) |
| Hover docs | `tools/lsp/eshkol_lsp.cpp` | 491–518 (`get_doc()`) |
| Hover handler | `tools/lsp/eshkol_lsp.cpp` | 783–832 (`handle_hover()`) |
| Go-to-definition | `tools/lsp/eshkol_lsp.cpp` | 835–872 (`handle_definition()`) |
| Word extraction | `tools/lsp/eshkol_lsp.cpp` | 879–909 (`get_word_at()`) |
| Define extraction | `tools/lsp/eshkol_lsp.cpp` | 912–964 (`extract_defines()`) |
| JSON-RPC transport | `tools/lsp/eshkol_lsp.cpp` | 297–357 (`JsonRpcTransport`) |
| VSCode extension activate | `tools/vscode-eshkol/src/extension.ts` | 12–67 (`activate()`) |
| LSP client startup | `tools/vscode-eshkol/src/extension.ts` | 69–108 (`startLspClient()`) |
| Compiler flag parsing | `exe/eshkol-run.cpp` | 38–57 (`long_options[]`) |
| AST dump implementation | `exe/eshkol-run.cpp` | 2533–2569 |
| IR dump implementation | `exe/eshkol-run.cpp` | 2663–2678 |
| REPL IR/DL env vars | `lib/repl/repl_jit.cpp` | 573–579 |

---

## 7. Package Manager (eshkol-pkg)

### 7.1 Overview

| Property | Value |
|----------|-------|
| Source file | `tools/pkg/eshkol_pkg.cpp` |
| Binary name | `eshkol-pkg` |
| License | MIT |
| Manifest format | TOML (`eshkol.toml`) |
| Registry | Git-based package registry |

The package manager is a standalone CLI tool for managing Eshkol projects and dependencies. It uses a minimal inline TOML parser (no external dependencies) and a git-based package registry for dependency resolution.

### 7.2 Commands

| Command | Description |
|---------|-------------|
| `eshkol-pkg init` | Create a new `eshkol.toml` manifest in the current directory |
| `eshkol-pkg build` | Compile the current project using settings from `eshkol.toml` |
| `eshkol-pkg install` | Install all dependencies listed in `eshkol.toml` |
| `eshkol-pkg add <package>` | Add a dependency to `eshkol.toml` |
| `eshkol-pkg remove <package>` | Remove a dependency from `eshkol.toml` |
| `eshkol-pkg search <query>` | Search the package registry |
| `eshkol-pkg publish` | Publish the current package to the registry |
| `eshkol-pkg run` | Build and run the project |
| `eshkol-pkg clean` | Remove build artifacts |

### 7.3 Manifest Format

```toml
[package]
name = "my-project"
version = "0.1.0"
entry = "src/main.esk"

[dependencies]
math-utils = "1.0.0"
```

---

## 8. Web Compilation Server

For web platform compilation (WASM output), see [Web Platform](WEB_PLATFORM.md). The `--wasm` flag on `eshkol-run` compiles to WebAssembly format, and the bytecode VM subsystem provides an alternative execution model documented in the bytecode VM documentation.

---

## 9. Additional Tool Documentation

- **[VS Code Extension](VSCODE_EXTENSION.md)** — Installation, commands, configuration, syntax highlighting, snippets
- **[Command-Line Reference](COMMAND_LINE_REFERENCE.md)** — Complete flag reference for `eshkol-run` and `eshkol-repl`
- **[Runtime Configuration](RUNTIME_CONFIGURATION.md)** — Environment variables, config files, resource limits

---

## 10. See Also

- `docs/breakdown/README.md` — component index
- `docs/ESHKOL_V1_ARCHITECTURE.md` — compiler pipeline architecture
- `docs/API_REFERENCE.md` — full runtime API including environment variables
- `CONTRIBUTING.md` — build instructions and contribution guidelines
- `lib/frontend/parser.cpp` — parser and `transformInternalDefinesToLetrec`
- `lib/backend/llvm_codegen.cpp` — primary code generator (29K lines)
