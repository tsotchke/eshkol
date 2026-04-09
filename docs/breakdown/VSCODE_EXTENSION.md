# VS Code Extension

**Status:** Production (v1.1.13)
**Applies to:** Eshkol compiler v1.1-accelerate and later

---

## Overview

The Eshkol VS Code extension provides syntax highlighting, code completion, hover documentation, go-to-definition, live diagnostics, and code snippets for `.esk` files. It integrates with the Eshkol Language Server Protocol (LSP) server for intelligent editor features and provides commands for compiling and running Eshkol programs directly from VS Code.

| Property | Value |
|----------|-------|
| Source directory | `tools/vscode-eshkol/` |
| Publisher | `tsotchke` |
| Extension ID | `tsotchke.eshkol` |
| VS Code engine | `^1.75.0` |
| Runtime dependency | `vscode-languageclient ^9.0.1` |

---

## Installation

### From Source

```bash
cd tools/vscode-eshkol
npm install
npm run compile
```

Then install into VS Code using one of these methods:

**Option A -- Development mode:**
```bash
code --extensionDevelopmentPath=$(pwd)
```

**Option B -- Package as VSIX:**
```bash
npx vsce package
```
Then in VS Code: `Extensions: Install from VSIX...` and select the generated `.vsix` file.

### Prerequisites

The extension requires the `eshkol-lsp` binary for IDE features (completion, hover, diagnostics). Build it from the Eshkol project:

```bash
cd /path/to/eshkol
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

The `eshkol-lsp` binary is placed in `build/`. Either add it to your system `PATH` or configure the path in VS Code settings.

---

## Commands

Three commands are available via the Command Palette (`Cmd+Shift+P` on macOS, `Ctrl+Shift+P` on Linux/Windows):

### Eshkol: Compile Current File

**Command ID:** `eshkol.compile`

Compiles the active `.esk` file to a native binary:

```
eshkol-run "file.esk" -o "file"
```

The output binary name is the source filename with the `.esk` extension removed.

### Eshkol: Compile and Run Current File

**Command ID:** `eshkol.run`

Compiles and immediately runs the binary:

```
eshkol-run "file.esk" -o "file" && "./file"
```

Both compile commands reuse an existing `Eshkol` terminal if one is open and still running.

### Eshkol: Restart Language Server

**Command ID:** `eshkol.restartLsp`

Stops the current LSP client, discards it, and starts a fresh connection. Useful when `eshkol-lsp` has crashed or has been rebuilt after a compiler update.

---

## Configuration

All settings are under the `eshkol.*` namespace in VS Code settings.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `eshkol.lsp.enabled` | boolean | `true` | Enable or disable the language server. When disabled, syntax highlighting and snippets still work, but completion, hover, diagnostics, and go-to-definition are unavailable. |
| `eshkol.lsp.path` | string | `""` | Absolute path to the `eshkol-lsp` executable. When empty, the extension searches the system `PATH` for `eshkol-lsp`. |
| `eshkol.compiler.path` | string | `""` | Absolute path to the `eshkol-run` executable. When empty, the extension searches the system `PATH` for `eshkol-run`. |

### Example settings.json

```json
{
    "eshkol.lsp.enabled": true,
    "eshkol.lsp.path": "/usr/local/bin/eshkol-lsp",
    "eshkol.compiler.path": "/usr/local/bin/eshkol-run"
}
```

---

## Syntax Highlighting

The extension registers the `eshkol` language ID for files with the `.esk` extension. The TextMate grammar (`syntaxes/eshkol.tmLanguage.json`) provides scoped highlighting for:

| Element | Scope | Examples |
|---------|-------|---------|
| Line comments | `comment.line.semicolon.eshkol` | `; comment` |
| Block comments | `comment.block.eshkol` | `#| block |#` |
| Strings | `string.quoted.double.eshkol` | `"hello"` |
| Character literals | `constant.character.eshkol` | `#\space`, `#\newline` |
| Booleans | `constant.language.boolean.eshkol` | `#t`, `#f` |
| Numbers | `constant.numeric.*` | `42`, `3.14`, `#xFF`, `#b1010`, `3+4i` |
| Special forms | `keyword.control.eshkol` | `define`, `lambda`, `if`, `let`, `cond` |
| Quote operators | `keyword.operator.eshkol` | `quote`, `quasiquote`, `unquote` |
| Builtin functions | `support.function.builtin.eshkol` | `+`, `-`, `car`, `cdr`, `map` |
| Function definitions | `entity.name.function.eshkol` | name in `(define (name ...))` |
| Identifiers | `variable.other.eshkol` | any other identifier |

---

## Code Snippets

16 code snippets are available for common Eshkol patterns:

| Prefix | Expansion | Description |
|--------|-----------|-------------|
| `defn` | `(define (name params) body)` | Define a named function |
| `def` | `(define name value)` | Define a variable |
| `lambda` | `(lambda (params) body)` | Anonymous function |
| `let` | `(let ((var val)) body)` | Local variable binding |
| `let*` | `(let* ((v1 val1) (v2 val2)) body)` | Sequential bindings |
| `if` | `(if test consequent alternate)` | Conditional expression |
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

---

## Language Configuration

The language configuration (`language-configuration.json`) defines:

- **Line comment:** `;`
- **Block comment:** `#| ... |#`
- **Auto-closing pairs:** `()`, `[]`, `{}`, `""`, `#| |#`
- **Auto-indent triggers:** `define`, `lambda`, `let`, `let*`, `letrec`, `begin`, `if`, `cond`, `case`, `when`, `unless`, `do`, `guard`, `with-region`, `borrow`, `match`
- **Auto-dedent:** Lines starting with `)`
- **Word pattern:** `[a-zA-Z_][a-zA-Z0-9_\-!?*+/<>=.]*` (Scheme identifier rules)

---

## LSP Features

When the language server is enabled, the following features are active:

### Diagnostics

Live parse-error reporting in the Problems panel. Detects unmatched parentheses and unclosed expressions. Diagnostics update on every keystroke (full document sync).

### Completion

Triggered by `(` or space. Provides three sources of completions:

- **33 keywords** (special forms): `define`, `lambda`, `if`, `let`, `cond`, `begin`, etc.
- **~80 builtin functions**: arithmetic, predicates, pairs/lists, strings, vectors, I/O, tensors, autodiff, parallel primitives
- **Document-local definitions**: functions and variables defined in the current file

### Hover

Shows documentation and type information for keywords, builtins, and local definitions. 16 functions have detailed markdown documentation (including `define`, `lambda`, `gradient`, `parallel-map`, `tensor`, etc.).

### Go-to-Definition

Navigates to the definition of a symbol within the current document. Cross-file navigation is not yet supported.

---

## Task Provider

The extension provides a custom `eshkol` task type for use in `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "type": "eshkol",
            "task": "build",
            "file": "src/main.esk",
            "label": "Build Eshkol project"
        }
    ]
}
```

---

## Troubleshooting

**LSP not starting:** Verify that `eshkol-lsp` is on your `PATH` or configured via `eshkol.lsp.path`. Check the Output panel (select "Eshkol" from the dropdown) for error messages.

**No syntax highlighting:** Ensure the file has a `.esk` extension. The extension activates on the `eshkol` language ID, which is associated with `.esk` files.

**Stale diagnostics after rebuilding the compiler:** Use the `Eshkol: Restart Language Server` command to reconnect with the updated `eshkol-lsp` binary.

---

## See Also

- [Developer Tools](DEVELOPER_TOOLS.md) -- Full LSP server implementation details
- [Command-Line Reference](COMMAND_LINE_REFERENCE.md) -- Compiler flags used by the extension
- [Getting Started](GETTING_STARTED.md) -- Build instructions for the compiler and tools
