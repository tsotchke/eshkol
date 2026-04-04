# Command-Line Reference

**Status:** Production (v1.1.11)
**Applies to:** Eshkol compiler v1.1-accelerate and later

---

## eshkol-run (AOT Compiler)

The `eshkol-run` binary is the ahead-of-time compiler and JIT execution engine. It parses Eshkol source files, generates LLVM IR, compiles to native code, and optionally links into an executable.

### Usage

```
eshkol-run [options] <input.esk|input.o> [input.esk|input.o ...]
eshkol-run -e '<expression>'
eshkol-run -r <file.esk>
```

### Flags

#### Output Control

| Flag | Short | Argument | Description |
|------|-------|----------|-------------|
| `--output` | `-o` | `<path>` | Output path for the compiled binary. Without this flag, the binary is named from the first source file. |
| `--compile-only` | `-c` | (none) | Compile to an object file (`.o`) only; do not link into an executable. Also produces a `.bc` bitcode file for REPL JIT use. |
| `--shared-lib` | `-s` | (none) | Compile as a shared library (no `main` function). All symbols use `LinkOnceODRLinkage` so user programs can override them. Implies `--compile-only`. |
| `--wasm` | `-w` | (none) | Compile to WebAssembly (`.wasm`) format. Sets the target triple to `wasm32-unknown-unknown`. |
| `--emit-eskb` | `-B` | `<path>` | Emit ESKB bytecode format to the specified path. Used by the bytecode VM subsystem. |

**Examples:**

```bash
# Compile to native binary
eshkol-run hello.esk -o hello

# Compile to object file only
eshkol-run module.esk -c -o module.o

# Compile as shared library
eshkol-run mylib.esk -s -o mylib.o

# Compile to WebAssembly
eshkol-run app.esk -w -o app.wasm

# Emit bytecode
eshkol-run program.esk -B program.eskb
```

#### JIT Execution

| Flag | Short | Argument | Description |
|------|-------|----------|-------------|
| `--eval` | `-e` | `<expression>` | JIT-evaluate a single Eshkol expression and print the result. Does not compile to disk. |
| `--run` | `-r` | (none) | JIT-run a source file without compiling to disk. Uses LLVM OrcJIT for in-memory compilation. |

**Examples:**

```bash
# Evaluate an expression
eshkol-run -e '(+ 1 2 3)'
# Output: 6

# Run a file via JIT
eshkol-run -r program.esk
```

#### Debug and Diagnostics

| Flag | Short | Argument | Description |
|------|-------|----------|-------------|
| `--debug` | `-d` | (none) | Enable debug output. Sets log level to `ESHKOL_DEBUG`, prints LLVM IR to stdout after generation. |
| `--dump-ast` | `-a` | (none) | Dump the parsed AST to a `.ast` file (e.g., `program.ast`). |
| `--dump-ir` | `-i` | (none) | Dump the generated LLVM IR to a `.ll` file (e.g., `program.ll`). |
| `--debug-info` | `-g` | (none) | Emit DWARF debug info in the compiled binary, enabling source-level debugging with `lldb` or `gdb`. |

**Examples:**

```bash
# Dump AST for inspection
eshkol-run program.esk -o program --dump-ast
cat program.ast

# Dump LLVM IR for codegen debugging
eshkol-run program.esk -o program --dump-ir
cat program.ll

# Compile with debug info for lldb
eshkol-run program.esk -o program -g
lldb ./program

# Verbose debug output
eshkol-run -d program.esk -o program
```

#### Optimization

| Flag | Short | Argument | Description |
|------|-------|----------|-------------|
| `--optimize` | `-O` | `<0-3>` | Set the LLVM optimization level. `0` = none (fastest compile, easiest to debug IR), `1` = basic, `2` = standard, `3` = aggressive. |

**Examples:**

```bash
# No optimization (for readable IR)
eshkol-run program.esk -o program -O 0 --dump-ir

# Aggressive optimization
eshkol-run program.esk -o program -O 3
```

#### Type System

| Flag | Short | Argument | Description |
|------|-------|----------|-------------|
| `--strict-types` | (none) | (none) | Make type errors fatal (default: gradual typing with warnings). |
| `--unsafe` | (none) | (none) | Skip all type checks entirely. Use for maximum performance in trusted code. |

#### Library Management

| Flag | Short | Argument | Description |
|------|-------|----------|-------------|
| `--lib` | `-l` | `<name>` | Link a shared library to the resulting executable. Can be specified multiple times. |
| `--lib-path` | `-L` | `<directory>` | Add a directory to the library search path. Can be specified multiple times. |
| `--no-stdlib` | `-n` | (none) | Do not auto-load the standard library (`build/stdlib.o`). Use for minimal builds or when providing a custom stdlib. |

**Examples:**

```bash
# Link with a custom library
eshkol-run program.esk -o program -l mylib -L /path/to/libs

# Compile without standard library
eshkol-run minimal.esk -o minimal -n

# Use build directory as library path
eshkol-run program.esk -o program -L build/
```

#### Other

| Flag | Short | Description |
|------|-------|-------------|
| `--help` | `-h` | Print the help message and exit. |

### Input Files

`eshkol-run` accepts both `.esk` source files and `.o` object files. Multiple files can be provided and they will be compiled and linked together:

```bash
# Compile two source files together
eshkol-run main.esk utils.esk -o program

# Link a pre-compiled object with source
eshkol-run main.esk precompiled.o -o program
```

---

## eshkol-repl (Interactive REPL)

The `eshkol-repl` binary provides an interactive JIT-compiled read-eval-print loop with readline support, colorized output, and crash recovery.

### Starting the REPL

```bash
eshkol-repl
```

The REPL automatically loads the standard library at startup if `stdlib.o` is found in the build directory.

### REPL Commands

All REPL commands start with `:`. Eshkol expressions are entered directly without any prefix.

#### Session Control

| Command | Short | Description |
|---------|-------|-------------|
| `:help` | `:h` | Show available commands |
| `:quit` | `:q` | Exit the REPL. Also accepts `(exit)` or `exit`. |
| `:clear` | (none) | Clear the terminal screen |
| `:version` | `:v` | Show Eshkol version information |
| `:history` | (none) | Show the last 20 commands from the readline history |
| `:reset` | (none) | Clear tracked REPL state (note: JIT symbols persist until restart) |

#### Code Loading

| Command | Short | Description |
|---------|-------|-------------|
| `:load <file>` | `:l <file>` | Load and execute an Eshkol source file |
| `:reload` | `:r` | Reload the last loaded file |
| `:stdlib` | (none) | Manually load the standard library (if not loaded at startup) |

#### Inspection

| Command | Short | Description |
|---------|-------|-------------|
| `:type <expr>` | `:t <expr>` | Show the inferred type of an expression |
| `:ast <expr>` | (none) | Show the parsed AST structure of an expression |
| `:env` | `:e` | Show defined symbols in the current session |
| `:doc <name>` | `:d <name>` | Show documentation for a function or keyword |

#### Performance

| Command | Short | Description |
|---------|-------|-------------|
| `:time <expr>` | (none) | Execute an expression and show timing breakdown (parse time, JIT+run time, total) |

### REPL Usage Examples

```
> (+ 1 2 3)
6

> (define (fib n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))

> (fib 20)
6765

> :type (+ 1 2)
Type: integer

> :time (fib 30)
832040
--- Timing ---
  Parse:       12 us
  JIT+Run:  14523 us
  Total:    14535 us

> :load myfile.esk

> :doc map
(map f lst) -> list
Apply f to each element of lst, returning a new list.

> :quit
Goodbye!
```

### Multi-line Input

The REPL supports multi-line input. If an expression has unclosed parentheses, the REPL will prompt for continuation lines until all parentheses are balanced.

### Crash Recovery

The REPL installs signal handlers for SIGSEGV, SIGFPE, and SIGBUS during JIT execution. If a crash occurs during evaluation, the REPL recovers and prints an error message rather than terminating.

### Interrupt Handling

Press `Ctrl+C` to cancel the current expression evaluation. The REPL will return to the prompt.

---

## See Also

- [Developer Tools](DEVELOPER_TOOLS.md) -- Debug flags, IR inspection, REPL IR dumps
- [Runtime Configuration](RUNTIME_CONFIGURATION.md) -- Environment variables
- [Getting Started](GETTING_STARTED.md) -- Build instructions
- [Compilation Guide](COMPILATION_GUIDE.md) -- LLVM compilation pipeline details
