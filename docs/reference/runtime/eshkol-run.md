# `eshkol-run` — Compiler & JIT Driver

`eshkol-run` is the primary entry point: it compiles `.esk` source to native
code (AOT), evaluates or runs it in-process via the LLVM JIT, emits object files
/ shared libraries / WebAssembly / VM bytecode, and dumps ASTs and IR.

```
Usage: eshkol-run [options] <input.esk|input.o> [input.esk|input.o]
       eshkol-run -e '<expression>'   (JIT evaluate expression)
       eshkol-run -r <file.esk>       (JIT run file)
```

Reported version: `Eshkol Compiler v1.2.4-scale` (`--version`).

## Modes at a glance

| Mode | Invocation | Effect |
|------|-----------|--------|
| AOT compile + link | `eshkol-run in.esk -o out` | Produce a native executable |
| Object only | `eshkol-run -c in.esk -o in.o` | Emit an object file, no link |
| Shared library | `eshkol-run -s in.esk -o libin.so` | Emit a shared library |
| JIT run | `eshkol-run -r in.esk` | Compile in-memory and run immediately |
| JIT eval | `eshkol-run -e '(expr)'` | Evaluate one expression in-process |
| WebAssembly | `eshkol-run -w in.esk -o in.wasm` | Emit `.wasm` |
| VM bytecode | `eshkol-run --profile hosted-vm -B out.eskb in.esk` | Emit ESKB |

Multiple `.esk`/`.o` inputs may be given and are linked together.

## Full flag reference

### Output & compilation

| Flag | Alias | Meaning |
|------|-------|---------|
| `--output FILE` | `-o` | Output path (executable, object, `.so`, `.wasm`, per mode) |
| `--compile-only` | `-c` | Compile to an intermediate object file (no link) |
| `--emit-object` | | Alias for `--compile-only` |
| `--shared-lib` | `-s` | Compile to a shared library |
| `-fPIC` | | Accepted for build-system compatibility (no-op flag) |
| `--wasm` | `-w` | Compile to WebAssembly (`.wasm`) |
| `--no-stdlib` | `-n` | Do not auto-load the standard library |

### JIT execution

| Flag | Alias | Meaning |
|------|-------|---------|
| `--run FILE` | `-r` | JIT-run a file (compile in memory, execute) |
| `--eval 'EXPR'` | `-e` | JIT-evaluate a single expression |

### Optimization & debugging

| Flag | Alias | Meaning |
|------|-------|---------|
| `--optimize N` | `-O` | LLVM optimization level: `0` none, `1` basic, `2` full, `3` aggressive |
| `--debug` | `-d` | Add debugging information inside the program |
| `--debug-info` | `-g` | Emit DWARF debug info (source-level lldb/gdb) |
| `--dump-ast` | `-a` | Dump the AST to a `.ast` file |
| `--dump-ir` | `-i` | Dump LLVM IR to a `.ll` file |

### Type checking

| Flag | Meaning |
|------|---------|
| `--strict-types` | Type errors become fatal (default: gradual — warnings only) |
| `--unsafe` | Skip all type checks |

### Search paths

| Flag | Alias | Meaning |
|------|-------|---------|
| `-I DIR` | | Add a source/module search path |
| `--lib NAME` | `-l` | Link a shared library into the executable |
| `--lib-path DIR` | `-L` | Add a directory to the library search path |
| `-D NAME[=VALUE]` | | Accepted for build-system compatibility (no-op) |

### Targets & profiles

| Flag | Meaning |
|------|---------|
| `--target TRIPLE` | Set the LLVM target triple (cross-compilation) |
| `--profile NAME` | Execution profile (see below) |
| `--emit-eskb FILE` / `-B FILE` | Emit Eshkol VM bytecode (ESKB); requires a VM profile |
| `--require-vm-entry NAME` | Require a named VM entry in the emitted ESKB (VM profile only) |
| `--require-vm-entry-zero-arg NAME` | Require a named zero-argument VM entry in the emitted ESKB |

**Profiles:** `hosted-native`, `hosted-wasm`, `hosted-vm`,
`freestanding-kernel-native`, `freestanding-mcu-native`, `freestanding-vm`,
`embedded-vm`. See [`eshkol-vm-standalone`](eshkol-vm-standalone.md) for the ESKB
format and the VM profiles, and the [platform target matrix](../../platform/TARGET_SUPPORT_MATRIX.md)
for what each profile guarantees.

### Info

| Flag | Alias | Meaning |
|------|-------|---------|
| `--help` | `-h` | Print help |
| `--version` | | Print version |

## Verified examples

```sh
$ eshkol-run -r hello.esk                 # JIT run
42

$ eshkol-run -e '(display (+ 2 3))'       # JIT eval, explicit output
5

$ eshkol-run hello.esk -o hello && ./hello # AOT compile + run
hi

$ eshkol-run -c hello.esk -o hello.o      # object only  (25 KB .o)
$ eshkol-run -r hello.esk --dump-ir       # also writes hello.ll to CWD
$ eshkol-run -r hello.esk --dump-ast      # also writes hello.ast to CWD
```

> **Discrepancy (report only):** `--help` describes `-e` as "JIT evaluate an
> expression and print the result," but `eshkol-run -e '(+ 2 3)'` produces **no**
> output — the resulting value is not auto-printed. Use `display` explicitly
> (`-e '(display (+ 2 3))'`). Side effects and definitions do run.

`--dump-ir` / `--dump-ast` write `<source-basename>.ll` / `.ast` into the current
working directory.

## Related

- [Environment variables](environment-variables.md) — `ESHKOL_JIT_CACHE`,
  search paths, resource limits, and more.
- [JIT internals](jit-internals.md) — run cache, stdlib object cache, code-model
  notes.
- [`eshkol-repl`](eshkol-repl.md) — interactive REPL and the `--machine`
  warm-worker protocol.
