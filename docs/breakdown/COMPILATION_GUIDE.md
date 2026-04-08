# Compilation Guide for Eshkol v1.1-accelerate

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Building the Compiler](#building-the-compiler)
- [Build Options](#build-options)
- [Compiling Eshkol Programs](#compiling-eshkol-programs)
- [CLI Reference](#cli-reference)
- [REPL Usage](#repl-usage)
- [WebAssembly Target](#webassembly-target)
- [Debugging](#debugging)
- [Common Issues](#common-issues)

---

## Overview

Eshkol compiles programs to **native executables via LLVM 21**. The compilation pipeline:

```
.esk source → Macro expansion → AST → HoTT type checking → LLVM IR → Optimization → Native code
```

**Five executables:**
- `eshkol-run` - Ahead-of-time compiler and JIT runner
- `eshkol-repl` - Interactive REPL with JIT compilation
- `eshkol-pkg` - Package manager (registry, install, publish)
- `eshkol-lsp` - Language Server Protocol server (IDE integration)
- `eshkol-server` - Web platform server (HTTP API for compilation)

---

## Prerequisites

### Required

- **LLVM 21** (core code generation backend for lite/native builds)
- **CMake 3.14+** (build system)
- **C++20 compiler** (GCC 11+, Clang 14+, or Apple Clang 15+)
- **Ninja** (recommended build tool, faster than Make)

### Optional

- **Apple Accelerate** (macOS, auto-detected) - BLAS acceleration via AMX
- **OpenBLAS** (Linux, auto-detected) - BLAS acceleration
- **Metal framework** (macOS, auto-detected) - GPU acceleration
- **CUDA Toolkit** (Linux/NVIDIA, auto-detected) - GPU acceleration
- **MLIR + StableHLO** (for XLA backend builds)

---

## Building the Compiler

### Quick Build

```bash
# Clone repository
git clone https://github.com/tsotchke/eshkol.git
cd eshkol

# Configure with Ninja (recommended)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build (produces 5 executables)
cmake --build build -j$(nproc)

# Executables in build/
ls build/eshkol-*
# eshkol-run    - AOT compiler / JIT runner
# eshkol-repl   - Interactive REPL
# eshkol-pkg    - Package manager
# eshkol-lsp    - LSP server
# eshkol-server - Web server
```

### Platform-Specific Builds

#### Ubuntu/Debian

```bash
# Install dependencies
sudo apt-get install -y \
    llvm-21 llvm-21-dev clang-21 \
    cmake ninja-build build-essential \
    libopenblas-dev  # Optional: BLAS acceleration

# Build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

#### macOS

```bash
# Install via Homebrew
brew install llvm@21 cmake ninja

# Configure with LLVM path
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR=$(brew --prefix llvm@21)/lib/cmake/llvm

# Build
cmake --build build -j$(sysctl -n hw.ncpu)
```

Apple Accelerate (BLAS) and Metal (GPU) are auto-detected on macOS.

#### Windows

```powershell
# In Developer PowerShell for VS 2022
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T ClangCL `
    -DCMAKE_BUILD_TYPE=Release `
    -DLLVM_DIR="C:/Program Files/LLVM/lib/cmake/llvm"
cmake --build build --config Release --parallel
```

#### Fedora/RHEL

```bash
sudo dnf install llvm17-devel cmake ninja-build gcc-c++ openblas-devel
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Build Types

```bash
# Debug build (with assertions, no optimizations, sanitizer-ready)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug

# Release build (optimized, no debug info)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# RelWithDebInfo (optimized + debug symbols)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

---

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `ESHKOL_BLAS_ENABLED` | `ON` | BLAS acceleration (Accelerate on macOS, OpenBLAS on Linux) |
| `ESHKOL_GPU_ENABLED` | `ON` | GPU acceleration (Metal on macOS, CUDA on Linux) |
| `ESHKOL_XLA_ENABLED` | `OFF` | XLA/StableHLO backend for tensor fusion |
| `ESHKOL_ENABLE_ASAN` | `OFF` | Address Sanitizer |
| `ESHKOL_ENABLE_UBSAN` | `OFF` | Undefined Behavior Sanitizer |
| `ESHKOL_BUILD_TESTS` | `ON` | Build test suite |
| `STABLEHLO_ROOT` | `""` | Path to StableHLO build (for XLA mode) |

### Examples

```bash
# Full build with GPU + BLAS (default)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Minimal build (no GPU, no BLAS)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DESHKOL_GPU_ENABLED=OFF -DESHKOL_BLAS_ENABLED=OFF

# XLA build (requires MLIR + StableHLO)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DESHKOL_XLA_ENABLED=ON -DSTABLEHLO_ROOT=/path/to/stablehlo

# Debug build with sanitizers
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    -DESHKOL_ENABLE_ASAN=ON -DESHKOL_ENABLE_UBSAN=ON
```

---

## Compiling Eshkol Programs

### Basic Compilation

```bash
# Create program
cat > hello.esk << 'EOF'
(display "Hello, Eshkol!")
(newline)
EOF

# Compile to executable
eshkol-run hello.esk -o hello

# Run
./hello
# Output: Hello, Eshkol!
```

### JIT Execution

```bash
# Evaluate an expression directly (no binary produced)
eshkol-run -e '(+ 1 2 3)'
# Output: 6

# JIT-run a file (no binary produced)
eshkol-run -r script.esk
```

### Compilation Pipeline

`eshkol-run` performs these steps:

1. **Macro expansion** - Process `define-syntax` and `syntax-rules`
2. **S-expression parsing** - Build AST from source
3. **HoTT type checking** - Infer types, emit warnings (or errors with `--strict-types`)
4. **LLVM IR generation** - Translate AST to LLVM IR with tagged value encoding
5. **LLVM optimization** - Apply optimization passes (controlled by `-O` level)
6. **Native code generation** - LLVM backend produces machine code
7. **Linking** - Link with runtime, stdlib, and acceleration libraries (BLAS, GPU, XLA)
8. **Executable creation** - Produce final standalone binary

### Multi-File Projects

Eshkol uses `require`/`provide` for modules:

```scheme
;;; math-utils.esk
(provide square cube)

(define (square x) (* x x))
(define (cube x) (* x x x))

;;; main.esk
(require 'math-utils)

(display (square 5))
(newline)
```

Compile the main file (automatically loads dependencies):

```bash
eshkol-run main.esk -o main
./main
# Output: 25
```

### Standard Library

Programs can use the precompiled standard library:

```scheme
(require stdlib)

;; 350+ functions across 34 modules
(display (reverse (list 1 2 3)))  ; (3 2 1)
(display (filter even? (list 1 2 3 4 5)))  ; (2 4)
```

The stdlib is precompiled to `build/stdlib.o` and automatically linked.

---

## CLI Reference

### `eshkol-run` — Compiler and JIT Runner

```
Usage: eshkol-run [options] <input.esk|input.o> [input.esk|input.o]
       eshkol-run -e '<expression>'   (JIT evaluate expression)
       eshkol-run -r <file.esk>       (JIT run file)
```

| Flag | Short | Description |
|------|-------|-------------|
| `--help` | `-h` | Print help message |
| `--output <file>` | `-o` | Output to named binary (without this, runs the default binary name) |
| `--debug` | `-d` | Add debugging information to the program |
| `--debug-info` | `-g` | Emit DWARF debug info (enables lldb/gdb source-level debugging) |
| `--optimize <N>` | `-O` | LLVM optimization level: 0=none, 1=basic, 2=full, 3=aggressive |
| `--dump-ast` | `-a` | Dump the AST to a `.ast` file |
| `--dump-ir` | `-i` | Dump LLVM IR to a `.ll` file |
| `--compile-only` | `-c` | Compile to object file only (no linking) |
| `--shared-lib` | `-s` | Compile as a shared library |
| `--wasm` | `-w` | Compile to WebAssembly (`.wasm`) format |
| `--eval <expr>` | `-e` | JIT evaluate an expression and print the result |
| `--run` | `-r` | JIT run a file (interpret without creating a binary) |
| `--lib <name>` | `-l` | Link a shared library to the resulting executable |
| `--lib-path <dir>` | `-L` | Add a directory to the library search path |
| `--no-stdlib` | `-n` | Do not auto-load the standard library |
| `--strict-types` | | Type errors are fatal (default: gradual/warnings) |
| `--unsafe` | | Skip all type checks |

### Examples

```bash
# Compile with debug symbols for lldb
eshkol-run program.esk -o program -g

# Compile with aggressive optimization
eshkol-run program.esk -o program -O 3

# Compile with strict type checking (warnings → errors)
eshkol-run program.esk -o program --strict-types

# Inspect generated LLVM IR
eshkol-run program.esk -i
cat program.ll

# Quick one-liner evaluation
eshkol-run -e '(map (lambda (x) (* x x)) (list 1 2 3 4 5))'

# Compile to WebAssembly
eshkol-run program.esk --wasm -o program.wasm
```

---

## REPL Usage

### Starting the REPL

```bash
eshkol-repl
```

### REPL Session

```scheme
eshkol> (define (factorial n)
          (if (<= n 1) 1 (* n (factorial (- n 1)))))

eshkol> (factorial 20)
2432902008176640000

eshkol> (gradient (lambda (x) (* x x)) 3.0)
6.0

eshkol> :type (+ 1 2)
integer

eshkol> :time (factorial 1000)
[result]
Elapsed: 0.002s

eshkol> :quit
```

### REPL Commands

| Command | Alias | Description |
|---------|-------|-------------|
| `:help` | `:h` | Show help message |
| `:quit` | `:q` | Exit the REPL |
| `:cancel` | `:c` | Cancel multi-line input |
| `:clear` | | Clear the screen |
| `:env` | `:e` | Show defined symbols in environment |
| `:type <expr>` | `:t` | Show type of an expression |
| `:doc <name>` | `:d` | Show documentation for a function |
| `:ast <expr>` | | Show AST for an expression |
| `:time <expr>` | | Time execution of an expression |
| `:load <file>` | `:l` | Load and execute a file |
| `:reload` | `:r` | Reload the last loaded file |
| `:stdlib` | | Load the standard library |
| `:reset` | | Reset the REPL state |
| `:history [n]` | | Show command history (optionally last n entries) |
| `:version` | `:v` | Show version information |
| `:examples` | | Show example expressions |

### REPL Features

- **JIT compilation** - Each expression is compiled and executed via LLVM ORC JIT
- **Persistent state** - Definitions carry across inputs
- **Stdlib access** - Use `:stdlib` to load all 350+ standard library functions
- **Hot reload** - `:load` and `:reload` for iterative development
- **Command history** - Arrow keys navigate history, persisted across sessions

---

## WebAssembly Target

Eshkol can compile programs to WebAssembly:

```bash
eshkol-run program.esk --wasm -o program.wasm
```

This uses the WebAssembly backend built into LLVM 21. The generated `.wasm` file can be run in any WASM runtime (wasmtime, wasmer, browser, etc.).

---

## Debugging

### Inspecting Generated Code

```bash
# Dump LLVM IR to see what the compiler generates
eshkol-run program.esk -i
# Creates program.ll with full LLVM IR

# Dump AST to see parsed structure
eshkol-run program.esk -a
# Creates program.ast
```

### Source-Level Debugging

```bash
# Compile with DWARF debug info
eshkol-run program.esk -o program -g

# Debug with LLDB (macOS)
lldb ./program
(lldb) breakpoint set --name main
(lldb) run
(lldb) step
(lldb) bt

# Debug with GDB (Linux)
gdb ./program
(gdb) break main
(gdb) run
(gdb) next
(gdb) backtrace
```

### Memory Debugging

**Address Sanitizer (recommended):**

```bash
# Rebuild compiler with ASan enabled
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DESHKOL_ENABLE_ASAN=ON
cmake --build build -j$(nproc)

# Compile and run program — ASan reports memory errors with stack traces
eshkol-run program.esk -o program
./program
```

**Undefined Behavior Sanitizer:**

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DESHKOL_ENABLE_UBSAN=ON
cmake --build build -j$(nproc)
```

**Valgrind (Linux):**

```bash
valgrind --leak-check=full --show-leak-kinds=all ./program
```

---

## Common Issues

### LLVM not found

```
CMake Error: Could not find LLVM
```

**Solution:**

```bash
# Ubuntu/Debian
sudo apt-get install llvm-21-dev

# macOS
brew install llvm@21
export LLVM_DIR=$(brew --prefix llvm@21)/lib/cmake/llvm

# Then reconfigure
cmake -B build -G Ninja -DLLVM_DIR=$LLVM_DIR
```

### Type warnings

```
Warning: Type mismatch at line 10
Expected: integer
Found: real
```

Type checker emits warnings by default (gradual typing). Options:
- Fix the type mismatch in your code
- Use `--strict-types` to make warnings into errors
- Use `--unsafe` to skip type checking entirely

### BLAS not found

```
BLAS acceleration: DISABLED (no BLAS library found)
```

The compiler still works without BLAS, but matrix operations will be slower.

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# Fedora/RHEL
sudo dnf install openblas-devel

# macOS — Accelerate is built-in, should auto-detect
```

### Linker errors with stdlib

If you see duplicate symbol errors when overriding stdlib functions, ensure you're using the latest compiler build. Stdlib symbols use `LinkOnceODRLinkage` so user definitions take priority.

---

## See Also

- [Getting Started](GETTING_STARTED.md) - Installation and first programs
- [Compiler Architecture](COMPILER_ARCHITECTURE.md) - LLVM backend internals
- [Type System](TYPE_SYSTEM.md) - Understanding type warnings
- [Memory Management](MEMORY_MANAGEMENT.md) - Arena allocation
- [XLA Backend](XLA_BACKEND.md) - Accelerated tensor operations
- [GPU Acceleration](GPU_ACCELERATION.md) - Metal and CUDA backends
