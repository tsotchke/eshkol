# Compilation Guide for Eshkol

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Building the Compiler](#building-the-compiler)
- [Compiling Eshkol Programs](#compiling-eshkol-programs)
- [REPL Usage](#repl-usage)
- [Debugging](#debugging)
- [Common Issues](#common-issues)

---

## Overview

Eshkol compiles programs to **native executables via LLVM**. The compilation process:

```
.esk source → Macro expansion → AST → Type checking → LLVM IR → Optimization → Native code
```

**Two executables:**
- `eshkol-run` - Ahead-of-time compiler (generates standalone binaries)
- `eshkol-repl` - Interactive REPL with JIT compilation

---

## Prerequisites

### Required

- **LLVM 14+** (core code generation backend)
- **CMake 3.16+** (build system)
- **C++17 compiler** (GCC 9+, Clang 10+, or MSVC 2019+)

### Optional

- **GDB/LLDB** (debugging)
- **Valgrind** (memory debugging)
- **perf** (performance profiling)

---

## Building the Compiler

### Quick Build

```bash
# Clone repository
git clone https://github.com/tsotchke/eshkol.git
cd eshkol

# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Executables in build/
ls build/eshkol-*
# eshkol-run   - AOT compiler
# eshkol-repl  - JIT REPL
```

### Platform-Specific Builds

#### Ubuntu/Debian

```bash
# Install dependencies
sudo apt-get install -y \
    llvm-14 llvm-14-dev clang-14 \
    cmake build-essential

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

#### macOS

```bash
# Install via Homebrew
brew install llvm@14 cmake

# Configure with LLVM path
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR=$(brew --prefix llvm@14)/lib/cmake/llvm

# Build
cmake --build build -j$(sysctl -n hw.ncpu)
```

### Build Types

```bash
# Debug build (with assertions, no optimizations)
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Release build (optimized, no debug info)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# RelWithDebInfo (optimized + debug symbols)
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
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

# Compile (generates ./hello executable)
build/eshkol-run hello.esk

# Run
./hello
# Output: Hello, Eshkol!
```

### Compilation Process

`eshkol-run` performs these steps:

1. **Macro expansion** - Process `define-syntax` and `syntax-rules`
2. **S-expression parsing** - Build AST from source
3. **HoTT type checking** - Infer types, emit warnings
4. **LLVM IR generation** - Translate AST to LLVM intermediate representation
5. **LLVM optimization** - Apply -O3 optimization passes
6. **Native code generation** - LLVM backend produces machine code
7. **Executable creation** - Link and produce final binary

### Multi-File Projects

Eshkol uses `require` for modules:

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

Compile main file (automatically loads dependencies):

```bash
build/eshkol-run main.esk
./main
# Output: 25
```

---

## REPL Usage

### Starting the REPL

```bash
build/eshkol-repl
```

### REPL Session

```scheme
eshkol> (define (factorial n)
          (if (<= n 1) 1 (* n (factorial (- n 1)))))

eshkol> (factorial 5)
120

eshkol> (map factorial (list 1 2 3 4 5))
(1 2 6 24 120)

eshkol> (gradient (lambda (x) (* x x)) 3.0)
6.0

eshkol> (exit)
```

### REPL Commands

- **`(exit)`** or **Ctrl+D** - Exit REPL
- Definitions persist across inputs
- JIT compiles each expression immediately

---

## Debugging

### Compilation Debugging

Check what LLVM IR is generated:

```bash
# Set environment variable for LLVM IR dump
export ESHKOL_DUMP_IR=1
build/eshkol-run program.esk
# Prints LLVM IR to stderr before compilation
```

### Runtime Debugging

**With GDB:**

```bash
# Compile (executables include debug symbols by default)
build/eshkol-run program.esk

# Debug
gdb ./program
(gdb) break main
(gdb) run
(gdb) next
(gdb) print variable
(gdb) backtrace
```

**With LLDB (macOS):**

```bash
lldb ./program
(lldb) breakpoint set --name main
(lldb) run
(lldb) step
(lldb) print variable
(lldb) bt
```

### Memory Debugging

**Valgrind (Linux):**

```bash
valgrind --leak-check=full --show-leak-kinds=all ./program
```

**Address Sanitizer:**

```bash
# Rebuild compiler with ASan
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer"
cmake --build build

# Compile and run program
build/eshkol-run program.esk
./program
# ASan will report memory errors with stack traces
```

---

## Common Issues

### Issue: LLVM not found

```
CMake Error: Could not find LLVM
```

**Solution:**

```bash
# Ubuntu/Debian
sudo apt-get install llvm-14-dev

# macOS
brew install llvm@14
export LLVM_DIR=$(brew --prefix llvm@14)/lib/cmake/llvm

# Then reconfigure
cmake -B build -DLLVM_DIR=$LLVM_DIR
```

### Issue: Compilation fails with type warnings

```
Warning: Type mismatch at line 10
Expected: integer
Found: real
```

**Explanation:** Eshkol type checker emits warnings, not errors. Code still compiles. To fix warnings, add explicit type annotations or conversions.

### Issue: Executable not created

```
error: linker failed
```

**Solution:** Check that all required libraries are installed and LLVM version is 14+.

### Issue: Segmentation fault at runtime

**Debugging steps:**

1. Run with GDB: `gdb ./program`
2. Set breakpoint: `(gdb) break main`
3. Run: `(gdb) run`
4. When it crashes: `(gdb) backtrace`
5. Inspect variables: `(gdb) print var_name`

---

## See Also

- [Getting Started](GETTING_STARTED.md) - Installation and first programs
- [Compiler Architecture](COMPILER_ARCHITECTURE.md) - LLVM backend internals
- [Type System](TYPE_SYSTEM.md) - Understanding type warnings
- [Memory Management](MEMORY_MANAGEMENT.md) - Arena allocation, debugging leaks
