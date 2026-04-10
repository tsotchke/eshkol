# Frequently Asked Questions

## Installation

### What are the prerequisites?

- **LLVM 21** (required for lite/native builds)
- **CMake 3.14+** (build system)
- **C++20 compiler** (GCC 12+, Clang 15+, or MSVC 2022)
- **Ninja** (recommended, but Make works too)

### How do I install LLVM 21?

**macOS:**
```bash
brew install llvm@21
```

**Ubuntu/Debian:**
```bash
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null
echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-21 main" | sudo tee /etc/apt/sources.list.d/llvm.list
sudo apt update && sudo apt install llvm-21 llvm-21-dev
```

**Windows:**
Use native Visual Studio 2022 plus the official LLVM 21 SDK:
```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T ClangCL `
  -DCMAKE_BUILD_TYPE=Release `
  -DLLVM_DIR="C:/Program Files/LLVM/lib/cmake/llvm"
```

### The build fails with "llvm-config not found" or "LLVM_DIR not set"

Make sure LLVM 21 is on your PATH on macOS/Linux:
```bash
# macOS
export PATH="/opt/homebrew/opt/llvm@21/bin:$PATH"

# Linux (if installed to /usr/lib/llvm-21)
export PATH="/usr/lib/llvm-21/bin:$PATH"
```

On native Windows, set `LLVM_DIR` to the LLVM 21 SDK's `lib/cmake/llvm` directory before configuring.

Then rebuild:
```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Can I use an older LLVM release?

Lite/native builds are pinned to LLVM 21. Older or intermediate LLVM releases are unsupported. The only expected exception is a separately documented bundled XLA/StableHLO toolchain when that path carries its own LLVM/MLIR stack.

### Do I need to install anything to try Eshkol?

No. Visit [eshkol.ai](https://eshkol.ai) for a browser-based REPL that runs entirely in WebAssembly. No installation required.

---

## Language

### What is the difference between a vector and a tensor?

**Vectors** (`(vector 1 2 3)` or `#(1 2 3)`) are heterogeneous — each element is a 16-byte tagged value that can hold any type (integers, strings, closures, etc.). They're Scheme's standard vector type.

**Tensors** (`(make-tensor '(3 4) 0.0)`) are homogeneous arrays of doubles — each element is 8 bytes. They support N-dimensional shapes, automatic GPU dispatch, and integrate with the automatic differentiation system. Use tensors for numerical computing; use vectors for general-purpose collections.

### Is Eshkol R7RS compatible?

Yes, Eshkol implements a substantial subset of R7RS Scheme including: `lambda`, `define`, `let`/`let*`/`letrec`, `if`/`cond`/`case`/`when`/`unless`, `call/cc`, `dynamic-wind`, `guard`/`raise`, `values`/`call-with-values`, `syntax-rules`, `do`, tail call optimization, and the full numeric tower (integers, rationals, reals, complex numbers).

Some R7RS features are extended: the `define-library` system is planned for v1.3, and the type system adds HoTT-based gradual typing beyond what R7RS specifies.

### What does "homoiconic" mean?

Code is data. Every Eshkol expression is an S-expression — a data structure that the program can read, manipulate, and evaluate at runtime. Lambda closures retain their symbolic S-expression forms even after compilation. This enables metaprogramming at native speed.

### How does automatic differentiation work?

Eshkol provides three modes:

1. **Symbolic** (`diff`): Compile-time expression transformation. Zero runtime cost.
2. **Forward-mode** (`derivative`): Dual number arithmetic. Efficient for functions with few inputs.
3. **Reverse-mode** (`gradient`): Computational graph construction. Efficient for functions with many inputs (e.g., neural networks).

All three are compiler primitives, not library functions. The compiler knows how to differentiate your code.

### What is the consciousness engine?

22 compiled primitives spanning three paradigms:

1. **Logic programming**: `unify`, `walk`, `make-substitution`, `make-kb`, `kb-assert!`, `kb-query` — Prolog-style symbolic reasoning
2. **Active inference**: `make-factor-graph`, `fg-add-factor!`, `fg-infer!`, `free-energy`, `expected-free-energy` — probabilistic graphical models with belief propagation
3. **Global workspace**: `make-workspace`, `ws-register!`, `ws-step!` — attention-based module competition (Baars 1988, Bengio 2017)

These are compiled to native code, not interpreted. Belief propagation runs at C speed.

---

## Performance

### Is Eshkol fast?

Eshkol compiles to native machine code via LLVM. Arithmetic and tensor operations are competitive with C. The arena memory allocator runs in O(1) constant time with zero garbage collection pauses.

### Does Eshkol use garbage collection?

No. Eshkol uses Ownership-Aware Lexical Regions (OALR) — arena-based allocation where memory is freed deterministically at scope boundaries. This provides microsecond-scale worst-case guarantees suitable for real-time systems.

### How does GPU dispatch work?

Tensor operations dispatch through a four-tier hierarchy based on operand size:
1. **SIMD** (< 64 elements): SSE/AVX/NEON micro-kernels
2. **cBLAS/Accelerate** (64-100K elements): Optimized BLAS libraries
3. **Metal/CUDA** (> 100K elements): GPU acceleration
4. **XLA** (very large or dynamic shapes): StableHLO/MLIR

This is transparent to user code — the same expression compiles to different backends automatically.

---

## Deployment

### What platforms does Eshkol support?

- macOS (Apple Silicon ARM64, Intel x86_64)
- Linux (x86_64, ARM64)
- Windows (native x86_64 via Visual Studio 2022 + LLVM 21)
- WebAssembly (browser)

### Can I deploy Eshkol programs as standalone binaries?

Yes:
```bash
eshkol-run program.esk -o program
./program  # standalone executable, no runtime dependency
```

### Can Eshkol run in the browser?

Yes. Eshkol compiles to WebAssembly. The project website ([eshkol.ai](https://eshkol.ai)) is itself written in Eshkol — 1,500+ lines compiled to a 502KB WASM binary. A bytecode VM REPL also runs in the browser for interactive evaluation.
