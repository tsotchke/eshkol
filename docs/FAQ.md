# Frequently Asked Questions

## Installation

### What are the prerequisites?

- **LLVM 21** (the default pinned major for lite/native builds; the source
  compiles against LLVM **18 through 24** — see below)
- **CMake 3.14+** (build system)
- **C++20 compiler** (GCC 12+, Clang 15+, or MSVC 2022)
- **Ninja** (recommended, but Make works too)
- No image codec packages: PNG, JPEG, BMP and WebP read/write are built into the
  runtime on every platform

### How do I install LLVM 21?

**macOS:**
```bash
brew install llvm@21
```

**Ubuntu/Debian:**
```bash
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null
echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-21 main" | sudo tee /etc/apt/sources.list.d/llvm.list
sudo apt update && sudo apt install llvm-21 llvm-21-dev pkg-config
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

### Can I use a different LLVM release?

Yes, within a range. The build **pins one major** so a mixed toolchain cannot
be assembled by accident, and that major defaults to **21**; the compiler source
itself builds against **LLVM 18 through 24**, with the version shims in
`inc/eshkol/backend/llvm_compat.h`. Select a different one at configure time:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DESHKOL_REQUIRED_LLVM_MAJOR=24
```

A major outside 18-24 is unsupported. The separately documented bundled
XLA/StableHLO toolchain is its own case: that path carries its own LLVM/MLIR
stack.

### Do I need to install anything to try Eshkol?

No. Visit [eshkol.ai](https://eshkol.ai) for a browser-based REPL that runs entirely in WebAssembly. No installation required.

---

## Language

### What is the difference between a vector and a tensor?

**Vectors** (`(vector 1 2 3)` or `#(1 2 3)`) are heterogeneous — each element is a 16-byte tagged value that can hold any type (integers, strings, closures, etc.). They're Scheme's standard vector type.

**Tensors** (`(make-tensor '(3 4) 0.0)`) are homogeneous arrays of doubles — each element is 8 bytes. They support N-dimensional shapes, automatic GPU dispatch, and integrate with the automatic differentiation system. Use tensors for numerical computing; use vectors for general-purpose collections.

### Is Eshkol R7RS compatible?

Yes, Eshkol implements a substantial subset of R7RS Scheme including: `lambda`, `define`, `let`/`let*`/`letrec`, `if`/`cond`/`case`/`when`/`unless`, `call/cc`, `dynamic-wind`, `guard`/`raise`, `values`/`call-with-values`, `syntax-rules`, `do`, tail call optimization, and the full numeric tower (integers, rationals, reals, complex numbers).

Some R7RS features are extended: `define-library` and R7RS `import` forms lower through Eshkol's existing module system, including `only`, `except`, `rename`, and `prefix` import sets. The type system adds HoTT-based gradual typing beyond what R7RS specifies.

R7RS 7.1.1's third `<identifier>` production — vertical-line symbols like `'|weird sym|` and the empty symbol `'||` — is supported by both the native and bytecode-VM readers and writers. See [Complete Language Specification §2.1.6](COMPLETE_LANGUAGE_SPECIFICATION.md#216-symbol-interned-symbol).

### What does "homoiconic" mean?

Code is data. Every Eshkol expression is an S-expression — a data structure that the program can read, manipulate, and evaluate at runtime. Lambda closures retain their symbolic S-expression forms even after compilation. This enables metaprogramming at native speed.

### How does automatic differentiation work?

Eshkol provides three modes:

1. **Symbolic** (`diff`): Compile-time expression transformation. Zero runtime cost.
2. **Forward-mode** (`derivative`): Dual number arithmetic. Efficient for functions with few inputs.
3. **Reverse-mode** (`gradient`): Computational graph construction. Efficient for functions with many inputs (e.g., neural networks).

All three are compiler primitives, not library functions. The compiler knows how to differentiate your code. `gradient` is exact reverse-mode AD however the callable is reached — named directly, passed through a function parameter, wrapped, or applied in curried form; `(gradient f pt)` through a wrapper and the curried `((gradient f) pt)` are byte-identical to the direct call, with no finite-difference fallback in the gradient path.

### What is the consciousness engine?

22 compiled primitives spanning three paradigms:

1. **Logic programming**: `unify`, `walk`, `make-substitution`, `make-kb`, `kb-assert!`, `kb-query` — Prolog-style symbolic reasoning
2. **Active inference**: `make-factor-graph`, `fg-add-factor!`, `fg-infer!`, `free-energy`, `expected-free-energy` — probabilistic graphical models with belief propagation
3. **Global workspace**: `make-workspace`, `ws-register!`, `ws-step!` — attention-based module competition (Baars 1988, Bengio 2017)

These are compiled to native code, not interpreted. Belief propagation runs at C speed.

---

## Networking & Systems

### Can Eshkol talk to the network?

Yes, since v1.4.0-connection. `core.net` provides TCP and UDP sockets and Unix
domain sockets; `core.tls` provides TLS client and server contexts and
sessions; there is an HTTP/1.1 client (`http-request`) and a non-blocking
HTTP/1.1 server on the portable event loop, plus WebSocket with RFC-validated
framing and bounded Server-Sent Events streaming. `core.http_server` adds a
pure-Scheme request parser and router with `/health`, `/ready` and `/metrics`
fallbacks.

```scheme
(require core.net)
(call-with-tcp-connection "example.org" 80
  (lambda (sock)
    (tcp-write sock "GET / HTTP/1.1\r\nHost: example.org\r\n\r\n")
    (display (tcp-read sock 4096))))
```

### How do I avoid leaking a socket?

Two ways, and they compose. Use the scoped forms — `call-with-tcp-connection`,
`call-with-tcp-listener`, `call-with-tcp-accept`, `call-with-udp-socket`,
`call-with-tls-client` — which close the handle on normal return *and* on a
raise. Then compile with `--strict-resources` (ADR-0056): the compiler tracks
sockets, TLS sessions, files and regions use-exactly-once, so an unclosed or
twice-closed handle is a compile-time error rather than something you discover
in production.

### How do network errors surface?

As structured conditions, not bare errno values. `network-error?` identifies
them and `network-error-code`, `network-error-operation` and
`network-error-reason` read the detail; TLS has the matching `tls-error?`
family. Check `(tls-available?)` before assuming a build carries TLS instead of
silently falling back to plaintext.

### Is anything guaranteed about distributed results?

Socket collectives (broadcast, all-reduce, all-gather, reduce-scatter,
barrier) reduce in rank order, so the same ranks and the same inputs give the
same bits on every run. Durable writes go through a write-ahead log and commit
atomically: a checkpoint lands whole, or the previous state stays intact.
Subprocesses run confined under Darwin Seatbelt or Linux seccomp with Landlock,
and a platform with no confinement mechanism refuses the launch rather than
running unconfined.

---

## Mathematics & Verification

### What is the mathematics programme?

A set of libraries (v1.4.0-connection) that state mathematics the way a
mathematician states it — a site, a sheaf, a topos, a ring — and decide each
claim exactly, by enumeration and exact linear algebra, never by sampling:
`core.sheaf`, `core.topos`, `core.ring`, `core.category`, `core.simplicial`,
`core.homological`, `core.theory`, `core.manifold`, `core.curvature`,
`core.exact_linalg` and the Stochastic Binary Lambda Calculus in `core.sblc`.
A reader's guide is in [guide/MATHEMATICS.md](guide/MATHEMATICS.md).

### Are the proofs actually checked?

Yes. 395 top-level theorems across 12 standalone proof files are
machine-checked in Lean 4.12.0 in CI, with 0 `sorry` and 0 custom axioms, and
replayed independently (ADR-0061, ADR-0063). 86 example programs are verified
alongside them.

### What is NOT in v1.4.0-connection?

Consciousness Engine v2 (CE-12) and the Eshkol OS track (K1-K6) are later
work, and multi-host distributed computing beyond single-host socket
collectives is still ahead. Backend parity is a documented subset, not complete
parity: the VM parity manifest carries 997 rows, 322 of them `gap`, and each
construct's engine coverage is listed in
[FEATURE_MATRIX.md](FEATURE_MATRIX.md).

---

## Performance

### Is Eshkol fast?

Eshkol compiles to native machine code via LLVM. Arithmetic and tensor operations are competitive with C. The arena memory allocator runs in O(1) constant time with zero garbage collection pauses.

### Does Eshkol use garbage collection?

No. Eshkol uses Ownership-Aware Lexical Regions (OALR) — arena-based allocation where memory is freed deterministically at scope boundaries. This provides microsecond-scale worst-case guarantees suitable for real-time systems. As of v1.3.4, a resident tick/daemon loop reclaims its per-iteration garbage automatically — even when it mutates persistent state every iteration — so RSS stays flat without any explicit `with-region` annotation (`with-region` remains available for scratch regions but is no longer required). One deliberate exclusion: a loop body containing a `gradient` op, a `set!` or a `tensor-set!` is disqualified from the automatic nursery by design, and an AD training step trips all three — scope those steps with `with-region` or a `region-open`/`region-close` handle.

That "no" is about Eshkol's own semantics: Eshkol values are never traced and no Eshkol program pauses for a collection. It is not a claim that no collector may run anywhere in the process. A garbage-collected guest language can be **hosted** on regions — the guest heap is a region with its own collector inside it, on a declared byte budget, with Eshkol tracing nothing. Planned; see [ADR-0011](design/adr/0011-guest-collector-adapter.md).

The *automatic* nursery above is a **native-engine** capability (`eshkol-run`, JIT and AOT). The **bytecode VM** has no nursery, but `with-region` reclaims there too as of the Stage-1 region evacuator — measured flat at 26 MB across 1 000/4 000/16 000 iterations of the same fixture, against 793 MB with the evacuator disabled and 704 MB for an unwrapped control (commit `487c2a62`, #461). So a resident VM loop needs an explicit `with-region` where a native one does not, and outside a region the VM heap still grows monotonically (it says so when the growth crosses a heap budget). The one region surface the VM does not reclaim is the handle API (`region-open` / `region-close`), which announces that at the point of use — see [memory model](reference/runtime/memory-model.md#which-engine-reclaims).

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
- Windows (native x86_64 and ARM64 via Visual Studio 2022 + LLVM 21). x86_64 is covered by the Lite, XLA and CUDA lanes; ARM64 builds and passes the hosted `windows-arm64-lite` and `windows-arm64-xla` CI lanes and ships those two packages, with no mesh/self-verified ARM64 hardware coverage behind them. CUDA packages are x86_64-only.
- WebAssembly (browser)

### Can I deploy Eshkol programs as standalone binaries?

Yes:
```bash
eshkol-run program.esk -o program
./program  # standalone executable, no runtime dependency
```

### Can Eshkol run in the browser?

Yes. Eshkol compiles to WebAssembly. The project website ([eshkol.ai](https://eshkol.ai)) is written in Eshkol. Its browser bytecode VM evaluates the REPL and runnable documentation blocks, and tensor operations use WebGPU when a suitable adapter is available. A REPL answer is emitted as a complete line; `(display "hi")` alone remains a fragment until `(newline)`.
