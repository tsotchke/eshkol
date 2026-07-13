# Eshkol Structure Notes

Independent architecture analysis of `tsotchke/eshkol` (forked to
`waefrebeorn/eshkol_audit` — note the `_audit` suffix — upstream-synced). Written 2026-07-11 during the
cross-validation audit that produced `cross-validation/`.

> Devil's-advocate reminder: this is a *large* repo (157 KB CMakeLists, 79 test
> dirs, 1,648 commits). Marketing docs (README/ANNOUNCEMENT/ROADMAP) describe an
> ambitious "consciousness engine" + HoTT foundations. The **code** is a
> Scheme-dialect compiler with an LLVM backend + an AD system + a native VM.
> Treat docs as intent; treat `lib/`, `inc/`, `exe/` as fact.

## 1. What it actually is

A **Scheme-based language ("Eshkol")** for mathematical computing:
- Functional Lisp syntax (homoiconic `.esk` files).
- **Native automatic differentiation** (forward + reverse mode) as a compiler
  primitive — the headline feature.
- Compiles to native via **LLVM** (`lib/backend/llvm_codegen.cpp`).
- Also has a **bytecode VM** (`eshkol_vm.c`, `vm_native.c`) that runs a 64-opcode
  core with dual-number (AD) propagation; compiles to **WebAssembly** (59 DOM
  bindings) for the web REPL.

## 2. Directory map (real, from disk)

```
eshkol/
├── exe/                     # entry points (C++ main files)
│   ├── eshkol-repl.cpp      # REPL/JIT driver
│   ├── eshkol-run.cpp       # batch runner
│   └── eshkol-server.cpp    # HTTP server (web platform)
├── lib/
│   ├── core/                # language runtime + math builtins (C++/C/.esk)
│   │   ├── ast.cpp, parser..  # AST + frontend parsing (C++)
│   │   ├── llvm_codegen.cpp   # LLVM IR emission (THE biggest file, 325 commits)
│   │   ├── autodiff_codegen.cpp  # AD → IR
│   │   ├── eshkol_vm.c, vm_native.c  # bytecode VM + native dispatch
│   │   ├── bignum.cpp, logic.cpp, crypto_primitives.c, image_io.c, json.esk ...
│   │   ├── ad/              # AD tape + builtins
│   │   ├── control/ functional/ logic/ list/ data/  # Scheme libraries
│   │   └── manifold.esk     # ★ Riemannian geometry (Euclidean/Poincaré/sphere)
│   ├── frontend/            # parser.cpp (82 commits) — source → AST
│   ├── backend/             # codegen: LLVM + VM (heaviest dir)
│   ├── math/ math.esk       # numeric builtins (Scheme)
│   ├── ml/                  # ML primitives
│   ├── tensor/              # tensor ops
│   ├── quantum/             # quantum simulation glue (NEW: moonlab integration design, #260)
│   ├── random/ signal/ types/ bridge/ ffi/ repl/ agent/ web/ test/
│   └── stdlib.esk          # standard library
├── inc/eshkol/              # public C/C++ headers (eshkol.h, llvm_backend.h, ...)
├── tests/                   # 79 test directories (unit + integration)
├── docs/                    # huge doc tree (architecture/, design/, platform/, ...)
├── bindings/python/         # Python FFI
├── bench/ benchmarks/       # perf corpora
├── docker/ nix/ packaging/  # build/deploy (LLVM 21, CUDA, XLA variants)
├── web/ site/               # the Eshkol-written website
├── scripts/ tools/          # dev tooling
├── CMakeLists.txt           # 157 KB — the build graph
├── .gitlab-ci.yml RELEASE_NOTES.md CHANGELOG.md ROADMAP.md ANNOUNCEMENT.md
└── LICENSE (MIT)
```

## 3. Build (fact, from CMakeLists + CI)

- **Prerequisites:** CMake ≥ 3.14, **LLVM 21** (`LLVM_MAJOR: '21'` in CI),
  C17/C++20 toolchain.
- **Commands:**
  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build
  cmake --build build --target eshkol-repl
  ```
- **Why it won't build in this WSL env:** no LLVM 21 (only the LLM toolchain is
  present). The `lib/core/manifold.esk` math is *pure Scheme* and runs under the
  REPL/JIT/AOT paths, but building the REPL needs the full LLVM toolchain.
- **Web build** targets WASM (works without a native LLVM install in principle,
  but still routed through the same codegen).

## 4. The math we cross-validated (`lib/core/manifold.esk`)

A self-contained constant-curvature differential-geometry library in Scheme:
- Manifolds: Euclidean (K=0), **Poincaré ball (K=−1)**, Spherical (K=+1).
- Ops: `manifold-exp-map`, `manifold-log-map`, `manifold-distance`,
  `manifold-parallel-transport`, **`manifold-christoffel`**,
  `manifold-sectional-curvature`, `manifold-ricci-curvature`,
  `manifold-scalar-curvature`, `manifold-riemann-tensor`.
- Product manifolds (Euclidean × Poincaré) supported per-factor.
- Its header explicitly states the Poincaré formulas "agree with the GRR/qLLM
  dylib geometry up to convention; the dylib is the source of truth." → i.e. the
  `.esk` forms are *mirrors* of a separate C dylib, with known convention drift.

### ★ Devil's-advocate finding (see cross-validation/REPORT.md)

`manifold-exp-map` puts the conformal factor `lam = 2/(1−|p|²)` **inside** the
`tanh`: `factor = tanh(0.5·lam·|v|)/|v|`. Cross-validation against the manifold's
own geodesic invariant `dist(p, exp_p(v)) = const·|v|` shows it **fails for
p ≠ 0** (ratio drifts >0.15 across random bases). WuBuMath's *own* `exp_0` (single
tanh, no lam) is the consistent one. Documented verbatim + reproducible in
`cross-validation/`. The analytic Christoffel symbols, however, are **correct**
and agree with WuBuMath's independent RK4 geodesic acceleration.

## 5. Where our work plugs in

`waefrebeorn/WuBuMath` already ported + validated the *math* from tsotchke:
- `libirrep` → SO(3) exp/log/geodesic, Wigner 3j / Clebsch-Gordan.
- `quantum_geometric_tensor` → generic RK4 geodesic integrator.
- `moonlab` → SU(2)_k anyon model (fusion, R-matrix, quantum 6j).
- `eshkol/manifold.esk` → Poincaré/sphere geometry **cross-validated** here.

The fork keeps `manifold.esk` verbatim (so the bug is reproducible) and adds
`cross-validation/` with a standalone C port + test that proves agreement with
WuBuMath and flags the exp-map discrepancy. This is the "audit + notes + agents md"
deliverable requested on the fork.

## 6. Open questions / things NOT verified

- The "consciousness engine" (22 primitives), HoTT foundations, and web platform
  are documented but were **not** exercised (need a full LLVM 21 build + runtime).
- `lib/backend/vm_geometric.c` **does exist** (native IDs 804–861). It is a
  *dispatcher*: with `ESHKOL_GEOMETRIC_ENABLED` it calls the external
  `semiclassical_qllm` library (NOT vendored in this repo); otherwise it uses a
  portable fallback holding only a scalar `{type, dim, curvature}` — no metric /
  Christoffel / geodesic code. The actual Riemannian math is in `lib/core/manifold.esk`.
  So the geometric-VM claim is **under-substantiated by the shipped default build**,
  not absent from the tree.
- `lib/quantum/` is real committed code (`quantum_rng.c/.h`, `quantum_rng_wrapper.c/.h`),
  not a design stub. Its entropy/"quantum" claims are debunked in the sibling fork
  `waefrebeorn/quantum_rng_audit`. PR #260 adds moonlab-integration *design* docs on top.
