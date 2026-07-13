# AGENTS.md

Guidance for AI agents and contributors working in this audit fork of
**`tsotchke/eshkol`**.

- **This repo (fork):** `waefrebeorn/eshkol_audit` (the GitHub slug — note the
  `_audit` suffix; do NOT call it `waefrebeorn/eshkol`).
- **Upstream:** `git@github.com:tsotchke/eshkol.git` (kept as the `upstream`
  remote, pointing at the local clone `/home/wubu/tsotchke/eshkol`).
- **Local working dir:** `/home/wubu/eshkol_audit` (matches the repo slug).

## Purpose of this fork

This is a **cross-validation audit fork**. The math implemented here (notably
`lib/core/manifold.esk`) is compared, numerically and symbolically, against
`waefrebeorn/WuBuMath` — an independent C implementation of the same geometry /
Lie-group / representation-theory primitives. See `cross-validation/`.

Rule of thumb: **do not "fix" upstream's code silently.** If you find a math
error, reproduce it in `cross-validation/`, document it in `REPORT.md`, and keep
the upstream file verbatim so the discrepancy is reproducible. Submit fixes
upstream via PR; keep the fork as the evidence trail.

## Repository at a glance

- **Language:** Scheme dialect (`.esk`) for stdlib + math; **C++** for
  frontend/codegen (`lib/frontend`, `lib/backend`, `lib/core/*.cpp`); **C** for
  the VM and runtime builtins (`lib/backend/*.c`, `lib/core/*.c`).
- **Compiler:** LLVM 21 backend (`lib/backend/llvm_codegen.cpp`) + a bytecode VM
  (`eshkol_vm.c`, `vm_native.c`) that also targets WebAssembly.
- **Headline feature:** native automatic differentiation (forward + reverse),
  exact, not numerical.
- **Math we care about:** `lib/core/manifold.esk` (Riemannian geometry:
  Euclidean / Poincaré / sphere; exp/log maps, geodesic distance, parallel
  transport, **Christoffel / sectional / Ricci / scalar / Riemann curvature**).

## Build (requires LLVM 21 — NOT available in minimal WSL)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --build build --target eshkol-repl
```
Prereqs: CMake ≥ 3.14, **LLVM 21**, C17/C++20. The web target builds WASM.
See `ESHKOL_STRUCTURE.md` §3 for the exact CI toolchain.

> In environments without LLVM 21 you **cannot build the REPL/VM**. You can still
> *read* `lib/core/manifold.esk` (pure Scheme) and port/audit its math in plain C
> under `cross-validation/` (no LLVM needed).

## How the cross-validation works

`cross-validation/` holds a **standalone C port** of `manifold.esk`'s formulas
plus a test that:
1. Checks WuBuMath's own `exp_0` + distance are self-consistent.
2. Reproduces the exp-map geodesic-invariant bug (documented, not hidden).
3. Confirms `manifold.esk`'s analytic Christoffel symbols agree with WuBuMath's
   independent RK4 geodesic acceleration.
4. Verifies analytic curvature (K=−1 / +1 / 0; scalar R; Ricci).

Build + run the cross-validation (no LLVM needed):
```bash
cd cross-validation
gcc -std=c11 -O3 -I. wubu_poincare_geom.c test_crossval.c -o crossval -lm
./crossval
```

## Repo layout quick-ref

| Path | What | Lang |
|------|------|------|
| `lib/core/manifold.esk` | ★ Riemannian geometry (audit target) | Scheme |
| `lib/core/llvm_codegen.cpp` | LLVM IR emission (largest file) | C++ |
| `lib/backend/eshkol_vm.c` | bytecode VM | C |
| `lib/frontend/parser.cpp` | source → AST | C++ |
| `lib/quantum/` | real vendored code: `quantum_rng.c/.h` + `quantum_rng_wrapper.c/.h` (moonlab quantum-sim integration; PR #260 design docs also present) | C |
| `lib/agent/quantum.esk` | agent-facing quantum bindings (added in v1.3.3) | Scheme |
| `inc/eshkol/` | public headers | C/C++ |
| `tests/` | 79 integration/unit dirs | mixed |
| `cross-validation/` | **our** standalone C port + tests | C |
| `ESHKOL_STRUCTURE.md` | detailed architecture notes | md |
| `AGENTS.md` | this file | md |

## Conventions for agents

- **Commits:** conventional (`feat:`/`fix:`/`docs:`/`test:`), small and scoped.
- **Math changes:** always add/extend a `cross-validation/` test proving the
  numeric invariant. No theorem without a check (Lean proof OR numeric).
- **Upstream sync:** `git fetch upstream && git merge upstream/main` on a branch,
  then PR. Never force-push `main`.
- **Secrets:** none. The fork has no credentials; do not add any.
- **Big files:** GitHub rejects >100 MB on push. Keep build artifacts local.

## Known pitfalls (from the audit)

- `manifold-exp-map` is NOT a consistent geodesic exponential off the origin:
  the tangent norm is chordal at the origin (`dist(0,exp_0(v))=2|v|` exactly)
  but the per-base conformal prefactor `lam(p)` is folded into the tangent
  magnitude without a matching rescale in the distance formula, so
  `dist(p,exp_p(v))/|v|` drifts 2.83→4.96 for `p≠0`. The `tanh(λ|v|/2)` form
  itself is the textbook-correct Poincaré exp map; the defect is a norm-scaling
  mismatch between exp-map and distance, not the tanh argument. See
  `cross-validation/REPORT.md` F1.
- `lib/backend/vm_geometric.c` **does exist** (native IDs 804–861). It is a
  *dispatcher*, NOT standalone geometry: when `ESHKOL_GEOMETRIC_ENABLED` is set
  it delegates to the external `semiclassical_qllm` library (NOT vendored here),
  and otherwise uses a portable fallback that stores only a scalar
  `{type, dim, curvature}` — no metric / Christoffel / geodesic math. The
  **real** Riemannian geometry (exp/log maps, Christoffel, curvature) lives in
  `lib/core/manifold.esk`. So "vm_geometric exists but the actual math is in
  manifold.esk" — not "vm_geometric doesn't exist."
- `lib/quantum/quantum_rng.c` is real committed code (not design-only). Its
  entropy/quantum claims are what `waefrebeorn/quantum_rng_audit` debunks.
- The "consciousness engine" / HoTT claims are documentation intent, not verified
  by the committed code. Treat as unproven until exercised with a full build.
