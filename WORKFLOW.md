# WORKFLOW — A Rigorous but Inviting Pipeline for tsotchke's Work

This fork exists to cross-validate `tsotchke/*` math against an independent
implementation (`waefrebeorn/WuBuMath`) and to demonstrate a workflow that makes
"proven / quantum / non-deterministic" claims *falsifiable*. The goal is
collaboration, not confrontation: every finding is reproduced, documented, and
offered back as a PR.

## The pipeline (3 gates, all mandatory)

### Gate 1 — Read the source, not the README
Marketing docs (README / ANNOUNCEMENT / ROADMAP) describe *intent*. Code
describes *fact*. We grep the actual functions, read the bodies, and never trust a
claim that isn't backed by a function we can compile.

> eshkol: "consciousness engine" / HoTT are doc intent, unexercised by code.
> quantum_rng: "63.999872 bits/sample" is hardcoded text, no code computes it.

### Gate 2 — Reproduce every claim numerically (C contract)
Standalone C test, no heavy deps, that checks the *geometric invariant* or
*statistical property* the claim implies.

- eshkol: `dist(p, exp_p(v)) = const·|v|` must hold. It fails for p≠0 → bug.
- quantum_rng: same seed → same stream must hold. It fails (seed ignored) → not a
  reproducible PRNG.

These live in `cross-validation/` (eshkol) and `audit/` (quantum_rng). Build +
run is one `gcc` line. No LLVM, no CUDA.

### Gate 3 — Formalize the true claims (Lean 4)
Theorems that survive Gate 2 get a Lean proof in `waefrebeorn/WuBuMath/lean/WubuProofs/`
(CI: `lake build` + sorry-audit, 0 sorry enforced). This closes the loop from
"it ran" to "it's proven."

> Already proven (WuBuMath, 0 sorry): `PoincareBall` (dist formula, conformal
> metric, curvature scaling), `FiberBundle` (SO(3) closure, principal bundle),
> `NestedHyperbolicSpaces` (nested-balls, φ-curvature), `PowerTower` (π↑↑4 bounds).

## How tsotchke's work plugs in (concrete next steps)

| tsotchke repo | What's true | What to do | WuBu artifact to build on |
|---|---|---|---|
| `eshkol/manifold.esk` | Christoffel/curvature correct; exp-map buggy | PR fix + Lean `exp_map_geodesic_invariant` | `PoincareBall.lean`, `wubu_poincare_geom.c` |
| `libirrep` | SO(3)/SE(3) exp-log, CG/Wigner — production-grade | Port proofs to `FiberBundle.lean` | existing SO(3) commutation proof |
| `quantum_geometric_tensor` | RK4 geodesic integrator (real) | Port + prove geodesic-eq contract | `wubu_manifold.c` (already done) |
| `moonlab` | SU(2)_k anyon fusion/R/6j (real) | Port + prove fusion invariants | `wubu_anyon.c` (already done) |
| `quantum_rng` | competent classical mixer; false "quantum" claim | PR: fix build, real seed, measure entropy | WuBu RNG audit + Lean min-entropy lemma |

## Invitation template (rigorous but welcoming)

> "We cross-validated your `<repo>` against an independent implementation.
> Your `<X>` is correct. We found `<one specific bug>` — reproduced here:
> `<path>`. Fix + test attached. Want to co-author a PR? We can also formalize
> `<invariant>` in Lean as a shared proof artifact."

## Principles (from the Mind Palace DA loop)
- **No theorem without a check** — Lean proof OR numeric contract, never neither.
- **Keep upstream verbatim** — document bugs, don't silently rewrite.
- **Reproduce, don't assert** — every finding has a one-command repro.
- **Assume good faith** — "quantum" is analogy until proven malice; fix is labeling.
- **Upstream PRs, not forks-as-truth** — the fork is evidence; the PR is the gift.

## Cross-reference
- eshkol-audit: `.hermes/mind-palace/{prestige_prompt,goal-mantra,state,overnight-map}.md`,
  `plans/devils_advocate_v1.md`, `cross-validation/REPORT.md`, `ESHKOL_STRUCTURE.md`.
- quantum_rng-audit: `.hermes/mind-palace/{...}`, `plans/devils_advocate_v1.md`,
  `AUDIT.md`, `audit/determinism_test.c`.
- Reference impl: `waefrebeorn/WuBuMath` (`lean/WubuProofs/`, `src/math/`, `VALIDATION.md`).
