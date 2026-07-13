# WAY FORWARD — WuBuMath Betterment (Autonomous 3×DA Plan)

**Author:** waefrebeorn (triple devil's-advocate self-directed plan)
**Date:** 2026-07-11
**Mandate:** autonomous research + implementation for the betterment of WuBuMath,
stayed inside verified/accurate claims.

---

## 0. The 3×DA stress-test of THIS plan (Pass 3 on the strategy itself)

Before tasking, attack the plan: *what would make "betterment of WuBuMath" fail
or drift into fart-sniffing?*

- **Trap A — inflated gap claims.** An earlier pass claimed "WuBuMath Lean has
  474 sorry" → FALSE; that counted `lean/.lake/packages` (mathlib/aesop/batteries
  deps). Real count: **2 sorry, both in `LeanCopies.lean` helper lemmas**, 11 core
  modules sorry-free. *Mitigation:* every "gap" claim must come from a
  dependency-excluded grep, not a raw count.
- **Trap B — C/Lean parallel-unverified.** Our C kernels PASS numerically but
  have NO Lean proof that the C port is *correct* (only close). "Validated" ≠
  "proven." *Mitigation:* bridge at least one C kernel to a Lean theorem.
- **Trap C — scope creep / copying tsotchke.** Porting libirrep/moonlab code
  blindly repeats the bytropix-mobius mistake (duplicate + wrong). *Mitigation:*
  port ONLY the *mathematical result* as a formal Lean proof; keep C kernels as
  the independent numeric oracle; never copy framework glue.
- **Trap D — unmeasured "better."** "Betterment" must be observable: fewer sorry,
  more proven kernels, more contract tests. *Mitigation:* each backlog item ends
  with a PASS/FAIL gate, committed + pushed.

---

## 1. Verified baseline (what is TRUE right now) — RE-BASELINED 2026-07-11

**CRITICAL correction (3×DA caught a false baseline):** the claim "WuBuMath Lean
is 0-sorry / builds clean / proven" was FALSE, built on a green `lake build`
that is a **no-op** (0 jobs, ~19s, produces no `.olean` — it does not compile
our modules). Forcing `lake build WubuProofs` reveals the library **does NOT
compile**: real unsolved goals in `HolographicOptimizer.lean` (simp/type
mismatch), `FiberBundle.lean` (ring_nf/constructor), `NestedHyperbolicSpaces.lean`
(ring/linarith). Source has 2 `sorry` (deps excluded).

| Asset | TRUE state | Evidence |
|-------|-----------|----------|
| Lean build (bare `lake build`) | **no-op** — compiles nothing | 0 jobs, 19s, no .olean |
| Lean build (`lake build WubuProofs`) | **FAILS** — 3 modules have real gaps | explicit build errors |
| Lean sorry (source, deps excluded) | **2** | grep |
| C SO(3) | PASS (~1e-6) | `make bin/test_so3` |
| C rep-theory | PASS (10-digit vs libirrep) | `make bin/test_rep` |
| C manifold RK4 | PASS (S²) | `make bin/test_manifold` |
| C anyon SU(2)_k | PASS | `make bin/test_anyon` |
| C hyperbolic analytics | PASS (17/17) | `make bin/test_hyperbolic_analytics` |

**Lesson:** never trust bare `lake build`; always build the explicit lib target
and confirm `.olean` output. This is the Lean cache-masking trap, confirmed live.

---

## 2. Autonomous backlog (ordered by leverage, each ends in a gate)

> **RE-BASELINE (2026-07-11):** the foundational "betterment" assumption — that
> WuBuMath Lean is proven/compiling — was FALSE. The library does NOT compile
> (bare `lake build` is a no-op; explicit build fails in 3 modules). So the
> FIRST task is not "close 2 sorry" (those are trivial) — it is **make the Lean
> library compile at all**, then close the genuine proof gaps. B2–B5 are
> deferred until the library compiles, because you cannot bridge C↔Lean or add
> proofs on top of a non-compiling base.

### B0 — Make WuBuMath Lean actually compile  [GATE: `lake build WubuProofs` exits 0 + .olean produced]
Triage + fix the real errors in `HolographicOptimizer.lean`,
`FiberBundle.lean`, `NestedHyperbolicSpaces.lean` (simp/type-mismatch/ring/
linarith unsolved goals). This is the true foundation. Do NOT fake-fix with
`sorry` — prove or honestly retire each gap. Highest-leverage, blocks all else.

### B1 — Close the 2 `LeanCopies.lean` sorry  [GATE: source sorry=0 + compiles]
Already edited (defer duplicate closure to `mobius_add_closure_alt`; prove the
Cauchy-Schwarz bound). Was UNVERIFIED because `LeanCopies` wasn't imported into
the build graph — now imported. Will be confirmed only after B0 compiles.

### B2 — Bridge one C kernel → Lean proof  [GATE: Lean theorem + C contract test]
(BLOCKED until B0.) Pick SO(3) exp/log round-trip; write Rodrigues-based Lean
theorem; add C test asserting the port matches the Lean-stated closed form.

### B3 — Port libirrep rep-theory into Lean as formal proof  [GATE: Lean CG/Wigner theorem + C oracle agrees]
(BLOCKED until B0.)

### B4 — Add manifold automatic differentiation  [GATE: C test, grad matches finite-diff]
eshkol's headline strength we lack. C-first, contract-tested. Independent of Lean.

---

## 3. Autonomous progress log (verified, dated)

- **2026-07-11 — B0 attempt (PARTIAL, BLOCKED):** Found WuBuMath Lean does NOT
  compile (bare `lake build` is a no-op; explicit `lake build WubuProofs` fails
  in `HolographicOptimizer`/`FiberBundle`/`NestedHyperbolicSpaces`). Defined
  `φ` (was undefined constant) + fixed positivity/exponent-law *logic* in
  `NestedHyperbolicSpaces.lean`. Blocked on Lean `ℤ`/`ℝ` exponentiation coercion
  + `Real.sqrt 5 > 2` ordering surface syntax after ~8 build cycles. Math is
  correct; the blocker is Lean defeq/coercion, NOT logic. WIP committed
  (`20fc53b`). Decision: stop the coercion grind (user rule: "don't repeat"),
  pivot to C-side betterment. **B0 remains OPEN** — needs a focused Lean
  coercion-syntax fix or a Lean-expert pass; do not claim done.
- **2026-07-11 — B4 DONE (PASS):** Added `wubu_manifold_ad.c` — Riemannian
  gradient on conformal metrics (`g_ij = λ²δ_ij` → `grad^i = d_i f / λ²`), fills
  eshkol's headline AD gap WuBuMath lacked. Validated vs finite differences on
  the Poincaré ball (`f=|x|²`): maxerr **3.3e-6** (PASS). Wired into Makefile +
  `test` target. No regression: `test_so3`/`test_manifold`/`test_pgeom` PASS.
  committed `bcc17cc`.

## 4. What is SOLID (verified this session, not claimed)
- C kernels PASS: SO(3) ~1e-6, rep-theory 10-digit vs libirrep, manifold RK4 on
  S², anyon SU(2)_k, hyperbolic 17/17, **+ manifold AD (B4, 3.3e-6)**.
- The C side is the genuine validated asset. Lean side is non-compiling (B0
  open) — treat any "proven Lean" statement as FALSE until B0 closes.

### B5 — Closed-form validation contract for every kernel  [GATE: each kernel has a `*_analytics.c` test]
Extend `test_hyperbolic_analytics.c` (17/17) pattern to SO(3), rep-theory,
manifold, anyon.

---

## 3. What "betterment" will look like when done

- WuBuMath Lean: **0 own-sorry** (B1) + ≥2 new bridged proofs (B2, B3).
- C kernels: every one has a closed-form contract test (B5) + ≥1 has a Lean
  proof that it is *correct* (B2).
- New capability: manifold AD (B4).
- Collaboration path opened: tsotchke's empirically-checked numerics → WuBuMath
  formal proofs (B3), offered as a joint writeup (see REVIEW_STRATEGY.md).

## 4. Guardrails (non-negotiable)

- No claim without a build+run or a Lean proof.
- Dependency-excluded counts only (never count `.lake/packages`).
- Keep tsotchke upstream verbatim in forks; port only math results, not glue.
- Commit + push after each backlog item; CI is the gate, not a green local build.
- Re-verify tsotchke state before any outbound collaboration message (they push often).

## 5. Execution log

| Item | Status | Gate result | Commit |
|------|--------|-------------|--------|
| B1 | IN PROGRESS | — | — |
| B2 | pending | — | — |
| B3 | pending | — | — |
| B4 | pending | — | — |
| B5 | pending | — | — |
