# Cross-Repo Review Strategy: tsotchke/* vs waefrebeorn/WuBuMath

**Purpose:** A fair, evidence-grounded strategy for reviewing tsotchke's 15 repos
against WuBuMath, so that any writeup we send (or keep) stays inside *verified*
claims. This is the org-scale triple devil's-advocate output (2026-07-11).

**Golden rule (anti-fart-sniffing):** grade every repo by what we *built and
ran*, never by README prose or by our own prior summaries. Two of our own prior
claims were already caught and deleted this session:
- "WuBuMath Lean is 0-sorry" → FALSE; source grep finds **474 `sorry` tactics**.
- "quantum_rng has low entropy (~7.99)" → FALSE framing; the stream is a *good
  uniform* PRNG (~7.9998 bits/byte Shannon); the defect is *unsubstantiated*
  entropy claims, not low entropy.

---

## Pass 1 — Analytical core (what is actually verified on each side)

### WuBuMath (waefrebeorn/WuBuMath, HEAD ac21a24)
C kernels, **built + run, PASS** (Makefile, not hand-glob):
| Kernel | Validation | Result |
|--------|-----------|--------|
| SO(3) exp/log/geodesic (`wubu_so3`) | round-trip + geodesic | PASS (~1e-6) |
| Rep-theory Wigner-3j / CG (`wubu_rep_theory`) | vs libirrep's own outputs | PASS (10-digit) |
| Manifold RK4 geodesic (`wubu_manifold`) | S² great-circle arc length | PASS |
| Hyperbolic analytics (`test_hyperbolic_analytics`) | closed-form contracts | PASS (17/17) |
| SU(2)_k anyon (`test_anyon`) | fusion / R-matrix / 6j | PASS |

Lean layer: **474 `sorry` tactics** — partial proofs only. Real, non-trivial
theorems exist (SO(3) group closure, bundle structure) but the "0-sorry /
fully proven" claim is RETRACTED. CI (`lean.yml`) enforces clean-rebuild +
zero-new-sorry on push, which prevents cache-masking.

### tsotchke (15 repos, full-depth cloned)
| Repo | Math relevance | Empirical validation we verified | Verdict |
|------|---------------|----------------------------------|---------|
| **libirrep** | Rep theory (Wigner-D/CG/ED) | `tests/` + `examples/EXPECTED_OUTPUT.md` with kagome / Anderson / LSWT-vs-ED published-result cross-checks | **STRONG** — gold-standard empirical validation |
| **moonlab** | Anyon / fusion / quantum sim | real `src/quantum`, `src/optimization/fusion`, anyon examples | **STRONG** numerics; QC framing needs check |
| **quantum_geometric_tensor** | Geodesic / Christoffel / Riemannian geometry | `differential_geometry.c` has a real RK4 geodesic integrator (we ported it → PASS on S²) | **REAL math**, heavy framework |
| **eshkol** | Poincaré/sphere manifold.esk | exp-map has an off-origin bug (dist/|v| drifts 2.83→4.96) | correct formulas, 1 real bug |
| **quantum_rng** | "quantum" RNG | stream is uniform (~8 bits/byte) but 63.99 claim uncomputed | marketing-over-claim |
| other 10 (gpt2, mnist, asteroids, PINN, PIMC, spinNN, tensorcore, llm-arbitrator, classical_rng, homebrew-eshkol) | ML / sim / tooling | not math-audit targets here | out of scope for geometry/rep-theory comparison |

**Formal proofs:** tsotchke has **ZERO** `.lean`/`.coq`/`.v` files across all 15
repos. WuBuMath has a Lean layer (partial). This is the single clean
differentiator: *WuBuMath proves, tsotchke validates numerically.*

---

## Pass 2 — Structural integrity (complementary vs competing)

The two projects are **complementary, not directly competing**, on the math
axis:

- **WuBuMath** = proof-first + clean dependency-free C kernels. Strength:
  formal guarantees (where proven), auditable code. Weakness: Lean layer is
  mostly `sorry` (474), and it has *no* applied ML/quantum-sim surface.
- **tsotchke/libirrep + moonlab** = empirically-validated numerics with
  published-result contracts. Strength: real, checked numbers (kagome12 E0/N,
  Anderson <n>). Weakness: no formal proofs; some repos over-claim
  (eshkol geometry bug, quantum_rng fake entropy, moonlab "quantum" labeling of
  an RDSEED-default path).

**Honest framing for a review we send tsotchke:** lead with *specific real
credit* (libirrep's EXPECTED_OUTPUT.md contract; moonlab's anyon simulator;
qgt's real geodesic integrator), then raise the *verifiable* gaps as
curiosity/learning, never accusation:
1. eshkol `manifold-exp-map` off-origin invariant (reproduced two ways).
2. quantum_rng entropy claim unsubstantiated (measurements attached).
3. moonlab "quantum RNG" path is classical RDSEED by default.
4. Offer: WuBuMath's Lean layer could *formally prove* libirrep/moonlab's
   empirically-checked results (a genuine collaboration, not a takedown).

---

## Pass 3 — Adversarial stress test (what would break our review)

- **If WuBuMath's 474 sorry are mostly in non-core files** (e.g. examples), our
  "partial proofs" line still holds but the *magnitude* is overstated → action:
  quantify how many sorry are in `WubuProofs/` core vs elsewhere before sending
  any "partially proven" claim. (OPEN — not yet counted per-dir.)
- **If libirrep's EXPECTED_OUTPUT.md is hand-typed, not machine-generated**, its
  "gold standard" status weakens → action: confirm it is generated by a fixed
  seed harness (it is referenced as such; verify by re-running one example).
- **If tsotchke fixes the eshkol exp-map bug before we send**, our F1 is moot →
  action: re-pull + re-run `da_eshkol_exp.c` immediately before any outreach.
- **Circular reasoning trap:** "WuBuMath is better because it has proofs" vs
  "tsotchke is better because it has checked numerics" — both true on different
  axes. A review must state BOTH, not pick a winner.

---

## Working strategy (the actionable part)

1. **Re-verify before outreach.** Any claim we send tsotchke must be re-executed
   within this session: re-pull the repo, rebuild, re-run the reproduction.
   Never send a claim from a prior summary.
2. **Send only the friendly version.** Sharp adversarial breakdown stays private
   (in these forks). Outbound message = one quote block, specific credit first,
   gaps as curiosity. (See devils-advocate-audit skill §"Delivering Audit
   Findings to the Audited Party".)
3. **Pin every kernel to closed form.** For any new port, add a test asserting
   the C kernel equals the closed-form formula the Lean file proves. This is the
   anti-fart-sniffing guard; it makes "self-validated" math un-hideable.
4. **Keep upstream verbatim in forks.** Bugs stay visible as evidence for PRs;
   we do not silently "fix" audited files.
5. **Offer collaboration, not competition.** The natural next step is porting
   libirrep/moonlab's checked numerics into WuBuMath's Lean as formal proofs —
   that is the accurate, mutually-beneficial simulation strategy.

## Open items (do NOT assert closed)
- Per-directory sorry count in WuBuMath Lean (core vs examples).
- Re-run libirrep example to confirm EXPECTED_OUTPUT.md is machine-generated.
- Re-pull eshkol/moonlab/quantum_rng before any outbound message (they pushed
  today; verify no fix landed).
