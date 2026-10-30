---
kind: explanation
status: current
owner-area: types
since: v1.9.0
sources:
  - inc/eshkol/types/hott_cubical.h
  - inc/eshkol/types/hott_forms.h
  - lib/types/cubical/interval.cpp
  - lib/types/cubical/eval.cpp
  - lib/types/cubical/conv.cpp
  - lib/types/cubical/elab.cpp
  - lib/types/cubical/decl.cpp
  - lib/types/hott_forms.cpp
  - lib/hott/foundations.esk
  - lib/hott/numbers.esk
  - lib/hott/circle.esk
  - lib/hott/truncation.esk
  - lib/hott/suspension.esk
  - tests/hott/programs/pi1_circle.esk
  - tests/hott/agda/Pi1Circle.agda
  - scripts/run_hott_gate.sh
---
# ADR-0043: Identity types, univalence and higher inductive types

**Status:** Accepted
**Date:** 2026-10-30
**Scope:** the identity types of Eshkol's HoTT layer, path induction,
transport, function extensionality, univalence, higher inductive types, and
the computational presentation that makes all of them compute.
**Depends on:** ADR-0004 (one quantitative dependent type system), ADR-0013
(the gradual type relation), ADR-0035 (Kripke-Joyal semantics: mere
propositions and propositional truncation are the HoTT side of local truth).

## Context

Eshkol's documentation describes its type system as grounded in Homotopy Type
Theory. The HoTT layer (`inc/eshkol/types/hott_types.h`) had universes, Pi
types, a universe of erased propositions and a gradual relation, but no type
of identifications: no identity types, no path induction, no transport, no
univalence and no higher inductive types. ADR-0004 deferred those features
because the systems types on the roadmap do not depend on them. This record
builds them.

Homotopy Type Theory has two standard presentations.

1. **Book HoTT** (The Univalent Foundations Program, *Homotopy Type Theory*,
   2013): Martin-Lof identity types with J, univalence as an axiom, higher
   inductive types whose path constructors have propositional computation
   rules. An axiom does not compute: transport along `ua e` is stuck, `ap f
   loop` does not reduce to the image of the loop, and a closed integer
   produced by a proof need not reduce to a numeral. Making it compute needs
   rewrite rules added one by one, without a canonicity result.
2. **Cubical type theory** (Cohen, Coquand, Huber, Mortberg, "Cubical Type
   Theory: a constructive interpretation of the univalence axiom", 2018,
   CCHM; Coquand, Huber, Mortberg, "On Higher Inductive Types in Cubical Type
   Theory", LICS 2018, CHM18; Angiuli, Brunerie, Coquand, Favonia, Harper,
   Licata, "Syntax and models of Cartesian cubical type theory", 2021,
   ABCFHL; implemented in Cubical Agda, Vezzosi, Mortberg, Abel, JFP 2021):
   paths are maps out of an interval, Kan composition is part of every type,
   univalence is a theorem proved with Glue types, and higher inductive types
   are ordinary data types with path constructors. Every term computes, and
   closed terms of inductive types reduce to constructors (Huber, "Canonicity
   for cubical type theory", 2019).

## Decision

### 1. The presentation: CCHM with the transp/hcomp split

Eshkol's kernel is a cubical type theory: the De Morgan interval of CCHM,
dependent path types `PathP`, the Kan operations `transp` and `hcomp` in the
form Cubical Agda uses, Glue types, and higher inductive types with `hcomp`
as a constructor (CHM18). The Cartesian interval of ABCFHL was considered; the
De Morgan interval was chosen because its connections (`min`, `max`) and
reversal (`~`) give short proofs of the path algebra, and because it is the
interval of Cubical Agda, which serves as an independent cross-check of every
value the kernel computes (section 6).

The kernel is `eshkol::hott::cubical` (`inc/eshkol/types/hott_cubical.h`,
`lib/types/cubical/`): values in weak head normal form, closures for term
binders, nominal dimension binders (a fresh atom and a value body), so that a
value can be restricted to a face and every stuck form re-reduces when a
substitution unblocks it.

### 2. What computes definitionally

These hold by evaluation; the kernel's conversion checker decides them.

- beta, eta and delta for Pi, Sigma and definitions; eta for paths (two paths
  are equal when they agree at a fresh dimension) and for Glue elements.
- The interval equations of a free De Morgan algebra (formulas are compared
  by a canonical disjunctive normal form) and the endpoints of path
  application: `p i0` is `a` for `p : PathP A a b`.
- `transp` along any line, on Pi, Sigma, PathP, the universe, Glue types and
  inductive types (structurally on constructors and on `hcomp`); along a line
  constant on its face `phi`, `transp` is the identity where `phi` holds; along
  the constant line of an inductive type whose parameters do not vary, it is
  the identity (also on variables), as in Cubical Agda.
- `hcomp` on Pi, Sigma, PathP and Glue; in a universe it is a Glue type
  (CCHM 7.2) whose equivalences are the transports of the faces; in an
  inductive type it is a constructor that eliminators compute on (CHM18).
  With an empty system, `hcomp` is the identity.
- **Univalence**: `ua e` is the Glue line `Glue B [i=0 |-> (A, e), i=1 |-> (B,
  idEquiv B)]`, and `transport (ua e) x` reduces to `transp^B (fst e x)`;
  when `B` is an inductive type with fixed parameters that is `fst e x`
  itself. `transport (ua sucEquiv) (pos 4)` is `(pos 5)`.
- **Function extensionality**: `funExt h = \i x. h x i`; `happly (funExt h)`
  is `h` definitionally.
- **Higher inductive types**: an eliminator computes on point constructors,
  on path constructors (`ap f loop` is the image of `loop`, definitionally),
  and on `hcomp`.
- **J** on `refl` computes when transport in the motive is the identity on
  constant lines (for example any inductive motive).

### 3. What is a term (a proof, not a reduction)

- `J-refl`: the computation rule of J in general, `J P d refl = d`, is a path
  (CCHM; in Cubical Agda it is `transportRefl`).
- `ua-beta`: `transport (ua e) x = fst e x` in general is a path.
- The groupoid laws: `rUnit`, `lUnit`, `rCancel`, `lCancel` are squares built
  with `hcomp`; `symInvol` is `refl`.
- The structure of equivalences: `iso-to-equiv` (the isoToIsEquiv proof of the
  Cubical Agda library), `path-to-equiv`, contractible fibers.
- `decodeEncode` (by J) and `windingIntLoop` (by induction on the integers),
  which make `OmegaS1-equiv-Int : Equiv OmegaS1 Int` and the path
  `OmegaS1=Int : Path U OmegaS1 Int`.

### 4. Soundness guards

The kernel is a logic, so it refuses programs that would make it
inconsistent: universes are predicative (`U l : U (lsuc l)`, cumulative,
with explicit level arguments; `U : U` is rejected); data types are strictly
positive; recursion is structural (a definition calls itself only on a
variable bound by a recursive argument of its own split); a split covers
every constructor and agrees with the boundary of every path constructor;
every composition side meets its base and sides agree where faces overlap;
`transp` is checked constant where its formula holds; Glue clauses carry
equivalences into the base. A path constructor whose boundary is built from
constructors and the constructor's own arguments transports structurally;
for any other boundary (set truncation's `squash0`, whose sides are paths
among its arguments) transport adds the CHM18 correction, a composition from
the boundary of the transported arguments to the transport of the boundary.

### 5. Surface syntax and integration

Two forms carry kernel syntax inside Eshkol programs:

```scheme
(hott decl ...)       ; data, define, example, import, builtin; checked at compile time
(hott-eval term)      ; replaced by the normal form of term, as a quoted datum
```

The parser keeps both as reader datums (`syntax_record_use`) under one
operator, `ESHKOL_HOTT_OP`. The type checker hands them to the kernel owned by
the compilation's `TypeEnvironment` (persisting across REPL inputs); a
rejected form is an error in every mode and code generation is refused, the
same discipline as linearity. The bytecode VM reads source with its own
reader; it reaches the same kernel through `eshkol_vm_install_hott_form`,
which the drivers install next to the linear check, so there is one kernel and
one rule on every engine. The library (`lib/hott/*.esk`) is embedded into the
kernel at configure time and loaded with `(import module)`. The syntax is
documented in `docs/reference/language/hott.md` and `docs/guide/HOTT.md`.

### 6. Evidence

- The flagship: the fundamental group of the circle, by encode-decode
  (Licata-Shulman 2013; HoTT Book 8.1). `winding (loop . loop)` is `(pos 2)`,
  `winding (loop . loop . loop^-1)` is `(pos 1)`, `winding (intLoop (negsuc
  3))` is `(negsuc 3)`, transport along `OmegaS1=Int` sends `loop . loop` to
  `(pos 2)`, and transport back sends `(pos 3)` to a loop of winding number 3.
- `tests/hott/agda/Pi1Circle.agda` states the same definitions against Cubical
  Agda's builtin primitives and proves every one of these values by `refl`
  (Agda 2.8.0, `--cubical --safe`).
- `scripts/run_hott_gate.sh` (CTest `hott_gate`) runs `tests/hott/programs`
  through the kernel driver, the JIT, AOT binaries and the bytecode VM against
  their `.expected` output, and `tests/hott/reject` (the negative corpus)
  through the kernel and both compiler routes.

## Relation to ADR-0004 and ADR-0035

ADR-0004 kept the kernel of the systems types small and predicative; this
kernel is separate and predicative too, and it is where the HoTT features
ADR-0004 left for later now live. ADR-0035 reads a U2 proposition as a mere
proposition and propositional truncation as the image of `A -> 1`; the
kernel's `isProp`, `Trunc` and `Trunc-rec` are those notions as computing
types.

## Consequences

- "Grounded in Homotopy Type Theory" is a statement about the kernel:
  identity types, univalence and higher inductive types are checked and
  computed by the compiler.
- Proofs are compile-time objects; `hott-eval` extracts computed values (a
  winding number, a numeral) into the running program, identically on every
  engine.
- The kernel is independent of the gradual checker of ADR-0013: a HoTT form
  never weakens or strengthens the typing of ordinary Eshkol code.
