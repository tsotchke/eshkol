---
kind: explanation
status: current
owner-area: numeric
since: v1.4.0
sources:
  - lib/backend/arithmetic_codegen.cpp
  - lib/core/runtime_taylor.c
  - lib/core/bignum.cpp
  - lib/core/rational.cpp
  - lib/core/i128_runtime.cpp
  - lib/backend/vm_ops.c
  - lib/backend/vm_dual.c
  - lib/backend/vm_native.c
  - lib/core/algebraic_domain.esk
  - lib/core/symbolic.esk
  - lib/core/dbsp.esk
---
# ADR-0034: Ring carriers: the functor-of-points evaluation model

- **Status:** Proposed
- **Date:** 2026-10-24
- **Decision owners:** numeric tower, AD, VM, `core.symbolic`, `core.dbsp` and exact linear algebra maintainers
- **Depends on:** ADR-0027 (recursive Taylor level carrier), ADR-0025 (complex values carry derivatives), ADR-0009 (DBSP), ADR-0004 (type system), ADR-0012 (object ABI)
- **Scope:** every place a number is added, subtracted, multiplied, negated, inverted, compared for equality or built from an integer, on native code generation (JIT and AOT), the native runtime, the bytecode VM and the browser VM, and the library carriers built on them
- **Evidence:** a pure-Eshkol prototype on the released 1.3.6 compiler (29 checks, no compiler change), described under Evidence

## Context

### The idea

Grothendieck's functor of points says that a space X is known by its points
with values in every commutative ring R, the sets X(R), together with the maps
X(R) -> X(S) induced by every ring map R -> S. A numeric program built from
`+`, `-`, `*`, integer constants and division by units is a polynomial (or
rational) map, and so it is a morphism of such functors: it has a value at an
R-point for every ring R, and every ring map commutes with it.

Read this way, facilities that Eshkol implements separately are one facility
with different carriers:

| Carrier | What evaluation over it is | Where Eshkol has it today |
|---|---|---|
| R (flonums), exact Q | ordinary evaluation; the numeric tower | native tower dispatch, VM tower dispatch |
| Z/2^128 | fixed-width wrapping integers | `HEAP_SUBTYPE_I128` (`lib/core/i128_runtime.cpp:165`) |
| k[e]/e^2 | forward-mode AD; the tangent space is Spec k[e]/e^2 | native 8-jet, VM `VmDual` |
| k[e]/e^(n+1) | Taylor jets, higher-order AD | native `esh_taylor_t`, VM Taylor and level kinds |
| k[e1,e2]/(e1^2, e2^2) | nested first-order AD, hyper-duals | 8-jet slots, VM `VmHyperDual` |
| Q(alpha), Q(zeta_n) | exact number fields | `core.algebraic_field`, `core.cyclotomic` |
| F_p | modular arithmetic | the GF(p) coefficient field of `core.symbolic` (open as #852) |
| Z/p^k, Z_p | Newton iteration is Hensel lifting (formal smoothness) | none |
| Q[x], Q[[x]] | symbolic evaluation | `core.symbolic` `poly`/`series`, `core.polynomial_exact` |
| Z[G] | DBSP Z-sets: finitely supported G -> Z | `core.dbsp` |
| Spec Z | run an exact algorithm in every fibre F_p and glue by CRT | none |

The last row is the payoff of the reading. An integer polynomial computation
commutes with every reduction Z -> F_p, so an exact answer can be computed in
several small prime fields, where every element fits a machine word, and
recovered by the Chinese remainder theorem plus rational reconstruction, with
a prime count fixed in advance by a proven bound (Hadamard's inequality for a
determinant). This is the multimodular method of computer algebra.

### How each facility is implemented today

The survey used ICC (`adr list`, `find-symbol`, `trace-callers`,
`duplicate-implementations`, `impact-analysis`, `search-chunks`) on the v1.4
integration head and was confirmed by reading the code. A *dispatch site* is a
place that inspects the representation of a number to choose which ring's
arithmetic performs a ring operation (`+ - *`, negation, unit inverse, integer
power, equality). Order, `quotient`/`remainder`/`modulo` and transcendental
functions are not ring operations and are not counted.

**Native code generation: 10 sites.** Each of the generic operators in
`lib/backend/arithmetic_codegen.cpp` emits its own ladder of carrier tests:

| Operator | Line | Carrier arms in the ladder |
|---|---|---|
| `ArithmeticCodegen::add` | 1700 | 9: i128, Taylor tower, 8-jet, bignum, rational, complex, reverse-tape node, double, int64 with overflow promotion |
| `ArithmeticCodegen::sub` | 1928 | 9: the same |
| `ArithmeticCodegen::mul` | 2155 | 9: the same |
| `ArithmeticCodegen::div` | 2386 | 9: the same |
| `ArithmeticCodegen::neg` | 2802 | 4 |
| `ArithmeticCodegen::compare` | 3339 | 7 |
| `ArithmeticCodegen::pow` | 3644 | 4 |
| `withComplexCarrierDispatch` | 543 | complex carrier wrapper around every binary ladder |
| `withADBinaryDispatch` | 644 | reverse-tape wrapper around every binary ladder |
| `withADUnaryDispatch` | 845 | reverse-tape wrapper around every unary ladder |

The ladder order is load-bearing: the AD carriers are tested before the tower
subtypes so that an exact literal operand cannot take the dispatch from a live
jet (the comment at `arithmetic_codegen.cpp:1761` records why). That ordering
is a coercion lattice expressed as control flow, repeated in seven functions.

**Native runtime: 7 sites.** `exact_binary` (`lib/core/runtime_taylor.c:332`);
`num_binary` (`runtime_taylor.c:1960`), the coefficient ring of the ADR-0027
level carrier, with its own arms for reverse-tape nodes, Taylor towers, the
8-jet, complex values, exact numbers and doubles; `eshkol_taylor_binary_tagged`
(`runtime_taylor.c:1779`); `eshkol_bignum_binary_tagged`
(`lib/core/bignum.cpp:943`); `eshkol_rational_binary_tagged`
(`lib/core/rational.cpp:985`); `eshkol_i128_binary_tagged`
(`lib/core/i128_runtime.cpp:165`); `eshkol_jet_tensor_binary`
(`lib/core/runtime_tensor_math.cpp:395`).

**Bytecode VM: 14 sites.** `vm_op_arith` (`lib/backend/vm_ops.c:122`), a
ten-arm ladder (reverse tape, i128, complex, hyper-dual, dual, rational,
exact division, bignum, int64, double); `vm_exec_eq`, `vm_exec_lt`,
`vm_exec_gt` (`vm_ops.c:212`, `227`, `242`); the native-call groups in
`vm_dispatch_native` for complex (`lib/backend/vm_native.c:10849`, ids
307-313), i128 (`vm_native.c:11041`, ids 2103-2119), rational
(`vm_native.c:11139`, ids 330-336), dual (`vm_native.c:11465`, ids 370-376)
and hyper-dual (`vm_native.c:11563`, ids 1905-1909); `vm_dual_add`,
`vm_dual_sub`, `vm_dual_mul`, `vm_dual_div` (`lib/backend/vm_dual.c:1145`,
`1160`, `1175`, `1195`), each a four-arm ladder (level, Taylor, exact dual,
float dual); `vm_bignum_arith` (`vm_native.c:7787`).

**Total: 31 dispatch sites carrying more than 100 carrier arms.**

**Nine implementations of the same series algebra.** The truncated-series
recurrences (Cauchy product, series division, `exp`, `log`, `sin`/`cos`,
constant power) exist once per coefficient representation:

| Tier | Coefficients | Location |
|---|---|---|
| `tr_*` | double | `runtime_taylor.c:776`-`1160` |
| `tre_*` | exact tagged | `runtime_taylor.c:1178`-`1390` |
| `ddual_*` | double with a tangent companion | `runtime_taylor.c:1229`-`1366` |
| `level_ser_*` | any number (ADR-0027 level) | `runtime_taylor.c:2165` onward |
| 8-jet algebra | three nilpotent directions | `runtime_taylor.c` `jet_mul_raw`, `jet_compose` |
| `AutodiffCodegen::dual*` | LLVM IR, 30 functions | `lib/backend/autodiff_codegen.cpp:1669`-`2218`, `15415`-`15524` |
| VM Taylor | double or exact | `vm_dual.c:298` (`taylor_binary`), `362` (`taylor_unary`) |
| VM level | any `VmDual` | `vm_dual.c:628`-`1080` (`lv_*`) |
| VM hyper-dual | f, f1, f2, f12 | `lib/backend/vm_hyperdual.c:47`-`69` |

`lib/core/taylor_recurrences.def` already makes the *primitive set* one table
(33 rows) shared by the runtime and the IR emitter. The *algebra* behind each
row is still written once per tier.

**Eight library carrier families,** each with its own operation vocabulary:

| Family | Operations | Location |
|---|---|---|
| `algebraic-domain` | a record of closures: copy, add, sub, mul, inverse, equal?, zero?, unit?, zero, one | `lib/core/algebraic_domain.esk:15`; 100+ `algebraic-domain-ref` call sites in 25 files |
| `core.ad.algebraic` | Taylor and hyper-dual lifts of a number field | `lib/core/ad/algebraic.esk:107`, `114` |
| `core.algebraic_field` | Q(alpha) | `lib/core/algebraic_field.esk:96`, `99` |
| `core.cyclotomic` | Q(zeta_n) | `lib/core/cyclotomic.esk:112`, `124` |
| `core.polynomial_exact` | Q[x] | `lib/core/polynomial_exact.esk:27`, `29` |
| `core.symbolic` | Q[x1..xn], truncated series; #852 adds a field switch `F` (`sf-add`, `sf-mul`, ...) choosing between Q and GF(p) | `lib/core/symbolic.esk:458`, `470`, `665`, `684` |
| `core.dbsp` | Z-set sum, negation, integer scaling, bilinear join | `lib/core/dbsp.esk:202`, `209`, `217`, `272` |
| exact linear algebra | the tower, with exactness checks | `lib/core/exact_linalg.esk`, `lib/core/certified_solve.esk:7`-`10` |

`algebraic-domain` is a ring-carrier record in all but name, restricted to
number fields and their AD lifts. #852's field switch is a second one,
restricted to Q and GF(p). Neither can be used by the other, by `core.dbsp`,
or by the compiler.

`duplicate-implementations` reports no textual duplicates among these: the
duplication is semantic (the same algebra over different coefficient types),
which a token-similarity scan does not see. `search-chunks` finds no CRT,
rational-reconstruction-from-residues or Hensel facility in the tree.

## Decision

### 1. The ring-carrier protocol

A **ring carrier** is a commutative ring with identity, given as data:

| Field | Meaning |
|---|---|
| `zero`, `one` | the identities |
| `add`, `neg`, `mul` | the ring operations (`sub` is derived) |
| `from-integer` | the unique ring map Z -> R |
| `equal?` | decidable equality |
| `unit?`, `inverse` | the unit group; `inverse` is defined exactly on units and raises otherwise |

and optional structure, each a declared capability rather than a convention:

| Capability | Meaning |
|---|---|
| `characteristic` | n >= 0 with ker(Z -> R) = nZ |
| `field?` | every nonzero element is a unit |
| `nilpotency` | n when the carrier is K[e]/e^n over a base K, with its `base`, `seed` and `coefficient` maps |
| `exact?` | the operations satisfy the laws exactly (false for flonums) |
| `primal` | the ring map to the base carrier (e -> 0); branches and comparisons read it |
| `analytic` | `exp`, `log`, `sin`, ... defined on the carrier; requires a Q-algebra (characteristic 0) |
| `order` | a total order compatible with the ring (R and Q; not F_p, not jets beyond the primal) |
| `euclidean` | `quotient`, `remainder`, `gcd` (Z, K[x]) |

**Laws.** `(add, zero, neg)` is an abelian group, `(mul, one)` a commutative
monoid, `mul` distributes over `add`, and `from-integer` is a ring map. For an
`exact?` carrier the laws are equalities and are checked by a property gate;
for flonums they hold to rounding, and the carrier says so.

**Ring maps** are first-class too: a *base change* is a map of carriers that
commutes with the operations (Z -> Q, Q -> R, Z -> F_p, Q -> Q[e]/e^n,
K[x] -> K[e]/e^n by x -> a + e, Z[G] -> Q by a character). The tower's
contagion rules are exactly the base-change maps Z -> Q -> R applied to the
narrower operand.

### 2. Registered once

Every carrier is registered once, in one registry per engine compiled from one
source:

- **Library:** `core.ring` exports `make-ring-carrier`, `ring-ref` and the
  generic operations `r+ r- r* r-neg r-inv r-expt r= r-int`, plus the
  constructors `make-zmod`, `make-jets`, `make-polys` and `zset-group-ring`.
  `algebraic-domain` becomes a constructor that returns a ring carrier, keeping
  its present API; #852's field switch becomes `(make-zmod p)` and
  `rationals`.
- **Native runtime:** one table `eshkol_ring_carrier_t` (id, name,
  characteristic, capability bits, `primal`, and the operation entry points)
  indexed by a carrier id. The tower carriers (int64, bignum, rational,
  double, i128, complex), the series carriers (8-jet, Taylor tower, level) and
  the reverse-tape node are rows of this table.
- **VM:** the same table, compiled from the same C sources into the VM and the
  browser VM, so a carrier is one implementation on both substrates (the
  property `.icc/ad-carrier-manifest.yaml` calls `shared: true`).

Registration is the only place a carrier's arithmetic is written. A new
carrier adds one row; it does not edit a ladder.

### 3. Dispatch: base change when the carrier is known, one dynamic dispatch otherwise

- **Static.** When the carrier of every operand is known at compile time
  (from type inference, a literal, or a declared parameter), code generation
  instantiates the operation for that carrier. Specialising a generic
  function to a carrier *is* base change of the morphism along the ring map
  into that carrier, so a monomorphised function is the same morphism, not a
  copy of it. The int64 and double fast paths of today's ladders are the
  monomorphised instances of the tower carriers and stay inline.
- **Dynamic.** Otherwise one call reaches `eshkol_ring_binary(op, a, b)`
  (`vm_ring_binary` on the VM). It reads both carrier ids, finds their join in
  the coercion lattice (the smallest carrier both base-change into), applies
  the base-change maps and calls the join's operation. The lattice is a table
  written once; the ladder order of section "Context" becomes data. The cached
  outlined helper that code generation already emits per operator holds the
  one call.

A value never changes carrier implicitly except along a lattice edge. F_p and
Z/m are not in the tower lattice: combining an element of F_p with a rational
raises, as #852's coefficient fields already do.

### 4. Exactness

An `exact?` carrier never produces an inexact element: an operation that
would leave it raises (`core.cyclotomic` and #852 already behave this way).
The tower keeps R7RS contagion as the lattice edge Q -> R. Series carriers
inherit exactness coefficientwise, as ADR-0027 requires: a coefficient is
exact while everything it was computed from is exact.

### 5. AD nesting

A differentiation pass of epoch E and order n evaluates its function over the
carrier K[e_E]/e_E^(n+1), where K is the carrier of the enclosing passes.
Nested passes therefore evaluate over

    K[e_E1]/e_E1^(n1+1) [e_E2]/e_E2^(n2+1) ...  =  K[e_E1, e_E2, ...]/(e_E1^(n1+1), e_E2^(n2+1), ...)

with one nilpotent generator per epoch. Perturbation confusion is impossible
by construction: two perturbations combine only when they are the same
generator, which is ADR-0027's epoch rule stated as ring theory. ADR-0027's
level carrier is the dynamic representation of this tower of carriers and is
kept; this record makes its coefficient ring (`num_binary`) the registry
instead of a private ladder. The hyper-dual k[e1,e2]/(e1^2,e2^2) and the
8-jet are fixed-shape instances of the same tower and become registered
carriers with monomorphised arithmetic.

Reverse mode is not a carrier: the cotangent map is the transpose of the
tangent map, and the tape (ADR-0002, ADR-0016) stays its own facility. The
reverse-tape node remains a row of the registry only so that the lattice
places it, as the ladders do today.

DBSP fits the same picture. A Z-set is an element of the free Z-module on
rows, and of the group ring Z[G] when rows form an abelian group; the
bilinear join rule of ADR-0009, `d(a * b) = da * b + a * db + da * db`, is
the product rule in a carrier where the square of the change is kept
(an exact difference), against k[e]/e^2 where it is truncated (a
linearization). Z-set weights become a carrier parameter, Z by default.

### 6. Computation over Spec Z

`core.multimodular` provides, for any computation written against the
protocol:

- the fibre evaluation: run the computation over `(make-zmod p)` for primes p
  chosen below the machine-word bound, skipping primes where a required
  element is not a unit (an "unlucky" prime);
- CRT gluing to a residue modulo M, the product of the primes;
- the symmetric lift for an integer result and Wang's rational reconstruction
  for a rational result, which is unique when M > 2ND for numerator and
  denominator bounds N and D;
- the bounds themselves: Hadamard's inequality for determinants and Cramer
  numerators, so the prime count is fixed before the first fibre runs and the
  answer is proven, never a probabilistic early termination.

Correctness is base change: an integer polynomial map commutes with Z -> F_p,
and Z/M is the product of the F_p.

### 7. The VM path

The VM keeps one arithmetic opcode per operation. `vm_op_arith` and the
`vm_exec_*` comparisons call `vm_ring_binary`, and the native-call groups for
complex, i128, rational, dual and hyper-dual arithmetic become registry rows.
`VmDual` and `VmHyperDual` become the VM representations of registered series
carriers. The browser VM is rebuilt from the same sources. The VM parity
table gains one row per registered carrier.

### 8. What this retires

- the carrier arms of the seven operator ladders in
  `arithmetic_codegen.cpp` and the three wrapper dispatchers (they become one
  outlined dynamic dispatch plus monomorphised fast paths);
- the private coefficient ladder of `num_binary` and the separate
  `exact_binary` dispatch;
- eight of the nine series-algebra tiers: one generic recurrence set over a
  coefficient carrier, with the double tier kept as its monomorphised
  instance;
- the VM's `vm_op_arith` ladder, the five per-carrier native-call groups and
  the four-arm ladders inside `vm_dual_*`;
- `algebraic-domain`'s private record shape, #852's `F` switch and the
  fixed integer weights of `core.dbsp`, all as instances of one library
  protocol.

The `forked_carrier_budget` of `.icc/ad-carrier-manifest.yaml` shrinks as the
VM carriers become shared registry rows.

## Staged plan

Each rung ships only what its gate proves; work not yet in the tree is
Planned for the rung named.

| Release | Stage | Gate |
|---|---|---|
| v1.4.0-connection | This record. The evidence program joins `examples/` as `mathematics_functor_of_points.esk` once it has passed native JIT and AOT on Linux x86-64 as well as macOS arm64. | The program's 29 checks on native JIT and AOT on both platforms. |
| v1.4.1 | Library protocol `core.ring` and `core.multimodular`. `algebraic-domain` returns a ring carrier; `core.symbolic` coefficient fields and `core.dbsp` weights take a carrier. | Ring-law property gate on every exact carrier; the 1.4.0 example rewritten against `core.ring`; existing `algebraic_*`, `symbolic_*` and `dbsp` suites unchanged in output. |
| v1.4.2 | Native runtime registry `eshkol_ring_carrier_t` and `eshkol_ring_binary`; `num_binary` and `exact_binary` dispatch through it; one generic series recurrence set over a coefficient carrier, the double tier its monomorphised instance. | `tests/ad/nested_towers_matrix_test.esk`, `nested_operator_matrix_test.esk` and the exact-arithmetic suites byte-identical; the Taylor unary route guard; benchmark gate on the double tier (no slowdown beyond noise). |
| v1.4.3 | Code generation: the operator ladders become monomorphised fast paths plus one outlined registry call; static base change when the carrier is known. | Full native suite on JIT and AOT; AD composition oracle; codegen size and compile-time measurements against v1.4.2. |
| v1.4.4 | VM and browser VM on the same registry; `VmDual`/`VmHyperDual` as registered carriers; one VM parity row per carrier. | VM parity corpus; `ad_carrier_model_clean` with `forked_carrier_budget` lowered; browser VM rebuilt from the same sources. |
| v1.4.5-accelerate | Fibre-parallel kernels: batched F_p arithmetic on CPU vector units, GPU and XLA (word-size modular GEMM), CRT on the host; jets over F_p batched the same way. | Multimodular determinant and solve against `exact-det`/`exact-solve` on growing sizes; throughput measured against the bignum path. |
| v1.5.0 | Carriers in the type system (ADR-0004): a ring constraint on type variables, base change as functoriality, guaranteed monomorphisation; `define-ring-carrier` for user carriers so that plain `+` reaches them. | Type-directed monomorphisation coverage; user-carrier conformance suite; the laws gate extended to user carriers. |

## Risks, each a build item

| Risk | Build item |
|---|---|
| Dynamic dispatch costs time on the hot path. | Monomorphised int64/double paths stay inline (v1.4.3); the registry call is reached only where today's ladder would reach a heap carrier; the v1.4.2 and v1.4.3 benchmark gates. |
| Flonums do not satisfy the ring laws exactly. | The `exact?` capability; the laws gate is exact only on exact carriers, and the flonum carrier states "laws to rounding". |
| Transcendental functions do not exist on every ring. | The `analytic` capability, defined only on Q-algebras; jets over F_p of order >= p need divided powers and are refused until a divided-power carrier is registered. |
| Unlucky primes in the multimodular method. | Detected per fibre (a required unit is zero mod p) and skipped; the bound counts usable primes only, as the evidence example does. |
| Perturbation confusion when carriers nest. | One nilpotent generator per epoch (section 5), the ADR-0027 invariant; the nested AD matrices are the gate. |
| A carrier id needs a place in the object representation. | Carrier ids ride the existing heap subtypes during v1.4.2-v1.4.4; a general ring-element header goes through the ADR-0012 object ABI migration. |
| The VM has no dual-carrying tensor. | Tensors over a carrier are K-modules; the module extension of the registry is the v1.4.4 item that also closes the blocker named in the AD carrier manifest. |
| Library carriers and the compiler registry could drift. | One law suite run against both; v1.5.0 makes the library carriers registry rows. |

## Alternatives rejected

- **Keep adding ladder arms.** Each new number kind edits seven code
  generation ladders, the runtime ladders and the VM ladder, and the ordering
  between arms stays implicit control flow.
- **Operator overloading on records only (library-level, no compiler
  registry).** It unifies the library carriers but leaves the native and VM
  numeric carriers forked, which is where most of the dispatch sites are.
- **One universal number type (everything a series over a bignum rational).**
  It loses the machine-word fast paths and the exactness distinctions the
  tower guarantees.

## Evidence

The evidence program (planned as `examples/mathematics_functor_of_points.esk`)
defines a ring-carrier record of closures, writes one polynomial `f(x) = 3x^5 - 2x^3 + x^2 - 7x + 4` once
against it, and evaluates it over R, Q, Q[e]/e^2, Q[e]/e^6, F_7, F_7[e]/e^6,
Q[x] and Z[t, 1/t] (the latter as `core.dbsp` Z-sets, with the ring product a
DBSP join on the trivial key). Each result is checked against an independent
reference: `derivative` and `taylor` on the same function for the AD
carriers, the direct formula for R and Q, reduction mod 7 for F_7, evaluation
at seven rationals for Q[x], and characters t -> c for the group ring. The
same Newton step finds sqrt 2 over R and the 7-adic square root of 2 over
Z/7^16 in four steps, with the derivative taken by the dual carrier over each
ring. A 6x6 integer matrix with 13-digit entries has its determinant computed
in 9 fibres F_p near 2^30 (the Hadamard bound has 79 digits) and glued by
CRT, equal to the bignum Bareiss determinant of `core.exact_linalg`, and the
first component of the solution of A x = b is recovered by CRT and rational
reconstruction, equal to `exact-solve`. The program runs on the released
1.3.6 compiler with no change: 29 checks pass on native JIT and AOT on
macOS arm64. Linux x86-64 qualification is the remaining step before it joins
`examples/`.

## References

- A. Grothendieck, J. Dieudonne, *Elements de geometrie algebrique I*,
  Publ. Math. IHES 4 (1960), the functor of points.
- D. Eisenbud, J. Harris, *The Geometry of Schemes*, Springer GTM 197 (2000),
  sections I.4 and VI.
- J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
  Cambridge (2013), chapters 5 (CRT, rational reconstruction) and 15 (Hensel
  lifting).
- P. S. Wang, M. J. T. Guy, J. H. Davenport, "P-adic reconstruction of
  rational numbers", SIGSAM Bulletin 16 (1982).
- B. A. Pearlmutter, J. M. Siskind, "Lazy multivariate higher-order
  forward-mode AD", POPL 2007, and "Perturbation confusion and referential
  transparency" (2005).
- M. Budiu, T. Chajed, F. McSherry, L. Ryzhyk, V. Tannen, "DBSP: Automatic
  incremental view maintenance for rich query languages", PVLDB 16 (2023).
