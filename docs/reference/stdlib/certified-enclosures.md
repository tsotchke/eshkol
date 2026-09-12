# Certified enclosures — directed rounding and rigorous Taylor models

**Sources**: [`lib/backend/llvm_codegen.cpp`](../../../lib/backend/llvm_codegen.cpp) (`codegenNextafter`), [`lib/backend/vm_native.c`](../../../lib/backend/vm_native.c) (native ids 2228/2229), [`lib/core/ad/rigorous_interval.esk`](../../../lib/core/ad/rigorous_interval.esk), [`lib/core/ad/rigorous_taylor_models.esk`](../../../lib/core/ad/rigorous_taylor_models.esk)
**Require**: `(require core.ad.taylor_models)` (re-exports the whole family); `fl-next-up`/`fl-next-down` are runtime builtins, always available with no require

This is the **rigorous** (proof-backed) layer beneath the existing
**validated** `core.ad.interval` and `core.ad.taylor_models`
(epsilon-widening and sampled-remainder approximations — their exported
symbols are listed in
[the stdlib index](INDEX.md) and
[shipped_exports.md](shipped_exports.md)). Every enclosure
described here is a proof: given operand intervals that soundly contain
their true real values, the result interval provably contains the true
real result. The validated modules are **completely unchanged by
default** — this document is additive.

## 1. Directed rounding primitives (runtime)

`fl-next-up` and `fl-next-down` are unary `double -> double` builtins
wrapping the C99 libm function `nextafter`: `fl-next-up x` = the next
representable double strictly greater than `x`; `fl-next-down x` = the
next one strictly less. Both are wired into **both execution engines**:

- **Native (JIT and AOT share the same LLVM backend)**: `codegenNextafter`
  in `lib/backend/llvm_codegen.cpp`, declared via `declareBinaryMathFunc`
  in `lib/backend/builtin_factory_codegen.cpp`. Deliberately bypasses the
  generic `codegenMathFunction` dispatcher (no AD-dual, no tensor-map, no
  Taylor-tower fast path — directed rounding has no sound derivative and
  is not vector-mapped) and rejects complex/dual/AD-node/tensor operands
  with a typed error rather than silently misreading their payload as a
  double.
- **Bytecode VM**: native ids 2228 (`fl-next-up`) and 2229
  (`fl-next-down`) in `lib/backend/eshkol_vm.c`'s `BUILTINS[]` table,
  dispatched in `lib/backend/vm_native.c`'s `vm_dispatch_native`. Same
  reject list (complex/dual/tensor/closure), same `nextafter` call.

**Rounding mechanism: `nextafter`, not `fesetround`.** Directed rounding
classically means computing with the FPU rounding mode steered toward
±infinity. This compiler deliberately does **not** use `fesetround`: its
FPU control-word state is ambient, per-thread machine state that does not
portably survive a JIT-compiled call boundary, a native-vs-VM substrate
switch, or every backend this compiler targets (wasm has no FPU
rounding-mode control at all). `nextafter` is a pure function with
identical behaviour everywhere. The soundness argument: an IEEE-754
round-to-nearest op (or a libm call correctly rounded to within 1 ulp)
differs from the true real result by at most 0.5 ulp of the *computed*
result; `nextafter` steps by exactly the local ulp (correct across
power-of-two boundaries and subnormals, unlike a hand-rolled
"multiply by 1±epsilon" formula), so `fl-next-down(r) <= true <=
fl-next-up(r)` always holds after a single elementary operation.

Every interval primitive in `core.ad.rigorous_interval` builds each
result endpoint from **exactly one** such nudge (never a chain of
unnudged floating ops feeding one endpoint), which is what keeps the
soundness argument valid throughout.

**Self-test.** `tests/stdlib/certified_enclosures_test.esk` §1 proves
outward rounding on 0.1, `1/3.0`, the smallest positive denormal (`5e-324`),
`-0.1`, and `0.0`: each brackets its argument (`fl-next-up x > x`,
`fl-next-down x < x`) and round-trips exactly (`fl-next-down(fl-next-up
x) = x`), which is only possible if the step taken really is the true
single ulp.

## 2. Outward-rounded interval arithmetic — `core.ad.rigorous_interval`

`ia+ ia- ia* ia/ ia-neg ia-ipow ia-scale ia-sqrt ia-exp ia-log ia-sin
ia-cos ia-atan ia-pi` operate on the same `(lo . hi)` pair representation
`core.ad.interval` uses (`interval-of`/`interval-lo`/`interval-hi`/
`interval-contains?`/... all work unchanged on their results).

**Exactness policy.** `ia+`/`ia-`/`ia*`/`ia/` inspect their operand
endpoints: when every endpoint relevant to a result endpoint is exact
(int/bignum/ratnum), the result is computed with ordinary exact generic
arithmetic — zero rounding error, R7RS contagion already guarantees it.
The moment any relevant endpoint is inexact, every operand is coerced to
a double and the outward nudge is applied. Mixed exact/inexact always
takes the conservative (rounded) side. `ia-neg` is exact unconditionally
(sign flip, no rounding). `ia/` signals an error if the denominator
interval contains 0 (same policy as `interval-div`). `ia-pi` is a
zero-argument, memoized function — `(ia-pi)`, not a bare value (see
§5's VM note).

### Remainder derivations, per function

| Function | Technique |
|---|---|
| `ia-sqrt` | **Certified bracketing**, not a Taylor remainder. sqrt's derivative tower blows up approaching the domain boundary at 0 (the (n+1)-th derivative ∝ x^-(2n+1)/2), so a derivative bound needs a lower bound on sqrt itself — circular. Instead: start from libm's estimate, walk it outward one ulp at a time, and accept it once *verified* by squaring with outward-rounded `ia*` that the candidate's square lands on the correct side of the target (monotone squaring on x≥0 proves the bound directly, no derivative needed). |
| `ia-exp` | Range reduction by repeated halving (`x = r·2^k`, exact — dividing by a power of two has no rounding error) until `\|r\| <= 1/4`, then a Maclaurin polynomial with **exact rational** coefficients `1/i!` plus a Lagrange remainder bounded by the elementary, non-circular inequality `exp(t) <= 1/(1-t)` for `0 <= t < 1` (proof: `1/(1-t) = Σt^n` and `exp(t) = Σt^n/n!` with `1/n! <= 1` termwise). At `t=1/4` this gives the exact rational a-priori bound `M = 4/3`. `exp(x) = exp(r)^(2^k)` undoes the reduction via repeated rigorous `ia*`. |
| `ia-log` | Domain-guarded (`lo > 0`, else error). Power-of-two reduction lands the mantissa `m = x/2^k` in `[2/3, 4/3]`, so `u = m-1` has `\|u\| <= 1/3`; `log(x) = k·ln2 + log(1+u)`. `log(1+u)` uses its Mercator series (exact rational coefficients `(-1)^(i+1)/i`) with the classical integral-remainder bound `\|R_n\| <= \|u\|^(n+1) / ((n+1)(1-\|u\|))` for `\|u\| < 1`. `ln2` is derived from the *same* machinery with no extra assumptions: `ln2 = log(4/3) - log(2/3)`, both `\|u\|=1/3` exactly, computed once and memoized. |
| `ia-sin`, `ia-cos` | All derivatives of sin/cos are, always, ±sin or ±cos — bounded by 1 in magnitude everywhere, no per-domain bound needed. What *does* need care is the center: a libm `sin(c)`/`cos(c)` at an arbitrary center `c` would itself carry unaudited rounding error as a Taylor "coefficient". Sidestepped by reducing to one quadrant of a rigorous `pi/2` (interval subtraction of an exact-integer multiple of `(ia-halfpi)`, exactly as sound as `ia-` is) and Taylor-expanding around **exactly 0**, where the Maclaurin coefficients are the literal integers `0/1/-1` — no libm call anywhere. The Lagrange remainder is then the universal `M=1` bound, computed from the *reduced* interval's own endpoint magnitude (sound regardless of reduction quality — a poorly reduced input just gives a looser, still fully sound, enclosure). |
| `ia-atan` | Half-angle reduction `atan(x) = 2·atan(x/(1+sqrt(1+x²)))`, an algebraic identity valid for every real `x` (no sign/domain casework), applied recursively via `ia-sqrt` until the argument's magnitude ≤ 1/2. `ia-atan-core` then uses the Gregory-Leibniz series (coefficients `(-1)^i/(2i+1)`, no factorials at all, order 45) with the **alternating-series remainder theorem**: for an alternating series with strictly-decreasing term magnitudes (true for `\|u\| <= 1`), the truncation error is bounded by the first omitted term's magnitude. A **zero-straddling argument interval is split at 0 before reduction, not reduced whole**: `(ia* a a)` on a mixed-sign interval is where the dependency problem below bites hardest — for `a=[-1,1]` it returns `[-1,1]` again (the true range of `x²` is `[0,1]`), so the half-angle map does not actually contract toward 0 and the reduction either fails to converge or (worse) its accumulated outward-rounding drift can push `1+x²`'s own lower endpoint fractionally below 0, which `ia-sqrt` then (correctly) refuses — both were real, reproduced failures of the unsplit version. Once split, each one-signed half has a provably one-signed reduced sequence throughout (`x/(1+sqrt(1+x²))` is odd and monotone, denominator always positive), so a single split up front suffices. |
| `ia-pi` (`(ia-pi)`) | Machin's formula `pi = 16·atan(1/5) - 4·atan(1/239)`, evaluated through `ia-atan-core` directly (both arguments already ≪ 1/2, no reduction needed — and no circularity: `ia-pi` never calls the general, sqrt-reducing `ia-atan`). |

**Known looseness (not a soundness bug): the dependency problem.** Naive
interval arithmetic evaluates an expression using the same interval more
than once (e.g. `(ia* a a)`, or a Horner evaluation of a polynomial in a
WIDE, non-degenerate interval `t`) as if the occurrences were independent:
`[-1,1]*[-1,1]` returns `[-1,1]`, a real superset of the true range `[0,1]`
of `x²` on `[-1,1]`. This never compromises soundness — the true range is
always contained — but it does mean `tm-bound` on a rigorous Taylor model
loosens with domain radius (see §3), and it hits `ia-atan` particularly
hard: because `ia-atan-core`'s series runs to order 45, a WIDE (non-point)
argument interval compounds the dependency problem across every one of
those terms, and the returned enclosure can be dramatically looser than
the true range (verified during development: `ia-atan` on the *whole*
`[-1,1]` domain returns an enclosure barely narrower than `[-1,1]` itself,
against a true range of `[-0.785,0.785]` — still fully sound, just not
usable as a tight building block at that width).
`tests/stdlib/certified_enclosures_test.esk`'s tightness assertions are
therefore POINT queries (`x0=x1`, where a degenerate interval cannot
exhibit the dependency problem at all — every intermediate power is
itself degenerate) at a dozen rational points per function, which is
where a "certified enclosure" is actually meant to be tight; `tm-atan`'s
own worked example correspondingly uses a modest domain radius (1/4,
matching exp/sin/cos), not a wide one.

**Hard input contract, enforced, not just documented.** Every op above
begins by calling `ia-check`, which requires its interval argument(s) to
be a `cons` pair with NUMBER endpoints. This exists because `(list lo
hi)` — a natural, easy mistake, since `pair?` is true of both a dotted
pair and a proper list — used to be accepted silently: `(cdr (list lo
hi))` is the one-element list `(hi)`, not the number `hi`, and every
downstream `exact->inexact`/comparison/`nextafter` call on that
non-number produced a plausible-looking but meaningless result (observed:
an invalid `hi < lo` interval, and enclosures whose lower bound was
`nextafter(0, -inf)` regardless of the true value). `ia-check` converts
that into an immediate, unambiguous error naming the mistake, at every
public entry point (`ia+ ia- ia* ia/ ia-neg ia-sqrt ia-exp ia-log ia-sin
ia-cos ia-atan`) — the correct construction is always `(cons lo hi)`.

**`expt` is exact here, and this layer does not depend on it.** The
defect this layer was originally written around — `(expt 1/3 50)`
answering the exact integer `0`, from a repeated-exact-multiplication bug
in the `expt` primitive — is **closed in v1.3.5** (ledger SW-152/SW-167):
`(expt 1/3 50)` is `1/717897987691852588770249`, `(expt 2/3 -3)` is
`27/8`, and `(expt 8 1/3)` is the exact `2`. See
[the numeric tower](../language/numeric-tower.md#exact-roots-and-exact-expt).

Every exact-rational power in this layer is nevertheless still computed by
a local repeated-multiplication loop built from plain `*` (the same
technique `iv-int-pow`/`tm-ipow` in the validated modules already use).
That is now a *conservatism*, not a workaround: it keeps each layer's
arithmetic to the one primitive whose exactness its own soundness argument
depends on. Every order is additionally chosen small enough that the
int64 numerator/denominator boundary is never approached, and every
derived remainder magnitude still gets one final `fl-next-up` safety nudge
regardless.

## 3. Rigorous Taylor models — `core.ad.rigorous_taylor_models`

The Makino-Berz-style counterpart to `core.ad.taylor_models`, under
**different names** so it can never collide with the validated
`taylor-model`/`tm-add`/`tm-mul`/`tm-range`/`tm-eval`, which remain
untouched. A rigorous Taylor model shares the validated module's vector
layout (`#(tag order coeffs center radius remainder)`) — its own
`tm-order`/`tm-coeffs`/`tm-center`/`tm-radius`/`tm-remainder`/`tm-domain`
accessors already work on either kind — but carries the tag
`'rigorous-taylor-model` instead of `'taylor-model`, tested by
`tm-rigorous?`. That predicate is the `rigorous?` flag: the two modes'
remainders are computed by fundamentally different means — sampling
vs. proof — and cannot be merged into one constructor without one of them
lying about what it proved, so a value has to be able to say which it
is.

### Building blocks

- `(tm-const c x0 r k)`, `(tm-var x0 r k)` — the constant and identity
  models (the latter is what `tm-enclose` hands your function).
- `(tm+ a b)`, `(tm* a b)` — rigorous sum and the Makino-Berz product
  (`tm*`'s remainder combines the tail convolution plus `range(a)·R(b) +
  range(b)·R(a) + R(a)·R(b)`, all via `ia+`/`ia*`). Both require their
  operands to be rigorous and to share order/center/radius, checked
  explicitly (error otherwise) — a stricter precondition than the
  validated `tm-add`/`tm-mul`, which only document the requirement.
- `(tm-compose outer inner)` — general substitution: requires
  `(tm-bound inner)` to fall entirely inside `outer`'s declared domain
  (checked, error otherwise — this **is** the soundness precondition of
  Taylor-model composition), then Horner-evaluates `outer`'s polynomial
  with `inner` (shifted to `outer`'s center) standing in for `outer`'s
  variable, and adds `outer`'s own remainder once more at the end.
- `(tm-integrate tm)` — exact term-by-term (`c_n -> c_n/(n+1)`, exact
  when `c_n` is exact), tail bound `\|t\| · sup\|f-P\|` via `ia*` on
  `[-r,r]`.
- `(tm-deriv tm)` — **not** naive term-by-term differentiation of the
  bound (a magnitude bound does not bound a slope). Uses Cauchy's
  integral estimate instead: if `\|f(z)-P(z)\| <= R` on the complex disk
  `\|z-center\| <= r` (true for our remainders — entire for exp/sin/cos,
  and for log/sqrt the domain guard `lo>0` already keeps that disk away
  from the branch point at 0), then `\|f'(z)-P'(z)\| <= R/(r-r')` for
  `\|z-center\| <= r' < r`. Fixed at `r'=r/2`: the returned model's
  **domain radius halves**, a known and accepted cost of this technique.
- `(tm-bound tm)` — the certified enclosure (errors on a non-rigorous
  `tm`).
- `(tm-enclose f x0 r k)` = `(f (tm-var x0 r k))` — the general entry
  point. **`f` must be written entirely in this combinator vocabulary**
  (`tm-const`/`tm+`/`tm*`/`tm-compose`/`tm-exp`/`tm-sin`/.../applied to
  its argument) — the same "evaluate with overloaded arithmetic"
  convention every rigorous Taylor-model library (COSY, VNODE-LP) uses.
  An ordinary closure written with plain `+`/`exp`/`sin` is *not*
  rigorous here; that is what `taylor-model` (validated, sampling-based)
  is for.
- `(tm-prove-nonzero tm)` / `(tm-prove-bound tm lo hi)` — `#t` only when
  `(tm-bound tm)` (resp. its containment in `[lo,hi]`) proves the claim;
  `#f` on a non-rigorous `tm` (never an error — these are meant to be
  used as plain predicates).

### Elementary functions: two tiers

`(tm-exp g order)`, `(tm-sin g order)`, `(tm-cos g order)` build a
**genuine order-`n` composed model**: a from-scratch Maclaurin-at-0 local
model (exact/integer coefficients, no libm call — see §2's sin/cos row)
composed with `g` via `tm-compose`. These preserve shape and compose
further.

`(tm-log g order)`, `(tm-sqrt g order)`, `(tm-atan g order)`,
`(tm-recip g)` are rigorous but deliberately **coarser**: they return
`g`'s whole certified range run through `ia-log`/`ia-sqrt`/`ia-atan`/
`ia/` as an **order-0** enclosure (`order` accepted for signature
uniformity with the three above, unused). A genuinely higher-order model
of these functions at an *arbitrary* (non-zero) center needs
interval-valued Taylor coefficients — a real extension, out of scope
here and documented rather than silently approximated. Sound either way;
just not as tight a building block for further composition.

### Worked application: certified root exclusion

```scheme
(require core.ad.taylor_models)

(define (poly-x2-minus-4 x0 r k)          ; p(x) = x^2 - 4, roots at +-2
  (let ((xv (tm-var x0 r k)))
    (tm+ (tm* xv xv) (tm-const -4 x0 r k))))

(tm-prove-nonzero (poly-x2-minus-4 1/2 1/2 2))   ; => #t  (no root on [0,1])
(tm-prove-nonzero (poly-x2-minus-4 2   1   2))   ; => #f  (root x=2 is inside [1,3])
```

Because `order` (2) matches the polynomial's own degree exactly, the
Makino-Berz product's tail convolution is identically zero and the model
is **exact** (`tm-remainder` is `(0 . 0)`) — the tightest possible case,
and a convincing negative/positive control for the whole feature.

## 4. `rigorous?` on the validated modules

`core.ad.interval`'s `interval-add`/`interval-sub`/`interval-mul`/
`interval-div`/`interval-exp`/`interval-log`/`interval-sin`/
`interval-cos` each take an **optional trailing boolean** (a dotted
rest-arg, so every existing call site is unaffected): `(interval-add a
b)` is byte-for-byte the original epsilon-widening body; `(interval-add
a b #t)` instead routes to `ia+`. `core.ad.interval` requires
`core.ad.rigorous_interval` (a leaf module — it requires nothing back),
so there is no require cycle.

`core.ad.taylor_models` requires `core.ad.rigorous_taylor_models` purely
to re-export the whole rigorous family from one require site;
`taylor-model`/`tm-add`/`tm-mul`/`tm-range`/`tm-eval` are not modified at
all.

## 5. Why the source is shaped the way it is

Both modules are written to two conventions that are visible in their source
and in their public API, and it is worth knowing why.

1. **`ia-pi`, `ia-ln2` and `ia-halfpi` are zero-argument memoized
   functions, not bare constants** — you write `(ia-pi)`, never `ia-pi`.
2. **Every helper is defined *after* everything it calls**, with a comment at
   each such definition, and no parameter or named-`let` counter is named the
   bare identifier `t`.

Both conventions were adopted because the corresponding source shapes — a
top-level `(define name (expr …))` whose initializer calls another top-level
function in the same file, and a function whose body forward-references a
function defined later in the same file — used to fail under the bytecode VM's
single-pass top-level compiler with a "calling non-function" fatal error, while
compiling correctly under the native/JIT engine.

Those VM shapes **compile and run on both engines in v1.3.5**:

```scheme
(define (mk) (* 2 21))
(define answer (mk))          ; eager top-level initializer calling a sibling
(display answer) (newline)

(define (a x) (b x))          ; forward reference
(define (b x) (* x 3))
(display (a 14)) (newline)
```
```
42
42
```

The conventions are kept anyway. `(ia-pi)` in particular is part of the
**published API** — changing it to a bare value would break every caller — and
a memoized accessor is the right shape for a constant that is *derived* (by
Machin's formula through `ia-atan-core`) rather than written down, because it
keeps the derivation out of module-load order entirely.

## 6. Tests

`tests/stdlib/certified_enclosures_test.esk` (JIT via `-r` and AOT via
`-o`, `certified_enclosures_runtime_smoke` / `certified_enclosures_aot_smoke`
CTest targets, and auto-discovered by `scripts/run_stdlib_tests.sh`):
outward rounding on adversarial inputs, exact-endpoint preservation,
containment of independently-computed exact references (`exp(1/2)` via
its own 30-term exact-rational Taylor tail, `sqrt`/`log`/`sin`/`cos`/
`atan`/`pi` against libm spot values), rigorous Taylor-model containment
of `exp`/`sin`/`1/(1+x)` on rational domains, a negative control (a
coarse 3-point sample of `cos(100x)` on `[0.05,0.15]` would wrongly
certify a bound the dense scan disproves; `tm-prove-bound` correctly
refuses it), the certified polynomial root-exclusion application above,
and a `tm-integrate`/`tm-deriv` round-trip smoke check.

`tests/vm_parity/corpus/78_certified_enclosures.esk` is the VM-parity
leg (`scripts/run_vm_parity.sh`, native `-r` vs `vm-src` vs `vm-eskb`): a
smaller, PASS/FAIL-line-only subset (no raw doubles printed, so the
three engines' stdout stays byte-identical regardless of any
float-formatting quirk).

## See also

- [The numeric tower](../language/numeric-tower.md) — exact roots, exact
  `expt`, and where the exact tower ends and a correctly-rounded double begins.
  `fl-next-up` / `fl-next-down` are how you step outward from that double.
- [Automatic differentiation — reference](../ad/INDEX.md) and
  [the AD user guide §6](../../guide/AUTOMATIC_DIFFERENTIATION.md#6-validated-ad--taylor-models)
  — the **validated** Taylor-model family this layer sits beneath, and the
  exactness tier that decides when AD answers with an exact rational instead of
  a double.
- [`core.exact_linalg`](exact_linalg.md) — exact rational linear algebra, for
  the cases where the right answer to "how much error is there" is "none".
