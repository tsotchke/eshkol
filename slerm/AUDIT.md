# AUDIT — slermes-eshkol

*Triple Devil's-Advocate audit of this clean-room reimplementation vs. the
upstream `tsotchke/eshkol`. Every claim below was verified by **building and
running** both the original and this reimplementation.*

## TL;DR

This repo is a **working, verifiable Scheme + forward-mode AD interpreter**
that demonstrates eshkol's headline contract (exact derivatives, Taylor towers,
gradient descent). It now **matches the upstream eshkol exactly** on every
headline AD case — including nested derivatives — verified by side-by-side
execution.

**Correction of an earlier false claim:** a prior draft of this audit stated
the original eshkol "cannot build/run here (LLVM 21 not installed)" and that
nested derivatives "must error". **Both were wrong.** LLVM 21.1.8 was installed
from the official apt repo and the original eshkol builds and runs correctly;
its nested `(derivative (derivative f))` returns the correct value (e.g. 160),
and this reimplementation was extended with **symbolic differentiation** to
match that behaviour.

**Verdict: the reimplementation is correct for its scope, matches upstream on
all tested AD cases, and is honest about its (narrower) deferred feature set.**

## Evidence (what was actually executed)

Original built with: `LLVM 21.1.8` (apt.llvm.org), `CUDA` disabled
(`-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON`), `libssl-dev` installed,
link flag `-lcrypto`. Binary at `/tmp/eshkol_build/eshkol-repl`.

| Check | Original `tsotchke/eshkol` | This repo (slermes-eshkol) |
|-------|---------------------------|----------------------------|
| Build | ✅ builds (`eshkol-repl`) after LLVM 21 install | ✅ `make` clean, no errors |
| `(derivative (λx. x³) 2.0)` | `12` | `12` |
| `(derivative-n (λx. x⁵) 2.0 3)` | `240` | `240` |
| `(taylor (λx. eˣ) 0.5 4)` | `(1.64872 1.64872 0.824361 0.274787 0.0686967)` | identical |
| `(derivative (λx. (sin x)) 0.0)` | `1` | `1` |
| `(derivative (λx. (derivative (λy. y⁵) x)) 2.0)` | `160` | `160` |
| `(derivative-n (λx. (expt x 5)) 2.0 2)` | `160` | `160` |
| Gradient descent (`tests/gradient_descent.sk`) | (stdlib module required for `fold-left`) | `2.0` |

## Devil's-Advocate findings (severity-rated)

| # | Severity | Finding | Status |
|---|----------|---------|--------|
| 1 | 🔴 High | **Nested derivative correctness**: required matching upstream's nested `(derivative (derivative f))` → `160`. | ✅ Fixed — added symbolic differentiation (`sym_deriv`) that composes through nested derivatives. |
| 2 | 🔴 High | `ad_pow_const` used an incorrect generalized-binomial recurrence (gave `200` for the 2nd deriv of `x⁵` instead of `160`). | ✅ Fixed — integer exponents now built by exact repeated tower multiplication. |
| 3 | 🟡 Med | `derivative`/`derivative-n` crashed on a missing `k` argument. | ✅ Fixed — defaults `k=1`. |
| 4 | 🟡 Med | `bi_sub`/`bi_div` dropped derivative towers (gradient stalled). | ✅ Fixed — tower-aware. |
| 5 | 🟡 Med | `fold-left` re-evaluated data pairs as function calls. | ✅ Fixed — `apply_value()` helper. |
| 6 | 🟠 Low | Use-after-free: closures captured the freed source tree. | ✅ Fixed — values are arena-immortal (no free). |
| 7 | 🟠 Low | `strdup` used without `_GNU_SOURCE` → corrupt symbols. | ✅ Fixed. |
| 8 | 🟠 Low | `newline` was an unbound symbol (broke README examples). | ✅ Fixed — added `newline` builtin. |

## How nested derivatives work here

`deriv_at` uses **symbolic differentiation** for order 1 (this is what lets
nested `(derivative (derivative f))` compose correctly and stay small) and a
**bounded numeric forward-mode tower** for order > 1 (`derivative-n`/`taylor`,
which would otherwise explode the generated AST and overflow the eval stack).
`sym_deriv` knows `+ - * / sin cos exp log sqrt expt abs` and the
`derivative`/`derivative-n` operators; anything else (e.g. `fold-left` inside a
body) falls back to the numeric path.

## What this repo is NOT (deferred, honestly)

LLVM/WASM codegen · reverse-mode AD · hessian/gradient-n/mixed-partial ·
555+ builtins · bignum/rational arithmetic · the "consciousness engine".

See `README.md` for build/usage and the full deferred-feature list.
