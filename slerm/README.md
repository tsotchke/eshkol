# slerm — clean-room reimplementation of `tsotchke/eshkol`

This folder is the **clean-room, from-scratch C11 reimplementation** that backs
the audit in this fork's root `AUDIT.md` / `REPORT.md`. It is *not* a fork of
upstream — it was written independently to verify eshkol's headline contract
(exact forward-mode automatic differentiation) by actually running it and
comparing against the original.

## What it demonstrates

A small Scheme + AD interpreter (`slermes-eshkol`) implementing:

- `derivative f x0` → `f'(x0)` (exact for smooth `f`)
- `derivative-n f x0 k` → `f^{(k)}(x0)`
- `taylor f x0 k` → Taylor coefficients `f^{(m)}(x0)/m!`
- **Nested derivatives** `(derivative (derivative f))` — supported via
  symbolic differentiation, matching upstream eshkol.
- Gradient descent (see `tests/gradient_descent.sk` → `2`).

## Build & run

```bash
make                 # builds ./slermes-eshkol
./slermes-eshkol tests/gradient_descent.sk
./slermes-eshkol tests/negative.sk
```

Requires `gcc` and `-lm`. No LLVM needed (unlike upstream).

## Parity with the original (verified by execution)

| Case | Original `tsotchke/eshkol` | This slerm |
|------|----------------------------|------------|
| `(derivative x³ 2.0)` | `12` | `12` |
| `(derivative-n x⁵ 2.0 3)` | `240` | `240` |
| `(taylor eˣ 0.5 4)` | `(1.64872 1.64872 0.824361 0.274787 0.0686967)` | identical |
| `(derivative sin 0)` | `1` | `1` |
| `(derivative cos 0)` | `0` | `0` |
| `(taylor sin 0 4)` | `(0 1 0 −0.166667 0)` | identical |
| `(taylor cos 0 4)` | `(1 0 −0.5 0 0.0416667)` | identical |
| `(derivative (derivative y⁵) x) 2.0` | `160` | `160` |
| `(derivative-n (expt x 5) 2.0 2)` | `160` | `160` |
| gradient descent | `2` (with `--stdlib`) | `2` |

## Why this exists

The audit's job is to show, by running *both* the original and an independent
implementation, that eshkol's AD claims hold (and where they don't). This slerm
is the independent implementation. Deferred (documented, not claimed): LLVM/WASM
codegen, reverse-mode AD, the 555+ builtin library, the "consciousness engine".

See `AUDIT.md` (this folder) for the full finding list, and the fork root
`AUDIT.md` for the cross-validation against `WuBuMath`.
