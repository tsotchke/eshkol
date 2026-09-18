---
kind: explanation
status: current
owner-area: ad
since: v1.3.5
sources:
  - inc/eshkol/eshkol.h
  - lib/backend/arithmetic_codegen.cpp
  - lib/backend/complex_codegen.cpp
  - lib/backend/vm_complex.c
  - lib/backend/vm_native.c
  - .icc/ledger/entries/SW-180.yaml
---
# ADR-0025: A complex value carries the derivative of its components

**Status:** Accepted
**Ledger:** SW-180; follow-up SW-191
**Scope:** Native LLVM code generation (JIT and AOT) and the bytecode VM.

## Context

A complex number under real-parameter differentiation is a pair of real
numbers, and each of them may carry a derivative. `make-rectangular` and
`make-polar` took the primal of their arguments on both engines, so a
derivative entering a complex value was dropped at construction and every
derivative through complex arithmetic was 0, with exit status 0 (SW-180).
The Butterworth filter design in `lib/signal/filters.esk` computes its poles
this way, so every derivative with respect to a filter cutoff was 0.

Two further boundaries dropped the same derivative: the VM's generic
arithmetic tested the dual carrier before complex, so a dual meeting a
complex value was read as a real number, and `real-part`, `conjugate` and
`magnitude` of a real dual flattened it to its primal.

## Decision

Native. A complex value whose component carries a derivative keeps the plain
`{real, imag}` pair first and appends the two components as tagged values
(`eshkol_complex_carrier_t`), marked by `ESHKOL_COMPLEX_CARRIER_FLAG` in the
flags byte. A reader that does not know about carriers (the printer,
equality, the FFT) reads a correct complex number from the same pointer. Every
complex operation on such a value is its component formula evaluated by the
generic real arithmetic and the value-level math functions, so a forward jet,
a reverse-tape node and a Taylor tower each flow through with the rules they
already have and no operation needs a rule per carrier.
`ArithmeticCodegen::makeRectangular` is the one constructor;
`withComplexCarrierDispatch` wraps `+ - * /` ahead of the real-carrier
dispatch; one helper per builtin kind serves `real-part`, `imag-part`,
`magnitude`, `angle`, `conjugate`, `make-polar` and `exp`, `log`, `sqrt`,
`sin`, `cos`, `tan`. A real carrier is a complex number with no imaginary
part and is no longer flattened.

VM. The VM's forward carrier is a first-order dual, so `VmComplex` carries the
matching tangent pair and each VM complex operation propagates it by its own
rule in `vm_complex.c`: sum, product and quotient rules, `f'(z) dz` for the
holomorphic functions, and explicit rules for `conjugate`, `magnitude` and
`angle`. The payload stays plain doubles, so no heap walker needs a case.
`vm_real_with_tangent` is the one boundary that lifts a real operand into
complex arithmetic; a carrier this representation cannot hold (a Taylor tower,
a hyper-dual) is refused with a message. `vm_op_arith` sends a complex
operand to the complex path before the dual path. The same boundary gives
`atan`, `atan2`, `asin`, `acos`, `sinh` and `cosh` their first-order rules;
they had read a dual as its primal.

Where no derivative exists (`magnitude` and `angle` at `0+0i`, `log` and
`sqrt` at the origin, `expt` at a zero base) and a derivative is flowing, the
operation raises. A derivative is never answered as 0.

## Consequences

- `(derivative (lambda (x) (magnitude (make-rectangular x 4.0))) 3.0)` is
  `0.6` on both engines, forward and reverse, and the Butterworth cutoff
  derivatives are exact (`tests/ad/complex_value_derivative_test.esk`, corpus
  program 89).
- `derivative` stays an operator over the reals: a complex evaluation point is
  refused on both engines (the VM had answered 0). Complex-point
  differentiation is a build item awaiting a ruling on its contract for
  non-holomorphic functions (SW-191).
- The native carrier holds components of any carrier kind; the VM's holds a
  first-order tangent, the same order its dual has.
