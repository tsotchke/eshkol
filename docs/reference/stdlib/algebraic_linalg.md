---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/algebraic_domain.esk
  - lib/core/algebraic_linalg.esk
  - tests/stdlib/algebraic_linalg_test.esk
---

The [matrix calculus example](../../../examples/algebraic_matrix_calculus.esk)
shows repeated right-hand-side solves over a selected field and a Taylor
coefficient solve with the same bottom-field factorization.

`algebraic-factor-solve-transpose` reuses the retained LU for `A^T*x=b`,
including nested coefficient recurrences and a full transposed residual.
