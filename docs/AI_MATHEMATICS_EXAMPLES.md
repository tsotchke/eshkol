# AI-driven mathematics examples

Status: verified on the Linux x86-64 lite lane with native AOT, native JIT, and
the hosted bytecode VM.

These examples reproduce public finite or algebraic witnesses. They do not
claim to rediscover the results: each program checks the supplied witness with
exact integer or rational arithmetic, or with AD where the example explicitly
uses floating-point inputs.

## Jacobian conjecture counterexample

Source: <https://www.ulam.ai/research/jacobian.pdf>

`examples/mathematics_jacobian_counterexample.esk` verifies the map

```text
A = 1 + xy
B = A^2 z + y^2(4 + 3xy)
P = AB
Q = y + 3xB
R = 2x - 3x^2y - x^3z
```

It checks the three rational preimages of `(-1/4,0,0)`, the binary cubic
`2pS^3 - qS^2T + 2ST^2 - rT^3`, the discriminant fiber-count cases, reverse
mode AD at multiple double-valued points, and an exact determinant identity.
The identity check evaluates the explicit rational partial derivatives on a
`9 x 8 x 3` grid. The determinant has per-variable degree bounds `(8,7,2)`,
so this grid is larger than the bound in every variable.

The Theorem 5.1 family is also checked for two parameter choices, with the
predicted determinant `-2c lambda^2`.

The AD checks intentionally use double vectors. Exact-rational vector inputs
remain tracked by `SW-136`; the program contains the required TODO at the
replacement point.

## AlphaTensor matrix multiplication

Source: <https://github.com/google-deepmind/alphatensor>

`examples/mathematics_alphatensor_3x3_gf2.esk` and
`examples/mathematics_alphatensor_gf2.esk` contract the public rank-23 3x3 and
rank-47 4x4 factorizations over `F_2`. Factor rows and columns are represented
as exact integer bit masks. Because the map is bilinear, expansion on every
pair of matrix units is a complete exact tensor check: 81 pairs for 3x3 and
256 pairs for 4x4.

## FunSearch cap set

Source: <https://github.com/google-deepmind/funsearch>

`examples/mathematics_funsearch_cap_set.esk` implements the public explicit
construction of 512 points in `AG(8,3)`. It enumerates the 3^8 ambient points,
selects the four construction classes, and checks every unordered pair against
the third point on its affine line. The resulting 130,816 exact pair checks
must find no third point in the set.

## Running the family

The normal examples suite discovers these files automatically because the
repository's examples convention is a flat set of `.esk` files:

```bash
./scripts/run_examples_tests.sh
```

For mode-specific checks, use the same `eshkol-run` binary with its AOT, JIT,
and VM options as described in the runtime reference. The examples suite is
part of `scripts/run_all_tests.sh`, so these files run in CI with the other
public examples.
