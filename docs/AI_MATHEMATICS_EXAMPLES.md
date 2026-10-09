# AI-driven mathematics examples

Status: SHIPPED. At release SHA `60f345def` all four programs print
`RESULT: ALL PASS` with `Failed: 0` in native JIT, native AOT and the hosted
bytecode VM (macOS arm64, Release build; the per-program outcomes and timings are
in [the mathematics catalogue](MATHEMATICS_EXAMPLES.md) and
`docs/examples/measurements.json`). The family was first verified on the Linux
x86-64 lite lane with native AOT, native JIT, and the hosted bytecode VM; that
earlier lane was not re-measured for this release.

These examples reproduce public finite or algebraic witnesses. They do not
claim to rediscover the results: each program checks the supplied witness with
exact integer or rational arithmetic, or with AD where the example explicitly
uses floating-point inputs.

## Jacobian conjecture counterexample

Source: "A counterexample to the Jacobian conjecture", research note dated
July 20, 2026, <https://www.ulam.ai/research/jacobian.pdf>. The extracted PDF
text carries no author line; the note credits the explicit formula to an
announcement by L. Alpöge. Its Theorem 3.1 states that the map `F = (P, Q, R)`
below has `det Jac F = -2` and that
`F(0, 0, -1/4) = F(1, -3/2, 13/2) = F(-1, 3/2, 13/2) = (-1/4, 0, 0)`, so `F` is
a Keller map that is not injective, hence not a polynomial automorphism of
`C^3`; by stabilization `(x, y, z, u_4, ..., u_n) -> (F(x, y, z), u_4, ..., u_n)`
it gives a counterexample in every dimension `n >= 3`. The note also shows the
image of `F` is Zariski open with complement of codimension two, and gives the
families of Theorem 5.1.

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
mode AD through a nested closure factory at both double-valued points and
exact-rational vector inputs, and an exact determinant identity.
The identity check evaluates the explicit rational partial derivatives on a
`9 x 8 x 3` grid. The determinant has per-variable degree bounds `(8,7,2)`,
so this grid is larger than the bound in every variable.

The Theorem 5.1 family is also checked for two parameter choices, with the
predicted determinant `-2c lambda^2`.

The AD determinant checks use a tolerance of `1e-9`, including when the input
vector contains exact rationals: the AD carrier is inexact internally. The
separate determinant identity uses explicit polynomial partial derivatives
and exact rational arithmetic over the degree-bounded grid. These are two
different checks; the tolerance-based AD result is not the exact-arithmetic
identity certificate.

The exact-grid argument is a complete proof of the identity, not a sample: a
polynomial in `(x, y, z)` of degree at most `(8, 7, 2)` in the three variables
that vanishes on a `9 x 8 x 3` grid of distinct points is identically zero, so
`det Jac F + 2` vanishing on the grid proves `det Jac F = -2` everywhere. The
three preimages are checked by exact rational evaluation. At `60f345def` the
program passes all 12 checks in JIT, AOT and VM.

## AlphaTensor matrix multiplication

Source: A. Fawzi, M. Balog, A. Huang et al., "Discovering faster matrix
multiplication algorithms with reinforcement learning", Nature 610, 47–53
(2022), <https://doi.org/10.1038/s41586-022-05172-4>; factorizations published
at <https://github.com/google-deepmind/alphatensor>. Over `F_2` the paper's
rank-47 decomposition of the 4x4 multiplication tensor improves on the rank 49
of Strassen's algorithm applied recursively; the rank 23 for 3x3 equals
Laderman's 1976 bound and is included as a second, smaller witness.

`examples/mathematics_alphatensor_3x3_gf2.esk` and
`examples/mathematics_alphatensor_gf2.esk` contract the public rank-23 3x3 and
rank-47 4x4 factorizations over `F_2`. Factor rows and columns are represented
as exact integer bit masks. Because the map is bilinear, expansion on every
pair of matrix units is a complete exact tensor check: 81 pairs for 3x3 and
256 pairs for 4x4. Both programs print exactly those counts at `60f345def`
(`Basis pairs checked: 81` and `Basis pairs checked: 256`). The check certifies
the supplied decompositions over `F_2`; it neither searches for them nor proves
that 23 or 47 is the minimal rank.

## FunSearch cap set

Source: B. Romera-Paredes, M. Barekatain, A. Novikov et al., "Mathematical
discoveries from program search with large language models", Nature 625,
468–475 (2024), <https://doi.org/10.1038/s41586-023-06924-6>; witnesses
published at <https://github.com/google-deepmind/funsearch>. FunSearch found a
cap set of size 512 in `F_3^8`, larger than the previously best known 496.

`examples/mathematics_funsearch_cap_set.esk` implements the public explicit
construction of 512 points in `AG(8,3)`. It enumerates the 3^8 ambient points,
selects the four construction classes, and checks every unordered pair against
the third point on its affine line. The resulting 130,816 exact pair checks
must find no third point in the set; at `60f345def` the program prints
`Exact pair checks: 130816` and passes. This certifies that the 512 points form
a cap; it does not rerun FunSearch or bound the maximum cap size in `AG(8,3)`.

## Running the family

Each program prints one `PASS:`/`FAIL:` line per check, `Passed:`/`Failed:`
counts and a final `RESULT: ALL PASS` or `RESULT: FAILURES DETECTED`. The
verdict is the program's `RESULT:` line, and the measured outcomes in the
catalogue read it. Each process also exits 0 on `RESULT: ALL PASS` and 1 on
`RESULT: FAILURES DETECTED`, so a failing check is a nonzero exit as well as
a printed line.

The normal examples suite discovers these files automatically because the
repository's examples convention is a flat set of `.esk` files:

```bash
./scripts/run_examples_tests.sh
```

For mode-specific checks, use `eshkol-run -r` (JIT) and `eshkol-run -o`
(AOT) as in the commands below, as described in the runtime reference. The VM
outcome above was measured with the hosted VM harness that the CTest `vm_*`
entries use:

```bash
ESHKOL_VM_NO_DISASM=1 ./build/eshkol-vm-standalone-test examples/mathematics_jacobian_counterexample.esk
``` The examples suite is
part of `scripts/run_all_tests.sh`, so these files run in CI with the other
public examples.

<!-- example-catalogue:ai-inventory:start -->
The catalogue contains **4 public witness programs** in this family. Every one of these 4 programs was run at release SHA `60f345def` (Eshkol Compiler v1.3.6-evolve, Release build, macOS arm64, measured 2026-10-09) in native JIT and native AOT. **4** completed in both modes; **4** of those printed their own verdict `RESULT: ALL PASS` with `Failed: 0` in both modes. Wall times measured with 6 programs running concurrently on a shared machine; CPU user/sys from wait4. Each complete description separates exact identity checks from numerical AD comparisons.

- [Rank-23 matrix multiplication over F₂](MATHEMATICS_EXAMPLES.md#mathematics-alphatensor-3x3-gf2): [mathematics_alphatensor_3x3_gf2.esk](../examples/mathematics_alphatensor_3x3_gf2.esk#L1); measured JIT / AOT: PASS / PASS.
- [Rank-47 matrix multiplication over F₂](MATHEMATICS_EXAMPLES.md#mathematics-alphatensor-gf2): [mathematics_alphatensor_gf2.esk](../examples/mathematics_alphatensor_gf2.esk#L1); measured JIT / AOT: PASS / PASS.
- [Explicit 512-cap in AG(8,3)](MATHEMATICS_EXAMPLES.md#mathematics-funsearch-cap-set): [mathematics_funsearch_cap_set.esk](../examples/mathematics_funsearch_cap_set.esk#L1); measured JIT / AOT: PASS / PASS.
- [Polynomial Jacobian map and fibers](MATHEMATICS_EXAMPLES.md#mathematics-jacobian-counterexample): [mathematics_jacobian_counterexample.esk](../examples/mathematics_jacobian_counterexample.esk#L1); measured JIT / AOT: PASS / PASS.

```bash
mkdir -p .scratch/example-manual
./build/eshkol-run -r examples/mathematics_alphatensor_3x3_gf2.esk
./build/eshkol-run -L./build examples/mathematics_alphatensor_3x3_gf2.esk -o .scratch/example-manual/mathematics_alphatensor_3x3_gf2 && .scratch/example-manual/mathematics_alphatensor_3x3_gf2

./build/eshkol-run -r examples/mathematics_alphatensor_gf2.esk
./build/eshkol-run -L./build examples/mathematics_alphatensor_gf2.esk -o .scratch/example-manual/mathematics_alphatensor_gf2 && .scratch/example-manual/mathematics_alphatensor_gf2

./build/eshkol-run -r examples/mathematics_funsearch_cap_set.esk
./build/eshkol-run -L./build examples/mathematics_funsearch_cap_set.esk -o .scratch/example-manual/mathematics_funsearch_cap_set && .scratch/example-manual/mathematics_funsearch_cap_set

./build/eshkol-run -r examples/mathematics_jacobian_counterexample.esk
./build/eshkol-run -L./build examples/mathematics_jacobian_counterexample.esk -o .scratch/example-manual/mathematics_jacobian_counterexample && .scratch/example-manual/mathematics_jacobian_counterexample
```
<!-- example-catalogue:ai-inventory:end -->
