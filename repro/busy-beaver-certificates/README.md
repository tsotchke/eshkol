# Source-bound Busy Beaver resource certificates

Two MIT Eshkol examples compose existing exact ring and polynomial owners:

* A symbolic word proof reconstructs the known five-state champion's complete
  47,176,870-transition blank-tape halt in fifteen macro judgments. The checker
  verifies the literal transition table, monotone sweeps, parametric tape
  framing, summation induction and three residue branches.
* The six-state A.4 checker's six-coefficient cost algebra verifies its 5-by-5
  matrix embedding over twelve independent polynomial variables. Sixteen exact
  dense/structured powers cover both base operators, through exponent 4,608.
  A polynomial control distinguishes reversed composition. This verifies the
  algebra, not the complete A.4 word theorem or orbit.

The five-state table is `1RB1LC_1RC1RB_1RD0LE_1LA1LD_1RZ0LA`.
Its established laws appear in
[Michel, section 5.2.1](https://arxiv.org/html/0906.3749v7#S5.SS2.SSS1).
One cost unit is one raw write/move transition, including entry into halt.
This is neither a new Busy Beaver bound nor a lambda beta-cost theorem.
Word framing and induction are part of the trusted executable kernel,
not proof-assistant formalization or automatic law discovery.

```sh
python3 run_gate.py --eshkol-bin /path/to/eshkol-run --source-root /path/to/eshkol
```

Use PR #865 at `fb8a9415b3dd05bb82ebf42db2fe894240f8b4f0`.
Prerequisites: Python 3, C11 compiler and Eshkol. The independent C reference
executes every raw transition. The gate compares all 131,072 final tape cells,
halt state, head -12,242 and 4,098 ones with the certificate. Eighteen exact
macro queries include parameters through 2^1,024; their huge tapes are not
materialized. Certificate checking never calls the direct simulator.

Controls reject altered machine source, sweeps, affine/summed/composed costs,
inexact polynomials, possibly empty symbolic reads and invalid/budgeted inputs.
JIT and separately linked AOT must produce identical complete output.
The released runtime duplicates the sign of some printed negative bignums;
query transport explicitly prints a sign and nonnegative magnitude.

`results.json` records the measured scope. No Eshkol speedup is inferred
from the original implementation's timings. The MIT grant covers this packet
only. VM/browser, the six-state word theorem and its long orbit are outside
this qualification. The CTest wrapper runs the exact example and a
readiness self-test without requiring a new public API.

The focused CTest wrapper targets POSIX native builds; Windows is unqualified.

Run `python3 self_test.py` to demonstrate that the actual readiness gate
rejects a successful backend returning incorrect data. Infrastructure failures
are distinguished from the expected rejection.


The gate explicitly sets `ESHKOL_JIT_CACHE=0` in its child environment so the
`-r` leg cannot inherit the persistent executable-cache route. `self_test.py`
intercepts the gate's first child boundary with that setting unset and with it
enabled in the caller; both cases must pass `0` to the child without mutating
the caller. This contained environment check starts no compiler or Busy Beaver
workload.

The included `results.json` is the earlier certificate/reference-tape receipt.
It predates this explicit cache policy and alone does not establish an
independent in-process JIT run. Mathematical certificate checks, the complete
reference-tape comparison and a fresh native run of the corrected gate remain
separate qualification steps; this environment correction does not reissue
those results.
