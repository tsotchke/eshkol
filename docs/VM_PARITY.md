# VM Parity Ratchet

Eshkol ships two executable back ends:

- the **native LLVM codegen** (`lib/backend/llvm_codegen.cpp`), used by
  `eshkol-run` for both `-r` (JIT) and AOT builds; and
- the **bytecode VM** (`lib/backend/vm_compiler.c`, `vm_native.c`,
  `eshkol_vm.c`, `eshkol-vm-standalone`, the ESKB format and the `hosted-vm`
  profile), used for the browser/WASM playground and embedded hosting.

The VM implements a *subset* of the language. Before v1.3.0-evolve that subset
was undeclared: nothing forced a decision when a feature landed in the codegen
but not the VM, and nothing recorded which shared behaviors silently diverged.
The **VM parity ratchet** makes the subset explicit and makes drift impossible
to miss.

> Status: the ratchet, manifest, and gate shipped with the v1.3.0-evolve
> release (PR #118 — `scripts/run_vm_parity.sh`,
> `scripts/vm_parity_audit.py`, `tests/vm_parity/`). The counts below are from
> the v1.3.5-evolve audit.

### v1.3.5-evolve parity backlog controls

- **Arity probes (PR-03).** `scripts/p8/p8_arity_sweep.py` generates one
  compilation unit per builtin and probe kind, then compares the canonical
  value or fatal-diagnostic class across native JIT, the optional native AOT
  route, and the hosted VM. A compile-time native refusal and a runtime VM
  refusal are both `FATAL:arity`; neither is allowed to become a value or a
  swallowed `ERR`. Timeouts remain explicitly unmeasured and cannot erase a
  prior baseline entry.
- **One canonical arity diagnostic, one arity fact.** `FATAL:arity` above is a
  claim about the text an engine prints, so the text has a single source:
  `inc/eshkol/core/arity_contract.h` owns the class marker
  (`Arity mismatch: `) and the canonical sentence
  (`<procedure> expects N argument(s) but got M`). The bytecode VM's compiler
  renders it, the LLVM backend renders it, and
  `eshkol_arity_error_current()` prepends the marker to the ~200 lowering
  guards that have something more specific to say
  (`string->utf8 requires 1 to 3 arguments`). Native lowering used to answer
  `(ceiling)` with `ceil requires exactly 1 argument` — the LLVM intrinsic's
  name, no contract named — so the ratchet could not tell that both engines
  had refused the call, and `ceiling`, `char->integer`, `exact->inexact`,
  `numerator` and `tanh` read as native-vs-VM divergences over behaviour that
  was already identical.

  The *number* is single-sourced too. `BUILTINS[]` in
  `lib/backend/eshkol_vm.c` is the one arity table — it is also what
  `scripts/gen_language_surface.py` turns into
  `tests/coverage/language_surface.json` — and both engines read it through
  `eshkol_builtin_min_arity()`. `llvm_codegen.cpp` no longer transcribes those
  numbers into a map of its own; it names only the builtins whose lowering has
  no arity guard, and `scripts/check_builtin_min_arity.py` fails the build if
  one of those names stops being backed by the table.

  That scope is a subset of the table on purpose. `arity` in `BUILTINS[]` is
  the OPCODE'S OPERAND COUNT, not the caller's obligation, and the VM applies
  it only where the call actually reaches the raw op: a name the VM compiler
  special-cases (`round`, `string->utf8`) never does.
- **A DOCUMENTED OPTIONAL ARGUMENT IS A LEGAL CALL ON BOTH ENGINES.** The
  caller-minimum column the note above called for now exists, and it is
  derived rather than typed: `scripts/gen_builtin_min_arity.py` reads the
  fixed-arity macro the native dispatch expands
  (`THREE_ARG_BUILTIN(stringPadLeft, …)` is an exact arity, stated by the code
  that runs), the arity guard the lowering enforces (`substring requires 2 or 3
  arguments: (substring string start [end])`), and the declarative documented
  signature, in that order of authority, and writes the answer into
  `min_arity` for all 742 rows. `scripts/check_builtin_min_arity.py` imports
  that same module and re-derives, so editing a guard or a signature without
  regenerating the table fails the build rather than opening a new divergence.

  Nineteen rows carry an argument a caller may omit. For those the VM used to
  refuse `(substring s 1)`, `(make-vector 3)`, `(make-string 3)`,
  `(read-line)`, `(append)`, `(append lst)`, `(bytevector-append)` and
  `(hash-ref table key)` — all legal, all accepted by native. Three of them are
  the R7RS variadic FOLDS, and they are derived from the code native runs
  rather than from prose: `append` from the core-module definition native
  compiles (`(define (append . lists) (cond ((null? lists) '()) …))`),
  `bytevector-append` and `gcd`/`lcm` from their lowering's own `num_vars == 0`
  case. A row DECLARED variadic was the worse half of the defect: it makes no
  minimum claim, so the compile-time check never fired for it, and `(gcd)` got
  all the way to the closure call and died there wanting both operands. REFUSING WAS ONLY HALF OF IT: the VM's builtin body
  loads a fixed number of operands and its runtime closure check wants exactly
  that many, so `hash-ref`, whose `min_arity` had been filled in by hand, got
  past the compiler and then died at runtime with
  `arity mismatch: expected 3 arguments, got 2`. The call site therefore
  supplies the omitted operands as `ESHKOL_ABSENT_ARG` — the VM's unspecified
  value, which no source expression evaluates to — and the native op reads that
  marker and applies the DOCUMENTED DEFAULT: `substring`'s missing `end` is the
  string's length, `make-vector`'s missing fill is `0`, `read-line`'s missing
  port is the current input. Lowering a row's minimum without teaching its op
  that default would turn a refused call into a wrong answer, so the two always
  land in the same change.
  `tests/vm_parity/corpus/82_builtin_optional_argument_arity.esk` calls every
  one of the nineteen at its minimum and at its maximum arity;
  `tests/diagnostics/arity_below_documented_minimum` pins the refusal BELOW the
  minimum to the canonical sentence, quoting the minimum (`substring expects 2
  arguments`) rather than the opcode's three operands, on both engines.
- **Canonical gap evidence (PR-05).** Every `gap` row in `PARITY.tsv` has a
  matching row in `tests/vm_parity/GAP_DISPOSITIONS.tsv`. The sidecar records
  an explicit disposition, a live `found/` reproducer when one exists, or the
  deterministic generated probe route used for an unimplemented surface. The
  sidecar currently contains **330** rows; `op:LET_VALUES` was promoted to
  `vm-supported` after `corpus/28_multiple_values_complete.esk` verified both
  VM routes.
- **Differential floor (PR-10).** `scripts/run_engine_parity_coverage.py`
  reports both overall and high-risk differential evidence. The readiness
  event is `PASS` only when the monotonic overall floor and the high-risk
  floor in `ENGINE_PARITY_BASELINE.json` hold, with no new divergence and no
  regression of a program previously observed on both engines. A passing
  name-resolution or one-engine coverage run cannot satisfy this criterion.
- **Per-form VM evidence (D-03 (i)).** The VM used to record coverage only
  from builtin dispatch, so every construct it lowers inline — the arithmetic
  and comparison opcodes, and `if`/`let`/`cond`/`do`/`lambda` — earned no
  differential credit at all and `(display (+ 1 2))` produced no VM trace
  file. The VM compiler now emits `OP_LANGUAGE_COVERAGE_FORM` at the head of
  every compiled `(name ...)` form when
  `ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR` is set, carrying the same stable
  31-bit head-symbol hash the call marker uses, and the marker survives ESKB
  serialization so the standalone VM binary and the `--profile hosted-vm`
  route report identically. Differential coverage measured 303/1137 (26.65%)
  and high-risk 152/473 (32.14%) on `integration/astra-v135`.
- **The high-risk floor is a measured ratchet, not a literal (PR-13).**
  `ENGINE_PARITY_BASELINE.json`'s `high_risk_differential_floor` used to be a
  hardcoded `1.0` (100%), written by `--update-baseline` as a literal rather
  than read from any run's own output — it had never once been measured,
  because it could not be: running native alone over the gate's default
  corpus, the parser records only 171 of 473 high-risk constructs (36.15%) —
  the **corpus ceiling** — since a construct no corpus program mentions under
  native can never earn differential credit on either engine no matter what
  the VM implements. A floor above that ceiling was unreachable by
  construction from the day it was written, and
  `engine_semantic_parity_threshold` failed on every run since.
  `scripts/run_engine_parity_coverage.py --update-baseline` now writes both
  `differential_floor` and `high_risk_differential_floor` from this exact
  run's own measured fractions — the same ratchet discipline
  `differential_floor` already had — and both `run_engine_parity_coverage.py`
  and `scripts/check_engine_parity_threshold.py` independently refuse to
  *grade* (not update) a baseline whose recorded floor exceeds its own run's
  ceiling, reporting it as a malformed baseline rather than a failed run
  (`scripts/check_engine_parity_threshold.py --self-test` proves this: a
  floor above the ceiling is rejected as malformed, a run below the recorded
  floor fails, a run at or above it passes). The current measured values on
  `integration/astra-v135`: differential coverage 312/1137 (27.44%,
  ceiling 426/1137 or 37.47%), high-risk 153/473 (32.35%, ceiling 171/473 or
  36.15%).
- **Raising the high-risk floor is corpus growth, not VM work (DD-15,
  build item, target v1.4).** The 320 high-risk constructs no corpus program
  under native currently mentions at all, broken down by surface category:
  194 `tensor_ad`, 56 `geometry`, 38 `numeric`, 14 `consciousness`, 7
  `control_flow`, 6 `memory_region`, 5 `macro_syntax`. None of these can gain
  differential evidence until a `tests/vm_parity/corpus/*.esk` program
  exercises them under native — the VM's own coverage is a separate,
  already-gated axis (`vm_parity_gate`, above). `tensor_ad` is the large
  majority of the gap, so it is the first family to target.

### v1.3.5-evolve surface closure

- **PR-02 closes the historical stdlib surface baseline.** The desktop VM now
  loads the canonical `lib/stdlib.esk` dependency closure before user source
  in source execution, REPL initialization, and ESKB emission. The
  execution-backed surface probe tested all 323 historical baseline entries:
  `0` native-resolves/VM-does-not divergences and `323` baseline entries fixed.
  `tests/vm_parity/SURFACE_BASELINE.tsv` is retained as a header-only,
  zero-entry ratchet. The probe requires `stdlib` explicitly on both engines,
  and its default probe files live under `.scratch` rather than a system
  temporary directory.

### v1.3.4-evolve parity changes

- **2-D matmul-surface parity lands on the hosted VM** (corrected 2026-08-25
  from "COMPLETE" — conformity audit item g5). `arange` (1-, 2-, and
  3-argument forms), nested-literal tensor operands, and multi-dimensional
  `tensor-ref` / `tensor-set!` now compute the same answers on the bytecode VM
  as on native codegen. The parity corpus gains `31_tensor_matmul`; the former
  matmul-surface `gap` rows for that corpus are retired to `vm-supported`. The
  corpus itself is 2-D only, with small exactly-representable values, and does
  not cover batched or rank-3+ contraction — 37 `PARITY.tsv` rows still carry
  "tensor linalg/manipulation fid missing in VM". Full tensor-linalg VM parity
  is a build item, target v1.5.0.
- **Reverse/forward-mode `gradient` is now `vm-supported` (#337).** The VM lowers
  an arity-resolved forward/reverse-mode `gradient` — direct, through a callable
  parameter, and curried — byte-identical to native codegen across the `native`,
  `vm-src`, and `vm-eskb` axes (`corpus/32_gradient_reverse.esk`,
  `gradient_callable_arity_test.esk` 25/25 on the VM). `op:GRADIENT` and
  `op:DERIVATIVE` move from `gap` to `vm-supported`; higher-order nesting
  (gradient-of-derivative / Taylor tower, `op:DERIVATIVE_N`) stays native-only.
  The public low-level AD tape surface (`ad-pow`, `ad-gradient-of`,
  `ad-value-of`, `ad-tape-length`) is also complete on JIT and AOT.
- **`(the <type> expr)` is `native-only-justified`.** The checked type
  ascription is a compile-time construct on the native type checker with no VM
  surface; it is a runtime no-op, so a VM program that omits it computes the
  identical result. The contradiction diagnostic added in v1.3.4 is likewise
  compile-time and emits no code, so runtime parity is unchanged.

## The manifest

`tests/vm_parity/PARITY.tsv` is `name<TAB>status<TAB>justification`, with three
statuses:

| status | meaning |
|---|---|
| `vm-supported` | the VM resolves the name / implements the op |
| `native-only-justified` | conscious, permanent waiver (FFI, OALR regions, static type syntax, OS/process, parallel runtime, front-end module machinery) — justification mandatory |
| `gap` | acknowledged hole **or a verified behavioral divergence** (rows referencing `found/*.esk` name symbols present on both surfaces that compute different answers) — justification mandatory |

**"Justification mandatory" is formally true (0 rows have an empty
justification field) and substantively uneven** — corrected 2026-08-25,
conformity audit item g2: 20 of the 44 `native-only-justified` rows share one
boilerplate string, and roughly 55% of the 330 `gap` rows share four bulk
strings; only a minority carry a per-symbol argument. Raising justification
specificity across the ledger is a low-priority build item — the field is
present and non-empty everywhere, which is what the ledger schema enforces
today.

Seeded 2026-07-03 from the live extraction and continuously re-audited with
probe runs on `eshkol-vm-standalone-test` vs native `-r`: **956 rows — 582
`vm-supported`, 44 `native-only-justified`, 330 `gap`** (counted from
`tests/vm_parity/PARITY.tsv`). The separate gap-evidence sidecar is checked by
`scripts/canonicalize_vm_gaps.py` before the runtime stages. The three most
recent promotions are
`op:LOGIC_VAR`, `op:WALK` and `walk`, retired to `vm-supported` when the
logic-variable representation was unified across the engines (task #100).

**A status is a claim about the running system, and is now checked as one.**
This audit validates the ledger against SOURCE TEXT — names scraped from the
C++ dispatch table in `llvm_codegen.cpp` and the op enum — so it can neither
see Scheme-level stdlib procedures nor tell whether a `vm-supported` row is
true. `scripts/run_surface_parity.py` closes that: it probes every name on
BOTH engines and fails when native resolves a name the VM does not while the
ledger is silent or claims `vm-supported`. It is what found `assq`, `assv`,
`memv`, `partition` and `string-contains` — all resolvable natively, all
aborting the VM with "undefined variable", none of them in this ledger, while
this audit reported OK.
Verified behavioral divergences remain explicit `gap` rows with reproducible
programs under `tests/vm_parity/found/`. Every gap, including an unimplemented
surface row without a historical `found/` file, is also recorded with an
explicit disposition and a live generated probe in
`tests/vm_parity/GAP_DISPOSITIONS.tsv`; the sidecar is a stage-1 gate.

**`tests/vm_parity/SURFACE_BASELINE.tsv` — the retired delta** (added
2026-08-25, conformity audit item g6, cross-referenced from FEATURE_MATRIX.md
d9 and KNOWN_ISSUES.md e6). The historical 323-name baseline was fully
retested in PR-02: no native-resolved name remained absent from the desktop VM,
and the file now contains zero entries. The 956-row `PARITY.tsv` accounting
therefore no longer has an untracked surface backlog, although its 330
behavioral `gap` rows remain a separate contract.

## The ratchet workflow

`scripts/vm_parity_audit.py` extracts two surfaces on every run:

- **codegen surface** — every builtin the LLVM backend dispatches on
  (`func_name == "…"`, `function_return_types[…]`, the `math_builtins` sets)
  plus every member of the `eshkol_op_t` AST enum;
- **VM surface** — every name the VM can resolve: the `BUILTINS[]` native table
  in `eshkol_vm.c`, the special-form dispatch in `vm_compiler.c` /
  `vm_parser.c`, the Scheme prelude compiled into every VM
  (`vm_prelude_source.h`), and the canonical `stdlib` dependency closure
  loaded by desktop VM compilation.

The audit **fails** if any codegen symbol is absent from *both* the VM surface
and `PARITY.tsv`. So when you add a language feature:

1. You add a builtin or AST op to the native codegen.
2. `scripts/run_vm_parity.sh` (stage 1) fails with
   `RATCHET <name>: … add VM support or a justified manifest row`.
3. You either
   - **teach the VM** — add the fid + name binding; the audit then passes with
     no manifest change and the corpus differential keeps you honest; or
   - **waive it consciously** — add a `PARITY.tsv` row with status
     `native-only-justified` (permanent) or `gap` (acknowledged hole), each
     requiring a justification.

The audit also fails on stale `vm-supported` claims (a row naming a builtin the
VM surface no longer contains) and on `gap` / `native-only-justified` rows with
no justification.

## The differential gate

`scripts/run_vm_parity.sh` (honors `BUILD_DIR`, default `build/`) runs
**four** stages (corrected 2026-08-25 from "three", conformity audit item
g4):

1. **AUDIT** (stage 1) — the ratchet above, codegen-vs-VM surface audit.
2. **CORPUS** (stage 2) — a VM-vs-native differential over `tests/vm_parity/corpus/`
   (programs inside the VM's verified subset) across axes:
   - `native`  — `./build/eshkol-run -r f.esk`
   - `vm-src`  — `./build/eshkol-vm-standalone-test f.esk`
   - `vm-eskb` — emit ESKB via `--profile hosted-vm --emit-eskb`, then run it
     through `eshkol-vm-standalone-test`.
3. **OOS** (stage 3) — programs outside the VM's verified subset
   (`tests/vm_parity/oos/`) must fail cleanly on the VM, not fabricate a value.
4. **FATAL** (stage 4) — programs whose first failing form is fatal must fail
   closed (nonzero exit) on both substrates.

Any divergence outside the manifest, at any stage, is a failure. Last
remeasured 2026-08-25 against `4bf871a0`: **188 passed, 0 failed**, exit 0.
The corpus has grown since that commit (83 to 84 files, `found/` 36 to 39), so
this figure must be regenerated by `BUILD_DIR=build scripts/run_vm_parity.sh`
on the v1.3.5-evolve release cut before it is quoted as a release gate
(`evidence/audit/06_vm_parity.log` in this resolution's evidence root;
corrects the stale "140/140" figure carried in `docs/KNOWN_ISSUES.md`
before this pass, conformity audit item e3).

```bash
BUILD_DIR=build scripts/run_vm_parity.sh
```

Verified behavioral divergences are recorded as `gap` rows referencing a repro
under `tests/vm_parity/found/` (for example, the VM's `display` appends a
newline per call). The parity gate also reruns every filed program in `found/`;
normalized agreement fails the gate until the program is moved to
`tests/vm_parity/resolved/` or promoted into `corpus/`. This keeps the active
contract precise without retaining stale defect claims.

## Floating-point determinism across engines

Parity is a claim about **bits**, not about closeness: the corpus differential
compares printed floats raw, so a one-ulp difference fails the gate exactly
like a wrong answer. Two things make that achievable.

- **Eshkol's own codegen never contracts.** `llvm_codegen.cpp` emits `fmul` /
  `fadd` and sets no fast-math flags and no `llvm.fmuladd`, so an Eshkol-level
  `(+ (* a b) c)` rounds twice on every target.
- **The C runtime and the VM are compiled with `-ffp-contract=off`**, set
  project-wide in `CMakeLists.txt` and repeated on the Emscripten command line
  in `scripts/run_wasm_differential.sh`. Contraction of `a * b + c` into a
  single, singly-rounded multiply-add is a *per-target* liberty: AArch64 and
  x86-64-with-FMA take it, and WebAssembly cannot, because the instruction set
  has no scalar f64 FMA. Left at the compiler default (`-ffp-contract=on`) one
  and the same C kernel therefore produces different bits on different
  engines, with nothing in the source to show it.

That is not hypothetical. The forward-mode dual quotient rule shared by
`eshkol_tensor_layer_norm_dual` (`lib/core/runtime_tensor_math.cpp`) and
`vm_tensor_dual_div` (`lib/backend/vm_tensor_ops.c`) is written
`a.tangent * inv - a.primal * b.tangent * inv2`. Contracted, the layer-norm
tangent in `tests/vm_parity/corpus/551_tensor_transformer_dual.esk` is
`0.20413179969792875`; evaluated as written it is `0.20413179969792872`. The
native builds fused it, the WASM build could not, and the execute-and-diff lane
failed on the last digit of one printed double while every other byte matched.

A kernel that genuinely wants a fused, singly-rounded product must call `fma()`
explicitly: `fma()` is correctly rounded on both libm implementations in play,
so it is identical on every engine, whereas *contraction* is whatever the
back end happens to be able to do. The rule is therefore "evaluate binary64
arithmetic as written, and spell fusion out when you want it" — never a
tolerance in the differential, and never rounding the printed digits.

See also [TESTING.md](TESTING.md) for the full adversarial-testing overview.
Reclassified cases are listed in [tests/vm_parity/resolved/README.md](../tests/vm_parity/resolved/README.md).
