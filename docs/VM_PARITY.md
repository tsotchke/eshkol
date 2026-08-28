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
boilerplate string, and roughly 55% of the 331 `gap` rows share four bulk
strings; only a minority carry a per-symbol argument. Raising justification
specificity across the ledger is a low-priority build item — the field is
present and non-empty everywhere, which is what the ledger schema enforces
today.

Seeded 2026-07-03 from the live extraction and continuously re-audited with
probe runs on `eshkol-vm-standalone-test` vs native `-r`: **956 rows — 581
`vm-supported`, 44 `native-only-justified`, 331 `gap`** (counted from
`tests/vm_parity/PARITY.tsv`; the 936/562/45 figures quoted here previously
predated several ratchet promotions). The three most recent promotions are
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
programs under `tests/vm_parity/found/`.

**`tests/vm_parity/SURFACE_BASELINE.tsv` — the retired delta** (added
2026-08-25, conformity audit item g6, cross-referenced from FEATURE_MATRIX.md
d9 and KNOWN_ISSUES.md e6). The historical 323-name baseline was fully
retested in PR-02: no native-resolved name remained absent from the desktop VM,
and the file now contains zero entries. The 956-row `PARITY.tsv` accounting
therefore no longer has an untracked surface backlog, although its 331
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

Any divergence outside the manifest, at any stage, is a failure. Remeasured
2026-08-25 against `4bf871a0`: **188 passed, 0 failed**, exit 0
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

See also [TESTING.md](TESTING.md) for the full adversarial-testing overview.
Reclassified cases are listed in [tests/vm_parity/resolved/README.md](../tests/vm_parity/resolved/README.md).
