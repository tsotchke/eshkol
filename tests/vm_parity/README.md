# VM parity ratchet (adversarial testing campaign — P5)

Eshkol ships two executable back ends: the native LLVM codegen
(`lib/backend/llvm_codegen.cpp`) and the bytecode VM
(`lib/backend/vm_compiler.c`, `vm_native.c`, `eshkol_vm.c`,
`eshkol-vm-standalone`, the ESKB format and the `hosted-vm` profile).
Before this harness existed, the VM covered an **undeclared** subset of the
language: nothing forced a decision when a feature landed in the codegen but
not in the VM, and nothing recorded which VM behaviors silently diverged.

This directory makes the subset explicit and makes drift impossible to miss.

## The ratchet

`scripts/vm_parity_audit.py` extracts two surfaces on every run:

* **codegen surface** — every builtin name the LLVM backend dispatches on
  (`func_name == "..."`, `function_return_types[...]`, the `math_builtins`
  sets) plus every member of the `eshkol_op_t` AST enum (`op:NAME` rows);
* **VM surface** — every name the VM can resolve: the `BUILTINS[]`
  first-class native table in `eshkol_vm.c`, the special-form dispatch in
  `vm_compiler.c` / `vm_parser.c`, the Scheme prelude compiled into every VM
  (`vm_prelude_source.h`), and the canonical `stdlib` dependency closure
  loaded by desktop VM compilation.

The audit **fails** if any codegen symbol is absent from BOTH the VM surface
and `PARITY.tsv`. So the workflow when you add a language feature is:

1. You add a builtin or AST op to the native codegen.
2. `scripts/run_vm_parity.sh` (stage 1) fails with
   `RATCHET <name>: ... add VM support or a justified manifest row`.
3. You either
   * **teach the VM** (add the fid + name binding; the audit then passes with
     no manifest change and the corpus differential keeps you honest), or
   * **waive it consciously** — add a `PARITY.tsv` row with status
     `native-only-justified` (permanent, justification mandatory) or `gap`
     (acknowledged hole, justification mandatory, counted in every audit
     report).

The audit also fails on stale `vm-supported` claims (a manifest row naming a
builtin the VM surface no longer contains) and on `gap`/`native-only-justified`
rows without a justification. Rows for symbols that left the codegen surface
are warnings — tidy them when convenient.

## PARITY.tsv

`name<TAB>status<TAB>justification`, statuses:

| status | meaning |
|---|---|
| `vm-supported` | the VM resolves the name / implements the op |
| `native-only-justified` | conscious permanent waiver (FFI, OALR regions, static type syntax, OS/process, parallel runtime, front-end module machinery) |
| `gap` | acknowledged hole **or verified behavioral divergence** (rows referencing `found/*.esk` are names present on both surfaces that compute different answers) |

Seeded 2026-07-03 from the live extraction, hand-verified with probe runs on
`eshkol-vm-standalone-test` vs native `-r`: 956 rows — 581 `vm-supported`,
44 `native-only-justified`, 331 `gap`. **On the v1.3.5-evolve cut the manifest
is 961 rows — 604 `vm-supported`, 46 `native-only-justified`, 311 `gap`**: the
ratchet has moved 23 names out of `gap` and into VM support since the seed,
which is the only direction it moves. PR-02 retired the separate
`SURFACE_BASELINE.tsv` ratchet: its historical 323 names now produce zero
native-resolved/VM-missing divergences, and the file is retained with its
header and no rows so the count can never grow back.

## The differential gate

`scripts/run_vm_parity.sh` (uses `BUILD_DIR`, default `build/`; needs
`eshkol-run`, `stdlib`, `eshkol-vm-standalone-test`):

* **stage 1** — the surface audit above;
* **stage 2** — runs every program in `corpus/` (148 programs on the
  v1.3.5-evolve cut, all inside the VM's *verified* subset: arithmetic, floats, comparisons, recursion, TCO,
  closures + `set!`, let-family, named let, higher-order functions, lists,
  strings, `make-vector` vectors, `cond`/`case`/`when`/`unless`, flat `do`,
  `set!` from a `do` body, flonum integer division (`modulo`/`remainder`
  sign conventions under exactness contagion), quasiquote, rewrite-only
  macros, `guard`/`raise`, `call/cc`, `values`,
  `define-record-type`, a sieve) under three axes — native `eshkol-run -r`,
  `vm-src` (the VM's own compiler), and `vm-eskb`
  (`--profile hosted-vm --emit-eskb` + VM) — and byte-compares
  newline-normalized stdout;
* **stage 3** — asserts the 5 probes in `oos/` (http-get, hash tables,
  `match`, `eval`, `read-file`) fail **cleanly** on the VM: a clear stderr
  diagnostic and no fabricated stdout value;
* **stage 4** — asserts the 12 probes in `fatal/` fail **closed** (see below).

It emits `PASSED/FAILED <nodeid>` lines plus `kind:"vm_parity"` JSON-L
events into `scripts/icc_traces/vm_parity.jsonl`, consumed by the
`vm-parity` target in `.icc/completion-oracles.yaml`.

### Naming a new `corpus/` file

The `NN_` prefix is **ordering only** — the gate globs the directory and does
not read the number, and duplicates already exist (`31_`, `36_`, `42_`) from
branches that picked the same next slot in parallel. So: pick a number that is
free on master **and** on every open branch that adds a corpus file
(`git ls-tree --name-only origin/<branch> tests/vm_parity/corpus/`), and never
assume "highest + 1" is yours. A collision is an add/add conflict at merge time,
and renumbering then means chasing every reference to the old name.

### Normalization — why newlines are stripped

The VM's `display` appends a newline after every call
(`found/display_newline_per_call.esk`), inserting newlines where native has
none, so no per-line normalization can align the two streams. The gate
therefore strips banner/log lines and then removes ALL newline characters
from both sides before comparing. Value divergences, dropped output and
fabricated output all still surface; only newline-placement divergences are
masked — which is exactly the filed quirk. VM failure is detected via BOTH the
exit status and stderr markers (`ERROR`, `FRAME OVERFLOW`, `unhandled native
call`); the VM used to exit 0 on every fatal runtime error, which stage 4 now
gates directly.

## fatal/ — fail-closed probes

Programs whose first failing form is fatal on **both** substrates. Stage 4 of
the gate requires each to exit NONZERO, name the failure on stderr, and print
nothing past the fatal form (each probe ends with a `MUST-NOT-PRINT`
sentinel). This is the fail-open ratchet: a fatal VM error may never again look
like a successful run to a shell or to CI.

## found/ — verified divergences (in-subset programs, wrong answers)

Every file is a minimal repro with native-vs-VM expected output in its
header. The set below is what `found/` holds on the v1.3.5-evolve cut; 27
further reproducers filed here since 2026-07 have been reclassified into
`resolved/` because their outputs now agree (see `resolved/README.md`).

| repro | divergence |
|---|---|
| `display_newline_per_call.esk` | display appends a newline per call — the divergence the harness normalizes around |
| `case_lambda_wrong_clause.esk` | `case-lambda` picks the wrong clause |
| `ad_gradient_wrong.esk` | `gradient`/`jacobian`/`hessian` silently wrong |
| `error_object_irritants_empty.esk` | `error-object-irritants` always `()` (`error` is a 1-arg native, fid 237) |
| `frame_overflow_exit_zero.esk` | non-tail depth ~300 → FRAME OVERFLOW (the VM now exits nonzero; the depth limit remains) |
| `quotient_inexact_native_vm.esk` | `quotient` with an inexact operand comes back **exact** and **wraps past 2^63**; `(remainder <flonum> 0.0)` answers `+nan.0` where every other representation raises |
| `tensor_predicate_on_literal.esk` | `tensor?` disagrees on a reader `#(...)` literal (every size and rank property agrees) |
| `with_region_explicit_quote_body_vm.esk` | `(with-region (quote name))` — the one `with-region` spelling the two readers disagree on |

Control fixtures, which document an intentionally one-sided or non-defect
behavior rather than a divergence, and which the gate reruns without treating
as stale:

| control | what it pins |
|---|---|
| `vm_tail_arity_ok.esk` | mutual tail recursion between procedures of differing arity is already O(1) on the VM |
| `vm_tail_indirect_ok.esk` | an indirect tail call through a procedure parameter is already O(1) on the VM |

Divergences where **native is the wrong side** are filed here too, rather than
matched in the VM — native codegen is not VM-owned. On the v1.3.5-evolve cut
that set is empty: every native-side reproducer filed since 2026-07
(`tensor_nested_collection_native.esk`, `tensor_ref_component_oob_native.esk`,
`tensor_set_oob_silent_native.esk`, `tensor_shape_empty_native_is_right.esk`)
has been reclassified into `resolved/`.

These are deliberately **not** in `corpus/` (they would hold the gate red);
each is referenced from its `PARITY.tsv` gap row. When a divergence is fixed
in the VM, move its repro into `corpus/` and flip the manifest row to
`vm-supported` — the gate then guards the fix forever. The mirror rule holds
for the native side: when native is repaired, the repro is **promoted out of
`found/`** into `corpus/` in the same change, so the table above never claims
a defect the compiler no longer has. Retiring the file is part of the fix, not
follow-up work — a `found/` entry asserting a divergence that no longer
reproduces is worse than no entry at all, because it tells the next reader to
expect a bug that is gone.

Retired this way so far — `bignum_div_inexact_zero_native.esk` →
`corpus/53_bignum_inexact_zero_division.esk`, `do_set_param_native.esk` →
`corpus/51_do_body_set_mutation.esk`, `float_remainder_modulo_native.esk`
→ `corpus/50_flonum_integer_division.esk`, `namedlet_escaped_closure_native_segv.esk`
plus its VM-side counterpart `namedlet_escaped_closure_vm_routes.esk` →
`corpus/47_namedlet_escaped_closure.esk`, and
`tensor_vector_built_nested_native.esk` + `tensor_ragged_literal_native.esk` →
`corpus/46_tensor_literal_spellings.esk`. (`corpus/52` was claimed by
`#394`'s `52_numeric_tag_dispatch.esk` on master; files were renumbered
to the next free slot when the branches merged.) The complete retired set —
27 reproducers whose native and VM outputs now agree — is listed in
`resolved/README.md`, each keeping its original report and measured expected
values in its source header.

The parity gate also reruns every `.esk` file still under `found/` on native
and VM. A file whose outputs now agree is reported as stale and fails the
gate until it is moved to `resolved/` with its measured result, or promoted
into `corpus/` when it is now part of the supported parity contract. The
recheck is deliberately separate from the corpus baseline so a filed claim
cannot silently become either a false defect or an untracked regression.
Control fixtures are marked `CONTROL` in their header and remain in `found/`
when they document an intentionally one-sided or non-defect behavior; the
gate reruns them but does not treat them as stale defects.

## Regenerating

* audit only: `python3 scripts/vm_parity_audit.py`
* codegen/VM surface dumps: `--dump-codegen` / `--dump-vm`
* fresh manifest skeleton (after mass changes): `--seed`
* full gate: `scripts/run_vm_parity.sh` (`--audit-only`, `--no-eskb`)
