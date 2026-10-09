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
44 `native-only-justified`, 331 `gap`. PR-02 retired the separate
`SURFACE_BASELINE.tsv` ratchet: its historical 323 names now produce zero
native-resolved/VM-missing divergences.

At v1.3.6-evolve the stage-1 audit reports a codegen surface of 946 symbols
(834 builtins + 112 ops), a VM surface of 1,636 names, and 962 manifest rows:
622 `vm-supported`, 45 `native-only-justified`, 295 `gap`; 39 further symbols
are on both surfaces and need no row. The audit also lists, as warnings, rows
whose symbol has left both the codegen surface and the Scheme stdlib (the
retired `dnc-*`/`sdnc-*` builtins among them).

## The differential gate

`scripts/run_vm_parity.sh` (uses `BUILD_DIR`, default `build/`; needs
`eshkol-run`, `stdlib`, `eshkol-vm-standalone-test`):

* **stage 1** — the surface audit above;
* **stage 2** — runs every program in `corpus/` (180 programs at v1.3.6-evolve, inside the VM's
  *verified* subset: arithmetic, floats, comparisons, recursion, TCO,
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
  diagnostic and no fabricated stdout value.

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

## found/ — verified divergences (in-subset programs, differing answers)

Every file is a minimal repro with native-vs-VM expected output in its
header. At v1.3.6-evolve one divergence remains filed here, alongside two `CONTROL`
fixtures:

| repro | divergence |
|---|---|
| `display_newline_per_call.esk` | display appends a newline per call (the newline normalization above masks exactly this) |
| `vm_tail_arity_ok.esk` | `CONTROL`: mutual tail calls between procedures of different arity are O(1) on the VM |
| `vm_tail_indirect_ok.esk` | `CONTROL`: an indirect tail call through a procedure parameter is O(1) on the VM |

The rest of the set filed while building this gate in 2026-07 has since
converged, and each repro moved out of `found/`:

| repro | divergence when filed | where it lives now |
|---|---|---|
| `char_type_collapsed.esk` | chars displayed as integers | `resolved/` |
| `ad_gradient_wrong.esk` | `gradient`/`jacobian`/`hessian` disagreed with native | `resolved/` |
| `logic_walk_unresolved.esk` | `walk` did not resolve bindings | retired; the logic-variable contract is asserted by value in `tests/logic/` |
| `float_display_1e10.esk` | large-float format `1e+10` vs `10000000000` | `resolved/` |
| `map_two_lists_eskb_route.esk` | multi-list `map` dropped lists on the ESKB route | `resolved/` |
| `consecutive_do_state_leak.esk` | consecutive top-level `do` loops interfered | `resolved/` |
| `define_after_do_corrupted.esk` | a top-level `do` disturbed later top-level defines | `resolved/` |
| `do_composition_broken.esk` | nested `do` lost iterations | `resolved/` |
| `when_tail_call_no_tco.esk` | tail calls through `when` bodies were not TCO'd | `resolved/` |
| `bignum_exact_rational.esk` | exact bignum-rational conversion | `corpus/63_bignum_exact_rational.esk` |
| `internal_define_then_body_form.esk` | internal `define` followed by a body form lost its slot | `corpus/73_internal_define_then_body_form.esk` |
| `sqrt_exact_negative.esk` | `(sqrt -4)` returned `+nan.0` rather than the complex `+2i` | `resolved/` |
| `error_object_irritants_roundtrip.esk` | `error-object-irritants` ordering, empty lists and re-raise | `corpus/error_object_irritants_roundtrip.esk` |

Divergences where **native is the side that departs from the specified result**
(filed rather than mirrored in the VM; native codegen is not VM-owned):

| repro | divergence |
|---|---|

These are deliberately **not** in `corpus/` (they would hold the gate red);
each is referenced from its `PARITY.tsv` gap row. When a divergence is fixed
in the VM, move its repro into `corpus/` and flip the manifest row to
`vm-supported` — the gate then guards the fix forever. The mirror rule holds
for the native side: when native is repaired, the repro is **promoted out of
`found/`** into `corpus/` in the same change, so the table above never claims
a divergence the compiler no longer has. Retiring the file is part of the fix, not
follow-up work — a `found/` entry asserting a divergence that no longer
reproduces is worse than no entry at all, because it tells the next reader to
expect a divergence that is gone.

Retired this way so far — `bignum_div_inexact_zero_native.esk` →
`corpus/53_bignum_inexact_zero_division.esk`, `do_set_param_native.esk` →
`corpus/51_do_body_set_mutation.esk`, `float_remainder_modulo_native.esk`
→ `corpus/50_flonum_integer_division.esk`, `namedlet_escaped_closure_native_segv.esk`
plus its VM-side counterpart `namedlet_escaped_closure_vm_routes.esk` →
`corpus/47_namedlet_escaped_closure.esk`, and
`tensor_vector_built_nested_native.esk` + `tensor_ragged_literal_native.esk` →
`corpus/46_tensor_literal_spellings.esk`; `quotient_inexact_native_vm.esk` →
`corpus/77_inexact_division_contagion.esk`. (`corpus/52` was claimed by
`#394`'s `52_numeric_tag_dispatch.esk` on master; files were renumbered
to the next free slot when the branches merged.)
`tensor_predicate_on_literal.esk` →
`corpus/100_tensor_reader_literal_classification.esk` now that reader-origin
rectangular numeric vectors answer `tensor?` consistently on both substrates.

The parity gate also reruns every `.esk` file still under `found/` on native
and VM. A file whose outputs now agree is reported as stale and fails the
gate until it is moved to `resolved/` with its measured result, or promoted
into `corpus/` when it is now part of the supported parity contract. The
recheck is deliberately separate from the corpus baseline so a filed claim
cannot silently become either an outdated divergence claim or an untracked regression.
Control fixtures are marked `CONTROL` in their header and remain in `found/`
when they document an intentionally one-sided behaviour that is not a
divergence; the gate reruns them but does not treat them as outdated entries.

## Regenerating

* audit only: `python3 scripts/vm_parity_audit.py`
* codegen/VM surface dumps: `--dump-codegen` / `--dump-vm`
* fresh manifest skeleton (after mass changes): `--seed`
* full gate: `scripts/run_vm_parity.sh` (`--audit-only`, `--no-eskb`)
