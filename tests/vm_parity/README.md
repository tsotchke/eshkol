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
  `vm_compiler.c` / `vm_parser.c`, and the Scheme prelude compiled into
  every VM (`vm_prelude_source.h`).

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
`eshkol-vm-standalone-test` vs native `-r`: 912 rows — 520 `vm-supported`,
41 `native-only-justified`, 351 `gap`.

## The differential gate

`scripts/run_vm_parity.sh` (uses `BUILD_DIR`, default `build/`; needs
`eshkol-run`, `stdlib`, `eshkol-vm-standalone-test`):

* **stage 1** — the surface audit above;
* **stage 2** — runs every program in `corpus/` (25 programs inside the VM's
  *verified* subset: arithmetic, floats, comparisons, recursion, TCO,
  closures + `set!`, let-family, named let, higher-order functions, lists,
  strings, `make-vector` vectors, `cond`/`case`/`when`/`unless`, flat `do`,
  quasiquote, rewrite-only macros, `guard`/`raise`, `call/cc`, `values`,
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

### Normalization — why newlines are stripped

The VM's `display` appends a newline after every call
(`found/display_newline_per_call.esk`), inserting newlines where native has
none, so no per-line normalization can align the two streams. The gate
therefore strips banner/log lines and then removes ALL newline characters
from both sides before comparing. Value divergences, dropped output and
fabricated output all still surface; only newline-placement divergences are
masked — which is exactly the filed quirk. The VM also exits 0 on fatal
runtime errors (`found/error_exit_code_zero.esk`), so VM failure is detected
via stderr markers (`ERROR`, `FRAME OVERFLOW`, `unhandled native call`),
never via exit codes.

## found/ — verified divergences (in-subset programs, wrong answers)

Every file is a minimal repro with native-vs-VM expected output in its
header. Filed while building this gate, 2026-07:

| repro | divergence |
|---|---|
| `display_newline_per_call.esk` | display appends a newline per call |
| `error_exit_code_zero.esk` | exit code 0 after fatal runtime error |
| `exact_division_lost.esk` | `(/ 1 3)` → `0.333333`, not `1/3` |
| `expt_bignum_to_float.esk` | `(expt 2 100)` → float, not bignum |
| `force_returns_promise.esk` | `force` returns the promise object |
| `let_values_silent_zero.esk` | `let-values` binds wrong values silently |
| `case_lambda_wrong_clause.esk` | `case-lambda` picks the wrong clause |
| `dynamic_wind_after_twice.esk` | after-thunk runs twice |
| `char_type_collapsed.esk` | chars display as integers |
| `equal_eq_structural_false.esk` | `equal?`/`eq?` → `#f` on equal lists / same symbols |
| `vector_literal_empty.esk` | `#(...)` literals become `()` |
| `vector_constructor_empty.esk` | `(vector ...)` → `()`; `vector->list` fid 140 unimplemented; `list->vector` fid collides with `memq` (139) |
| `symbol_string_unhandled_fid.esk` | `symbol->string`/`string->symbol` fids 184/185 unimplemented |
| `write_does_not_quote.esk` | `write` emits display syntax |
| `iota_returns_empty.esk` | `iota` returns `()` |
| `recursive_macro_zero.esk` | recursive syntax-rules macros → wrong value |
| `macro_set_top_level.esk` | set!-mutating macro: three-way divergence (native side suspect too) |
| `ad_gradient_wrong.esk` | `gradient`/`jacobian`/`hessian` silently wrong |
| `logic_walk_unresolved.esk` | `walk` does not resolve bindings |
| `float_display_1e10.esk` | large-float format `1e+10` vs `10000000000` |
| `map_two_lists_eskb_route.esk` | multi-list `map` correct on vm-src, drops lists on the ESKB route (stale prelude cache) |
| `consecutive_do_state_leak.esk` | consecutive top-level `do` loops corrupt each other |
| `define_after_do_corrupted.esk` | a top-level `do` corrupts later top-level defines |
| `do_composition_broken.esk` | nested `do` loses iterations; `do`+`when` spins forever |
| `frame_overflow_exit_zero.esk` | non-tail depth ~300 → FRAME OVERFLOW, exit 0 |
| `when_tail_call_no_tco.esk` | tail calls through `when` bodies are not TCO'd |

These are deliberately **not** in `corpus/` (they would hold the gate red);
each is referenced from its `PARITY.tsv` gap row. When a divergence is fixed
in the VM, move its repro into `corpus/` and flip the manifest row to
`vm-supported` — the gate then guards the fix forever.

## Regenerating

* audit only: `python3 scripts/vm_parity_audit.py`
* codegen/VM surface dumps: `--dump-codegen` / `--dump-vm`
* fresh manifest skeleton (after mass changes): `--seed`
* full gate: `scripts/run_vm_parity.sh` (`--audit-only`, `--no-eskb`)
