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
> the current v1.3.3-evolve candidate audit.

## The manifest

`tests/vm_parity/PARITY.tsv` is `name<TAB>status<TAB>justification`, with three
statuses:

| status | meaning |
|---|---|
| `vm-supported` | the VM resolves the name / implements the op |
| `native-only-justified` | conscious, permanent waiver (FFI, OALR regions, static type syntax, OS/process, parallel runtime, front-end module machinery) — justification mandatory |
| `gap` | acknowledged hole **or a verified behavioral divergence** (rows referencing `found/*.esk` name symbols present on both surfaces that compute different answers) — justification mandatory |

Seeded 2026-07-03 from the live extraction and continuously re-audited with
probe runs on `eshkol-vm-standalone-test` vs native `-r`: **916 rows — 540
`vm-supported`, 44 `native-only-justified`, 332 `gap`**. Verified behavioral
divergences remain explicit `gap` rows with reproducible programs under
`tests/vm_parity/found/`.

## The ratchet workflow

`scripts/vm_parity_audit.py` extracts two surfaces on every run:

- **codegen surface** — every builtin the LLVM backend dispatches on
  (`func_name == "…"`, `function_return_types[…]`, the `math_builtins` sets)
  plus every member of the `eshkol_op_t` AST enum;
- **VM surface** — every name the VM can resolve: the `BUILTINS[]` native table
  in `eshkol_vm.c`, the special-form dispatch in `vm_compiler.c` /
  `vm_parser.c`, and the Scheme prelude compiled into every VM
  (`vm_prelude_source.h`).

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

`scripts/run_vm_parity.sh` (honors `BUILD_DIR`, default `build/`) runs three
stages:

1. **AUDIT** — the ratchet above.
2. **CORPUS** — a VM-vs-native differential over `tests/vm_parity/corpus/`
   (programs inside the VM's verified subset) across axes:
   - `native`  — `./build/eshkol-run -r f.esk`
   - `vm-src`  — `./build/eshkol-vm-standalone-test f.esk`
   - `vm-eskb` — emit ESKB via `--profile hosted-vm --emit-eskb`, then run it
     through `eshkol-vm-standalone-test`.
3. **Verdict** — any divergence outside the manifest is a failure.

```bash
BUILD_DIR=build scripts/run_vm_parity.sh
```

Verified behavioral divergences are recorded as `gap` rows referencing a repro
under `tests/vm_parity/found/` (for example, the VM's `display` appends a
newline per call). This turns "the VM is roughly compatible" into a precise,
enforced, and continuously re-verified contract.

See also [TESTING.md](TESTING.md) for the full adversarial-testing overview.
