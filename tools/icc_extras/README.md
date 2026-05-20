# icc_extras — Eshkol-specific tooling on top of `infinite_context_coder`

The base `infinite_context_coder` (ICC, at `~/Desktop/infinite_context_coder`)
gives us call-graph, include-graph, chunked symbol/file index, git-history memory,
and a regex-based `guard-diff`. That covers routing/scoping decisions for the
v1.2 carry-forward and the mechanical extractions (#205).

What it does **not** cover, and what this directory adds:

| Need | Tool here |
|------|-----------|
| JIT/AOT/VM cross-mode parity audit | `parity_ledger.json` + `generate_parity_ledger.py` |
| Project-specific structural-invariant lints | `codegen_audit_rules.json` (preset for ICC's `guard-diff`) |
| LLVM-IR validity check on every emit | covered in the codegen layer (see `lib/backend/llvm_codegen.cpp` `verifyModule` calls); audit notes in `verifier_coverage.md` |

These three artefacts are the v1.3 prep that goes in front of #206
("v1.3 architectural codegen rewrite"). They catch the classes of bug that
have actually bitten us — Bug F/G (parallel workers across modes), the 35-gap
bignum audit, the tagged-value-data-field-{4} class, the
findFreeVariablesImpl-coverage class, the closure-in-loop-PHI class — at
audit time, not at runtime.

## Layout

```
tools/icc_extras/
├── README.md                          this file
├── parity_ledger.schema.json          JSON Schema (Draft 7) for the ledger
├── parity_ledger.json                 the ledger itself (committed; reviewed)
├── generate_parity_ledger.py          scrapes the codebase and emits the ledger
├── codegen_audit_rules.json           ICC guard-diff rules (regex + skip-context)
├── codegen_audit.py                   runs the rules against the source tree and
│                                      compares results against the saved baseline
├── audit_baseline.json                the accepted baseline of known findings
│                                      (deviations from this file fail the audit)
└── verifier_coverage.md               audit of where verifyModule already runs
```

## Running

```sh
# Regenerate the ledger from the current source tree
python3 tools/icc_extras/generate_parity_ledger.py \
    --repo-root . \
    --out tools/icc_extras/parity_ledger.json

# Audit a diff against the eshkol-codegen rules
python3 ~/Desktop/infinite_context_coder/scripts/codebase_tool.py guard-diff \
    --repo eshkol_lang \
    --rules tools/icc_extras/codegen_audit_rules.json \
    --base origin/master
```

## What "parity status" means

For each operation/builtin, the ledger records one of:

- **`ok`** — AOT and VM implementations exist and are believed to be
  semantically equivalent. Tests exercise both paths.
- **`vm-partial`** — VM implementation exists but is a subset (e.g.
  `gradient` in the VM is scalar-only, AOT is multi-variable).
- **`vm-absent`** — Builtin exists in AOT only. The VM raises
  "unknown native" or returns `()`. Often intentional (e.g. parallel
  primitives in WASM build).
- **`aot-partial`** — AOT implementation has a known limitation that
  the VM has worked around (rare; mainly older-AD-only ops).
- **`divergent`** — Both implementations exist but produce different
  results for some input. **Always a bug.** Refer to `notes` field.
- **`unverified`** — Status not yet hand-checked. New entries default
  here until a maintainer audits and flips them.

The generator never *infers* the status — that's a human call. It only
fills in the structural fields (paths, line numbers, test cross-refs)
and flips entries to `unverified` when it finds a new mismatch in the
AOT/VM presence.
