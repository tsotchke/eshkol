# P8 "escape-closure" — escape analysis ledger (internal)

Goal: for every externally-reported bug CLASS of the 2026-07 cycle, add a
generator/gate to our own framework such that the class would have been caught
BY US FIRST. This ledger maps each escape to its originating fix commit, the
root cause, the P8 axis that now closes it, and the retro-catch evidence.

Internal document — it may name the full commit history. The public-facing
`docs/TESTING.md` section uses neutral wording (no reporter/consumer names).

Worktree: `test/p8-escape-closure` off `origin/master` @ `5cb02c8a`.
Pre-fix retro-catch build: `010053c8` (parent of #331 `6e2215ed`) — chosen
because it precedes **every** relevant fix of the wave (verified with
`git merge-base --is-ancestor`): #310/#328 printer, #330 callable-arity, #331
scope race, #338/#339/#340/#343 AD binding-form. A SINGLE pre-fix build
therefore reproduces the entire escape set.

> Note on the brief's SHA: the brief cited `401808ef` as the concurrency
> pre-fix SHA. `401808ef` is the FIX for #330 (callable arity); its parent is
> `6e2215ed` (#331, the scope-race fix). The correct pre-fix commit for the
> scope race is `010053c8` (#331's parent), which is what we build.

---

## Escape → axis map

| # | Axis | Originating fix (commit) | Root cause of the escape | Why our tests missed it | Retro-catch |
|---|------|--------------------------|--------------------------|-------------------------|-------------|
| 1 | binding-form | #343 `d79a06ea`, #338 `735985d9` (also #339/#340) | AD differentiation POINT classified by AST node kind, not runtime value; VAR/`(the …)`/general-call points mis-routed → SIGSEGV or silent-zero | Tests only ever wrote the point as a `#(…)` literal (a provable collection); the ambiguous forms were never swept | CONFIRMED |
| 2 | indirection | #330 `401808ef` | `gradient` runtime-closure branch ignored callable arity when `f` came through a parameter/wrapper/curry | Tests always named `f` directly; the wrapper/curried/2-level forms were never swept | CONFIRMED (integrated) |
| 3 | arity-sweep | #322/#327 `ea1a5956` (VM matmul) | VM special-form dispatch diverged from native (arange arity, nested-literal operands, multi-dim ref/set) | No gate asserted native==VM across the documented builtin surface | by construction + ratchet |
| 4 | property-oracle | #310 `0c25e0c1`, #328 `f3018f84` (printer) | 6-significant-figure float printing instead of shortest round-trip | The chibi differential's OWN normalizer reformatted every float to `%.6g` on BOTH sides — shared-defect blindness via lossy normalization | CONFIRMED |
| 5 | concurrency-fuzz | #331 `6e2215ed` | bump-arena scope stack is single-threaded; parallel-map workers concurrently push/pop it → dangling structure, nondeterministic ~50% | One fixed-shape fixture existed; the trigger space (closure body × threshold-straddling n × repetition) was not swept | CONFIRMED |
| 6 | five-way-surface | ad-pow/ad-tape (VM-only; see edge_v134 note) | a builtin documented + VM-dispatched but NOT native-registered | No gate cross-checked a builtin's presence across doc/manifest/native/VM/provide | by construction (ad-pow captured as `native_missing`) |
| 7 | fault-injection | #334 `7cb1d5d1` | a broken generated-program link under `-r` fell back to a reduced in-process run and exited 0 | Point tests existed; the fault space (missing/unopenable/malformed/bad-require/broken-link/hang × -r/AOT) was not swept as a matrix | CONFIRMED + 2 NEW masking cells found |
| 8 | mem-profiles | training-loop OOM class (ESH-0039/#81, ESH-0214e wave) | scope/region reclamation leak makes RSS grow with WORK | Fixed-size RSS ceilings are machine-specific and set too loose | by construction (flat-RSS ratio invariant) |
| 9 | packaging | Homebrew build class (#344 `fa8b8c7b`) | system-dependency resolution / formula build only exercised by hand | No lane built the in-repo formula from source | nightly lane (advisory) |

---

## Retro-catch evidence (pre-fix build `010053c8` vs master `5cb02c8a`)

All generators are pure Python; the ONLY difference between the columns is the
`eshkol-run` / VM binary they execute against.

### Axis 1 — binding-form (`scripts/p8/gen_ad_escape.py`)
- **master**: 35/35 gated files PASS; the `(list …)` field-op cell is quarantined
  as XKNOWN (see NEW BUG ESH-0360 below).
- **pre-fix `010053c8`**: **34 / 35 gated files CAUGHT**.
  - scalar hessian/derivative point forms → `SIGSEGV (rc=139)` (the #339 class:
    `hessian` hard-classified `ESHKOL_VAR` as scalar).
  - `(the (vector any) …)` gradient point → codegen error `Unknown function: the`
    (the ambiguous `(the …)` form was not accepted as a point pre-#343).
  - gradient/hessian at multi-form points → `rc=1` whole-file evaluation failure.

### Axis 4 — property oracles (`scripts/p8/gen_property_oracles.py`)
- **master**: `number->string(sqrt 2)` = `1.4142135623730951`; round-trip `#t`;
  numrt corpus 92/92 PASS.
- **pre-fix `010053c8`**: `number->string(sqrt 2)` = `1.41421`; round-trip `#f`;
  numrt corpus **79 FAIL** across 4 files (`prop_numrt_00..03`). This is exactly
  the precision the chibi differential's `%.6g` normalizer had been collapsing.

### Axis 5 — concurrency fuzz (`scripts/p8/gen_concurrency_fuzz.py`)
- **master**: 6/6 shapes PASS (each: 5 threshold-straddling n × 20 repeats vs the
  serial-map oracle), ~6 s.
- **pre-fix `010053c8`**: **6 / 6 shapes CAUGHT** on a single pass — `rc=134`
  (SIGABRT, "car/cdr: not a pair") and `rc=139` (SIGSEGV). The heavy scope-op
  triggers drive detection well above the ~50% single-run rate.

### Axis 7 — fault injection (`scripts/p8/p8_fault_injection.sh`)
- 7 hard-gate cells PASS on master (retro-guarding #334: broken `--lib` under
  `-r`/AOT, malformed-source AOT, bad output dir, undefined symbol, missing
  source under `-r`).

---

## NEW real bugs found by P8 (UNFIXED — recorded, reported, quarantined)

### ESH-0360 — jacobian/divergence/curl SIGSEGV at a `(list …)` point (axis 1)
Found on master `5cb02c8a`. `jacobian` / `divergence` / `curl` SIGSEGV (rc=139)
when the differentiation point is built with `(list …)`. The identical point as
`#(…)` / `(vector …)` / `(tensor …)` / VAR-bound / let-bound / fnret /
`(the (vector any) …)` is correct, AND `gradient`/`hessian`/`laplacian` at a
`(list …)` point are correct. The #343 `cons->svec` point normalization was
applied to the scalar-output operators but NOT to the vector-field operators.
- Repro: `tests/escape_matrix/found/ESH-0360_field_ops_list_point_segv.esk`
- Quarantine: `KNOWN_CRASH` in `gen_ad_escape.py` → emitted as a `;; P8-XCRASH`
  file; the runner tolerates it as XKNOWN and reports XPASS (promote to gate)
  when fixed.

### ESH-0361 — exit-0 masking in toolchain fault paths (axis 7)
Found on master `5cb02c8a`. Several `-r`/AOT fault inputs exit **0**, masking
the failure from any build system that checks `$?`:
- **`eshkol-run MISSING.esk -o out` (AOT)** → exit 0 AND writes a 5 MB binary
  (the most harmful: a build ships an executable that never held the program).
- **`eshkol-run -r SYNTAX-ERR.esk`** → exit 0 despite printing a parse error.
- `-r` unreadable file, `-r`/AOT `(require missing.module)` → exit 0 (the last
  two are possibly lenient-by-design; quarantined pending triage).
- Contrast: broken `--lib` and malformed-source-AOT correctly exit nonzero (the
  #334 fix), so the exit-code contract is inconsistent across paths.
- Repro: `tests/escape_matrix/found/ESH-0361_aot_missing_input_exit0.md`
- Quarantine: `xmask=1` cells in `p8_fault_injection.sh` (XKNOWN; XPASS-on-fix).

---

## Determinism / disk / seeds
- Every generator is a pure function of a fixed seed (8801 AD, 8803 arity,
  8804 property, 8805 concurrency). Regeneration is byte-identical.
- Corpora are generated into a per-run `mktemp -d` removed on exit; the
  orchestrator enforces a 512 MB corpus disk cap. Sanitizer/hang cells use a
  perl alarm or a background SIGKILL (macOS has no `timeout(1)`).
- Baselines (`arity_parity_baseline.json`, `five_way_baseline.json`) are
  shrink-only ratchets: a key may leave the baseline (fixed) but a NEW key fails
  the gate.
