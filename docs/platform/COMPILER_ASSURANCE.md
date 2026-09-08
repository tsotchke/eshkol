# Compiler assurance executions

`python3 scripts/run_compiler_assurance.py` executes the production closed-enum
gate twelve times in an isolated source projection: four repeats each with zero,
one, and two handwritten AD dispatcher case labels removed. The gate must pass
all baselines, reject all treatments, and produce increasing finding counts whose
separation exceeds the measured within-dose range. Counts have resolution one;
zero observed variation is measured repeatability, not a claim about all possible
machines, changes, or gate implementations.

The control found an actual false green: a registry include previously exempted
all manual `INLINE` cases from missing-member checks. Only disposition macros
that emit case labels now receive generated coverage credit.

Raw subprocess argv, stdout, stderr, exit code, timing, execution UUID, source
hash, gate hash, and Git revision are retained under
`.icc/evidence/compiler-assurance/raw/`. The sibling `closed-enum/` directory
contains treatment/score observations linked to those receipts. `--verify`
rejects an empty, incomplete, duplicated, nonresponsive, or noisy corpus.
Evidence is freshly replaced on each run; committed historic PASS records are
not used. CI uploads hidden evidence files even when the gate fails.

Run actual compiler capabilities separately:

```sh
python3 scripts/run_compiler_assurance.py --runtime-only \
  --binary build/eshkol-run --output .icc/evidence/compiler-capabilities
```

This executes captured closures, case-lambda, parameter calls, and apply with a
capture, checks exact output, and hashes the binary before and after execution.
This establishes the identity of the binary exercised, not that an arbitrary
external binary was built from the current checkout. CI supplies its own freshly
built compiler. The source-mutation experiment measures the correctness gate;
it does not claim that the mutated compiler was rebuilt or executed.

Both lanes emit completion-oracle events under `scripts/icc_traces/` and ICC
history under `.icc/runtime-traces/`. CTest registers both lanes and their
self-tests. The assurance CI job runs the mutation corpus; linux-x64-lite runs
the compiler capabilities after its build.

For ICC vacuity inspection:

```sh
icc vacuous-assertions --repo REPO \
  --include-prefix scripts/run_compiler_assurance.py \
  --evidence-glob '.icc/evidence/compiler-assurance/closed-enum/*.json' \
  --treatment-key dose --format json
```

Inspect scan counts, not just ICC's top-level status: the installed ICC version
can exclude evidence under an ancestor named `.worktrees` and report PASS after
reading zero corpus/history files. The local `--verify` gate independently
requires twelve executions and three treatment levels. No suppression or
phantom-API allowlist is supplied by this framework.
