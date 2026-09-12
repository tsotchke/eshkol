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

## Compiler architecture and public API linkage

`gate_compiler_architecture.py` discovers AST operation switches throughout
`lib/frontend` and `lib/backend`, derives the operation set from `eshkol_op_t`,
and refuses omitted cases or defaults. Even an intentionally partial analysis
must state its remaining cases explicitly. Nested switch labels cannot cover a
parent. The companion lexical consumer scan finds direct LLVM calls associated
with unpacking closure metadata and permits that ABI dispatch only in
`lib/backend/llvm_codegen.cpp:codegenClosureCall`. It is a structural source
policy, not a C++ semantic or data-flow proof: unusual indirect-call spellings
require extending its recognizer. Its self-tests cover newly introduced source
files and duplicate dispatcher names as well as missing cases and defaults.

The current integration base fails this strict policy: 32 discovered AST
switches have omissions and defaults, and seven direct callable consumers
remain outside the canonical dispatcher. These findings are blocking; the gate
has no baseline or exception list. Run the gate to obtain exact paths, line
numbers, and missing operations. This does not label intentionally partial
analyses as proven runtime defects; they violate the explicit-routing policy.

`gate_public_api_linkage.py` derives every `eshkol_*` function prototype from
the umbrella public header and generates a volatile function-pointer relocation
for each. The CMake `compiler_public_api_linkage` executable must compile with
the real header, link the actual compiler archive and dependencies, and execute.
This currently covers 104 prototypes, including C++ overload/linkage names;
private backend classes and optional standalone headers are outside this gate.
A real-linker self-test demonstrates that adding a declared-but-undefined API
fails, while providing its definition links and executes. No `nm` spelling
heuristic or phantom-API allowlist substitutes for linking.

The runtime corpus also exports `gate_frame_observations.jsonl` in ICC's stated
criteria certificate format. Each repeat actually sets a different
`PYTHONHASHSEED`; ICC pairs equivalent source hashes across these execution
frames. The measured run produced 12 observations and 18 pairs with noise 0.0
findings. An evidence projection outside `.worktrees` let ICC's vacuity detector
read all 12 treatment receipts and one complete treatment cohort. Preserve the
raw receipts when transferring this corpus into an ICC artifact store.
