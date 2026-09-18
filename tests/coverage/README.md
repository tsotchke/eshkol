# Language-surface coverage tracking

This directory is the backbone for **TOTAL-LANGUAGE exposure-engine coverage**:
the mechanism that measures which of Eshkol's user-facing constructs the
generative exposure engines actually exercise, names the gap, and turns
"cover the whole language" into a measurable ICC completion-oracle criterion.

Two numbers, and neither is hardcoded in prose: the **live surface total** is
whatever `language_surface.json` holds (regenerated from compiler sources, so it
grows the moment a builtin lands), and the **enforced floor** is
`coverage_policy.json`'s `minimum_covered`, which only ever increases and is
1,078 as of the v1.3.4 line. A surface that has grown past the floor is not a
failure — it is the next ratchet.

## Artifacts

| File | Producer | What it is |
|---|---|---|
| `language_surface.json` | `scripts/gen_language_surface.py` | Ground-truth manifest: every core builtin, tracked Agent FFI API, special form, AST op, and prelude fn, each categorised by risk. Extracted from source — never hand-maintained. |
| `coverage_policy.json` | monotonic ratchet | Minimum covered count/fraction and the categories that must reach zero uncovered before TOTAL-LANGUAGE completion. The floor can only increase. |
| `coverage_run.json` | `scripts/language_coverage.py --update-committed-run` | Committed snapshot of the sidecar: covered / total, covered fraction, the nested effective policy block, and covered + uncovered names by category. **Build output by nature** — an ordinary gate run writes the sidecar to the gitignored `build/coverage/coverage_run.json` instead, so running the gate never dirties the tree. Refresh this copy deliberately (see below). |
| `coverage_gap.md` | analysis | Human-readable gap report ranked by silent-wrong risk. |
| `release_record.json` | release cut | The release tag, the previous release's tag, the date, status label and the CTest and VM-parity totals. `scripts/check_surface_counts.py` grades every release-facing document and generated site page against it, and `--sync` rewrites their claims from it. |
| `changelog_no_user_facing_change.json` | hand-maintained, gated | The merged pull requests of the release range that deliberately have no `CHANGELOG.md` entry, each with a class from a closed set and a specific reason. Graded by `scripts/check_changelog_completeness.py` (see below). |
| `doc_front_matter_baseline.json` | `scripts/check_doc_front_matter.py --update-baseline` | The number of documentation pages that carry no front-matter block. It can only go down (see below). |

Collect evidence and regenerate everything:

```sh
python3 scripts/gen_language_surface.py     # -> language_surface.json
# With both a default test build and an opt-in quantum build present, this
# creates fresh isolated traces, runs the complete deterministic corpus, and
# proves the 100% policy without reusing evidence from an earlier shell:
BUILD_DIR=build QUANTUM_BUILD_DIR=build-quantum \
  ./scripts/run_language_coverage.sh

# Existing trace directories may still be aggregated explicitly for forensic
# comparison or CI artifact replay:
LANGUAGE_COVERAGE_RUNTIME_TRACE_DIRS=/tmp/core-trace:/tmp/quantum-trace \
  BUILD_DIR=build ./scripts/run_language_coverage.sh
```

The corpus and instrumentation are deterministic; the trace is per-process TSV
so concurrent test workers can append evidence without sharing mutable state.

### The gate never writes a tracked file

An ordinary run writes its sidecar to `build/coverage/coverage_run.json`, which
is gitignored, so `git status --porcelain` is empty afterwards. Refresh the
**committed** sidecar only when it is meant to change — at a release cut, or
whenever `coverage_policy.json` moves — with the explicit flag:

```sh
BUILD_DIR=build QUANTUM_BUILD_DIR=build-quantum \
  ./scripts/run_language_coverage.sh --update-committed-run
```

That is the only supported way to regenerate `coverage_run.json`; review the
resulting diff like any other artifact. Its nested `coverage_policy` block is
derived from the live `coverage_policy.json`, so a committed sidecar whose
`minimum_covered` disagrees with the policy floor simply means it was not
regenerated after the last ratchet.

## How the manifest is built (ground truth)

`gen_language_surface.py` parses four sources directly so the surface can never
silently drift from the compiler:

1. **Native first-class closure table** — `lib/backend/eshkol_compiler.c`
   `static const BuiltinDef BUILTINS[]` (`{name, native_id, arity}`).
2. **Bytecode VM table** — `lib/backend/eshkol_vm.c` `BUILTINS[]`.
3. **LLVM AOT dispatch** — every `func_name == "name"` in
   `lib/backend/llvm_codegen.cpp`. This is the AOT intrinsic surface and adds
   ~299 builtins absent from the id-tables (R7RS IO/mutation, the NN/optimizer/
   linalg surface, atomics, the extended numeric tower).
4. **Special forms** — `lib/frontend/parser.cpp` `get_operator_type`
   (keyword → `eshkol_op_t`) plus the directly-dispatched forms
   (`begin`, `define-library`, `delay`, `named-let`, ...), and the
   `eshkol_op_t` enum from `inc/eshkol/eshkol.h`.
5. **Tracked Agent FFI APIs** — the Moonlab `provide` surfaces in
   `lib/agent/quantum.esk` and `lib/agent/pqc.esk`. These entries are marked
   `agent_ffi`, not falsely attributed to the core VM/native builtin tables.

Each builtin records which backend(s) register it (`native`, `vm`,
`native_llvm`) so a construct that exists in only one backend is visible.

A row in either `BUILTINS[]` table may also carry a trailing block comment
reading `mirrors: <public-name>`, which the generator copies onto the entry as
`"mirrors"`. It marks the row as one engine's private spelling of a public
construct — an arity split (`_newline1` for the explicit-port `newline`, whose
native codegen takes the optional port directly), or a lower-level handle form
(`make-euclidean-manifold-handle` for `core.manifold`'s
`make-euclidean-manifold`) — rather than a construct of its own. Registering a
row on both engines makes the annotation redundant, and the generator then
fails rather than letting a stale one stand.

The annotation exists because the cross-surface gates could otherwise relate
the two spellings only by name identity, so renaming a private spelling apart
from its public one reported a backend asymmetry that did not exist. It cannot
excuse a real gap: `scripts/p8/five_way_surface.py` resolves the named public
construct on the native surface itself before it treats the row as covered.

## How coverage is measured (dynamic)

`language_coverage.py` is the "ICC tracks the language dynamically" mechanism:

1. The parser records exact source spelling and location (`P`) only when
   `ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR` is set before compilation. It also
   records fully accepted top-level forms (`A`) and expected compile-time
   rejections (`R`, currently `syntax-error`).
2. LLVM code generation records reached AST nodes (`G`) and injects lightweight
   runtime hooks into that instrumented module. Executed operations (`O`) and
   direct calls (`C`) are emitted by the running JIT/AOT program.
3. The bytecode compiler serializes three exact execution witnesses. Native
   calls carry their native-ID alias marker (`V name`), direct Scheme closure
   calls carry a stable 31-bit FNV-1a marker (`V hash @call`) validated
   immediately beside the actual `CALL`/`TAIL_CALL` dispatch, and every
   compiled `(name ...)` form carries the same stable hash as a per-form
   marker (`V hash @form`) at the head of its lowering. The first two fire
   only from builtin dispatch, so before the third existed the arithmetic and
   comparison opcode fast paths and every inline special form produced no VM
   evidence at all — `(display (+ 1 2))` wrote no VM trace file whatsoever.
   `language_coverage.py` resolves hashes only against the checked-in manifest
   and rejects collisions rather than granting ambiguous credit.
4. `language_coverage.py` grants ordinary builtins and runtime forms credit only
   from `O`/`C`/validated `V`. A parser spelling is joined to execution by normalized
   source+line+column, so aliases and reader forms remain auditable after parser
   lowering. `A`/`G` can credit only an explicit allowlist of forms whose
   semantics are compile-time (for example `define`, `require`, and
   `define-syntax`); negative forms require an `R` event.
5. The source-head collector remains as a diagnostic. Its
   `source_exposed_only_names` receive **zero release credit**. A call in an
   untaken branch has `P` and `G`, but no `O`/`C`, and is therefore uncovered.
6. The regression test `scripts/test_runtime_language_coverage.py` exercises a
   real untaken branch, exact ESKB native aliases, exact serialized direct
   Scheme calls, per-form markers on both engine binaries (the standalone VM
   and the `--profile hosted-vm` ESKB route), the differential gate's own
   hash resolution, collision rejection, and an unset trace environment.

Normal generated programs contain no hooks unless tracing was enabled in the
compiler process. Parser dispatch has one cached false branch in production;
trace formatting/allocation occurs only in an opt-in run. Trace writes are
deduplicated per process and flushed in batches.

## Execution-backed guarantee and the deficit ratchet (A1)

The gate certifies **executed behaviour**, never lexical name-presence. Two
numbers are always reported side by side:

- **Execution-backed** — the only gated number. A construct counts as covered
  only when it dispatched/executed in a passing run (`O`/`C`/validated `V`) or,
  for the bounded compile-time-form allowlist, was parsed and
  accepted/code-generated (`A`/`G`). This is the fraction the policy floor and
  the ICC oracle enforce.
- **Lexical exposure** (`lexical_covered` / `spelled_but_unproven`) — a pure
  diagnostic. A construct is "lexically covered" when its name merely appears as
  an application head somewhere in the corpus or a generator. It earns **zero**
  release credit; a name that is spelled but never executed is reported under
  `spelled_but_unproven` and stays uncovered.

`language_coverage.py` enforces this at runtime through
`verify_execution_backed_invariant`: the credited set must be a subset of the
runtime/compile-time evidence, and no source-only head may be counted. If a
future refactor ever routes the lexical `collect_heads` output back into the
gate, that assertion trips instead of silently inflating the number. The
build-free guard `scripts/test_language_coverage_gate.py` pins the same property
with synthetic traces so it can run without a compiler.

`execution_deficit.json` is the monotonic **deficit ledger**: the categorised
list of manifest constructs that lack execution evidence — i.e. the work queue.
It also carries the ratchet baseline (`baseline_execution_backed_covered` /
`_fraction`, `deficit_names`). The gate fails if execution-backed coverage drops
below the baseline **or** if any construct not already in the ledger becomes
uncovered (the deficit list grows). Regenerate it from a fresh run with:

```sh
./scripts/run_language_coverage.sh --write-execution-deficit
```

Writing refuses to record a larger deficit unless `--allow-deficit-growth` is
passed with an explicit, reviewed regression, so the claim is never walked down
silently. The gate emits an `execution_backed_language_coverage` runtime_event
carrying both numbers and the ratchet verdict; the ICC oracle criterion of the
same name consumes it.

## Wiring into the ICC completion-oracle

The coverage fraction is designed to be an oracle criterion, not a one-off
report. `language_coverage.py --emit-runtime-event` prints two long-form ICC
`runtime_event` records: the monotonic floor and the final high-risk-complete
criterion. `--trace PATH` writes them as fresh JSONL evidence:

```json
{"kind": "runtime_event", "event": "language_surface_coverage",
 "name": "language_surface_coverage", "value": "PASS",
 "covered_fraction": 1.0, "covered": 1078, "surface_total": 1078,
 "status": "PASSED"}
```

The integration mirrors `define_loop_flat_rss_aot` and the other release
pillars:

1. `scripts/run_language_coverage.sh` first proves the checked-in manifest is
   source-current, then runs the tracker and writes
   `scripts/icc_traces/language_surface_coverage.jsonl`.
2. `coverage_policy.json` owns the one-way floor. A command-line threshold may
   raise it for an exploratory run but cannot lower it.
3. `eshkol-compiler-readiness` requires the floor event to PASS.
4. `total-language-coverage` additionally requires
   `language_surface_high_risk_complete=PASS`. Phase 4 closes both that
   criterion and the complete surface: every construct in the live manifest has
   deterministic execution evidence and the monotonic policy floor is 100%.
5. Gate the campaign after regenerating traces:
   `icc readiness --repo eshkol --target total-language-coverage --trace-dir scripts/icc_traces`.

Because the threshold ratchets, the oracle enforces monotonic progress toward
total-language coverage: any engine change that drops a previously-covered
construct fails the gate, and the only way to raise the bar is to genuinely
exercise more of the surface.

## Categories (risk buckets)

Every construct is tagged with one category, ordered by silent-wrong risk for
prioritisation: `numeric`, `tensor_ad`, `geometry`, `control_flow`,
`consciousness`, `higher_order`, `list_pair`, `vector`, `string_char`, `hash`,
`predicate`, `io_port`, `binding_form`, `macro_syntax`, `module`,
`memory_region`, `misc_core`, `ffi_system`, `misc`. See `coverage_gap.md` for
the remaining lower-risk surface and the next monotonic ratchet.

## The changelog accounts for every merged pull request

`scripts/check_changelog_completeness.py` is a build-free release gate. It
reads `tag` and `previous_tag` from `release_record.json`, walks
`<previous_tag>..HEAD`, and derives every merged pull request number from the
commit subjects alone (a squash subject ending in `(#N)`, or
`Merge pull request #N ...`); it needs no network. Each number must have
exactly one home:

- a `#N` reference inside the `## [<version>]` section of `CHANGELOG.md` for the
  record's tag, or inside `## [Unreleased]`; a reference in an older release
  section does not count; or
- an entry in `changelog_no_user_facing_change.json`:
  `{"pr": N, "class": ..., "reason": ...}`, where the class is one of `ci`,
  `test-only`, `docs-only`, `release-machinery`, `build-internal`,
  `merge-integration` or `reverted-or-superseded`, and the reason says what the
  pull request changed and why a user cannot observe it.

Anything else is reported as unaccounted, with its commit subject, and the gate
fails. The ledger is graded as strictly as the changelog: an unknown class, a
reason under 25 characters or a generic one, a duplicate, an entry whose pull
request is not in the range (stale), an entry whose pull request is also
referenced in the changelog (one home per pull request), entries out of
numeric order, or a `release` that differs from the record's tag all fail.
Prefer a changelog entry whenever a user could observe the change; the ledger
is for CI wiring, tests, release machinery, integration merges and internal
refactors.

The gate fails closed. Without git, without the previous tag, in a shallow
clone, or when the range yields no pull request at all, it exits 2 (`NO_DATA`)
or 1 rather than passing over an empty list; fetch the full history and tags
(`git fetch --unshallow --tags`) and run it again.

```sh
python3 scripts/check_changelog_completeness.py --no-trace   # grade the tree
python3 scripts/check_changelog_completeness.py --self-test  # every rule, red and green
# In a pull request, before it merges: its own ledger entry is not stale
python3 scripts/check_changelog_completeness.py --no-trace --pending-pr <N>
```

CI runs it in the `assurance-gates` job (full-history checkout, runs on
docs-only pull requests too), and `scripts/release_autopilot.py` runs it with
the other documentation checks before a release. When the record's tag moves to
the next release, set the ledger's `release` to the new tag and delete the
entries the gate then reports as stale.

## Documentation front matter

`scripts/check_doc_front_matter.py` is a build-free gate over every tracked
`*.md` under `docs/` (except the generated `docs/api/`) and the root project
pages. The block and its vocabulary are defined in `docs/DOCUMENTATION.md`.

- A page without front matter is valid and is counted. The count is recorded
  in `doc_front_matter_baseline.json`: a higher count fails (a new page carries
  the block), and a lower count fails until the number is lowered with
  `--update-baseline`, so it only goes down. `--update-baseline` refuses to
  raise it unless `--allow-increase` is passed for a reviewed scope change.
- A page with front matter must have exactly the keys `kind`, `status`,
  `owner-area`, `since` and `sources` (plus `superseded-by` when, and only
  when, the status is `superseded`), values from the closed sets, source paths
  that exist, and its H1 as the first line after the block. A `report` is never
  `current`.
- A current tutorial, guide, reference or explanation page carries no release
  narrative outside code fences; a line that must keep such a phrase carries
  `<!-- evergreen: allow <reason> -->` (on the line or the one above), and every
  exemption is printed.
- A current page that links to a historical or superseded page says
  "historical", "superseded", "dated" or "archived" on the same line.

```sh
python3 scripts/check_doc_front_matter.py --no-trace          # grade the tree
python3 scripts/check_doc_front_matter.py --self-test         # every rule, red and green
python3 scripts/check_doc_front_matter.py --update-baseline   # after adding front matter to a page
```

CI runs it in the `assurance-gates` job, and `scripts/release_autopilot.py`
runs it with the other documentation checks.
