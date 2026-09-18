---
kind: project
status: current
owner-area: docs
since: v1.3.5
sources:
  - tests/coverage/release_record.json
  - scripts/check_surface_counts.py
  - scripts/check_changelog_completeness.py
  - scripts/check_doc_front_matter.py
  - scripts/doc_audit/check_doc_examples.py
  - scripts/doc_audit/extract_examples.py
---
# Writing Eshkol Documentation

How the documentation is organised, where a new page goes, how a page gets its
facts, how examples are executed, and what each documentation gate checks. The
decision record behind this system is
[ADR 0019 — Evergreen documentation architecture](design/adr/0019-evergreen-documentation-architecture.md).

The short version: **a fact has one source, an example runs, a page says what
is true now, and a change updates its pages in the same pull request.**

## Contents

1. [Where a page goes](#1-where-a-page-goes)
2. [Front matter](#2-front-matter)
3. [Evergreen wording](#3-evergreen-wording)
4. [Facts come from sources](#4-facts-come-from-sources)
5. [Examples are executed](#5-examples-are-executed)
6. [Accounting for a pull request](#6-accounting-for-a-pull-request)
7. [Historical pages](#7-historical-pages)
8. [When the page and the code disagree](#8-when-the-page-and-the-code-disagree)
9. [The gates, and how to run them](#9-the-gates-and-how-to-run-them)
10. [The v1.4.0 plan](#10-the-v140-plan)

## 1. Where a page goes

| Kind | Its one job | Location |
|---|---|---|
| `tutorial` | Teach by doing, in order. The reader types what the page says. | `docs/tutorials/` |
| `guide` | Help a user do one task or understand one subsystem as a user. | `docs/guide/` |
| `reference` | Be complete and exact. Generated where a machine source exists. | `docs/reference/`, `docs/api/` |
| `explanation` | Say why it works this way: architecture, design, decisions. | `docs/breakdown/`, `docs/design/`, `docs/design/adr/` |
| `project` | Speak for the project: release copy, policies, process. | repository root, `docs/*.md`, `docs/platform/` |
| `report` | Record a dated measurement, audit or review. | `docs/reports/`, dated files under `docs/design/` |

Decide the kind first. If a draft is doing two jobs (a reference table inside a
tutorial, a design argument inside a reference page), split it and link the
halves. Then:

1. Create the page with [front matter](#2-front-matter).
2. Link it from [`docs/README.md`](README.md), and from the pages a reader would
   arrive from. No page is an orphan; `icc unreferenced-docs` reports zero.
3. A new Architecture Decision Record takes the next free number (one number
   per record, never reused), goes in the
   [ADR index](design/adr/README.md), and is registered with
   `icc adr import-markdown --repo <alias> --dir docs/design/adr`.

## 2. Front matter

Every page you create or substantially edit starts with this block, before the
title:

```yaml
---
kind: guide            # tutorial | guide | reference | explanation | project | report
status: current        # current | historical | superseded
owner-area: types      # the subsystem that owns the page
since: v1.3.5          # the release in which the page first shipped
sources:               # machine sources and source files the page's facts come from
  - lib/types/type_relation.cpp
---
```

- `owner-area` is one of `language`, `types`, `ad`, `tensors`, `stdlib`,
  `runtime`, `vm`, `memory`, `build`, `platform`, `web`, `gpu`, `quantum`,
  `agent`, `testing`, `release`, `docs`, `project`.
- `since` is a release tag. For an existing page it is the first release that
  contains the file.
- `sources` lists repository paths that exist: the manifests the page renders
  figures from, and the source files whose behaviour it describes. Write
  `sources: []` when there are none. This list is what connects a code change
  to the page.
- A `superseded` page adds `superseded-by: <path>`. A `report` is `historical`
  or `superseded`, never `current`.

Pages without front matter are valid and are counted. The count lives in
`tests/coverage/doc_front_matter_baseline.json` and can only go down: when you
add front matter to a page, lower the baseline with
`python3 scripts/check_doc_front_matter.py --update-baseline`.

## 3. Evergreen wording

Tutorials, guides, reference and explanation pages describe **what is true
now**, in the present tense. They carry no release narrative: not "new in",
"recently", "now fixed", "no longer", "this release adds", and no account of how
something used to misbehave. That material belongs in `CHANGELOG.md`,
`RELEASE_NOTES.md`, `ANNOUNCEMENT.md` and [`UPGRADING.md`](UPGRADING.md).

Where provenance helps the reader, annotate the statement: "(since v1.3.5)".

| Instead of | Write |
|---|---|
| "The checker now descends into `cond` bodies." | "The checker examines every `cond` body (since v1.3.5)." |
| "Previously this crashed; it has been fixed." | State the behaviour. The fix is a `CHANGELOG.md` line. |
| "New in v1.3.5: `json-get-in` takes a default." | "`(json-get-in obj path [default])`" with the default described. |

The front-matter gate enforces this on every evergreen page that carries front
matter. A line that must keep such a phrase (a quotation, a heading of a
historical table) carries `<!-- evergreen: allow <reason> -->`.

## 4. Facts come from sources

A number, date, tag or status that more than one page states is rendered from
its source, never retyped.

| Fact | Source | How a page states it |
|---|---|---|
| Release tag, previous tag, date, status, CTest and VM-parity totals | `tests/coverage/release_record.json` | Write the claim once, then `python3 scripts/check_surface_counts.py --sync` rewrites every graded statement; a `<!-- release-record:KEY -->` span renders a record value in place |
| Language constructs, builtin count | `tests/coverage/coverage_policy.json`, `tests/coverage/language_surface.json` | Graded by `check_surface_counts.py` in every registered document |
| Public C and C++ API | Doxygen comments in `inc/eshkol/**/*.h` | `python3 scripts/gen_api_docs.py`; never edit `docs/api/` by hand |
| Language surface manifest | the builtin tables | `python3 scripts/gen_language_surface.py` |
| Test suite inventory | the registered suites | `python3 scripts/check_test_coverage.py` |
| Source line counts cited in prose | `wc -l` at the commit | Graded as typed claims by `scripts/check_doc_claims_residual.py` |
| Silent-wrong ledger aggregate | `.icc/ledger/entries/*.yaml` | `python3 scripts/gen_silent_wrong_ledger.py` |
| Browser import glue | `scripts/wasm_flat_ad_imports.fragment.js`, `scripts/wasm_core_import_keys.json` | `python3 scripts/generate_wasm_import_glue.py --write`, verified by `--check` |
| Site pages | the Markdown sources | `scripts/build-site-content.sh`; never edit `site/static/content/` by hand |

Do not quote a figure from memory, from a previous release's notes or from
another page. Measure it, or cite the manifest that holds it. A figure in a
dated measurement ("measured on the v1.3.4 cut") is a historical statement: say
what it was measured on, and leave it alone afterwards.

## 5. Examples are executed

Every fenced `scheme` block in a gated scope is run by
`scripts/doc_audit/check_doc_examples.py` on the JIT (`eshkol-run -r`, which is
what the REPL runs) and as an AOT binary, in CI and in the release evidence run.
The gated scopes are the entries of `GATED_SCOPES` in
`scripts/doc_audit/extract_examples.py`: `docs/tutorials`, and the guides that
have joined page by page.

```sh
python3 scripts/doc_audit/check_doc_examples.py --eshkol-run build/eshkol-run \
    --only docs/guide/GRADUAL_TYPING.md
```

An example must exit 0 within its time limit without writing an error
diagnostic, and what it prints must be what the page says:

- `;; => value` on the line of a form, or on the line under it, is compared with
  the value the build shows. The annotation names a value, so `12.0` matches a
  printed `12` and `"abc"` matches `abc`.
- A block that starts with `> ` is a REPL transcript: the lines under each input
  are its expected output.
- A bare, `text` or `output` fence directly under an example is its exact
  standard output. Show compiler diagnostics (standard error) in a `console`
  fence, which is not compared, and take the text from a real run.

A block is tried on its own and then after the page's earlier examples, so a
later example may use an earlier definition.

### Markers

When an example cannot be checked, say so in an HTML comment on the line above
its fence. The rendered page does not change.

| Marker | Meaning |
|---|---|
| `<!-- doc-example: skip <reason>: <why> -->` | Not executed. |
| `<!-- doc-example: run-only <reason>: <why> -->` | Executed and must exit 0; its output is not compared. |
| `<!-- doc-example: known-defect <LEDGER-ID>: <what the page promises> -->` | The page states the designed behaviour and the build does not deliver it yet. The ledger entry must be open. The example is still run, and the gate fails the day it passes, so the marker cannot outlive the defect. |
| `<!-- doc-example: file <name>: <what it is> -->` | The block is also written next to the page's later examples as `<name>`, for `require`, `load` or file I/O. Not an exclusion. |
| `<!-- doc-example: output stdout: <what it is> -->` | On a `text` fence further down the page: it is the standard output of the nearest example above it. |

`<reason>` is one of `pseudo-code`, `fragment`, `platform-specific`,
`nondeterministic`, `interactive`, `external-resource`. Nothing is skipped by
heuristic. Every marked example is printed with its reason on every run, and the
number of marked examples per file is ratcheted in
`scripts/doc_audit/example_gate_baseline.json` in both directions: after
removing a marker, lower it with `--update-baseline`.

### Putting a page or a directory under the gate

1. Make every example on the page pass, or mark it, using `--only <page>`.
2. Add the path to a scope in `GATED_SCOPES`
   (`scripts/doc_audit/extract_examples.py`). A path is a directory or a single
   file; a new scope is one more key.
3. Record it: `python3 scripts/doc_audit/check_doc_examples.py --eshkol-run
   build/eshkol-run --scope <scope> --update-baseline`, and review the diff of
   `example_gate_baseline.json`.

Nothing else names a documentation path. CI runs every gated scope, and the
change classifier (`scripts/ci_change_class.py`) reads the same table, so a
pull request that touches only a gated page is classified `tests-only` and the
lanes that execute the examples run for it.

Outside the gated scopes the rule is the same and you are the gate: run every
sample you add or change on the JIT and as an AOT binary, and paste what it
printed.

## 6. Accounting for a pull request

`scripts/check_changelog_completeness.py` walks the merged pull requests between
the release record's `previous_tag` and `HEAD`. Each one must be in exactly one
of two places:

- referenced as `#N` in the `CHANGELOG.md` section for the release (or in
  `[Unreleased]`), if a user could observe the change: language, library,
  runtime, tooling, build, packaging, documentation a user reads, examples,
  benchmarks;
- listed in `tests/coverage/changelog_no_user_facing_change.json` with a class
  and a specific reason, if no user could: CI wiring, a test-only change,
  release machinery, an integration merge.

The changelog line is part of the change. Write it in the pull request that
makes the change: `Added` for a capability, `Changed` for behaviour a user will
notice, `Fixed` for a correction, stated plainly. Release copy
(`README.md`, `RELEASE_NOTES.md`, `ANNOUNCEMENT.md`, the press sheets, the site)
states what is true and carries no account of defects.

## 7. Historical pages

A dated report, an audit and a superseded design are records. Do not bring
their figures up to date: mark the page instead.

```yaml
---
kind: report
status: historical
owner-area: testing
since: v1.3.5
sources: []
---
```

A current page that links to a historical or superseded page says so on the same
line ("the dated audit of 2026-08-25", "superseded by ..."). When a later
document corrects a figure, the correction is propagated to every current page
that cites it; `icc citation-binding` and `icc supersession-propagation` list
the citations still to fix.

## 8. When the page and the code disagree

Decide which one is the design before touching either.

- The page is wrong about what exists (a signature, a flag, a count, a path):
  fix the page.
- The page states the designed behaviour and the implementation lags: the page
  stays as written. File the gap as a ledger entry under `.icc/ledger/entries/`,
  and if an executed example shows it, mark that example `known-defect` with the
  entry's id.

A page is never edited down to match a defect.

## 9. The gates, and how to run them

Build-free, from the repository root:

```sh
python3 scripts/check_surface_counts.py --no-trace        # counts, dates, tag and status against their sources
python3 scripts/check_changelog_completeness.py --no-trace # every merged pull request is accounted for
python3 scripts/check_doc_front_matter.py --no-trace      # front matter, evergreen wording, historical links, unmarked-page ratchet
python3 scripts/gen_api_docs.py --check                   # docs/api/ matches the headers
python3 scripts/gen_language_surface.py --check           # the surface manifest matches the builtin tables
python3 scripts/check_ledger_integrity.py                 # ledger entries are well-formed
python3 scripts/check_test_coverage.py                    # the suite inventory is current
python3 scripts/check_disclosure.py --base origin/master --head HEAD   # nothing private in the diff
```

Each has a `--self-test` that proves it can fail. With a build:

```sh
python3 scripts/doc_audit/check_doc_examples.py --eshkol-run build/eshkol-run
```

With the repository's code index (ICC), after `icc reindex`:

```sh
icc doc-typed-claims --repo <alias>                      # counts, line figures and paths against the tree
python3 scripts/check_doc_claims_residual.py --icc-bin <icc> --repo <alias>
icc citation-binding --repo <alias>                      # withdrawn figures still cited as current
icc supersession-propagation --repo <alias>              # corrections not yet propagated
icc unreferenced-docs --repo <alias>                     # orphan pages
icc doc-coverage --repo <alias>                          # public symbols without documentation
```

A wrong typed claim is corrected in the text. Only a finding the detector
misattributes (a cross-repository citation, an aggregate bound to the wrong
file) or a dated historical statement goes in
`.icc/doc-claims-allowlist.yaml`, with a reason in the style of the existing
entries.

## 10. The v1.4.0 plan

Recorded in ADR 0019 and listed here so that a contributor can pick one up.

**Generated references.** A CLI reference for every binary harvested from its
`--help`; an environment-variable registry harvested from the `getenv` call
sites; a diagnostics catalogue harvested from the compiler's messages; a stdlib
export index harvested from `provide` forms. Each gets a `--check` freshness
gate like `gen_api_docs.py`.

**The doc-impact gate.** The `sources` lists and the doc-symbol graph map each
public symbol, builtin, flag, environment variable and diagnostic to its pages.
A pull request that changes one of them touches the mapped pages or carries a
reasoned "no documentation impact" entry.

**Coverage.** The example gate over `docs/guide`, `docs/reference` and the
README samples; front matter on every page; the ADR registry gate; typed claims,
citations and supersessions as a required check.

**Pages.** Guides, in priority order from user questions and the feature
inventory: tensors and a training loop; exact arithmetic and the numeric tower;
memory, regions and long-running programs; parallel programming; modules and
packages; the REPL and machine mode; embedding in C and C++; Python bindings;
the agent FFI; the bytecode VM and when to use it; WebAssembly and the web
platform; GPU backends; quantum backends; logic and knowledge bases; debugging
and reading diagnostics; performance and profiling; testing Eshkol programs.
Reference: file formats (ESKM); the ABI and embedding API; the generated
references above. Project: governance and support policy (supported platforms,
host compilers, deprecation policy, versioning).
