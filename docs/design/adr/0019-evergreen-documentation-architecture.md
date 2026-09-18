# ADR 0019 — Evergreen documentation architecture

- Status: Accepted — partially implemented: the v1.3.5 stage below is in the tree; later stages Proposed
- Date: 2026-09-17
- Decision owners: Eshkol maintainers; documentation and release assurance maintainers
- Related: ADR-0010 (closed-loop assurance), ADR-0014 (release invariant contracts),
  [`docs/DOCUMENTATION.md`](../../DOCUMENTATION.md) (the contributor guide to this system)

## Context

Documentation that is written by hand in more than one place goes stale between
sweeps. A release fact (a date, a count, a status) is restated in a dozen
documents; an example claims an output nothing executes; a behaviour change
merges and no page that describes the behaviour is touched; a dated report keeps
being read as a current statement. Periodic sweeps repair the symptoms and the
repair does not last, because nothing connects a change in the code to the pages
that describe it. The documentation has to grow with the project by
construction: every class of drift needs a source of truth and a gate, not a
reviewer's memory.

## Decision

### Principles

1. **One source per fact.** A fact that appears in more than one document is
   rendered from a machine source and never retyped. The sources are
   `tests/coverage/release_record.json` (tag, previous tag, date, status,
   evidence totals), `tests/coverage/language_surface.json` and
   `tests/coverage/coverage_policy.json` (constructs, builtins), the generated
   `docs/api/` (public headers), and the package manifest (release assets).
2. **Every example runs.** A code block that claims output is executed by the
   documentation example gate on every engine it claims. An example that cannot
   run carries an explicit marker with a reason, and the marked count can only
   go down.
3. **Every claim is typed and checked.** Counts, line figures, file paths,
   citations and corrections are extracted and graded against the tree. A wrong
   claim is corrected in the text; a finding the detector misattributes is
   recorded in the allowlist with its reason.
4. **A code change names its pages.** A pull request that changes a public
   symbol, builtin, flag, environment variable or diagnostic touches the pages
   that describe it, or carries a reasoned statement that no page is affected.
   A merged pull request is referenced by the changelog or listed in the
   no-user-facing-change ledger.
5. **Evergreen pages carry no release narrative.** Tutorials, guides, reference
   and explanation pages describe what is true now, in the present tense.
   Release-scoped statements live in `CHANGELOG.md`, `RELEASE_NOTES.md`,
   `ANNOUNCEMENT.md` and `docs/UPGRADING.md`. A page may annotate a statement
   "(since v1.3.5)"; it does not say "new" or "recently".
6. **History is dated, not rewritten.** Reports, audits and superseded designs
   carry a status in their front matter. Their figures are frozen, and a current
   page that links to one says that it is historical.
7. **Docs are the blueprint.** When a page and the code disagree, first decide
   which one is the design. If the page is, the gap is a build item with a
   ledger entry, and the page is not edited down.

### Four kinds of page, and the project pages

| Kind | Job | Location | Grows when |
|---|---|---|---|
| `tutorial` | Learn by doing, in order | `docs/tutorials/` | A capability needs a guided first use |
| `guide` | Solve one task, or learn one subsystem as a user | `docs/guide/` | Users ask "how do I" |
| `reference` | Complete and exact, generated where possible | `docs/reference/`, `docs/api/` | The surface changes |
| `explanation` | Why it works this way | `docs/breakdown/`, `docs/design/adr/` | A design decision is made |

Two further kinds complete the tree. `project` pages speak for the project as a
whole: `README.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`, `ROADMAP.md`,
`CONTRIBUTING.md`, `SECURITY.md`, `docs/KNOWN_ISSUES.md`, `docs/UPGRADING.md`,
`docs/TROUBLESHOOTING.md`, `docs/FAQ.md`, `docs/GLOSSARY.md`,
`docs/DOCUMENTATION.md` and the release process. `report` pages are dated
records: audits, measurement reports and review notes. Every page is reachable
from `docs/README.md`.

### Front matter

Every page begins with a YAML block:

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

A `superseded` page adds `superseded-by: <path>`. A `report` is never `current`.
The `owner-area` vocabulary is closed: `language`, `types`, `ad`, `tensors`,
`stdlib`, `runtime`, `vm`, `memory`, `build`, `platform`, `web`, `gpu`,
`quantum`, `agent`, `testing`, `release`, `docs`, `project`. `sources` is what
the doc-impact gate reads: a change to a listed source is a change the page has
to account for.

Front matter is adopted incrementally. A page that is created or substantially
edited gains it; the count of pages without it is recorded in
`tests/coverage/doc_front_matter_baseline.json` and can only go down.

### Gates

All gates are build-free unless noted, and each has a self-test.

| Gate | Fails when | State at v1.3.5 |
|---|---|---|
| Surface and release-record consistency (`scripts/check_surface_counts.py`) | a stated count, date, tag or status disagrees with its source | In force |
| Documentation example execution (`scripts/doc_audit/check_doc_examples.py`, needs a build) | an unmarked example fails, or the marked count changes | In force for `docs/tutorials` and the gated guides; scope grows page by page |
| Changelog completeness (`scripts/check_changelog_completeness.py`) | a merged pull request in the release range is neither referenced by the changelog nor listed in `tests/coverage/changelog_no_user_facing_change.json` | In force |
| Front matter, evergreen wording and historical links (`scripts/check_doc_front_matter.py`) | a front-matter block is malformed, names a source that does not exist, an evergreen page carries release narrative, a current page presents a historical page as current, or the count of pages without front matter differs from the recorded baseline (it may only be lowered, with `--update-baseline`) | In force for pages that carry front matter |
| Generated API reference (`scripts/gen_api_docs.py --check`) | `docs/api/` differs from a fresh harvest of the public headers | In force |
| Language surface (`scripts/gen_language_surface.py --check`) | the surface manifest differs from the builtin tables | In force |
| Typed claims (`scripts/check_doc_claims_residual.py`, needs ICC) | a wrong typed claim is neither corrected, allowlisted with a reason, nor an open build item | In force in the release evidence run |
| Citations, supersessions, contradictions (ICC) | a withdrawn figure is still cited as current, or a correction is not propagated | Run for each documentation change; a required check from v1.4.0 |
| Site release facts | the site's tag, headline or install links disagree with the release record | In force with the site build |
| Generated reference freshness | the CLI, environment-variable, diagnostics or stdlib export reference differs from a fresh harvest | v1.4.0 |
| Doc impact | a changed public symbol, flag, variable or diagnostic whose mapped pages are untouched and unexcused | v1.4.0 |
| ADR registry | a duplicate ADR number, an unregistered ADR, or a status that disagrees with the code | v1.4.0 |

### The site is a projection

The website publishes the same tree. It is never a second copy: release facts
on the site come from the release record, and a page manifest (file, slug,
navigation section, kind) drives both the `docs/README.md` navigation and the
site build.

### Staging

- **v1.3.5.** Every existing claim graded and dispositioned; the project pages
  (`UPGRADING`, `TROUBLESHOOTING`, `RELEASE_PROCESS`, `DOCUMENTATION`,
  `GLOSSARY`); the changelog-completeness gate; front matter and its check on
  every page created or substantially edited; this record.
- **v1.4.0.** Generated references: the CLI reference harvested from each
  binary's `--help`, the environment-variable registry harvested from the
  `getenv` call sites, the diagnostics catalogue harvested from the compiler's
  messages, the stdlib export index harvested from `provide` forms. The
  doc-impact gate as a required check. The example gate over guides, reference
  pages and the README. Front matter on every page. The ADR registry gate. The
  guide inventory in `docs/DOCUMENTATION.md`, written in priority order.
- **v1.5.0.** Per-release archived documentation on the site, search, API
  examples executed per symbol, and a translation-ready structure.

### The rule for every change

A change that alters behaviour updates the pages that describe it in the same
pull request, runs the documentation gates and reports their output. A change
that adds a capability adds its reference entry, at least one runnable example
and a changelog line. "Docs later" is not a state.

## Consequences

A fact has one home, so correcting it is one edit and a `--sync`. A reader can
tell a current statement from a dated one by the page's status. A pull request
that would leave the changelog or a gated page behind fails before it merges.
The cost is a front-matter block per page and a marker on every example that
cannot run; both are small, local and checked.

Front matter is a leading YAML block: the site build consumes it as metadata,
so it does not appear in the rendered site pages, and pages without it remain valid, which is what lets adoption be incremental.

## Verification

`scripts/check_doc_front_matter.py --self-test` and
`scripts/check_changelog_completeness.py --self-test` exercise each rule red and
green. Both gates, `check_surface_counts.py` and the example gate's self-test
run in the `assurance-gates` CI job and in the release documentation checks.
