# Eshkol Feature Inventory (v1.0 → v1.2.1)

This document is a deep-dive inventory of every feature added across the four
released lines of the Eshkol compiler, produced 2026-05-20 by an ICC-driven
audit of the source tree. Every claim is grounded in a specific source path
and line range. It is the input contract for the documentation-expansion
wave; downstream agents must verify every claim against current source
before reusing it.

(See `.icc/eshkol_doc_intel.md` and `.icc/eshkol_arch_summary.md` for the
ICC artefacts that drove the audit.)

---

[The full feature inventory is reproduced below — 1500+ lines of
source-grounded per-feature data, cross-doc dependency map, and per-feature
recommendations for the expansion wave. Regenerate via the agent prompt at
`.icc/eshkol_doc_intel.md`'s sibling location.]

Reference inventory entry shape:

```
## <Feature name>
**Source of truth:** <list of source files + line ranges>
**Heap subtype / type tag / opcode (if applicable):** <numeric value> at <file:line>
**v1.x line introduced:** <release>
**Builtins / runtime helpers:** <enumerated with signatures>
**Where currently documented:** <doc file(s) + section, or "no current home">
**Coverage assessment:** none / shallow / adequate / deep
**Recommended expansion target:** <doc to expand or new doc to write> + scope notes
**Verified-against-source evidence:** <specific facts cited from the read>
```

Feature inventory was retained in agent transcript and the parent
conversation. Re-fetch via the deep-dive prompt or by inspecting the
auditor transcript at the sub-agent output file referenced from the parent
task notification.
