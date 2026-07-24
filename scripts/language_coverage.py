#!/usr/bin/env python3
"""language_coverage.py — execution-backed language-surface coverage.

Runs the generative exposure engines plus the opt-in quantum test corpus,
collects the BUILTINS and SPECIAL FORMS that generated code actually executes,
diffs
that against the ground-truth manifest (tests/coverage/language_surface.json),
and reports covered% plus the categorised list of UNCOVERED constructs — the
constructs no engine exercises today.

This is the "ICC tracks every part of the language dynamically" mechanism: each
run emits a coverage sidecar (tests/coverage/coverage_run.json) whose
`covered_fraction` and `uncovered` fields are consumable as an ICC
completion-oracle runtime_event (see tests/coverage/README.md).

Engines measured:
  * scripts/gen_generative_corpus.py        (run_generative_differential.py)
  * tests/ad_adversarial/gen_ad_adversarial.py (run_ad_adversarial.sh)
  * tests exercised by scripts/run_all_tests.sh (complete CI suite)
  * tests/quantum/*.esk                     (quantum-macos CI lane)

Usage:
  python3 scripts/language_coverage.py --runtime-trace-dir DIR [DIR ...]
      [--json OUT] [--emit-runtime-event]
"""

import argparse
import glob
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests", "ad_adversarial"))

MANIFEST = os.path.join(REPO, "tests", "coverage", "language_surface.json")
POLICY = os.path.join(REPO, "tests", "coverage", "coverage_policy.json")
# Monotonic execution-backed deficit ledger. This is the work queue AND the
# named-construct ratchet baseline: the gate fails if execution-backed coverage
# drops below its recorded floor or if any construct not already listed here
# becomes uncovered (the deficit list grows). Lexical name-presence earns zero
# credit; it is reported only as the diagnostic `spelled_but_unproven` set.
EXECUTION_DEFICIT = os.path.join(REPO, "tests", "coverage",
                                 "execution_deficit.json")

# Scheme test roots exercised by scripts/run_all_tests.sh. Keep this list in
# the same order as that suite's TEST_SCRIPTS array. The text scan is diagnostic
# only: it explains which constructs appear in the committed corpus, but earns
# no coverage credit. Release credit comes exclusively from P/A/G/O/C/R/V runtime
# traces, so an untaken branch or dead helper remains uncovered.
CI_TEST_GLOBS = (
    "tests/features/*.esk", "tests/stdlib/*.esk", "tests/lists/*.esk",
    "tests/memory/*.esk", "tests/modules/*.esk", "tests/types/*.esk",
    "tests/typesystem/*.esk", "tests/autodiff/*.esk",
    "tests/autodiff_debug/*.esk", "tests/manifold/*.esk", "tests/ml/*.esk",
    "tests/neural/*.esk",
    "tests/json/*.esk", "tests/system/*.esk", "tests/complex/*.esk",
    "tests/parser/*.esk", "tests/control_flow/*.esk", "tests/logic/*.esk",
    "tests/bignum/*.esk", "tests/rational/*.esk", "tests/parallel/*.esk",
    "tests/signal/*.esk", "tests/xla/*.esk", "tests/gpu/*.esk",
    "tests/error_handling/*.esk", "tests/macros/*.esk", "tests/repl/*.esk",
    "tests/web/*.esk", "tests/tco/*.esk", "tests/io/*.esk", "tests/ffi/*.esk",
    "tests/r7rs/*.esk",
    "tests/benchmark/*.esk", "tests/benchmarks/*.esk", "tests/migration/*.esk",
    "tests/codegen/**/*.esk", "tests/numeric/*.esk", "tests/sicp/**/*.esk",
    "tests/v1_2_edge_cases/*.esk",
)

# Deterministic VM extension probes run by scripts/run_vm_surface_tests.sh.
# The broader tests/vm directory also contains host-side-effect probes (signals,
# servers, process replacement), so list the pure numeric/stateful subset
# explicitly instead of claiming unexecuted coverage.
CI_VM_SURFACE_TESTS = (
    "tests/vm/geometric_fallback_numeric_regression.esk",
    "tests/vm/riemannian_adam_state_regression.esk",
    "tests/vm/kb_factor_graph_extensions_regression.esk",
    "tests/vm/workspace_introspection_regression.esk",
    "tests/vm/ad_tape_lowlevel_regression.esk",
    "tests/vm/vm_kb_tensor_test.esk",
)

CI_VM_SURFACE_GLOBS = (
    "tests/vm/*_surface_regression.esk",
)

CI_SURFACE_EXTENSION_TESTS = (
    "tests/dnc/dnc_test.esk",
    "tests/quant/dequant_test.esk",
    "tests/sdnc/sdnc_api_test.esk",
    "tests/v1_3_edge_cases/tensor_dtype_test.esk",
    "tests/ad/one_pass_gradient_test.esk",
    "tests/ad/sparse_tensors_test.esk",
    "tests/ad/taylor_tower_test.esk",
    "tests/macros/negative/syntax_error_negative.esk",
)

# reader macros -> the special form they expand to
READER_MACROS = {
    "'": "quote",
    "`": "quasiquote",
    ",@": "unquote-splicing",
    ",": "unquote",
}

# Forms whose complete behavior occurs while parsing or generating a module,
# rather than by evaluating an expression in the generated program. These earn
# credit only when an exact source location has both a parser-dispatch event and
# a successful top-level-accept/codegen event. Ordinary calls and runtime forms
# must have an execution event; merely appearing in generated IR is not enough.
COMPILE_TIME_FORMS = {
    "define", "define-values", "define-type", "define-syntax",
    "define-record-type", "define-library", "extern", "extern-var",
    "import", "require", "load", "provide", "include", "include-ci",
    "cond-expand", "let-syntax", "letrec-syntax", "syntax-error",
}

NEGATIVE_COMPILE_TIME_FORMS = {"syntax-error"}


def vm_call_name_hash(name):
    """Match the VM compiler's stable non-negative 31-bit FNV-1a hash."""
    value = 2166136261
    for byte in name.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value & 0x7FFFFFFF


def normalize_trace_source(source):
    """Return a stable repo-relative identity for a trace source path."""
    source = source.replace("\\", "/")
    if source in ("", "<unknown>", "unknown"):
        return "<unknown>"
    absolute = os.path.abspath(source)
    try:
        if os.path.commonpath((REPO, absolute)) == REPO:
            return os.path.relpath(absolute, REPO).replace("\\", "/")
    except ValueError:
        pass
    for marker in ("/tests/", "/lib/", "/examples/", "/benchmarks/"):
        if marker in source:
            return marker.strip("/") + "/" + source.split(marker, 1)[1]
    return source


def repo_relative_or_absolute(path):
    """Return a repo-relative path when possible, otherwise an absolute path.

    os.path.commonpath raises ValueError when paths live on different Windows
    drives. Keep sidecar generation portable instead of letting an external
    trace directory make the coverage gate crash.
    """
    absolute = os.path.abspath(path)
    try:
        if os.path.commonpath((REPO, absolute)) == REPO:
            return os.path.relpath(absolute, REPO).replace("\\", "/")
    except ValueError:
        pass
    return absolute.replace("\\", "/")


def load_runtime_evidence(trace_dirs):
    """Parse strict TSV traces and return execution-backed surface names.

    P records identify exact source spelling. O/C records prove evaluation at
    the source location. A/G records are accepted only for the explicit
    compile-time-form allowlist. This is what excludes calls in untaken
    branches: they have P and G records but no O/C record. V records are exact
    bytecode-VM native dispatches whose alias marker survived ESKB
    serialization and was validated against the dispatched native ID.
    """
    trace_paths = []
    for trace_dir in trace_dirs:
        trace_paths.extend(glob.glob(os.path.join(trace_dir, "**", "*.tsv"),
                                     recursive=True))
    trace_paths = sorted(set(trace_paths))
    if not trace_paths:
        raise RuntimeError("no runtime language-coverage TSV files found in: %s"
                           % ", ".join(trace_dirs))

    parsed = []
    executed_locations = set()
    executed_positions = set()
    accepted_locations = set()
    accepted_positions = set()
    generated_locations = set()
    generated_positions = set()
    direct_calls = set()
    vm_calls = set()
    vm_call_hashes = set()
    rejected_forms = set()
    counts = {kind: 0 for kind in ("P", "A", "G", "O", "C", "R", "V")}

    for path in trace_paths:
        with open(path, encoding="utf-8", errors="strict") as fh:
            for line_number, raw in enumerate(fh, 1):
                raw = raw.rstrip("\n")
                if not raw:
                    continue
                fields = raw.split("\t")
                kind = fields[0]
                expected = 6 if kind in ("P", "R", "V") else 5
                if kind not in counts or len(fields) != expected:
                    raise RuntimeError("malformed runtime trace %s:%d: %r"
                                       % (path, line_number, raw))
                try:
                    source = normalize_trace_source(fields[1])
                    line = int(fields[2])
                    column = int(fields[3])
                    operation = int(fields[4]) if kind != "C" else None
                    location = (source, line, column, operation)
                except ValueError as exc:
                    raise RuntimeError("invalid runtime trace %s:%d: %s"
                                       % (path, line_number, exc)) from exc
                counts[kind] += 1
                if kind == "P":
                    parsed.append((location, fields[5]))
                elif kind == "V":
                    if fields[5] == "@call":
                        vm_call_hashes.add(operation)
                    else:
                        vm_calls.add(fields[5])
                elif kind == "R":
                    if fields[5] in NEGATIVE_COMPILE_TIME_FORMS:
                        rejected_forms.add(fields[5])
                elif kind == "C":
                    direct_calls.add(fields[4])
                elif kind == "O":
                    executed_locations.add(location)
                    executed_positions.add(location[:3])
                elif kind == "A":
                    accepted_locations.add(location)
                    accepted_positions.add(location[:3])
                elif kind == "G":
                    generated_locations.add(location)
                    generated_positions.add(location[:3])

    # Resolve executed direct Scheme callsites only against the checked-in
    # language manifest.  Refuse manifest collisions: an ambiguous marker can
    # never grant credit.  Non-surface helper/user-function hashes are ignored.
    manifest = load_manifest()
    manifest_names = {
        item["name"]
        for section in ("builtins", "special_forms", "prelude")
        for item in manifest[section]
        if not item["name"].startswith("_")
    }
    names_by_hash = {}
    for name in manifest_names:
        names_by_hash.setdefault(vm_call_name_hash(name), set()).add(name)
    for name_hash in vm_call_hashes:
        candidates = names_by_hash.get(name_hash, set())
        if len(candidates) > 1:
            raise RuntimeError(
                "ambiguous serialized-VM call hash %d maps to: %s"
                % (name_hash, ", ".join(sorted(candidates)))
            )
        if len(candidates) == 1:
            vm_calls.update(candidates)

    covered = set(direct_calls) | set(vm_calls) | set(rejected_forms)
    runtime_spelling_matches = set()
    compile_time_matches = set()
    spellings_by_position = {}
    for location, name in parsed:
        spellings_by_position.setdefault(location[:3], set()).add(name)
    for location, name in parsed:
        if location[0] == "<unknown>":
            continue
        position = location[:3]
        unambiguous_position = len(spellings_by_position[position]) == 1
        if (location in executed_locations or
                (unambiguous_position and position in executed_positions)):
            covered.add(name)
            runtime_spelling_matches.add(name)
        elif (name in COMPILE_TIME_FORMS and
              (location in accepted_locations or location in generated_locations or
               (unambiguous_position and
                (position in accepted_positions or position in generated_positions)))):
            covered.add(name)
            compile_time_matches.add(name)

    # These helpers are compiler-generated by the corresponding executed
    # promise forms and have no user-spelled call site of their own.
    if "delay" in covered:
        covered.add("%make-lazy-promise")
    if "delay-force" in covered:
        covered.add("%make-lazy-promise-force")

    return {
        "covered_names": covered,
        "direct_call_names": direct_calls,
        "vm_dispatch_names": vm_calls,
        "rejected_compile_time_names": rejected_forms,
        "runtime_spelling_matches": runtime_spelling_matches,
        "compile_time_matches": compile_time_matches,
        "trace_paths": trace_paths,
        "event_counts": counts,
    }


def collect_heads(text):
    """Return the set of symbols that appear as an application/operator head, plus
    the special forms introduced by reader macros (', `, , ,@) and #( vectors."""
    heads = set()
    # Named let is reader syntax `(let name ((binding value) ...) body ...)`,
    # not a literal `(named-let ...)` head.  Record the semantic special form
    # when that executed source shape is present; otherwise the manifest can
    # never credit the real named-let stress tests.
    if re.search(r"\(\s*let\s+[^\s()]+\s*\(\s*\(", text):
        heads.add("named-let")
    # reader-macro forms actually present in the source
    if "#(" in text:
        heads.add("vector")           # vector literal reader syntax
    for rm, form in READER_MACROS.items():
        if rm in text:
            heads.add(form)
    # walk tokens; a symbol immediately following '(' is a head
    i, n = 0, len(text)
    expect_head = False
    while i < n:
        c = text[i]
        if c == ";":                  # line comment
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == '"':                  # string literal
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\":
                    i += 1
                i += 1
            i += 1
            expect_head = False
            continue
        if c == "#" and i + 1 < n and text[i + 1] == "\\":  # char literal
            i += 2
            while i < n and (text[i].isalnum()):
                i += 1
            expect_head = False
            continue
        if c == "(":
            expect_head = True
            i += 1
            continue
        if c in ") \t\n\r'`,":
            i += 1
            continue
        # read an atom
        j = i
        while j < n and text[j] not in "() \t\n\r\";":
            j += 1
        atom = text[i:j]
        if expect_head and atom:
            heads.add(atom)
        expect_head = False
        i = j
    return heads


def gen_generative_corpus_text():
    import gen_generative_corpus as g
    progs = g.generate_programs()
    return "\n".join(body for _, _, body in progs)


def gen_ad_adversarial_text():
    import gen_ad_adversarial as a
    probes = a.Gen().generate()
    parts = [a.PRELUDE, a.SUMMARY]
    for p in probes:
        parts.append("\n".join(p["lines"]))
    return "\n".join(parts)


def quantum_test_text():
    """Return the exact quantum-enabled test corpus exercised by CI.

    This source is reported only as diagnostic exposure. The corresponding
    quantum runtime trace is what earns coverage credit.
    """
    paths = sorted(glob.glob(os.path.join(REPO, "tests", "quantum", "*.esk")))
    if not paths:
        raise RuntimeError("no quantum coverage tests found")
    parts = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            parts.append(fh.read())
    # The gated quantum corpus loads these two modules and exercises their
    # public wrappers end-to-end.  Include the exact loaded module sources so
    # compiler intrinsics such as vqe-energy-primitive receive transitive,
    # execution-backed coverage instead of requiring a fake direct test call.
    for relative in ("lib/agent/quantum.esk", "lib/agent/pqc.esk"):
        module_path = os.path.join(REPO, relative)
        if not os.path.isfile(module_path):
            raise RuntimeError("quantum module source missing: %s" % relative)
        with open(module_path, encoding="utf-8", errors="replace") as fh:
            parts.append(fh.read())
        paths.append(module_path)
    return "\n".join(parts), [os.path.relpath(path, REPO) for path in paths]


def complete_ci_test_text():
    """Return the committed Scheme corpus exercised by run_all_tests.sh.

    Paths are de-duplicated because a recursive and a direct glob may overlap.
    Empty roots are tolerated across feature-gated checkouts, but the complete
    corpus itself must be non-empty. This text is diagnostic only; runtime
    traces, not source presence, determine release coverage.
    """
    paths = set()
    for pattern in CI_TEST_GLOBS:
        paths.update(glob.glob(os.path.join(REPO, pattern), recursive=True))
    paths.update(os.path.join(REPO, path) for path in CI_VM_SURFACE_TESTS)
    for pattern in CI_VM_SURFACE_GLOBS:
        paths.update(glob.glob(os.path.join(REPO, pattern)))
    paths.update(os.path.join(REPO, path) for path in CI_SURFACE_EXTENSION_TESTS)
    paths = sorted(path for path in paths if os.path.isfile(path))
    if not paths:
        raise RuntimeError("complete CI Scheme corpus is empty")
    parts = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            parts.append(fh.read())
    return "\n".join(parts), [os.path.relpath(path, REPO) for path in paths]


def load_manifest():
    with open(MANIFEST) as fh:
        return json.load(fh)


def load_policy(path):
    with open(path, encoding="utf-8") as fh:
        policy = json.load(fh)
    required = {
        "schema_version",
        "minimum_covered",
        "minimum_covered_fraction",
        "baseline_surface_total",
        "high_risk_categories",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError("coverage policy missing keys: %s" % ", ".join(missing))
    return policy


def write_runtime_events(path, events):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for event in events:
            json.dump(event, fh, sort_keys=True)
            fh.write("\n")


def verify_execution_backed_invariant(covered, runtime_names, surface,
                                      source_heads):
    """Refuse to certify unless every credited construct has runtime evidence.

    This is the permanent tripwire against re-introducing name-presence as a
    coverage credit path: `covered` must be a subset of the runtime/compile-time
    evidence set, and no construct that is *only* present in corpus text (a
    source head with no O/C/V/allowlisted-A/G witness) may appear as covered. A
    future refactor that routes the lexical `collect_heads` output back into the
    gate trips this assertion instead of silently inflating the number.
    """
    covered_names = set(covered)
    leaked = covered_names - set(runtime_names)
    if leaked:
        raise RuntimeError(
            "execution-backed invariant violated: %d construct(s) counted as "
            "covered without runtime evidence (name-presence leak): %s"
            % (len(leaked), ", ".join(sorted(leaked))))
    source_only = (set(surface) & set(source_heads)) - set(runtime_names)
    lexical_leak = source_only & covered_names
    if lexical_leak:
        raise RuntimeError(
            "execution-backed invariant violated: %d lexically-spelled but "
            "unexecuted construct(s) counted as covered: %s"
            % (len(lexical_leak), ", ".join(sorted(lexical_leak))))


def build_execution_deficit(surface, covered, uncovered_by_category,
                            execution_fraction, lexical_covered,
                            spelled_but_unproven, execution_only_names):
    """Return the serializable execution-backed deficit ledger."""
    surface_total = len(surface)
    deficit_names = sorted(name for names in uncovered_by_category.values()
                           for name in names)
    return {
        "schema_version": 1,
        "doctrine": (
            "Execution-backed evidence is the ONLY coverage gate. A construct "
            "is covered only when it dispatched/executed in a passing run "
            "(O/C/V) or, for the bounded compile-time-form allowlist, was "
            "parsed and accepted/generated (A/G). Lexical name-presence in "
            "corpus/generator text earns ZERO credit and is reported under "
            "`spelled_but_unproven` for diagnostics only. This ledger is the "
            "monotonic work queue: the gate FAILS if execution-backed coverage "
            "drops below baseline_execution_backed_covered / _fraction, or if "
            "any construct not already in `deficit_names` becomes uncovered "
            "(the deficit list grows). Regenerate with "
            "scripts/language_coverage.py --write-execution-deficit; growth is "
            "refused without --allow-deficit-growth so the claim is never "
            "walked down silently."),
        "evidence_mode": "runtime-execution",
        "surface_total": surface_total,
        "baseline_execution_backed_covered": len(covered),
        "baseline_execution_backed_fraction": round(execution_fraction, 4),
        "baseline_lexical_covered": lexical_covered,
        "baseline_lexical_fraction": (round(lexical_covered / surface_total, 4)
                                      if surface_total else 0.0),
        "deficit_total": len(deficit_names),
        "deficit_by_category": {category: sorted(names) for category, names
                                in sorted(uncovered_by_category.items())},
        "deficit_names": deficit_names,
        "spelled_but_unproven": sorted(spelled_but_unproven),
        "execution_only_names": sorted(execution_only_names),
    }


def load_execution_deficit(path):
    """Return the committed deficit baseline, or None when absent."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def evaluate_deficit_ratchet(current, baseline):
    """Compare the fresh deficit against the committed baseline.

    The ratchet is monotonic: execution-backed coverage may only hold or rise,
    and the named deficit set may only hold or shrink. Any construct that was
    not already a known deficit becoming uncovered — or the count/fraction
    regressing — fails the gate and names the offending constructs.
    """
    current_deficit = set(current["deficit_names"])
    if baseline is None:
        return {
            "has_baseline": False,
            "pass": True,
            "coverage_dropped": False,
            "deficit_grew": False,
            "new_deficit_names": [],
            "baseline_covered": current["baseline_execution_backed_covered"],
            "baseline_fraction": current["baseline_execution_backed_fraction"],
            "baseline_deficit_total": current["deficit_total"],
        }
    baseline_deficit = set(baseline.get("deficit_names", []))
    baseline_covered = int(baseline.get("baseline_execution_backed_covered", 0))
    baseline_fraction = float(
        baseline.get("baseline_execution_backed_fraction", 0.0))
    baseline_total = int(baseline.get("deficit_total", len(baseline_deficit)))
    new_deficit_names = sorted(current_deficit - baseline_deficit)
    coverage_dropped = (
        current["baseline_execution_backed_covered"] < baseline_covered
        or current["baseline_execution_backed_fraction"] < baseline_fraction)
    deficit_grew = bool(new_deficit_names) or len(current_deficit) > baseline_total
    return {
        "has_baseline": True,
        "pass": not coverage_dropped and not deficit_grew,
        "coverage_dropped": coverage_dropped,
        "deficit_grew": deficit_grew,
        "new_deficit_names": new_deficit_names,
        "baseline_covered": baseline_covered,
        "baseline_fraction": baseline_fraction,
        "baseline_deficit_total": baseline_total,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=os.path.join(REPO, "tests", "coverage",
                                                   "coverage_run.json"))
    ap.add_argument("--runtime-trace-dir", action="append", required=True,
                    help="directory containing execution TSV traces (repeatable)")
    ap.add_argument("--emit-runtime-event", action="store_true",
                    help="print ICC runtime_event JSON lines to stdout")
    ap.add_argument("--trace", default=None,
                    help="write fresh ICC runtime_event JSONL evidence to PATH")
    ap.add_argument("--policy", default=POLICY,
                    help="monotonic coverage policy (default: %(default)s)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="raise the policy floor for this run (cannot lower it)")
    ap.add_argument("--require-zero-high-risk", action="store_true",
                    help="also fail until every policy high-risk construct is covered")
    ap.add_argument("--execution-deficit", default=EXECUTION_DEFICIT,
                    help="monotonic execution-backed deficit ledger "
                         "(default: %(default)s)")
    ap.add_argument("--no-deficit-ratchet", action="store_true",
                    help="skip the deficit-growth ratchet (diagnostic runs only)")
    ap.add_argument("--write-execution-deficit", action="store_true",
                    help="rewrite the committed deficit ledger from this run; "
                         "refuses to grow the deficit without "
                         "--allow-deficit-growth")
    ap.add_argument("--allow-deficit-growth", action="store_true",
                    help="permit --write-execution-deficit to record a larger "
                         "deficit (only with an explicit, reviewed regression)")
    args = ap.parse_args()

    manifest = load_manifest()
    policy = load_policy(args.policy)
    # surface = builtins + special forms + prelude higher-order functions.
    surface = {}
    for b in manifest["builtins"]:
        surface[b["name"]] = b["category"]
    for s in manifest["special_forms"]:
        surface.setdefault(s["name"], s["category"])
    for p in manifest["prelude"]:
        surface.setdefault(p["name"], p["category"])
    # internal-only helpers (leading underscore) are not user-facing surface.
    surface = {k: v for k, v in surface.items() if not k.startswith("_")}

    import gen_generative_corpus as generative
    unknown_targets = sorted(generative.surface_probe_heads() - set(surface))
    if unknown_targets:
        raise RuntimeError(
            "deterministic surface probes name constructs absent from the manifest: %s"
            % ", ".join(unknown_targets)
        )

    ci_test_text, ci_test_paths = complete_ci_test_text()
    quantum_text, quantum_paths = quantum_test_text()
    engine_text = {
        "gen_generative_corpus.py": gen_generative_corpus_text(),
        "gen_ad_adversarial.py": gen_ad_adversarial_text(),
        "scripts/run_all_tests.sh (complete CI Scheme corpus)": ci_test_text,
        "tests/quantum/*.esk (quantum-enabled CI)": quantum_text,
    }
    heads_by_engine = {name: collect_heads(text)
                       for name, text in engine_text.items()}
    source_heads = set().union(*heads_by_engine.values())
    runtime_evidence = load_runtime_evidence(args.runtime_trace_dir)
    runtime_names = runtime_evidence["covered_names"]

    covered = {k: v for k, v in surface.items() if k in runtime_names}
    uncovered = {k: v for k, v in surface.items() if k not in runtime_names}

    # Permanent tripwire: certify nothing that lacks runtime evidence. This is
    # the guard that keeps lexical name-presence demoted to a diagnostic — if a
    # future change ever routes source_heads back into the credited set, this
    # raises instead of silently inflating the coverage number.
    verify_execution_backed_invariant(covered, runtime_names, surface,
                                      source_heads)

    # Lexical exposure is reported for the truth-gap, never gated. A surface
    # construct is "lexically covered" when its name merely appears as an
    # application head in the corpus/generator text; "spelled_but_unproven" are
    # the names present in that text yet lacking execution evidence.
    lexical_names = set(surface) & source_heads
    lexical_covered = len(lexical_names)
    spelled_but_unproven = sorted(lexical_names - set(covered))
    execution_only_names = sorted(set(covered) - lexical_names)

    total = len(surface)
    frac = len(covered) / total if total else 0.0
    policy_floor = float(policy["minimum_covered_fraction"])
    requested_floor = args.threshold if args.threshold is not None else policy_floor
    effective_floor = max(policy_floor, requested_floor)
    high_risk_categories = set(policy["high_risk_categories"])
    uncovered_high_risk = {
        name: category for name, category in uncovered.items()
        if category in high_risk_categories
    }
    floor_pass = (
        len(covered) >= int(policy["minimum_covered"])
        and frac >= effective_floor
    )
    high_risk_pass = not uncovered_high_risk

    # categorised uncovered, ranked by silent-wrong-risk then size
    RISK_ORDER = ["numeric", "tensor_ad", "geometry", "control_flow",
                  "consciousness", "higher_order", "list_pair", "vector",
                  "string_char", "hash", "predicate", "io_port", "binding_form",
                  "macro_syntax", "module", "memory_region", "misc_core",
                  "ffi_system", "misc"]
    by_cat = {}
    for k, v in uncovered.items():
        by_cat.setdefault(v, []).append(k)
    cov_by_cat = {}
    for k, v in covered.items():
        cov_by_cat.setdefault(v, []).append(k)

    ranked = sorted(by_cat.items(),
                    key=lambda kv: (RISK_ORDER.index(kv[0])
                                    if kv[0] in RISK_ORDER else 99,
                                    -len(kv[1])))

    # Execution-backed deficit ledger + monotonic ratchet. The deficit IS the
    # categorised uncovered set; the ratchet fails the gate if execution-backed
    # coverage regresses or the named deficit grows beyond the committed floor.
    deficit = build_execution_deficit(
        surface, covered, dict(ranked), frac, lexical_covered,
        spelled_but_unproven, execution_only_names)
    baseline_deficit = load_execution_deficit(args.execution_deficit)

    if args.write_execution_deficit:
        if (baseline_deficit is not None and not args.allow_deficit_growth):
            grew = evaluate_deficit_ratchet(deficit, baseline_deficit)
            if grew["coverage_dropped"] or grew["deficit_grew"]:
                print(
                    "\nREFUSED: --write-execution-deficit would grow the "
                    "committed deficit (new: %s). Fix coverage or pass "
                    "--allow-deficit-growth with a reviewed regression."
                    % (", ".join(grew["new_deficit_names"]) or "count regressed"),
                    file=sys.stderr)
                return 1
        os.makedirs(os.path.dirname(args.execution_deficit), exist_ok=True)
        with open(args.execution_deficit, "w", encoding="utf-8") as fh:
            json.dump(deficit, fh, indent=2)
            fh.write("\n")
        print("Wrote execution-backed deficit ledger: %s (deficit=%d)"
              % (args.execution_deficit, deficit["deficit_total"]))
        return 0

    ratchet = evaluate_deficit_ratchet(deficit, baseline_deficit)
    deficit_pass = args.no_deficit_ratchet or ratchet["pass"]

    out = {
        "surface_total": total,
        "covered": len(covered),
        "uncovered": len(uncovered),
        "covered_fraction": round(frac, 4),
        "coverage_policy": {
            "minimum_covered": int(policy["minimum_covered"]),
            "minimum_covered_fraction": policy_floor,
            "effective_threshold": effective_floor,
            "baseline_surface_total": int(policy["baseline_surface_total"]),
            "floor_pass": floor_pass,
            "high_risk_categories": sorted(high_risk_categories),
            "uncovered_high_risk": len(uncovered_high_risk),
            "high_risk_complete": high_risk_pass,
        },
        "covered_by_category": {k: len(v) for k, v in sorted(cov_by_cat.items())},
        "uncovered_by_category": {k: sorted(v) for k, v in ranked},
        "covered_names": sorted(covered),
        "evidence_mode": "runtime-execution",
        "execution_backed_covered": len(covered),
        "execution_backed_fraction": round(frac, 4),
        "lexical_covered": lexical_covered,
        "lexical_fraction": round(lexical_covered / total, 4) if total else 0.0,
        "spelled_but_unproven": spelled_but_unproven,
        "execution_only_names": execution_only_names,
        "execution_deficit": {
            "ledger": repo_relative_or_absolute(args.execution_deficit),
            "deficit_total": deficit["deficit_total"],
            "deficit_by_category": deficit["deficit_by_category"],
            "ratchet_pass": deficit_pass,
            "ratchet_enforced": not args.no_deficit_ratchet,
            "has_baseline": ratchet["has_baseline"],
            "coverage_dropped": ratchet["coverage_dropped"],
            "deficit_grew": ratchet["deficit_grew"],
            "new_deficit_names": ratchet["new_deficit_names"],
            "baseline_covered": ratchet["baseline_covered"],
            "baseline_fraction": ratchet["baseline_fraction"],
        },
        "runtime_trace_dirs": args.runtime_trace_dir,
        "runtime_trace_files": [repo_relative_or_absolute(path)
                                for path in runtime_evidence["trace_paths"]],
        "runtime_event_counts": runtime_evidence["event_counts"],
        "runtime_direct_call_names": sorted(runtime_evidence["direct_call_names"]),
        "runtime_vm_dispatch_names": sorted(runtime_evidence["vm_dispatch_names"]),
        "runtime_rejected_compile_time_names": sorted(
            runtime_evidence["rejected_compile_time_names"]),
        "runtime_spelling_matches": sorted(runtime_evidence["runtime_spelling_matches"]),
        "compile_time_spelling_matches": sorted(runtime_evidence["compile_time_matches"]),
        "source_exposed_names": sorted(set(surface) & source_heads),
        "source_exposed_only_names": sorted((set(surface) & source_heads) - set(covered)),
        "source_engines": list(engine_text),
        "source_exposed_by_engine": {
            name: len(set(surface) & engine_heads)
            for name, engine_heads in heads_by_engine.items()
        },
        "quantum_test_paths": quantum_paths,
        "complete_ci_test_paths": ci_test_paths,
        "source_exposed_by_quantum_tests": sorted(set(surface) &
                                                   heads_by_engine["tests/quantum/*.esk (quantum-enabled CI)"]),
    }
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")

    print("Language-surface coverage (runtime execution evidence)")
    print("  surface constructs : %d" % total)
    print("  EXECUTION-backed   : %d (%.1f%%)  [the only gated number]"
          % (len(covered), 100 * frac))
    print("  lexical exposure   : %d (%.1f%%)  [diagnostic only, zero credit]"
          % (lexical_covered, 100 * lexical_covered / total if total else 0.0))
    print("  spelled-but-unproven: %d (in corpus text, no execution evidence)"
          % len(spelled_but_unproven))
    print("  uncovered          : %d" % len(uncovered))
    print("  policy floor       : %d constructs, %.2f%% — %s"
          % (int(policy["minimum_covered"]), 100 * effective_floor,
             "PASS" if floor_pass else "FAIL"))
    if ratchet["has_baseline"]:
        print("  deficit ratchet    : baseline %d covered / %d deficit — %s%s"
              % (ratchet["baseline_covered"], ratchet["baseline_deficit_total"],
                 "PASS" if ratchet["pass"] else "FAIL",
                 "" if not ratchet["new_deficit_names"]
                 else " (new: %s)" % ", ".join(ratchet["new_deficit_names"][:8])))
    else:
        print("  deficit ratchet    : no committed baseline (bootstrapping)")
    print("  high-risk uncovered: %d — %s"
          % (len(uncovered_high_risk), "COMPLETE" if high_risk_pass else "OPEN"))
    print("  sidecar            : %s" % args.json)
    print("  trace files        : %d" % len(runtime_evidence["trace_paths"]))
    print("  runtime events     : %s" % ", ".join(
        "%s=%d" % item for item in sorted(runtime_evidence["event_counts"].items())))
    print("\nUncovered by category (highest silent-wrong risk first):")
    for cat, names in ranked:
        cov_n = len(cov_by_cat.get(cat, []))
        tot = cov_n + len(names)
        print("  %-14s %3d/%-3d uncovered  (e.g. %s)"
              % (cat, len(names), tot, ", ".join(sorted(names)[:6])))

    event_timestamp = time.time()
    events = [
        {
            "kind": "runtime_event",
            "event": "language_surface_coverage",
            "name": "language_surface_coverage",
            "value": "PASS" if floor_pass else "FAIL",
            "covered_fraction": round(frac, 4),
            "covered": len(covered),
            "surface_total": total,
            "threshold": effective_floor,
            "status": "PASSED" if floor_pass else "FAILED",
            "timestamp": event_timestamp,
            "confidence": 1.0,
        },
        {
            "kind": "runtime_event",
            "event": "language_surface_high_risk_complete",
            "name": "language_surface_high_risk_complete",
            "value": "PASS" if high_risk_pass else "FAIL",
            "uncovered_high_risk": len(uncovered_high_risk),
            "uncovered_by_category": {
                category: sum(1 for value in uncovered_high_risk.values()
                              if value == category)
                for category in sorted(high_risk_categories)
            },
            "status": "PASSED" if high_risk_pass else "FAILED",
            "timestamp": event_timestamp,
            "confidence": 1.0,
        },
        {
            # A1: the certification is execution-backed evidence, never
            # name-presence. This event carries the truth-gap (execution vs
            # lexical) and the monotonic deficit-ratchet verdict so the ICC
            # oracle can gate on proven execution and refuse silent regression.
            "kind": "runtime_event",
            "event": "execution_backed_language_coverage",
            "name": "execution_backed_language_coverage",
            "value": "PASS" if (floor_pass and deficit_pass) else "FAIL",
            "evidence_mode": "runtime-execution",
            "execution_backed_covered": len(covered),
            "execution_backed_fraction": round(frac, 4),
            "lexical_covered": lexical_covered,
            "lexical_fraction": round(lexical_covered / total, 4) if total else 0.0,
            "spelled_but_unproven": len(spelled_but_unproven),
            "deficit_total": deficit["deficit_total"],
            "deficit_grew": ratchet["deficit_grew"],
            "coverage_dropped": ratchet["coverage_dropped"],
            "new_deficit_names": ratchet["new_deficit_names"],
            "surface_total": total,
            "threshold": effective_floor,
            "status": "PASSED" if (floor_pass and deficit_pass) else "FAILED",
            "timestamp": event_timestamp,
            "confidence": 1.0,
        },
    ]
    if args.trace:
        write_runtime_events(args.trace, events)
        print("  ICC trace          : %s" % args.trace)
    if args.emit_runtime_event:
        for event in events:
            print(json.dumps(event, sort_keys=True))

    if not floor_pass:
        print("\nFAIL: coverage %d/%d (%.4f) is below policy floor "
              "%d constructs and %.4f"
              % (len(covered), total, frac, int(policy["minimum_covered"]),
                 effective_floor), file=sys.stderr)
        return 1
    if not deficit_pass:
        reasons = []
        if ratchet["coverage_dropped"]:
            reasons.append(
                "execution-backed coverage regressed below baseline %d/%.4f"
                % (ratchet["baseline_covered"], ratchet["baseline_fraction"]))
        if ratchet["deficit_grew"]:
            reasons.append(
                "deficit grew (new uncovered: %s)"
                % (", ".join(ratchet["new_deficit_names"]) or "count increased"))
        print("\nFAIL: execution-backed deficit ratchet — %s"
              % "; ".join(reasons), file=sys.stderr)
        return 1
    if args.require_zero_high_risk and not high_risk_pass:
        print("\nFAIL: %d high-risk constructs remain uncovered"
              % len(uncovered_high_risk), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
