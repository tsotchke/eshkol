#!/usr/bin/env python3
"""language_coverage.py — dynamic language-surface coverage for the exposure engines.

Runs the generative exposure engines plus the opt-in quantum test corpus,
collects the set of BUILTINS and
SPECIAL FORMS their generated (and executed) programs actually contain, diffs
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
  python3 scripts/language_coverage.py [--json OUT] [--emit-runtime-event]
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

# Scheme test roots exercised by scripts/run_all_tests.sh.  Keep this list in
# the same order as that suite's TEST_SCRIPTS array.  Source-scanning here is
# legitimate coverage evidence because those files are self-checking and the
# complete CI suite executes the corresponding directories; unlike a blanket
# tests/**/*.esk glob this deliberately excludes filed divergences, minimized
# crashers, and other diagnostic artifacts that are not run as gates.
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
    "tests/benchmark/*.esk", "tests/benchmarks/*.esk", "tests/migration/*.esk",
    "tests/codegen/**/*.esk", "tests/numeric/*.esk", "tests/sicp/**/*.esk",
    "tests/v1_2_edge_cases/*.esk",
)

# Deterministic VM extension probes run by scripts/run_vm_surface_tests.sh.
# The broader tests/vm directory also contains host-side-effect probes (signals,
# servers, process replacement), so list the pure numeric/stateful subset
# explicitly instead of claiming unexecuted coverage.
CI_VM_SURFACE_TESTS = (
    "tests/vm/geometric_surface_regression.esk",
    "tests/vm/geometric_fallback_numeric_regression.esk",
    "tests/vm/riemannian_adam_state_regression.esk",
    "tests/vm/kb_factor_graph_extensions_regression.esk",
    "tests/vm/workspace_introspection_regression.esk",
    "tests/vm/ad_tape_lowlevel_regression.esk",
    "tests/vm/vm_kb_tensor_test.esk",
    "tests/vm/numeric_alias_surface_regression.esk",
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


def collect_heads(text):
    """Return the set of symbols that appear as an application/operator head, plus
    the special forms introduced by reader macros (', `, , ,@) and #( vectors."""
    heads = set()
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

    These tests require Moonlab and therefore run only in the separate opt-in
    macOS lane. Their source still participates in this manifest calculation:
    every file is self-checking and is executed by that lane, matching the
    source-scan contract used for the deterministic generative engines.
    """
    paths = sorted(glob.glob(os.path.join(REPO, "tests", "quantum", "*.esk")))
    if not paths:
        raise RuntimeError("no quantum coverage tests found")
    parts = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            parts.append(fh.read())
    return "\n".join(parts), [os.path.relpath(path, REPO) for path in paths]


def complete_ci_test_text():
    """Return the committed Scheme corpus exercised by run_all_tests.sh.

    Paths are de-duplicated because a recursive and a direct glob may overlap.
    Empty roots are tolerated across feature-gated checkouts, but the complete
    corpus itself must be non-empty so a broken path list cannot manufacture a
    plausible zero-coverage result.
    """
    paths = set()
    for pattern in CI_TEST_GLOBS:
        paths.update(glob.glob(os.path.join(REPO, pattern), recursive=True))
    paths.update(os.path.join(REPO, path) for path in CI_VM_SURFACE_TESTS)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=os.path.join(REPO, "tests", "coverage",
                                                   "coverage_run.json"))
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
    heads = set().union(*heads_by_engine.values())
    # Parser-lowered promise helpers are genuinely executed even though the
    # source corpus contains their public syntax rather than the internal call
    # heads.  Record the lowering explicitly so the manifest measures runtime
    # exposure rather than merely lexical spelling.
    if "delay" in heads:
        heads.add("%make-lazy-promise")
    if "delay-force" in heads:
        heads.add("%make-lazy-promise-force")

    covered = {k: v for k, v in surface.items() if k in heads}
    uncovered = {k: v for k, v in surface.items() if k not in heads}

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
        "engines": list(engine_text),
        "covered_by_engine": {
            name: len(set(surface) & engine_heads)
            for name, engine_heads in heads_by_engine.items()
        },
        "quantum_test_paths": quantum_paths,
        "complete_ci_test_paths": ci_test_paths,
        "exercised_by_quantum_tests": sorted(set(surface) &
                                              heads_by_engine["tests/quantum/*.esk (quantum-enabled CI)"]),
    }
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")

    print("Language-surface coverage (exposure engines)")
    print("  surface constructs : %d" % total)
    print("  covered            : %d (%.1f%%)" % (len(covered), 100 * frac))
    print("  uncovered          : %d" % len(uncovered))
    print("  policy floor       : %d constructs, %.2f%% — %s"
          % (int(policy["minimum_covered"]), 100 * effective_floor,
             "PASS" if floor_pass else "FAIL"))
    print("  high-risk uncovered: %d — %s"
          % (len(uncovered_high_risk), "COMPLETE" if high_risk_pass else "OPEN"))
    print("  sidecar            : %s" % args.json)
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
    if args.require_zero_high_risk and not high_risk_pass:
        print("\nFAIL: %d high-risk constructs remain uncovered"
              % len(uncovered_high_risk), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
