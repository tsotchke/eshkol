#!/usr/bin/env python3
"""Check or render the reviewed example guides; never execute an example.

Two inputs drive the generated pages:

* ``docs/examples/catalogue.json`` -- the reviewed semantics of every tracked
  example: what it computes, the published result it reproduces and its
  references, its arithmetic scope, its checks and its limits.
* ``docs/examples/measurements.json`` -- what the release build measured when
  every example was run: JIT and AOT outcome, timing, the program's own verdict
  line and the computed values that matter. It is written by
  ``--ingest RECEIPTS`` from a receipts file produced by an external runner;
  this script reads receipts and logs but never runs a program.

A measurement is bound to the exact source bytes it measured. When an example
changes, its SHA-256 no longer matches and the check fails until the program is
re-run and the receipts are re-ingested.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


class CatalogueError(ValueError):
    pass


FAMILIES = ("ESHKOL_NS_EXAMPLES", "ESHKOL_IPM_EXAMPLES", "ESHKOL_AI_WITNESS_EXAMPLES")
FAMILY_PREFIX = {"ESHKOL_NS_EXAMPLES": "ns", "ESHKOL_IPM_EXAMPLES": "ipm", "ESHKOL_AI_WITNESS_EXAMPLES": "aiw"}
FIELDS = ("title", "purpose", "algorithm", "domain", "arithmetic", "validation", "limitations", "prerequisites", "literature_review")
AI_NAMES = {"mathematics_jacobian_counterexample", "mathematics_alphatensor_gf2", "mathematics_alphatensor_3x3_gf2", "mathematics_funsearch_cap_set"}
CATALOGUE_SCHEMA = "eshkol.example-catalogue.v2"
MEASUREMENTS = "docs/examples/measurements.json"
MEASUREMENTS_SCHEMA = "eshkol.example-measurements.v1"
# PASS: the program's own verdict line is "RESULT: ALL PASS" with "Failed: 0".
# RAN-OK: no self-verdict; exit 0 and no runtime-error line (the general runner's standard).
STATUSES = {"PASS", "RAN-OK", "FAIL", "TIMEOUT", "COMPILE-FAIL", "NOT-RUN"}
REFERENCE_STATES = {"checked", "source-comment", "repository", "citation unverified"}
KEY_OUTPUT_LIMIT = 16


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CatalogueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def inventory(root):
    result = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", "examples"], capture_output=True)
    if result.returncode:
        raise CatalogueError("Git example inventory failed; no outputs may be written")
    try:
        paths = sorted(p for p in result.stdout.decode("utf-8").split("\0") if p.endswith(".esk"))
    except UnicodeError as exc:
        raise CatalogueError("Git inventory is not UTF-8") from exc
    if not paths or len(set(paths)) != len(paths):
        raise CatalogueError("Git example inventory is empty or duplicated; refusing to write")
    for path in paths:
        if not re.fullmatch(r"examples/(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+\.esk", path):
            raise CatalogueError(f"unsupported example path: {path}")
        if not (root / path).is_file() or (root / path).is_symlink():
            raise CatalogueError(f"tracked example missing or symlinked: {path}")
    return paths


def registration_matrix(root, paths):
    text = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    groups = {}
    for family in FAMILIES:
        matches = list(re.finditer(rf"\bset\({family}\s+([^)]*)\)", text))
        if len(matches) != 1:
            raise CatalogueError(f"expected one nonempty registration list: {family}")
        tokens = re.sub(r"#[^\n]*", "", matches[0][1]).split()
        if not tokens or len(tokens) % 2:
            raise CatalogueError(f"empty or odd criterion/source list: {family}")
        prior_if = text.rfind("if(", 0, matches[0].start())
        if prior_if < 0 or not text[prior_if:].startswith("if(ESHKOL_BUILD_TESTS AND TARGET eshkol-run)"):
            raise CatalogueError(f"authored build/target condition changed: {family}")
        prefix = FAMILY_PREFIX[family]
        start = matches[0].end()
        loop_end = text.find("endforeach()", start)
        block = text[start:loop_end] if loop_end >= 0 else ""
        for mode in ("jit", "aot"):
            if block.count(f"add_test(NAME ${{_{prefix}_name}}_{mode}") != 1:
                raise CatalogueError(f"missing/duplicate authored {mode} registration for {family}")
        if f'"${{CMAKE_CURRENT_SOURCE_DIR}}/examples/${{_{prefix}_file}}.esk"' not in block or 'PASS_REGULAR_EXPRESSION "RESULT: ALL PASS"' not in block:
            raise CatalogueError(f"unrecognized source/verdict registration contract: {family}")
        if f'COMMAND $<TARGET_FILE:eshkol-run> -r "${{_{prefix}_src}}"' not in block or f"'${{_{prefix}_src}}' && '${{CMAKE_CURRENT_BINARY_DIR}}/${{_{prefix}_name}}_aot'" not in block:
            raise CatalogueError(f"native JIT/AOT command routing changed: {family}")
        if f'ENVIRONMENT "ESHKOL_PATH=${{CMAKE_CURRENT_SOURCE_DIR}}/lib"' not in block[block.find(f'set_tests_properties(${{_{prefix}_name}}_aot'):]:
            raise CatalogueError(f"AOT stdlib environment missing: {family}")
        if f'list(GET {family} ${{_{prefix}_i}} _{prefix}_name)' not in block or f'list(GET {family} ${{_{prefix}_j}} _{prefix}_file)' not in block:
            raise CatalogueError(f"criterion/source routing changed: {family}")
        rows, names = [], set()
        for criterion, stem in zip(tokens[::2], tokens[1::2]):
            source = f"examples/{stem}.esk"
            if not re.fullmatch(r"[A-Za-z0-9_]+", criterion) or criterion in names or source not in paths:
                raise CatalogueError(f"invalid, duplicate or untracked registration: {criterion}/{source}")
            names.add(criterion)
            rows.append({"criterion": criterion, "source": source, "modes": ["jit", "aot"]})
        groups[family] = {"criteria": len(rows), "distinct_sources": len({r["source"] for r in rows}), "ctest_entries": 2 * len(rows), "rows": rows}
    names = [r["criterion"] for group in groups.values() for r in group["rows"]]
    if len(set(names)) != len(names):
        raise CatalogueError("criterion is registered in competing families")
    return groups


def load_catalogue(root, path, paths, matrix):
    try:
        catalogue = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except (OSError, ValueError) as exc:
        raise CatalogueError(f"invalid catalogue: {exc}") from exc
    if not isinstance(catalogue, dict) or catalogue.get("schema") != CATALOGUE_SCHEMA or not isinstance(catalogue.get("entries"), list):
        raise CatalogueError("unsupported catalogue schema or entries")
    entries, seen = catalogue["entries"], set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("path") not in paths or entry["path"] in seen:
            raise CatalogueError("catalogue has unknown or duplicate example")
        seen.add(entry["path"])
        if any(not isinstance(entry.get(field), str) or not entry[field].strip() for field in FIELDS):
            raise CatalogueError(f"missing reviewed semantics: {entry['path']}")
        if entry.get("review_status") != "source-inspected" or entry.get("kind") not in {"mathematics", "example"}:
            raise CatalogueError("invalid review/kind status")
        if entry["kind"] != ("mathematics" if Path(entry["path"]).name.startswith("mathematics_") else "example"):
            raise CatalogueError("mathematics inventory classification drift")
        if "execution_evidence" in entry or "measured" in entry:
            raise CatalogueError("catalogue cannot carry execution evidence; outcomes come only from " + MEASUREMENTS)
        references = entry.get("references")
        if not isinstance(references, list) or (entry["kind"] == "mathematics" and not references):
            raise CatalogueError(f"mathematics entries need references: {entry['path']}")
        for ref in references:
            if (not isinstance(ref, dict) or not isinstance(ref.get("citation"), str) or not ref["citation"].strip()
                    or ref.get("verification") not in REFERENCE_STATES
                    or ("link" in ref and not (isinstance(ref["link"], str) and ref["link"].startswith(("https://", "http://"))))):
                raise CatalogueError(f"invalid reference record: {entry['path']}")
        patterns = entry.get("key_output", [])
        if not isinstance(patterns, list) or any(not isinstance(p, str) or not p for p in patterns):
            raise CatalogueError(f"key_output must be a list of patterns: {entry['path']}")
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise CatalogueError(f"invalid key_output pattern in {entry['path']}: {exc}") from exc
        source = root / entry["path"]
        if entry.get("source_sha256") != digest(source):
            raise CatalogueError(f"source changed; semantic review required before updating fingerprint: {entry['path']}")
        lines = source.read_text(encoding="utf-8").splitlines()
        anchors = entry.get("source_anchors")
        if not isinstance(anchors, list) or len(anchors) < 2:
            raise CatalogueError("implementation/validation source anchors required")
        for anchor in anchors:
            if (not isinstance(anchor, dict) or type(anchor.get("line")) is not int or not 1 <= anchor["line"] <= len(lines)
                    or anchor.get("text") != lines[anchor["line"] - 1]):
                raise CatalogueError(f"source anchor changed: {entry['path']}")
        expected = [{"family": family, "criterion": row["criterion"], "modes": row["modes"]}
                    for family, group in matrix.items() for row in group["rows"] if row["source"] == entry["path"]]
        if entry.get("registrations") != expected:
            raise CatalogueError(f"catalogue/registration matrix mismatch: {entry['path']}")
        if any(r["family"] == "ESHKOL_NS_EXAMPLES" for r in expected) and not isinstance(entry.get("paper_sections"), str):
            raise CatalogueError("Navier–Stokes source sections must be curated")
    if seen != set(paths):
        raise CatalogueError("catalogue omits tracked examples: " + ", ".join(sorted(set(paths) - seen)))
    runner = catalogue.get("runner_contract")
    if not isinstance(runner, dict) or runner.get("path") != "scripts/run_examples_tests.sh" or runner.get("source_sha256") != digest(root / runner["path"]):
        raise CatalogueError("general examples runner changed; review scope/skip policy")
    text = (root / runner["path"]).read_text()
    if (runner.get("discovery") != "examples/*.esk" or runner.get("mode") != "aot"
            or 'for test_file in examples/*.esk; do' not in text or '-L./"$BUILD_DIR" "$test_file" -o "$ESHKOL_TEST_BIN"' not in text):
        raise CatalogueError("invalid general runner discovery/mode")
    skip = text.split("example_should_skip() {", 1)[-1].split("print_empty_examples_summary()", 1)[0]
    quantum = re.search(r"\n\s*([a-z0-9_.|]+)\)\s*\n\s*if \[.*ESHKOL_QUANTUM_ENABLED", skip)
    excluded = re.search(r"\n\s*(selene_[^\n]+)\)\s*\n(?:\s*SKIP_REASON=[^\n]*\n)?\s*return 0", skip)
    if not quantum or not excluded or runner.get("quantum_sources") != ["examples/" + p for p in quantum[1].split("|")] or runner.get("excluded_patterns") != excluded[1].split("|") or runner.get("quantum_condition") != "ESHKOL_QUANTUM_ENABLED=ON":
        raise CatalogueError("general runner conditional/exclusion scope disagrees with review")
    return catalogue


def _validate_run(record, compiled, context="run", require_log=False):
    if not isinstance(record, dict) or record.get("status") not in STATUSES:
        raise CatalogueError(f"invalid {context} status")
    status = record["status"]
    exit_code = record.get("exit")
    compile_exit = record.get("compile_exit")
    result_line = record.get("result_line")
    if status in {"PASS", "RAN-OK"}:
        if exit_code != 0:
            raise CatalogueError(f"{context}: {status} requires exit 0")
        if status == "PASS":
            if not isinstance(result_line, str) or "RESULT: ALL PASS" not in result_line:
                raise CatalogueError(f"{context}: PASS requires a RESULT: ALL PASS receipt")
            if result_line.startswith("RESULT: ALL PASS") and "Failed:" in result_line and not re.search(r"Failed:\s*0(?:\D|$)", result_line):
                raise CatalogueError(f"{context}: PASS receipt reports failures")
            if record.get("failed") is not None and record.get("failed") != 0:
                raise CatalogueError(f"{context}: PASS receipt reports failed cases")
        elif result_line is not None:
            raise CatalogueError(f"{context}: RAN-OK must have no self-verdict line")
        if status == "RAN-OK" and any(record.get(k) is not None for k in ("passed", "failed")):
            raise CatalogueError(f"{context}: RAN-OK must not carry verdict counts")
        if compiled and compile_exit != 0:
            raise CatalogueError(f"{context}: successful AOT run requires compile exit 0")
    elif status == "COMPILE-FAIL":
        if not compiled or not isinstance(compile_exit, int) or compile_exit == 0:
            raise CatalogueError(f"{context}: COMPILE-FAIL requires a nonzero compile exit")
        if exit_code is not None:
            raise CatalogueError(f"{context}: COMPILE-FAIL must not have a run exit")
    elif status == "NOT-RUN":
        return True
    elif not isinstance(exit_code, int) or exit_code == 0:
        raise CatalogueError(f"{context}: {status} requires a nonzero exit")
    if require_log:
        log = record.get("log")
        compile_log = record.get("compile_log")
        if status == "COMPILE-FAIL":
            if not compile_log:
                raise CatalogueError(f"{context}: missing compile log")
        elif not log:
            raise CatalogueError(f"{context}: missing run log")
    numeric = ("wall_s", "cpu_s", "compile_s") if compiled else ("wall_s", "cpu_s")
    if any(record.get(k) is not None and not isinstance(record[k], (int, float)) for k in numeric):
        raise CatalogueError(f"{context}: invalid timing field")
    return True


def _run_record(record, compiled):
    try:
        _validate_run(record, compiled)
    except CatalogueError:
        return False
    if not isinstance(record, dict):
        return False
    numeric = ("wall_s", "cpu_s", "compile_s") if compiled else ("wall_s", "cpu_s")
    return all(record.get(k) is None or isinstance(record[k], (int, float)) for k in numeric)


def load_measurements(root, catalogue):
    """Return the validated measurement document, or None when none is committed."""
    path = root / MEASUREMENTS
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except (OSError, ValueError) as exc:
        raise CatalogueError(f"invalid measurements: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != MEASUREMENTS_SCHEMA or not isinstance(data.get("records"), dict):
        raise CatalogueError("unsupported measurements schema")
    if not re.fullmatch(r"[0-9a-f]{40}", str(data.get("release_sha", ""))):
        raise CatalogueError("measurements must name the full release SHA")
    records = data["records"]
    paths = {e["path"] for e in catalogue["entries"]}
    if set(records) != paths:
        raise CatalogueError("measurements do not cover exactly the catalogued examples")
    for entry in catalogue["entries"]:
        record = records[entry["path"]]
        if record.get("source_sha256") != entry["source_sha256"]:
            raise CatalogueError(f"measurement is for different source bytes; re-run and re-ingest: {entry['path']}")
        if not _run_record(record.get("jit"), False) or not _run_record(record.get("aot"), True):
            raise CatalogueError(f"invalid JIT/AOT measurement: {entry['path']}")
        if "note" in record and not isinstance(record["note"], str):
            raise CatalogueError(f"invalid measurement note: {entry['path']}")
        lines = record.get("key_output")
        if not isinstance(lines, list) or any(not isinstance(x, str) for x in lines):
            raise CatalogueError(f"invalid key_output lines: {entry['path']}")
        ran = any(record[m].get("status") in ("PASS", "RAN-OK") for m in ("jit", "aot"))
        for pattern in entry.get("key_output", []) if ran else []:
            if not any(re.search(pattern, line) for line in lines):
                raise CatalogueError(f"key_output pattern has no measured line in {entry['path']}: {pattern}")
        expected = {f"{r['criterion']}_{m}" for r in entry["registrations"] for m in r["modes"]} | {"vm_" + Path(entry["path"]).stem}
        ctest = record.get("ctest", [])
        if not isinstance(ctest, list) or {c.get("name") for c in ctest} - expected:
            raise CatalogueError(f"CTest measurement names an unregistered criterion: {entry['path']}")
    return data


def _log_lines(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _receipt_log_lines(base, run, context):
    """Read a required receipt log and reject absent or empty evidence."""
    name = run.get("compile_log") if run.get("status") == "COMPILE-FAIL" else run.get("log")
    if not name:
        raise CatalogueError(f"{context}: missing required receipt log")
    path = base / name
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise CatalogueError(f"{context}: cannot read receipt log {name}: {exc}") from exc
    if not any(line.strip() for line in lines):
        raise CatalogueError(f"{context}: receipt log {name} is empty")
    if run.get("status") == "PASS":
        expected = run.get("result_line")
        if not any(line.strip() == expected.strip() for line in lines):
            raise CatalogueError(f"{context}: declared PASS verdict is absent from receipt log {name}")
    return lines


def _select_key_lines(lines, patterns):
    # Loader banners, ICC trace echoes and GPU device banners are runtime
    # chatter, not program output (and device banners describe the host).
    # A one-line JSON receipt is kept whole (its verdict field comes last).
    lines = [line.rstrip()[:2000 if line.startswith("{") else 240] for line in lines if not line.startswith(("[REPL]", "ICC-EVENT ", "[GPU]"))]
    if not patterns:
        chosen = [line for line in lines if line.strip()][:5]
    else:
        chosen = []
        for line in lines:
            if any(re.search(p, line) for p in patterns) and line not in chosen:
                chosen.append(line)
    return chosen[:KEY_OUTPUT_LIMIT]


def _run_summary(run, compiled):
    if not isinstance(run, dict):
        return {"status": "NOT-RUN"}
    keys = ["status", "exit", "wall_s", "passed", "failed", "result_line"]
    out = {k: run.get(k) for k in keys}
    out["cpu_s"] = round((run.get("user_s") or 0) + (run.get("sys_s") or 0), 2) if run.get("user_s") is not None else None
    if compiled:
        out["compile_exit"] = run.get("compile_exit")
        out["compile_s"] = run.get("compile_wall_s")
    return out


def ingest(root, receipts_path, junit_paths=(), gates_path=None, not_run=None, notes=None):
    """Write docs/examples/measurements.json from an external runner's receipts.

    Receipts carry per-program JIT/AOT records and log paths relative to the
    receipts file; JUnit XML files carry CTest results; the optional gates JSON
    records whole-suite runs (the general examples runner, the WGSL artifact)."""
    import xml.etree.ElementTree as ET
    root = Path(root).resolve()
    paths = inventory(root)
    matrix = registration_matrix(root, paths)
    catalogue = load_catalogue(root, root / "docs/examples/catalogue.json", paths, matrix)
    receipts_path = Path(receipts_path).resolve()
    receipts = json.loads(receipts_path.read_text(encoding="utf-8"))
    if receipts.get("schema") != "eshkol.example-receipts.v1":
        raise CatalogueError("unsupported receipts schema")
    base = receipts_path.parent
    raw_records = receipts.get("records")
    if not isinstance(raw_records, list) or any(not isinstance(r, dict) or not isinstance(r.get("path"), str) for r in raw_records):
        raise CatalogueError("receipts records must be a list of path records")
    by_path = {}
    for item in raw_records:
        if item["path"] in by_path:
            raise CatalogueError(f"duplicate receipt record: {item['path']}")
        by_path[item["path"]] = item
    ctest = {}
    for junit in junit_paths:
        for case in ET.parse(junit).getroot().iter("testcase"):
            status = "Passed" if case.get("status") in (None, "run", "passed") and case.find("failure") is None and case.find("skipped") is None else "Failed"
            if case.find("skipped") is not None:
                status = "Not run"
            ctest[case.get("name")] = {"name": case.get("name"), "status": status, "seconds": round(float(case.get("time", 0)), 2)}
    not_run = not_run or {}
    records = {}
    for entry in sorted(catalogue["entries"], key=lambda e: e["path"]):
        rec = by_path.get(entry["path"])
        if rec is None:
            reason = not_run.get(entry["path"])
            if not reason:
                raise CatalogueError(f"receipts omit {entry['path']} and no not-run reason was given")
            records[entry["path"]] = {"source_sha256": entry["source_sha256"], "jit": {"status": "NOT-RUN"},
                                      "aot": {"status": "NOT-RUN"}, "not_run_reason": reason, "ctest": [], "key_output": []}
            continue
        if rec["source_sha256"] != entry["source_sha256"]:
            raise CatalogueError(f"receipt measured different source bytes: {entry['path']}")
        jit, aot = rec.get("jit"), rec.get("aot")
        _validate_run(jit, False, f"{entry['path']} JIT", require_log=True)
        _validate_run(aot, True, f"{entry['path']} AOT", require_log=True)
        logs = {}
        for label, run in (("JIT", jit), ("AOT", aot)):
            if run.get("status") != "NOT-RUN":
                logs[label] = _receipt_log_lines(base, run, f"{entry['path']} {label}")
        primary = jit if jit and jit.get("status") in ("PASS", "RAN-OK") else (aot if aot and aot.get("log") else jit)
        lines = logs.get("JIT", []) if primary is jit else logs.get("AOT", [])
        record = {"source_sha256": rec["source_sha256"], "jit": _run_summary(jit, False), "aot": _run_summary(aot, True),
                  "ctest": [ctest[n] for n in [f"{r['criterion']}_{m}" for r in entry["registrations"] for m in r["modes"]] + ["vm_" + Path(entry["path"]).stem] if n in ctest],
                  "key_output": _select_key_lines(lines, entry.get("key_output", [])),
                  "output_lines": len(lines)}
        bad = [run for run in (jit, aot) if run and run.get("status") not in ("PASS", "RAN-OK")]
        if bad:
            excerpt = []
            for run in bad:
                log = run.get("log") or run.get("compile_log")
                tail = _log_lines(base / log) if log else []
                excerpt += [l[:240] for l in tail if re.match(r"^(FAIL:|RESULT:|error|Error|ERROR)", l)][:8] or [l[:240] for l in tail[-6:]]
            record["failure_excerpt"] = excerpt[:16]
        if (notes or {}).get(entry["path"]):
            record["note"] = notes[entry["path"]]
        records[entry["path"]] = record
    data = {"schema": MEASUREMENTS_SCHEMA, "release_sha": receipts["release_sha"], "compiler_version": receipts.get("compiler_version"),
            "build_type": receipts.get("build_type"), "platform": receipts.get("platform"),
            "measured_on": receipts.get("started", "")[:10], "parallel_jobs": receipts.get("parallel_jobs"),
            "timing_note": receipts.get("timing_note"),
            "commands": {"jit": "ESHKOL_JIT_CACHE=0 ./build/eshkol-run -r <program>",
                         "aot": "./build/eshkol-run -L./build <program> -o <binary> && <binary>",
                         "ctest": "ctest --test-dir build -R '<criterion>_(jit|aot)$'"},
            "gates": json.loads(Path(gates_path).read_text(encoding="utf-8")) if gates_path else {},
            "records": records}
    destination = root / MEASUREMENTS
    destination.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    load_measurements(root, catalogue)
    return {"records": len(records), "measurements": MEASUREMENTS}


def anchor(entry):
    return Path(entry["path"]).stem.replace("_", "-")


def source_link(entry, line=1):
    return f"[{Path(entry['path']).name}](../{entry['path']}#L{line})"


def manual(entry):
    if entry["path"] == "examples/wgsl_artifact/generate.esk":
        return "python3 examples/wgsl_artifact/build_artifact.py"
    source = shlex.quote(entry["path"])
    output = shlex.quote(".scratch/example-manual/" + Path(entry["path"]).stem)
    return f"./build/eshkol-run -r {source}\n./build/eshkol-run -L./build {source} -o {output} && {output}"


def fmt_seconds(value):
    if value is None:
        return "n/a"
    return f"{value:.2f} s" if value < 10 else f"{value:.1f} s"


def run_phrase(run, compiled):
    status = run.get("status", "NOT-RUN")
    if status == "NOT-RUN":
        return "**NOT-RUN**"
    if status == "COMPILE-FAIL":
        return f"**COMPILE-FAIL** (compiler exit {run.get('compile_exit')})"
    parts = []
    if run.get("passed") is not None:
        parts.append(f"Passed: {run['passed']}, Failed: {run.get('failed')}")
    elif status != "PASS":
        parts.append(f"exit {run.get('exit')}")
    if compiled and run.get("compile_s") is not None:
        parts.append(f"compile {fmt_seconds(run['compile_s'])}, run {fmt_seconds(run.get('wall_s'))}")
    else:
        parts.append(fmt_seconds(run.get("wall_s")))
    return f"**{status}** ({'; '.join(parts)})"


def measured_text(entry, measurements):
    if measurements is None:
        return ["**Measured outcome:** no measurement is committed for this checkout; see `" + MEASUREMENTS + "`."]
    record = measurements["records"][entry["path"]]
    sha = measurements["release_sha"][:9]
    head = f"**Measured at `{sha}`** ({measurements.get('compiler_version')}, {measurements.get('build_type')}, {measurements.get('platform')}): "
    if record["jit"].get("status") == "NOT-RUN" and record["aot"].get("status") == "NOT-RUN":
        return [head + "**NOT-RUN**. " + record.get("not_run_reason", "")]
    text = head + f"native JIT {run_phrase(record['jit'], False)}; native AOT {run_phrase(record['aot'], True)}."
    if record.get("ctest"):
        text += " CTest: " + ", ".join(f"`{c['name']}` {c['status']} ({fmt_seconds(c.get('seconds'))})" for c in record["ctest"]) + "."
    verdict = record["jit"].get("result_line") or record["aot"].get("result_line")
    if verdict:
        text += f" Verdict line: `{verdict}`."
    rows = [text]
    if record.get("key_output"):
        label = "Computed values (verbatim program output):" if entry.get("key_output") else "First lines of output (verbatim):"
        rows.append(label + "\n\n```text\n" + "\n".join(line.rstrip() for line in record["key_output"]) + "\n```")
    if record.get("note"):
        rows.append("**Measurement note:** " + record["note"])
    if record.get("failure_excerpt"):
        rows.append("**Output at this SHA:**\n\n```text\n" + "\n".join(record["failure_excerpt"]) + "\n```")
    return rows


def reference_text(entry):
    if not entry.get("references"):
        return []
    items = []
    for ref in entry["references"]:
        line = "- " + ref["citation"]
        if ref.get("link"):
            line += f" <{ref['link']}>"
        note = {"checked": "", "source-comment": " (as cited in the program's source comments)",
                "repository": " (repository document)", "citation unverified": " (citation unverified)"}[ref["verification"]]
        items.append(line + note)
    return ["**References:**\n\n" + "\n".join(items)]


def entry_text(entry, measurements=None):
    fields = (('Purpose', 'purpose'), ('Implemented algorithm', 'algorithm'), ('Domain and parameters', 'domain'),
              ('Arithmetic', 'arithmetic'), ('Checks in the source', 'validation'), ('Limits', 'limitations'), ('Prerequisites', 'prerequisites'))
    rows = [f'<a id="{anchor(entry)}"></a>', f"## {entry['title']}", f"Source: {source_link(entry)}. SHA-256: `{entry['source_sha256']}`."]
    if entry["kind"] == "mathematics":
        rows += [f"**Published result:** {entry['literature_review']}"]
    rows += reference_text(entry)
    rows += [f"**{label}:** {entry[key]}" for label, key in fields]
    rows += measured_text(entry, measurements)
    refs = ", ".join(f"[line {a['line']}](../{entry['path']}#L{a['line']})" for a in entry['source_anchors'])
    rows += [f"Implementation/check anchors: {refs}.",
             "Manual commands from the repository root, after satisfying the prerequisites:", "```bash\n" + manual(entry) + "\n```"]
    return "\n\n".join(rows)


def marker(text, name, content):
    start, end = f"<!-- example-catalogue:{name}:start -->", f"<!-- example-catalogue:{name}:end -->"
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) >= text.index(end):
        raise CatalogueError(f"missing, duplicate or inverted generated markers: {name}")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return before + start + "\n" + content.rstrip() + "\n" + end + after


def status_pair(entry, measurements):
    if measurements is None:
        return "not measured"
    record = measurements["records"][entry["path"]]
    return f"{record['jit'].get('status')} / {record['aot'].get('status')}"


def outcome_summary(group, measurements, noun):
    """One sentence counting measured outcomes for a group of entries."""
    if measurements is None:
        return f"No measurement is committed for this checkout (`{MEASUREMENTS}` is absent), so these {noun} report source review only."
    sha = measurements["release_sha"][:9]
    records = [measurements["records"][e["path"]] for e in group]
    ok = {"PASS", "RAN-OK"}
    both = sum(r["jit"].get("status") in ok and r["aot"].get("status") in ok for r in records)
    verdict = sum(r["jit"].get("status") == "PASS" and r["aot"].get("status") == "PASS" for r in records)
    not_run = [e for e, r in zip(group, records) if r["jit"].get("status") == "NOT-RUN" and r["aot"].get("status") == "NOT-RUN"]
    failed = [e for e, r in zip(group, records) if e not in not_run and not (r["jit"].get("status") in ok and r["aot"].get("status") in ok)]
    text = (f"Every one of these {len(group)} {noun} was run at release SHA `{sha}` ({measurements.get('compiler_version')}, "
            f"{measurements.get('build_type')} build, {measurements.get('platform')}, measured {measurements.get('measured_on')}) in native JIT and native AOT. "
            f"**{both}** completed in both modes; **{verdict}** of those printed their own verdict `RESULT: ALL PASS` with `Failed: 0` in both modes")
    quiet = both - verdict
    text += (f", and the other {quiet} {'prints' if quiet == 1 else 'print'} diagnostics or a receipt without a `RESULT:` line (status `RAN-OK`: exit 0, no runtime-error line; the entry quotes the output)." if quiet else ".")
    if failed:
        text += " **Without a passing verdict at this SHA:** " + ", ".join(f"[{e['title']}](#{anchor(e)})" for e in failed) + " (its entry shows the output)."
    if not_run:
        text += " **Not run:** " + ", ".join(f"[{e['title']}](#{anchor(e)})" for e in not_run) + " (the entry states why)."
    if measurements.get("timing_note"):
        text += " " + measurements["timing_note"]
    return text


def render(root, catalogue, matrix, measurements=None):
    entries = sorted(catalogue['entries'], key=lambda e:e['path'])
    math = [e for e in entries if e['kind'] == 'mathematics']
    others = [e for e in entries if e['kind'] != 'mathematics']
    flat = [e for e in entries if len(Path(e['path']).parts) == 2]
    runner = catalogue['runner_contract']
    excluded = [e for e in flat if any(fnmatch.fnmatch(Path(e['path']).name, pattern) for pattern in runner['excluded_patterns'])]
    quantum = [e for e in flat if e['path'] in runner['quantum_sources']]
    sha = measurements["release_sha"][:9] if measurements else None
    front_matter = """---
kind: guide
status: current
owner-area: docs
since: v1.3.6
sources:
  - docs/examples/catalogue.json
  - docs/examples/measurements.json
  - CMakeLists.txt
  - scripts/run_examples_tests.sh
---"""
    gates = (measurements or {}).get("gates") or {}
    gate_text = []
    for name, gate in sorted(gates.items()):
        gate_text.append(f"`{gate.get('command', name)}`: {gate.get('summary', '')}")
    index = [front_matter, "# Example catalogue", "<!-- Generated by scripts/build_example_catalogue.py from reviewed docs/examples/catalogue.json and docs/examples/measurements.json. -->",
        f"This catalogue covers **{len(entries)} tracked programs**, including **{len(math)} mathematics programs**. Each entry gives the reviewed source behavior, the authored test registrations and the outcome measured when the program was run at the release SHA.",
        outcome_summary(entries, measurements, "programs"),
        "The [mathematics guide](MATHEMATICS_EXAMPLES.md) gives each mathematical program’s published result, references, algorithm, domain, arithmetic, checks, limits and measured values. The [Navier–Stokes guide](NAVIER_STOKES_EXAMPLES.md) and [AI mathematics guide](AI_MATHEMATICS_EXAMPLES.md) retain their explanatory narratives and references.",
        f"The general [examples runner](../scripts/run_examples_tests.sh) discovers **{len(flat)} flat programs** and compiles/runs them with native AOT. **{len(quantum)}** require `ESHKOL_QUANTUM_ENABLED=ON`; **{len(excluded)}** current paths match the existing proprietary/unreleased exclusion patterns. The **{len(entries)-len(flat)} nested artifact generator** is outside that glob. A successful process is distinct from an asserted mathematical verdict: `PASS` below means the program printed its own `RESULT: ALL PASS` with `Failed: 0`; `RAN-OK` means a program without a self-verdict exited 0 with no runtime-error line.",
        "The NS/IPM CMake lists declare native JIT/AOT tests under `ESHKOL_BUILD_TESTS AND TARGET eshkol-run`. The bytecode VM runs four of the cohomology programs under its own CTest entries; VM support for the other examples is not claimed here."]
    if gate_text:
        index.append("Whole-suite runs at the same SHA:\n\n" + "\n".join("- " + g for g in gate_text))
    index += ["For manual AOT commands below, create `.scratch/example-manual` once from the repository root.",
        f"To check documentation drift: `python3 scripts/build_example_catalogue.py --check`. After deliberate source/semantic review, update the catalogue, re-run the examples, ingest the receipts with `--ingest`, and run `--write`; the generator never updates fingerprints or executes examples, and a measurement whose source SHA-256 no longer matches fails the check.",
        "| Program | Reviewed purpose | Authored dedicated criteria | General AOT scope | Measured JIT / AOT |\n|---|---|---|---|---|"]
    table_rows=[]
    for entry in entries:
        target = 'MATHEMATICS_EXAMPLES.md' if entry['kind']=='mathematics' else 'EXAMPLES.md'
        title = f"[{entry['title']}]({target}#{anchor(entry)})"
        criteria = ', '.join(f"`{r['criterion']}` (JIT/AOT)" for r in entry['registrations']) or 'No NS/IPM criterion'
        scope = 'Nested artifact pipeline' if entry not in flat else 'Conditional quantum lane' if entry in quantum else 'Declared exclusion' if entry in excluded else 'Flat discovery'
        table_rows.append(f"| {source_link(entry)} | {title} | {criteria} | {scope} | {status_pair(entry, measurements)} |")
    index[-1] += '\n' + '\n'.join(table_rows)
    index += ['\n# Language and application examples', 'These descriptions report implemented checks or printed diagnostics; output prose is not treated as an assertion.']
    index += [entry_text(e, measurements) for e in others]
    math_text = [front_matter, '# Mathematics examples', '<!-- Generated by scripts/build_example_catalogue.py from reviewed docs/examples/catalogue.json and docs/examples/measurements.json. -->',
        f"**{len(math)} programs**, each reproducing a published or classical result by computation. Exact finite constructions, bilinear/polynomial identity certificates, finite formal expansions and numerical comparisons have different scopes, stated per entry under **Arithmetic** and **Limits**.",
        outcome_summary(math, measurements, "programs"),
        "Each entry gives: the published result being reproduced and its references; what the program computes and by which algorithm; the domain and parameters; the arithmetic and its scope; the checks the source asserts; the honest limits; and the measured outcome at the release SHA with the computed values that matter, quoted verbatim from the program's output. A reference marked *as cited in the program's source comments* is taken from the program header; *(citation unverified)* marks one whose bibliographic details could not be confirmed.",
        "[All examples and run scope](EXAMPLES.md) · [Navier–Stokes narrative](NAVIER_STOKES_EXAMPLES.md) · [AI witness narrative](AI_MATHEMATICS_EXAMPLES.md)",
        '| Authored CMake family | Programs | Criteria | Native JIT/AOT entries |\n|---|---:|---:|---:|']
    matrix_rows=[]
    for family, group in matrix.items():
        matrix_rows.append(f"| `{family}` | {group['distinct_sources']} | {group['criteria']} | {group['ctest_entries']} |")
    math_text[-1] += '\n' + '\n'.join(matrix_rows)
    no_criteria = [e for e in math if not e['registrations']]
    if no_criteria:
        names = ', '.join(f"`{Path(e['path']).stem}`" for e in no_criteria)
        verb = "have" if len(no_criteria) != 1 else "has"
        pronoun = "they are" if len(no_criteria) != 1 else "it is"
        no_criteria_note = f" {len(no_criteria)} program{'s' if len(no_criteria) != 1 else ''} ({names}) {verb} no dedicated criterion; {pronoun} run by the general examples runner."
    else:
        no_criteria_note = " Every mathematics program carries a dedicated criterion."
    math_text += ['\nRegistration is conditional on `ESHKOL_BUILD_TESTS AND TARGET eshkol-run`. Repeated criteria for one program are counted separately; the localization source has three NS criteria.' + no_criteria_note,
        '| Program | Measured JIT / AOT | Dedicated criteria |\n|---|---|---|' + ''.join(f"\n| [{e['title']}](#{anchor(e)}) | {status_pair(e, measurements)} | {', '.join('`'+r['criterion']+'`' for r in e['registrations']) or 'general runner only'} |" for e in math)]
    math_text += [entry_text(e, measurements) for e in math]
    ns_entries = [e for e in entries if any(r['family']=='ESHKOL_NS_EXAMPLES' for r in e['registrations'])]
    ns = (root / 'docs/NAVIER_STOKES_EXAMPLES.md').read_text()
    group = matrix['ESHKOL_NS_EXAMPLES']
    summary = f"The authored matrix contains **{group['distinct_sources']} programs, {group['criteria']} criteria and {group['ctest_entries']} CTest entries**, each with native JIT and AOT variants, under `ESHKOL_BUILD_TESTS AND TARGET eshkol-run`. Localization supplies three criteria. " + outcome_summary(ns_entries, measurements, "programs").replace("](#", "](MATHEMATICS_EXAMPLES.md#")
    table = ['| Program | Paper sections | Implemented calculation | CTest criteria | Measured JIT / AOT |','|---|---|---|---|---|']
    for entry in ns_entries:
        criteria=', '.join('`'+r['criterion']+'`' for r in entry['registrations'])
        table.append(f"| {source_link(entry)} | {entry['paper_sections']} | {entry['algorithm']} | {criteria} | [{status_pair(entry, measurements)}](MATHEMATICS_EXAMPLES.md#{anchor(entry)}) |")
    tail = (f"The criterion table above lists the `_jit` and `_aot` variants; the measured outcome of each, at `{sha}`, is in the table's last column and in each program's entry in the [mathematics catalogue](MATHEMATICS_EXAMPLES.md)."
            if measurements else "The criterion table above declares `_jit` and `_aot` variants; it is not a receipt that either ran.")
    commands = 'Prerequisites and scope are stated in the [complete mathematics catalogue](MATHEMATICS_EXAMPLES.md). Run from the repository root after building the compiler and stdlib.\n\n```bash\nmkdir -p .scratch/example-manual\n' + '\n\n'.join(manual(e) for e in ns_entries) + "\n\nctest --test-dir build --output-on-failure -R '^ns_'\n```\n\n" + tail
    ns = marker(marker(marker(ns,'ns-summary',summary),'ns-table','\n'.join(table)),'ns-commands',commands)
    ai=(root/'docs/AI_MATHEMATICS_EXAMPLES.md').read_text()
    ai_entries=[e for e in entries if Path(e['path']).stem in AI_NAMES]
    ai_block=(f"The catalogue contains **{len(ai_entries)} public witness programs** in this family. " + outcome_summary(ai_entries, measurements, "programs").replace("](#", "](MATHEMATICS_EXAMPLES.md#")
              + " Each complete description separates exact identity checks from numerical AD comparisons.\n\n"
              + '\n'.join(f"- [{e['title']}](MATHEMATICS_EXAMPLES.md#{anchor(e)}): {source_link(e)}; measured JIT / AOT: {status_pair(e, measurements)}." for e in ai_entries)
              + '\n\n```bash\nmkdir -p .scratch/example-manual\n'+'\n\n'.join(manual(e) for e in ai_entries)+'\n```')
    ai=marker(ai,'ai-inventory',ai_block)
    readme=(root/'examples/README.md').read_text()
    overview=[f"The reviewed catalogue covers **{len(entries)} programs**, including **{len(math)} mathematics programs**. Start with the [complete guide](../docs/EXAMPLES.md) or the [mathematics guide](../docs/MATHEMATICS_EXAMPLES.md) for published results, references, algorithms, domains, arithmetic, checks, limits, measured outcomes and per-program commands.",
        f"The general runner discovers **{len(flat)} flat sources** for native AOT, with **{len(quantum)} quantum-conditional programs** and the existing declared exclusions. The nested [WGSL generator](wgsl_artifact/README.md) follows its artifact pipeline. Dedicated mathematics registration contains NS **{matrix['ESHKOL_NS_EXAMPLES']['distinct_sources']}/{matrix['ESHKOL_NS_EXAMPLES']['criteria']}/{matrix['ESHKOL_NS_EXAMPLES']['ctest_entries']}**, IPM **{matrix['ESHKOL_IPM_EXAMPLES']['distinct_sources']}/{matrix['ESHKOL_IPM_EXAMPLES']['criteria']}/{matrix['ESHKOL_IPM_EXAMPLES']['ctest_entries']}** and AI-witness **{matrix['ESHKOL_AI_WITNESS_EXAMPLES']['distinct_sources']}/{matrix['ESHKOL_AI_WITNESS_EXAMPLES']['criteria']}/{matrix['ESHKOL_AI_WITNESS_EXAMPLES']['ctest_entries']}** programs/criteria/JIT-AOT entries, under the authored build condition.",
        outcome_summary(entries, measurements, "programs").replace("](#", "](../docs/EXAMPLES.md#") if measurements is None else
        outcome_summary(entries, measurements, "programs").replace("](#mathematics-", "](../docs/MATHEMATICS_EXAMPLES.md#mathematics-").replace("](#", "](../docs/EXAMPLES.md#"),
        "Each entry distinguishes asserted checks from printed diagnostics. Cross-host byte equality and backend dispatch need their own measured evidence.",
        "Run from the repository root with an already built compiler and stdlib, as described in the [Quickstart](../docs/QUICKSTART.md):\n\n```bash\nmkdir -p .scratch/example-manual\n./build/eshkol-run -r examples/hello.esk\n./build/eshkol-run -L./build examples/hello.esk -o .scratch/example-manual/hello\n.scratch/example-manual/hello\n```",
        "| Program | Source-backed purpose | Measured JIT / AOT | Detailed guide |\n|---|---|---|---|"]
    rows=[]
    for entry in entries:
        local=Path(entry['path']).relative_to('examples').as_posix()
        guide='MATHEMATICS_EXAMPLES.md' if entry['kind']=='mathematics' else 'EXAMPLES.md'
        rows.append(f"| [{local}]({local}#L1) | {entry['purpose']} | {status_pair(entry, measurements)} | [{entry['title']}](../docs/{guide}#{anchor(entry)}) |")
    overview[-1]+='\n'+'\n'.join(rows)
    readme=marker(readme,'examples-overview','\n\n'.join(overview))
    if measurements:
        claim = {'release_sha': measurements['release_sha'], 'source': MEASUREMENTS,
                 'summary': outcome_summary(entries, measurements, 'programs')}
    else:
        claim = 'No measurement committed; source inventory and authored registrations only.'
    data={'schema':'eshkol.example-test-matrix.v1','authored_condition':'ESHKOL_BUILD_TESTS AND TARGET eshkol-run','programs':len(entries),'mathematics_programs':len(math),'families':matrix,'general_runner':{'discovery':runner['discovery'],'mode':runner['mode'],'flat_programs':len(flat),'quantum_sources':runner['quantum_sources'],'excluded_patterns':runner['excluded_patterns'],'current_excluded_sources':[e['path'] for e in excluded],'nested_sources':[e['path'] for e in entries if e not in flat]},'execution_claim':claim}
    return {'docs/EXAMPLES.md':'\n\n'.join(index)+'\n','docs/MATHEMATICS_EXAMPLES.md':'\n\n'.join(math_text)+'\n','docs/NAVIER_STOKES_EXAMPLES.md':ns,'docs/AI_MATHEMATICS_EXAMPLES.md':ai,'docs/examples/test-matrix.json':json.dumps(data,indent=2,ensure_ascii=False)+'\n','examples/README.md':readme}


def verify_links(root, outputs):
    for relative, text in outputs.items():
        if not relative.endswith('.md'): continue
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            if target.startswith(('http:', 'https:', 'mailto:', '#')): continue
            destination, _, fragment=target.partition('#')
            path=(root/relative).parent/destination
            try: path.resolve().relative_to(root.resolve())
            except ValueError as exc: raise CatalogueError(f'link escapes repository: {target}') from exc
            key=path.resolve().relative_to(root.resolve()).as_posix()
            if key not in outputs and not path.exists():
                raise CatalogueError(f'broken local link in {relative}: {target}')
            if fragment.startswith('L') and fragment[1:].isdigit():
                if not path.is_file() or not 1 <= int(fragment[1:]) <= len(path.read_text().splitlines()):
                    raise CatalogueError(f'broken source line anchor: {target}')
            elif fragment and key in outputs and f'id="{fragment}"' not in outputs[key]:
                raise CatalogueError(f'broken generated section anchor: {target}')


def build(root, write=False):
    root=Path(root).resolve()
    paths=inventory(root)
    matrix=registration_matrix(root,paths)
    catalogue=load_catalogue(root,root/'docs/examples/catalogue.json',paths,matrix)
    measurements=load_measurements(root,catalogue)
    outputs=render(root,catalogue,matrix,measurements)
    verify_links(root,outputs)
    changed=[name for name,text in outputs.items() if not (root/name).is_file() or (root/name).read_text()!=text]
    if changed and not write:
        raise CatalogueError('generated documentation drift: '+', '.join(changed))
    # Every read, semantic/matrix check and link check finishes before writes.
    for name in changed:
        destination=root/name
        if destination.is_symlink(): raise CatalogueError('generated destination cannot be a symlink')
    for name in changed:
        destination=root/name; destination.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=destination.parent,prefix='.'+destination.name+'.',delete=False) as handle:
            handle.write(outputs[name]); temporary=Path(handle.name)
        temporary.replace(destination)
    return {'programs':len(paths),'mathematics':sum(e['kind']=='mathematics' for e in catalogue['entries']),'changed':changed}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--write',action='store_true')
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--ingest',type=Path,metavar='RECEIPTS',help='write docs/examples/measurements.json from an external runner receipts file')
    parser.add_argument('--junit',type=Path,action='append',default=[],help='CTest JUnit XML to attach (with --ingest)')
    parser.add_argument('--gates',type=Path,help='JSON describing whole-suite runs (with --ingest)')
    parser.add_argument('--not-run',type=Path,help='JSON {path: reason} for programs the receipts omit (with --ingest)')
    parser.add_argument('--notes',type=Path,help='JSON {path: note} attached to measurements (with --ingest)')
    args=parser.parse_args()
    try:
        if args.ingest:
            reasons=json.loads(args.not_run.read_text()) if args.not_run else None
            notes=json.loads(args.notes.read_text()) if args.notes else None
            result=ingest(args.root,args.ingest,args.junit,args.gates,reasons,notes)
        else:
            result=build(args.root,args.write)
    except (CatalogueError,OSError,UnicodeError,ValueError) as exc:
        print(f'example catalogue: FAIL: {exc}',file=sys.stderr);return 1
    print('example catalogue: PASS '+json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':
    raise SystemExit(main())
