#!/usr/bin/env bash
#
# nLab conformance gate (v1.4.0-connection).
#
# Runs, from an existing CMake build, the six CTest entries of the nLab
# conformance suite (tests/nlab/nlab_{topos,space,ring}_test.esk, native JIT and
# AOT) and checks their claim lines against the manifest tests/nlab/claims.tsv:
#   - every claim marked `computed` printed "PASS: [page#k]" exactly once in
#     both the JIT and the AOT run;
#   - no claim printed FAIL, and every claim line is in the manifest;
#   - the coverage line of docs/reports/NLAB_CONFORMANCE.md states the
#     manifest's numbers (pages, pages with computed claims, computed and
#     planned claims).
#
# It writes one event to $TRACE_DIR/v14_nlab_conformance.jsonl:
#   {"kind": "v1_4_connection", "name": "v14_nlab_conformance", "value": PASS|FAIL}
# with the coverage numbers in the snippet. Absence of a test is a FAIL.
#
#   BUILD_DIR  (env, default "build")   the CMake build directory
#   TRACE_DIR  (env, default scripts/icc_traces)
#   JOBS       (env, default 2)          parallel CTest jobs

set -u
set -o pipefail
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) BUILD_DIR_PATH="$BUILD_DIR" ;;
    *)  BUILD_DIR_PATH="$REPO_ROOT/$BUILD_DIR" ;;
esac
if [ ! -x "$BUILD_DIR_PATH/eshkol-run" ]; then
    echo "run_nlab_conformance.sh: $BUILD_DIR_PATH/eshkol-run not found; build first" >&2
    exit 2
fi
BUILD_DIR_PATH="$(cd "$BUILD_DIR_PATH" && pwd -P)" || exit 2

TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/v14_nlab_conformance.jsonl"

WORK="$BUILD_DIR_PATH/test-work/nlab-conformance"
mkdir -p "$WORK"
export ESHKOL_PATH="$REPO_ROOT/lib${ESHKOL_PATH:+:$ESHKOL_PATH}"

TESTS='^(nlab_(topos|space|ring)_(jit|aot))$'
EXPECTED=6

ctest_log="$WORK/ctest.log"
(cd "$BUILD_DIR_PATH" && ctest -R "$TESTS" -V -j "${JOBS:-2}") > "$ctest_log" 2>&1
ctest_status=$?

python3 - "$ctest_log" "$REPO_ROOT/tests/nlab/claims.tsv" "$REPO_ROOT/docs/reports/NLAB_CONFORMANCE.md" \
          "$TRACE_FILE" "$ctest_status" "$EXPECTED" <<'PY'
import json, os, re, sys
log, manifest, report, trace, ctest_status, expected = sys.argv[1:7]
ctest_status, expected = int(ctest_status), int(expected)
text = open(log, encoding="utf-8", errors="replace").read()

names = dict(re.findall(r'^\s*Start\s+(\d+):\s+(\S+)', text, re.M))
ran = re.findall(r'^\s*\d+/\d+ Test +#\d+: (\S+) \.+\s*(\S+)', text, re.M)
passed_tests = sorted(n for n, r in ran if r == "Passed")
claims = {"jit": {}, "aot": {}}
for num, verdict, page, k in re.findall(r'^(\d+): (PASS|FAIL): \[([^\]#]+)#(\d+)\]', text, re.M):
    name = names.get(num, "")
    engine = "jit" if name.endswith("_jit") else "aot" if name.endswith("_aot") else None
    if engine is None:
        continue
    claims[engine].setdefault((page, int(k)), []).append(verdict)

computed, planned = set(), set()
for line in open(manifest, encoding="utf-8"):
    if line.startswith("#") or not line.strip():
        continue
    page, k, status = line.rstrip("\n").split("\t")[:3]
    (computed if status == "computed" else planned).add((page, int(k)))

problems = []
for engine in ("jit", "aot"):
    got = claims[engine]
    for c in sorted(computed):
        v = got.get(c)
        if v != ["PASS"]:
            problems.append(f"{engine}: {c[0]}#{c[1]} -> {v}")
    for c in sorted(set(got) - computed):
        problems.append(f"{engine}: {c[0]}#{c[1]} printed but not in claims.tsv")

pages = {p for p, _ in computed | planned}
cpages = {p for p, _ in computed}
coverage = f"pages={len(pages)} computed-pages={len(cpages)} computed={len(computed)} planned={len(planned)}"
stated = re.search(r'<!-- nlab-coverage: ([^>]*?) -->', open(report, encoding="utf-8").read())
if not stated or stated.group(1).strip() != coverage:
    problems.append(f"report coverage line {stated.group(1) if stated else None!r} != {coverage!r}")
if ctest_status != 0 or len(ran) != expected or len(passed_tests) != expected:
    problems.append(f"ctest {len(passed_tests)}/{expected} passed (ran {len(ran)}, exit {ctest_status})")

jit_ok = sum(1 for c in computed if claims["jit"].get(c) == ["PASS"])
aot_ok = sum(1 for c in computed if claims["aot"].get(c) == ["PASS"])
summary = (f"ctest {len(passed_tests)}/{expected}; claims PASS jit {jit_ok}/{len(computed)}, aot {aot_ok}/{len(computed)}; "
           f"coverage {coverage}")
value = "PASS" if not problems else "FAIL"
snippet = summary if not problems else summary + "; " + "; ".join(problems[:20])
tmp = trace + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(json.dumps({"kind": "v1_4_connection", "name": "v14_nlab_conformance", "value": value,
                        "snippet": snippet[:2000], "confidence": 0.95}, ensure_ascii=False) + "\n")
os.replace(tmp, trace)
print(f"nLab conformance gate: {value} ({snippet})")
sys.exit(0 if value == "PASS" else 1)
PY
status=$?
if [ "$status" -ne 0 ]; then
    tail -n 40 "$ctest_log"
fi
exit "$status"
