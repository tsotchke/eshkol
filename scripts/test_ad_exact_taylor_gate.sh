#!/usr/bin/env bash
# Failure-injection selftest for the GUW oracle runner's execution evidence.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/ad-exact-taylor-selftest.XXXXXX")" || exit 2
trap 'rm -rf -- "$TMP"' EXIT
mkdir -p "$TMP/build"
touch "$TMP/build/CTestTestfile.cmake" "$TMP/build/eshkol-run"
chmod +x "$TMP/build/eshkol-run"
cat > "$TMP/fake-ctest" <<'PY'
#!/usr/bin/env python3
import os, sys, xml.etree.ElementTree as ET
args = sys.argv[1:]
path = args[args.index("--output-junit") + 1]
mode = os.environ.get("FAKE_CTEST_MODE", "pass")
if mode == "timeout":
    raise SystemExit(124)
if mode == "missing-report":
    raise SystemExit(0)
if mode == "malformed-xml":
    with open(path, "w", encoding="utf-8") as report:
        report.write("<testsuites><testsuite>")
    raise SystemExit(0)
root = ET.Element("testsuites")
suite = ET.SubElement(root, "testsuite", tests="2",
                      failures="1" if mode == "zero-exit-failed-xml" else "0",
                      errors="0", skipped="0")
names = ["ad_guw_order8_jit_smoke", "ad_guw_order8_aot_smoke"]
if mode == "zero-match":
    names = []
for name in names:
    case = ET.SubElement(suite, "testcase", name=name, classname="ad", time="0")
    if mode in ("fixture-fail", "zero-exit-failed-xml") and name == names[0]:
        ET.SubElement(case, "failure", message="injected intentional fixture failure")
ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
if mode == "emit-event-failure":
    os.chmod(os.path.join(os.environ["TRACE_DIR"], "ad_exact_taylor.jsonl"), 0o444)
raise SystemExit(1 if mode == "fixture-fail" else 0)
PY
chmod +x "$TMP/fake-ctest"

run_case() {
  mode="$1" expected="$2"
  dir="$TMP/traces-$mode"
  if FAKE_CTEST_MODE="$mode" BUILD_DIR="$TMP/build" CTEST="$TMP/fake-ctest" \
       TRACE_DIR="$dir" scripts/run_ad_exact_taylor_gate.sh >/dev/null 2>&1; then
    rc=0
  else
    rc=$?
  fi
  if [ "$expected" = "NO_EVENT" ]; then
    python3 - "$dir/ad_exact_taylor.jsonl" "$rc" <<'PY'
import json, sys
path, rc = sys.argv[1], int(sys.argv[2])
assert rc != 0, "gate returned success when ICC event writing failed"
try:
    event = json.loads(open(path, encoding="utf-8").read())
except (OSError, json.JSONDecodeError):
    event = None
assert event is None or event.get("value") != "PASS", event
PY
    return
  fi
  python3 - "$dir/ad_exact_taylor.jsonl" "$expected" "$rc" <<'PY'
import json, sys
event = json.loads(open(sys.argv[1], encoding="utf-8").read())
expected, rc = sys.argv[2], int(sys.argv[3])
assert event["value"] == expected, (event, expected)
assert (rc == 0) == (expected == "PASS"), (rc, expected)
PY
}

run_case pass PASS
run_case zero-match FAIL
run_case fixture-fail FAIL
run_case zero-exit-failed-xml FAIL
run_case malformed-xml FAIL
run_case missing-report FAIL
run_case timeout FAIL
run_case emit-event-failure NO_EVENT

mkdir -p "$TMP/not-a-directory"
touch "$TMP/not-a-directory/file"
if BUILD_DIR="$TMP/build" CTEST="$TMP/fake-ctest" TRACE_DIR="$TMP/not-a-directory/file/child" \
     scripts/run_ad_exact_taylor_gate.sh >/dev/null 2>&1; then
  echo "gate returned success when trace directory creation failed" >&2
  exit 1
else
  rc=$?
fi
[ "$rc" -eq 2 ]
echo "PASS: AD exact Taylor gate rejects zero-match, bad reports, timeout, and trace-write failures"
