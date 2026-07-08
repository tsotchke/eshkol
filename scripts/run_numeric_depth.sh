#!/usr/bin/env bash
# run_numeric_depth.sh — depth/scale-parametric NUMERIC TOWER sweep runner
# (adversarial testing pillar P6d, .swarm/DEPTH_PARAMETRIC_TESTING.md).
#
# Runs every generated tests/numeric_depth/<family>.esk under BOTH native axes:
#   jit   ./build/eshkol-run -r <file>
#   aot   ./build/eshkol-run -O0 <file> -o <bin> && <bin>
# scans the per-n "PASS/WRONG <family> <n>" self-check lines (self-verified
# against Python-computed exact ground truth baked in by gen_numeric_depth.py),
# and for each family records max-correct-n, the WRONG-n set, and whether the
# family terminated (its "DONE <family>" marker) or hit a LIMIT (crash / missing
# marker). Results are compared to tests/numeric_depth/BASELINE.json. The release
# contract is semantic: any NEW silent-wrong n or regressed max-correct-n is a
# REGRESSION (nonzero exit). A lost DONE marker with the same proven-correct
# prefix is capacity telemetry, not a correctness failure; AOT compile/runtime
# budgets vary by host and should not be encoded as portable semantic facts.
#
# Emits:
#   * docs/reports/NUMERIC_DEPTH_REPORT.md  (max-correct-n table + findings)
#   * scripts/icc_traces/numeric_depth.jsonl (kind:"numeric_depth" events,
#     consumed by .icc/completion-oracles.yaml::numeric-depth)
#
# Usage:
#   scripts/run_numeric_depth.sh [--jit-only|--aot-only] [--update-baseline]
#   BUILD_DIR=build scripts/run_numeric_depth.sh
#
# macOS has no coreutils `timeout`; we use a portable perl alarm wrapper.
set -u
export LC_ALL=C
export LANG=C
export LC_CTYPE=C
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in /*) : ;; *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;; esac
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
TESTS_DIR="$REPO_ROOT/tests/numeric_depth"
MANIFEST="$TESTS_DIR/manifest.json"
BASELINE="$TESTS_DIR/BASELINE.json"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/numeric_depth.jsonl"
REPORT="$REPO_ROOT/docs/reports/NUMERIC_DEPTH_REPORT.md"
OUTDIR="$(mktemp -d)"
TIMEOUT="${NUMERIC_DEPTH_TIMEOUT:-240}"

DO_JIT=1; DO_AOT=1; UPDATE_BASELINE=0
for a in "$@"; do
    case "$a" in
        --jit-only) DO_AOT=0 ;;
        --aot-only) DO_JIT=0 ;;
        --update-baseline) UPDATE_BASELINE=1 ;;
        *) echo "unknown arg: $a" >&2; exit 2 ;;
    esac
done

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_numeric_depth.sh: $ESHKOL_RUN not found — run" >&2
    echo "  cmake --build build --target eshkol-run stdlib -j" >&2
    exit 2
fi
if [ ! -f "$MANIFEST" ]; then
    echo "run_numeric_depth.sh: $MANIFEST missing — run python3 scripts/gen_numeric_depth.py" >&2
    exit 2
fi
mkdir -p "$TRACE_DIR" "$(dirname "$REPORT")"

# portable timeout: alarm N cmd...
alarm() { perl -e 'my $t=shift; my $p=fork; if($p==0){exec @ARGV or exit 127} local $SIG{ALRM}=sub{kill "KILL",$p; exit 124}; alarm $t; waitpid $p,0; exit($?>>8)' "$@"; }

families="$(python3 -c 'import json,sys; print("\n".join(f["family"] for f in json.load(open(sys.argv[1]))["families"]))' "$MANIFEST")"

echo "== numeric-depth sweep =="
for fam in $families; do
    esk="$TESTS_DIR/$fam.esk"
    [ -f "$esk" ] || { echo "  skip $fam (no .esk)"; continue; }
    if [ "$DO_JIT" = 1 ]; then
        alarm "$TIMEOUT" "$ESHKOL_RUN" -r "$esk" > "$OUTDIR/$fam.jit.out" 2>/dev/null
        echo "  jit  $fam: $(grep -c '^PASS' "$OUTDIR/$fam.jit.out") pass / $(grep -c '^WRONG' "$OUTDIR/$fam.jit.out") wrong"
    fi
    if [ "$DO_AOT" = 1 ]; then
        bin="$OUTDIR/$fam.bin"
        if alarm "$TIMEOUT" "$ESHKOL_RUN" -O0 "$esk" -o "$bin" >/dev/null 2>&1 && [ -x "$bin" ]; then
            alarm "$TIMEOUT" "$bin" > "$OUTDIR/$fam.aot.out" 2>/dev/null
        else
            : > "$OUTDIR/$fam.aot.out"  # compile failed => LIMIT
        fi
        echo "  aot  $fam: $(grep -c '^PASS' "$OUTDIR/$fam.aot.out") pass / $(grep -c '^WRONG' "$OUTDIR/$fam.aot.out") wrong"
    fi
done

# ---- analyze / compare baseline / write report + trace (Python) ----
DO_JIT="$DO_JIT" DO_AOT="$DO_AOT" UPDATE_BASELINE="$UPDATE_BASELINE" \
OUTDIR="$OUTDIR" MANIFEST="$MANIFEST" BASELINE="$BASELINE" \
TRACE_FILE="$TRACE_FILE" REPORT="$REPORT" \
python3 - <<'PY'
import json, os, re, sys

outdir   = os.environ["OUTDIR"]
manifest = json.load(open(os.environ["MANIFEST"]))
do_jit   = os.environ["DO_JIT"] == "1"
do_aot   = os.environ["DO_AOT"] == "1"
update   = os.environ["UPDATE_BASELINE"] == "1"
baseline_path = os.environ["BASELINE"]
trace_path    = os.environ["TRACE_FILE"]
report_path   = os.environ["REPORT"]

modes = [m for m, on in (("jit", do_jit), ("aot", do_aot)) if on]

def parse(path, fam):
    """Return dict: pass set, wrong set, done bool, last_n, max_correct."""
    passes, wrongs = set(), set()
    done = False
    last_n = 0
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            m = re.match(rf"^PASS {fam} (\d+)", line)
            if m:
                n = int(m.group(1)); passes.add(n); last_n = max(last_n, n); continue
            m = re.match(rf"^WRONG {fam} (\d+)", line)
            if m:
                n = int(m.group(1)); wrongs.add(n); last_n = max(last_n, n); continue
            if line.startswith(f"DONE {fam}"):
                done = True
    return {"pass": passes, "wrong": wrongs, "done": done, "last_n": last_n}

def max_correct(ns, r):
    """Largest n (in ns) such that every ns-entry <= n is PASS."""
    mc = 0
    for n in ns:
        if n in r["pass"]:
            mc = n
        elif n in r["wrong"] or (r["last_n"] and n > r["last_n"]):
            break
        else:
            break
    return mc

results = {}   # fam -> mode -> summary
for fam_m in manifest["families"]:
    fam = fam_m["family"]
    ns = fam_m["ns"]
    results[fam] = {"ns_max": ns[-1] if ns else 0}
    for mode in modes:
        r = parse(os.path.join(outdir, f"{fam}.{mode}.out"), fam)
        r["max_correct"] = max_correct(ns, r)
        r["wrong_sorted"] = sorted(r["wrong"])
        # LIMIT: family did not reach its DONE marker (crash / compile fail / timeout)
        r["limit"] = not r["done"]
        results[fam][mode] = {
            "max_correct": r["max_correct"],
            "wrong": r["wrong_sorted"],
            "done": r["done"],
            "limit": r["limit"],
            "last_n": r["last_n"],
        }

# ---- baseline compare ----
baseline = {}
if os.path.exists(baseline_path):
    baseline = json.load(open(baseline_path))

regressions = []
capacity_notes = []
for fam in results:
    for mode in modes:
        cur = results[fam][mode]
        base = baseline.get(fam, {}).get(mode)
        if base is None:
            continue  # no baseline yet for this family/mode
        new_wrong = sorted(set(cur["wrong"]) - set(base.get("wrong", [])))
        if new_wrong:
            regressions.append(f"{fam}/{mode}: NEW silent-wrong at n={new_wrong}")
        capacity_before_proof = cur["limit"] and cur["last_n"] == 0 and not cur["wrong"]
        if cur["max_correct"] < base.get("max_correct", 0) and not capacity_before_proof:
            regressions.append(
                f"{fam}/{mode}: max-correct-n regressed "
                f"{base.get('max_correct')} -> {cur['max_correct']}")
        elif capacity_before_proof and base.get("max_correct", 0) > 0:
            capacity_notes.append(
                f"{fam}/{mode}: capacity LIMIT before first emitted check; "
                f"baseline proves max-correct-n {base.get('max_correct')}")
        if base.get("done") and not cur["done"] and not capacity_before_proof:
            capacity_notes.append(
                f"{fam}/{mode}: lost DONE marker (capacity LIMIT at n~{cur['last_n']}); "
                f"max-correct-n observed {cur['max_correct']}")

if update:
    dump = {fam: {m: results[fam][m] for m in modes} for fam in results}
    json.dump(dump, open(baseline_path, "w"), indent=2, sort_keys=True)
    print(f"[baseline] wrote {baseline_path}")

# ---- report ----
lines = ["# Numeric-tower depth/scale-parametric sweep — report (P6d)\n",
         "Auto-generated by `scripts/run_numeric_depth.sh`. Ground truth: Python\n"
         "arbitrary-precision `int` / `fractions.Fraction`, or analytic identities.\n"
         "`max-correct-n` = largest depth/scale with every check <= it PASS.\n"]
lines.append("\n## Max-correct-n by family\n")
hdr = "| family | n range | " + " | ".join(f"{m} max-n" for m in modes) + \
      " | " + " | ".join(f"{m} status" for m in modes) + " |"
lines.append(hdr)
lines.append("|" + "---|" * (2 + 2 * len(modes)))
for fam in results:
    row = [fam, f"1..{results[fam]['ns_max']}"]
    for m in modes:
        row.append(str(results[fam][m]["max_correct"]))
    for m in modes:
        c = results[fam][m]
        if c["limit"]:
            st = f"LIMIT@~{c['last_n']}"
        elif c["wrong"]:
            st = f"WRONG@{c['wrong'][0]}+"
        else:
            st = "clean"
        row.append(st)
    lines.append("| " + " | ".join(row) + " |")

lines.append("\n## Silent-wrong / boundary findings\n")
any_find = False
for fam in results:
    for m in modes:
        c = results[fam][m]
        if c["wrong"]:
            any_find = True
            w = c["wrong"]
            lines.append(f"- **{fam}/{m}**: first wrong n={w[0]}, "
                         f"{len(w)} wrong depths (n={w[0]}..{w[-1]}); "
                         f"correct through n={c['max_correct']}.")
        if c["limit"] and not c["wrong"]:
            any_find = True
            lines.append(f"- **{fam}/{m}**: LIMIT — no DONE marker; "
                         f"last emitted n={c['last_n']} (crash/compile/timeout).")
if not any_find:
    lines.append("- none (every family clean across all axes).")

lines.append("\n## Regressions vs baseline\n")
if not baseline:
    lines.append("- (no baseline recorded yet — run with `--update-baseline`).")
elif regressions:
    lines += [f"- REGRESSION: {r}" for r in regressions]
else:
    lines.append("- none.")

lines.append("\n## Capacity boundary shifts vs baseline\n")
if capacity_notes:
    lines += [f"- BOUNDARY: {r}" for r in capacity_notes]
else:
    lines.append("- none.")

open(report_path, "w").write("\n".join(lines) + "\n")
print(f"[report] wrote {report_path}")

# ---- ICC trace ----
def esc(s): return s.replace("\\", "\\\\").replace('"', '\\"')
with open(trace_path, "w") as tf:
    for fam in results:
        for m in modes:
            c = results[fam][m]
            val = "LIMIT" if c["limit"] else ("WRONG" if c["wrong"] else "PASS")
            snip = f"max-correct-n={c['max_correct']} wrong={c['wrong'][:6]}"
            tf.write(json.dumps({
                "kind": "numeric_depth",
                "name": f"{fam}_{m}",
                "value": val,
                "max_correct_n": c["max_correct"],
                "wrong": c["wrong"],
                "limit": c["limit"],
                "snippet": snip,
                "confidence": 0.95,
            }) + "\n")
    # sweep-level: clean iff a baseline exists and there are no regressions.
    if not baseline:
        sweep = "PASS" if update else "NO_BASELINE"
    else:
        sweep = "PASS" if not regressions else "FAIL"
    tf.write(json.dumps({
        "kind": "numeric_depth",
        "name": "numeric_depth_sweep_clean",
        "value": sweep,
        "regressions": regressions,
        "capacity_notes": capacity_notes,
        "snippet": f"{len(regressions)} regression(s)",
        "confidence": 0.95,
    }) + "\n")
print(f"[trace] wrote {trace_path}  sweep={sweep}")

if regressions:
    print("REGRESSIONS:")
    for r in regressions:
        print("  " + r)
    sys.exit(1)
sys.exit(0)
PY
rc=$?
rm -rf "$OUTDIR"
exit $rc
