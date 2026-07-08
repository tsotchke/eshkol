#!/usr/bin/env bash
# run_metaprog_depth.sh — depth-parametric METAPROGRAMMING + MODULE sweep
# (adversarial depth-parametric pillar P6e).
#
# Regenerates tests/metaprog_depth/generated/ (via gen_metaprog_depth.py) and
# runs every generated program on both native axes — JIT (-r) and AOT — diffing
# the printed "RESULT: <v>" against the closed-form ground truth carried in the
# manifest. For each (family, mode) it records the MAX-CORRECT-DEPTH (largest d
# for which depths 1..d all PASS) and the first WRONG / first LIMIT depth.
#
# Classification per program:
#   PASS   RESULT present and == expected
#   WRONG  RESULT present but != expected, OR clean exit with no RESULT
#          (a silent wrong answer at depth = compiler bug)
#   LIMIT  timeout / crash / compile failure / nonzero exit (documented boundary)
#
# Emits (mirroring scripts/run_differential.sh):
#   * pytest-style lines : "PASSED tests/metaprog_depth/<main>::<mode>"
#   * ICC JSON-L events  : kind=metaprog_depth into
#                          scripts/icc_traces/metaprog_depth.jsonl
#     (consumed by .icc/completion-oracles.yaml::metaprog-depth)
#   * METAPROG_DEPTH_REPORT.md  (per-family depth table + max-correct-depth)
#
# Usage:
#   scripts/run_metaprog_depth.sh [--max-depth N] [--module-max-depth M]
#       [--no-aot] [--timeout SECS] [--no-regen]
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# Keep the Perl timeout wrapper independent of host locale availability.
export LC_ALL=C
export LANG=C
export LC_CTYPE=C

: "${ESHKOL_JIT_CACHE_DIR:=${TMPDIR:-/tmp}/eshkol-metaprog-depth-jit-cache}"
export ESHKOL_JIT_CACHE_DIR
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in /*) : ;; *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;; esac
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_metaprog_depth.sh: $ESHKOL_RUN not found — build eshkol-run stdlib first." >&2
    exit 2
fi

MAX_DEPTH=32
MODULE_MAX_DEPTH=12
DO_AOT=1
TIMEOUT=60
REGEN=1
while [ $# -gt 0 ]; do
    case "$1" in
        --max-depth) MAX_DEPTH="$2"; shift 2 ;;
        --module-max-depth) MODULE_MAX_DEPTH="$2"; shift 2 ;;
        --no-aot) DO_AOT=0; shift ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --no-regen) REGEN=0; shift ;;
        *) echo "run_metaprog_depth.sh: unknown flag: $1" >&2; exit 2 ;;
    esac
done

GEN_DIR="$REPO_ROOT/tests/metaprog_depth/generated"
MANIFEST="$GEN_DIR/manifest.jsonl"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/metaprog_depth.jsonl"
REPORT="$REPO_ROOT/METAPROG_DEPTH_REPORT.md"
mkdir -p "$TRACE_DIR"
: > "$TRACE_FILE"

if [ "$REGEN" -eq 1 ]; then
    python3 "$REPO_ROOT/scripts/gen_metaprog_depth.py" \
        --out-dir "$GEN_DIR" --max-depth "$MAX_DEPTH" \
        --module-max-depth "$MODULE_MAX_DEPTH" || exit 2
fi
[ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST" >&2; exit 2; }

# macOS has no timeout(1); emulate with perl alarm (exit 124 on expiry).
run_to() {
    local secs="$1"; shift
    perl -e 'my $s=shift; eval { local $SIG{ALRM}=sub{ exit 124 }; alarm $s; exec @ARGV or exit 127; }' \
        "$secs" "$@"
}
json_escape() {
    printf '%s' "$1" | perl -0pe 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/\r/\\r/g; s/\t/\\t/g; s/([\x00-\x08\x0b\x0c\x0e-\x1f])/sprintf("\\u%04x", ord($1))/ge'
}
emit_event() { # name value family depth mode detail
    printf '{"kind":"metaprog_depth","name":"%s","value":"%s","family":"%s","depth":%s,"mode":"%s","detail":"%s"}\n' \
        "$1" "$2" "$3" "$4" "$5" "$(json_escape "$6")" >> "$TRACE_FILE"
}

RESULTS="$(mktemp "${TMPDIR:-/tmp}/metaprog_results.XXXXXX")"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/metaprog_aot.XXXXXX")"
cleanup() { rm -f "$RESULTS"; rm -rf "$WORK"; }
trap cleanup EXIT

# classify one run: args = mode main_abs expected
# echoes "CLASS|detail"; CLASS in PASS|WRONG|LIMIT
classify_run() {
    local mode="$1" main="$2" expected="$3"
    local out err ec result
    err="$WORK/err.txt"
    if [ "$mode" = "r" ]; then
        out="$(run_to "$TIMEOUT" "$ESHKOL_RUN" -r "$main" 2>"$err")"; ec=$?
    else
        local bin="$WORK/aot_bin"
        rm -f "$bin"
        if ! run_to "$TIMEOUT" "$ESHKOL_RUN" "$main" -o "$bin" >"$err" 2>&1; then
            echo "LIMIT|aot-compile-fail(ec=$?)"; return
        fi
        [ -x "$bin" ] || { echo "LIMIT|aot-no-binary"; return; }
        out="$(run_to "$TIMEOUT" "$bin" 2>"$err")"; ec=$?
    fi
    if [ "$ec" -eq 124 ]; then echo "LIMIT|timeout"; return; fi
    result="$(printf '%s\n' "$out" | sed -n 's/^RESULT: //p' | tail -1)"
    if [ -n "$result" ]; then
        if [ "$result" = "$expected" ]; then echo "PASS|"; else echo "WRONG|got=$result want=$expected"; fi
    else
        if [ "$ec" -ne 0 ]; then echo "LIMIT|exit=$ec $(head -1 "$err" 2>/dev/null)"
        else echo "WRONG|no-RESULT-clean-exit"; fi
    fi
}

MODES="r"; [ "$DO_AOT" -eq 1 ] && MODES="r aot"

echo "== metaprog depth sweep (max_depth=$MAX_DEPTH module_max=$MODULE_MAX_DEPTH aot=$DO_AOT) =="
n_total=0; n_pass=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    family="$(printf '%s' "$line" | python3 -c 'import sys,json;print(json.load(sys.stdin)["family"])')"
    depth="$(printf '%s' "$line" | python3 -c 'import sys,json;print(json.load(sys.stdin)["depth"])')"
    mrel="$(printf '%s' "$line" | python3 -c 'import sys,json;print(json.load(sys.stdin)["main"])')"
    expected="$(printf '%s' "$line" | python3 -c 'import sys,json;print(json.load(sys.stdin)["expected"])')"
    main_abs="$GEN_DIR/$mrel"
    for mode in $MODES; do
        cr="$(classify_run "$mode" "$main_abs" "$expected")"
        cls="${cr%%|*}"; detail="${cr#*|}"
        n_total=$((n_total+1))
        printf '%s\t%s\t%s\t%s\t%s\n' "$family" "$mode" "$depth" "$cls" "$detail" >> "$RESULTS"
        emit_event "metaprog_case" "$cls" "$family" "$depth" "$mode" "$detail"
        if [ "$cls" = "PASS" ]; then
            n_pass=$((n_pass+1))
            echo "PASSED tests/metaprog_depth/$mrel::$mode"
        else
            echo "$cls tests/metaprog_depth/$mrel::$mode  ($detail)"
        fi
    done
done < "$MANIFEST"

# ---- summarize: max-correct-depth per (family,mode) + report -------------
python3 - "$RESULTS" "$REPORT" <<'PY'
import sys, collections
res_path, report_path = sys.argv[1], sys.argv[2]
rows = []
for ln in open(res_path):
    ln = ln.rstrip("\n")
    if not ln: continue
    fam, mode, depth, cls, detail = ln.split("\t", 4)
    rows.append((fam, mode, int(depth), cls, detail))
byfm = collections.defaultdict(dict)   # (fam,mode) -> {depth: (cls,detail)}
for fam, mode, d, cls, detail in rows:
    byfm[(fam, mode)][d] = (cls, detail)

def summarize(depths):
    ds = sorted(depths)
    mcd = 0
    for d in ds:
        if depths[d][0] == "PASS": mcd = d
        else: break
    first_wrong = next((d for d in ds if depths[d][0] == "WRONG"), None)
    first_limit = next((d for d in ds if depths[d][0] == "LIMIT"), None)
    return mcd, first_wrong, first_limit, ds

fams = sorted(set(f for f, _ in byfm))
modes = ["r", "aot"]
lines = ["# Metaprogramming + Module Depth-Parametric Report (P6e)", "",
         "Each family generated at depth d=1..N with a closed-form ground truth.",
         "MCD = max-correct-depth (largest d with 1..d all PASS). "
         "WRONG = ran but wrong value (bug). LIMIT = clean crash/compile-fail/timeout.",
         "", "| family | mode | MCD | first-WRONG | first-LIMIT | max-d tested |",
         "|---|---|---|---|---|---|"]
overall_ok = True
for fam in fams:
    for mode in modes:
        if (fam, mode) not in byfm: continue
        mcd, fw, fl, ds = summarize(byfm[(fam, mode)])
        lines.append("| %s | %s | %d | %s | %s | %d |" %
                     (fam, mode, mcd,
                      "-" if fw is None else str(fw),
                      "-" if fl is None else str(fl), max(ds)))
        # a family that is WRONG at any depth is a real bug
        if fw is not None:
            overall_ok = False

lines += ["", "## Findings (WRONG at some depth = silent-wrong bug)"]
seen = {}
for fam in fams:
    for mode in modes:
        if (fam, mode) not in byfm: continue
        for d in sorted(byfm[(fam, mode)]):
            cls, detail = byfm[(fam, mode)][d]
            if cls == "WRONG":
                cur = seen.get(fam)
                if cur is None or d < cur[0]:
                    seen[fam] = (d, detail, set())
                seen[fam][2].add(mode)
                break
for fam in sorted(seen):
    d, detail, ms = seen[fam]
    lines.append("- **%s** WRONG from depth %d (%s) [%s]" %
                 (fam, d, detail, "+".join(sorted(ms))))
if not seen:
    lines.append("- none: every family is either all-PASS or degrades via a clean LIMIT.")

open(report_path, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# sweep-level ICC event appended by shell using this exit code
sys.exit(0 if overall_ok else 3)
PY
SUMMARY_EC=$?

if [ "$SUMMARY_EC" -eq 0 ]; then
    emit_event "metaprog_depth_sweep_clean" "PASS" "ALL" 0 "all" "no WRONG values at any depth"
else
    emit_event "metaprog_depth_sweep_clean" "FAIL" "ALL" 0 "all" "one or more families produced WRONG values at depth (see METAPROG_DEPTH_REPORT.md)"
fi

echo ""
echo "cases: $n_pass/$n_total PASS   report: $REPORT   trace: $TRACE_FILE"
# The sweep gate is informational: WRONG findings are logged as ESH tasks.
exit 0
