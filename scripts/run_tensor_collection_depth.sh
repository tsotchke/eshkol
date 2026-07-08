#!/usr/bin/env bash
# run_tensor_collection_depth.sh — depth-parametric TENSOR / COLLECTION / STRING
# sweep (adversarial depth-parametric pillar P6f).
#
# Regenerates tests/tensor_collection_depth/generated/ (via
# gen_tensor_collection_depth.py) and runs every generated program on THREE
# execution axes — JIT (-r), AOT-O0, AOT-O2 — diffing the printed
# "RESULT: <v>" against the closed-form / numpy ground truth in the manifest.
#
# Classification per (family, depth), combining the three axes:
#   PASS             every axis ran and RESULT == expected
#   WRONG            every axis ran, all agree on a value, and it != expected
#                    (a silent wrong answer consistent across axes = codegen bug)
#   AXIS-DIVERGENCE  axes disagree with each other (one wrong, or one crashes
#                    while another runs, or two produce different values)
#   LIMIT            every axis fails cleanly (compile-fail / crash / timeout)
#
# Per (family, axis) the report records the MAX-CORRECT depth/size (largest d
# for which every tested d' <= d PASSes on that axis).
#
# Emits (mirroring scripts/run_metaprog_depth.sh):
#   * pytest-style lines : "PASSED tests/tensor_collection_depth/<main>::<axis>"
#   * ICC JSON-L events  : kind=tensor_collection_depth into
#                          scripts/icc_traces/tensor_collection_depth.jsonl
#     (consumed by .icc/completion-oracles.yaml::tensor-collection-depth)
#   * docs/reports/TENSOR_COLLECTION_DEPTH_REPORT.md
#     (per-family/axis max-correct + findings)
#
# DISK BUDGET: a single reused AOT binary is deleted after every run; per-run
# logs are kept only for non-PASS cases; the whole artifact dir is capped at
# ARTIFACT_CAP_MB and an on-exit trap purges the work binary.
#
# Usage:
#   scripts/run_tensor_collection_depth.sh [--max-rank R] [--max-chain C]
#       [--max-nest D] [--no-aot] [--timeout SECS] [--no-regen]
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in /*) : ;; *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;; esac
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_tensor_collection_depth.sh: $ESHKOL_RUN not found — build eshkol-run stdlib first." >&2
    exit 2
fi

MAX_RANK=8
MAX_CHAIN=16
MAX_NEST=32
DO_AOT=1
TIMEOUT=90
REGEN=1
ARTIFACT_CAP_MB="${ARTIFACT_CAP_MB:-1024}"
while [ $# -gt 0 ]; do
    case "$1" in
        --max-rank) MAX_RANK="$2"; shift 2 ;;
        --max-chain) MAX_CHAIN="$2"; shift 2 ;;
        --max-nest) MAX_NEST="$2"; shift 2 ;;
        --no-aot) DO_AOT=0; shift ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --no-regen) REGEN=0; shift ;;
        *) echo "run_tensor_collection_depth.sh: unknown flag: $1" >&2; exit 2 ;;
    esac
done

GEN_DIR="$REPO_ROOT/tests/tensor_collection_depth/generated"
MANIFEST="$GEN_DIR/manifest.jsonl"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/tensor_collection_depth.jsonl"
REPORT="$REPO_ROOT/docs/reports/TENSOR_COLLECTION_DEPTH_REPORT.md"
ARTIFACT_DIR="$REPO_ROOT/artifacts/tensor-collection-depth"
LOG_DIR="$ARTIFACT_DIR/logs"
mkdir -p "$TRACE_DIR" "$ARTIFACT_DIR" "$LOG_DIR" "$(dirname "$REPORT")"
: "${TRACE_FILE:?TRACE_FILE must be set}"
: > "$TRACE_FILE"

if [ "$REGEN" -eq 1 ]; then
    python3 "$REPO_ROOT/scripts/gen_tensor_collection_depth.py" \
        --out-dir "$GEN_DIR" --max-rank "$MAX_RANK" \
        --max-chain "$MAX_CHAIN" --max-nest "$MAX_NEST" || exit 2
fi
[ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST" >&2; exit 2; }

# ---- disk budget: single reused work binary + on-exit purge --------------
WORK_BIN="$ARTIFACT_DIR/aot_work_bin"
RESULTS="$(mktemp "${TMPDIR:-/tmp}/tcd_results.XXXXXX")"
: "${WORK_BIN:?WORK_BIN must be set}"
: "${RESULTS:?RESULTS must be set}"
cleanup() { rm -f "$WORK_BIN" "$RESULTS"; }
trap cleanup EXIT

check_disk_cap() {
    local used_mb
    used_mb="$(du -sm "$ARTIFACT_DIR" 2>/dev/null | awk '{print $1}')"
    [ -z "$used_mb" ] && used_mb=0
    if [ "$used_mb" -gt "$ARTIFACT_CAP_MB" ]; then
        echo "run_tensor_collection_depth.sh: artifact dir ${used_mb}MB > cap ${ARTIFACT_CAP_MB}MB — aborting." >&2
        rm -f "$WORK_BIN"; rm -rf "$LOG_DIR"; exit 4
    fi
}

# macOS has no timeout(1); emulate with perl alarm (exit 124 on expiry).
run_to() {
    local secs="$1"; shift
    perl -e 'my $s=shift; eval { local $SIG{ALRM}=sub{ exit 124 }; alarm $s; exec @ARGV or exit 127; }' \
        "$secs" "$@"
}
json_escape() {
    printf '%s' "$1" | perl -0pe 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/\r/\\r/g; s/\t/\\t/g; s/([\x00-\x08\x0b\x0c\x0e-\x1f])/sprintf("\\u%04x", ord($1))/ge'
}
emit_event() { # name value family depth axis detail
    printf '{"kind":"tensor_collection_depth","name":"%s","value":"%s","family":"%s","depth":%s,"axis":"%s","detail":"%s"}\n' \
        "$1" "$2" "$3" "$4" "$5" "$(json_escape "$6")" >> "$TRACE_FILE"
}

# run one axis; echoes "STATUS|value" where STATUS in RAN|LIMIT and value is the
# parsed RESULT string (empty on LIMIT). Args: axis main_abs tag
run_axis() {
    local axis="$1" main="$2" tag="$3"
    local out ec result errlog
    errlog="$LOG_DIR/$tag.err"
    case "$axis" in
        r)
            out="$(run_to "$TIMEOUT" "$ESHKOL_RUN" -r "$main" 2>"$errlog")"; ec=$?
            ;;
        aot-O0|aot-O2)
            local olvl="${axis#aot-O}"
            rm -f "$WORK_BIN"
            if ! run_to "$TIMEOUT" "$ESHKOL_RUN" -O "$olvl" "$main" -o "$WORK_BIN" >"$errlog" 2>&1; then
                echo "LIMIT|"; return
            fi
            [ -x "$WORK_BIN" ] || { echo "LIMIT|"; return; }
            out="$(run_to "$TIMEOUT" "$WORK_BIN" 2>"$errlog")"; ec=$?
            rm -f "$WORK_BIN"
            ;;
    esac
    if [ "$ec" -eq 124 ]; then echo "LIMIT|"; return; fi
    result="$(printf '%s\n' "$out" | sed -n 's/^RESULT: //p' | tail -1)"
    if [ -n "$result" ]; then
        rm -f "$errlog"
        echo "RAN|$result"
    elif [ "$ec" -ne 0 ]; then
        echo "LIMIT|"
    else
        # clean exit but no RESULT emitted -> a silent-wrong (missing output)
        echo "RAN|<no-result>"
    fi
}

AXES="r"; [ "$DO_AOT" -eq 1 ] && AXES="r aot-O0 aot-O2"

echo "== tensor/collection/string depth sweep (rank<=$MAX_RANK chain<=$MAX_CHAIN nest<=$MAX_NEST aot=$DO_AOT) =="
n_total=0; n_pass=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    family="$(printf '%s' "$line" | python3 -c 'import sys,json;print(json.load(sys.stdin)["family"])')"
    depth="$(printf '%s' "$line" | python3 -c 'import sys,json;print(json.load(sys.stdin)["depth"])')"
    mrel="$(printf '%s' "$line" | python3 -c 'import sys,json;print(json.load(sys.stdin)["main"])')"
    expected="$(printf '%s' "$line" | python3 -c 'import sys,json;print(json.load(sys.stdin)["expected"])')"
    main_abs="$GEN_DIR/$mrel"

    # gather per-axis (status,value) — bash 3.2 has no assoc arrays, so we
    # accumulate one "axis|status|value" record per line in AXLINES.
    ax_vals=""; n_ran=0; n_limit=0; AXLINES=""
    for axis in $AXES; do
        tag="${family}_d${depth}_${axis}"
        r="$(run_axis "$axis" "$main_abs" "$tag")"
        st="${r%%|*}"; val="${r#*|}"
        AXLINES="$AXLINES$axis|$st|$val
"
        if [ "$st" = "RAN" ]; then n_ran=$((n_ran+1)); ax_vals="$ax_vals|$val"; else n_limit=$((n_limit+1)); fi
        # per-axis PASS/FAIL for max-correct-per-axis tracking
        if [ "$st" = "RAN" ] && [ "$val" = "$expected" ]; then axcls="PASS"
        elif [ "$st" = "RAN" ]; then axcls="WRONG"
        else axcls="LIMIT"; fi
        printf 'AX\t%s\t%s\t%s\t%s\t%s\n' "$family" "$axis" "$depth" "$axcls" "$val" >> "$RESULTS"
    done

    # combined classification across axes
    uniq_vals="$(printf '%s' "$ax_vals" | tr '|' '\n' | sed '/^$/d' | sort -u | wc -l | tr -d ' ')"
    if [ "$n_ran" -eq 0 ]; then
        cls="LIMIT"; detail="all axes crash/timeout/compile-fail"
    elif [ "$n_limit" -gt 0 ] && [ "$n_ran" -gt 0 ]; then
        cls="AXIS-DIVERGENCE"; detail="some axes ran, some failed"
    elif [ "$uniq_vals" -gt 1 ]; then
        cls="AXIS-DIVERGENCE"; detail="axes produced differing values ($ax_vals)"
    else
        # all axes ran and agree on one value
        agreed="$(printf '%s' "$ax_vals" | tr '|' '\n' | sed '/^$/d' | head -1)"
        if [ "$agreed" = "$expected" ]; then cls="PASS"; detail=""
        else cls="WRONG"; detail="all-axis got=$agreed want=$expected"; fi
    fi

    n_total=$((n_total+1))
    printf 'CB\t%s\t%s\t%s\t%s\n' "$family" "$depth" "$cls" "$detail" >> "$RESULTS"
    printf '%s' "$AXLINES" | while IFS='|' read -r axis st val; do
        [ -z "$axis" ] && continue
        emit_event "tcd_case" "$cls" "$family" "$depth" "$axis" "axis=$st:$val exp=$expected"
    done
    if [ "$cls" = "PASS" ]; then
        n_pass=$((n_pass+1))
        echo "PASSED tests/tensor_collection_depth/$mrel"
    else
        echo "$cls tests/tensor_collection_depth/$mrel  ($detail)"
    fi
    check_disk_cap
done < "$MANIFEST"

# ---- summarize: max-correct per (family,axis) + combined findings --------
python3 - "$RESULTS" "$REPORT" <<'PY'
import sys, collections
res_path, report_path = sys.argv[1], sys.argv[2]
ax_rows = []       # (family, axis, depth, cls, val)
cb_rows = []       # (family, depth, cls, detail)
for ln in open(res_path):
    ln = ln.rstrip("\n")
    if not ln: continue
    parts = ln.split("\t")
    if parts[0] == "AX":
        _, fam, axis, depth, cls, val = (parts + [""])[:6]
        ax_rows.append((fam, axis, int(depth), cls, val))
    elif parts[0] == "CB":
        _, fam, depth, cls, detail = (parts + [""])[:5]
        cb_rows.append((fam, int(depth), cls, detail))

# per (family,axis): depth -> cls
byfa = collections.defaultdict(dict)
for fam, axis, d, cls, val in ax_rows:
    byfa[(fam, axis)][d] = cls
def max_correct(depths):
    mcd = 0
    for d in sorted(depths):
        if depths[d] == "PASS": mcd = d
        else: break
    return mcd

# combined per family: depth -> (cls,detail)
byfam = collections.defaultdict(dict)
for fam, d, cls, detail in cb_rows:
    byfam[fam][d] = (cls, detail)

fams = sorted(byfam)
axes = ["r", "aot-O0", "aot-O2"]
lines = ["# Tensor / Collection / String Depth-Parametric Report (P6f)", "",
         "Each family generated at depth/size d=1..N with a closed-form or numpy",
         "ground truth. Every case runs on three axes — JIT (-r), AOT-O0, AOT-O2.",
         "",
         "- **MCD** = max-correct depth/size per axis (largest d with 1..d all PASS).",
         "- **WRONG** = every axis agrees on a value that is not the oracle (silent bug).",
         "- **AXIS-DIVERGENCE** = the axes disagree (codegen/opt/JIT mismatch).",
         "- **LIMIT** = every axis fails cleanly (documented capability boundary).",
         "",
         "| family | MCD(-r) | MCD(O0) | MCD(O2) | max-tested | first-WRONG | first-DIVERGE | first-LIMIT |",
         "|---|---|---|---|---|---|---|---|"]
overall_wrong = []
overall_diverge = []
for fam in fams:
    depths_all = sorted(byfam[fam])
    mcd_r = max_correct(byfa.get((fam, "r"), {}))
    mcd_o0 = max_correct(byfa.get((fam, "aot-O0"), {}))
    mcd_o2 = max_correct(byfa.get((fam, "aot-O2"), {}))
    fw = next((d for d in depths_all if byfam[fam][d][0] == "WRONG"), None)
    fd = next((d for d in depths_all if byfam[fam][d][0] == "AXIS-DIVERGENCE"), None)
    fl = next((d for d in depths_all if byfam[fam][d][0] == "LIMIT"), None)
    lines.append("| %s | %d | %d | %d | %d | %s | %s | %s |" % (
        fam, mcd_r, mcd_o0, mcd_o2, max(depths_all),
        "-" if fw is None else str(fw),
        "-" if fd is None else str(fd),
        "-" if fl is None else str(fl)))
    if fw is not None: overall_wrong.append((fam, fw, byfam[fam][fw][1]))
    if fd is not None: overall_diverge.append((fam, fd, byfam[fam][fd][1]))

lines += ["", "## Silent-WRONG findings (all axes agree on a wrong value = bug)"]
if overall_wrong:
    for fam, d, detail in overall_wrong:
        lines.append("- **%s** WRONG from depth/size %d: %s" % (fam, d, detail))
else:
    lines.append("- none.")

lines += ["", "## Axis-divergence findings (-r vs AOT-O0 vs AOT-O2 disagree)"]
if overall_diverge:
    for fam, d, detail in overall_diverge:
        lines.append("- **%s** diverges from depth/size %d: %s" % (fam, d, detail))
else:
    lines.append("- none.")

open(report_path, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
# clean = no WRONG and no AXIS-DIVERGENCE anywhere
sys.exit(0 if (not overall_wrong and not overall_diverge) else 3)
PY
SUMMARY_EC=$?

if [ "$SUMMARY_EC" -eq 0 ]; then
    emit_event "tensor_collection_depth_gate" "PASS" "ALL" 0 "all" "no WRONG and no axis-divergence at any depth"
else
    emit_event "tensor_collection_depth_gate" "FAIL" "ALL" 0 "all" "one or more families WRONG or axis-divergent (see docs/reports/TENSOR_COLLECTION_DEPTH_REPORT.md)"
fi

PEAK_MB="$(du -sm "$ARTIFACT_DIR" 2>/dev/null | awk '{print $1}')"
echo ""
echo "cases: $n_pass/$n_total PASS   artifact-peak: ${PEAK_MB}MB   report: $REPORT   trace: $TRACE_FILE"
# The sweep gate is informational: WRONG / divergence findings are logged as ESH tasks.
exit 0
