#!/usr/bin/env bash
# run_stress.sh — P4 extreme stress harness (adversarial testing campaign).
#
# Runs every program listed in tests/stress/budgets.tsv under the JIT (-r)
# and/or AOT with EXPLICIT budgets asserted by THIS runner (not the program):
#   * wall-time ceiling  (perl alarm; macOS has no timeout(1))
#   * max-RSS ceiling    (/usr/bin/time -l "maximum resident set size")
#   * exit code 0
#   * required stdout substring
#
# Verdicts per (file, mode):
#   PASS      all budgets met, expected output present
#   FAIL      exit != 0 (no signal) or expected output missing (wrong value)
#   CRASH     killed by a signal other than SIGALRM (SIGSEGV/SIGILL/SIGBUS/…)
#   HANG      killed by the alarm (SIGALRM) — wall-time ceiling exceeded
#   OVER-RSS  ran fine but exceeded the RSS ceiling (unbounded-memory class)
#   OVER-TIME finished under the alarm but past the wall-time ceiling
#   XKNOWN    row is pinned to a documented-open bug (xknown column) and did
#             not PASS — recorded, does not fail the gate
#   XPASS     an xknown row PASSED — FAILS the gate (stale XKNOWN: promote the
#             row to a normal budget row and close the ledger task)
#
# Special classes (budgets.tsv "class" column):
#   jitcache  r-only: 50 sequential -r invocations sharing one fresh JIT cache
#             dir. Run 1 gets the row timeout (cold); runs 2..50 must each
#             finish within STRESS_WARM_CEILING_S (default 5s).
#   rep3      run 3x per mode; all three stdouts must be byte-identical and
#             pass the normal budget checks (spawn/join flake detector).
#
# ICC wiring (mirrors scripts/run_sicp_smoke.sh):
#   * pytest-style lines : "PASSED tests/stress/<file>::<mode>" / "FAILED …"
#   * JSON-L events      : kind=stress_smoke, name=stress_<base>_<mode>,
#                          into scripts/icc_traces/stress_smoke.jsonl,
#                          consumed by .icc/completion-oracles.yaml::stress-budget
#   * summary event      : name=stress_suite_green PASS/FAIL
#
# Usage: scripts/run_stress.sh [--quick] [--no-aot] [--only <substring>]
#   --quick   run only rows with quick=1 and cap jitcache at 5 iterations (CI)
#   --no-aot  skip AOT mode everywhere
#   --only S  run only rows whose file contains S
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
STRESS_DIR="$REPO_ROOT/tests/stress"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/stress_smoke.jsonl"
mkdir -p "$TRACE_DIR"
: > "$TRACE_FILE"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
    *) ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
esac
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_stress.sh: $BUILD_DIR/eshkol-run not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
    exit 2
fi

QUICK=0; DO_AOT=1; ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --quick) QUICK=1 ;;
        --no-aot) DO_AOT=0 ;;
        --only) shift; ONLY="${1:-}" ;;
        *) echo "run_stress.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

WARM_CEILING_S="${STRESS_WARM_CEILING_S:-5}"
JITCACHE_RUNS=50
[ "$QUICK" -eq 1 ] && JITCACHE_RUNS=5

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-stress.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
# Fresh, harness-private JIT cache: first -r per file is genuinely cold.
export ESHKOL_JIT_CACHE_DIR="$WORK/jit-cache"
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

# Regenerate the large generated sources (deterministic).
bash "$STRESS_DIR/gen_stress_sources.sh" >/dev/null

json_escape() {
    printf '%s' "$1" | perl -0pe 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/\r/\\r/g; s/\t/\\t/g; s/([\x00-\x08\x0b\x0c\x0e-\x1f])/sprintf("\\u%04x", ord($1))/ge'
}

emit_event() { # name value snippet
    printf '{"kind":"stress_smoke","name":"%s","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$(json_escape "$1")" "$(json_escape "$2")" "$(json_escape "$3")" >> "$TRACE_FILE"
}

# run_budgeted <timeout_s> <cmd...>
# Globals set: RB_RC RB_WALL_S RB_RSS_MB RB_OUT_FILE
RB_OUT_FILE="$WORK/out.txt"
RB_TIME_FILE="$WORK/time.txt"
run_budgeted() {
    local tmo="$1"; shift
    local t0 t1
    t0=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
    # /usr/bin/time -l writes rusage (incl. "maximum resident set size", bytes
    # on macOS) to stderr → captured separately so program stderr stays with
    # the program output for diagnosis.
    # </dev/null: keep the budgets.tsv read-loop's stdin away from programs.
    /usr/bin/time -l perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
        "$tmo" "$@" > "$RB_OUT_FILE" 2> "$RB_TIME_FILE" < /dev/null
    RB_RC=$?
    t1=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
    RB_WALL_S=$(perl -e 'printf "%.2f", $ARGV[1]-$ARGV[0]' "$t0" "$t1")
    RB_RSS_MB=$(awk '/maximum resident set size/{printf "%d", $1/1048576}' "$RB_TIME_FILE")
    [ -n "$RB_RSS_MB" ] || RB_RSS_MB=0
    # program stderr (compile errors etc.) is interleaved into time file; keep
    # the non-rusage part with the output for the failure snippet.
    grep -vE 'real .*user .*sys|maximum resident set size|average .* size|page reclaims|page faults|swaps|block .* operations|messages (sent|received)|signals received|context switches|instructions retired|cycles elapsed|peak memory footprint' \
        "$RB_TIME_FILE" >> "$RB_OUT_FILE" 2>/dev/null || true
}

# classify <rc> <wall_s> <rss_mb> <timeout_s> <rss_ceiling_mb> <expect> <out_file>
classify() {
    local rc="$1" wall="$2" rss="$3" tmo="$4" ceil="$5" expect="$6" out="$7"
    if [ "$rc" -eq 142 ]; then echo "HANG"; return; fi          # 128+SIGALRM
    if [ "$rc" -gt 128 ]; then echo "CRASH"; return; fi
    if [ "$rc" -ne 0 ]; then echo "FAIL"; return; fi
    if [ "$expect" != "-" ] && ! grep -qF -- "$expect" "$out"; then echo "FAIL"; return; fi
    if [ "$ceil" != "-" ] && [ "$rss" -gt "$ceil" ]; then echo "OVER-RSS"; return; fi
    if perl -e 'exit(($ARGV[0] > $ARGV[1]) ? 0 : 1)' "$wall" "$tmo"; then echo "OVER-TIME"; return; fi
    echo "PASS"
}

total=0; passed=0; failed=0; xknown_n=0; xpassed=0
declare -a fail_lines=()

# record <file> <mode> <verdict> <xknown> <detail>
record() {
    local file="$1" mode="$2" v="$3" xk="$4" detail="$5"
    local base; base=$(basename "$file" .esk)
    total=$((total+1))
    if [ "$xk" != "-" ]; then
        if [ "$v" = "PASS" ]; then
            xpassed=$((xpassed+1))
            printf '  XPASS   tests/stress/%s::%s  (%s is documented-open but PASSED — promote it!)\n' "$file" "$mode" "$xk"
            echo "XPASS tests/stress/$file::$mode"
            emit_event "stress_${base}_${mode}" "XPASS" "$file $mode -> XPASS ($xk) $detail"
            fail_lines+=("XPASS $file::$mode ($xk)")
        else
            xknown_n=$((xknown_n+1))
            printf '  XKNOWN  tests/stress/%s::%s  [%s] (%s)\n' "$file" "$mode" "$v" "$xk"
            echo "XFAIL tests/stress/$file::$mode"
            emit_event "stress_${base}_${mode}" "XKNOWN" "$file $mode -> $v ($xk) $detail"
        fi
        return
    fi
    emit_event "stress_${base}_${mode}" "$([ "$v" = PASS ] && echo PASS || echo FAIL)" "$file $mode -> $v $detail"
    if [ "$v" = "PASS" ]; then
        passed=$((passed+1))
        printf '  PASS    tests/stress/%s::%s  (%s)\n' "$file" "$mode" "$detail"
        echo "PASSED tests/stress/$file::$mode"
    else
        failed=$((failed+1))
        printf '  %-7s tests/stress/%s::%s  (%s)\n' "$v" "$file" "$mode" "$detail"
        echo "FAILED tests/stress/$file::$mode"
        fail_lines+=("$v $file::$mode $detail")
    fi
}

snippet_of() { tail -c 300 "$RB_OUT_FILE" | tr '\n' ' ' | cut -c1-200; }

echo "P4 stress harness → $TRACE_FILE"
echo "  quick=$QUICK aot=$DO_AOT eshkol-run=$ESHKOL_RUN"
echo

while IFS=$'\t' read -r file mode class timeout_s rss_r rss_aot quick xknown expect; do
    case "$file" in \#*|"") continue ;; esac
    if [ "$QUICK" -eq 1 ] && [ "$quick" != "1" ] && [ "$class" != "jitcache" ]; then continue; fi
    if [ -n "$ONLY" ] && [ "${file#*"$ONLY"}" = "$file" ]; then continue; fi
    src="$STRESS_DIR/$file"
    if [ ! -f "$src" ]; then
        record "$file" "r" "FAIL" "$xknown" "source missing"
        continue
    fi

    # ── jitcache class: 50 sequential warm-path -r runs ────────────────────
    if [ "$class" = "jitcache" ]; then
        cache="$WORK/jitcache-probe"; rm -rf "$cache"; mkdir -p "$cache"
        verdict="PASS"; detail=""
        for i in $(seq 1 "$JITCACHE_RUNS"); do
            lim="$timeout_s"; [ "$i" -gt 1 ] && lim="$WARM_CEILING_S"
            ESHKOL_JIT_CACHE_DIR="$cache" run_budgeted "$lim" "$ESHKOL_RUN" -r "$src"
            v=$(classify "$RB_RC" "$RB_WALL_S" "$RB_RSS_MB" "$lim" "$rss_r" "$expect" "$RB_OUT_FILE")
            if [ "$v" != "PASS" ]; then
                verdict="$v"; detail="run $i/$JITCACHE_RUNS: rc=$RB_RC wall=${RB_WALL_S}s rss=${RB_RSS_MB}MB limit=${lim}s :: $(snippet_of)"
                break
            fi
            [ "$i" -eq 1 ] && detail="cold=${RB_WALL_S}s"
            [ "$i" -gt 1 ] && detail="$detail warm${i}=${RB_WALL_S}s"
        done
        # keep only first+last warm sample in the detail to stay readable
        detail=$(printf '%s' "$detail" | awk '{ if (NF<=3) print; else print $1, $2, "...", $NF }')
        record "$file" "r" "$verdict" "$xknown" "$detail"
        continue
    fi

    reps=1; [ "$class" = "rep3" ] && reps=3

    # ── JIT (-r) ────────────────────────────────────────────────────────────
    if [ "$mode" = "r" ] || [ "$mode" = "both" ]; then
        verdict="PASS"; detail=""; ref_out=""
        for rep in $(seq 1 "$reps"); do
            run_budgeted "$timeout_s" "$ESHKOL_RUN" -r "$src"
            v=$(classify "$RB_RC" "$RB_WALL_S" "$RB_RSS_MB" "$timeout_s" "$rss_r" "$expect" "$RB_OUT_FILE")
            detail="rc=$RB_RC wall=${RB_WALL_S}s rss=${RB_RSS_MB}MB"
            if [ "$v" != "PASS" ]; then verdict="$v"; detail="$detail :: $(snippet_of)"; break; fi
            if [ "$reps" -gt 1 ]; then
                if [ "$rep" -eq 1 ]; then ref_out=$(cat "$RB_OUT_FILE")
                elif [ "$ref_out" != "$(cat "$RB_OUT_FILE")" ]; then
                    verdict="FAIL"; detail="nondeterministic stdout across rep $rep (flake)"; break
                fi
            fi
        done
        record "$file" "r" "$verdict" "$xknown" "$detail"
    fi

    # ── AOT ──────────────────────────────────────────────────────────────────
    if [ "$DO_AOT" -eq 1 ] && { [ "$mode" = "aot" ] || [ "$mode" = "both" ]; }; then
        bin="$WORK/$(basename "$file" .esk).bin"; rm -f "$bin"
        run_budgeted "$timeout_s" "$ESHKOL_RUN" "$src" -o "$bin"
        if [ "$RB_RC" -ne 0 ] || [ ! -x "$bin" ]; then
            v="FAIL"; [ "$RB_RC" -eq 142 ] && v="HANG"; [ "$RB_RC" -gt 128 ] && [ "$RB_RC" -ne 142 ] && v="CRASH"
            record "$file" "aot" "$v" "$xknown" "compile rc=$RB_RC wall=${RB_WALL_S}s :: $(snippet_of)"
        else
            verdict="PASS"; detail=""; ref_out=""
            for rep in $(seq 1 "$reps"); do
                run_budgeted "$timeout_s" "$bin"
                v=$(classify "$RB_RC" "$RB_WALL_S" "$RB_RSS_MB" "$timeout_s" "$rss_aot" "$expect" "$RB_OUT_FILE")
                detail="rc=$RB_RC wall=${RB_WALL_S}s rss=${RB_RSS_MB}MB"
                if [ "$v" != "PASS" ]; then verdict="$v"; detail="$detail :: $(snippet_of)"; break; fi
                if [ "$reps" -gt 1 ]; then
                    if [ "$rep" -eq 1 ]; then ref_out=$(cat "$RB_OUT_FILE")
                    elif [ "$ref_out" != "$(cat "$RB_OUT_FILE")" ]; then
                        verdict="FAIL"; detail="nondeterministic stdout across rep $rep (flake)"; break
                    fi
                fi
            done
            record "$file" "aot" "$verdict" "$xknown" "$detail"
            rm -f "$bin"
        fi
    fi
done < "$STRESS_DIR/budgets.tsv"

echo
suite="PASS"
if [ "$failed" -ne 0 ] || [ "$xpassed" -ne 0 ]; then suite="FAIL"; fi
emit_event "stress_suite_green" "$suite" "pass=$passed fail=$failed xknown=$xknown_n xpass=$xpassed total=$total quick=$QUICK"
echo "Stress summary: $passed/$((passed+failed)) budget probes PASS; $xknown_n XKNOWN (documented-open), $xpassed XPASS; $total total."
if [ "${#fail_lines[@]}" -gt 0 ]; then
    echo "Failures:"
    for l in "${fail_lines[@]}"; do echo "  - $l"; done
fi
echo "Trace written: $TRACE_FILE"
[ "$suite" = "PASS" ] || exit 1
exit 0
