#!/usr/bin/env bash
# run_region_parity.sh — the whole-program half of xla_region_formation_result_parity.
#
# Runs every program in tests/xla/regions/ twice: once as it has always run,
# and once with ESHKOL_XLA_REGIONS=1, which makes the compiler outline its
# maximal device-eligible subgraphs and call them on the device where the
# subtree used to be evaluated. The two runs' STDOUT must agree.
#
# Stdout only, on purpose: the trace and the LLVM vectorizer remarks go to
# stderr, and a comparison that included them would be comparing diagnostics
# rather than results.
#
# A program that forms no rewritable region is still run twice and still
# compared — it is the control that says the regions-on path did not change a
# program it was not supposed to touch.
#
# Everything this writes lives under <repo>/.scratch. Never /tmp.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 2

BUILD_DIR="${XLA_GATE_BUILD_DIR:-$REPO_ROOT/.scratch/xla_gate/build}"
RUN="$BUILD_DIR/eshkol-run"
CORPUS="${1:-$REPO_ROOT/tests/xla/regions}"
WORK="$REPO_ROOT/.scratch/region_parity"
mkdir -p "$WORK"

if [ ! -x "$RUN" ]; then
    echo "run_region_parity.sh: $RUN not built"
    exit 2
fi

total=0; agreed=0; failed=0; no_region=0
printf '%-42s %8s %8s %9s  %s\n' program regions off on verdict
printf '%s\n' "--------------------------------------------------------------------------------"

for f in "$CORPUS"/*.esk; do
    name="$(basename "$f")"
    total=$((total + 1))

    # Regions off: the program exactly as it has always run.
    "$RUN" -r "$f" > "$WORK/$name.off" 2> "$WORK/$name.off.err"
    off_rc=$?

    # Regions on. The report tells us how many regions were rewritten, which
    # is what separates "agreed because the device got it right" from "agreed
    # because nothing went to the device".
    ESHKOL_XLA_REGIONS=1 \
    ESHKOL_XLA_PJRT=1 \
    ${ESHKOL_PJRT_PLUGIN_PATH:+ESHKOL_PJRT_PLUGIN_PATH="$ESHKOL_PJRT_PLUGIN_PATH"} \
    ESHKOL_XLA_REGION_REPORT="$WORK/$name.report.json" \
        "$RUN" -r "$f" > "$WORK/$name.on" 2> "$WORK/$name.on.err"
    on_rc=$?

    regions="$(grep -c '"shape_signature"' "$WORK/$name.report.json" 2>/dev/null || echo 0)"
    [ "$regions" = "0" ] && no_region=$((no_region + 1))

    if [ "$off_rc" -ne 0 ] || [ "$on_rc" -ne 0 ]; then
        printf '%-42s %8s %8d %9d  FAIL (exit)\n' "$name" "$regions" "$off_rc" "$on_rc"
        failed=$((failed + 1))
        continue
    fi
    if diff -q "$WORK/$name.off" "$WORK/$name.on" > /dev/null 2>&1; then
        printf '%-42s %8s %8s %9s  PASS\n' "$name" "$regions" ok ok
        agreed=$((agreed + 1))
    else
        printf '%-42s %8s %8s %9s  FAIL (output differs)\n' "$name" "$regions" ok ok
        diff "$WORK/$name.off" "$WORK/$name.on" | head -6 | sed 's/^/      /'
        failed=$((failed + 1))
    fi
done

printf '\nwhole_program_region_parity: %s (%d of %d programs agree; %d formed no region)\n' \
    "$([ "$failed" -eq 0 ] && echo PASS || echo FAIL)" "$agreed" "$total" "$no_region"
[ "$failed" -eq 0 ]
