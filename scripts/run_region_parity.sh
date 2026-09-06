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

# The same search order run_xla_gate.sh uses, so a plugin found by the gate is
# found here. An empty answer leaves the device path to its own discovery,
# which then reports honestly that it found nothing.
PLUGIN="${ESHKOL_PJRT_PLUGIN_PATH:-}"
if [ -z "$PLUGIN" ]; then
    for candidate in \
        "$HOME"/.local/lib/python3.1[0-9]/site-packages/jaxlib/cpu_plugin.so \
        "$HOME"/.local/lib/python3.1[0-9]/site-packages/jax_plugins/xla_cpu/xla_cpu_pjrt_plugin.so \
        /usr/lib/pjrt/pjrt_c_api_cpu_plugin.so; do
        [ -f "$candidate" ] && PLUGIN="$candidate" && break
    done
fi

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

    # ESHKOL_JIT_CACHE=0 ON BOTH RUNS, and it is not a convenience.
    #
    # The JIT's object cache is keyed on the source, not on the settings the
    # source was compiled under, so a regions-on request was served the
    # regions-off object compiled moments earlier. The pass never ran, every
    # program reported zero regions, and all fourteen rows "agreed" — a
    # vacuous pass of exactly the kind this stage exists to make impossible.
    # Disabling the cache on both sides makes the two runs like-for-like.
    # The cache key itself should carry the region setting; that is a compiler
    # change and a separate build item.

    # Regions off: the program exactly as it has always run.
    env ESHKOL_JIT_CACHE=0 "$RUN" -r "$f" > "$WORK/$name.off" 2> "$WORK/$name.off.err"
    off_rc=$?

    # Regions on. The report tells us how many regions were rewritten, which
    # is what separates "agreed because the device got it right" from "agreed
    # because nothing went to the device".
    env ESHKOL_JIT_CACHE=0 ESHKOL_XLA_REGIONS=1 ESHKOL_XLA_PJRT=1 \
        ESHKOL_PJRT_PLUGIN_PATH="$PLUGIN" \
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
    # THE COMPARISON IS NUMERIC, NOT TEXTUAL, and the reason is in the
    # contract: the device computes in f32 (TPU has no f64) while the host
    # computes in f64, so a byte-identical stdout would only ever mean the
    # region did not run. docs/design/ESHKOL_S_FRAGMENT.md's f32 bounds apply
    # — 1e-5 for exact operations, 1e-3 for a chain containing an approximated
    # elementary function, which is the tolerance class of the loosest op in
    # the region.
    #
    # Everything that is NOT a number still has to match exactly. That is what
    # catches a structural change: a region that returned #(2.209) where the
    # subtree returned 2.209 has the same number and a different skeleton, and
    # a purely numeric comparison would have called it equal.
    tol=0.00001
    if grep -qE '"(tanh|exp|log|sin|cos|sqrt|tensor-sqrt|tensor-exp|tensor-log|sigmoid|atanh|softmax)"' \
            "$WORK/$name.report.json" 2>/dev/null; then
        tol=0.001
    fi

    sed -E 's/-?[0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?/N/g' "$WORK/$name.off" > "$WORK/$name.off.skel"
    sed -E 's/-?[0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?/N/g' "$WORK/$name.on"  > "$WORK/$name.on.skel"
    grep -oE '\-?[0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?' "$WORK/$name.off" > "$WORK/$name.off.num"
    grep -oE '\-?[0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?' "$WORK/$name.on"  > "$WORK/$name.on.num"

    if ! diff -q "$WORK/$name.off.skel" "$WORK/$name.on.skel" > /dev/null 2>&1; then
        printf '%-42s %8s %8s %9s  FAIL (output structure differs)\n' "$name" "$regions" ok ok
        diff "$WORK/$name.off.skel" "$WORK/$name.on.skel" | head -4 | sed 's/^/      /'
        failed=$((failed + 1))
        continue
    fi

    worst=$(paste "$WORK/$name.off.num" "$WORK/$name.on.num" | awk -v tol="$tol" '
        BEGIN { worst = 0; bad = 0; n = 0 }
        {
            n++
            h = $1 + 0; d = $2 + 0
            e = h - d; if (e < 0) e = -e
            a = h; if (a < 0) a = -a
            ok = (e <= tol) || (a > 0 && e <= tol * a)
            if (!ok) bad++
            r = (a > 0) ? e / a : e
            if (r > worst) worst = r
        }
        END {
            if (nlines_off != n) { }
            printf "%d %d %.3e", bad, n, worst
        }' nlines_off=$(wc -l < "$WORK/$name.off.num"))
    set -- $worst
    bad="$1"; count="$2"; worst_rel="$3"

    off_count=$(wc -l < "$WORK/$name.off.num" | tr -d " ")
    on_count=$(wc -l < "$WORK/$name.on.num" | tr -d " ")
    if [ "$off_count" != "$on_count" ]; then
        printf '%-42s %8s %8s %9s  FAIL (%s numbers off, %s on)\n' \
            "$name" "$regions" ok ok "$off_count" "$on_count"
        failed=$((failed + 1))
        continue
    fi
    if [ "${bad:-1}" = "0" ]; then
        printf '%-42s %8s %8s %9s  PASS (%s values, worst rel %s, tol %s)\n' \
            "$name" "$regions" ok ok "$count" "$worst_rel" "$tol"
        agreed=$((agreed + 1))
    else
        printf '%-42s %8s %8s %9s  FAIL (%s of %s values outside %s, worst rel %s)\n' \
            "$name" "$regions" ok ok "$bad" "$count" "$tol" "$worst_rel"
        failed=$((failed + 1))
    fi
done

printf '\nwhole_program_region_parity: %s (%d of %d programs agree; %d formed no region)\n' \
    "$([ "$failed" -eq 0 ] && echo PASS || echo FAIL)" "$agreed" "$total" "$no_region"
[ "$failed" -eq 0 ]
