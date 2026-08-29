#!/usr/bin/env bash
# Measure a real consumer's cold AOT source-to-object compile.
#
# Usage:
#   bench/consumer_aot_compile_bench.sh <build-dir> <consumer-root> <work-root> [ceiling-s]
#
# The caller supplies work-root so a CI lane can place all generated files on
# its lane filesystem. The command does not link or execute the consumer; this
# isolates compiler/module emission from consumer runtime configuration.
set -u

BUILD_DIR="${1:?usage: consumer_aot_compile_bench.sh <build-dir> <consumer-root> <work-root> [ceiling-s]}"
CONSUMER_ROOT="${2:?usage: consumer_aot_compile_bench.sh <build-dir> <consumer-root> <work-root> [ceiling-s]}"
WORK_ROOT="${3:?usage: consumer_aot_compile_bench.sh <build-dir> <consumer-root> <work-root> [ceiling-s]}"
CEILING_S="${4:-300}"

ESHKOL_RUN="$BUILD_DIR/eshkol-run"
SOURCE="$CONSUMER_ROOT/src/main.esk"
OBJECT="$WORK_ROOT/consumer-main.o"
LOG="$WORK_ROOT/consumer-aot.log"

[ -x "$ESHKOL_RUN" ] || { echo "consumer_aot_compile_bench: compiler not found: $ESHKOL_RUN" >&2; exit 1; }
[ -f "$SOURCE" ] || { echo "consumer_aot_compile_bench: source not found: $SOURCE" >&2; exit 1; }
mkdir -p "$WORK_ROOT"

SECONDS=0
COMPILE_EXIT=0
ESHKOL_PATH="$CONSUMER_ROOT/lib:$BUILD_DIR/../lib" \
    /usr/bin/time -f 'max_rss_kb=%M' \
    timeout --signal=KILL "${CEILING_S}s" \
    "$ESHKOL_RUN" --emit-object -O 0 -o "$OBJECT" "$SOURCE" \
    >"$LOG" 2>&1 || COMPILE_EXIT=$?
ELAPSED_S="$SECONDS"

RSS_KB="$(sed -n 's/^max_rss_kb=//p' "$LOG" | tail -n 1)"
[ -n "$RSS_KB" ] || RSS_KB=null

if [ "$COMPILE_EXIT" -eq 124 ] || [ "$COMPILE_EXIT" -eq 137 ]; then
    printf '{"status":"fail","reason":"ceiling_exceeded","elapsed_s":%s,"ceiling_s":%s,"max_rss_kb":%s}\n' \
        "$ELAPSED_S" "$CEILING_S" "$RSS_KB"
    exit 1
fi
if [ "$COMPILE_EXIT" -ne 0 ] || [ ! -s "$OBJECT" ]; then
    printf '{"status":"fail","reason":"compile_error","exit":%s,"elapsed_s":%s,"ceiling_s":%s,"max_rss_kb":%s}\n' \
        "$COMPILE_EXIT" "$ELAPSED_S" "$CEILING_S" "$RSS_KB"
    exit 1
fi

OBJECT_BYTES="$(stat -c '%s' "$OBJECT")"
printf '{"status":"pass","elapsed_s":%s,"ceiling_s":%s,"max_rss_kb":%s,"object_bytes":%s}\n' \
    "$ELAPSED_S" "$CEILING_S" "$RSS_KB" "$OBJECT_BYTES"
