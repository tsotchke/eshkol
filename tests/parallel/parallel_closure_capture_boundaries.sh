#!/usr/bin/env bash
# Native JIT/AOT regression for dynamically sized parallel closure environments.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
WORK_DIR="${1:?usage: $0 <work-dir>}"
TRACE_DIR="${TRACE_DIR:-$ROOT_DIR/scripts/icc_traces}"
TRACE_FILE="$TRACE_DIR/parallel_closure_capture_boundaries.jsonl"

mkdir -p "$WORK_DIR" "$TRACE_DIR"
: > "$TRACE_FILE"

python3 "$ROOT_DIR/tests/parallel/generate_closure_capture_boundaries.py" "$WORK_DIR"

for count in 31 32 33 200 4096; do
    source_file="$WORK_DIR/closure_capture_${count}.esk"
    jit_out="$WORK_DIR/jit_${count}.out"
    aot_file="$WORK_DIR/aot_${count}"
    aot_compile_out="$WORK_DIR/aot_compile_${count}.out"
    aot_out="$WORK_DIR/aot_${count}.out"

    "$ESHKOL_RUN" --no-stdlib -O0 -r "$source_file" \
        >"$jit_out" 2>"$WORK_DIR/jit_${count}.err"
    "$ESHKOL_RUN" --no-stdlib -O0 -o "$aot_file" "$source_file" \
        >"$aot_compile_out" 2>"$WORK_DIR/aot_compile_${count}.err"
    "$aot_file" >"$aot_out" 2>"$WORK_DIR/aot_${count}.err"

    expected_sum=$((count - 1))
    expected_values=""
    for input in 1 2 3 4 5 6 7 8; do
        value=$((expected_sum + input))
        if [ -n "$expected_values" ]; then
            expected_values+=" "
        fi
        expected_values+="$value"
    done
    expected_line="RESULT ${count} SERIAL=(${expected_values}) PARALLEL=(${expected_values})"

    for output in "$jit_out" "$aot_out"; do
        grep -Fqx "$expected_line" "$output"
        grep -Fqx "PASS: parallel-map" "$output"
        if [ "$count" -ne 4096 ]; then
            grep -Fqx "PASS: parallel-execute" "$output"
            grep -Fqx "PASS: parallel-for-each" "$output"
        fi
    done

    printf '{"kind":"parallel_closure_capture_boundary","name":"closure_capture_%s","value":"PASS","confidence":0.99}\n' \
        "$count" >> "$TRACE_FILE"
    echo "PASS: ${count} captures native JIT/AOT serial-parallel equality"
done

echo "PASS: parallel closure capture boundaries 31, 32, 33, 200, 4096"
