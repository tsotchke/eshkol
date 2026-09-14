#!/usr/bin/env bash
# Fault-inject expected- and actual-transcript normalization failures. Neither
# may collapse to an empty string and accidentally compare equal.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUNNER="$REPO_ROOT/scripts/run_continuation_tests.sh"
TEMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/continuation-normalizer-test.XXXXXX")
trap 'rm -rf "$TEMP_ROOT"' EXIT

FAKE_BUILD="$TEMP_ROOT/build"
mkdir -p "$FAKE_BUILD"

cat > "$FAKE_BUILD/eshkol-run" <<'SH'
#!/bin/sh
set -eu
repo=$ESHKOL_CONT_TEST_REPO_ROOT
if [ "$1" = "-r" ]; then
    fixture=${2##*/}
    fixture=${fixture%.esk}
    cat "$repo/tests/continuations/expected/$fixture.txt"
elif [ "$1" = "-o" ]; then
    exe=$2
    fixture=${3##*/}
    fixture=${fixture%.esk}
    {
        printf '%s\n' '#!/bin/sh'
        printf 'cat "$ESHKOL_CONT_TEST_REPO_ROOT/tests/continuations/expected/%s.txt"\n' "$fixture"
    } > "$exe"
    chmod +x "$exe"
else
    exit 2
fi
SH

cat > "$FAKE_BUILD/eshkol-vm-standalone-test" <<'SH'
#!/bin/sh
set -eu
fixture=${1##*/}
fixture=${fixture%.esk}
cat "$ESHKOL_CONT_TEST_REPO_ROOT/tests/continuations/expected/$fixture.txt"
SH
chmod +x "$FAKE_BUILD/eshkol-run" "$FAKE_BUILD/eshkol-vm-standalone-test"

cat > "$TEMP_ROOT/fail-normalizer" <<'SH'
#!/bin/sh
set -eu
count=0
if [ -f "$ESHKOL_CONT_NORMALIZER_COUNT" ]; then
    count=$(cat "$ESHKOL_CONT_NORMALIZER_COUNT")
fi
count=$((count + 1))
printf '%s\n' "$count" > "$ESHKOL_CONT_NORMALIZER_COUNT"
if [ "${ESHKOL_CONT_NORMALIZER_FAIL_CALL:-all}" = all ] ||
   [ "$count" -eq "$ESHKOL_CONT_NORMALIZER_FAIL_CALL" ]; then
    exit 23
fi
cat
SH
chmod +x "$TEMP_ROOT/fail-normalizer"

fixture_count=$(find "$REPO_ROOT/tests/continuations" -maxdepth 1 -name '*.esk' | wc -l | tr -d ' ')

run_injected_case() {
    local mode=$1 fail_call=$2 output="$TEMP_ROOT/$1.log" count_file="$TEMP_ROOT/$1.count"
    if ESHKOL_CONT_TEST_REPO_ROOT="$REPO_ROOT" \
       ESHKOL_CONT_NORMALIZER="$TEMP_ROOT/fail-normalizer" \
       ESHKOL_CONT_NORMALIZER_COUNT="$count_file" \
       ESHKOL_CONT_NORMALIZER_FAIL_CALL="$fail_call" \
       BUILD_DIR="$FAKE_BUILD" \
       ESHKOL_CONT_WORK="$TEMP_ROOT/work-$mode" \
       bash "$RUNNER" > "$output" 2>&1; then
        echo "normalizer_failure_test: $mode injection unexpectedly passed" >&2
        cat "$output" >&2
        return 1
    fi
    printf '%s\n' "$output"
}

expected_log=$(run_injected_case expected all)
grep -q 'FAILED .*::expected-normalization' "$expected_log"
if grep -q '^PASSED ' "$expected_log"; then
    echo "normalizer_failure_test: expected injection emitted a false PASSED" >&2
    cat "$expected_log" >&2
    exit 1
fi
grep -q "continuations: 0 passed, $fixture_count failed" "$expected_log"

# Let the expected transcript normalize, then fail the first actual transcript.
actual_log=$(run_injected_case actual 2)
grep -q 'FAILED .*::native-jit::transcript-normalization' "$actual_log"
first_fixture=$(find "$REPO_ROOT/tests/continuations" -maxdepth 1 -name '*.esk' | sort | head -1)
first_label=${first_fixture#"$REPO_ROOT/"}
first_label=${first_label%.esk}
if grep -q "^PASSED $first_label::native-jit$" "$actual_log"; then
    echo "normalizer_failure_test: actual injection emitted a false native JIT PASSED" >&2
    cat "$actual_log" >&2
    exit 1
fi
grep -q "continuations: $((fixture_count * 3 - 1)) passed, 1 failed" "$actual_log"

echo "normalizer_failure_test.sh: PASS"
