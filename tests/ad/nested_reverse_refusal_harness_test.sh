#!/bin/sh
# Ensure the refusal harness rejects a silent success and an unrelated failure.
set -eu

if [ "$#" -ne 3 ]; then
    echo "FAIL: usage: $0 HARNESS SOURCE BUILD_DIR" >&2
    exit 2
fi
harness=$1
source_file=$2
build_dir=$3
expected='reverse arithmetic: a nested reverse-mode derivative cannot consume an outer reverse-mode value; differentiate the outer level with forward mode or restructure the computation'
work=$(mktemp -d "$build_dir/nested-ad-harness.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

cat > "$work/success.sh" <<'RUNNER'
#!/bin/sh
printf '#(0 0)\n'
exit 0
RUNNER
cat > "$work/startup_failure.sh" <<'RUNNER'
#!/bin/sh
printf 'unrelated startup failure\n' >&2
exit 1
RUNNER
cat > "$work/signal_failure.sh" <<RUNNER
#!/bin/sh
printf '%s\n' '$expected' >&2
exit 139
RUNNER
chmod +x "$work/success.sh" "$work/startup_failure.sh" "$work/signal_failure.sh"

if sh "$harness" "$work/success.sh" "$source_file" "$build_dir" jit "$expected" >"$work/success.log" 2>&1; then
    echo "FAIL: harness accepted a successful zero result" >&2
    cat "$work/success.log" >&2
    exit 1
fi
if ! grep -F 'succeeded; expected refusal' "$work/success.log" >/dev/null; then
    echo "FAIL: zero-result control failed for an unrelated reason" >&2
    cat "$work/success.log" >&2
    exit 1
fi

if sh "$harness" "$work/startup_failure.sh" "$source_file" "$build_dir" jit "$expected" >"$work/startup.log" 2>&1; then
    echo "FAIL: harness accepted an unrelated startup failure" >&2
    cat "$work/startup.log" >&2
    exit 1
fi
if ! grep -F 'without the intended refusal' "$work/startup.log" >/dev/null; then
    echo "FAIL: startup-failure control failed for an unrelated reason" >&2
    cat "$work/startup.log" >&2
    exit 1
fi

if sh "$harness" "$work/signal_failure.sh" "$source_file" "$build_dir" jit "$expected" >"$work/signal.log" 2>&1; then
    echo "FAIL: harness accepted a signal-style exit with the expected diagnostic" >&2
    cat "$work/signal.log" >&2
    exit 1
fi
if ! grep -F 'signal-style code' "$work/signal.log" >/dev/null; then
    echo "FAIL: signal control failed for an unrelated reason" >&2
    cat "$work/signal.log" >&2
    exit 1
fi
printf 'PASS: refusal harness rejects zero-result, unrelated-failure, and signal controls\n'
