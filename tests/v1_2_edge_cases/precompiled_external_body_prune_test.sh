#!/usr/bin/env bash
# ESH-0088: precompiled-module external declarations must not typecheck bodies.
set -euo pipefail

ESHKOL_RUN="${1:-${ESHKOL_RUN:-}}"
if [ -z "$ESHKOL_RUN" ]; then
    if [ -x "./build/eshkol-run" ]; then
        ESHKOL_RUN="./build/eshkol-run"
    elif [ -x "./build-verify/eshkol-run" ]; then
        ESHKOL_RUN="./build-verify/eshkol-run"
    else
        echo "FAIL: precompiled_external_body_prune_test could not locate eshkol-run" >&2
        exit 1
    fi
fi

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "FAIL: precompiled_external_body_prune_test eshkol-run is not executable: $ESHKOL_RUN" >&2
    exit 1
fi

case "$ESHKOL_RUN" in
    /*) ;;
    */*) ESHKOL_RUN="$(cd "$(dirname "$ESHKOL_RUN")" && pwd)/$(basename "$ESHKOL_RUN")" ;;
    *) ESHKOL_RUN="$(command -v "$ESHKOL_RUN")" ;;
esac

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-external-prune.XXXXXX")"
case "$tmpdir" in
    "${TMPDIR:-/tmp}"/eshkol-external-prune.*) ;;
    *) echo "FAIL: unsafe tmpdir: $tmpdir" >&2; exit 1 ;;
esac
trap 'rm -rf "$tmpdir"' EXIT

mkdir -p "$tmpdir/lib/core"

fixture_runner="$tmpdir/eshkol-run"
cp "$ESHKOL_RUN" "$fixture_runner"
chmod +x "$fixture_runner"

cat > "$tmpdir/lib/stdlib.esk" <<'EOF_STDLIB'
(require core.external_probe)
EOF_STDLIB

cat > "$tmpdir/lib/core/external_probe.esk" <<'EOF_MODULE'
(provide external-bad-body)

(define (external-bad-body x)
  (+ x "not-a-number"))
EOF_MODULE

cat > "$tmpdir/use_external_probe.esk" <<'EOF_USE'
(require core.external_probe)
(define (main) (external-bad-body 1))
EOF_USE

# The filename is what marks the library as precompiled; compile-only mode does
# not link this dummy object.
: > "$tmpdir/stdlib.o"

set +e
(
    cd "$tmpdir"
    "$fixture_runner" --strict-types --compile-only \
        use_external_probe.esk stdlib.o \
        -o use_external_probe.o
) > "$tmpdir/compile.log" 2>&1
status=$?
set -e

if [ "$status" -ne 0 ]; then
    echo "FAIL: external precompiled body was scanned during compile-only" >&2
    cat "$tmpdir/compile.log" >&2
    exit 1
fi

if grep -Eq "not-a-number|external-bad-body.*body type|Type errors detected" "$tmpdir/compile.log"; then
    echo "FAIL: external precompiled body leaked into typechecking diagnostics" >&2
    cat "$tmpdir/compile.log" >&2
    exit 1
fi

if [ ! -s "$tmpdir/use_external_probe.o" ]; then
    echo "FAIL: compile-only did not produce an object" >&2
    cat "$tmpdir/compile.log" >&2
    exit 1
fi

echo "PASS: precompiled_external_body_prune_test"
