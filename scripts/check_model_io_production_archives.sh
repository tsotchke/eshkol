#!/usr/bin/env bash
# Prove test-enabled builds still ship production model-I/O archive objects.

set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 LIBESHKOL_RUNTIME LIBESHKOL_STATIC" >&2
    exit 2
fi

TMP_PARENT="${ESHKOL_TEST_TMPDIR:-${TMPDIR:-/tmp}}"
WORK="$(mktemp -d "$TMP_PARENT/eshkol-model-io-archive.XXXXXX")" || exit 2
cleanup() {
    local rc=$?
    trap - EXIT
    rm -rf -- "$WORK"
    return "$rc"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

for archive in "$@"; do
    if [ ! -f "$archive" ]; then
        echo "FAIL: archive not found: $archive" >&2
        exit 1
    fi
    if strings "$archive" | grep -Fq 'ESHKOL_TEST_MODEL_IO_FAIL'; then
        echo "FAIL: test failpoint string present in $archive" >&2
        exit 1
    fi

    member="$(ar t "$archive" | grep -E '(^|/)model_io_atomic\.c\.o$' | head -n 1)"
    if [ -z "$member" ]; then
        echo "FAIL: model_io_atomic object missing from $archive" >&2
        exit 1
    fi
    object="$WORK/$(basename "$archive").model_io_atomic.o"
    ar p "$archive" "$member" >"$object"
    if strings "$object" | grep -Fq 'ESHKOL_TEST_MODEL_IO_FAIL' ||
       nm -u "$object" | awk '{print $NF}' | grep -Eq '^_?getenv$'; then
        echo "FAIL: production model_io_atomic object contains environment failpoints: $archive" >&2
        exit 1
    fi
done

echo "PASS: production archives contain no model-I/O environment failpoints"
