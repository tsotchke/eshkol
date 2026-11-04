#!/usr/bin/env bash
# Compile the native stream contract without the compiler/dependency aggregate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-subprocess-stream.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
case "$(uname -s)" in
    Linux) backend=epoll ;;
    Darwin|FreeBSD) backend=kqueue ;;
    *) echo 'Unsupported: native POSIX stream contract'; exit 77 ;;
esac
CC="${CC:-cc}"
CXX="${CXX:-c++}"
sanitize=()
if [ "${ESHKOL_STREAM_SANITIZE:-0}" = 1 ]; then
    sanitize=(-fsanitize=address,undefined -fno-omit-frame-pointer)
fi
flags=(-std=c11 -D_GNU_SOURCE -D_XOPEN_SOURCE=700 -O2 -g -Wall -Wextra -I"$ROOT/inc" -pthread)
if [ "$(uname -s)" = Darwin ]; then
    flags+=(-D_DARWIN_C_SOURCE=1)
fi
runtime_objects=()
for file in tests/agent/subprocess_stream_test.c lib/agent/c/agent_subprocess.c \
            lib/core/event_loop.c "lib/core/event_loop_${backend}.c"; do
    object="$WORK/$(basename "$file").o"
    "$CC" "${flags[@]}" "${sanitize[@]}" -c "$ROOT/$file" -o "$object"
    case "$file" in lib/*) runtime_objects+=("$object") ;; esac
done
"$CXX" -std=c++17 -O2 -g -Wall -Wextra -pthread "${sanitize[@]}" -I"$ROOT/inc" \
    -c "$ROOT/lib/core/runtime_handle.cpp" -o "$WORK/runtime_handle.o"
"$CXX" -pthread "${sanitize[@]}" "$WORK"/*.o -o "$WORK/subprocess-stream-test"
python3 - "$WORK/subprocess-stream-test" <<'PY'
import subprocess
import sys
raise SystemExit(subprocess.run([sys.argv[1]], timeout=45).returncode)
PY

cat > "$WORK/missing-checker.c" <<'C'
#include <eshkol/core/event_loop.h>
#include <eshkol/core/subprocess_stream.h>
#include <stdio.h>
int main(void) {
    int64_t loop = eshkol_event_loop_open_handle(8);
    if (loop <= 0) return 1;
    int64_t status = eshkol_process_stream_open(loop,
        "/definitely-unavailable-eshkol-executable", ".", NULL);
    if (status != ESHKOL_PROCESS_STREAM_EDENIED) return 2;
    if (eshkol_event_loop_close_handle(loop) != 0) return 3;
    if (eshkol_process_stream_runtime_shutdown() != 0) return 4;
    puts("PASS: missing native capability checker denies acquisition");
    return 0;
}
C
"$CC" "${flags[@]}" "${sanitize[@]}" -c "$WORK/missing-checker.c" -o "$WORK/missing-checker.o"
"$CXX" -pthread "${sanitize[@]}" "$WORK/missing-checker.o" \
    "${runtime_objects[@]}" "$WORK/runtime_handle.o" -o "$WORK/missing-checker"
"$WORK/missing-checker"

# GNU/ELF wrapping injects a clock failure without a production test hook.
if [ "$(uname -s)" = Linux ]; then
    "$CC" "${flags[@]}" "${sanitize[@]}" \
        -c "$ROOT/tests/agent/subprocess_stream_clock_test.c" -o "$WORK/clock-test.o"
    "$CXX" -pthread "${sanitize[@]}" -Wl,--wrap=clock_gettime \
        "$WORK/clock-test.o" "${runtime_objects[@]}" "$WORK/runtime_handle.o" \
        -o "$WORK/clock-test"
    python3 - "$WORK/clock-test" <<'PY'
import subprocess
import sys
raise SystemExit(subprocess.run([sys.argv[1]], timeout=5).returncode)
PY
fi
