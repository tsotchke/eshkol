#!/usr/bin/env bash
# Run against a source checkout containing the SSE, PROCESS and Unix socket implementations.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <composed Eshkol checkout>" >&2
    exit 2
fi
test_dir=$(cd -- "$(dirname -- "$0")" && pwd -P)
source_dir=$(cd -- "$1" && pwd -P)
case "$(uname -s)" in
    Linux) backend=epoll ;;
    Darwin) backend=kqueue ;;
    *) echo "SKIP: requires Linux epoll or macOS kqueue"; exit 77 ;;
esac
for file in inc/eshkol/agent_http_sse_adapter.h inc/eshkol/core/subprocess_stream.h \
            inc/eshkol/core/unix_socket_stream.h lib/agent/c/agent_unix_socket_stream.c; do
    if [[ ! -f "$source_dir/$file" ]]; then
        echo "SKIP: composition lacks $file"
        exit 77
    fi
done

printf 'Composed commit and tree: '
git -C "$source_dir" show -s --format='%H %T'
if ! git -C "$source_dir" diff --quiet HEAD --; then
    echo 'Composed checkout has tracked changes'
fi

# Keep the private Unix socket pathname short on both platforms.
work=$(mktemp -d /tmp/eshkol-sse-mix.XXXXXXXX)
trap 'rm -rf -- "$work"' EXIT
sandbox="$work/peer"
mkdir -m 700 "$sandbox"
CC="${CC:-clang}"
CXX="${CXX:-clang++}"
flags=(-O2 -g)
if [[ ${ESHKOL_SSE_MIXED_SANITIZE:-0} == 1 ]]; then
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    if [[ "$backend" == epoll ]]; then
        export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1:halt_on_error=1}"
    fi
    export UBSAN_OPTIONS="${UBSAN_OPTIONS:-halt_on_error=1}"
fi
flags+=(-pthread -I"$source_dir/inc")
sources=(
    "$test_dir/http_sse_mixed_loop_test.c"
    "$source_dir/lib/agent/c/agent_http_server.c"
    "$source_dir/lib/agent/c/agent_subprocess.c"
    "$source_dir/lib/agent/c/agent_unix_socket_stream.c"
    "$source_dir/lib/core/resource_lease.c"
    "$source_dir/lib/core/event_loop.c"
    "$source_dir/lib/core/event_loop_${backend}.c"
)
objects=()
for file in "${sources[@]}"; do
    object="$work/$(basename "$file").o"
    "$CC" -std=gnu11 "${flags[@]}" -DESHKOL_SSE_ADAPTER_TESTING -c "$file" -o "$object"
    objects+=("$object")
done
"$CXX" -std=c++17 "${flags[@]}" -c "$source_dir/lib/core/runtime_handle.cpp" \
    -o "$work/runtime_handle.o"
"$CXX" "${flags[@]}" "${objects[@]}" "$work/runtime_handle.o" -o "$work/verify"
python3 - "$work/verify" "$sandbox" <<'PY'
import subprocess
import sys
raise SystemExit(subprocess.run(sys.argv[1:], timeout=40).returncode)
PY
