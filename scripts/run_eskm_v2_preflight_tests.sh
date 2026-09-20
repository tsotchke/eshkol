#!/usr/bin/env bash
# Private parser tests: C17 + C++ consumers, no Eshkol/LLVM build required.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d "${TMPDIR:-/tmp}/eskm-v2-preflight.XXXXXX")"
trap 'rm -rf -- "$work"' EXIT
cc="${CC:-cc}"
cxx="${CXX:-c++}"
python="${PYTHON:-python3}"
nm="${NM:-nm}"
flags=(-Wall -Wextra -Werror -pedantic -I"$root/lib/core")
sanitizers=()
case "${ESKM_V2_SANITIZE:-0}" in
    0) ;;
    1) sanitizers=(-fsanitize=address,undefined -fno-omit-frame-pointer
                   -fno-sanitize-recover=all) ;;
    *) echo 'ESKM_V2_SANITIZE must be 0 or 1' >&2; exit 2 ;;
esac

# Audit the uninstrumented object: only byte comparison/copying/zeroing and the
# compiler's stack protector may be external. Together with the reviewed
# nonrecursive source, this checks that validation has no allocator dependency.
"$cc" -std=c17 "${flags[@]}" -O0 -c "$root/lib/core/eskm_v2_preflight.c" \
    -o "$work/import-audit.o"
"$nm" -u "$work/import-audit.o" > "$work/imports.txt"
"$python" - "$work/imports.txt" <<'PY'
from pathlib import Path
import sys

allowed = {"memcmp", "bcmp", "memcpy", "memmove", "memset",
           "__memset_chk", "_memset_chk", "memset_chk",
           "__stack_chk_fail", "__stack_chk_fail_local"}
for line in Path(sys.argv[1]).read_text().splitlines():
    symbol = line.split()[-1]
    if symbol not in allowed and symbol.removeprefix("_") not in allowed:
        raise SystemExit(f"Unexpected parser import: {symbol}")
print("PASS: parser object has no allocator or Eshkol runtime imports")
PY

"$cc" -std=c17 "${flags[@]}" "${sanitizers[@]-}" -O1 -g \
    -c "$root/lib/core/eskm_v2_preflight.c" -o "$work/parser.o"
"$cc" -std=c17 "${flags[@]}" "${sanitizers[@]-}" -O1 -g \
    "$root/tests/core/eskm_v2_preflight_test.c" "$work/parser.o" \
    -o "$work/preflight-test"
"$cxx" -std=c++17 "${flags[@]-}" "${sanitizers[@]-}" -O1 -g \
    "$root/tests/core/eskm_v2_preflight_cpp_test.cpp" "$work/parser.o" \
    -o "$work/cpp-test"

fixtures="$root/tests/core/fixtures/eskm-v2"
"$work/preflight-test" "$fixtures"
"$work/cpp-test"
"$python" "$root/scripts/check_eskm_v2_fixtures.py" --self-test \
    --fixtures "$fixtures" --test-executable "$work/preflight-test"
