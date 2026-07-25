#!/usr/bin/env bash
#
# Copyright (C) tsotchke
#
# SPDX-License-Identifier: MIT
#
# Regression guard for `-g` (DWARF debug info) code generation.
#
# `-g` builds were completely broken and invisible for an entire release line.
# Every top-level define -- including the `:external` stdlib defines a
# `(require ...)` produces, whose bodies live in the pre-linked stdlib object --
# was handed a *definition* DISubprogram at declaration time. A bodyless LLVM
# function may only carry a uniqued declaration subprogram, so the module failed
# verification before a single byte of output existed:
#
#   function declaration may only have a unique !dbg attachment
#   ptr @caadr
#
# The only assertion covering `-g` at the time checked that an "-O2/-O3" marker
# was *absent* from the log -- a condition a compiler that never produces output
# satisfies for free. So this test asserts on artifacts and behaviour instead:
#
#   1. -g compiles, links, and the binary runs with the expected output, over
#      the very list accessors that used to break it (caadr/cadar/caddr/cddr);
#   2. -g composes with every explicit optimization level (-O0..-O3);
#   3. the debug info is actually *useful*, not merely present: the emitted
#      DWARF names the user's functions and carries line entries pointing at the
#      .esk source. Skipped only if no DWARF reader exists on the host.
#
# Every one of those fails loudly if the compiler stops producing output.
#
# Usage: debug_info_emission_test.sh <path-to-eshkol-run> <build-dir>

set -u

TEST_NAME="debug_info_emission_test"
ESHKOL_RUN="${1:-}"
BUILD_DIR="${2:-}"

fail() {
    echo "FAIL: $TEST_NAME: $*"
    exit 1
}

if [ -z "$ESHKOL_RUN" ] || [ ! -x "$ESHKOL_RUN" ]; then
    fail "eshkol-run not executable: '$ESHKOL_RUN'"
fi
if [ -z "$BUILD_DIR" ] || [ ! -d "$BUILD_DIR" ]; then
    fail "build directory not found: '$BUILD_DIR'"
fi

ESHKOL_RUN="$(cd "$(dirname "$ESHKOL_RUN")" && pwd)/$(basename "$ESHKOL_RUN")"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-debuginfo.XXXXXX")" || fail "mktemp failed"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

SRC="$WORK/debug_info_probe.esk"
cat > "$SRC" <<'ESK'
(define nested (list 1 (list 20 21) (list 30 (list 31 32)) 4))

(define (double x)
  (* x 2))

(define (describe lst)
  (double (car lst)))

(define (main)
  (display (caadr nested))
  (newline)
  (display (cadar (list (list 100 101) 2)))
  (newline)
  (display (car (caddr nested)))
  (newline)
  (display (cddr (list 1 2 3 4)))
  (newline)
  (display (describe (list 7 8)))
  (newline)
  0)
ESK

EXPECTED="20
101
30
(3 4)
14"

# --- 1 + 2: -g alone and -g with each explicit optimization level -----------
for opt in "" "-O0" "-O1" "-O2" "-O3"; do
    label="-g ${opt:-(default opt level)}"
    bin="$WORK/probe$(echo "${opt:-def}" | tr -d '-')"
    rm -f "$bin" "$bin.tmp.o"

    log="$WORK/compile.log"
    if ! ( cd "$WORK" && "$ESHKOL_RUN" -g $opt -L"$BUILD_DIR" "$SRC" -o "$bin" ) \
            > "$log" 2>&1; then
        echo "--- compiler output ---"
        tail -20 "$log"
        fail "'$label' failed to compile (exit non-zero)"
    fi
    [ -x "$bin" ] || fail "'$label' reported success but produced no executable"

    actual="$("$bin" 2>&1)" || fail "'$label' binary exited non-zero"
    if [ "$actual" != "$EXPECTED" ]; then
        echo "--- expected ---"; echo "$EXPECTED"
        echo "--- actual ---";   echo "$actual"
        fail "'$label' binary produced the wrong output"
    fi
    echo "  ok: $label compiles, links and runs correctly"
done

# --- 3: the debug info has to be usable ------------------------------------
# The DWARF lives in the object file the driver leaves beside the executable
# (that object is also what a debugger follows via the Mach-O debug map), so
# read whichever of the two the host's reader understands.
DWARF_TOOL=""
for candidate in llvm-dwarfdump dwarfdump; do
    if command -v "$candidate" >/dev/null 2>&1; then DWARF_TOOL="$candidate"; break; fi
done
# objdump can read the line table too, and exists on hosts with neither above.
if [ -z "$DWARF_TOOL" ] && command -v objdump >/dev/null 2>&1; then
    DWARF_TOOL="objdump"
fi

if [ -z "$DWARF_TOOL" ]; then
    echo "  skip: no DWARF reader on this host (llvm-dwarfdump/dwarfdump/objdump)"
else
    OBJ="$WORK/probedef.tmp.o"
    TARGET="$OBJ"
    [ -f "$TARGET" ] || TARGET="$WORK/probedef"
    [ -e "$TARGET" ] || fail "neither the -g object nor the executable exists to inspect"

    if [ "$DWARF_TOOL" = "objdump" ]; then
        info="$(objdump --dwarf=info "$TARGET" 2>/dev/null)"
        lines="$(objdump --dwarf=decodedline "$TARGET" 2>/dev/null)"
    else
        info="$("$DWARF_TOOL" --debug-info "$TARGET" 2>/dev/null)"
        lines="$("$DWARF_TOOL" --debug-line "$TARGET" 2>/dev/null)"
    fi

    if [ -z "$info" ]; then
        fail "no .debug_info in the -g output: debug info was not emitted at all"
    fi
    # A user function must be named in the DWARF. Present-but-useless debug info
    # (a compile unit and nothing else) fails here rather than passing quietly.
    for fn in double describe; do
        case "$info" in
            *"\"$fn\""*|*"$fn"*) ;;
            *) fail "DWARF does not name the user function '$fn'" ;;
        esac
    done
    case "$info" in
        *debug_info_probe.esk*) ;;
        *) fail "DWARF does not reference the .esk source file" ;;
    esac
    if [ -z "$lines" ]; then
        fail "no line-number program in the -g output: line attribution unusable"
    fi
    # Require actual line rows, not just an empty line-table header.
    if ! printf '%s\n' "$lines" | grep -qE '(0x0*[0-9a-f]{4,}[[:space:]]+[0-9]+[[:space:]]+[0-9]+)|(debug_info_probe\.esk[[:space:]]*:?[[:space:]]*[0-9]+)'; then
        echo "--- line table ---"
        printf '%s\n' "$lines" | head -20
        fail "line table has no source rows: -g emitted debug info that cannot attribute code"
    fi
    echo "  ok: DWARF names user functions and carries source line rows ($DWARF_TOOL)"
fi

echo "PASS: $TEST_NAME"
exit 0
