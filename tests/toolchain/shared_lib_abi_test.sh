#!/usr/bin/env bash
#
# Copyright (C) tsotchke
#
# SPDX-License-Identifier: MIT
#
# ABI contract for `--shared-lib`: a function exported by an Eshkol shared
# library must be callable across the REAL platform C ABI.
#
# THE DEFECT THIS GUARDS
#
# `eshkol_tagged_value_t` is a 16-byte struct, and Eshkol's internal calling
# convention returns and passes it as an LLVM first-class struct of five fields
# {i8, i8, i16, i32, i64}. That is not how a C compiler passes the same struct.
# The backend flattens the aggregate into ONE RETURN REGISTER PER FIELD:
#
#     _abi-int:  mov w0, #1      ; type
#                mov w1, #0x10   ; flags   <- a C caller reads this as the payload
#                mov w2, #0      ; reserved
#                mov w3, #0      ; padding
#                mov w4, #42     ; data    <- the payload, stranded in x4
#                ret
#
# while a C caller compiled against eshkol_tagged_value_t follows AAPCS/SysV and
# reads the value out of x0:x1 (rax:rdx). So `(define (abi-int) 42)` returned
# {type=1, flags=0x00, data=16} -- the flags byte masquerading as the payload --
# through ctypes and through a compiled C harness alike, at every -O level.
# Arguments were corrupted the same way: with two tagged parameters the
# flattened fields overflow the argument registers and the callee reads its
# second argument off the stack.
#
# The fix coerces the EXPORT BOUNDARY only (llvm_codegen.cpp
# emitSharedLibraryExportWrappers): the exported name becomes a thunk with the
# platform C signature and the body moves to `<name>__eshkol_internal_abi`.
#
# WHAT THIS TEST ASSERTS
#
#   1. Every documented return shape survives the boundary -- exact integer,
#      a second integer, an inexact double, and a heap string pointer -- with
#      the correct type tag AND flags AND payload, not just a plausible number.
#   2. ARGUMENTS survive it too: a two-tagged-argument function is called with
#      C-constructed tagged values and must return their real sum.
#   3. Both callers agree: a clang-compiled harness using the real
#      `eshkol_tagged_value_t` from inc/eshkol/eshkol.h, and python ctypes.
#   4. At -O0 and -O2 (the defect was optimizer-independent, and so is the fix).
#   5. NEGATIVE DIRECTION: the same harness, pointed at the unwrapped
#      `<name>__eshkol_internal_abi` symbol that still speaks the internal
#      convention, must observe the corruption. If it does not, the harness
#      cannot see the defect and assertions 1-4 prove nothing.
#
# Usage: shared_lib_abi_test.sh <path-to-eshkol-run> <build-dir> [source-root]

set -u

TEST_NAME="shared_lib_abi_test"
ESHKOL_RUN="${1:-}"
BUILD_DIR="${2:-}"
SOURCE_ROOT="${3:-}"

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

# Pin the compiler binary and the build tree by absolute path: the test cds into
# its own scratch directory and must not resolve either through $PWD.
ESHKOL_RUN="$(cd "$(dirname "$ESHKOL_RUN")" && pwd)/$(basename "$ESHKOL_RUN")"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"

if [ -z "$SOURCE_ROOT" ]; then
    SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
else
    SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd)"
fi
INC_DIR="$SOURCE_ROOT/inc"
[ -f "$INC_DIR/eshkol/eshkol.h" ] || fail "public header not found under '$INC_DIR'"

case "$(uname -s)" in
    Darwin) LIB_EXT="dylib" ;;
    *)      LIB_EXT="so" ;;
esac

# Both consumers are REQUIRED, and so is `nm`. This test's entire claim is that
# two independent callers -- a compiled C consumer and a runtime FFI consumer --
# agree with each other across the boundary. One of them silently sitting out
# would leave that claim half-checked while still printing PASS, which is the
# failure mode the defect itself had. A missing tool is a broken environment,
# reported as such.
CC="${CC:-cc}"
command -v "$CC" >/dev/null 2>&1 || fail "no C compiler ('$CC') to build the ABI harness"
command -v nm >/dev/null 2>&1 \
    || fail "no 'nm': cannot verify which symbols the library exports"

PYTHON=""
for candidate in "${PYTHON:-}" python3 python; do
    [ -n "$candidate" ] || continue
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
done
[ -n "$PYTHON" ] \
    || fail "no python3: the ctypes leg is a required second consumer, not an optional one"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-sharedlibabi.XXXXXX")" || fail "mktemp failed"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

# ── the library under test ────────────────────────────────────────────────────
# One function per return shape the ABI has to carry, plus one that takes two
# tagged values by value so the ARGUMENT direction is covered too.
cat > "$WORK/abi_module.esk" <<'ESK'
(define (abi-int) 42)
(define (abi-int-2) 777)
(define (abi-real) 3.5)
(define (abi-text) "eshkol-abi")
(define (abi-add a b) (+ a b))
ESK

# ── the C harness ─────────────────────────────────────────────────────────────
# Uses the real public struct, so it is compiled with exactly the ABI any
# consumer gets. `expect_*` checks type, flags and payload -- a harness that
# only checked the payload would have passed the days when flags was garbage.
cat > "$WORK/abi_harness.c" <<'CSRC'
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <dlfcn.h>
#include "eshkol/eshkol.h"

typedef eshkol_tagged_value_t tv_t;

static int failures = 0;

static void report(const char* label, tv_t v) {
    printf("    %-14s type=%-3u flags=0x%02x data.int=%-20" PRId64
           " data.double=%g data.ptr=0x%" PRIx64 "\n",
           label, (unsigned)v.type, (unsigned)v.flags, v.data.int_val,
           v.data.double_val, v.data.ptr_val);
}

static void expect_int(const char* label, tv_t v, int64_t want) {
    report(label, v);
    if (v.type != ESHKOL_VALUE_INT64) {
        printf("    MISMATCH %s: type %u, want %u\n", label,
               (unsigned)v.type, (unsigned)ESHKOL_VALUE_INT64);
        failures++;
    }
    if (v.flags != ESHKOL_VALUE_EXACT_FLAG) {
        printf("    MISMATCH %s: flags 0x%02x, want 0x%02x\n", label,
               (unsigned)v.flags, (unsigned)ESHKOL_VALUE_EXACT_FLAG);
        failures++;
    }
    if (v.data.int_val != want) {
        printf("    MISMATCH %s: data %" PRId64 ", want %" PRId64 "\n",
               label, v.data.int_val, want);
        failures++;
    }
}

static void expect_double(const char* label, tv_t v, double want) {
    report(label, v);
    if (v.type != ESHKOL_VALUE_DOUBLE) {
        printf("    MISMATCH %s: type %u, want %u\n", label,
               (unsigned)v.type, (unsigned)ESHKOL_VALUE_DOUBLE);
        failures++;
    }
    if (v.flags != ESHKOL_VALUE_INEXACT_FLAG) {
        printf("    MISMATCH %s: flags 0x%02x, want 0x%02x\n", label,
               (unsigned)v.flags, (unsigned)ESHKOL_VALUE_INEXACT_FLAG);
        failures++;
    }
    if (v.data.double_val != want) {
        printf("    MISMATCH %s: data %g, want %g\n", label,
               v.data.double_val, want);
        failures++;
    }
}

static void expect_string(const char* label, tv_t v, const char* want) {
    report(label, v);
    if (!ESHKOL_IS_STRING_COMPAT(v)) {
        printf("    MISMATCH %s: not a string (type=%u ptr=0x%" PRIx64 ")\n",
               label, (unsigned)v.type, v.data.ptr_val);
        failures++;
        return;
    }
    const char* text = (const char*)(uintptr_t)v.data.ptr_val;
    if (strcmp(text, want) != 0) {
        printf("    MISMATCH %s: '%s', want '%s'\n", label, text, want);
        failures++;
    }
}

static void* need(void* handle, const char* name) {
    void* sym = dlsym(handle, name);
    if (!sym) {
        printf("    MISSING symbol '%s': %s\n", name, dlerror());
        failures++;
    }
    return sym;
}

int main(int argc, char** argv) {
    /* Unbuffered: this harness calls into a freshly dlopened library across a
     * hand-written ABI boundary, so a crash is a REPORTABLE OUTCOME, not an
     * accident. With the default block buffering on a pipe, a segfault takes
     * every line printed so far down with it and the log shows only that the
     * process died -- which of the calls died is exactly what you need to
     * know, and exactly what you lose. */
    setvbuf(stdout, NULL, _IONBF, 0);

    if (argc < 2) { printf("usage: abi_harness <library>\n"); return 2; }

    void* handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!handle) { printf("    dlopen failed: %s\n", dlerror()); return 2; }

    void (*lib_init)(void*) = (void (*)(void*))need(handle, "__eshkol_lib_init__");
    void* (*global_arena)(void) = (void* (*)(void))need(handle, "get_global_arena");
    if (failures) return 2;
    lib_init(global_arena());

    tv_t (*abi_int)(void)   = (tv_t (*)(void))need(handle, "abi-int");
    tv_t (*abi_int_2)(void) = (tv_t (*)(void))need(handle, "abi-int-2");
    tv_t (*abi_real)(void)  = (tv_t (*)(void))need(handle, "abi-real");
    tv_t (*abi_text)(void)  = (tv_t (*)(void))need(handle, "abi-text");
    tv_t (*abi_add)(tv_t, tv_t) = (tv_t (*)(tv_t, tv_t))need(handle, "abi-add");
    if (failures) return 2;

    printf("  returns:\n");
    expect_int("abi-int", abi_int(), 42);
    expect_int("abi-int-2", abi_int_2(), 777);
    expect_double("abi-real", abi_real(), 3.5);
    expect_string("abi-text", abi_text(), "eshkol-abi");

    printf("  tagged arguments:\n");
    expect_int("add(20,22)", abi_add(eshkol_make_int64(20, true),
                                     eshkol_make_int64(22, true)), 42);
    expect_int("add(1000,-7)", abi_add(eshkol_make_int64(1000, true),
                                       eshkol_make_int64(-7, true)), 993);
    expect_int("add(0,777)", abi_add(eshkol_make_int64(0, true),
                                     eshkol_make_int64(777, true)), 777);

    /* NEGATIVE DIRECTION -- see the header comment. The unwrapped body still
     * speaks Eshkol's internal first-class-struct convention. Calling it
     * through the C ABI must NOT produce 42: that corruption is the defect,
     * and a harness that cannot observe it cannot certify the fix either. */
    printf("  negative control (unwrapped internal-ABI symbol):\n");
    tv_t (*raw_int)(void) =
        (tv_t (*)(void))dlsym(handle, "abi-int__eshkol_internal_abi");
    if (!raw_int) {
        printf("    MISSING abi-int__eshkol_internal_abi: the export thunk is "
               "not in place at all\n");
        failures++;
    } else {
        tv_t raw = raw_int();
        report("raw abi-int", raw);
        if (raw.type == ESHKOL_VALUE_INT64 &&
            raw.flags == ESHKOL_VALUE_EXACT_FLAG &&
            raw.data.int_val == 42) {
            printf("    NEGATIVE CONTROL FAILED: the internal-convention "
                   "symbol also read back correctly, so this harness cannot "
                   "detect the ABI defect it is meant to guard\n");
            failures++;
        } else {
            printf("    ok: internal-convention symbol reads back corrupt "
                   "(type=%u flags=0x%02x data=%" PRId64 ") as it must\n",
                   (unsigned)raw.type, (unsigned)raw.flags, raw.data.int_val);
        }
    }

    dlclose(handle);
    if (failures) {
        printf("  %d ABI mismatch(es)\n", failures);
        return 1;
    }
    printf("  all C-ABI checks agree\n");
    return 0;
}
CSRC

# ── the ctypes harness ────────────────────────────────────────────────────────
# An independent consumer: no Eshkol headers, no compiler, just the platform
# ABI as ctypes implements it. Note the subscript form lib["__eshkol_lib_init__"]
# -- getattr() on a dunder name is swallowed by Python's attribute protocol.
cat > "$WORK/abi_harness.py" <<'PYSRC'
import ctypes
import sys

EXACT_FLAG = 0x10
INEXACT_FLAG = 0x20
VALUE_INT64 = 1
VALUE_DOUBLE = 2
VALUE_HEAP_PTR = 8


class Data(ctypes.Union):
    _fields_ = [("int_val", ctypes.c_int64),
                ("double_val", ctypes.c_double),
                ("ptr_val", ctypes.c_uint64)]


class Tagged(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint8),
                ("flags", ctypes.c_uint8),
                ("reserved", ctypes.c_uint16),
                ("data", Data)]


def make_int(value):
    tv = Tagged()
    tv.type = VALUE_INT64
    tv.flags = EXACT_FLAG
    tv.reserved = 0
    tv.data.int_val = value
    return tv


failures = []


def report(label, tv):
    print("    %-14s type=%-3d flags=0x%02x data.int=%-20d data.double=%g"
          % (label, tv.type, tv.flags, tv.data.int_val, tv.data.double_val))


def expect(label, tv, want_type, want_flags, check, want_repr):
    report(label, tv)
    if tv.type != want_type:
        failures.append("%s: type %d, want %d" % (label, tv.type, want_type))
    if tv.flags != want_flags:
        failures.append("%s: flags 0x%02x, want 0x%02x"
                        % (label, tv.flags, want_flags))
    if not check(tv):
        failures.append("%s: payload wrong, want %s" % (label, want_repr))


def main(path):
    lib = ctypes.CDLL(path)

    # Subscript, not getattr: ctypes' __getattr__ never sees dunder names.
    lib_init = lib["__eshkol_lib_init__"]
    lib_init.argtypes = [ctypes.c_void_p]
    lib_init.restype = None
    arena = lib["get_global_arena"]
    arena.restype = ctypes.c_void_p
    lib_init(arena())

    def fn(name, argtypes=()):
        f = lib[name]
        f.argtypes = list(argtypes)
        f.restype = Tagged
        return f

    print("  returns:")
    expect("abi-int", fn("abi-int")(), VALUE_INT64, EXACT_FLAG,
           lambda t: t.data.int_val == 42, 42)
    expect("abi-int-2", fn("abi-int-2")(), VALUE_INT64, EXACT_FLAG,
           lambda t: t.data.int_val == 777, 777)
    expect("abi-real", fn("abi-real")(), VALUE_DOUBLE, INEXACT_FLAG,
           lambda t: t.data.double_val == 3.5, 3.5)

    text = fn("abi-text")()
    report("abi-text", text)
    if text.type != VALUE_HEAP_PTR:
        failures.append("abi-text: type %d, want %d" % (text.type, VALUE_HEAP_PTR))
    elif not text.data.ptr_val:
        failures.append("abi-text: null string pointer")
    else:
        chars = ctypes.string_at(ctypes.c_void_p(text.data.ptr_val), 10)
        if chars != b"eshkol-abi":
            failures.append("abi-text: %r, want b'eshkol-abi'" % chars)

    print("  tagged arguments:")
    add = fn("abi-add", (Tagged, Tagged))
    expect("add(20,22)", add(make_int(20), make_int(22)), VALUE_INT64,
           EXACT_FLAG, lambda t: t.data.int_val == 42, 42)
    expect("add(1000,-7)", add(make_int(1000), make_int(-7)), VALUE_INT64,
           EXACT_FLAG, lambda t: t.data.int_val == 993, 993)
    expect("add(0,777)", add(make_int(0), make_int(777)), VALUE_INT64,
           EXACT_FLAG, lambda t: t.data.int_val == 777, 777)

    # Negative direction: the unwrapped internal-convention symbol must look
    # corrupt from here too.
    print("  negative control (unwrapped internal-ABI symbol):")
    try:
        raw_fn = fn("abi-int__eshkol_internal_abi")
    except (AttributeError, OSError):
        failures.append("negative control: no abi-int__eshkol_internal_abi "
                        "symbol -- the export thunk is not in place at all")
        raw_fn = None
    if raw_fn is not None:
        raw = raw_fn()
        report("raw abi-int", raw)
        if (raw.type == VALUE_INT64 and raw.flags == EXACT_FLAG
                and raw.data.int_val == 42):
            failures.append("negative control: the internal-convention symbol "
                            "read back correctly, so ctypes cannot detect the "
                            "defect")
        else:
            print("    ok: internal-convention symbol reads back corrupt "
                  "(type=%d flags=0x%02x data=%d) as it must"
                  % (raw.type, raw.flags, raw.data.int_val))

    if failures:
        for message in failures:
            print("    MISMATCH %s" % message)
        return 1
    print("  all ctypes checks agree")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
PYSRC

"$CC" -O0 -I"$INC_DIR" "$WORK/abi_harness.c" -o "$WORK/abi_harness" \
    $( [ "$(uname -s)" = "Darwin" ] || echo -ldl ) \
    > "$WORK/harness_build.log" 2>&1 \
    || { sed -n '1,40p' "$WORK/harness_build.log"; fail "could not build the C ABI harness"; }

for opt in 0 2; do
    echo "--- -O$opt ---"
    lib_stem="abi_O$opt"
    build_log="$WORK/build_O$opt.log"
    ( cd "$WORK" && "$ESHKOL_RUN" "abi_module.esk" --shared-lib "-O$opt" \
        -o "$lib_stem" ) > "$build_log" 2>&1
    status=$?
    lib_path="$WORK/lib${lib_stem}.${LIB_EXT}"
    if [ $status -ne 0 ]; then
        sed -n '1,40p' "$build_log"
        # Name the one non-ABI cause that looks identical from here: the Eshkol
        # runtime archive is not built position-independent, so an ELF linker
        # refuses to put it in a -shared output. That is a build-configuration
        # defect, not an export-ABI one, and it must not be misread as this
        # test's subject.
        if grep -qE "recompile with -fPIC|can not be used when making a shared object" \
               "$build_log"; then
            fail "--shared-lib -O$opt could not be linked: the runtime archive is not position-independent (relocation rejected in a -shared link). This is a BUILD CONFIGURATION defect, not the export ABI: the runtime objects need POSITION_INDEPENDENT_CODE"
        fi
        fail "--shared-lib -O$opt exited $status"
    fi
    [ -f "$lib_path" ] || {
        sed -n '1,40p' "$build_log"
        fail "--shared-lib -O$opt produced no library at '$lib_path'"
    }

    # The export thunk must be the symbol a host finds under the source name,
    # and the unwrapped body must still be there under its decorated name.
    syms="$(nm -g "$lib_path" 2>/dev/null || true)"
    [ -n "$syms" ] || fail "-O$opt: nm read no symbols out of '$lib_path'"
    for want in "abi-int" "abi-add" "abi-int__eshkol_internal_abi" \
                "abi-add__eshkol_internal_abi" "__eshkol_lib_init__"; do
        case "$syms" in
            *"$want"*) ;;
            *) fail "-O$opt: '$want' is not exported: the C ABI export thunk was not emitted as specified" ;;
        esac
    done

    "$WORK/abi_harness" "$lib_path" || fail "-O$opt: C harness rejected the library ABI"
    "$PYTHON" "$WORK/abi_harness.py" "$lib_path" \
        || fail "-O$opt: ctypes harness rejected the library ABI"
done

# The relocatable object flavour must NOT be wrapped: those objects are linked
# into other Eshkol modules that call with the internal convention.
( cd "$WORK" && "$ESHKOL_RUN" "abi_module.esk" --shared-lib -c -o "obj_flavour.o" ) \
    > "$WORK/build_obj.log" 2>&1 \
    || { sed -n '1,40p' "$WORK/build_obj.log"; fail "--shared-lib -c failed"; }
[ -f "$WORK/obj_flavour.o" ] || fail "--shared-lib -c produced no object"
obj_syms="$(nm -g "$WORK/obj_flavour.o" 2>/dev/null || true)"
[ -n "$obj_syms" ] || fail "nm read no symbols out of the --shared-lib -c object"
case "$obj_syms" in
    *__eshkol_internal_abi*)
        fail "--shared-lib -c object carries C-ABI export thunks; Eshkol consumers of that object call with the internal convention" ;;
esac
case "$obj_syms" in
    *abi-int*) ;;
    *) fail "--shared-lib -c object does not export 'abi-int' at all" ;;
esac
echo "  ok: --shared-lib -c object keeps the internal convention (no thunks)"

echo "PASS: $TEST_NAME"
exit 0
