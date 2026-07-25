#!/usr/bin/env bash
# run_wasm_differential.sh — WASM execute-and-diff differential lane.
#
# Closes the WASM-correctness cell in the assurance matrix: until now
# scripts/run_web_tests.sh only asserted that a produced .wasm is a valid
# binary (magic bytes) — nothing ever EXECUTED Eshkol-compiled WASM and
# diffed its output against native.  This lane does.
#
# WHAT COMPILES TO WASM
#   The faithful "Eshkol in WASM" surface is the bytecode VM
#   (lib/backend/eshkol_vm.c) compiled to WebAssembly via Emscripten from
#   lib/backend/vm_wasm_repl.c — the same module that powers the browser
#   REPL.  Its `run_program` export runs a whole program in BATCH mode
#   (compile prelude + program, execute, NO REPL auto-print), identical to
#   `eshkol-vm-standalone <file>` and `eshkol-run -r <file>`.  Crucially the
#   VM's own C `display`/`write`/number-formatting is what gets compiled to
#   WASM, so the captured stdout is a genuine product of WASM execution —
#   NOT a JS re-implementation of Eshkol formatting.  (The native-codegen
#   `--wasm` path routes display through `eshkol_display_value`, a no-op
#   stub in the browser glue, so it produces no diffable text and is not a
#   candidate for an output-diff lane.)
#
# AXES (per corpus program)
#   native   $BUILD_DIR/eshkol-run -r <file>            (native LLVM codegen)
#   wasm     node scripts/lib/wasm_diff_runner.js <mod> <file>
#              where <mod> is the Emscripten VM built fresh from current
#              source by this script (so a VM regression that only manifests
#              in WASM is caught — a stale checked-in artifact would test
#              nothing about current code).
#
# NORMALIZATION
#   Reuses the vm-parity normalization: strip banner/log lines, then remove
#   ALL newline characters from both sides before byte comparison.  The VM's
#   `display` appends a newline per call (filed:
#   tests/vm_parity/found/display_newline_per_call.esk) and Emscripten
#   line-buffers stdout, so newline PLACEMENT is not comparable — but every
#   other byte is.  Float text is compared RAW (newline stripping never
#   touches digits): a shortest-round-trip vs 17-sig-fig divergence, a
#   dropped value, or a fabricated value all still surface.
#
# SUBSET
#   Supported subset = tests/vm_parity/corpus/ — the curated set the VM
#   verifiably supports (its native-VM output already matches native codegen,
#   gated by scripts/run_vm_parity.sh).  Over that subset, native codegen ==
#   native VM, so any wasm-VM-vs-native VALUE divergence isolates the WASM
#   compilation as the cause.  With WASM_DIFF_FULL=1 the lane additionally
#   sweeps tests/differential/corpus/ (consumed READ-ONLY): programs whose
#   surface the VM does not support are classified EXCLUDED with a per-file
#   reason (the native-VM also errors) — never silently skipped.
#
# SELF-REPORTING (house style, mirrors scripts/run_vm_parity.sh)
#   * pytest-style lines : "PASSED tests/.../<file>::wasm-vs-native"
#   * ICC JSON-L events  : kind=wasm_parity into
#                          scripts/icc_traces/wasm_parity.jsonl, consumed by
#                          .icc/completion-oracles.yaml::wasm-execute-diff
#
# Usage: scripts/run_wasm_differential.sh [--quick] [--full] [--keep]
#   --quick   supported subset only (default; the CI-PR mode)
#   --full    also sweep tests/differential/corpus (== WASM_DIFF_FULL=1; nightly)
#   --keep    keep the work dir for inspection
#
# Exit: 0 = all green, 1 = a real divergence/failure, 2 = misuse/missing build,
#       77 = prerequisites unavailable (emcc or node) — SKIP, not a failure.
set -u

export LC_ALL=C
export LC_CTYPE=C
export LANG=C

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/wasm_parity.jsonl"
mkdir -p "$TRACE_DIR"
: "${TRACE_FILE:?TRACE_FILE must be set}"
: > "$TRACE_FILE"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) : ;;
    *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;;
esac
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
# Optional native-VM binary — used only as a diagnostic tie-break and to
# classify out-of-subset (differential) programs; never gates the subset.
VM_BIN="$BUILD_DIR/eshkol-vm-standalone-test"
[ -x "$VM_BIN" ] || VM_BIN="$BUILD_DIR/eshkol-vm-standalone"
[ -x "$VM_BIN" ] || VM_BIN=""

# Where the Emscripten VM-diff module is built (under build/, gitignored).
WASM_DIFF_DIR="${WASM_DIFF_DIR:-$BUILD_DIR/wasm-diff}"
WASM_MODULE="$WASM_DIFF_DIR/eshkol-vm-diff.js"
RUNNER_JS="$REPO_ROOT/scripts/lib/wasm_diff_runner.js"
VM_WASM_SRC="$REPO_ROOT/lib/backend/vm_wasm_repl.c"
# Per-file overrides for the supported subset (documented exclusions + xfails).
MANIFEST="$REPO_ROOT/tests/wasm_diff/EXCLUSIONS.tsv"

DO_FULL="${WASM_DIFF_FULL:-0}"
KEEP=0
for arg in "$@"; do
    case "$arg" in
        --quick) DO_FULL=0 ;;
        --full)  DO_FULL=1 ;;
        --keep)  KEEP=1 ;;
        *) echo "run_wasm_differential.sh: unknown flag: $arg" >&2; exit 2 ;;
    esac
done

TIMEOUT_RUN="${WASM_DIFF_TIMEOUT:-60}"

# macOS has no timeout(1); emulate with perl alarm (exit 124 on expiry).
run_guarded() { # seconds cmd...
    perl -e 'my $s=shift; eval { local $SIG{ALRM}=sub{ exit 124 }; alarm $s; exec @ARGV or exit 127; }' \
        "$1" "${@:2}"
}

json_escape() {
    printf '%s' "$1" | perl -0pe 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/\r/\\r/g; s/\t/\\t/g; s/([\x00-\x08\x0b\x0c\x0e-\x1f])/sprintf("\\u%04x", ord($1))/ge'
}

emit_event() { # name value snippet
    printf '{"kind":"wasm_parity","name":"%s","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$(json_escape "$1")" "$(json_escape "$2")" "$(json_escape "$3")" >> "$TRACE_FILE"
}

pass=0; fail=0; excluded=0; xfail=0
report() { # PASS|FAIL nodeid event_name snippet
    if [ "$1" = "PASS" ]; then
        pass=$((pass+1)); echo "PASSED $2"
    else
        fail=$((fail+1)); echo "FAILED $2 — $4"
    fi
    emit_event "$3" "$1" "$4"
}
report_excluded() { # nodeid event_name reason
    excluded=$((excluded+1))
    echo "EXCLUDED $1 — $3"
    emit_event "$2" "EXCLUDED" "$3"
}
report_xfail() { # nodeid event_name reason
    xfail=$((xfail+1))
    echo "XFAIL $1 — $3"
    emit_event "$2" "XFAIL" "$3"
}
# XPASS = a documented XFAIL that unexpectedly matched native: the bug is
# fixed, so the ratchet FAILS to force the manifest row to be removed.
report_xpass() { # nodeid event_name reason
    fail=$((fail+1))
    echo "FAILED $1 — XPASS: $3"
    emit_event "$2" "XPASS" "$3"
}

# Look up a file's manifest class (EXCLUDED|XFAIL) and reason by basename.esk.
manifest_class()  { awk -F'\t' -v b="$1" '!/^#/ && NF>=3 && $1==b {print $2; exit}' "$MANIFEST" 2>/dev/null; }
manifest_reason() { awk -F'\t' -v b="$1" '!/^#/ && NF>=3 && $1==b {print $3; exit}' "$MANIFEST" 2>/dev/null; }

# Reuse the vm-parity normalization verbatim: strip VM/loader/GPU/compiler
# noise, then remove ALL newline characters.  Spaces and every non-newline
# byte (crucially, float digits) are preserved for a RAW comparison.
normalize() { # infile outfile
    perl -ne 'next if
        /^WARN/ or /^INFO:/ or /^DEBUG/ or
        /^\[ESKB\]/ or /^\[GPU\]/ or /^\s*\[compiled:/ or
        /^=== Eshkol VM/ or /^=== Execution complete ===/ or
        /^remark:/ or /^warning: <unknown>/ or
        /^WASM-RUNNER-/;
        print' "$1" | tr -d '\n' > "$2"
}

# A wasm run is "clean" iff the runner did not trap/abort and emitted no
# fatal VM markers on stderr.  (The VM exits 0 even on fatal errors, so
# markers — not exit codes — are the failure signal, same as vm-parity.)
wasm_run_clean() { # errfile rc
    [ "$2" -eq 0 ] || return 1
    ! grep -qE "WASM-RUNNER-EXCEPTION|WASM-RUNNER-ABORT|WASM-RUNNER-FATAL|ERROR|OVERFLOW|unhandled native call|Assertion|abort" "$1"
}
# ── prerequisites ────────────────────────────────────────────────────────
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_wasm_differential.sh: need $ESHKOL_RUN — build with:" >&2
    echo "  cmake --build build --target eshkol-run stdlib -j" >&2
    exit 2
fi

# emcc may live in an emsdk that has to be sourced; try $EMSDK if not on PATH.
if ! command -v emcc >/dev/null 2>&1; then
    if [ -n "${EMSDK:-}" ] && [ -f "$EMSDK/emsdk_env.sh" ]; then
        # shellcheck disable=SC1091
        . "$EMSDK/emsdk_env.sh" >/dev/null 2>&1 || true
    fi
fi
if ! command -v emcc >/dev/null 2>&1; then
    echo "run_wasm_differential.sh: SKIP — emcc (Emscripten) not found on PATH." >&2
    echo "  Install/activate emsdk to build the VM-diff WASM module, then re-run." >&2
    emit_event "wasm_parity_gate" "SKIP" "emcc unavailable — WASM diff lane not exercised"
    exit 77
fi
NODE_BIN="${NODE:-node}"
if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
    echo "run_wasm_differential.sh: SKIP — node not found on PATH." >&2
    emit_event "wasm_parity_gate" "SKIP" "node unavailable — WASM diff lane not exercised"
    exit 77
fi

# ── build the Emscripten VM-diff module from CURRENT source ───────────────
# Rebuild when the module is missing or older than the VM source / this
# script, so the lane always tests the working tree.
need_build=1
if [ -f "$WASM_MODULE" ] && [ -f "${WASM_MODULE%.js}.wasm" ]; then
    if [ "$WASM_MODULE" -nt "$VM_WASM_SRC" ] && [ "$WASM_MODULE" -nt "$0" ]; then
        # Also newer than the VM core it #includes.
        if [ "$WASM_MODULE" -nt "$REPO_ROOT/lib/backend/eshkol_vm.c" ]; then
            need_build=0
        fi
    fi
fi
if [ "$need_build" -eq 1 ]; then
    echo "== building Emscripten VM-diff module (emcc) =="
    mkdir -p "$WASM_DIFF_DIR"
    # STACK_SIZE=8MB: Emscripten's default C stack is only 64KB, but the VM's
    # recursive compiler/parser (nested forms, tensor programs) needs the same
    # headroom the native VM gets from an 8MB OS stack.  Under-provisioning
    # silently overflows the wasm stack into linear memory and corrupts results
    # (e.g. tests/vm_parity/corpus/31_tensor_matmul.esk traps "out of bounds"
    # at 64KB, passes at 8MB) — a build limit, NOT a codegen divergence, so we
    # provision to parity rather than mask it in normalization.
    if ! emcc -O2 -s WASM=1 -s MODULARIZE=1 -s EXPORT_NAME='EshkolVMDiff' \
            -s ENVIRONMENT=node -s ERROR_ON_UNDEFINED_SYMBOLS=0 \
            -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
            -s EXPORTED_FUNCTIONS='["_run_program","_fflush","_malloc","_free"]' \
            -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=67108864 -s STACK_SIZE=8388608 \
            -DESHKOL_VM_WASM -DESHKOL_VM_NO_DISASM \
            -I"$REPO_ROOT/inc" -I"$REPO_ROOT/lib/backend" "$VM_WASM_SRC" \
            -o "$WASM_MODULE" -lm 2> "$WASM_DIFF_DIR/emcc.log"; then
        echo "run_wasm_differential.sh: emcc build FAILED:" >&2
        tail -20 "$WASM_DIFF_DIR/emcc.log" >&2
        emit_event "wasm_parity_gate" "FAIL" "emcc build of VM-diff module failed"
        exit 1
    fi
    # eshkol_qrng_*/eshkol_linear_solve are native leaf runtime deps not
    # linked into the VM WASM; they become aborting stubs, so any program
    # that calls them fails cleanly (never silently mis-executes).
    echo "   built: $WASM_MODULE ($(wc -c < "${WASM_MODULE%.js}.wasm" | tr -d ' ') bytes)"
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-wasm-diff.XXXXXX")"
: "${WORK:?WORK must be set}"
if [ "$KEEP" -eq 1 ]; then
    echo "   work dir (kept): $WORK"
else
    trap 'rm -rf "$WORK"' EXIT
fi
export ESHKOL_JIT_CACHE_DIR="$WORK/jit-cache"
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

# ── one program: native -r vs wasm VM ─────────────────────────────────────
# $1 file  $2 nodeid-prefix  $3 supported(1)|extended(0)
diff_one() {
    local f="$1" prefix="$2" supported="$3"
    local base; base=$(basename "$f" .esk)
    local d="$WORK/$base"; mkdir -p "$d"
    local nodeid="$prefix/$base.esk"

    # Per-file manifest override (supported subset only).  EXCLUDED programs
    # cannot run in the WASM sandbox at all → skip without executing.  XFAIL
    # programs are known-buggy in WASM → run, expect divergence, ratchet on
    # an unexpected match.
    local mclass="" mreason=""
    if [ "$supported" -eq 1 ]; then
        mclass="$(manifest_class "$base.esk")"
        mreason="$(manifest_reason "$base.esk")"
    fi
    if [ "$mclass" = "EXCLUDED" ]; then
        report_excluded "$nodeid::wasm-vs-native" "wasm_${base}" "$mreason"
        return
    fi

    run_guarded "$TIMEOUT_RUN" "$ESHKOL_RUN" -r "$f" >"$d/native.raw" 2>"$d/native.err"
    local nrc=$?
    normalize "$d/native.raw" "$d/native.out"

    if [ $nrc -ne 0 ]; then
        # Native reference itself failed. For the supported subset this is a
        # hard error (corpus must be green natively). For the extended sweep
        # it means the program is not a valid native baseline → exclude.
        if [ "$supported" -eq 1 ]; then
            report FAIL "$nodeid::wasm-vs-native" "wasm_${base}" \
                "native -r exited $nrc (supported corpus must be green natively): $(head -c 140 "$d/native.err")"
        else
            report_excluded "$nodeid::wasm-vs-native" "wasm_${base}" \
                "native -r exited $nrc — not a valid native baseline"
        fi
        return
    fi

    # Extended sweep (tests/differential/corpus) is a native-CODEGEN corpus:
    # keep only the files the native VM already reproduces byte-for-byte, since
    # the WASM profile IS that VM.  Everything else is a pre-existing
    # VM-vs-codegen gap — documented EXCLUDED, never failed: a WASM VM that
    # faithfully mirrors a native VM which itself differs from codegen is not a
    # WASM bug.  (The supported subset, tests/vm_parity/corpus, is already
    # curated to VM==codegen by scripts/run_vm_parity.sh, so it skips this.)
    if [ "$supported" -eq 0 ]; then
        if [ -z "$VM_BIN" ]; then
            report_excluded "$nodeid::wasm-vs-native" "wasm_${base}" \
                "no native VM binary to classify the VM-supported surface (build eshkol-vm-standalone-test)"
            return
        fi
        ESHKOL_VM_NO_DISASM=1 run_guarded "$TIMEOUT_RUN" "$VM_BIN" "$f" >"$d/nvm.raw" 2>"$d/nvm.err"
        local vrc=$?
        normalize "$d/nvm.raw" "$d/nvm.out"
        if [ $vrc -ne 0 ] || grep -qE "ERROR|OVERFLOW|unhandled native call|undefined variable|Assertion|Segmentation" "$d/nvm.err"; then
            report_excluded "$nodeid::wasm-vs-native" "wasm_${base}" \
                "native VM errors on this program — VM-unsupported surface, out-of-subset"
            return
        fi
        if ! cmp -s "$d/native.out" "$d/nvm.out"; then
            report_excluded "$nodeid::wasm-vs-native" "wasm_${base}" \
                "native VM diverges from native codegen (pre-existing VM-vs-codegen gap; the WASM VM mirrors the native VM) — out-of-subset for the VM/WASM profile"
            return
        fi
    fi

    # From here the file is genuinely in the VM/WASM-supported subset: any WASM
    # divergence below is a real WASM issue, not a VM gap.
    run_guarded "$TIMEOUT_RUN" "$NODE_BIN" "$RUNNER_JS" "$WASM_MODULE" "$f" \
        >"$d/wasm.raw" 2>"$d/wasm.err"
    local wrc=$?
    normalize "$d/wasm.raw" "$d/wasm.out"

    if ! wasm_run_clean "$d/wasm.err" "$wrc"; then
        if [ "$mclass" = "XFAIL" ]; then
            # Known WASM bug that manifests as a trap/error — expected.
            report_xfail "$nodeid::wasm-vs-native" "wasm_${base}" \
                "known WASM bug (errored as expected): $mreason"
        else
            # A supported program the WASM VM could not execute cleanly is a
            # real WASM failure — surface it, do not mask it.
            report FAIL "$nodeid::wasm-vs-native" "wasm_${base}" \
                "WASM run failed (rc=$wrc): $(head -c 180 "$d/wasm.err")"
        fi
        return
    fi

    if ! cmp -s "$d/native.out" "$d/wasm.out"; then
        if [ "$mclass" = "XFAIL" ]; then
            report_xfail "$nodeid::wasm-vs-native" "wasm_${base}" \
                "known WASM bug (diverged as expected): native=<$(head -c 90 "$d/native.out")> wasm=<$(head -c 90 "$d/wasm.out")> :: $mreason"
        else
            report FAIL "$nodeid::wasm-vs-native" "wasm_${base}" \
                "output diverges: native=<$(head -c 120 "$d/native.out")> wasm=<$(head -c 120 "$d/wasm.out")>"
        fi
    elif [ "$mclass" = "XFAIL" ]; then
        # Documented WASM bug now matches native → the fix landed; fail so the
        # manifest row is removed and the file becomes fully gated (ratchet).
        report_xpass "$nodeid::wasm-vs-native" "wasm_${base}" \
            "$base.esk is marked XFAIL but WASM now matches native — remove its row from $MANIFEST"
    else
        report PASS "$nodeid::wasm-vs-native" "wasm_${base}" \
            "identical newline-normalized output ($(wc -c < "$d/native.out" | tr -d ' ') bytes)"
    fi
}

# ── supported subset: tests/vm_parity/corpus (must all PASS) ──────────────
echo "== WASM execute-and-diff: supported subset (tests/vm_parity/corpus) =="
CORPUS="$REPO_ROOT/tests/vm_parity/corpus"
shopt -s nullglob
corpus_files=("$CORPUS"/*.esk)
if [ "${#corpus_files[@]}" -eq 0 ]; then
    echo "run_wasm_differential.sh: no corpus files in $CORPUS" >&2
    exit 2
fi
for f in "${corpus_files[@]}"; do
    diff_one "$f" "tests/vm_parity/corpus" 1
done

# ── extended sweep: tests/differential/corpus (READ-ONLY; nightly) ────────
if [ "$DO_FULL" -eq 1 ]; then
    echo
    echo "== WASM execute-and-diff: extended sweep (tests/differential/corpus, read-only) =="
    EXT="$REPO_ROOT/tests/differential/corpus"
    ext_files=("$EXT"/*.esk)
    if [ "${#ext_files[@]}" -eq 0 ]; then
        echo "run_wasm_differential.sh: WASM_DIFF_FULL set but no files in $EXT" >&2
    else
        for f in "${ext_files[@]}"; do
            diff_one "$f" "tests/differential/corpus" 0
        done
    fi
fi

# ── gate ──────────────────────────────────────────────────────────────────
echo
echo "wasm-execute-diff: $pass passed, $fail failed, $xfail xfail (known WASM bugs), $excluded excluded"
if [ $fail -eq 0 ]; then
    emit_event "wasm_parity_gate" "PASS" \
        "$pass checks green; $xfail documented WASM bugs (xfail); $excluded out-of-subset excluded with reasons"
    echo "Trace written: $TRACE_FILE"
    exit 0
else
    emit_event "wasm_parity_gate" "FAIL" "$fail of $((pass+fail)) checks failed (unexpected divergence or XPASS)"
    echo "Trace written: $TRACE_FILE"
    exit 1
fi
