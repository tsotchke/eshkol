#!/usr/bin/env bash
# run_vm_parity.sh — VM parity gate (adversarial P5).
#
# Three stages:
#
#   1. AUDIT   scripts/vm_parity_audit.py — every symbol on the native
#              codegen surface must be VM-supported or consciously waived in
#              tests/vm_parity/PARITY.tsv.  Adding a language feature without
#              updating the VM or the manifest fails here (the ratchet).
#
#   2. CORPUS  VM-vs-native differential over tests/vm_parity/corpus/
#              (programs inside the VM's verified subset).  Axes:
#                native   ./build/eshkol-run -r f.esk
#                vm-src   ./build/eshkol-vm-standalone-test f.esk
#                vm-eskb  ./build/eshkol-run --profile hosted-vm
#                             --emit-eskb f.eskb f.esk
#                         && ./build/eshkol-vm-standalone-test f.eskb
#              The VM's `display` appends a newline per call (filed:
#              tests/vm_parity/found/display_newline_per_call.esk), so
#              normalization strips banner/log lines and then removes ALL
#              newline characters from both sides before byte comparison.
#              Value divergences, dropped output and fabricated output all
#              still surface; only newline-placement divergences are masked
#              (that is exactly the filed quirk).  VM failure is detected via
#              BOTH the exit status and ERROR/WARNING markers on stderr; the
#              VM used to exit 0 on every fatal runtime error, which stage 4
#              now gates directly.
#
#   3. OOS     Programs outside the subset (tests/vm_parity/oos/) must fail
#              CLEANLY on the VM: a clear diagnostic on stderr and no
#              fabricated value on stdout.
#
#   4. FATAL   Programs whose FIRST failing form is fatal on both substrates
#              (tests/vm_parity/fatal/) must FAIL CLOSED: nonzero exit, a
#              diagnostic on stderr, and no output past the fatal form.  This
#              is the fail-open ratchet — a fatal VM error may never again
#              look like a successful run to a shell or to CI.
#
# Emits (mirroring scripts/run_sicp_smoke.sh):
#   * pytest-style lines : "PASSED tests/vm_parity/<file>::<check>"
#   * ICC JSON-L events  : kind=vm_parity into
#                          scripts/icc_traces/vm_parity.jsonl, consumed by
#                          .icc/completion-oracles.yaml::vm-parity
#
# Usage: scripts/run_vm_parity.sh [--no-eskb] [--audit-only]
set -u

# Keep the Perl timeout/json helpers portable. Some macOS hosts inherit a
# UTF-8 locale name that Perl cannot materialize; the C locale is sufficient
# for this byte-oriented gate and avoids false infrastructure failures.
export LC_ALL=C
export LC_CTYPE=C
export LANG=C

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/vm_parity.jsonl"
mkdir -p "$TRACE_DIR"
: "${TRACE_FILE:?TRACE_FILE must be set}"
: > "$TRACE_FILE"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) : ;;
    *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;;
esac
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
VM_BIN="$BUILD_DIR/eshkol-vm-standalone-test"
[ -x "$VM_BIN" ] || VM_BIN="$BUILD_DIR/eshkol-vm-standalone"

DO_ESKB=1
AUDIT_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --no-eskb) DO_ESKB=0 ;;
        --audit-only) AUDIT_ONLY=1 ;;
        *) echo "run_vm_parity.sh: unknown flag: $arg" >&2; exit 2 ;;
    esac
done

TIMEOUT_RUN="${VM_PARITY_TIMEOUT:-60}"

# macOS has no `timeout(1)`; emulate with perl alarm (exit 124 on expiry).
run_guarded() { # seconds cmd...
    perl -e 'my $s=shift; eval { local $SIG{ALRM}=sub{ exit 124 }; alarm $s; exec @ARGV or exit 127; }' \
        "$1" "${@:2}"
}

json_escape() {
    printf '%s' "$1" | perl -0pe 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/\r/\\r/g; s/\t/\\t/g; s/([\x00-\x08\x0b\x0c\x0e-\x1f])/sprintf("\\u%04x", ord($1))/ge'
}

emit_event() { # name value snippet
    printf '{"kind":"vm_parity","name":"%s","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$(json_escape "$1")" "$(json_escape "$2")" "$(json_escape "$3")" >> "$TRACE_FILE"
}

# ICC architecture invariants consume canonical test_result evidence rather
# than the parity gate's richer domain-specific events.  Emit both: the
# vm_parity event remains the completion-oracle payload, while test_result is
# the runtime proof for INV-dispatch-table-completeness.
emit_test_result() { # name PASS|FAIL snippet
    local passed=false
    [ "$2" = "PASS" ] && passed=true
    printf '{"kind":"test_result","name":"%s","value":{"passed":%s,"summary":"%s"},"timestamp":%s}\n' \
        "$(json_escape "$1")" "$passed" "$(json_escape "$3")" "$(date +%s)" >> "${TRACE_FILE:?}"
}

pass=0; fail=0
report() { # PASS|FAIL nodeid event_name snippet
    if [ "$1" = "PASS" ]; then
        pass=$((pass+1)); echo "PASSED $2"
    else
        fail=$((fail+1)); echo "FAILED $2 — $4"
    fi
    emit_event "$3" "$1" "$4"
}

# ── stage 1: surface audit (the ratchet) ────────────────────────────────
echo "== stage 1: codegen-vs-VM surface audit =="
audit_out=$(python3 "$REPO_ROOT/scripts/vm_parity_audit.py" 2>&1); audit_rc=$?
echo "$audit_out"
if [ $audit_rc -eq 0 ]; then
    report PASS "tests/vm_parity/PARITY.tsv::surface-audit" "vm_parity_audit" \
        "$(echo "$audit_out" | grep 'manifest rows' | head -1)"
else
    report FAIL "tests/vm_parity/PARITY.tsv::surface-audit" "vm_parity_audit" \
        "$(echo "$audit_out" | grep '^vm_parity_audit: FAIL' | head -3)"
fi

if [ "$AUDIT_ONLY" -eq 1 ]; then
    gate_status="$([ $fail -eq 0 ] && echo PASS || echo FAIL)"
    gate_summary="audit-only: $pass passed, $fail failed"
    emit_event "vm_parity_gate" "$gate_status" "$gate_summary"
    emit_test_result "vm_parity_gate" "$gate_status" "$gate_summary"
    echo; echo "vm-parity (audit-only): $pass passed, $fail failed"
    [ $fail -eq 0 ] || exit 1
    exit 0
fi

if [ ! -x "$ESHKOL_RUN" ] || [ ! -x "$VM_BIN" ]; then
    echo "run_vm_parity.sh: need $ESHKOL_RUN and $VM_BIN — build with:" >&2
    echo "  cmake --build build --target eshkol-run stdlib eshkol-vm-standalone-test -j" >&2
    exit 2
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-vm-parity.XXXXXX")"
: "${WORK:?WORK must be set}"
trap 'rm -rf "$WORK"' EXIT
export ESHKOL_JIT_CACHE_DIR="$WORK/jit-cache"
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

# Normalize an output capture:
#   * strip VM banners, ESKB loader lines, GPU init logs, compiler noise;
#   * remove ALL newline characters (the filed display-per-call-newline
#     divergence inserts newlines where native has none, so per-line
#     normalization cannot align the two — the newline-free byte stream is
#     the strongest comparison the quirk permits; spaces are preserved).
normalize() { # infile outfile
    perl -ne 'next if
        /^WARN/ or /^INFO:/ or /^DEBUG/ or
        /^\[ESKB\]/ or /^\[GPU\]/ or /^\s*\[compiled:/ or
        /^=== Eshkol VM/ or /^=== Execution complete ===/ or
        /^remark:/ or /^warning: <unknown>/;
        print' "$1" | tr -d '\n' > "$2"
}

vm_stderr_clean() { # errfile -> 0 if no ERROR/abort markers
    # The VM exits 0 even on fatal errors, so stderr markers are the only
    # failure signal: ERROR, FRAME OVERFLOW (silent-empty-stdout death,
    # found/frame_overflow_exit_zero.esk) and unhandled-fid warnings
    # (found/symbol_string_unhandled_fid.esk) all mean the run is invalid.
    ! grep -qE "ERROR|OVERFLOW|unhandled native call|Assertion|Segmentation|abort" "$1"
}

# ── stage 2: corpus differential ────────────────────────────────────────
echo
echo "== stage 2: corpus differential (native -r vs vm-src$([ $DO_ESKB -eq 1 ] && printf ' vs vm-eskb')) =="
CORPUS="$REPO_ROOT/tests/vm_parity/corpus"
shopt -s nullglob
corpus_files=("$CORPUS"/*.esk)
if [ "${#corpus_files[@]}" -eq 0 ]; then
    echo "run_vm_parity.sh: no corpus files in $CORPUS" >&2
    exit 2
fi

for f in "${corpus_files[@]}"; do
    base=$(basename "$f" .esk)
    d="$WORK/$base"; mkdir -p "$d"

    native_args=(-r "$f")
    case "$base" in
        17_guard_raise|18_call_cc)
            # These primitives are compiler/runtime builtins and do not depend
            # on the Scheme stdlib.  Loading the full stdlib makes LLVM spend
            # roughly three minutes optimizing either tiny probe on macOS
            # (measured 190.90 s for guard versus <1 s with --no-stdlib), which
            # turns a semantic parity gate into a compile-throughput timeout.
            # Keep the probes exact while isolating the control-flow surface.
            native_args=(-n -r "$f")
            ;;
    esac
    run_guarded "$TIMEOUT_RUN" "$ESHKOL_RUN" "${native_args[@]}" >"$d/native.raw" 2>"$d/native.err"
    nrc=$?
    normalize "$d/native.raw" "$d/native.out"

    ESHKOL_VM_NO_DISASM=1 run_guarded "$TIMEOUT_RUN" "$VM_BIN" "$f" >"$d/vmsrc.raw" 2>"$d/vmsrc.err"
    vrc=$?
    normalize "$d/vmsrc.raw" "$d/vmsrc.out"

    nodeid="tests/vm_parity/corpus/$base.esk"
    if [ $nrc -ne 0 ]; then
        report FAIL "$nodeid::native-vs-vm-src" "corpus_${base}_vmsrc" \
            "native -r exited $nrc (corpus programs must be green natively)"
    elif [ $vrc -ne 0 ] || ! vm_stderr_clean "$d/vmsrc.err"; then
        report FAIL "$nodeid::native-vs-vm-src" "corpus_${base}_vmsrc" \
            "vm-src errored (rc=$vrc, stderr: $(head -c 160 "$d/vmsrc.err"))"
    elif ! cmp -s "$d/native.out" "$d/vmsrc.out"; then
        report FAIL "$nodeid::native-vs-vm-src" "corpus_${base}_vmsrc" \
            "output diverges: native=<$(head -c 120 "$d/native.out")> vm=<$(head -c 120 "$d/vmsrc.out")>"
    else
        report PASS "$nodeid::native-vs-vm-src" "corpus_${base}_vmsrc" \
            "identical newline-normalized output"
    fi

    if [ $DO_ESKB -eq 1 ]; then
        eskb="$d/prog.eskb"
        run_guarded "$TIMEOUT_RUN" "$ESHKOL_RUN" --profile hosted-vm --emit-eskb "$eskb" "$f" \
            >"$d/eskb.compile.out" 2>"$d/eskb.compile.err"
        erc=$?
        if [ $erc -ne 0 ] || [ ! -f "$eskb" ]; then
            report FAIL "$nodeid::native-vs-vm-eskb" "corpus_${base}_vmeskb" \
                "eskb emit failed rc=$erc"
            continue
        fi
        ESHKOL_VM_NO_DISASM=1 run_guarded "$TIMEOUT_RUN" "$VM_BIN" "$eskb" >"$d/vmeskb.raw" 2>"$d/vmeskb.err"
        brc=$?
        normalize "$d/vmeskb.raw" "$d/vmeskb.out"
        if [ $brc -ne 0 ] || ! vm_stderr_clean "$d/vmeskb.err"; then
            report FAIL "$nodeid::native-vs-vm-eskb" "corpus_${base}_vmeskb" \
                "vm-eskb errored (rc=$brc, stderr: $(head -c 160 "$d/vmeskb.err"))"
        elif ! cmp -s "$d/native.out" "$d/vmeskb.out"; then
            report FAIL "$nodeid::native-vs-vm-eskb" "corpus_${base}_vmeskb" \
                "output diverges: native=<$(head -c 120 "$d/native.out")> vm-eskb=<$(head -c 120 "$d/vmeskb.out")>"
        else
            report PASS "$nodeid::native-vs-vm-eskb" "corpus_${base}_vmeskb" \
                "identical newline-normalized output"
        fi
    fi
done

# ── stage 3: out-of-subset probes must fail cleanly ─────────────────────
echo
echo "== stage 3: out-of-subset probes (clean VM error, no fabricated value) =="
OOS="$REPO_ROOT/tests/vm_parity/oos"
oos_files=("$OOS"/*.esk)
for f in "${oos_files[@]}"; do
    base=$(basename "$f" .esk)
    d="$WORK/$base"; mkdir -p "$d"
    ESHKOL_VM_NO_DISASM=1 run_guarded "$TIMEOUT_RUN" "$VM_BIN" "$f" >"$d/vm.raw" 2>"$d/vm.err"
    normalize "$d/vm.raw" "$d/vm.out"
    nodeid="tests/vm_parity/oos/$base.esk"
    # Clean failure = a clear diagnostic on stderr AND no fabricated value
    # on stdout.  Exit status is asserted separately, in stage 4.
    if ! grep -qE "ERROR|undefined variable" "$d/vm.err"; then
        report FAIL "$nodeid::fails-cleanly" "oos_${base}" \
            "no diagnostic on stderr; VM may have silently mis-executed an unsupported feature"
    elif [ -s "$d/vm.out" ] && ! grep -qE '^\(\)$' "$d/vm.out"; then
        report FAIL "$nodeid::fails-cleanly" "oos_${base}" \
            "fabricated stdout despite unsupported feature: <$(head -c 120 "$d/vm.out")>"
    else
        report PASS "$nodeid::fails-cleanly" "oos_${base}" \
            "clear diagnostic, no fabricated value"
    fi
done

# ── stage 4: fatal errors must FAIL CLOSED ──────────────────────────────
#
# The VM used to return 0 from main() unconditionally, so a program killed by
# a fatal runtime error — dropping every remaining top-level form — was
# indistinguishable from a clean run for shells, Makefiles and CI.  Each probe
# in tests/vm_parity/fatal/ must, on BOTH substrates, (1) exit NONZERO,
# (2) name the failure on stderr, and (3) not print the sentinel that follows
# the failing form.  This is the same fail-open discipline already enforced at
# the driver and FFI boundaries.
echo
echo "== stage 4: fatal-error exit status (fail closed, both substrates) =="
FATAL="$REPO_ROOT/tests/vm_parity/fatal"
fatal_files=("$FATAL"/*.esk)
for f in "${fatal_files[@]}"; do
    base=$(basename "$f" .esk)
    d="$WORK/fatal_$base"; mkdir -p "$d"
    nodeid="tests/vm_parity/fatal/$base.esk"

    run_guarded "$TIMEOUT_RUN" "$ESHKOL_RUN" -n -r "$f" >"$d/native.raw" 2>"$d/native.err"
    nrc=$?
    ESHKOL_VM_NO_DISASM=1 run_guarded "$TIMEOUT_RUN" "$VM_BIN" "$f" >"$d/vm.raw" 2>"$d/vm.err"
    vrc=$?

    if [ $nrc -eq 0 ]; then
        report FAIL "$nodeid::native-exits-nonzero" "fatal_${base}_native" \
            "native -r exited 0 on a fatal error"
    else
        report PASS "$nodeid::native-exits-nonzero" "fatal_${base}_native" \
            "native exited $nrc"
    fi

    if [ $vrc -eq 0 ]; then
        report FAIL "$nodeid::vm-exits-nonzero" "fatal_${base}_vm" \
            "VM exited 0 on a fatal error (fail-open regression)"
    elif ! grep -qE "ERROR|OVERFLOW|BY ZERO" "$d/vm.err"; then
        report FAIL "$nodeid::vm-exits-nonzero" "fatal_${base}_vm" \
            "VM exited $vrc but printed no diagnostic on stderr"
    elif grep -q "MUST-NOT-PRINT" "$d/vm.raw"; then
        report FAIL "$nodeid::vm-exits-nonzero" "fatal_${base}_vm" \
            "VM continued past the fatal form"
    else
        report PASS "$nodeid::vm-exits-nonzero" "fatal_${base}_vm" \
            "VM exited $vrc with a diagnostic and stopped at the fatal form"
    fi
done

# ── gate ─────────────────────────────────────────────────────────────────
echo
echo "vm-parity: $pass passed, $fail failed"
if [ $fail -eq 0 ]; then
    gate_summary="$pass checks green (audit + corpus + oos)"
    emit_event "vm_parity_gate" "PASS" "$gate_summary"
    # Name the production dispatcher explicitly so ICC can bind this full
    # source+serialized-bytecode parity run to the implementation boundary it
    # exercises, rather than heuristically classifying the dispatcher as an
    # untested backend-shaped path.
    emit_event "vm_dispatch_native" "PASS" \
        "vm_dispatch_native exercised by $gate_summary"
    emit_test_result "vm_parity_gate" "PASS" "$gate_summary"
    echo "Trace written: $TRACE_FILE"
    exit 0
else
    gate_summary="$fail of $((pass+fail)) checks failed"
    emit_event "vm_parity_gate" "FAIL" "$gate_summary"
    emit_test_result "vm_parity_gate" "FAIL" "$gate_summary"
    echo "Trace written: $TRACE_FILE"
    exit 1
fi
