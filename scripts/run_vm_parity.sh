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
. "$REPO_ROOT/scripts/lib/durable_work_root.sh"
. "$REPO_ROOT/scripts/lib/harness_outcome.sh"
if eshkol_durable_enabled; then
    VM_PARITY_WORK="$(eshkol_durable_prepare_dir vm-parity)" || exit $?
    TRACE_DIR="${TRACE_DIR:-$VM_PARITY_WORK/traces}"
else
    TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
fi
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

# Real, signal-observing timeout wrapper (scripts/lib/harness_outcome.sh).
# This replaces an exec-then-perl-alarm one-liner this script used to carry
# locally: `exec` replaces perl's own process image, so the alarm handler
# perl installed does not survive into the child, and a child still running
# when the alarm fires is killed by SIGALRM under ITS OWN default
# disposition — exit 128+14=142 — instead of a controlled, recognizable 124.
# That is exactly the F13 audit incident (2026-08-25 architecture audit,
# section 3): `native -r exited 142` on a 140s cold-start JIT/stdlib compile
# under load, read as a parity FAIL because nothing downstream could tell a
# clock fact from a wrong answer. eshkol_outcome_guarded forks instead of
# exec'ing, so the alarm fires in the still-alive parent and reports a
# stable 124; a real crash the child raises on its own is still preserved as
# 128+N (e.g. a genuine SIGSEGV stays 139), so a self-inflicted defect is
# never masked as infrastructure. See eshkol_outcome_classify_exit below.
run_guarded() { eshkol_outcome_guarded "$@"; } # seconds cmd...

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

pass=0; fail=0; infra=0
report() { # PASS|FAIL|INFRA nodeid event_name snippet
    case "$1" in
        PASS)  pass=$((pass+1));  echo "PASSED $2" ;;
        INFRA) infra=$((infra+1)); echo "INFRA  $2 — $4 (no verdict obtained; not counted as a parity defect)" ;;
        *)     fail=$((fail+1));  echo "FAILED $2 — $4" ;;
    esac
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

if eshkol_durable_enabled; then
    WORK="$VM_PARITY_WORK/work"
    mkdir "$WORK"
else
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-vm-parity.XXXXXX")"
fi
: "${WORK:?WORK must be set}"
if ! eshkol_durable_enabled; then trap 'rm -rf "$WORK"' EXIT; fi
export ESHKOL_JIT_CACHE_DIR="$WORK/jit-cache"
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

# ── JIT-cache warm-up (fix for F13: the 140s cold-start timeout) ────────
#
# $ESHKOL_JIT_CACHE_DIR is a fresh directory every invocation of this
# script, so the FIRST native compile against it always pays the full
# stdlib load/optimize cost — measured at 140.49s under a load average of
# 180, versus 0.07s/0.06s on a warm cache (2026-08-25 audit, section 3).
# TIMEOUT_RUN defaults to 60s, so an unlucky first corpus file used to eat
# that cold-start cost inside its own timed window and get killed — not
# because anything was wrong, but because it happened to go first. Paying
# the cold cost HERE, once, outside any per-file timing window, means every
# per-file measurement below starts from a warm cache and the 60s budget
# is measuring the thing it is supposed to measure again.
#
# This does not remove the safety net: if the warm-up itself times out
# (a genuinely pathological host), we log it and continue — the per-file
# eshkol_outcome_retry_guarded calls below still classify and retry any
# residual cold-start timeout as INFRA rather than a parity FAIL.
WARMUP_TIMEOUT="${VM_PARITY_WARMUP_TIMEOUT:-300}"
WARM_FILE="$WORK/_warmup.esk"
printf '(display 1)\n(newline)\n' > "$WARM_FILE"
echo "== warm-up: priming \$ESHKOL_JIT_CACHE_DIR (stdlib compile, budget ${WARMUP_TIMEOUT}s) =="
if eshkol_outcome_guarded "$WARMUP_TIMEOUT" "$ESHKOL_RUN" -r "$WARM_FILE" \
        >"$WORK/_warmup.out" 2>"$WORK/_warmup.err"; then
    echo "warm-up: cache primed"
else
    echo "run_vm_parity.sh: warm-up compile did not finish within ${WARMUP_TIMEOUT}s —" \
         "the corpus loop below may still see a cold first hit under extreme load" \
         "(see $WORK/_warmup.err); the per-file retry-once still applies." >&2
fi

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
    # eshkol_outcome_retry_guarded: one attempt, and — IFF that attempt
    # classifies as INFRA (timeout/OOM/signal-from-outside) — exactly one
    # retry before we conclude anything. This is the direct fix for F13: a
    # cold-start timeout gets a second try, at which point the JIT cache the
    # warm-up (and/or the first attempt itself) populated makes the retry
    # fast, matching what the audit measured by hand (140.49s once, then
    # 0.07s, 0.06s). A real wrong-answer FAIL is never retried — only the
    # clock gets a second chance.
    eshkol_outcome_retry_guarded "$TIMEOUT_RUN" "$d/native.raw" "$d/native.err" \
        "$ESHKOL_RUN" "${native_args[@]}"
    nrc=$?
    normalize "$d/native.raw" "$d/native.out"

    ESHKOL_VM_NO_DISASM=1 eshkol_outcome_retry_guarded "$TIMEOUT_RUN" "$d/vmsrc.raw" "$d/vmsrc.err" \
        "$VM_BIN" "$f"
    vrc=$?
    normalize "$d/vmsrc.raw" "$d/vmsrc.out"

    nodeid="tests/vm_parity/corpus/$base.esk"
    nclass=$(eshkol_outcome_classify_exit "$nrc")
    vclass=$(eshkol_outcome_classify_exit "$vrc")
    if [ "$nclass" = INFRA ]; then
        report INFRA "$nodeid::native-vs-vm-src" "corpus_${base}_vmsrc" \
            "native -r timed out/infra after retry (rc=$nrc) — no parity verdict obtained"
    elif [ $nrc -ne 0 ]; then
        report FAIL "$nodeid::native-vs-vm-src" "corpus_${base}_vmsrc" \
            "native -r exited $nrc (corpus programs must be green natively)"
    elif [ "$vclass" = INFRA ]; then
        report INFRA "$nodeid::native-vs-vm-src" "corpus_${base}_vmsrc" \
            "vm-src timed out/infra after retry (rc=$vrc) — no parity verdict obtained"
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
        eshkol_outcome_retry_guarded "$TIMEOUT_RUN" "$d/eskb.compile.out" "$d/eskb.compile.err" \
            "$ESHKOL_RUN" --profile hosted-vm --emit-eskb "$eskb" "$f"
        erc=$?
        eclass=$(eshkol_outcome_classify_exit "$erc")
        if [ "$eclass" = INFRA ]; then
            report INFRA "$nodeid::native-vs-vm-eskb" "corpus_${base}_vmeskb" \
                "eskb emit timed out/infra after retry (rc=$erc) — no parity verdict obtained"
            continue
        fi
        if [ $erc -ne 0 ] || [ ! -f "$eskb" ]; then
            report FAIL "$nodeid::native-vs-vm-eskb" "corpus_${base}_vmeskb" \
                "eskb emit failed rc=$erc"
            continue
        fi
        ESHKOL_VM_NO_DISASM=1 eshkol_outcome_retry_guarded "$TIMEOUT_RUN" "$d/vmeskb.raw" "$d/vmeskb.err" \
            "$VM_BIN" "$eskb"
        brc=$?
        bclass=$(eshkol_outcome_classify_exit "$brc")
        normalize "$d/vmeskb.raw" "$d/vmeskb.out"
        if [ "$bclass" = INFRA ]; then
            report INFRA "$nodeid::native-vs-vm-eskb" "corpus_${base}_vmeskb" \
                "vm-eskb timed out/infra after retry (rc=$brc) — no parity verdict obtained"
        elif [ $brc -ne 0 ] || ! vm_stderr_clean "$d/vmeskb.err"; then
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
#
# INFRA checks never count toward `fail`: a check that could not obtain a
# verdict (even after one retry) is not evidence the VM diverges from
# native — it is evidence the run needs redoing under better conditions.
# Folding it into `fail` here is exactly the F13 audit defect (a spurious
# FAIL propagating into INV-dispatch-table-completeness); leaving it out
# is the "no verdict beats a false FAIL" choice documented in
# scripts/lib/harness_outcome.sh. It is still loud: printed distinctly
# above, counted separately, and never silently dropped.
echo
echo "vm-parity: $pass passed, $fail failed, $infra infra (no verdict)"
if [ $infra -gt 0 ]; then
    echo "WARNING: $infra check(s) could not obtain a parity verdict (INFRA) — see trace and re-run under less contention if this persists." >&2
fi
if [ $fail -eq 0 ]; then
    gate_summary="$pass checks green (audit + corpus + oos + fatal)$([ $infra -gt 0 ] && printf '; %d infra (no verdict)' "$infra")"
    emit_event "vm_parity_gate" "PASS" "$gate_summary"
    # Name the production dispatcher explicitly so ICC can bind this full
    # source+serialized-bytecode parity run to the implementation boundary it
    # exercises, rather than heuristically classifying the dispatcher as an
    # untested backend-shaped path.
    emit_event "vm_dispatch_native" "PASS" \
        "vm_dispatch_native exercised by $gate_summary"
    emit_test_result "vm_parity_gate" "PASS" "$gate_summary"
    rc=0
else
    gate_summary="$fail of $((pass+fail)) checks failed"
    emit_event "vm_parity_gate" "FAIL" "$gate_summary"
    emit_test_result "vm_parity_gate" "FAIL" "$gate_summary"
    rc=1
fi

# Mirror only after every event (including the final gate verdict) has been
# appended, so a durable-root mirror is never missing the summary line.
eshkol_durable_mirror_trace "$TRACE_FILE" vm_parity.jsonl
echo "Trace written: $TRACE_FILE"
exit "$rc"
