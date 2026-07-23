#!/usr/bin/env bash
# transitive_ffi_link_test.sh — the generated-program native-dependency scan
# must walk the SAME transitive source closure as compilation (load + import +
# require, at every depth), and any generated-program compile/link failure under
# -r must be FATAL (never a fallback continuation that exits zero).
#
# Two coupled defects this regression pins:
#   (A) A helper reached only through (load ...)/(import ...) — not a top-level
#       (require agent.*) — used to drop its agent-FFI link requirement, so the
#       produced binary referenced qllm_process_* with the archive omitted.
#   (B) When the -r persistent-cache child's AOT compile/link failed, eshkol-run
#       fell back to a reduced in-process run and exited 0, certifying a build
#       that never linked.
#
# Neutral and self-contained: it synthesizes its own root/mid/leaf sources.

set -uo pipefail

ESHKOL_RUN="${1:-${ESHKOL_RUN:-}}"
if [ -z "$ESHKOL_RUN" ]; then
    if [ -x "./build/eshkol-run" ]; then
        ESHKOL_RUN="./build/eshkol-run"
    else
        echo "FAIL: transitive_ffi_link_test could not locate eshkol-run" >&2
        exit 1
    fi
fi
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "FAIL: transitive_ffi_link_test eshkol-run is not executable: $ESHKOL_RUN" >&2
    exit 1
fi

# The agent-FFI archive must be present for a real link; otherwise this host
# cannot exercise the linking path at all. Skip cleanly (this is not a failure).
run_dir="$(cd "$(dirname "$ESHKOL_RUN")" && pwd)"
agent_archive=""
for cand in "$run_dir/libeshkol-agent-ffi.a" "$run_dir/../lib/libeshkol-agent-ffi.a"; do
    if [ -f "$cand" ]; then agent_archive="$cand"; break; fi
done
if [ -z "$agent_archive" ]; then
    echo "SKIP: libeshkol-agent-ffi.a not found next to eshkol-run — nothing to link"
    echo "PASS: transitive_ffi_link_test"
    exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fail() { echo "FAIL: transitive_ffi_link_test — $1" >&2; exit 1; }

# nm may or may not exist; symbol assertions are skipped (not failed) without it.
have_nm=0
if command -v nm >/dev/null 2>&1; then have_nm=1; fi

# ── Fixtures: a real MULTI-LEVEL (load) graph ────────────────────────────────
#   root.esk  loads mid.esk
#   mid.esk   loads leaf.esk        (two levels of indirection)
#   leaf.esk  (require agent.subprocess) and references a qllm_process_* builtin
# The FFI requirement lives two loads deep, behind zero top-level (require agent.*).
cat > "$tmp/leaf.esk" <<'EOF'
(require agent.subprocess)
;; Referencing run-argv-capture forces the compiled program to emit
;; qllm_process_* symbol references, which only resolve if the driver linked
;; the agent-FFI archive. Guarded so the test never depends on a spawn actually
;; succeeding in a sandbox — the link requirement exists regardless.
(define (leaf-touch-ffi)
  (if #f (run-argv-capture (list "/bin/echo" "x")) #t))
EOF
cat > "$tmp/mid.esk" <<'EOF'
(load "leaf.esk")
(define (mid-touch-ffi) (leaf-touch-ffi))
EOF
cat > "$tmp/root.esk" <<'EOF'
(require stdlib)
(load "mid.esk")
(mid-touch-ffi)
(display "TRANSITIVE_FFI_LINK_OK")
(newline)
EOF

MARK="TRANSITIVE_FFI_LINK_OK"

# ── 1. AOT: the transitive requirement links; the binary runs and exits 0 ────
aot_bin="$tmp/root_aot"
if ! "$ESHKOL_RUN" "$tmp/root.esk" -o "$aot_bin" > "$tmp/aot_build.log" 2>&1; then
    cat "$tmp/aot_build.log" >&2
    fail "AOT build of the transitive-load FFI program failed (defect A: link requirement dropped)"
fi
[ -x "$aot_bin" ] || fail "AOT build produced no executable"
if ! "$aot_bin" > "$tmp/aot_run.out" 2>&1; then
    cat "$tmp/aot_run.out" >&2
    fail "AOT binary did not run cleanly"
fi
grep -q "^${MARK}\$" "$tmp/aot_run.out" || fail "AOT binary did not print the expected marker"
if [ "$have_nm" -eq 1 ]; then
    # Dump symbols to files first: `nm | grep -q` would give grep an early exit
    # that SIGPIPEs nm, and with `set -o pipefail` that fails the pipeline even
    # on a successful match. Read from the file instead.
    nm -u "$aot_bin" > "$tmp/aot_undef.sym" 2>/dev/null || true
    nm "$aot_bin" > "$tmp/aot_all.sym" 2>/dev/null || true
    if grep -q "qllm_process" "$tmp/aot_undef.sym"; then
        fail "AOT binary has UNRESOLVED qllm_process_* symbols — agent-FFI archive was not linked (defect A)"
    fi
    grep -q "qllm_process" "$tmp/aot_all.sym" || \
        fail "AOT binary does not define qllm_process_* — agent-FFI archive missing from link (defect A)"
fi

# ── 2. -r JIT (persistent cache): same program links, runs, exits 0 ──────────
cache1="$tmp/cache1"
if ! ESHKOL_JIT_CACHE_DIR="$cache1" "$ESHKOL_RUN" -r "$tmp/root.esk" \
        > "$tmp/jit_run.out" 2> "$tmp/jit_run.err"; then
    cat "$tmp/jit_run.err" >&2
    fail "-r run of the transitive-load FFI program failed"
fi
grep -q "^${MARK}\$" "$tmp/jit_run.out" || fail "-r run did not print the expected marker"

# ── 3. No over-linking: a plain program must NOT link the agent-FFI archive ──
cat > "$tmp/plain.esk" <<'EOF'
(display "PLAIN_NO_FFI")
(newline)
EOF
plain_bin="$tmp/plain_aot"
"$ESHKOL_RUN" -d "$tmp/plain.esk" -o "$plain_bin" > "$tmp/plain_build.log" 2>&1 || \
    fail "plain (no-FFI) AOT build failed"
# The emitted link command must not name the agent-FFI archive. Extract the
# link line to a variable (avoid grep|grep pipelines under pipefail).
plain_link_line="$(grep "Linking object files:" "$tmp/plain_build.log" || true)"
case "$plain_link_line" in
    *eshkol-agent-ffi*)
        fail "plain program's link command newly references the agent-FFI archive (over-linking regression)"
        ;;
esac
if [ "$have_nm" -eq 1 ]; then
    nm "$plain_bin" > "$tmp/plain_all.sym" 2>/dev/null || true
    if grep -q "qllm_process" "$tmp/plain_all.sym"; then
        fail "plain program's binary links qllm_process_* it never uses (over-linking regression)"
    fi
fi
# And the FFI program's link command DOES name it (positive control for -d grep).
"$ESHKOL_RUN" -d "$tmp/root.esk" -o "$tmp/root_aot_d" > "$tmp/ffi_build.log" 2>&1 || \
    fail "instrumented AOT build of the FFI program failed"
ffi_link_line="$(grep "Linking object files:" "$tmp/ffi_build.log" || true)"
case "$ffi_link_line" in
    *eshkol-agent-ffi*) ;;
    *) fail "transitive-FFI program's link command omits the agent-FFI archive (defect A)" ;;
esac

# ── 4. Fatal link under -r (defect B) ────────────────────────────────────────
# Force the generated-program link to fail with an intentionally-unavailable
# native dependency. Under -r this MUST exit nonzero and MUST NOT print the
# program's output via a reduced in-process fallback.
cat > "$tmp/fault.esk" <<'EOF'
(display "SHOULD_NOT_APPEAR")
(newline)
EOF
BOGUS="eshkol_no_such_native_lib_xyzzy_$$"
cache2="$tmp/cache2"
set +e
ESHKOL_JIT_CACHE_DIR="$cache2" "$ESHKOL_RUN" -r --lib "$BOGUS" "$tmp/fault.esk" \
    > "$tmp/fault.out" 2> "$tmp/fault.err"
fault_status=$?
set -e 2>/dev/null || true
if [ "$fault_status" -eq 0 ]; then
    fail "-r masked a link failure: exited 0 for a program whose generated-program link could not succeed (defect B)"
fi
if grep -q "SHOULD_NOT_APPEAR" "$tmp/fault.out"; then
    fail "-r ran a reduced in-process fallback after the native link failed and printed program output (defect B)"
fi

# The direct AOT path must likewise fail nonzero for the same broken link.
set +e
"$ESHKOL_RUN" --lib "$BOGUS" "$tmp/fault.esk" -o "$tmp/fault_aot" \
    > "$tmp/fault_aot.log" 2>&1
aot_fault_status=$?
set -e 2>/dev/null || true
if [ "$aot_fault_status" -eq 0 ]; then
    fail "AOT accepted a program whose link could not succeed (should exit nonzero)"
fi

echo "PASS: transitive_ffi_link_test"
