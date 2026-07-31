#!/usr/bin/env bash
# load_path_engine_parity_test.sh — one command, one answer.
#
# WHAT THIS GUARDS
#   `eshkol-run -r prog.esk` reaches the language through more than one engine.
#   With a single input file and no -d/--dump-*/coverage tracing it goes through
#   the persistent JIT run cache, which compiles ahead of time; bypass the cache
#   (ESHKOL_JIT_CACHE=0, or -d, or --dump-ast/--dump-ir, or several inputs, or an
#   active $ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR) and it goes through the
#   in-process LLVM JIT instead. `eshkol-run prog.esk -o prog` is a third lane.
#
#   Those lanes each carried their own copy of the module search order, and the
#   copies disagreed: the AOT copy resolved a relative `(load "sib.esk")` beside
#   the SOURCE FILE, the JIT copy against the process's WORKING DIRECTORY. So
#   the same command on the same file printed one answer with the cache warm and
#   a different one (or "Module not found") with it bypassed — a divergence that
#   surfaced only in whichever harness happened to set one of the bypass
#   switches. run_language_coverage.sh sets the coverage trace dir, so the whole
#   qllm suite aborted under it while passing everywhere else.
#
#   The contract is documented in docs/reference/language/modules.md: the
#   requiring FILE's directory first, then the working directory, then
#   $ESHKOL_PATH/-I, then the install. This test pins BOTH halves of it and
#   demands every engine agree byte for byte.
#
# WHY A DECOY
#   A same-named file is planted in the working directory. A lane that resolves
#   against the cwd finds the decoy and prints a DIFFERENT number — so this test
#   fails on a difference, not merely on an error. Deleting the decoy would let
#   a cwd-rooted lane pass by accident.
#
# USAGE
#   bash tests/toolchain/load_path_engine_parity_test.sh [path/to/eshkol-run]
set -uo pipefail

ESHKOL_RUN="${1:-${ESHKOL_RUN:-}}"
if [ -z "$ESHKOL_RUN" ]; then
    for candidate in ./build/eshkol-run ./build-verify/eshkol-run ./b/eshkol-run; do
        if [ -x "$candidate" ]; then ESHKOL_RUN="$candidate"; break; fi
    done
fi
if [ -z "$ESHKOL_RUN" ] || [ ! -x "$ESHKOL_RUN" ]; then
    echo "FAIL: load_path_engine_parity_test could not locate an executable eshkol-run" >&2
    exit 1
fi
case "$ESHKOL_RUN" in
    /*) : ;;
    *) ESHKOL_RUN="$(pwd)/$ESHKOL_RUN" ;;
esac

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-load-parity.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }

# ── fixture ──────────────────────────────────────────────────────────────
# proj/main.esk       loads a sibling, and a file one directory down
# proj/sib.esk        the sibling that must win over the cwd decoy
# proj/nested/deep.esk    loads ITS OWN sibling — the search root must follow
# proj/nested/deeper.esk  the load stack down, not stay pinned at main.esk
mkdir -p "$tmpdir/proj/nested" "$tmpdir/elsewhere/shared"

cat > "$tmpdir/proj/sib.esk" <<'EOF'
(define (sib-value) 21)
EOF

cat > "$tmpdir/proj/nested/deeper.esk" <<'EOF'
(define (deeper-value) 100)
EOF

cat > "$tmpdir/proj/nested/deep.esk" <<'EOF'
(load "deeper.esk")
(define (deep-value) (+ (deeper-value) 5))
EOF

cat > "$tmpdir/proj/main.esk" <<'EOF'
(load "sib.esk")
(load "nested/deep.esk")
(display (* 2 (sib-value)))
(newline)
(display (deep-value))
(newline)
EOF

# The decoy: same spelling, different answer, sitting in the working directory.
cat > "$tmpdir/elsewhere/sib.esk" <<'EOF'
(define (sib-value) 999)
EOF

# The other half of the contract: a program whose load target exists ONLY
# relative to the working directory (the repo-root spelling the corpus uses,
# e.g. (load "tests/.../fixture.esk") run from the repo root). Tier 2 must
# still fire, or that whole camp of tests stops resolving.
cat > "$tmpdir/elsewhere/shared/helper.esk" <<'EOF'
(define (helper-value) 7)
EOF
cat > "$tmpdir/proj/cwdrel.esk" <<'EOF'
(load "shared/helper.esk")
(display (* 3 (helper-value)))
(newline)
EOF

expected_main=$'42\n105'
expected_cwdrel=$'21'

# ── engines ──────────────────────────────────────────────────────────────
# Each runs with cwd = $tmpdir/elsewhere, i.e. NOT the source file's directory,
# which is the only condition under which the two rules differ at all.
run_engine() { # engine-name source-file -> stdout on fd 1
    local engine="$1" src="$2"
    local out rc
    case "$engine" in
        jit-cache)
            out="$(cd "$tmpdir/elsewhere" && \
                ESHKOL_JIT_CACHE_DIR="$tmpdir/cache" \
                "$ESHKOL_RUN" -r "$src" 2>/dev/null)"
            rc=$?
            ;;
        jit-in-process)
            out="$(cd "$tmpdir/elsewhere" && \
                ESHKOL_JIT_CACHE=0 \
                "$ESHKOL_RUN" -r "$src" 2>/dev/null)"
            rc=$?
            ;;
        jit-coverage-trace)
            # The exact bypass run_language_coverage.sh takes.
            mkdir -p "$tmpdir/covtrace"
            out="$(cd "$tmpdir/elsewhere" && \
                ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR="$tmpdir/covtrace" \
                "$ESHKOL_RUN" -r "$src" 2>/dev/null)"
            rc=$?
            ;;
        aot)
            local bin="$tmpdir/aot-$(basename "$src" .esk)"
            if ! (cd "$tmpdir/elsewhere" && \
                    "$ESHKOL_RUN" "$src" -o "$bin" >/dev/null 2>&1); then
                echo "__COMPILE_FAILED__"
                return 1
            fi
            out="$(cd "$tmpdir/elsewhere" && "$bin" 2>/dev/null)"
            rc=$?
            ;;
        *)
            echo "__UNKNOWN_ENGINE__"; return 1 ;;
    esac
    printf '%s' "$out"
    return "$rc"
}

ENGINES="jit-cache jit-in-process jit-coverage-trace aot"

check_program() { # label source-file expected
    local label="$1" src="$2" expected="$3"
    local baseline="" baseline_engine="" agreed=0 total=0
    for engine in $ENGINES; do
        local got rc
        total=$((total + 1))
        got="$(run_engine "$engine" "$src")"
        rc=$?
        if [ "$rc" -ne 0 ]; then
            fail "$label: engine '$engine' exited $rc (output: ${got//$'\n'/ | })"
            continue
        fi
        if [ "$got" != "$expected" ]; then
            fail "$label: engine '$engine' produced '${got//$'\n'/ | }', expected '${expected//$'\n'/ | }'"
            continue
        fi
        if [ -z "$baseline_engine" ]; then
            baseline="$got"; baseline_engine="$engine"; agreed=1
        elif [ "$got" != "$baseline" ]; then
            fail "$label: engine '$engine' disagrees with '$baseline_engine' — one command, two answers"
        else
            agreed=$((agreed + 1))
        fi
    done
    # Report the count that actually agreed, never the count attempted: a line
    # reading "4 engines agree" above two FAILs is how a parity test stops
    # being read.
    echo "  $label: $agreed/$total engines agree on '${baseline//$'\n'/ | }'"
}

echo "load_path_engine_parity_test: source-file-relative load, cwd holds a decoy"
check_program "file-relative load (+ nested, + decoy in cwd)" \
    "$tmpdir/proj/main.esk" "$expected_main"

echo "load_path_engine_parity_test: cwd-relative load (project-root spelling)"
check_program "cwd-relative load" \
    "$tmpdir/proj/cwdrel.esk" "$expected_cwdrel"

# ── the decoy must genuinely be reachable ────────────────────────────────
# If the decoy could never be found by anything, its presence proves nothing.
# Run a program FROM the working directory that loads it: that must print 1998,
# confirming the file is loadable and that the 42 above is a real precedence
# decision rather than an inability to see the decoy at all.
cat > "$tmpdir/elsewhere/decoy_probe.esk" <<'EOF'
(load "sib.esk")
(display (* 2 (sib-value)))
(newline)
EOF
decoy_out="$(cd "$tmpdir/elsewhere" && ESHKOL_JIT_CACHE=0 \
    "$ESHKOL_RUN" -r "$tmpdir/elsewhere/decoy_probe.esk" 2>/dev/null)"
if [ "$decoy_out" != "1998" ]; then
    fail "decoy control: expected 1998 from the cwd copy, got '$decoy_out' (the decoy is not reachable, so the precedence assertion above is vacuous)"
else
    echo "  decoy control: the cwd copy is reachable (1998) and correctly loses to the sibling"
fi

if [ "$failures" -ne 0 ]; then
    echo "FAIL: load_path_engine_parity_test — $failures check(s) failed" >&2
    exit 1
fi

echo "PASS: load_path_engine_parity_test"
exit 0
