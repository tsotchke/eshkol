#!/usr/bin/env bash
#
# run_pgo_corpus_smoke.sh — Stage-1 consumer for bench/pgo_corpus/ (ADR 0007).
#
# ADR 0007 names bench/pgo_corpus/ (5 programs) as the training corpus for the
# eventual eshkol-pgo-train/-merge/-verify orchestration (Phase 1 of the PGO
# ADR). That orchestration does not exist yet (BUILD ITEM, targeted v1.5.0 —
# see the ADR's 2026-08-25 attainment note). Until it lands, this corpus had
# ZERO callers anywhere: nothing compiled it, nothing ran it, so a program
# could silently bit-rot (stale builtin, broken syntax) for months and no one
# would notice until someone tried to actually wire PGO training against it.
#
# This script is deliberately NOT full PGO wiring. It is the Stage-1 smoke
# consumer the corpus was missing: for every bench/pgo_corpus/*.esk file it
#   1. runs the program under the JIT (`eshkol-run -r`),
#   2. compiles + runs the program AOT (`eshkol-run -o`),
#   3. asserts both modes exit 0, produce non-empty output, and produce
#      BYTE-IDENTICAL stdout — a JIT/AOT divergence on this corpus is exactly
#      the kind of codegen bug PGO training would otherwise bake in silently.
#
# What full PGO wiring still needs (tracked against ADR 0007, not this
# script): the eshkol-pgo-train/-merge/-verify CMake targets that actually
# drive `-fprofile-instr-generate` -> `llvm-profdata merge` ->
# `-fprofile-instr-use` over this corpus, plus the Phase 0 O3-assertion
# runner across every build mode (currently O2, and only for artifact
# builds). That is a real orchestration project, not a one-line follow-up.
#
# Honours $BUILD_DIR (CI passes it via the matrix).
#
# Exit codes:
#   0 — every corpus file passed (JIT ok, AOT ok, outputs match, non-empty).
#   1 — at least one corpus file failed one of those checks.
#   2 — infrastructure problem (binary missing, or the corpus itself is
#       empty/missing — NO_DATA is graded as a hard failure, never a silent
#       pass: a gate with nothing to check proves nothing).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
RUN="$BUILD_DIR/eshkol-run"
CORPUS_DIR="bench/pgo_corpus"

export ESHKOL_PATH="$PROJECT_ROOT/lib${ESHKOL_PATH:+:$ESHKOL_PATH}"

if [ ! -x "$RUN" ]; then
    echo "ERROR: $RUN not found or not executable (set BUILD_DIR?)." >&2
    exit 2
fi

ESHKOL_TEST_ISOLATION_NO_TRAP=1
ESHKOL_TEST_LIB="$SCRIPT_DIR/lib/test_isolation.sh"
if [ ! -r "$ESHKOL_TEST_LIB" ]; then
    echo "FATAL: cannot read $ESHKOL_TEST_LIB" >&2
    echo "       (the shared test isolation and disk-cleanup prelude)." >&2
    echo "       Refusing to run: without it this suite has no scratch isolation" >&2
    echo "       and no bounded disk footprint." >&2
    exit 2
fi
source "$ESHKOL_TEST_LIB"
eshkol_test_isolation_init "pgo-corpus-smoke"
trap eshkol_test_isolation_cleanup EXIT

echo "========================================="
echo "  bench/pgo_corpus smoke consumer (ADR 0007, Stage 1)"
echo "========================================="

if [ ! -d "$CORPUS_DIR" ]; then
    echo "NO_DATA: $CORPUS_DIR does not exist." >&2
    exit 2
fi

CORPUS_FILES=()
while IFS= read -r -d '' f; do
    CORPUS_FILES+=("$f")
done < <(find "$CORPUS_DIR" -maxdepth 1 -name '*.esk' -print0 | sort -z)

if [ "${#CORPUS_FILES[@]}" -eq 0 ]; then
    echo "NO_DATA: $CORPUS_DIR contains no .esk files — nothing to smoke-test." >&2
    echo "         (This is graded as a FAILURE, not a pass: an empty corpus" >&2
    echo "         means the gate certifies nothing.)" >&2
    exit 2
fi

FAILURES=0
JIT_OUT="$ESHKOL_TEST_TMPDIR/jit.out"
AOT_OUT="$ESHKOL_TEST_TMPDIR/aot.out"
AOT_BIN="$ESHKOL_TEST_TMPDIR/pgo_corpus_aot"

for f in "${CORPUS_FILES[@]}"; do
    name="$(basename "$f")"
    echo ""
    echo "--- $name ---------------------------------"

    file_ok=1

    # stdout and stderr are captured SEPARATELY here (unlike the AOT run
    # below, where the compiled binary emits no compiler diagnostics at
    # runtime). `eshkol-run -r` compiles and runs in one process, and a
    # cold LLVM JIT compile of this file prints "remark:"/"warning:"
    # vectorizer diagnostics to stderr — real, but not part of the
    # program's own output, and not reproducible run-to-run (a warm
    # on-disk JIT cache suppresses them on a recompile-free rerun). Folding
    # them into stdout via 2>&1 made the JIT/AOT parity check below flag a
    # false divergence on every cold-cache run. Only stdout is compared.
    if "$RUN" -r "$f" > "$JIT_OUT" 2>"$ESHKOL_TEST_TMPDIR/jit_stderr.log"; then
        if [ ! -s "$JIT_OUT" ]; then
            echo "$name: FAIL (JIT produced no output)"
            cat "$ESHKOL_TEST_TMPDIR/jit_stderr.log"
            file_ok=0
        fi
    else
        cat "$JIT_OUT" "$ESHKOL_TEST_TMPDIR/jit_stderr.log"
        echo "$name: FAIL (JIT exited non-zero)"
        file_ok=0
    fi

    if "$RUN" -o "$AOT_BIN" "$f" > "$ESHKOL_TEST_TMPDIR/compile.log" 2>&1; then
        # Same stdout/stderr split as the JIT run above: the tensor corpus
        # program's first Metal GPU dispatch logs verbose pipeline-calibration
        # diagnostics to stderr (interleaved with the program's own
        # `display` output when merged), so only stdout is compared for
        # parity.
        if "$AOT_BIN" > "$AOT_OUT" 2>"$ESHKOL_TEST_TMPDIR/aot_run_stderr.log"; then
            if [ ! -s "$AOT_OUT" ]; then
                echo "$name: FAIL (AOT produced no output)"
                cat "$ESHKOL_TEST_TMPDIR/aot_run_stderr.log"
                file_ok=0
            fi
        else
            cat "$AOT_OUT" "$ESHKOL_TEST_TMPDIR/aot_run_stderr.log"
            echo "$name: FAIL (AOT binary exited non-zero)"
            file_ok=0
        fi
    else
        cat "$ESHKOL_TEST_TMPDIR/compile.log"
        echo "$name: FAIL (AOT compile failed)"
        file_ok=0
    fi
    # `eshkol-run -o <bin>` also leaves a `<bin>.tmp.o` object file sibling
    # to the binary on every compile — remove both, not just the binary,
    # or five corpus files' worth of intermediate objects accumulate for
    # the whole run (disk-budget rule: every harness cleans up after
    # itself).
    rm -f -- "$AOT_BIN" "$AOT_BIN.tmp.o"

    if [ "$file_ok" -eq 1 ]; then
        if ! diff -u "$JIT_OUT" "$AOT_OUT" > "$ESHKOL_TEST_TMPDIR/diff.txt"; then
            echo "$name: FAIL (JIT and AOT output diverge)"
            cat "$ESHKOL_TEST_TMPDIR/diff.txt" | sed 's/^/  /'
            file_ok=0
        fi
    fi

    if [ "$file_ok" -eq 1 ]; then
        echo "$name: PASS"
        cat "$JIT_OUT" | sed 's/^/  /'
    else
        FAILURES=$((FAILURES + 1))
    fi
done

echo ""
echo "========================================="
if [ "$FAILURES" -eq 0 ]; then
    echo "pgo-corpus-smoke: PASS (${#CORPUS_FILES[@]} corpus file(s), JIT+AOT+parity)"
    exit 0
else
    echo "pgo-corpus-smoke: FAIL ($FAILURES of ${#CORPUS_FILES[@]} corpus file(s) failed)"
    exit 1
fi
