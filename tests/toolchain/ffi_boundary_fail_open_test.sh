#!/usr/bin/env bash
# ffi_boundary_fail_open_test.sh — the two fail-open cells found during
# FFI-boundary hardening must stay closed, on BOTH execution paths (-r and AOT),
# and the correct-arity calls they guard must still work end to end.
#
# ESH-0362 — an arity error must be FATAL, never continue with a null binding.
#   `(define h (process-spawn-argv argv))` against the two-parameter definition
#   printed the named diagnostic "Arity mismatch: process-spawn-argv expects 2
#   arguments but got 1" and then KEPT GOING: codegen returned nullptr without
#   marking the compilation fatal, so `h` was bound to null, `(process-pid h)`
#   answered 0, and the next consumer of the handle dereferenced NULL. The AOT
#   form was worse — it wrote a complete binary and exited 0, shipping the
#   poisoned program. Related cells in the same class: a surplus argument was
#   silently dropped, and under `-r` the REPL slot-call path synthesised the
#   callee's signature from the CALL's argument count, so the mismatch was not
#   even diagnosed — the callee read its missing parameter out of a stale
#   register.
#
# ESH-0363 — a wrong-typed FFI pointer argument must raise a catchable type
#   error, never be dereferenced.
#   `(run-argv-capture argv 5000)` — 5000 in the positional `cwd` slot — was
#   IntToPtr'd straight through as `const char*` and died with
#   "SIGSEGV at address 0x1388" (= 5000). Now the wrapper's parameter check
#   fires first, and the codegen boundary guard backstops every extern.
#
# Asserted here: exit status, the preserved diagnostic wording, the ABSENCE of a
# written binary on the AOT arity cell, that no cell dies by SIGNAL (128+n, and
# specifically never 139/SIGSEGV), that the type error is catchable by `guard`,
# and that a correctly-called spawn still runs /bin/echo and captures its output.
#
# Self-contained: synthesizes its own fixtures.

set -uo pipefail

ESHKOL_RUN="${1:-${ESHKOL_RUN:-}}"
if [ -z "$ESHKOL_RUN" ]; then
    if [ -x "./build/eshkol-run" ]; then
        ESHKOL_RUN="./build/eshkol-run"
    else
        echo "FAIL: ffi_boundary_fail_open_test could not locate eshkol-run" >&2
        exit 1
    fi
fi
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "FAIL: ffi_boundary_fail_open_test eshkol-run is not executable: $ESHKOL_RUN" >&2
    exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
# Isolate the persistent -r run-cache: a cache entry built before the fix would
# otherwise serve the poisoned binary and hide the regression.
export ESHKOL_JIT_CACHE_DIR="$tmp/jit"; mkdir -p "$ESHKOL_JIT_CACHE_DIR"

fail() { echo "FAIL: ffi_boundary_fail_open_test — $1" >&2; exit 1; }

# run <outfile> <cmd...> : run, capture stdout+stderr, echo the exit code.
run() {
    local out="$1"; shift
    "$@" >"$out" 2>&1
    echo $?
}

# A fault must be reported, not crashed into. Exit codes above 128 are
# "died from signal n" on every POSIX shell; 139 is SIGSEGV specifically.
assert_not_signal() {
    local label="$1" ec="$2" log="$3"
    if [ "$ec" -eq 139 ]; then
        fail "$label died with SIGSEGV (139) instead of reporting an error: $(head -3 "$log" | tr '\n' ' ')"
    fi
    if [ "$ec" -gt 128 ]; then
        fail "$label died from signal $((ec - 128)) instead of reporting an error"
    fi
}

# ── fixtures ────────────────────────────────────────────────────────────────

# ESH-0362 cell 1: too FEW arguments — cwd omitted from a fixed-arity spawn.
too_few="$tmp/arity_too_few.esk"
cat > "$too_few" <<'ESK'
(require agent.subprocess)
(define h (process-spawn-argv (list "echo" "hi")))
(display "pid: ") (display (process-pid h)) (newline)
ESK

# ESH-0362 cell 2: too MANY arguments to a plain user function. The surplus used
# to be dropped on the floor with only a gradual-typing WARNING.
too_many="$tmp/arity_too_many.esk"
cat > "$too_many" <<'ESK'
(define (add2 a b) (+ a b))
(display (add2 1 2 99))
(newline)
ESK

# ESH-0363 cell: an integer in the positional `cwd` slot.
wrong_type="$tmp/ffi_wrong_type.esk"
cat > "$wrong_type" <<'ESK'
(require agent.subprocess)
(define r (run-argv-capture (list "/bin/echo" "hi") 5000))
(display r) (newline)
ESK

# ESH-0363 catchability: the same mistake, intercepted by `guard`. The program
# must survive the error and exit 0 having printed its own marker.
catchable="$tmp/ffi_catchable.esk"
cat > "$catchable" <<'ESK'
(require agent.subprocess)
(display (guard (e (#t "caught"))
           (run-argv-capture (list "/bin/echo" "hi") 5000)
           "not-reached"))
(newline)
ESK

# ESH-0363 backstop: a user-declared extern, called directly with an integer in
# a `ptr` parameter. No wrapper is involved, so this exercises the CODEGEN
# boundary guard itself — the part that covers every extern in the language, not
# just the ones with a hand-written parameter check.
raw_extern="$tmp/ffi_raw_extern.esk"
cat > "$raw_extern" <<'ESK'
(extern ptr c-getenv ptr :real getenv)
(display (c-getenv 5000))
(newline)
ESK

# ESH-0363 must not over-reach: #f is the spelling of a NULL pointer argument,
# and a real string must still pass. Both have to keep working.
raw_extern_ok="$tmp/ffi_raw_extern_ok.esk"
cat > "$raw_extern_ok" <<'ESK'
(extern ptr c-getenv ptr :real getenv)
(define p (c-getenv "PATH"))
(display (if (null? p) "unset" "set"))
(newline)
(extern ptr c-strchr ptr i32 :real strchr)
(display (if (null? (c-strchr #f 65)) "null-ok" "null-ok"))
(newline)
ESK

# Positive control: correct arity, correct types, real child process.
happy="$tmp/ffi_happy.esk"
cat > "$happy" <<'ESK'
(require agent.subprocess)
(define r (run-argv-capture (list "/bin/echo" "hello-ffi-guard") "."))
(display "exit=") (display (cdr (assq 'exit-code r))) (newline)
(display "out=") (display (cdr (assq 'stdout r)))
(define h (process-spawn-argv (list "/bin/echo" "spawned") "."))
(display "spawn-ok=") (display (if (process-running? h) "yes" (if h "yes" "no"))) (newline)
(process-wait h 10000)
(process-destroy h)
ESK

# ── ESH-0362 cell 1: too few arguments, -r ──────────────────────────────────
log="$tmp/too_few_r.log"
ec="$(run "$log" "$ESHKOL_RUN" -r "$too_few")"
assert_not_signal "-r arity-too-few" "$ec" "$log"
[ "$ec" -ne 0 ] || fail "-r arity-too-few exited 0 (ESH-0362: must be fatal, not a null binding)"
grep -q "Arity mismatch: process-spawn-argv expects 2 arguments but got 1" "$log" \
    || fail "-r arity-too-few lost the named diagnostic (hard-won contract): $(head -5 "$log" | tr '\n' ' ')"
grep -q "^pid: " "$log" \
    && fail "-r arity-too-few still RAN the program (printed a pid from a null handle)"
echo "  ok: -r arity-too-few exited $ec, named diagnostic preserved, program never ran"

# ── ESH-0362 cell 1: too few arguments, AOT — and NO binary written ─────────
aot_bin="$tmp/too_few.bin"
rm -f "$aot_bin"
log="$tmp/too_few_aot.log"
ec="$(run "$log" "$ESHKOL_RUN" "$too_few" -o "$aot_bin")"
assert_not_signal "AOT arity-too-few" "$ec" "$log"
[ "$ec" -ne 0 ] || fail "AOT arity-too-few exited 0 (ESH-0362: shipped a poisoned binary)"
[ ! -e "$aot_bin" ] || fail "AOT arity-too-few wrote a binary $aot_bin (must write nothing)"
grep -q "Arity mismatch: process-spawn-argv expects 2 arguments but got 1" "$log" \
    || fail "AOT arity-too-few lost the named diagnostic"
echo "  ok: AOT arity-too-few exited $ec, wrote no binary, named diagnostic preserved"

# ── ESH-0362 cell 2: too many arguments, -r and AOT ─────────────────────────
log="$tmp/too_many_r.log"
ec="$(run "$log" "$ESHKOL_RUN" -r "$too_many")"
assert_not_signal "-r arity-too-many" "$ec" "$log"
[ "$ec" -ne 0 ] || fail "-r arity-too-many exited 0 (ESH-0362: surplus argument silently dropped)"
grep -q "Arity mismatch" "$log" \
    || fail "-r arity-too-many produced no arity diagnostic: $(head -5 "$log" | tr '\n' ' ')"
echo "  ok: -r arity-too-many exited $ec with an arity diagnostic"

many_bin="$tmp/too_many.bin"
rm -f "$many_bin"
log="$tmp/too_many_aot.log"
ec="$(run "$log" "$ESHKOL_RUN" "$too_many" -o "$many_bin")"
assert_not_signal "AOT arity-too-many" "$ec" "$log"
[ "$ec" -ne 0 ] || fail "AOT arity-too-many exited 0 (surplus argument silently dropped)"
[ ! -e "$many_bin" ] || fail "AOT arity-too-many wrote a binary (must write nothing)"
echo "  ok: AOT arity-too-many exited $ec and wrote no binary"

# ── ESH-0363: wrong-typed FFI pointer argument, -r ──────────────────────────
log="$tmp/wrong_type_r.log"
ec="$(run "$log" "$ESHKOL_RUN" -r "$wrong_type")"
assert_not_signal "-r wrong-type FFI arg" "$ec" "$log"
[ "$ec" -ne 0 ] || fail "-r wrong-type FFI arg exited 0"
grep -qi "cwd" "$log" \
    || fail "-r wrong-type FFI arg did not name the offending parameter: $(head -5 "$log" | tr '\n' ' ')"
grep -q "5000" "$log" \
    || fail "-r wrong-type FFI arg did not report the offending value: $(head -5 "$log" | tr '\n' ' ')"
echo "  ok: -r wrong-type FFI arg exited $ec naming \`cwd\` and the value 5000"

# ── ESH-0363: same cell under AOT (compiles; must fail at run time, no fault) ─
wt_bin="$tmp/wrong_type.bin"
rm -f "$wt_bin"
log="$tmp/wrong_type_aot_build.log"
ec="$(run "$log" "$ESHKOL_RUN" "$wrong_type" -o "$wt_bin")"
if [ "$ec" -eq 0 ] && [ -x "$wt_bin" ]; then
    log="$tmp/wrong_type_aot_run.log"
    ec="$(run "$log" "$wt_bin")"
    assert_not_signal "AOT wrong-type FFI arg" "$ec" "$log"
    [ "$ec" -ne 0 ] || fail "AOT wrong-type FFI arg exited 0"
    grep -qi "cwd" "$log" \
        || fail "AOT wrong-type FFI arg did not name the offending parameter"
    echo "  ok: AOT wrong-type FFI arg exited $ec without faulting"
else
    echo "  skip: AOT link unavailable here (build exit $ec) — -r cell already asserted"
fi

# ── ESH-0363: the type error is CATCHABLE ───────────────────────────────────
log="$tmp/catchable.log"
ec="$(run "$log" "$ESHKOL_RUN" -r "$catchable")"
assert_not_signal "-r catchable FFI type error" "$ec" "$log"
if [ "$ec" -eq 0 ] && grep -q "caught" "$log"; then
    echo "  ok: FFI type error caught by guard, program continued and exited 0"
else
    fail "FFI type error was not catchable by guard (exit $ec): $(head -5 "$log" | tr '\n' ' ')"
fi

# ── ESH-0363 backstop: codegen guard on a bare user extern ──────────────────
log="$tmp/raw_extern_r.log"
ec="$(run "$log" "$ESHKOL_RUN" -r "$raw_extern")"
assert_not_signal "-r bare-extern integer in ptr param" "$ec" "$log"
[ "$ec" -ne 0 ] || fail "-r bare-extern integer in ptr param exited 0 (codegen guard did not fire)"
grep -q "FFI type error" "$log" \
    || fail "-r bare-extern cell did not raise the FFI boundary type error: $(head -5 "$log" | tr '\n' ' ')"
grep -q "c-getenv" "$log" \
    || fail "-r bare-extern cell did not name the extern"
grep -q "argument 1" "$log" \
    || fail "-r bare-extern cell did not name the argument position"
echo "  ok: -r bare-extern integer in ptr param raised the boundary type error naming c-getenv argument 1"

# The guard must not reject the legal spellings it sits next to.
log="$tmp/raw_extern_ok.log"
ec="$(run "$log" "$ESHKOL_RUN" -r "$raw_extern_ok")"
assert_not_signal "-r bare-extern legal pointer args" "$ec" "$log"
[ "$ec" -eq 0 ] || fail "-r bare-extern legal pointer args regressed (exit $ec): $(head -5 "$log" | tr '\n' ' ')"
grep -q "null-ok" "$log" \
    || fail "-r bare-extern: #f as a NULL pointer argument was rejected: $(head -5 "$log" | tr '\n' ' ')"
echo "  ok: -r bare-extern accepts a string and #f (NULL) in ptr params"

# ── positive control: correct arity + correct types still work end to end ───
log="$tmp/happy_r.log"
ec="$(run "$log" "$ESHKOL_RUN" -r "$happy")"
[ "$ec" -eq 0 ] || fail "-r correct-arity spawn regressed (exit $ec): $(head -10 "$log" | tr '\n' ' ')"
grep -q "exit=0" "$log" || fail "-r correct-arity spawn: /bin/echo did not report exit 0"
grep -q "hello-ffi-guard" "$log" || fail "-r correct-arity spawn did not capture child stdout"
grep -q "spawn-ok=yes" "$log" || fail "-r correct-arity process-spawn-argv returned no handle"
echo "  ok: -r correct-arity spawn ran /bin/echo and captured its output"

happy_bin="$tmp/happy.bin"
rm -f "$happy_bin"
log="$tmp/happy_aot_build.log"
ec="$(run "$log" "$ESHKOL_RUN" "$happy" -o "$happy_bin")"
if [ "$ec" -eq 0 ] && [ -x "$happy_bin" ]; then
    log="$tmp/happy_aot_run.log"
    ec="$(run "$log" "$happy_bin")"
    [ "$ec" -eq 0 ] || fail "AOT correct-arity spawn regressed (exit $ec): $(head -10 "$log" | tr '\n' ' ')"
    grep -q "hello-ffi-guard" "$log" || fail "AOT correct-arity spawn did not capture child stdout"
    echo "  ok: AOT correct-arity spawn ran /bin/echo and captured its output"
else
    echo "  skip: AOT link unavailable here (build exit $ec) — -r positive control already asserted"
fi

echo "PASS: ffi_boundary_fail_open_test"
exit 0
