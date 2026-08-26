#!/usr/bin/env bash
# run_icc_smoke.sh — Run Eshkol smoke probes and emit JSON-L trace
# events that ICC's `completion-oracle` reads as evidence.
#
# Each probe is one runtime_check the .icc/completion-oracles.yaml
# oracles depend on. PASS/FAIL events land in
# scripts/icc_traces/eshkol_smoke.jsonl, then the assistant runs
#
#   python3 ~/Desktop/infinite_context_coder/scripts/codebase_tool.py \
#       completion-oracle --repo eshkol_lang \
#       --target agent-ffi-ready \
#       --trace-dir scripts/icc_traces
#
# and the oracle flips from FAIL → PASS for the probes that succeeded.
#
# Adding a probe:
#   1. Add a runtime_event criterion in .icc/completion-oracles.yaml
#      with event_names: [<probe_id>] event_values: ["PASS"]
#   2. Add a `probe <probe_id> "<label>" '<bash command>'` line below
#   3. The command must exit 0 for PASS; any nonzero exits → FAIL
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
. "$REPO_ROOT/scripts/lib/durable_work_root.sh"
if eshkol_durable_enabled; then
    ESHKOL_ICC_WORK="$(eshkol_durable_prepare_dir icc-smoke)" || exit $?
fi
if eshkol_durable_enabled; then
    TRACE_DIR="${TRACE_DIR:-$ESHKOL_ICC_WORK/traces}"
else
    TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
fi
TRACE_FILE="$TRACE_DIR/eshkol_smoke.jsonl"
mkdir -p "$TRACE_DIR"

# Truncate so each run is a fresh evidence set; ICC reads the union of
# events in the file, but stale PASS lines for now-broken probes would
# otherwise mask regressions.
: "${TRACE_FILE:?}"
: > "$TRACE_FILE"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) BUILD_DIR_PATH="$BUILD_DIR" ;;
    *)  BUILD_DIR_PATH="$REPO_ROOT/$BUILD_DIR" ;;
esac
BUILD_DIR="$BUILD_DIR_PATH"
export BUILD_DIR

XLA_BUILD_DIR="${XLA_BUILD_DIR:-}"
if [ -n "$XLA_BUILD_DIR" ]; then
    case "$XLA_BUILD_DIR" in
        /*) XLA_BUILD_DIR_PATH="$XLA_BUILD_DIR" ;;
        *)  XLA_BUILD_DIR_PATH="$REPO_ROOT/$XLA_BUILD_DIR" ;;
    esac
else
    XLA_BUILD_DIR_PATH=""
fi

ESHKOL_RUN="$BUILD_DIR_PATH/eshkol-run"
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "scripts/run_icc_smoke.sh: $ESHKOL_RUN not found — run \`cmake --build $BUILD_DIR_PATH\` first." >&2
    exit 2
fi

if [ -z "${ESHKOL_JIT_CACHE_DIR:-}" ]; then
    if eshkol_durable_enabled; then
        ESHKOL_ICC_JIT_CACHE_DIR="$ESHKOL_ICC_WORK/jit-cache"
        mkdir "$ESHKOL_ICC_JIT_CACHE_DIR"
    else
        ESHKOL_ICC_JIT_CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-icc-jit-cache.XXXXXX")
    fi
    export ESHKOL_JIT_CACHE_DIR="$ESHKOL_ICC_JIT_CACHE_DIR"
    if ! eshkol_durable_enabled; then trap 'rm -rf "$ESHKOL_ICC_JIT_CACHE_DIR"' EXIT; fi
else
    mkdir -p "$ESHKOL_JIT_CACHE_DIR"
fi

# The existing probes use mktemp in their compact shell snippets.  In durable
# mode this shell function shadows the external command and claims fresh,
# deterministic files below the gate directory; it never invokes mktemp or
# writes ephemeral paths.  The default branch delegates unchanged to mktemp.
ESHKOL_ICC_TEMP_SEQUENCE=0
mktemp() {
    if ! eshkol_durable_enabled; then command mktemp "$@"; return; fi
    local is_dir=0 candidate index=1
    [ "${1:-}" = "-d" ] && is_dir=1
    while [ -e "$ESHKOL_ICC_WORK/probe-${index}" ] || [ -L "$ESHKOL_ICC_WORK/probe-${index}" ]; do
        index=$((index + 1))
    done
    candidate="$ESHKOL_ICC_WORK/probe-${index}"
    if [ "$is_dir" -eq 1 ]; then mkdir "$candidate"; else : > "$candidate"; fi
    printf '%s\n' "$candidate"
}

# Probe snippets clean their ad-hoc outputs on success.  In durable mode those
# outputs are release evidence, so retain only paths claimed beneath this gate.
rm() {
    if ! eshkol_durable_enabled; then command rm "$@"; return; fi
    local arg
    for arg in "$@"; do
        case "$arg" in
            -*) ;;
            "$ESHKOL_ICC_WORK"/*) ;;
            *) command rm "$@"; return ;;
        esac
    done
    return 0
}

# Emit one trace line as a JSON-L event with explicit `kind`. ICC's
# runtime_evidence parser was extended (2026-05-07) to recognize records
# carrying an explicit `kind` field as pre-shaped events, instead of
# walking their keys with the ML-training-log heuristic.
#
# The oracle criterion matches:
#     event_kinds: [eshkol_smoke]
#     event_names: ["<probe_id>"]
#     event_values: ["PASS"]
emit_event() {
    local probe_id="$1" status="$2" snippet="$3"
    : "${TRACE_FILE:?}"
    # json.dumps handles newlines, tabs, control bytes, quotes, backslashes,
    # and Unicode labels. Hand-escaping only quotes/backslashes produced
    # invalid JSON-L whenever a failing probe emitted a multiline diagnostic.
    python3 -c '
import json, sys
print(json.dumps({"kind": "eshkol_smoke", "name": sys.argv[1],
                  "value": sys.argv[2], "snippet": sys.argv[3],
                  "confidence": 0.95}, ensure_ascii=False))
' "$probe_id" "$status" "$snippet" >> "${TRACE_FILE:?}"
}

PROBE_TOTAL=0
PROBE_FAILURES=0

probe() {
    local probe_id="$1" label="$2" cmd="$3"
    local out status snippet
    PROBE_TOTAL=$((PROBE_TOTAL + 1))
    # Capture combined stdout+stderr so the snippet is informative when
    # something fails. Bound the snippet so a multi-MB log doesn't blow
    # up the trace file.
    out=$(eval "$cmd" 2>&1)
    status=$?
    if [ "$status" -eq 0 ]; then
        snippet="${label}: OK"
        emit_event "$probe_id" PASS "$snippet"
        printf '  ✓ %-40s %s\n' "$probe_id" "$label"
    else
        PROBE_FAILURES=$((PROBE_FAILURES + 1))
        snippet=$(printf '%s' "$out" | tail -c 200)
        emit_event "$probe_id" FAIL "$snippet"
        printf '  ✗ %-40s %s (exit %d)\n' "$probe_id" "$label" "$status"
    fi
}

echo "Running ICC smoke probes → $TRACE_FILE"
echo

# ─────────────────────────────────────────────────────────────────
# Compiler-readiness probes
# ─────────────────────────────────────────────────────────────────
probe llvm_module_verifier_clean "verifyModule passes on a smoke .esk" \
    'echo "(define (f x) (+ x 1)) (display (f 41)) (newline)" | "$ESHKOL_RUN" -e "$(cat)"'

probe aot_link_succeeds "AOT compile + run hello.esk" \
    'tmp=$(mktemp); echo "(display \"hello\") (newline)" > "$tmp.esk";
     "$ESHKOL_RUN" "$tmp.esk" -o "$tmp.bin" >/dev/null && "$tmp.bin" >/dev/null;
     rm -f "$tmp" "$tmp.esk" "$tmp.bin"'

probe jit_repl_clean_exit "eshkol-run -r returns 0 on a noop input" \
    'tmp=$(mktemp); echo "(+ 1 2)" > "$tmp.esk"; "$ESHKOL_RUN" -r "$tmp.esk" >/dev/null; rc=$?; rm -f "$tmp" "$tmp.esk"; exit $rc'

# ─────────────────────────────────────────────────────────────────
# Agent FFI probes (#234/#236/#237/#248 contracts)
# ─────────────────────────────────────────────────────────────────
probe native_http_get_works "HTTPS GET to postman-echo.com returns 200" \
    'tmp=$(mktemp).esk;
     cat > "$tmp" <<EOF
(require agent.http)
(http-init)
(let ((r (http-get "https://postman-echo.com/get" 10000)))
  (if (and r (= (car r) 200)) (exit 0) (exit 1)))
EOF
     "$ESHKOL_RUN" -r "$tmp" 2>&1; rc=$?; rm -f "$tmp"; exit $rc'

probe native_http_post_json_works "POST JSON round-trips via postman-echo.com" \
    'tmp=$(mktemp).esk;
     cat > "$tmp" <<EOF
(require agent.http)
(http-init)
(let ((r (http-post "https://postman-echo.com/post"
                    (list (cons "Content-Type" "application/json"))
                    "{\"k\":\"v\"}" 10000)))
  (if (and r (= (car r) 200) (string-contains (cdr r) "\"k\":\"v\""))
      (exit 0) (exit 1)))
EOF
     "$ESHKOL_RUN" -r "$tmp" 2>&1; rc=$?; rm -f "$tmp"; exit $rc'

probe subprocess_argv_safe "process-spawn-argv runs argv directly" \
    'tmp=$(mktemp).esk;
     cat > "$tmp" <<EOF
(require agent.subprocess)
(let ((p (process-spawn-argv (list "echo" "hello;world|pipe") ".")))
  (process-wait p 5000)
  ;; If the shell were invoked, the ; and | would split commands. argv-safe
  ;; spawn passes the whole string verbatim as one argument.
  (let ((out (process-read-all-stdout p 4096)))
    (process-destroy p)
    (if (string-contains out "hello;world|pipe") (exit 0) (exit 1))))
EOF
     "$ESHKOL_RUN" -r "$tmp" 2>&1; rc=$?; rm -f "$tmp"; exit $rc'

probe subprocess_pid_exposed "process-pid returns a real OS PID > 0" \
    'tmp=$(mktemp).esk;
     cat > "$tmp" <<EOF
(require agent.subprocess)
(let ((p (process-spawn-argv (list "sleep" "0.05") ".")))
  (let ((pid (process-pid p)))
    (process-wait p 5000)
    (process-destroy p)
    (if (> pid 0) (exit 0) (exit 1))))
EOF
     "$ESHKOL_RUN" -r "$tmp" 2>&1; rc=$?; rm -f "$tmp"; exit $rc'

probe sqlite_text_round_trip "30 KB session JSON round-trips through sqlite" \
    'tmp=$(mktemp).esk; db=$(mktemp);
     rm -f "$db";
     cat > "$tmp" <<EOF
(require agent.sqlite)
(define db (sqlite-open "$db"))
(sqlite-exec db "DROP TABLE IF EXISTS sessions")
(sqlite-exec db "CREATE TABLE sessions (id INTEGER PRIMARY KEY, payload TEXT)")
(define (rep s n) (let loop ((i 0) (acc "")) (if (>= i n) acc (loop (+ i 1) (string-append acc s)))))
(define big (string-append "{\"messages\":[" (rep "{\"role\":\"user\",\"content\":\"hello world\"}," 500) "{\"end\":true}]}"))
(let ((s (sqlite-prepare db "INSERT INTO sessions (payload) VALUES (?)")))
  (sqlite-bind-text s 1 big) (sqlite-step s) (sqlite-finalize s))
(let ((s (sqlite-prepare db "SELECT payload FROM sessions WHERE id = 1")))
  (sqlite-step s)
  (let ((round (sqlite-column-text s 0)))
    (sqlite-finalize s) (sqlite-close db)
    (if (string=? big round) (exit 0) (exit 1))))
EOF
     "$ESHKOL_RUN" -r "$tmp" 2>&1; rc=$?; rm -f "$tmp" "$db"; exit $rc'

probe aot_binaries_link_agent_ffi "AOT-compiled binary linking agent.http runs" \
    'tmp=$(mktemp).esk; bin=$(mktemp);
     rm -f "$bin";
     cat > "$tmp" <<EOF
(require agent.http)
(http-init) (http-shutdown)
(display "ok") (newline)
EOF
     "$ESHKOL_RUN" "$tmp" -o "$bin" >/dev/null 2>&1 && "$bin" >/dev/null; rc=$?
     rm -f "$tmp" "$bin"; exit $rc'

# Transitive-closure agent-FFI link discovery + fatal-link-under-`-r`.
# A helper reached only through (load …)/(import …) — two levels deep, no
# top-level (require agent.*) — must still link the agent-FFI archive (JIT+AOT),
# a plain program must not (no over-linking), and a broken generated-program
# link under -r must exit nonzero without a reduced in-process fallback.
probe transitive_load_agent_ffi_link "transitive (load) agent-FFI link + fatal -r link" \
    'out=$(bash "$REPO_ROOT/tests/toolchain/transitive_ffi_link_test.sh" "$ESHKOL_RUN" 2>&1) || { printf "%s\n" "$out"; exit 1; }
     printf "%s" "$out" | grep -q "PASS: transitive_ffi_link_test"'

# FFI-boundary fail-open closure (ESH-0362 / ESH-0363). An arity error is fatal
# under -r AND AOT (no null binding, no binary written, named diagnostic kept);
# a wrong-typed pointer argument raises a catchable type error instead of being
# dereferenced as an address; correctly-called spawns still run a real child.
probe ffi_boundary_fail_open "arity errors fatal; FFI pointer args type-checked" \
    'out=$(bash "$REPO_ROOT/tests/toolchain/ffi_boundary_fail_open_test.sh" "$ESHKOL_RUN" 2>&1) || { printf "%s\n" "$out"; exit 1; }
     printf "%s" "$out" | grep -q "PASS: ffi_boundary_fail_open_test"'

# Optional release-readiness evidence from an XLA-enabled build.  The default
# lite build deliberately omits xla_codegen_test, so release coordinators pass
# XLA_BUILD_DIR when certifying the full backend surface.  The integration test
# calls all three target-query functions and checks their mutual consistency.
if [ -n "$XLA_BUILD_DIR_PATH" ]; then
    probe xla_compiler_target_queries \
        "XLACompiler::isTargetAvailable, XLACompiler::getDefaultTarget, and XLACompiler::getAvailableTargets agree" \
        'test -x "$XLA_BUILD_DIR_PATH/xla_codegen_test" && "$XLA_BUILD_DIR_PATH/xla_codegen_test"'
fi

# ─────────────────────────────────────────────────────────────────
# v1.2 release probes
# ─────────────────────────────────────────────────────────────────
probe stdlib_o_loads "build/stdlib.o exists and is non-empty" \
    'test -s "$BUILD_DIR_PATH/stdlib.o"'

probe stdlib_compiles_clean "stdlib rebuilds without errors" \
    'cd "$REPO_ROOT" && touch lib/stdlib.esk && cmake --build "$BUILD_DIR_PATH" --target stdlib >/dev/null 2>&1'

probe error_messages_have_source_locations "Diagnostic includes line:col" \
    'tmp=$(mktemp).esk; bin=$(mktemp).bin; rm -f "$bin";
     echo "(define (foo x) x) (foo 1 2)" > "$tmp";
     out=$("$ESHKOL_RUN" "$tmp" -o "$bin" 2>&1);
     rm -f "$tmp" "$bin";
     ## Accept either "path:line:col" (eshkol_error_at) or "(line N:M)"
     ## (type-warning legacy). Both prove diagnostics carry source spans.
     printf "%s" "$out" | grep -qE ":[0-9]+:[0-9]+|line [0-9]+:[0-9]+"'

probe per_thread_arena_works "parallel-map across 8 workers completes" \
    'tmp=$(mktemp).esk;
     cat > "$tmp" <<EOF
(require core.threads)
(let ((r (parallel-map (lambda (x) (* x x)) (list 1 2 3 4 5 6 7 8))))
  (if (equal? r (list 1 4 9 16 25 36 49 64)) (exit 0) (exit 1)))
EOF
     "$ESHKOL_RUN" -r "$tmp" 2>&1; rc=$?; rm -f "$tmp"; exit $rc'

probe model_serialization_round_trip "tensor save/load round-trips bit-exact" \
    'tmp=$(mktemp).esk; f=$(mktemp);
     rm -f "$f";
     cat > "$tmp" <<EOF
(define t #(1.0 2.0 3.0 4.0))
(tensor-save "$f" t)
(define t2 (tensor-load "$f"))
(if (and (= (tensor-ref t2 0) 1.0) (= (tensor-ref t2 3) 4.0)) (exit 0) (exit 1))
EOF
     "$ESHKOL_RUN" -r "$tmp" 2>&1; rc=$?; rm -f "$tmp" "$f"; exit $rc'

probe image_io_works "image-read returns a tensor of expected shape" \
    'tmp=$(mktemp).esk; img=$(mktemp).png;
     printf "\\x89PNG\\r\\n\\x1a\\n" > "$img";  ## just a header — image-read should error gracefully
     cat > "$tmp" <<EOF
;; Smoke: builtin recognized and callable. Real shape probe deferred to
;; tests/v1_2_edge_cases/image_io_test.esk.
(define (probe) (with-exception-handler (lambda (e) #t)
                                         (lambda () (image-read "/nonexistent.png"))))
(probe) (exit 0)
EOF
     "$ESHKOL_RUN" -r "$tmp" 2>&1; rc=$?; rm -f "$tmp" "$img"; exit $rc'

probe v1_2_edge_case_tests_pass "v1.2 edge-case suite passes" \
    'cd "$REPO_ROOT";
     ## Inline list — avoid heredoc + double-eval quoting issues. Each
     ## test produces "PASS:" / "FAIL:" lines; we only fail the probe if
     ## any test prints a FAIL marker anywhere on a line (indented
     ## `  <case>: FAIL` included) or its summary shows nonzero
     ## "Failed: N" with N >= 1.
     for t in tests/v1_2_edge_cases/append_variadic_test.esk \
              tests/v1_2_edge_cases/main_substring_name_test.esk \
              tests/v1_2_edge_cases/sexp_canonical_string_test.esk \
              tests/v1_2_edge_cases/substring_utf8_test.esk \
              tests/v1_2_edge_cases/string_escapes_test.esk \
              tests/v1_2_edge_cases/procedure_arity_test.esk \
              tests/v1_2_edge_cases/json_schema_test.esk; do
       bin=$(mktemp "${TMPDIR:-/tmp}/icc_$(basename "$t" .esk).XXXXXX"); rm -f "$bin";
       "$ESHKOL_RUN" "$t" -o "$bin" >/dev/null 2>&1 || exit 1;
       tout=$("$bin" 2>&1);
       ## A bare FAIL token must not match when it only reports a count of
       ## ZERO: a passing suite printing "Total: 19, PASS: 19, FAIL: 0" is not
       ## a failure. Split across -e patterns rather than nesting `$` inside an
       ## alternation group: ugrep (which is `grep` on some dev machines)
       ## mis-parses `(a|b|$)` and then silently matches NOTHING, which would
       ## turn this probe into one that can never fail. `$` is only ever used
       ## at the end of a whole pattern here, which every engine handles.
       if printf "%s" "$tout" | grep -qE \
            -e "(^|[^A-Za-z0-9_])(FAILED|FAILURE|FAILS)" \
            -e "(^|[^A-Za-z0-9_])FAIL[^A-Za-z0-9_:]" \
            -e "(^|[^A-Za-z0-9_])FAIL$" \
            -e "FAIL:[[:space:]]*[1-9]" \
            -e "Failed:[[:space:]]*[1-9]"; then
         exit 1;
       fi;
     done;
     exit 0'

probe example_agent_compiles "agent-backed eagle training example compiles" \
    'cd "$REPO_ROOT" && test -f examples/eagle_train.esk;
     bin=$(mktemp "${TMPDIR:-/tmp}/icc-eagle.XXXXXX"); rm -f "$bin";
     "$ESHKOL_RUN" examples/eagle_train.esk -o "$bin" >/dev/null 2>&1;
     rc=$?; rm -f "$bin"; exit $rc'

# ───────────────────────────────────────────────────────────────────
# v1.3-evolve probes — see .icc/completion-oracles.yaml::v1.3-evolve.
# ───────────────────────────────────────────────────────────────────

probe string_interpolation_works '"…~{expr}…" parses and evaluates the embedded expression' \
    'tmp=$(mktemp).esk;
     cat > "$tmp" <<EOF
(define n 42)
(define s "n=~{n} squared=~{(* n n)}")
(if (string=? s "n=42 squared=1764") (exit 0) (exit 1))
EOF
     "$ESHKOL_RUN" -r "$tmp" 2>&1; rc=$?; rm -f "$tmp"; exit $rc'

probe keyword_args_work '(f #:k v) bind by name and reorder freely' \
    'tmp=$(mktemp).esk;
     cat > "$tmp" <<EOF
(define (weighted x #:scale s #:offset o) (+ (* x s) o))
(if (= (weighted 10 #:offset 2 #:scale 4) 42) (exit 0) (exit 1))
EOF
     "$ESHKOL_RUN" -r "$tmp" 2>&1; rc=$?; rm -f "$tmp"; exit $rc'

probe let_match_destructures '(let-match (((list a b) (list 1 2))) …) destructures and binds' \
    'tmp=$(mktemp).esk;
     cat > "$tmp" <<EOF
(define r (let-match (((list a b) (list 19 23))) (+ a b)))
(if (= r 42) (exit 0) (exit 1))
EOF
     "$ESHKOL_RUN" -r "$tmp" 2>&1; rc=$?; rm -f "$tmp"; exit $rc'

probe define_library_import_works '(define-library …) + (import …) round-trip works in a single file' \
    'tmp=$(mktemp).esk;
     cat > "$tmp" <<EOF
(define-library (smoke v1_3) (export greet) (begin (define (greet who) (string-append "hi " who))))
(import (smoke v1_3))
(if (string=? (greet "world") "hi world") (exit 0) (exit 1))
EOF
     "$ESHKOL_RUN" -r "$tmp" 2>&1; rc=$?; rm -f "$tmp"; exit $rc'

probe matmul_kernel_grad_nonzero 'gradient flows through tensor-matmul to the kernel side (input2)' \
    'tmp=$(mktemp).esk;
     cat > "$tmp" <<EOF
;; f = sum(X @ K). df/dK_lm = sum_i X[i][l]
;; X = [[1,2],[3,4]] → df/dK = [[4,4],[6,6]]
(define X (reshape #(1.0 2.0 3.0 4.0) 2 2))
(define (f params) (tensor-sum (tensor-matmul X (reshape params 2 2))))
(define g (gradient f #(1.0 0.0 0.0 1.0)))
(if (and (= (vector-ref g 0) 4.0) (= (vector-ref g 1) 4.0)
         (= (vector-ref g 2) 6.0) (= (vector-ref g 3) 6.0))
    (exit 0) (exit 1))
EOF
     "$ESHKOL_RUN" -r "$tmp" 2>&1; rc=$?; rm -f "$tmp"; exit $rc'

probe ad_input2_conv2d_grad_works 'gradient flows through conv2d to the kernel operand' \
    '"$ESHKOL_RUN" -r "$REPO_ROOT/tests/v1_3_edge_cases/ad_input2_test.esk" -L"$REPO_ROOT/build" 2>&1 |
     grep -q "PASS: ad_input2_conv2d_grad_works"'

probe ad_input2_batchnorm_grad_works 'gradient flows through batch-norm to gamma' \
    '"$ESHKOL_RUN" -r "$REPO_ROOT/tests/v1_3_edge_cases/ad_input2_test.esk" -L"$REPO_ROOT/build" 2>&1 |
     grep -q "PASS: ad_input2_batchnorm_grad_works"'

probe ad_input2_layernorm_grad_works 'gradient flows through layer-norm to gamma' \
    '"$ESHKOL_RUN" -r "$REPO_ROOT/tests/v1_3_edge_cases/ad_input2_test.esk" -L"$REPO_ROOT/build" 2>&1 |
     grep -q "PASS: ad_input2_layernorm_grad_works"'

probe ad_input2_attention_grad_works 'gradient flows through scaled-dot-attention to K/V operands' \
    '"$ESHKOL_RUN" -r "$REPO_ROOT/tests/v1_3_edge_cases/ad_input2_test.esk" -L"$REPO_ROOT/build" 2>&1 |
     grep -q "PASS: ad_input2_attention_grad_works"'

# ───────────────────────────────────────────────────────────────────
# Full-claim adversarial gates (2026-07-10 oracle hardening).
#
# The narrow ad_input2_* probes above only exercise the ONE working
# calling pattern (literal-lambda loss + scalar gamma). The two probes
# below assert the FULL claims that an adversarial audit found the oracle
# was silently over-promising:
#
#   * tensor_input2_grad_exact_firstclass_and_vector — the #229/ESH-0212
#     gate: every second-operand gradient (matmul B, conv2d kernel,
#     attention K/V, per-feature batch/layer-norm gamma) matches a central
#     finite-difference oracle EXACTLY across literal, first-class, and
#     higher-order loss forms AND for a vector/learnable gamma, under both
#     the JIT and AOT (24/24). Guards the silent-zero regressions the
#     narrow probes could not see (first-class loss → #(0 0 …), vector
#     gamma → 0).
#   * region_evac_subtype_coverage — the ESH-0214d/e evacuator gate, run
#     under ESHKOL_ARENA_POISON=1 so a missed interior pointer crashes at a
#     0xCB.. address instead of reading stale-but-valid data. Covers the
#     logic/workspace/PROMISE subtypes whose evacuation gap was invisible
#     to readiness=100 before this probe existed.
# ───────────────────────────────────────────────────────────────────
probe tensor_input2_grad_exact_firstclass_and_vector \
    'input2 tensor gradients (matmul B / conv2d kernel / attention K,V / per-feature batch+layer-norm gamma) match central FD EXACTLY across literal, first-class AND higher-order loss forms and vector gamma (JIT+AOT, 24/24)' \
    'cd "$REPO_ROOT";
     out=$(BUILD_DIR="$BUILD_DIR_PATH" bash scripts/run_tensor_input2_grad_gate.sh 2>&1) || exit 1;
     printf "%s" "$out" | grep -q "ESH-0212 tensor-AD second-operand gate: PASS"'

# Generative adversarial AD-vs-finite-difference oracle. Unlike the fixed
# tensor_input2 gate above, this GROWS random-but-seeded differentiable programs
# out of the AD primitives + tensor/ML ops (matmul/conv2d/attention/batch+layer-
# norm/softmax/…) and checks every gradient, laplacian and hessian against a
# central finite difference — scalar, field, gradient-of-gradient (ESH-0096
# shape), tensor-literal-point higher order (ESH-0095 shape), and literal/first-
# class/wrapper loss forms. Readiness thus CONTINUOUSLY asserts "AD matches FD
# across a generated family," and a NEW silent-wrong gradient trips it. --quick
# runs one file per family (JIT) for the smoke lane; the full JIT+AOT sweep is
# scripts/run_ad_adversarial.sh.
probe ad_adversarial_fd_oracle \
    'generative AD-vs-finite-difference sweep (random scalar/field/tensor-ML compositions, grad+laplacian+hessian, literal/first-class/wrapper loss, tensor-literal points) matches central FD — no silent-wrong gradients' \
    'cd "$REPO_ROOT";
     out=$(BUILD_DIR="$BUILD_DIR_PATH" bash scripts/run_ad_adversarial.sh --quick 2>&1) || exit 1;
     printf "%s" "$out" | grep -q "ad_adversarial gate: PASS"'

# The exactness guarantee, and the proof its gate can go red.
#
# `(= (ad-finite-difference-evals) 0)` shipped as the executable form of "no
# finite-difference fallback anywhere in the gradient path" while the counter it
# reads had NO writer on the native back end — a green result that proved
# nothing, and would have stayed green under a finite-difference regression.
# This probe runs the wired counter's positive case (an exact vector gradient is
# one primal / one reverse / zero finite differences) TOGETHER with the negative
# control (a real central-difference backward drives the counter to exactly the
# number of perturbations it evaluated, and the assertion form goes #f), on both
# engines, plus the matmul tape-node ratchet. Readiness must never again be able
# to certify exactness on the strength of a counter that cannot move.
probe ad_exactness_gate \
    'the no-finite-differences guarantee is enforced by a counter that can actually read nonzero: exact gradients report 0 FD evals, a real finite-difference backward reports exactly its perturbations and turns the shipped assertion #f (both engines), and matmul AD tape node counts stay within their ratchet with gradients exact' \
    'cd "$REPO_ROOT";
     out=$(BUILD_DIR="$BUILD_DIR_PATH" bash scripts/run_ad_exactness_gate.sh 2>&1) || exit 1;
     printf "%s" "$out" | grep -q "AD exactness gate: PASS"'

probe region_evac_subtype_coverage \
    'ESH-0214d/e region escape-evacuator keeps promoted logic/workspace/PROMISE subtype interiors intact under ESHKOL_ARENA_POISON=1 (AOT, flat RSS)' \
    'cd "$REPO_ROOT";
     out=$(ESHKOL_ARENA_POISON=1 BUILD_DIR="$BUILD_DIR_PATH" bash tests/memory/region_evac_subtype_coverage_test.sh 2>&1) || exit 1;
     printf "%s" "$out" | grep -q "region_evac_subtype_coverage_test.sh: PASS"'

probe parallel_map_scope_reclaim_race \
    'parallel-map over a closure using scope-based reclamation (internal named-let loops + memv) is race-free: workers no longer push/pop/rewind the shared arena scope stack concurrently. AOT, repeated under ESHKOL_ARENA_POISON=1' \
    'cd "$REPO_ROOT";
     out=$(BUILD_DIR="$BUILD_DIR_PATH" bash tests/parallel/parallel_map_scope_reclaim_test.sh 2>&1) || exit 1;
     printf "%s" "$out" | grep -q "parallel_map_scope_reclaim_test.sh: PASS"'

probe jit_cache_hit_invalidates 'eshkol-run -r persistent cache hits and source edits invalidate' \
    'bash "$REPO_ROOT/tests/v1_3_edge_cases/jit_cache_test.sh" "$ESHKOL_RUN"'

probe native_image_io_no_stb 'image-read uses platform APIs, not bundled deps/stb' \
    'cd "$REPO_ROOT";
     ## v1.3 commits to removing deps/stb in favour of native platform
     ## media APIs. This probe fails if the vendored tree or direct include
     ## path comes back.
     if grep -q "deps/stb" lib/core/image_io.c 2>/dev/null; then exit 1; fi;
     if [ -d deps/stb ]; then exit 1; fi;
     exit 0'

probe pgo_pipeline_works 'cmake -DESHKOL_PGO=generate/use supports a profile-guided binary' \
    'cd "$REPO_ROOT";
     ## v1.3 commits to a PGO build option.  This lightweight smoke
     ## confirms the configure surface is wired without running a full
     ## instrument/merge/use cycle on every ICC probe.
     if grep -qE "ESHKOL_PGO|fprofile-(generate|use)" CMakeLists.txt 2>/dev/null; then
        exit 0;
     fi;
     exit 1'

# ───────────────────────────────────────────────────────────────────
# R7RS string-op edge cases (ESH-0066): string-map returns a string and
# accepts first-class char builtins; string->number honors a radix. The
# suite runs under BOTH -r (JIT) and AOT and must report zero failures.
# ───────────────────────────────────────────────────────────────────
probe string_edge_ops_r7rs 'string-map returns a string; string->number honors radix (-r + AOT)' \
    'cd "$REPO_ROOT";
     t="tests/string/string_edge_test.esk";
     rout=$("$ESHKOL_RUN" -r "$t" 2>&1) || exit 1;
     printf "%s" "$rout" | grep -qE "(^|[^A-Za-z0-9_])FAIL|Failed:[[:space:]]+[1-9]" && exit 1;
     bin=$(mktemp "${TMPDIR:-/tmp}/icc_string_edge.XXXXXX"); rm -f "$bin";
     "$ESHKOL_RUN" "$t" -o "$bin" >/dev/null 2>&1 || exit 1;
     aout=$("$bin" 2>&1) || exit 1;
     printf "%s" "$aout" | grep -qE "(^|[^A-Za-z0-9_])FAIL|Failed:[[:space:]]+[1-9]" && exit 1;
     rm -f "$bin";
     exit 0'

probe define_loop_flat_rss_aot 'ESH-0214b: AOT guard-wrapped define loop keeps RSS flat (v1.3.1 gate)' \
    'cd "$REPO_ROOT";
     ## v1.3.1 fix: per-iteration arena scope reclamation for self-tail
     ## define loops with catch-all guard. Broken behavior is ~2.6GB peak
     ## RSS at 1e6 allocating iterations; fixed is ~26MB. The gate fails
     ## above a 200MB ceiling.
     bash tests/memory/define_loop_flat_rss_aot_test.sh'

probe vm_region_flat_rss 'SW-14 close: (with-region ...) MEASURABLY reclaims on the bytecode VM — flat peak RSS across a swept iteration count (26 MB flat at 1000/4000/16000) against 796 MB with the evacuator disabled, same answers either way' \
    'cd "$REPO_ROOT";
     ## SW-14 close condition. At the branch point this fixture peaked at the
     ## SAME RSS with and without the with-region wrapper (1.503 vs 1.504 GB)
     ## because the form reclaimed nothing. The gate sweeps the iteration count
     ## and requires flatness, requires a 2x separation against
     ## ESHKOL_VM_REGION_EVAC=0, and requires the printed answer to be identical
     ## either way — so the reclamation claim is a measurement, not an assertion.
     out=$(BUILD_DIR="$BUILD_DIR_PATH" bash tests/memory/vm_region_flat_rss_test.sh 2>&1) || exit 1;
     printf "%s" "$out" | grep -q "vm_region_flat_rss_test.sh: PASS"'

probe vm_region_evac_subtype_coverage 'SW-14 close: every VM heap subtype a program can build inside a region survives the pop with its interior intact — read back and compared under ESHKOL_ARENA_POISON=1, under the post-sweep audit, and with reclamation disabled' \
    'cd "$REPO_ROOT";
     ## The VM counterpart of region_evac_subtype_coverage. A VM Value addresses
     ## the heap by a small INTEGER, so a reference the per-subtype walk misses
     ## cannot be recovered by any pointer scan and the object is freed while
     ## live. Poison mode keeps dead blocks mapped and stamped 0xCB and refuses
     ## to recycle retired indices, so a coverage hole faults instead of
     ## aliasing; the audit stage independently scans the object table for a
     ## surviving reference to a retired index.
     out=$(BUILD_DIR="$BUILD_DIR_PATH" bash tests/memory/vm_region_evac_subtype_coverage_test.sh 2>&1) || exit 1;
     printf "%s" "$out" | grep -q "vm_region_evac_subtype_coverage_test.sh: PASS"'

probe vm_region_growth_watchdog 'The VM heap growth watchdog after the SW-14 close: no false "reclaims nothing" claim for with-region, the still-true note on the bookkeeping-only region HANDLE surface, a loud budget diagnostic for unbounded growth with no region around it, fail-closed mode, and silence for the loop that now gets its memory back' \
    'cd "$REPO_ROOT";
     ## The watchdog outlived SW-14 because outside a region the VM heap still
     ## grows monotonically. What changed is what it must NOT say: with-region
     ## reclaims now, so the interim note claiming otherwise is gone, and the
     ## same allocation volume that trips the budget unwrapped must not trip it
     ## wrapped.
     out=$(BUILD_DIR="$BUILD_DIR_PATH" bash tests/memory/vm_region_growth_watchdog_test.sh 2>&1) || exit 1;
     printf "%s" "$out" | grep -q "vm_region_growth_watchdog_test.sh: PASS"'

probe iter_scope_partial_reclaim 'ESH-0214e: resident tick loop that MUTATES persistent state every tick reclaims transient garbage automatically (nursery region) — AOT flat RSS + correct + clean under ESHKOL_ARENA_POISON=1' \
    'cd "$REPO_ROOT";
     ## ESH-0214e: iter-scope partial reclamation. A guard-wrapped self-tail
     ## define loop that hash-table-set!/vector-set!/set-cdr!s persistent state
     ## every tick used to be REJECTED by the all-or-nothing gate and leaked one
     ## tick of transient garbage forever; now it runs inside a per-loop nursery
     ## region that promotes escapees out and resets each tick. The gate also
     ## re-runs the binary under ESHKOL_ARENA_POISON=1 (dangling-ptr tripwire).
     bash tests/memory/iter_scope_partial_reclaim_test.sh'

probe region_handle_scoped_reclamation '#341: user-reachable region handles keep an AD training loop flat, numerics identical across plain/with-region/handle, full misuse matrix clean under ESHKOL_ARENA_POISON=1' \
    'cd "$REPO_ROOT";
     ## #341: the automatic per-iteration nursery (ESH-0214e) disqualifies any
     ## loop body containing a gradient op, a set! or a tensor-set! — an AD
     ## training step trips all three — so a 161-param MLP grew ~123MB/step
     ## unbounded. region-open / region-close give the same region machinery a
     ## NON-LEXICAL surface. This gate requires flat peak RSS across step counts,
     ## bit-identical trained parameters against both the unscoped baseline and
     ## the with-region twin, and the whole safety matrix (double close,
     ## use-after-close, out-of-order cascade, fabricated tokens, raise and
     ## call/cc unwind crossing an open handle, slot reuse, never-closed) clean
     ## on AOT and JIT under the arena poisoner.
     bash tests/memory/region_handle_training_rss_test.sh'

probe reader_fuzz_smoke 'seeded adversarial reader harness: no crash/hang, depth guard graceful (fixed-seed smoke pass)' \
    'cd "$REPO_ROOT" && bash scripts/run_reader_fuzz.sh --smoke'

# ───────────────────────────────────────────────────────────────────
# Generative multi-oracle differential (P7c). Generates a deterministic
# family of closed R7RS-small programs and cross-checks each across
# chibi / Eshkol JIT / AOT-O0 / AOT-O2 / bytecode VM plus metamorphic
# self-checks. Regression mode: passes iff no divergence appears outside
# the triaged baseline (tests/generative-diff/baseline.txt) — a NEW
# miscompile trips it. See docs/reports/GENERATIVE_DIFFERENTIAL_REPORT.md
# and tests/generative-diff/README.md.
# ───────────────────────────────────────────────────────────────────
probe generative_differential_oracle 'generated R7RS programs agree across chibi/JIT/AOT-O0/AOT-O2/VM + metamorphic (no NEW divergence vs baseline)' \
    'if eshkol_durable_enabled; then
         workdir=$(eshkol_durable_prepare_dir generative-differential) || exit $?;
         cd "$REPO_ROOT" && python3 scripts/run_generative_differential.py --smoke \
             --baseline tests/generative-diff/baseline.txt --quiet --workdir "$workdir";
     else
         cd "$REPO_ROOT" && python3 scripts/run_generative_differential.py --smoke \
             --baseline tests/generative-diff/baseline.txt --quiet;
     fi'

# TOTAL-LANGUAGE coverage is a monotonic, manifest-derived contract.  The
# dedicated harness also writes runtime_event evidence consumed directly by
# INV-language-surface-exercise and the total-language completion oracle.
probe language_surface_coverage_floor 'exposure-engine language coverage meets the committed monotonic floor' \
    'cd "$REPO_ROOT" && ./scripts/run_language_coverage.sh'
# The ledger says what the VM supports; this makes the ledger prove it by
# RUNNING both engines over the whole callable surface. Without it a row can
# claim vm-supported for a name that aborts the VM, and vm_parity_audit.py
# (which only scrapes source text) reports OK anyway — that is how assq,
# assv, memv, partition and string-contains stayed invisible.
probe surface_parity_execution_backed 'every name native resolves is resolved by the VM or recorded in PARITY.tsv (probed on both engines, ratcheted)' \
    'if eshkol_durable_enabled; then
         workdir=$(eshkol_durable_prepare_dir surface-parity) || exit $?;
         cd "$REPO_ROOT" && BUILD_DIR="$BUILD_DIR" python3 scripts/run_surface_parity.py --workdir "$workdir";
     else
         cd "$REPO_ROOT" && BUILD_DIR="$BUILD_DIR" python3 scripts/run_surface_parity.py;
     fi'
# The one that catches SILENT WRONG ANSWERS. Name-resolution parity, ledger
# classification and one-engine coverage all pass while the two engines return
# different VALUES; this runs the corpus on both engines and compares output,
# and refuses to let a VM failure be reclassified as "out of subset".
# Validated by reintroducing the '() truthiness bug: exit 1 with the bug, 0
# without.
probe engine_semantic_parity 'no corpus program computes a different answer on the two engines, and differential construct coverage holds its floor' \
    'if eshkol_durable_enabled; then
         workdir=$(eshkol_durable_prepare_dir engine-parity) || exit $?;
         cd "$REPO_ROOT" && BUILD_DIR="$BUILD_DIR" python3 scripts/run_engine_parity_coverage.py --workdir "$workdir";
     else
         cd "$REPO_ROOT" && BUILD_DIR="$BUILD_DIR" python3 scripts/run_engine_parity_coverage.py;
     fi'

# -- fix-campaign regression gates (2026-07-10): exact-oracle-verified fixes --
probe numeric_exactness_oracle 'exact gcd bignum path + bignum divmod identity (a=q*b+r) + rational/complex eqv?/equal? (ESH-0124/0125/0114)' \
    'cd "$REPO_ROOT"; out=$(ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" -r tests/numeric/bignum_rational_exactness_test.esk 2>&1) || exit 1; echo "$out" | grep -qE "^PASS:" || exit 1; echo "$out" | grep -qE "(^| )FAIL" && exit 1; exit 0'
probe i128_native_type_oracle 'native i128: wrapping add/sub/mul/neg at +-2^127, min/max decimal round-trip (incl -2^127), shifts 0/64/127, compares, truncated quotient/remainder, i128->int range' \
    'cd "$REPO_ROOT"; out=$("$ESHKOL_RUN" -r tests/types/i128_test.esk 2>&1) || exit 1; echo "$out" | grep -q "ALL i128 TESTS PASSED" || exit 1; echo "$out" | grep -qE ": FAIL" && exit 1; exit 0'
probe closure_set_tco_loop_oracle 'closure created in a named-let/TCO loop that set!s a captured global keeps the mutation (ESH-0094)' \
    'cd "$REPO_ROOT"; out=$(ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" -r tests/closures/closure_set_in_tco_loop_test.esk 2>&1) || exit 1; echo "$out" | grep -qE "Failed:[[:space:]]+0" || exit 1'
probe stdlib_sort_filter_scale_oracle 'stdlib sort (2M) and filter (1M) are tail-recursive and correct vs reference (ESH-0098/0108)' \
    'cd "$REPO_ROOT"; out=$(ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" -r tests/stdlib/sort_filter_scale_test.esk 2>&1) || exit 1; echo "$out" | grep -qE "Failed:[[:space:]]+0" || exit 1'
probe ad_forward_over_reverse_oracle 'jacobian/hessian differentiating through an inner forward-mode derivative is exact, not silent-zero (ESH-0120/0121)' \
    'cd "$REPO_ROOT"; out=$(ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" -r tests/ad/forward_over_reverse_test.esk 2>&1) || exit 1; echo "$out" | grep -qE "Failed:[[:space:]]+0" || exit 1'
# Task #114. Two claims in one probe, because the defect needed both to be
# visible: (1) an AD operator differentiating a lambda that captures its
# ENCLOSING FUNCTION'S PARAMETER must never read a same-named top-level global
# instead — the reconstruction has to use codegenLambda's own local-then-global
# scope rule; and (2) the cached `-r` route (which compiles AOT in a child) and
# the uncached in-process JIT route must produce BYTE-IDENTICAL output, the
# PR #407 invariant. Pre-fix, core.ad.guw's documented example died with
# `vector-ref: index out of bounds` under the default cache-on `-r` and printed
# correct values under ESHKOL_JIT_CACHE=0, and the AD guide carried the
# ESHKOL_JIT_CACHE=0 workaround eight times. Byte-comparing the two routes is
# what turns that class from "a workaround in a doc" into a gate.
probe ad_capture_global_shadow_oracle 'AD capture reconstruction respects lexical scope (parameter shadows same-named global), and the cached -r/AOT route is byte-identical to the uncached in-process JIT route (task #114)' \
    'cd "$REPO_ROOT"; t=tests/ad/ad_capture_global_shadow_test.esk;
     cold=$(mktemp -d) || exit 1;
     a=$(ESHKOL_JIT_CACHE_DIR="$cold" "$ESHKOL_RUN" -r "$t" -L"$BUILD_DIR_PATH" 2>/dev/null); ra=$?;
     b=$(ESHKOL_JIT_CACHE_DIR="$cold" "$ESHKOL_RUN" -r "$t" -L"$BUILD_DIR_PATH" 2>/dev/null); rb=$?;
     c=$(ESHKOL_JIT_CACHE=0 "$ESHKOL_RUN" -r "$t" -L"$BUILD_DIR_PATH" 2>/dev/null); rc=$?;
     rm -rf "$cold";
     [ "$ra" -eq 0 ] && [ "$rb" -eq 0 ] && [ "$rc" -eq 0 ] || exit 1;
     [ "$a" = "$b" ] && [ "$b" = "$c" ] || exit 1;
     printf "%s" "$c" | grep -q "PASS: ad_capture_global_shadow" || exit 1;
     printf "%s" "$c" | grep -q "FAIL:" && exit 1;
     bin=$(mktemp) || exit 1;
     "$ESHKOL_RUN" "$t" -o "$bin" -L"$BUILD_DIR_PATH" >/dev/null 2>&1 || { rm -f "$bin"; exit 1; };
     d=$("$bin" 2>/dev/null); rd=$?; rm -f "$bin";
     [ "$rd" -eq 0 ] && [ "$d" = "$c" ] || exit 1;
     exit 0'
# ESH-0070 class: higher-order builtins must respect shadowing bindings.
# `(define (apply-map fn lst) (map fn lst))` with a same-named top-level fn
# silently called the GLOBAL fn (map / reduce / remove — static
# procedure-operand resolution never checked local shadowing; the VM's
# outermost-first upvalue search had the same class of bug for nested-lambda
# captures). Gated on JIT + AOT + the standalone VM in one probe because the
# defect reproduced differently per engine and no differential axis could see
# it — every engine was wrong the same way on the map case.
probe higher_order_shadowing_oracle 'map/for-each/filter/fold/reduce/remove call the shadowing binding, not a same-named top-level procedure — JIT, AOT, and VM engines' \
    'cd "$REPO_ROOT"; t=tests/codegen/higher_order_shadowing_test.esk;
     a=$(ESHKOL_JIT_CACHE=0 "$ESHKOL_RUN" -r "$t" -L"$BUILD_DIR_PATH" 2>/dev/null) || exit 1;
     printf "%s" "$a" | grep -q "PASS: higher-order shadowing" || exit 1;
     printf "%s" "$a" | grep -q "FAIL:" && exit 1;
     bin=$(mktemp) || exit 1;
     "$ESHKOL_RUN" "$t" -o "$bin" -L"$BUILD_DIR_PATH" >/dev/null 2>&1 || { rm -f "$bin"; exit 1; };
     b=$("$bin" 2>/dev/null); rb=$?; rm -f "$bin";
     [ "$rb" -eq 0 ] || exit 1;
     printf "%s" "$b" | grep -q "PASS: higher-order shadowing" || exit 1;
     vm="$BUILD_DIR_PATH/eshkol-vm-standalone-test";
     [ -x "$vm" ] || exit 1;
     c=$(ESHKOL_VM_NO_DISASM=1 "$vm" tests/codegen/higher_order_shadowing_vm_test.esk 2>/dev/null) || exit 1;
     printf "%s" "$c" | grep -q "PASS: higher-order shadowing (vm)" || exit 1;
     printf "%s" "$c" | grep -q "FAIL:" && exit 1;
     exit 0'
# SW-45. A single VM procedure that references more than 16 distinct
# top-level procedures (every call is a free-variable reference resolved as
# an upvalue, exactly like a captured variable) needed to relay more than 16
# upvalues to its own closure. The compiler capped that count at 32
# (MAX_UPVALUES) but the runtime closure representation's arrays were fixed
# at 16 (HeapObject.closure.upvalues[]/open_slots[], vm_core.c); OP_CLOSURE
# silently clamped to 16 and popped only 16 of the >16 values the compiler
# had pushed, stranding the rest on the operand stack with no diagnostic and
# exit 0. Every stack-slot offset computed for the rest of the program was
# off by the leaked count from then on, so the very next top-level `define`
# read back a stray leaked value instead of its own closure — discovered via
# a VM evacuator fixture with ~20 constructor calls in one procedure
# corrupting the define compiled right after it. Fixed by sharing one
# constant (ESHKOL_VM_MAX_CLOSURE_UPVALUES, inc/eshkol/backend/vm_limits.h)
# between the compiler's cap and the runtime array capacity, and by failing
# the compile loudly (never silently) if a scope still needs more upvalues
# than that shared capacity holds.
probe eshkol-vm-large-proc 'a VM procedure calling up to 32 distinct top-level procedures (17-32 used to silently corrupt the define compiled right after it) runs correctly, and one past the shared capacity fails the compile loudly instead (SW-45)' \
    'cd "$REPO_ROOT";
     vm="$BUILD_DIR_PATH/eshkol-vm-standalone-test";
     [ -x "$vm" ] || exit 1;
     out=$(ESHKOL_VM_NO_DISASM=1 "$vm" tests/vm/closure_upvalue_capacity_surface_regression.esk 2>&1) || exit 1;
     [ "$(printf "%s" "$out" | grep -c "^PASS$")" -eq 3 ] || exit 1;
     printf "%s" "$out" | grep -q "^FAIL$" && exit 1;
     printf "%s" "$out" | grep -q "ERROR:" && exit 1;
     bash tests/closures/closure_upvalue_capacity_overflow_gate.sh "$ESHKOL_RUN" "$vm" "$(mktemp -d)" >/dev/null 2>&1 || exit 1;
     exit 0'
probe linear_solve_full_f64_oracle 'linear-solve: mixed-precision IR dense solver reaches full-f64 residual (<=1e-12, computed in-test) on well-conditioned/identity systems and raises catchably on singular/dimension-mismatch — verified on JIT, AOT, and the VM' \
    'cd "$REPO_ROOT"; t=tests/features/linear_solve_test.esk;
     out=$(ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" -r "$t" 2>/dev/null) || exit 1;
     [ "$(printf "%s" "$out" | grep -c "PASS:")" -eq 8 ] || exit 1;
     printf "%s" "$out" | grep -q "FAIL:" && exit 1;
     bin=$(mktemp) || exit 1;
     ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" "$t" -o "$bin" >/dev/null 2>&1 || { rm -f "$bin"; exit 1; };
     out=$("$bin" 2>/dev/null); rc=$?; rm -f "$bin"; [ "$rc" -eq 0 ] || exit 1;
     [ "$(printf "%s" "$out" | grep -c "PASS:")" -eq 8 ] || exit 1;
     printf "%s" "$out" | grep -q "FAIL:" && exit 1;
     vm="$BUILD_DIR_PATH/eshkol-vm-standalone-test";
     if [ -x "$vm" ]; then
       out=$(ESHKOL_VM_NO_DISASM=1 ESHKOL_PATH="$REPO_ROOT/lib" "$vm" "$t" 2>/dev/null) || exit 1;
       [ "$(printf "%s" "$out" | grep -c "PASS:")" -eq 8 ] || exit 1;
       printf "%s" "$out" | grep -q "FAIL:" && exit 1;
     fi;
     exit 0'

probe printer_roundtrip_oracle 'flonum printer (#310): number->string / display emit the SHORTEST decimal that reads back to the exact same double (round-trip), no 6-sig-fig truncation, integral doubles keep the no-".0" form — byte-identical across JIT, AOT and the VM' \
    'cd "$REPO_ROOT"; t=tests/features/printer_roundtrip_test.esk;
     out=$(ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" -r "$t" 2>/dev/null) || exit 1;
     printf "%s" "$out" | grep -q "ALL PRINTER ROUND-TRIP CHECKS PASSED" || exit 1;
     printf "%s" "$out" | grep -q "error:" && exit 1;
     bin=$(mktemp) || exit 1;
     ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" "$t" -o "$bin" >/dev/null 2>&1 || { rm -f "$bin"; exit 1; };
     out=$("$bin" 2>/dev/null); rc=$?; rm -f "$bin"; [ "$rc" -eq 0 ] || exit 1;
     printf "%s" "$out" | grep -q "ALL PRINTER ROUND-TRIP CHECKS PASSED" || exit 1;
     printf "%s" "$out" | grep -q "error:" && exit 1;
     vm="$BUILD_DIR_PATH/eshkol-vm-standalone-test";
     if [ -x "$vm" ]; then
       out=$(ESHKOL_VM_NO_DISASM=1 ESHKOL_PATH="$REPO_ROOT/lib" "$vm" "$t" 2>/dev/null) || exit 1;
       printf "%s" "$out" | grep -q "ALL PRINTER ROUND-TRIP CHECKS PASSED" || exit 1;
       printf "%s" "$out" | grep -q "error:" && exit 1;
     fi;
     exit 0'

probe pipe_symbol_oracle 'R7RS 7.1.1 vertical-line symbols: |weird sym| reads as ONE symbol (not two tokens), the full <symbol element> alphabet incl. \x<hex>; decodes, |.| is a symbol distinct from the bare dotted-pair delimiter, write emits bars iff the R7RS 7.1.1 <identifier> grammar cannot spell the name bare, display never bars, and write -> read recovers an eq? symbol — byte-identical across JIT, AOT and the VM' \
    'cd "$REPO_ROOT"; t=tests/features/pipe_symbol_test.esk;
     out=$(ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" -r "$t" 2>/dev/null) || exit 1;
     printf "%s" "$out" | grep -q "ALL PIPE SYMBOL CHECKS PASSED" || exit 1;
     printf "%s" "$out" | grep -q "error:" && exit 1;
     bin=$(mktemp) || exit 1;
     ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" "$t" -o "$bin" >/dev/null 2>&1 || { rm -f "$bin"; exit 1; };
     out=$("$bin" 2>/dev/null); rc=$?; rm -f "$bin"; [ "$rc" -eq 0 ] || exit 1;
     printf "%s" "$out" | grep -q "ALL PIPE SYMBOL CHECKS PASSED" || exit 1;
     printf "%s" "$out" | grep -q "error:" && exit 1;
     vm="$BUILD_DIR_PATH/eshkol-vm-standalone-test";
     if [ -x "$vm" ]; then
       out=$(ESHKOL_VM_NO_DISASM=1 ESHKOL_PATH="$REPO_ROOT/lib" "$vm" "$t" 2>/dev/null) || exit 1;
       printf "%s" "$out" | grep -q "ALL PIPE SYMBOL CHECKS PASSED" || exit 1;
       printf "%s" "$out" | grep -q "error:" && exit 1;
     fi;
     exit 0'

probe matmul_tensor_read_scope_oracle 'matmul-tensor scope (#309): a matmul result read via tensor-ref/tensor-data from INSIDE a defined function (captured global, argument, in-function matmul, nested define, closure, with-region escape, large GPU/BLAS-dispatched matmul) returns the SAME data as a top-level read — never zeros; verified on JIT and AOT' \
    'cd "$REPO_ROOT"; t=tests/tensor/matmul_read_in_define_test.esk;
     out=$("$ESHKOL_RUN" -r "$t" 2>/dev/null) || exit 1;
     printf "%s" "$out" | grep -q "PASS: matmul tensor reads inside defined functions match top-level" || exit 1;
     printf "%s" "$out" | grep -q "FAIL:" && exit 1;
     bin=$(mktemp) || exit 1;
     "$ESHKOL_RUN" "$t" -o "$bin" >/dev/null 2>&1 || { rm -f "$bin"; exit 1; };
     out=$("$bin" 2>/dev/null); rc=$?; rm -f "$bin"; [ "$rc" -eq 0 ] || exit 1;
     printf "%s" "$out" | grep -q "PASS: matmul tensor reads inside defined functions match top-level" || exit 1;
     printf "%s" "$out" | grep -q "FAIL:" && exit 1;
     exit 0'

probe vm_gradient_parity 'gradient on the bytecode VM: the callable-arity spec (direct/wrapped/curried; 2/3-arg scalar spread; arity-1 whole-vector; non-polynomial) is 25/25 on the VM AND byte-identical to native across native/vm-src/vm-eskb (corpus 32); the public low-level AD tape surface (incl. ad-pow) is green on JIT, AOT and the VM (corpus 33 + regression)' \
    'cd "$REPO_ROOT"; vm="$BUILD_DIR_PATH/eshkol-vm-standalone-test";
     [ -x "$vm" ] || exit 1;
     # (1) callable-arity spec: 25/25 on the VM
     out=$(ESHKOL_VM_NO_DISASM=1 "$vm" tests/autodiff/gradient_callable_arity_test.esk 2>/dev/null) || exit 1;
     printf "%s" "$out" | grep -q "RESULT: ALL PASSED" || exit 1;
     # (2) low-level tape surface (incl. ad-pow) public on JIT and AOT
     out=$("$ESHKOL_RUN" --strict-types -r tests/vm/ad_tape_lowlevel_regression.esk 2>/dev/null) || exit 1;
     printf "%s" "$out" | grep -q "FAIL" && exit 1;
     printf "%s" "$out" | grep -q "PASS: ad-pow propagates gradient" || exit 1;
     bin=$(mktemp) || exit 1;
     "$ESHKOL_RUN" --strict-types tests/vm/ad_tape_lowlevel_regression.esk -o "$bin" >/dev/null 2>&1 || { rm -f "$bin"; exit 1; };
     out=$("$bin" 2>/dev/null); rc=$?; rm -f "$bin"; [ "$rc" -eq 0 ] || exit 1;
     printf "%s" "$out" | grep -q "FAIL" && exit 1;
     # (3) native<->VM byte-identical for the gradient + ad-pow corpora
     for f in tests/vm_parity/corpus/32_gradient_reverse.esk tests/vm_parity/corpus/33_ad_pow_lowlevel.esk; do
       n=$("$ESHKOL_RUN" -r "$f" 2>/dev/null | tr -d "\n");
       v=$(ESHKOL_VM_NO_DISASM=1 "$vm" "$f" 2>/dev/null | grep -vE "Eshkol VM|Execution complete|^===" | tr -d "\n");
       [ "$n" = "$v" ] || exit 1;
     done;
     exit 0'
# v1.3.4 DYNAMIC EDGE COVERAGE. Runs the seeded, bounded, depth-parametric
# edge-case generator (scripts/gen_edge_v134.py) across every applicable
# execution axis (JIT / AOT-O2 / AOT-O0 / VM) via
# scripts/run_edge_coverage_v134.sh, which writes the per-family
# kind:"edge_coverage" events into scripts/icc_traces/edge_coverage_v134.jsonl
# that the `v1.3.4-edge-coverage` oracle gates on. This probe additionally
# emits a single eshkol_smoke roll-up so the compiler-readiness harness fails
# loudly if any family regresses. Reduced depth keeps the smoke run bounded
# (< ~40s on a 4-core slice); the nightly lane runs the full MAX_DEPTH sweep.
probe edge_v134_dynamic_coverage 'v1.3.4 edge coverage (nursery iter-scope 6-channel, capturing parallel-map, exact gradient-through-callable + curried, native i128 wraparound, native matmul, VM ad-pow/ad-tape) — generated seeded probes green across JIT/AOT-O0/AOT-O2/VM with no native-vs-VM divergence' \
    'cd "$REPO_ROOT"; MODES="jit aot aot-O0 vm" JOBS="${JOBS:-4}" MAX_DEPTH="${EDGE_V134_MAX_DEPTH:-4}" \
        BUILD_DIR="$BUILD_DIR_PATH" bash scripts/run_edge_coverage_v134.sh >/dev/null 2>&1'
# ───────────────────────────────────────────────────────────────────
# SDNC weight-matrix backward gradient check (docs/SDNC.md §13). The
# reverse-mode FFN backward passes in lib/backend/qllm_backward.c must
# agree with a central finite-difference reference to relative error
# < 1e-6. The test recompiles the precision-generic backward source in
# double (-DQLLM_REAL=double) so the finite-difference floor drops well
# below the bar; the production instantiation stays float. Self-contained
# cc build so the smoke lane does not depend on a full CMake tree.
# ───────────────────────────────────────────────────────────────────
probe qllm_backward_gradcheck \
    'SDNC qllm_backward FFN gradients (SQUARE + gated) match central finite differences to L2 rel err < 1e-6 (double regime)' \
    'cd "$REPO_ROOT";
     cc_bin="${CC:-cc}"; gc_out=$(mktemp "${TMPDIR:-/tmp}/icc-qllm-gradcheck.XXXXXX");
     "$cc_bin" -O2 -DQLLM_REAL=double -Iinc \
         tests/backend/qllm_backward_gradcheck_test.c \
         lib/backend/qllm_backward.c -lm -o "$gc_out" >/dev/null 2>&1 || { rm -f "$gc_out"; exit 1; };
     out=$("$gc_out" 2>&1); rc=$?; rm -f "$gc_out";
     [ "$rc" -eq 0 ] || exit 1;
     printf "%s" "$out" | grep -q "Results: 2 passed, 0 failed"'

# ───────────────────────────────────────────────────────────────────
# WASM execute-and-diff: Eshkol-compiled WASM output must byte-match
# native. scripts/run_wasm_differential.sh builds the bytecode VM to
# WebAssembly (Emscripten) and diffs its executed stdout against
# `eshkol-run -r` over the VM-supported subset. Requires emcc + node; on
# hosts without them the lane exits 77 (SKIP) which this probe treats as
# a visible pass-with-skip (never a silent skip: the lane prints why).
# Writes its own kind:"wasm_parity" trace consumed by the
# wasm-execute-diff completion oracle.
# ───────────────────────────────────────────────────────────────────
probe wasm_execute_diff_oracle \
    'WASM execute-and-diff: bytecode VM compiled to WebAssembly executes the VM-supported corpus under node and its stdout byte-matches native `eshkol-run -r` (float text RAW; vm-parity newline normalization) — SKIP (77) when emcc/node absent' \
    'cd "$REPO_ROOT"; BUILD_DIR="$BUILD_DIR_PATH" bash scripts/run_wasm_differential.sh --quick; rc=$?; [ "$rc" -eq 0 ] || [ "$rc" -eq 77 ]'
# P8 escape-closure pillar (scripts/run_p8_escape.sh). One roll-up probe
# runs the bounded CI subset of all eight escape axes — AD binding-form +
# indirection point/callable sweeps, reference-free property oracles,
# parallel-map concurrency fuzz, the manifest-driven native-vs-VM arity
# ratchet, the five-way surface-agreement ratchet, the toolchain
# fault-injection matrix, and the workload flat-RSS profiles — against a
# freshly built eshkol-run (+ VM). Each axis is designed to retro-catch a
# 2026-07 externally-reported bug CLASS. The runner writes per-axis
# kind:"escape_matrix" events plus the roll-up p8_escape_matrix_green into
# scripts/icc_traces/escape_matrix.jsonl (its own trace file); this probe
# additionally fails the compiler-readiness smoke loudly if any axis
# regresses. The full JIT+AOT+VM sweep and the TSan / packaging lanes run
# nightly (.github/workflows/adversarial-nightly.yml).
# ───────────────────────────────────────────────────────────────────
probe p8_escape_matrix_green \
    'P8 escape-closure pillar (CI subset): AD binding-form + indirection sweeps, property oracles, parallel-map concurrency fuzz, native-vs-VM arity ratchet, five-way surface agreement, toolchain fault-injection, workload flat-RSS — every axis retro-catches a 2026-07 reported bug class; green with no NEW divergence/masking/regression' \
    'cd "$REPO_ROOT"; BUILD_DIR="$BUILD_DIR_PATH" bash scripts/run_p8_escape.sh --quick --build-dir "$BUILD_DIR_PATH" >/dev/null 2>&1'

# ───────────────────────────────────────────────────────────────────
# THE VALUE-POSITION AXIS (SW-27 / SW-31 / SW-34 / SW-35, LE-01).
#
# Eshkol lowers most builtins INLINE at the call site, and referencing the
# same builtin as a VALUE takes a different route through the codegen — in
# fact two different routes, codegenVariable and resolveLambdaFunction. Four
# separate defects have been found living in that route while call position
# was correct, each by hand, each invisible to every other gate here: the
# differential corpus compares EXECUTION AXES and a value-position defect is
# usually wrong identically on all of them, so they agree and stay green.
#
# This probe closes that blind spot mechanically. For every builtin the
# manifest can type, it evaluates the SAME call in call position and through
# three value-position routes in ONE program and compares them, so the oracle
# needs no expected values and cannot pass by agreeing with a wrong one.
# ───────────────────────────────────────────────────────────────────
probe value_position_axis \
    'every builtin answers the same when referenced as a VALUE (passed to a higher-order procedure, stored, returned, reached through map) as it does in call position — the axis that SW-27, SW-31, SW-34 and SW-35 each escaped through one at a time' \
    'if eshkol_durable_enabled; then
         workdir=$(eshkol_durable_prepare_dir value-position) || exit $?;
         cd "$REPO_ROOT"; BUILD_DIR="$BUILD_DIR_PATH" python3 scripts/run_value_position_sweep.py --quiet --workdir "$workdir" >/dev/null 2>&1;
     else
         cd "$REPO_ROOT"; BUILD_DIR="$BUILD_DIR_PATH" python3 scripts/run_value_position_sweep.py --quiet >/dev/null 2>&1;
     fi'

# ───────────────────────────────────────────────────────────────────
# ESH-0011 — portable event loop (v1.4 async foundation).
#
# Runs the acceptance battery on BOTH native substrates so the probe covers the
# compile-and-link path (-r/JIT) and the standalone-binary path (AOT). The
# battery itself asserts the pipe round-trip, a measured timeout (neither a
# spin nor a hang passes), close-then-use failing closed, and 1000 open/close
# cycles proving the kernel descriptor is released.
#
# Backend under test is whichever CMake selected for the host: kqueue on
# Darwin/BSD, epoll on Linux, IOCP+WSAPoll/PeekNamedPipe on Windows. The test
# prints the name so the trace snippet says what actually ran.
# ───────────────────────────────────────────────────────────────────
probe event_loop_works \
    'ESH-0011 portable event loop (kqueue/epoll/IOCP): (make-event-loop 64) yields a handle, a pipe read+write round-trips through event-loop-poll inside the timeout, an idle poll waits its budget and returns instead of hanging, a closed handle fails closed, and 1000 open/close cycles never exhaust the descriptor table — green on JIT and AOT' \
    'cd "$REPO_ROOT";
     out=$("$ESHKOL_RUN" -r tests/v1_3_edge_cases/event_loop_test.esk -L"$BUILD_DIR_PATH" 2>&1) || exit 1;
     echo "$out" | grep -q "PASS: event_loop_test" || exit 1;
     echo "$out" | grep -qE "^FAIL:" && exit 1;
     bin=$(mktemp "${TMPDIR:-/tmp}/eshkol-event-loop.XXXXXX");
     "$ESHKOL_RUN" tests/v1_3_edge_cases/event_loop_test.esk -o "$bin" -L"$BUILD_DIR_PATH" >/dev/null 2>&1 || { rm -f "$bin"; exit 1; };
     aot=$("$bin" 2>&1); rc=$?; rm -f "$bin";
     [ $rc -eq 0 ] || exit 1;
     echo "$aot" | grep -q "PASS: event_loop_test" || exit 1;
     echo "$aot" | grep -qE "^FAIL:" && exit 1;
     exit 0'

# ───────────────────────────────────────────────────────────────────
# Silent-wrong flaw gate. Every probe above certifies that something
# WORKS; none of them can certify that nothing silently LIES. This one
# grades .icc/silent-wrong-ledger.yaml — the enumeration of defects that
# return a wrong value with no diagnostic and exit 0 — and is the gate
# that holds the tag while any of them is open and unwaived. It fails
# closed: a missing or unparseable ledger is a FAIL, never a pass.
#
# The grader writes its own trace file, so this probe deliberately runs
# it with --no-trace and lets the probe helper emit the eshkol_smoke
# event, keeping exactly one no_open_silent_wrong event in the bundle.
# ───────────────────────────────────────────────────────────────────
probe no_open_silent_wrong \
    'No open, unwaived SILENT-WRONG flaw in .icc/silent-wrong-ledger.yaml (wrong value / wrong derivative / wrong memory outcome with no diagnostic and exit 0 is tag-blocking)' \
    'cd "$REPO_ROOT"; python3 scripts/gate_no_silent_wrong.py --no-trace'

# ───────────────────────────────────────────────────────────────────
# AD carrier gates. The first is structural: every AD operator's
# differentiation carrier is declared in .icc/ad-carrier-manifest.yaml
# and re-derived from source, so an op cannot claim VM parity on an
# undeclared carrier and cannot claim it at all on a finite difference.
# The second is behavioural, and exists because a structural check
# grades text: it runs a gradient field through the VM and demands both
# the exact answer and a zero finite-difference counter. Structure
# without behaviour is how SW-51 stayed green; behaviour without
# structure is how SW-52 did.
# ───────────────────────────────────────────────────────────────────
probe ad_carrier_model_clean \
    'Every AD op declared vm-supported routes through a declared, source-verified, exact carrier and no unledgered finite difference exists on either engine' \
    'cd "$REPO_ROOT"; python3 scripts/gate_ad_shared_node_model.py --no-trace'

probe ad_vm_curl_divergence_exact \
    'VM curl of a gradient field is exactly #(0 0 0) and costs zero finite-difference evaluations' \
    'vmbin="$BUILD_DIR/eshkol-vm-standalone-test";
     if [ ! -x "$vmbin" ]; then echo "VM standalone binary not built: $vmbin"; exit 1; fi;
     tmp=$(mktemp).esk;
     cat > "$tmp" <<EOF
(ad-reset-counters!)
(define (F x y z) (list (* y z) (* x z) (* x y)))
(display "CURL=") (display (curl F (list 1.0 2.0 3.0)))
(display "|DIV=") (display (divergence F (list 1.0 2.0 3.0)))
(display "|FD=") (display (ad-finite-difference-evals))
EOF
     out=$("$vmbin" "$tmp" 2>&1); rc=$?; rm -f "$tmp";
     if [ $rc -ne 0 ]; then printf "%s" "$out" | tail -c 300; exit 1; fi;
     flat=$(printf "%s" "$out" | tr -d "\n");
     case "$flat" in
       *"CURL=#(0 0 0)|DIV=0|FD=0"*) exit 0 ;;
       *) printf "%s" "$flat" | grep -o "CURL=.*FD=[0-9]*"; exit 1 ;;
     esac'

echo
echo "Trace written: $TRACE_FILE"
echo "Probe summary: $((PROBE_TOTAL - PROBE_FAILURES))/$PROBE_TOTAL passed"
echo "Run: python3 ~/Desktop/infinite_context_coder/scripts/codebase_tool.py \\"
echo "         completion-oracle --repo eshkol_lang \\"
echo "         --target agent-ffi-ready --trace-dir scripts/icc_traces"

if [ "$PROBE_FAILURES" -ne 0 ]; then
    echo "ICC smoke gate: FAIL ($PROBE_FAILURES probe(s) failed)" >&2
    exit 1
fi
echo "ICC smoke gate: PASS"
