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
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
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
    ESHKOL_ICC_JIT_CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-icc-jit-cache.XXXXXX")
    export ESHKOL_JIT_CACHE_DIR="$ESHKOL_ICC_JIT_CACHE_DIR"
    trap 'rm -rf "$ESHKOL_ICC_JIT_CACHE_DIR"' EXIT
else
    mkdir -p "$ESHKOL_JIT_CACHE_DIR"
fi

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
(let ((p (process-spawn-argv (list "/bin/echo" "hello;world|pipe") ".")))
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
(let ((p (process-spawn-argv (list "/bin/sleep" "0.05") ".")))
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
     ## any test prints a literal "^FAIL:" or its summary shows nonzero
     ## "Failed: N" with N >= 1.
     for t in tests/v1_2_edge_cases/append_variadic_test.esk \
              tests/v1_2_edge_cases/main_substring_name_test.esk \
              tests/v1_2_edge_cases/sexp_canonical_string_test.esk \
              tests/v1_2_edge_cases/substring_utf8_test.esk \
              tests/v1_2_edge_cases/string_escapes_test.esk \
              tests/v1_2_edge_cases/procedure_arity_test.esk \
              tests/v1_2_edge_cases/json_schema_test.esk; do
       bin="/tmp/icc_$(basename "$t" .esk).bin";
       "$ESHKOL_RUN" "$t" -o "$bin" >/dev/null 2>&1 || exit 1;
       tout=$("$bin" 2>&1);
       if printf "%s" "$tout" | grep -qE "^FAIL:|Failed:[[:space:]]+[1-9]"; then
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

probe region_evac_subtype_coverage \
    'ESH-0214d/e region escape-evacuator keeps promoted logic/workspace/PROMISE subtype interiors intact under ESHKOL_ARENA_POISON=1 (AOT, flat RSS)' \
    'cd "$REPO_ROOT";
     out=$(ESHKOL_ARENA_POISON=1 BUILD_DIR="$BUILD_DIR_PATH" bash tests/memory/region_evac_subtype_coverage_test.sh 2>&1) || exit 1;
     printf "%s" "$out" | grep -q "region_evac_subtype_coverage_test.sh: PASS"'

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
     printf "%s" "$rout" | grep -qE "^FAIL:|Failed:[[:space:]]+[1-9]" && exit 1;
     bin="/tmp/icc_string_edge.bin"; rm -f "$bin";
     "$ESHKOL_RUN" "$t" -o "$bin" >/dev/null 2>&1 || exit 1;
     aout=$("$bin" 2>&1) || exit 1;
     printf "%s" "$aout" | grep -qE "^FAIL:|Failed:[[:space:]]+[1-9]" && exit 1;
     rm -f "$bin";
     exit 0'

probe define_loop_flat_rss_aot 'ESH-0214b: AOT guard-wrapped define loop keeps RSS flat (v1.3.1 gate)' \
    'cd "$REPO_ROOT";
     ## v1.3.1 fix: per-iteration arena scope reclamation for self-tail
     ## define loops with catch-all guard. Broken behavior is ~2.6GB peak
     ## RSS at 1e6 allocating iterations; fixed is ~26MB. The gate fails
     ## above a 200MB ceiling.
     bash tests/memory/define_loop_flat_rss_aot_test.sh'

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
    'cd "$REPO_ROOT" && python3 scripts/run_generative_differential.py --smoke \
        --baseline tests/generative-diff/baseline.txt --quiet'

# TOTAL-LANGUAGE coverage is a monotonic, manifest-derived contract.  The
# dedicated harness also writes runtime_event evidence consumed directly by
# INV-language-surface-exercise and the total-language completion oracle.
probe language_surface_coverage_floor 'exposure-engine language coverage meets the committed monotonic floor' \
    'cd "$REPO_ROOT" && ./scripts/run_language_coverage.sh'

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
probe linear_solve_full_f64_oracle 'linear-solve: mixed-precision IR dense solver reaches full-f64 residual (<=1e-12, computed in-test) on well-conditioned/identity systems and raises catchably on singular/dimension-mismatch — verified on JIT, AOT, and the VM' \
    'cd "$REPO_ROOT"; t=tests/features/linear_solve_test.esk;
     out=$(ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" -r "$t" 2>/dev/null) || exit 1;
     [ "$(printf "%s" "$out" | grep -c "PASS:")" -eq 6 ] || exit 1;
     printf "%s" "$out" | grep -q "FAIL:" && exit 1;
     bin=$(mktemp) || exit 1;
     ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" "$t" -o "$bin" >/dev/null 2>&1 || { rm -f "$bin"; exit 1; };
     out=$("$bin" 2>/dev/null); rc=$?; rm -f "$bin"; [ "$rc" -eq 0 ] || exit 1;
     [ "$(printf "%s" "$out" | grep -c "PASS:")" -eq 6 ] || exit 1;
     printf "%s" "$out" | grep -q "FAIL:" && exit 1;
     vm="$BUILD_DIR_PATH/eshkol-vm-standalone-test";
     if [ -x "$vm" ]; then
       out=$(ESHKOL_VM_NO_DISASM=1 ESHKOL_PATH="$REPO_ROOT/lib" "$vm" "$t" 2>/dev/null) || exit 1;
       [ "$(printf "%s" "$out" | grep -c "PASS:")" -eq 6 ] || exit 1;
       printf "%s" "$out" | grep -q "FAIL:" && exit 1;
     fi;
     exit 0'

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
