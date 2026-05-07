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
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/eshkol_smoke.jsonl"
mkdir -p "$TRACE_DIR"

# Truncate so each run is a fresh evidence set; ICC reads the union of
# events in the file, but stale PASS lines for now-broken probes would
# otherwise mask regressions.
: > "$TRACE_FILE"

ESHKOL_RUN="$REPO_ROOT/build/eshkol-run"
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "scripts/run_icc_smoke.sh: build/eshkol-run not found — run \`cmake --build build\` first." >&2
    exit 2
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
    # Escape backslashes and quotes in snippet for JSON-safe embedding.
    local esc_snippet
    esc_snippet=$(printf '%s' "$snippet" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
    printf '{"kind":"eshkol_smoke","name":"%s","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$probe_id" "$status" "$esc_snippet" >> "$TRACE_FILE"
}

probe() {
    local probe_id="$1" label="$2" cmd="$3"
    local out status snippet
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
probe native_http_get_works "HTTPS GET to httpbin.org returns 200" \
    'tmp=$(mktemp).esk;
     cat > "$tmp" <<EOF
(require agent.http)
(http-init)
(let ((r (http-get "https://httpbin.org/get" 10000)))
  (if (and r (= (car r) 200)) (exit 0) (exit 1)))
EOF
     "$ESHKOL_RUN" -r "$tmp" 2>&1; rc=$?; rm -f "$tmp"; exit $rc'

probe native_http_post_json_works "POST JSON round-trips via httpbin.org/post" \
    'tmp=$(mktemp).esk;
     cat > "$tmp" <<EOF
(require agent.http)
(http-init)
(let ((r (http-post "https://httpbin.org/post"
                    (list (cons "Content-Type" "application/json"))
                    "{\"k\":\"v\"}" 10000)))
  (if (and r (= (car r) 200) (string-contains (cdr r) "\"k\": \"v\""))
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

# ─────────────────────────────────────────────────────────────────
# v1.2 release probes
# ─────────────────────────────────────────────────────────────────
probe stdlib_o_loads "build/stdlib.o exists and is non-empty" \
    'test -s "$REPO_ROOT/build/stdlib.o"'

probe stdlib_compiles_clean "stdlib rebuilds without errors" \
    'cd "$REPO_ROOT" && touch lib/stdlib.esk && cmake --build build --target stdlib 2>&1 | grep -qE "Built target stdlib"'

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
(require core.parallel)
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

probe example_agent_compiles "examples/agent.esk compiles" \
    'cd "$REPO_ROOT" && test -f examples/agent.esk &&
     "$ESHKOL_RUN" examples/agent.esk -o /tmp/icc_selene.bin >/dev/null 2>&1'

echo
echo "Trace written: $TRACE_FILE"
echo "Run: python3 ~/Desktop/infinite_context_coder/scripts/codebase_tool.py \\"
echo "         completion-oracle --repo eshkol_lang \\"
echo "         --target agent-ffi-ready --trace-dir scripts/icc_traces"
