#!/usr/bin/env bash
# run_v14_connection_gate.sh — the v1.4-connection completion-oracle harness.
#
# .icc/completion-oracles.yaml's `v1.4-connection` target used to carry
# exactly one criterion for the whole release theme: a runtime_event nothing
# ever emitted (kind v1_4_connection / name connection_theme_deliverables,
# action `replace_with_v1_4_connection_oracle`). `icc readiness` graded that
# stub as a permanent, uninformative FAIL — it could never distinguish "v1.4
# hasn't started" from "v1.4 is half done" from "v1.4 shipped".
#
# This script is the replacement: ONE runtime_event probe per v1.4 deliverable
# (docs/COMPILER_ROADMAP.md's Networking/Concurrency/Wire-formats/Linear-types
# sections, ROADMAP.md's v1.4-connection list, docs/design/adr/0004's
# "resource-sound systems profile" exit gates, docs/design/adr/0000's Stage 3
# (v1.4.0) and Stage 4 (v1.4.1) gates, and docs/design/adr/0010's A8/A10-A13
# assurance gaps), each bound to real evidence:
#
#   * where a real, working primitive AND a real round-trip test already
#     exist, this script RUNS that evidence today and reports the actual
#     PASS/FAIL it produces (most of these were shipped ahead of the
#     roadmap's checkboxes and were simply never wired into the ICC oracle —
#     see the per-probe comments below for the executable evidence used);
#   * where this gate has no runnable or source-bound evidence for a
#     requirement, it emits an honest FAIL marked UNVERIFIED. Source presence
#     alone never earns a PASS, and an unverified status makes no claim that
#     the feature is absent;
#   * where external CI evidence is needed, the event requests a
#     source-bound CI receipt rather than inferring runner or hardware state.
#
# This script never fabricates a PASS. A criterion is PASS only when the
# thing it claims was actually exercised, this run, and produced the
# claimed result.
#
# Usage: scripts/run_v14_connection_gate.sh
#   BUILD_DIR   (env, default "build")   the CMake build directory
#   TRACE_DIR   (env, default scripts/icc_traces)

set -u
set -o pipefail
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) BUILD_DIR_PATH="$BUILD_DIR" ;;
    *)  BUILD_DIR_PATH="$REPO_ROOT/$BUILD_DIR" ;;
esac
if [ ! -d "$BUILD_DIR_PATH" ]; then
    echo "run_v14_connection_gate.sh: build directory not found: $BUILD_DIR_PATH" >&2
    exit 2
fi
BUILD_DIR_PATH="$(cd "$BUILD_DIR_PATH" && pwd -P)" || exit 2
ESHKOL_RUN="$BUILD_DIR_PATH/eshkol-run"
VM_RUN="$BUILD_DIR_PATH/eshkol-vm-standalone-test"

TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
# Evidence paths are absolute before first use (scripts/lib/evidence_paths.sh).
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
# shellcheck source=lib/checked_write.sh
. "$REPO_ROOT/scripts/lib/checked_write.sh"
TRACE_FILE="$TRACE_DIR/v1_4_connection.jsonl"
mkdir -p "$TRACE_DIR"
eshkol_require_output_file_path "$TRACE_FILE"
: > "$TRACE_FILE"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-v14-gate.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

PROBE_TOTAL=0
PROBE_FAILURES=0

emit_event() { # name PASS|FAIL snippet
    eshkol_require_output_file_path "$TRACE_FILE"
    python3 -c '
import json, sys
print(json.dumps({"kind": "v1_4_connection", "name": sys.argv[1], "value": sys.argv[2],
                  "snippet": sys.argv[3], "confidence": 0.95}, ensure_ascii=False))
' "$1" "$2" "$3" >> "$TRACE_FILE"
}

. "$REPO_ROOT/scripts/lib/connection_gate_probe.sh"

# #145 carries a second, immutable ICC receipt.  The ordinary sweep event is
# useful for the live counter, but cannot attest which source tree or runner
# binary produced it.  ICC's test-execution receipt stamps exact-clean-current
# source identity and hashes the declared executable as measurement input.
probe_http_server_source_receipt() {
    local icc_bin repo_name binding_output binding_status command_line jit_cache fingerprint_trace fingerprint_file out status snippet receipt_ok
    PROBE_TOTAL=$((PROBE_TOTAL + 1))
    icc_bin="${ICC_BIN:-$HOME/Desktop/infinite_context_coder/bin/icc}"
    if [ ! -x "$icc_bin" ]; then
        PROBE_FAILURES=$((PROBE_FAILURES + 1))
        emit_event v14_http_server_roundtrip FAIL "ICC receipt producer unavailable: $icc_bin"
        printf '  \xE2\x9C\x97 %-42s ICC receipt producer unavailable\n' v14_http_server_roundtrip
        return
    fi

    repo_name="${ICC_REPO_NAME:-eshkol}"
    binding_output=$(ICC_BIN="$icc_bin" ICC_REPO_NAME="$repo_name" \
        bash "$REPO_ROOT/tests/v1_2_edge_cases/http_server_smoke_test.sh" \
        --verify-icc-checkout-only 2>&1)
    binding_status=$?
    if [ "$binding_status" -ne 0 ]; then
        PROBE_FAILURES=$((PROBE_FAILURES + 1))
        emit_event v14_http_server_roundtrip FAIL "$binding_output"
        printf '  \xE2\x9C\x97 %-42s ICC repo checkout mismatch\n' v14_http_server_roundtrip
        return
    fi

    jit_cache="$WORK/http-server-jit-cache"
    # Keep the fingerprint receipt under the durable trace root; the rest of
    # the temporary probe state is safely removed when the sweep exits.
    fingerprint_trace="$TRACE_DIR/v14_http_server_build"
    mkdir -p "$jit_cache" "$fingerprint_trace"
    fingerprint_file="$fingerprint_trace/build_fingerprint.jsonl"
    command_line="env BUILD_DIR='$BUILD_DIR_PATH' TRACE_DIR='$fingerprint_trace' ICC_BIN='$icc_bin' ICC_REPO_NAME='$repo_name' ESHKOL_LIB_DIR='$BUILD_DIR_PATH' ESHKOL_JIT_CACHE_DIR='$jit_cache' bash ./tests/v1_2_edge_cases/http_server_smoke_test.sh"
    # The shell test invokes helper files transitively; ICC's command-seed
    # closure does not enumerate every sourced shell dependency. Disable that
    # relaxation so only an exact whole-tree source/index match is admissible;
    # any later commit retires this receipt until the round trip is rerun.
    out=$(ICC_RECEIPT_CLOSURE=0 "$icc_bin" test-execution-oracle \
        --repo "$repo_name" \
        --name v14_http_server_roundtrip \
        --cwd "$REPO_ROOT" \
        --declare-data "$ESHKOL_RUN" \
        --declare-data "$BUILD_DIR_PATH/stdlib.o" \
        --declare-data "$BUILD_DIR_PATH/stdlib.bc" \
        --declare-data "$fingerprint_file" \
        --declare-instrument scripts/run_v14_connection_gate.sh:http_server_smoke_test.sh \
        --command "$command_line" --format json 2>&1)
    status=$?
    receipt_ok=1
    if [ "$status" -eq 0 ]; then
        printf '%s' "$out" | python3 -c '
import json, re, sys
try:
    payload = json.load(sys.stdin)
    record = payload.get("record", {})
    value = record.get("value", {})
    source = value.get("source", {})
    data = (value.get("provenance", {}).get("data", {})
            if isinstance(value.get("provenance"), dict) else {})
    inputs = data.get("value") if isinstance(data, dict) else None
    binary, stdlib_object, stdlib_bitcode, fingerprint_path = sys.argv[1:]
    binary_rows = [row for row in inputs or []
                   if isinstance(row, dict) and row.get("path") == binary]
    required_paths = {stdlib_object, stdlib_bitcode, fingerprint_path}
    declared_paths = {row.get("path") for row in inputs or [] if isinstance(row, dict)}
    ok = (
        payload.get("ok") is True
        and record.get("schema") == "icc.test_execution.v1"
        and record.get("kind") == "test_result"
        and record.get("name") == "v14_http_server_roundtrip"
        and record.get("status") == "PASS"
        and source.get("dirty") is False
        and source.get("icc_fresh") is True
        and source.get("icc_source_fingerprint")
            == source.get("icc_index_source_fingerprint")
        and len(binary_rows) == 1
        and re.fullmatch(r"[0-9a-f]{64}", str(binary_rows[0].get("sha256") or ""))
        and required_paths.issubset(declared_paths)
        and all(
            re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256") or ""))
            for row in inputs or []
            if isinstance(row, dict) and row.get("path") in required_paths
        )
    )
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
' "$ESHKOL_RUN" "$BUILD_DIR_PATH/stdlib.o" "$BUILD_DIR_PATH/stdlib.bc" "$fingerprint_file" || receipt_ok=0
    else
        receipt_ok=0
    fi
    if [ "$receipt_ok" -eq 1 ]; then
        emit_event v14_http_server_roundtrip PASS "ICC test-execution receipt recorded; binary=$ESHKOL_RUN"
        printf '  \xE2\x9C\x93 %-42s source-bound ICC receipt\n' v14_http_server_roundtrip
    else
        PROBE_FAILURES=$((PROBE_FAILURES + 1))
        snippet=$(printf '%s' "$out" | tail -c 220)
        [ "$status" -eq 0 ] && snippet="ICC receipt did not bind a fresh source and runner SHA-256"
        emit_event v14_http_server_roundtrip FAIL "$snippet"
        printf '  \xE2\x9C\x97 %-42s ICC receipt failed (exit %d)\n' v14_http_server_roundtrip "$status"
    fi
}

# A requirement for which this gate has no current runnable or source-bound
# evidence. Keep it failing until a real probe or receipt can establish it;
# this status does not assert that the implementation is absent.
probe_unverified() {
    local probe_id="$1" reason="$2"
    PROBE_TOTAL=$((PROBE_TOTAL + 1))
    PROBE_FAILURES=$((PROBE_FAILURES + 1))
    emit_event "$probe_id" FAIL "UNVERIFIED: $reason"
    printf '  \xE2\x9C\x97 %-42s UNVERIFIED: %s\n' "$probe_id" "$reason"
}

# Requirements that need CI evidence stay failing until a source-bound receipt
# for the named workflow is supplied. Do not infer CI or runner availability.
probe_external_ci_evidence() {
    local probe_id="$1" workflow="$2" reason="$3"
    PROBE_TOTAL=$((PROBE_TOTAL + 1))
    PROBE_FAILURES=$((PROBE_FAILURES + 1))
    emit_event "$probe_id" FAIL "UNVERIFIED: no source-bound CI receipt supplied for $workflow — $reason"
    printf '  \xE2\x9C\x97 %-42s UNVERIFIED (CI receipt needed): %s — %s\n' "$probe_id" "$workflow" "$reason"
}

# A native TLS event is emitted only after the compiled capability and all
# registered C/JIT/AOT/Python-peer tests are present and pass in this run.
probe_tls_acceptance() {
    local compile_flags test_list required
    compile_flags="$BUILD_DIR_PATH/CMakeCache.txt"
    if [ ! -f "$compile_flags" ] || ! grep -Fq 'ESHKOL_TLS_AVAILABLE:INTERNAL=1' "$compile_flags"; then
        probe_unverified v14_tls_handshake_verified \
            "native OpenSSL TLS is unavailable in this build; no handshake PASS is emitted"
        return
    fi

    test_list="$(ctest --test-dir "$BUILD_DIR_PATH" -N -R '^tls_' 2>&1)" || {
        probe_unverified v14_tls_handshake_verified "CTest could not enumerate TLS acceptance tests"
        return
    }
    for required in tls_context_native tls_public_scheme_jit tls_public_scheme_aot \
                    tls_public_python_jit tls_public_python_aot \
                    tls_public_fatal_jit tls_public_fatal_aot; do
        if ! printf '%s\n' "$test_list" | grep -Fq "$required"; then
            probe_unverified v14_tls_handshake_verified \
                "required public TLS test is not registered: $required"
            return
        fi
    done
    probe_run v14_tls_handshake_verified \
        'native TLS C/JIT/AOT and Python SSL peer acceptance suite' \
        'ctest --test-dir "$BUILD_DIR_PATH" --output-on-failure -R "^tls_"' \
        '100% tests passed'
}

# Stage a minimal, real package directory from THIS build (mirroring the
# copy list in .github/workflows/release.yml's "Package Release Asset"
# step, which is check_package_manifest.py's own ground truth) so A10's
# producer can grade an actual staged install instead of only its synthetic
# --self-test fixtures. Files the current build did not produce are simply
# left unstaged — check_package_manifest.py then reports a real, honest
# violation for them rather than this function fabricating a PASS.
stage_v14_package_scratch() {
    local pkg_dir="$1" f
    mkdir -p "$pkg_dir/bin" "$pkg_dir/lib/eshkol" "$pkg_dir/share/eshkol/lib" "$pkg_dir/share/eshkol/cmake"

    for f in eshkol-run eshkol-repl; do
        [ -f "$BUILD_DIR_PATH/$f" ] && cp "$BUILD_DIR_PATH/$f" "$pkg_dir/bin/"
    done
    for f in stdlib.o stdlib.bc; do
        if [ -f "$BUILD_DIR_PATH/$f" ]; then
            cp "$BUILD_DIR_PATH/$f" "$pkg_dir/lib/"
            cp "$BUILD_DIR_PATH/$f" "$pkg_dir/lib/eshkol/"
        fi
    done
    for f in libeshkol-runtime.a libeshkol-agent-ffi.a eshkol-agent-pcre2.a eshkol-agent-sqlite3.a \
             eshkol-agent-zlib.a eshkol-agent-tree-sitter-grammars.a eshkol-agent-tree-sitter.a \
             eshkol-agent-yoga.a eshkol-agent-curl.a; do
        if [ -f "$BUILD_DIR_PATH/$f" ]; then
            cp "$BUILD_DIR_PATH/$f" "$pkg_dir/lib/"
            cp "$BUILD_DIR_PATH/$f" "$pkg_dir/lib/eshkol/"
        fi
    done

    [ -f "$REPO_ROOT/lib/stdlib.esk" ] && cp "$REPO_ROOT/lib/stdlib.esk" "$pkg_dir/share/eshkol/"
    [ -f "$REPO_ROOT/lib/math.esk" ] && cp "$REPO_ROOT/lib/math.esk" "$pkg_dir/share/eshkol/"
    [ -f "$REPO_ROOT/cmake/FindEshkol.cmake" ] && cp "$REPO_ROOT/cmake/FindEshkol.cmake" "$pkg_dir/share/eshkol/cmake/"
    [ -f "$REPO_ROOT/cmake/EshkolCompile.cmake" ] && cp "$REPO_ROOT/cmake/EshkolCompile.cmake" "$pkg_dir/share/eshkol/cmake/"

    while IFS= read -r -d '' source_file; do
        local rel_path="${source_file#"$REPO_ROOT"/lib/}"
        local dest_path="$pkg_dir/share/eshkol/lib/$rel_path"
        mkdir -p "$(dirname "$dest_path")"
        cp "$source_file" "$dest_path"
    done < <(find "$REPO_ROOT/lib" -type f -name '*.esk' -print0)

    for f in README.md LICENSE CHANGELOG.md RELEASE_NOTES.md; do
        [ -f "$REPO_ROOT/$f" ] && cp "$REPO_ROOT/$f" "$pkg_dir/"
    done

    # macOS-only runtime_dependencies entries (OpenSSL/LLVM/zstd closure):
    # staged for real via the same producer release.yml uses, on a best
    # effort basis. If Homebrew or a formula is unavailable this leaves the
    # category unstaged, and the manifest check below reports that honestly
    # as a real violation rather than this function papering over it.
    if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
        local openssl_prefix llvm_prefix zstd_prefix openssl_version llvm_version zstd_version
        openssl_prefix="$(brew --prefix openssl@3 2>/dev/null)" || true
        llvm_prefix="$(brew --prefix "llvm@${LLVM_MAJOR:-21}" 2>/dev/null)" || true
        zstd_prefix="$(brew --prefix zstd 2>/dev/null)" || true
        if [ -n "$openssl_prefix" ] && [ -n "$llvm_prefix" ] && [ -n "$zstd_prefix" ]; then
            openssl_version="$("$openssl_prefix/bin/openssl" version 2>/dev/null | awk '{print $2}')"
            llvm_version="$("$llvm_prefix/bin/llvm-config" --version 2>/dev/null)"
            zstd_version="$("$zstd_prefix/bin/zstd" --version 2>/dev/null | sed -E 's/.* v([^, ]+).*/\1/')"
            python3 "$REPO_ROOT/scripts/stage_macos_openssl_runtime.py" \
                --openssl-prefix "$openssl_prefix" --llvm-prefix "$llvm_prefix" --zstd-prefix "$zstd_prefix" \
                --package-dir "$pkg_dir" \
                --openssl-version "${openssl_version:-0}" --llvm-version "${llvm_version:-0}" \
                --zstd-version "${zstd_version:-0}" >/dev/null 2>&1 || true
        fi
    fi
    python3 "$REPO_ROOT/scripts/stage_third_party_licenses.py" \
        --build-dir "$BUILD_DIR_PATH" --package-dir "$pkg_dir" >/dev/null 2>&1 || true
}

# A10 (ADR-0010 gap A10, "no packaging manifest"): run the real producer
# (scripts/check_package_manifest.py) against an actual package directory
# staged from THIS build, the same way release.yml's "Package Release
# Asset" step does before archiving. The `toolchain_flag`/`type_diagnostic`
# manifest categories ADR-0010 also names for A10 are a separate, larger
# schema/checker extension with no existing producer to wire (see the ADR's
# own "remain open" note) — out of scope for this gate-wiring pass; they
# stay unverified rather than this probe claiming them covered.
probe_package_manifest_complete() {
    local pkg_dir="$WORK/v14-package-scratch"
    rm -rf "$pkg_dir"
    stage_v14_package_scratch "$pkg_dir"
    probe_run v14_assurance_a10_packaging_manifest \
        'A10 packaging manifest (ADR-0010): a package directory staged from this build is checked against .icc/package-manifest.yaml (binaries, stdlib artifacts/sources, static libraries, docs, CMake discovery modules)' \
        "python3 \"$REPO_ROOT/scripts/check_package_manifest.py\" --package-dir \"$pkg_dir\"" \
        '^package_manifest_complete: PASS'
}

# A11 (ADR-0010 gap A11, "no diagnostic golden corpus"): run the real
# producer (scripts/check_diagnostic_corpus.py) against tests/diagnostics/,
# the same invocation ci.yml uses post-build.
probe_diagnostic_corpus_clean() {
    probe_run v14_assurance_a11_diagnostic_corpus \
        'A11 diagnostic corpus (ADR-0010): every tests/diagnostics/<case>/ fixture reproduces its expected.json diagnostic code, message, and file:line:col span' \
        "python3 \"$REPO_ROOT/scripts/check_diagnostic_corpus.py\" --build-dir \"$BUILD_DIR_PATH\"" \
        '^diagnostic_corpus_clean: PASS'
}

# Portable tail transfer (v1.4.1-STAGE, S4): run the real producer
# (scripts/run_recursion_depth.sh) and grade this criterion on exactly the
# cells it emits for the mutual_tail_{cond,forms,arity} kinds — spelling-
# independence (ESH-0102b) and arity-independence (ESH-0102c) of the tail-
# transfer dispatcher — across every lane that producer runs them in: JIT
# (-r), AOT via LLVM `musttail`, and the portable non-AArch64 `aot-xfer`
# lane (ESHKOL_TAIL_TRANSFER_ONLY=1), which is the dispatcher this criterion
# is actually named for. The producer's trace path is its own (it does not
# honor this gate's TRACE_DIR), so it is read at its documented location.
#
# $(...) only guarantees the PRODUCER's own process has exited — guarded_exec.pl
# (scripts/lib/harness_outcome.sh's eshkol_outcome_guarded, which every cell in
# the producer runs through) SIGKILLs any process the command left behind in
# its group on normal exit too, but does not wait() for that kill to land
# before returning; a descendant still dying at that instant can hold a stdio
# pipe open a moment longer. That is a pipe-closing delay, not a file-content
# hazard by itself (the producer's own process writes the trace file directly
# and sequentially, with no child involved in those writes) — but this probe
# does not assume that proof holds forever as the producer evolves. Instead
# it waits for the producer's OWN explicit completion record
# (kind=recursion_depth, name=recursion_depth_gate, emitted as literally the
# last write before the producer exits) to appear in the file it just read,
# re-reading up to ~10s if it is not there yet, and fails closed with a clear
# message if it never appears. Only a read that already contains that
# completion record is trusted for the mutual_tail_{cond,forms,arity} match.
probe_portable_tail_transfer() {
    PROBE_TOTAL=$((PROBE_TOTAL + 1))
    local depth_script="$REPO_ROOT/scripts/run_recursion_depth.sh"
    local depth_trace="$REPO_ROOT/scripts/icc_traces/recursion_depth.jsonl"
    local depth_out depth_rc verdict
    if [ ! -x "$depth_script" ]; then
        PROBE_FAILURES=$((PROBE_FAILURES + 1))
        emit_event v14_1_portable_tail_transfer FAIL "producer missing: $depth_script"
        printf '  \xE2\x9C\x97 %-42s producer missing\n' v14_1_portable_tail_transfer
        return
    fi

    depth_out=$(BUILD_DIR="$BUILD_DIR_PATH" "$depth_script" 2>&1)
    depth_rc=$?

    verdict=$(python3 -c '
import json, re, sys, time
path, rc = sys.argv[1], sys.argv[2]
# Names are scripts/run_recursion_depth.sh'"'"'s own `${f%.esk}_${mode}`,
# e.g. rec_mutual_tail_cond_d100000000_r / _aot / _aot-xfer.
name_re = re.compile(r"^rec_mutual_tail_(cond|forms|arity)_d\d+_(r|aot|aot-xfer)$")
deadline = time.time() + 10.0
rows, saw_completion, last_err = [], False, None
while True:
    rows, saw_completion = [], False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                rows.append(row)
                if row.get("kind") == "recursion_depth" and row.get("name") == "recursion_depth_gate":
                    saw_completion = True
    except OSError as exc:
        last_err = str(exc)
    if saw_completion or time.time() >= deadline:
        break
    time.sleep(0.25)
if not saw_completion:
    detail = (": " + last_err) if last_err else ""
    print("producer trace at %s never showed its recursion_depth_gate completion record after waiting (exit %s)%s" % (path, rc, detail))
    raise SystemExit(0)
matches = [r for r in rows if r.get("kind") == "recursion_depth"
           and name_re.match(str(r.get("name", "")))]
if not matches:
    print("producer trace has no mutual_tail_{cond,forms,arity} cells despite a complete trace (exit %s)" % rc)
    raise SystemExit(0)
bad = [r for r in matches if r.get("value") != "PASS"]
if bad:
    names = ", ".join(sorted("%s=%s" % (r.get("name", "?"), r.get("value")) for r in bad))
    print("FAIL:%d/%d cells not PASS: %s" % (len(bad), len(matches), names[:400]))
else:
    print("OK:%d cells (JIT -r, AOT musttail, AOT portable aot-xfer dispatcher) all PASS" % len(matches))
' "$depth_trace" "$depth_rc")

    case "$verdict" in
        OK:*)
            emit_event v14_1_portable_tail_transfer PASS \
                "mutual_tail_{cond,forms,arity} depth cells, $verdict (scripts/run_recursion_depth.sh)"
            printf '  \xE2\x9C\x93 %-42s %s\n' v14_1_portable_tail_transfer "$verdict"
            ;;
        *)
            PROBE_FAILURES=$((PROBE_FAILURES + 1))
            emit_event v14_1_portable_tail_transfer FAIL \
                "$verdict -- producer tail: $(printf '%s' "$depth_out" | tail -c 220)"
            printf '  \xE2\x9C\x97 %-42s %s\n' v14_1_portable_tail_transfer "$verdict"
            ;;
    esac
}

echo "Running v1.4-connection oracle probes -> $TRACE_FILE"
echo

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_v14_connection_gate.sh: $ESHKOL_RUN not found - run \`cmake --build $BUILD_DIR_PATH\` first." >&2
    exit 2
fi

# ─────────────────────────────────────────────────────────────────
# Networking (docs/COMPILER_ROADMAP.md #145/#146/#148/#150/#161,
# ROADMAP.md v1.4-connection)
# ─────────────────────────────────────────────────────────────────

# #145 HTTP server: real primitives (lib/agent/c/agent_http_server.c
# http-server-create/-accept/-respond/-close, real bind/listen/accept/recv/
# send) + a real fork+client+server round trip over loopback, already
# CI-wired via the tests/v1_2_edge_cases/*.esk glob.
probe_http_server_source_receipt

# #148 Prometheus metrics + /metrics endpoint: lib/core/metrics.esk
# (make-counter/counter-inc!/metrics-render) is real and wired into
# http-standard-response's /metrics route (lib/core/http_server.esk). No
# existing test drove a real GET /metrics over a socket before this gate;
# tests/v1_2_edge_cases/metrics_http_roundtrip_test.sh (added alongside this
# script) does.
probe_run v14_metrics_http_roundtrip \
    'Prometheus /metrics (#148): real GET /metrics over loopback returns HELP/TYPE + counter value' \
    'BUILD_DIR="$BUILD_DIR_PATH" ./tests/v1_2_edge_cases/metrics_http_roundtrip_test.sh'

# Unix domain sockets: this probe runs the real round-trip (T1-T8) against a
# local AF_UNIX listener, including the environment-gated path.
run_unix_socket_roundtrip() {
    local sock="$WORK/v14_unix_socket_probe.sock"
    local module="$WORK/unix_socket_test.eskb"
    local out="$WORK/unix_socket_test.out"
    rm -f "$sock"
    python3 - "$sock" > "$WORK/unix_socket_listener.out" 2>&1 <<'PYEOF' &
import socket, sys, os
path = sys.argv[1]
if os.path.exists(path):
    os.remove(path)
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(15)
s.bind(path)
s.listen(1)
conn, _ = s.accept()
data = conn.recv(256)
if data == b"vm-socket-ping":
    conn.sendall(b"vm-socket-pong")
conn.close()
s.close()
PYEOF
    local listener_pid=$!
    local waited=0
    while [ ! -S "$sock" ] && [ "$waited" -lt 50 ]; do sleep 0.1; waited=$((waited + 1)); done
    "$ESHKOL_RUN" --profile hosted-vm --emit-eskb "$module" \
        tests/vm/unix_socket_surface_regression.esk >"$WORK/unix_socket_compile.out" 2>&1 || true
    ESHKOL_VM_UNIX_SOCKET_TEST="$sock" "$VM_RUN" "$module" >"$out" 2>&1
    local rc=$?
    wait "$listener_pid" 2>/dev/null
    rm -f "$sock"
    if [ "$rc" -ne 0 ]; then return "$rc"; fi
    grep -q "^FAIL" "$out" && return 1
    grep -q "T4 socket-recv reads response string:" "$out" || return 1
    return 0
}
if [ -x "$VM_RUN" ]; then
    if run_unix_socket_roundtrip; then
        PROBE_TOTAL=$((PROBE_TOTAL + 1))
        emit_event v14_unix_socket_roundtrip PASS \
            "unix-socket-connect/socket-send/socket-recv/socket-close round-trip against a real AF_UNIX listener (T1-T8 of tests/vm/unix_socket_surface_regression.esk, driven with ESHKOL_VM_UNIX_SOCKET_TEST set for the first time by this gate)"
        printf '  \xE2\x9C\x93 %-42s %s\n' v14_unix_socket_roundtrip "Unix domain socket real round-trip"
    else
        PROBE_TOTAL=$((PROBE_TOTAL + 1)); PROBE_FAILURES=$((PROBE_FAILURES + 1))
        emit_event v14_unix_socket_roundtrip FAIL "real AF_UNIX round-trip against tests/vm/unix_socket_surface_regression.esk failed"
        printf '  \xE2\x9C\x97 %-42s FAILED\n' v14_unix_socket_roundtrip
    fi
else
    probe_unverified v14_unix_socket_roundtrip "local VM runner is unavailable; a real AF_UNIX round-trip is required"
fi

# #150 Resource limits: real enforcement (lib/core/resource_limits.cpp,
# ESHKOL_MAX_HEAP/ESHKOL_TIMEOUT_MS/etc.) with a real CTest gate
# (tests/limits/resource_limits_enforcement_gate.sh) that CMake registers
# invoke the focused enforcement gate directly to record this run's result.
if [ -x "$VM_RUN" ]; then
    probe_run v14_resource_limits_enforced \
        'Resource limits (#150): CPU/wall-time/memory/tensor/string/stack ceilings terminate with documented exit codes, and an unreached ceiling changes nothing' \
        '"$REPO_ROOT/tests/limits/resource_limits_enforcement_gate.sh" "$ESHKOL_RUN" "$VM_RUN" "$WORK/limits"'
else
    probe_unverified v14_resource_limits_enforced "local VM runner is unavailable; resource-limit enforcement evidence is required"
fi

# Numeric TCP/UDP use the public module with binary payloads, deadlines,
# endpoints and stale-handle checks. Other protocols still require their own
# implementation and verified runtime evidence.
probe_run v14_tcp_roundtrip 'public core.net TCP loopback round trip' \
    'python3 "$REPO_ROOT/scripts/run_net_gate.py" --runner "$ESHKOL_RUN"'
probe_run v14_udp_roundtrip 'public core.net UDP loopback round trip' \
    'python3 "$REPO_ROOT/scripts/run_net_gate.py" --runner "$ESHKOL_RUN"'
probe_tls_acceptance
if [ -f "$BUILD_DIR_PATH/libeshkol-agent-ffi.a" ] &&
   [ -x "$ESHKOL_RUN" ] &&
   [ -x "$BUILD_DIR_PATH/eshkol-agent-websocket-server-test" ]; then
    probe_run v14_websocket_server_echo \
        'public agent.http_server WebSocket upgrade and echo through JIT/AOT; native frame, control, refusal, and backpressure contracts' \
        '"$BUILD_DIR_PATH/eshkol-agent-websocket-server-test" && python3 "$REPO_ROOT/scripts/test_websocket_server_public.py" --eshkol-run "$ESHKOL_RUN" --source "$REPO_ROOT" --build-dir "$BUILD_DIR_PATH"' \
        '^PASS: public Eshkol AOT WebSocket round-trip$'
else
    probe_unverified v14_websocket_server_echo \
        'optional agent FFI archive or public server test targets are unavailable in this build'
fi

# ─────────────────────────────────────────────────────────────────
# Concurrency (docs/COMPILER_ROADMAP.md #156-#160, M3)
# ─────────────────────────────────────────────────────────────────

# #158 already ships (v1.3.4) and already gates release readiness under
# kind eshkol_smoke/event_loop_works (.icc/completion-oracles.yaml). This
# probe re-emits the SAME evidence under the v1_4_connection family so the
# new oracle proves its own wiring end-to-end rather than relying solely on
# a criterion authored under a different kind before this stanza existed.
if [ -x "$ESHKOL_RUN" ]; then
    probe_run v14_async_io_event_loop \
        'Async I/O event loop (#158): kqueue/epoll/IOCP pipe round-trip via event-loop-poll, timeout, close-then-use fails closed, 1000 open/close cycles' \
        '"$ESHKOL_RUN" -r tests/v1_3_edge_cases/event_loop_test.esk' \
        '^PASS: event_loop_test$'
fi

# #156 threads + mutex + condvar: real pthread-backed core.threads.esk
# (make-mutex/mutex-lock!/make-condvar/make-thread/thread-join), and a real
# concurrency test — 8 parallel-map workers x 1000 mutex-protected
# increments on a shared cell, asserting the exact total (a real race would
# lose increments) — already CI-wired via the tests/v1_2_edge_cases/*.esk
# glob but never fed to the ICC oracle.
probe_run v14_threads_mutex_condvar \
    'Threads + mutex + condvar (#156): 8 parallel-map workers x 1000 mutex-protected increments preserve the exact total' \
    '"$ESHKOL_RUN" -r tests/v1_2_edge_cases/threads_mutex_concurrency_test.esk' \
    'PASS: mutex-protected counter is consistent across threads'

# #157 channels: real bounded-ring-buffer CSP channels (lib/core/channels.esk)
# built on core.threads, with a real parallel-map producer/consumer test.
probe_run v14_channels_csp \
    'Channels (#157): parallel-map producer/consumer traffic over a bounded CSP channel preserves the total and drains cleanly' \
    '"$ESHKOL_RUN" -r tests/v1_2_edge_cases/channels_test.esk' \
    '^Failed: 0$'

# #160 promises/futures: future/force/force-future are real pthread-backed
# VM builtins (lib/backend/vm_region_evac.c: "thunk/result Values, plus a
# live pthread mutex+cond"); core.threads' make-thread is built directly on
# (future thunk). tests/v1_2_edge_cases/future_force_test.esk is the #222
# regression that proves force actually runs the thunk (pre-fix it silently
# fell through to "return as-is").
probe_run v14_promises_futures \
    'Promises/futures (#160): future/force run the thunk and multiple futures resolve independently (regression #222)' \
    '"$ESHKOL_RUN" -r tests/v1_2_edge_cases/future_force_test.esk' \
    '^Failed: 0$'

# Atomic ops (unlisted #-number, "Atomic ops (CAS, fetch-add)" in the
# Concurrency table): real atomic-store!/-load/-exchange!/-compare-exchange!/
# -fetch-add!/-fetch-sub!/-fetch-and!/-fetch-or!/-fetch-xor! with explicit
# memory-order args. The only test today is SEQUENTIAL (API-shape) — no
# cross-thread CAS-under-contention probe exists, so this criterion proves
# the primitive surface only, not cross-thread atomicity; the label says so.
probe_run v14_atomics_surface \
    'Atomic ops: store/load/exchange/compare-exchange/fetch-add/sub/and/or/xor pass the sequential API-surface test' \
    '"$ESHKOL_RUN" -r tests/ffi/low_level_memory_surface_test.esk' \
    'Low-level FFI surface:.*0 failed'

# #159 fibers/coroutines and semaphores/barriers require end-to-end evidence.
probe_unverified v14_fibers_coroutines \
    'requires an end-to-end fibers or coroutines behavior test'
probe_unverified v14_semaphores_barriers \
    'requires an end-to-end semaphore or synchronization-barrier behavior test'

# ─────────────────────────────────────────────────────────────────
# Wire formats (docs/COMPILER_ROADMAP.md #162/#163/#175)
# ─────────────────────────────────────────────────────────────────

# #162 MessagePack: lib/core/msgpack.esk is a real, complete encoder/decoder
# over the deterministic subset (nil/bool/signed+unsigned ints to 32-bit/
# fixstr-str8-str16/bin8-bin16/fixarray-array16/fixmap-map16 — no ext-type
# hooks, no int64/float, a documented v1.8 wire-format substrate subset).
# tests/v1_2_edge_cases/msgpack_test.esk round-trips every type against
# exact wire bytes via core.testing's run-tests.
probe_run v14_msgpack_roundtrip \
    'MessagePack (#162): exact wire bytes, UTF-8 scalar boundaries, embedded NUL, malformed-text refusal and raw binary preservation' \
    'python3 scripts/run_msgpack_gate.py --runner "$ESHKOL_RUN" --vm-runner "$VM_RUN" --all-routes'

# #175 CAS + Merkle trees: lib/core/merkle.esk (merkle-leaf/-inode/-tree/
# -leaves/-proof/-verify plus make-cas/cas-put!/cas-get/cas-has?) backed by
# lib/core/merkle.c hash primitives. tests/v1_2_edge_cases/merkle_test.esk
# self-checks tree build/determinism/order-sensitivity, inclusion-proof
# verify (valid/tampered/empty), and full CAS round-trip incl. dedup.
probe_run v14_cas_merkle_roundtrip \
    'Content-addressable storage + Merkle trees (#175): tree build, inclusion-proof verify (valid/tampered/empty), CAS put/get/dedup round-trip' \
    '"$ESHKOL_RUN" -r tests/v1_2_edge_cases/merkle_test.esk' \
    '^Failed: 0$'

# #163 Protocol Buffers requires a schema-driven proto3 encode/decode test.
probe_unverified v14_protobuf_roundtrip \
    'requires a proto3 schema parser/compiler and generated encoder/decoder round-trip test'

# ─────────────────────────────────────────────────────────────────
# Linear resource types + borrow pattern (docs/design/adr/0004-type-
# system-trajectory.md "v1.4: the resource-sound systems profile")
# ─────────────────────────────────────────────────────────────────

# The generative linear/affine kernel is real and wired into the checker:
# Context::bindLinear/useLinear/checkLinearConstraints (lib/types/
# type_checker.cpp), exercised end-to-end today on the Qubit type (which
# carries TYPE_FLAG_LINEAR, same mechanism ADR-0004 proposes reusing for
# Own/socket typestate) via a real negative compile-fail fixture: a program
# that clones a linear Qubit must FAIL to compile under --strict-types (a
# type error, not a warning -- gradual/default mode only warns, which is
# why this probe passes --strict-types explicitly).
probe_run v14_linear_consume_exactly_once_static \
    'Linear/affine kernel (ADR-0004): a program that uses a linear Qubit twice is REJECTED at compile time under --strict-types (no-cloning), proving the same mechanism v1.4 needs for Own/socket typestate' \
    '"$ESHKOL_RUN" --strict-types tests/typesystem/qubit_no_cloning_test.esk -o "$WORK/qubit_bad" >/dev/null 2>&1; rc=$?; [ "$rc" -ne 0 ] && [ ! -e "$WORK/qubit_bad" ]'

# The opt-in native checked fragment must accept real scoped owners and reject
# the corresponding bad programs. Every static probe compiles a valid control
# as well as the negative fixture; blanket refusal cannot earn a PASS.
RESOURCE_COMPILED_TEST="$REPO_ROOT/tests/v1_4_resource_ownership/test_compiled.py"
RESOURCE_RUNTIME_TEST="$REPO_ROOT/tests/v1_4_resource_ownership/test_runtime.py"
if [ -f "$RESOURCE_COMPILED_TEST" ] && [ -f "$RESOURCE_RUNTIME_TEST" ]; then
    probe_run v14_socket_leak_rejected_statically \
        'native --strict-resources: real socket leak rejected, valid branch-close accepted' \
        'python3 "$RESOURCE_COMPILED_TEST" --runner "$ESHKOL_RUN" --lib-dir "$BUILD_DIR_PATH" --case positive-both-branches-close --case negative-leak --case negative-one-branch-leaks' \
        '^PASS negative-leak: exit=1 resource_diagnostic=True artifact=False$'
    probe_run v14_double_close_rejected_statically \
        'native --strict-resources: duplicate close rejected, valid owner accepted' \
        'python3 "$RESOURCE_COMPILED_TEST" --runner "$ESHKOL_RUN" --lib-dir "$BUILD_DIR_PATH" --case positive-both-branches-close --case negative-double-close --case negative-use-after-close' \
        '^PASS negative-double-close: exit=1 resource_diagnostic=True artifact=False$'
    probe_run v14_borrow_pattern_honored \
        'native --strict-resources: borrow compile controls and actual JIT/AOT port observations with one close' \
        'python3 "$RESOURCE_COMPILED_TEST" --runner "$ESHKOL_RUN" --lib-dir "$BUILD_DIR_PATH" --case positive-borrow-scope --case negative-consume-while-borrowed --case negative-loan-escape && python3 "$RESOURCE_RUNTIME_TEST" --runner "$ESHKOL_RUN" --lib-dir "$BUILD_DIR_PATH" --case borrow' \
        '^PASS borrow aot marker=RESULT: PASS$'
else
    probe_unverified v14_socket_leak_rejected_statically \
        'requires source-bound valid and leaked socket compile controls under --strict-resources'
    probe_unverified v14_double_close_rejected_statically \
        'requires source-bound valid and duplicate close compile controls under --strict-resources'
    probe_unverified v14_borrow_pattern_honored \
        'requires compiled borrow falsifiers and actual JIT/AOT borrow-scope observations'
fi

# ─────────────────────────────────────────────────────────────────
# Stage 4 / v1.4.1 (docs/design/adr/0000-unified-trajectory.md Stage 4:
# "OALR ABI v2 and portable tail transfer"). NOTE: v1.4.1 is its own
# release, one stage AFTER v1.4.0-connection (which this oracle targets),
# per the ADR's own staging and the maintainer-reviewed draft ladder
# (v1.4.0 "the systems profile" [S3] vs v1.4.1 "the ABI release" [S4]).
# These two criteria are included here because the task that authored this
# oracle explicitly asked for S4 coverage; they are scoped and labelled as
# v1.4.1-stage so a v1.4.0 readiness reader does not mistake them for
# blocking the v1.4.0 cut.
# ─────────────────────────────────────────────────────────────────
probe_unverified v14_1_oalr_abi_v2_header \
    'v1.4.1-STAGE (S4): requires a source-bound ABI v2 layout and compatibility contract check'
probe_portable_tail_transfer

# ─────────────────────────────────────────────────────────────────
# Assurance (docs/design/adr/0010-closed-loop-assurance.md A8, A10-A13)
# ─────────────────────────────────────────────────────────────────
probe_external_ci_evidence v14_assurance_a8_tsan_lane \
    '.github/workflows/adversarial-nightly.yml' \
    'requires the applicable A8 policy evidence and a source-bound TSan CI receipt'
probe_package_manifest_complete
probe_diagnostic_corpus_clean
# A12's leak-detection lane (ci.yml's ASan matrix, ASAN_OPTIONS=detect_leaks=1
# with .icc/lsan-suppressions.txt naming every known-acceptable retention) and
# VM region-evacuation poison path (lib/backend/vm_region_evac.c's
# ESHKOL_ARENA_POISON delegation) are real and already running; this fast
# local gate cannot rebuild an ASan-enabled toolchain to reproduce the lane
# itself, so it requests a source-bound CI receipt rather than inferring the
# lane's state, the same way A8's TSan lane is handled.
probe_external_ci_evidence v14_assurance_a12_lsan_lane \
    '.github/workflows/ci.yml' \
    'requires a source-bound ASan/LSan-lane CI receipt for the leak-detection and region-evacuation poison path'
probe_external_ci_evidence v14_assurance_a13_hosted_gpu_lane \
    '.github/workflows/gpu-execution-gate.yml' \
    'requires a source-bound A13 hosted-GPU CI receipt; runner availability is not inferred here'

# ─────────────────────────────────────────────────────────────────
# Distribution (W6, PENDING MAINTAINER RULING R7). These criteria need
# source-bound evidence for the v1.4.0-staged distribution requirements.
# ─────────────────────────────────────────────────────────────────
probe_unverified v14_distribution_pjrt_client_spike \
    'PENDING R7: requires source-bound evidence for the PJRT client spike requirement'
probe_unverified v14_distribution_xla_multi_device_single_host \
    'PENDING R7: requires source-bound multi-device single-host XLA execution evidence'
probe_unverified v14_distribution_native_collectives_over_sockets \
    'PENDING R7: requires source-bound native collective-over-sockets execution evidence'

echo
echo "Passed: $((PROBE_TOTAL - PROBE_FAILURES))"
echo "Failed: $PROBE_FAILURES"
echo "Total:  $PROBE_TOTAL"
echo "Trace written: $TRACE_FILE"

if [ "$PROBE_FAILURES" -eq 0 ]; then
    emit_event v14_connection_sweep_clean PASS "all $PROBE_TOTAL v1.4-connection probes green"
else
    emit_event v14_connection_sweep_clean FAIL "$PROBE_FAILURES of $PROBE_TOTAL v1.4-connection probes not satisfied (expected -- v1.4-connection has not shipped; this sweep is the honest running count, not a release gate)"
fi

exit 0
