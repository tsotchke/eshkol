#!/usr/bin/env bash
# run_sanitizer_fuzz.sh - ASan/UBSan oracle over differential and test corpora.
#
# The differential harness checks semantic agreement. This harness is additive:
# it ignores ordinary compile/runtime failures unless a sanitizer emits a report.

set -u
cd "$(dirname "$0")/.."

# Never let an aborting sanitizer process dump a (multi-GB, huge-shadow-memory)
# core. This is a hard peak-disk guard independent of ASAN_OPTIONS.
ulimit -c 0 2>/dev/null || true

REPO_ROOT="$(pwd)"
BUILD_DIR="build-asan-ubsan"
REPORT="SANITIZER_FUZZ_REPORT.md"
WORK_ROOT="artifacts/sanitizer-fuzz"
TRACE_FILE="scripts/icc_traces/sanitizer_fuzz.jsonl"
SEED=42
COUNT=200
TIMEOUT_RUN=90
TASK_BASE=160
SKIP_BUILD=0
SELF_TEST_ONLY=0
NO_GENERATED=0
LIMIT=0
# Default is a bounded gate sweep that completes fast and keeps disk peak tiny.
# The full generated+tests corpus is only run with --full / ESHKOL_FUZZ_FULL=1
# and is intended for a CI/mesh node with disk headroom, not interactive runs.
GATE_LIMIT="${ESHKOL_FUZZ_GATE_LIMIT:-150}"
FULL_SWEEP="${ESHKOL_FUZZ_FULL:-0}"
# Disk budget precedence: --max-artifacts-mb (env ESHKOL_FUZZ_MAX_MB) beats
# --max-artifacts-gb (env ESHKOL_FUZZ_MAX_GB); default is 300 MB.
DEFAULT_MAX_ARTIFACTS_MB=300
MAX_ARTIFACTS_MB="${ESHKOL_FUZZ_MAX_MB:-}"
MAX_ARTIFACTS_GB="${ESHKOL_FUZZ_MAX_GB:-}"
# Kernel-enforced per-file size cap (KiB) applied to the sweep children only,
# after the build. Bounds any runaway stdout/stderr (and stray temp/core files)
# so a single case cannot spike disk between budget checks. Comfortably above a
# per-case AOT binary (~25 MB) but far below the artifact cap.
SWEEP_FILE_CAP_KB="${ESHKOL_FUZZ_FILE_CAP_KB:-65536}"
MAX_RETAINED_CRASHES="${ESHKOL_FUZZ_MAX_RETAINED_CRASHES:-200}"

usage() {
    cat <<'EOF'
Usage: scripts/run_sanitizer_fuzz.sh [options]

Builds or reuses build-asan-ubsan and runs a corpus of tests/**/*.esk inputs
plus a deterministic generated differential fuzz corpus through -r and AOT,
writing SANITIZER_FUZZ_REPORT.md with ASan/UBSan reports deduped by top frame.

By default this runs a BOUNDED gate sweep (first 150 corpus files) so it
finishes fast and keeps the on-disk peak tiny (<300 MB). The ICC trace is
written for the bounded run, so the .icc sanitizer_fuzz gate still passes.
Use --full (or ESHKOL_FUZZ_FULL=1) to run the entire generated+tests corpus;
the full sweep is meant for a CI/mesh node with disk headroom, NOT interactive
runs.

Options:
  --skip-build             Reuse the existing sanitizer build.
  --build-dir DIR          Sanitizer build directory to use (default: build-asan-ubsan).
  --report PATH            Markdown report path (default: SANITIZER_FUZZ_REPORT.md).
  --work-dir DIR           Log/work directory (default: artifacts/sanitizer-fuzz).
  --trace-file PATH        ICC trace path (default: scripts/icc_traces/sanitizer_fuzz.jsonl).
  --seed N                 Seed for generated differential programs (default: 42).
  --count N                Generated differential program count (default: 200).
  --no-generated           Do not generate additional differential fuzz programs.
  --timeout SEC            Per process timeout (default: 90).
  --full                   Run the entire corpus, not the bounded gate sweep
                           (env: ESHKOL_FUZZ_FULL=1). For CI/mesh nodes only.
  --gate-limit N           Bounded gate sweep size (default: 150,
                           env: ESHKOL_FUZZ_GATE_LIMIT).
  --limit N                Run only the first N corpus files (debugging only;
                           overrides the bounded/full selection).
  --max-artifacts-mb N     Hard cap for sanitizer-fuzz work dir in MB
                           (default: 300, env: ESHKOL_FUZZ_MAX_MB).
  --max-artifacts-gb N     Hard cap in GB (env: ESHKOL_FUZZ_MAX_GB). Overridden
                           by --max-artifacts-mb when both are given.
  --max-retained-crashes N Cap retained sanitizer logs/repros (default: 200).
  --task-base N            First ESH id to allocate for sanitizer findings (default: 160).
  --self-test-only         Run only the synthetic ESH-0116-class sanitizer check.
  -h, --help               Show this help.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build) SKIP_BUILD=1; shift ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --report) REPORT="$2"; shift 2 ;;
        --work-dir) WORK_ROOT="$2"; shift 2 ;;
        --trace-file) TRACE_FILE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --count) COUNT="$2"; shift 2 ;;
        --no-generated) NO_GENERATED=1; shift ;;
        --timeout) TIMEOUT_RUN="$2"; shift 2 ;;
        --full) FULL_SWEEP=1; shift ;;
        --gate-limit) GATE_LIMIT="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --max-artifacts-mb) MAX_ARTIFACTS_MB="$2"; shift 2 ;;
        --max-artifacts-gb) MAX_ARTIFACTS_GB="$2"; shift 2 ;;
        --max-retained-crashes) MAX_RETAINED_CRASHES="$2"; shift 2 ;;
        --task-base) TASK_BASE="$2"; shift 2 ;;
        --self-test-only) SELF_TEST_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "run_sanitizer_fuzz.sh: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$BUILD_DIR" in
    /*) BUILD_DIR_ABS="$BUILD_DIR" ;;
    *) BUILD_DIR_ABS="$REPO_ROOT/$BUILD_DIR" ;;
esac
case "$REPORT" in
    /*) REPORT_ABS="$REPORT" ;;
    *) REPORT_ABS="$REPO_ROOT/$REPORT" ;;
esac
case "$WORK_ROOT" in
    /*) WORK_ROOT_ABS="$WORK_ROOT" ;;
    *) WORK_ROOT_ABS="$REPO_ROOT/$WORK_ROOT" ;;
esac
case "$TRACE_FILE" in
    /*) TRACE_FILE_ABS="$TRACE_FILE" ;;
    *) TRACE_FILE_ABS="$REPO_ROOT/$TRACE_FILE" ;;
esac

LOG_DIR="$WORK_ROOT_ABS/logs"
BIN_DIR="$WORK_ROOT_ABS/bin"
TMP_DIR="$WORK_ROOT_ABS/tmp"
REPRO_DIR="$WORK_ROOT_ABS/repros"
GENERATED_DIR="$WORK_ROOT_ABS/generated-differential"
RUNS_TSV="$WORK_ROOT_ABS/runs.tsv"
SELF_TSV="$WORK_ROOT_ABS/selftest.tsv"
CORPUS_RAW="$WORK_ROOT_ABS/corpus.raw"
CORPUS_LIST="$WORK_ROOT_ABS/corpus.txt"
AOT_BIN="$BIN_DIR/aot-test.bin"
RETAINED_SIGNATURES="$WORK_ROOT_ABS/retained-signatures.txt"
ESHKOL_RUN="$BUILD_DIR_ABS/eshkol-run"
TASK_DIR="$REPO_ROOT/.swarm/tasks"
STARTED_AT="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

case "$MAX_RETAINED_CRASHES" in
    ''|*[!0-9]*)
        echo "run_sanitizer_fuzz.sh: --max-retained-crashes must be a non-negative integer: $MAX_RETAINED_CRASHES" >&2
        exit 2
        ;;
esac

if ! MAX_ARTIFACTS_KB="$(python3 - "$MAX_ARTIFACTS_MB" "$MAX_ARTIFACTS_GB" "$DEFAULT_MAX_ARTIFACTS_MB" <<'PY'
import math
import sys

mb_arg, gb_arg, default_mb = sys.argv[1], sys.argv[2], sys.argv[3]


def parse(value, unit_kb, label):
    try:
        n = float(value)
    except ValueError:
        print("run_sanitizer_fuzz.sh: %s must be a positive number: %s" % (label, value),
              file=sys.stderr)
        raise SystemExit(1)
    if not math.isfinite(n) or n <= 0:
        print("run_sanitizer_fuzz.sh: %s must be a positive number: %s" % (label, value),
              file=sys.stderr)
        raise SystemExit(1)
    return max(1, int(math.ceil(n * unit_kb)))


# Precedence: --max-artifacts-mb, then --max-artifacts-gb, then default MB.
if mb_arg:
    print(parse(mb_arg, 1024, "--max-artifacts-mb"))
elif gb_arg:
    print(parse(gb_arg, 1024 * 1024, "--max-artifacts-gb"))
else:
    print(parse(default_mb, 1024, "default artifact cap"))
PY
)"; then
    exit 2
fi

cleanup_sanitizer_fuzz() {
    rm -rf "$BIN_DIR" "$TMP_DIR" "$GENERATED_DIR"
    rm -rf "$WORK_ROOT_ABS/self-test"
    rm -f "$CORPUS_RAW" "$CORPUS_LIST" "$RUNS_TSV" "$SELF_TSV" "$RETAINED_SIGNATURES" "$AOT_BIN"
}
trap cleanup_sanitizer_fuzz EXIT

rm -rf "$LOG_DIR" "$REPRO_DIR" "$BIN_DIR" "$TMP_DIR" "$GENERATED_DIR"
rm -f "$CORPUS_RAW" "$AOT_BIN"
if ! mkdir -p "$LOG_DIR" "$REPRO_DIR" "$BIN_DIR" "$TMP_DIR" "$GENERATED_DIR" "$(dirname "$TRACE_FILE_ABS")" "$TASK_DIR"; then
    echo "run_sanitizer_fuzz.sh: failed to create sanitizer-fuzz work directories under $WORK_ROOT_ABS" >&2
    exit 2
fi
: > "$RUNS_TSV" || exit 2
: > "$SELF_TSV" || exit 2
: > "$TRACE_FILE_ABS" || exit 2
: > "$RETAINED_SIGNATURES" || exit 2

# disable_coredump=1 stops the sanitizer runtime from producing a (potentially
# multi-GB, huge shadow-memory) core when it aborts. Reports still go to stderr
# (default log_path), never to an on-disk log file.
if [ "$(uname -s)" = "Darwin" ]; then
    DEFAULT_ASAN_OPTIONS="detect_leaks=0:halt_on_error=1:abort_on_error=1:disable_coredump=1:print_stacktrace=1:strict_string_checks=1:detect_stack_use_after_return=0:symbolize=1"
else
    DEFAULT_ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:abort_on_error=1:disable_coredump=1:print_stacktrace=1:strict_string_checks=1:detect_stack_use_after_return=0:symbolize=1"
fi
DEFAULT_UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1:abort_on_error=1:disable_coredump=1"
export ASAN_OPTIONS="${ASAN_OPTIONS:-$DEFAULT_ASAN_OPTIONS}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-$DEFAULT_UBSAN_OPTIONS}"

PEAK_ARTIFACT_KB=0
RETAINED_OUT=""
RETAINED_ERR=""

artifact_usage_kb() {
    if [ ! -d "$WORK_ROOT_ABS" ]; then
        echo 0
        return
    fi
    du -sk "$WORK_ROOT_ABS" 2>/dev/null | awk '{print $1 + 0}'
}

format_kb() {
    awk -v kb="$1" 'BEGIN {
        if (kb >= 1048576) printf "%.2f GiB", kb / 1048576;
        else if (kb >= 1024) printf "%.1f MiB", kb / 1024;
        else printf "%d KiB", kb;
    }'
}

check_disk_budget() {
    local context="$1"
    local used
    used="$(artifact_usage_kb)"
    if [ "$used" -gt "$PEAK_ARTIFACT_KB" ]; then
        PEAK_ARTIFACT_KB="$used"
    fi
    if [ "$used" -gt "$MAX_ARTIFACTS_KB" ]; then
        echo "run_sanitizer_fuzz.sh: artifact budget exceeded during $context: $WORK_ROOT_ABS uses $(format_kb "$used") > cap $(format_kb "$MAX_ARTIFACTS_KB") (--max-artifacts-gb $MAX_ARTIFACTS_GB, env ESHKOL_FUZZ_MAX_GB). Aborting." >&2
        exit 3
    fi
}

run_guarded() {
    LC_ALL=C LANG=C perl -MPOSIX=':sys_wait_h' -e '
        my $seconds = shift;
        my $pid = fork();
        die "fork failed: $!" unless defined $pid;
        if ($pid == 0) {
            setpgrp(0, 0);
            exec @ARGV or exit 127;
        }
        local $SIG{ALRM} = sub {
            kill "TERM", -$pid;
            select undef, undef, undef, 0.5;
            kill "KILL", -$pid;
            exit 124;
        };
        alarm $seconds;
        waitpid($pid, 0);
        my $status = $?;
        alarm 0;
        if (WIFEXITED($status)) {
            exit WEXITSTATUS($status);
        }
        if (WIFSIGNALED($status)) {
            exit 128 + WTERMSIG($status);
        }
        exit 125;
    ' \
        "$1" "${@:2}"
}

append_run() {
    local src="$1" axis="$2" phase="$3" rc="$4" out="$5" err="$6"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$src" "$axis" "$phase" "$rc" "$out" "$err" >> "$RUNS_TSV"
}

path_hash() {
    printf '%s' "$1" | cksum | awk '{print $1}'
}

has_sanitizer_report() {
    grep -E 'ERROR: AddressSanitizer:|runtime error:|UndefinedBehaviorSanitizer:DEADLYSIGNAL|SUMMARY: AddressSanitizer:|SUMMARY: UndefinedBehaviorSanitizer:' "$@" >/dev/null 2>&1
}

sanitizer_top_frame() {
    python3 - "$REPO_ROOT" "$@" <<'PY'
import os
import re
import sys

repo_root = os.path.abspath(sys.argv[1])
paths = sys.argv[2:]
lines = []
for path in paths:
    try:
        with open(path, "r", errors="replace") as f:
            lines.extend(f.read().splitlines())
    except OSError:
        pass

start = None
for i, line in enumerate(lines):
    if (
        "ERROR: AddressSanitizer:" in line
        or "UndefinedBehaviorSanitizer:DEADLYSIGNAL" in line
        or ("runtime error:" in line and "SUMMARY:" not in line)
        or "SUMMARY: AddressSanitizer:" in line
        or "SUMMARY: UndefinedBehaviorSanitizer:" in line
    ):
        start = i
        break

if start is None:
    raise SystemExit(1)

top = ""
for line in lines[start : start + 90]:
    if re.match(r"\s*#0\b", line):
        top = line.strip()
        break
if not top:
    for line in lines[start : start + 90]:
        if "SUMMARY:" in line:
            top = line.strip()
            break
if not top:
    top = lines[start].strip()

top = top.replace(repo_root, ".")
top = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", top)
top = re.sub(r"\s+", " ", top)
print(top[:240])
PY
}

retain_sanitizer_logs() {
    local src="$1" axis="$2" phase="$3" idx="$4" out="$5" err="$6"
    local top sig_hash src_hash retained_count base repro

    RETAINED_OUT=""
    RETAINED_ERR=""

    if ! has_sanitizer_report "$out" "$err"; then
        rm -f "$out" "$err"
        return
    fi

    top="$(sanitizer_top_frame "$out" "$err" || true)"
    if [ -z "$top" ]; then
        top="unknown sanitizer signature"
    fi

    if grep -Fx -- "$top" "$RETAINED_SIGNATURES" >/dev/null 2>&1; then
        rm -f "$out" "$err"
        return
    fi

    retained_count="$(wc -l < "$RETAINED_SIGNATURES" | tr -d ' ')"
    if [ "$retained_count" -ge "$MAX_RETAINED_CRASHES" ]; then
        rm -f "$out" "$err"
        return
    fi

    sig_hash="$(path_hash "$top")"
    src_hash="$(path_hash "$src")"
    RETAINED_OUT="$LOG_DIR/${idx}_${axis}_${phase}_${sig_hash}_${src_hash}.out"
    RETAINED_ERR="$LOG_DIR/${idx}_${axis}_${phase}_${sig_hash}_${src_hash}.err"
    mv "$out" "$RETAINED_OUT"
    mv "$err" "$RETAINED_ERR"
    printf '%s\n' "$top" >> "$RETAINED_SIGNATURES"

    base="$(basename "$src")"
    repro="$REPRO_DIR/${sig_hash}_${src_hash}_${base}.crash"
    if [ ! -f "$repro" ]; then
        cp "$src" "$repro" 2>/dev/null || true
    fi

    check_disk_budget "retaining sanitizer repro for ${src#$REPO_ROOT/}"
}

run_self_test() {
    local d="$WORK_ROOT_ABS/self-test"
    local src="$d/esh0116_unterminated_numeric_buffer.c"
    local bin="$d/esh0116_unterminated_numeric_buffer"
    local cout="$d/compile.out"
    local cerr="$d/compile.err"
    local out="$d/run.out"
    local err="$d/run.err"
    local cc_bin="${CC:-cc}"
    mkdir -p "$d"

    cat > "$src" <<'C'
#include <stddef.h>
#include <stdio.h>

__attribute__((noinline))
static long parse_digits_like_atoll(const char *buf) {
    long value = 0;
    for (size_t i = 0; buf[i] >= '0' && buf[i] <= '9'; i++) {
        value = value * 10 + (buf[i] - '0');
    }
    return value;
}

int main(void) {
    char buf[4] = {'1', '2', '3', '4'};
    volatile long parsed = parse_digits_like_atoll(buf);
    printf("%ld\n", parsed);
    return 0;
}
C

    if ! command -v "$cc_bin" >/dev/null 2>&1; then
        {
            printf 'status\tSKIP\n'
            printf 'summary\tC compiler not found: %s\n' "$cc_bin"
            printf 'source\t%s\n' "$src"
            printf 'stdout\t%s\n' "$out"
            printf 'stderr\t%s\n' "$err"
        } > "$SELF_TSV"
        return
    fi

    # Pass-path compile carries no debug info (-g0) so no .dSYM bundle is ever
    # produced; a symbolized repro is only rebuilt (below) if the case trips.
    "$cc_bin" -O1 -g0 -fsanitize=address,undefined -fno-sanitize-recover=all \
        -fno-omit-frame-pointer "$src" -o "$bin" >"$cout" 2>"$cerr"
    local crc=$?
    if [ "$crc" -ne 0 ] || [ ! -x "$bin" ]; then
        rm -f "$bin"
        rm -rf "$bin.dSYM"
        {
            printf 'status\tFAIL\n'
            printf 'summary\tSynthetic sanitizer repro failed to compile rc=%s\n' "$crc"
            printf 'source\t%s\n' "$src"
            printf 'stdout\t%s\n' "$cout"
            printf 'stderr\t%s\n' "$cerr"
        } > "$SELF_TSV"
        return
    fi

    run_guarded 15 "$bin" >"$out" 2>"$err"
    local rrc=$?
    rm -f "$bin"
    rm -rf "$bin.dSYM"
    if grep -E 'ERROR: AddressSanitizer:|runtime error:|SUMMARY: UndefinedBehaviorSanitizer:' "$out" "$err" >/dev/null 2>&1; then
        # The case tripped: rebuild THIS single source with -g into a temp path to
        # capture a symbolized repro, fold it into the retained stderr, then
        # delete the debug binary/dSYM so no debug bloat lingers.
        local symbin="$d/esh0116_symbolized"
        if "$cc_bin" -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
            -fno-omit-frame-pointer "$src" -o "$symbin" >/dev/null 2>&1 && [ -x "$symbin" ]; then
            run_guarded 15 "$symbin" >>"$out" 2>>"$err" || true
        fi
        rm -f "$symbin"
        rm -rf "$symbin.dSYM"
        {
            printf 'status\tPASS\n'
            printf 'summary\tSynthetic unterminated numeric buffer repro emitted a sanitizer report rc=%s\n' "$rrc"
            printf 'source\t%s\n' "$src"
            printf 'stdout\t%s\n' "$out"
            printf 'stderr\t%s\n' "$err"
        } > "$SELF_TSV"
    else
        {
            printf 'status\tFAIL\n'
            printf 'summary\tSynthetic unterminated numeric buffer repro did not emit a sanitizer report rc=%s\n' "$rrc"
            printf 'source\t%s\n' "$src"
            printf 'stdout\t%s\n' "$out"
            printf 'stderr\t%s\n' "$err"
        } > "$SELF_TSV"
    fi
}

collect_corpus() {
    : > "$CORPUS_RAW"
    if [ "$NO_GENERATED" -eq 0 ] && [ "$COUNT" -gt 0 ]; then
        python3 scripts/gen_differential.py \
            --seed "$SEED" --count "$COUNT" --emit-only "$GENERATED_DIR" || exit 2
        find "$GENERATED_DIR" -type f -name '*.esk' -print >> "$CORPUS_RAW" || exit 2
    fi
    find "$REPO_ROOT/tests" -type f -name '*.esk' -print >> "$CORPUS_RAW" || exit 2
    sort -u "$CORPUS_RAW" > "$CORPUS_LIST" || exit 2
}

run_r_axis() {
    local src="$1" idx="$2"
    local h out err rc
    h="$(path_hash "$src")"
    out="$TMP_DIR/${idx}_r_${h}.out"
    err="$TMP_DIR/${idx}_r_${h}.err"
    run_guarded "$TIMEOUT_RUN" "$ESHKOL_RUN" -r "$src" >"$out" 2>"$err"
    rc=$?
    retain_sanitizer_logs "$src" "r" "run" "$idx" "$out" "$err"
    append_run "$src" "r" "run" "$rc" "$RETAINED_OUT" "$RETAINED_ERR"
}

run_aot_axis() {
    local src="$1" idx="$2" axis="$3" opt="$4"
    local h bin cout cerr out err crc rrc
    h="$(path_hash "$src")"
    bin="$AOT_BIN"
    cout="$TMP_DIR/${idx}_${axis}_${h}.compile.out"
    cerr="$TMP_DIR/${idx}_${axis}_${h}.compile.err"
    out="$TMP_DIR/${idx}_${axis}_${h}.run.out"
    err="$TMP_DIR/${idx}_${axis}_${h}.run.err"

    rm -f "$bin"
    rm -rf "$bin.dSYM"
    run_guarded "$TIMEOUT_RUN" "$ESHKOL_RUN" "$opt" "$src" -o "$bin" >"$cout" 2>"$cerr"
    crc=$?
    retain_sanitizer_logs "$src" "$axis" "compile" "$idx" "$cout" "$cerr"
    append_run "$src" "$axis" "compile" "$crc" "$RETAINED_OUT" "$RETAINED_ERR"
    # Account for the freshly built (instrumented) binary while it still exists,
    # so the peak is measured accurately and a spike aborts immediately.
    check_disk_budget "compiled $axis binary for ${src#$REPO_ROOT/}"
    if [ "$crc" -ne 0 ] || [ ! -x "$bin" ]; then
        rm -f "$bin"
        rm -rf "$bin.dSYM"
        return
    fi

    run_guarded "$TIMEOUT_RUN" "$bin" >"$out" 2>"$err"
    rrc=$?
    check_disk_budget "ran $axis binary for ${src#$REPO_ROOT/}"
    retain_sanitizer_logs "$src" "$axis" "run" "$idx" "$out" "$err"
    append_run "$src" "$axis" "run" "$rrc" "$RETAINED_OUT" "$RETAINED_ERR"
    rm -f "$bin"
    rm -rf "$bin.dSYM"
}

write_report() {
    local finished_at="$1" total_sources="$2"
    python3 - "$REPORT_ABS" "$RUNS_TSV" "$SELF_TSV" "$TRACE_FILE_ABS" \
        "$TASK_DIR" "$TASK_BASE" "$REPO_ROOT" "$BUILD_DIR_ABS" "$SEED" "$COUNT" \
        "$total_sources" "$STARTED_AT" "$finished_at" "$ASAN_OPTIONS" \
        "$UBSAN_OPTIONS" "$WORK_ROOT_ABS" <<'PY'
import collections
import datetime
import json
import os
import re
import sys

(
    report_path,
    runs_tsv,
    self_tsv,
    trace_file,
    task_dir,
    task_base,
    repo_root,
    build_dir,
    seed,
    count,
    total_sources,
    started_at,
    finished_at,
    asan_options,
    ubsan_options,
    work_root,
) = sys.argv[1:]

task_base = int(task_base)
total_sources = int(total_sources)

def rel(path):
    path = os.path.abspath(path)
    root = os.path.abspath(repo_root)
    if path == root:
        return "."
    if path.startswith(root + os.sep):
        return path[len(root) + 1 :]
    return path

def md(s):
    return str(s).replace("|", "\\|").replace("\n", " ")

def read_text(path):
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return ""

def normalize_frame(line):
    s = line.strip().replace(os.path.abspath(repo_root), ".")
    s = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", s)
    s = re.sub(r"\s+", " ", s)
    return s[:240]

def report_starts(lines):
    starts = []
    for i, line in enumerate(lines):
        if "ERROR: AddressSanitizer:" in line:
            m = re.search(r"ERROR: AddressSanitizer:\s*([^ ]+)", line)
            starts.append((i, "AddressSanitizer", m.group(1) if m else "error", line.strip()))
        elif "UndefinedBehaviorSanitizer:DEADLYSIGNAL" in line:
            starts.append((i, "UndefinedBehaviorSanitizer", "DEADLYSIGNAL", line.strip()))
        elif "runtime error:" in line and "SUMMARY:" not in line:
            msg = re.sub(r"^.*runtime error:\s*", "", line.strip())
            starts.append((i, "UndefinedBehaviorSanitizer", msg[:160], line.strip()))

    if starts:
        return starts

    for i, line in enumerate(lines):
        if "SUMMARY: AddressSanitizer:" in line:
            m = re.search(r"SUMMARY: AddressSanitizer:\s*([^ ]+)", line)
            starts.append((i, "AddressSanitizer", m.group(1) if m else "summary", line.strip()))
        elif "SUMMARY: UndefinedBehaviorSanitizer:" in line:
            starts.append((i, "UndefinedBehaviorSanitizer", "summary", line.strip()))
    return starts

def extract_reports(text):
    lines = text.splitlines()
    reports = []
    for start, kind, subtype, headline in report_starts(lines):
        chunk = lines[start : min(len(lines), start + 90)]
        top = ""
        summary = ""
        for line in chunk:
            if re.match(r"\s*#0\b", line):
                top = normalize_frame(line)
                break
        for line in chunk:
            if "SUMMARY:" in line:
                summary = normalize_frame(line)
                if not top:
                    top = summary
                break
        if not top:
            top = normalize_frame(headline)
        snippet = "\n".join(line.rstrip() for line in chunk[:36]).replace(os.path.abspath(repo_root), ".")
        reports.append(
            {
                "kind": kind,
                "subtype": subtype,
                "top_frame": top,
                "summary": summary,
                "snippet": snippet[:6000],
            }
        )
    return reports

def read_tsv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 6:
                rows.append(
                    {
                        "source": parts[0],
                        "axis": parts[1],
                        "phase": parts[2],
                        "rc": parts[3],
                        "stdout": parts[4],
                        "stderr": parts[5],
                    }
                )
    return rows

def read_self(path):
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, "r", errors="replace") as f:
        for line in f:
            key, sep, value = line.rstrip("\n").partition("\t")
            if sep:
                data[key] = value
    return data

runs = read_tsv(runs_tsv)
self_data = read_self(self_tsv)
groups = collections.OrderedDict()
run_counts = collections.Counter((row["axis"], row["phase"], row["rc"]) for row in runs)
aot_o0_runs = sum(1 for row in runs if row["axis"] == "aot-o0" and row["phase"] == "run")
aot_o2_runs = sum(1 for row in runs if row["axis"] == "aot-o2" and row["phase"] == "run")
r_runs = sum(1 for row in runs if row["axis"] == "r" and row["phase"] == "run")
coverage_problems = []
if total_sources > 0 and r_runs == 0:
    coverage_problems.append("no -r records were produced")
if total_sources > 0 and aot_o0_runs == 0:
    coverage_problems.append("no AOT -O0 binaries were run")
if total_sources > 0 and aot_o2_runs == 0:
    coverage_problems.append("no AOT -O2 binaries were run")
coverage_status = "PASS" if not coverage_problems else "FAIL"
coverage_summary = (
    "-r runs=%d, AOT -O0 runs=%d, AOT -O2 runs=%d"
    % (r_runs, aot_o0_runs, aot_o2_runs)
)
if coverage_problems:
    coverage_summary += "; " + "; ".join(coverage_problems)

for row in runs:
    for stream in ("stdout", "stderr"):
        log_path = row[stream]
        text = read_text(log_path)
        if not text:
            continue
        for rep in extract_reports(text):
            key = "\t".join([rep["kind"], rep["subtype"], rep["top_frame"]])
            group = groups.setdefault(
                key,
                {
                    "kind": rep["kind"],
                    "subtype": rep["subtype"],
                    "top_frame": rep["top_frame"],
                    "summary": rep["summary"],
                    "hits": 0,
                    "samples": [],
                    "snippets": [],
                    "task_id": "",
                    "signature": key,
                },
            )
            group["hits"] += 1
            if len(group["samples"]) < 5:
                group["samples"].append(
                    {
                        "source": rel(row["source"]),
                        "axis": row["axis"],
                        "phase": row["phase"],
                        "rc": row["rc"],
                        "log": rel(log_path),
                    }
                )
            if len(group["snippets"]) < 1:
                group["snippets"].append(rep["snippet"])

existing_by_signature = {}
if os.path.isdir(task_dir):
    for name in sorted(os.listdir(task_dir)):
        if not re.match(r"ESH-\d{4}\.json$", name):
            continue
        path = os.path.join(task_dir, name)
        try:
            with open(path, "r", errors="replace") as f:
                data = json.load(f)
        except Exception:
            continue
        sig = data.get("sanitizer_signature")
        if sig:
            existing_by_signature[sig] = data.get("id", name[:-5])

def next_task_id(used):
    n = task_base
    while True:
        tid = "ESH-%04d" % n
        if tid not in used and not os.path.exists(os.path.join(task_dir, tid + ".json")):
            return tid
        n += 1

used_task_ids = set(existing_by_signature.values())
for group in groups.values():
    if group["signature"] in existing_by_signature:
        group["task_id"] = existing_by_signature[group["signature"]]
        continue
    tid = next_task_id(used_task_ids)
    used_task_ids.add(tid)
    group["task_id"] = tid
    sample = group["samples"][0] if group["samples"] else {}
    task = {
        "id": tid,
        "status": "open",
        "priority": "high",
        "workstream": "sanitizer-fuzz",
        "title": "Sanitizer fuzz: %s %s" % (group["kind"], group["subtype"]),
        "goal": (
            "Fix sanitizer finding from scripts/run_sanitizer_fuzz.sh. "
            "First repro: %(source)s via %(axis)s/%(phase)s. "
            "Top frame: %(top)s. Log: %(log)s."
            % {
                "source": sample.get("source", ""),
                "axis": sample.get("axis", ""),
                "phase": sample.get("phase", ""),
                "top": group["top_frame"],
                "log": sample.get("log", ""),
            }
        ),
        "status_note": "OPEN. Dedup signature is sanitizer kind/subtype plus normalized top frame.",
        "acceptance": [
            "Root cause is fixed or explicitly classified with a narrower XKNOWN.",
            "A minimal repro remains in tests/ or is added next to the affected corpus.",
            "scripts/run_sanitizer_fuzz.sh no longer reports this signature.",
        ],
        "blocked_by": [],
        "file_globs": [sample.get("source", "tests/**/*.esk")],
        "sanitizer_signature": group["signature"],
        "sanitizer_kind": group["kind"],
        "sanitizer_subtype": group["subtype"],
        "sanitizer_top_frame": group["top_frame"],
    }
    with open(os.path.join(task_dir, tid + ".json"), "w") as f:
        json.dump(task, f, indent=2)
        f.write("\n")

events = []
self_status = self_data.get("status", "FAIL")
self_summary = self_data.get("summary", "self-test metadata missing")
events.append(
    {
        "kind": "sanitizer_fuzz",
        "name": "sanitizer_fuzz_selftest_esh0116",
        "value": self_status,
        "snippet": self_summary,
        "confidence": 0.95,
    }
)
events.append(
    {
        "kind": "sanitizer_fuzz",
        "name": "sanitizer_fuzz_aot_coverage",
        "value": coverage_status,
        "snippet": coverage_summary,
        "confidence": 0.95,
    }
)

if groups:
    for group in groups.values():
        sample = group["samples"][0] if group["samples"] else {}
        events.append(
            {
                "kind": "sanitizer_fuzz",
                "name": "sanitizer_fuzz_finding_%s" % group["task_id"].lower().replace("-", "_"),
                "value": "FAIL",
                "snippet": "%s %s at %s first_repro=%s"
                % (group["kind"], group["subtype"], group["top_frame"], sample.get("source", "")),
                "confidence": 0.95,
            }
        )

clean_value = "PASS" if not groups else "FAIL"
events.append(
    {
        "kind": "sanitizer_fuzz",
        "name": "sanitizer_fuzz_clean",
        "value": clean_value,
        "snippet": "%d unique sanitizer signatures across %d run records"
        % (len(groups), len(runs)),
        "confidence": 0.95,
    }
)
with open(trace_file, "w") as f:
    for event in events:
        f.write(json.dumps(event, sort_keys=True) + "\n")

def command_for(sample):
    source = sample.get("source", "")
    axis = sample.get("axis", "")
    phase = sample.get("phase", "")
    runner = rel(os.path.join(build_dir, "eshkol-run"))
    if axis == "r":
        return "%s -r %s" % (runner, source)
    if axis == "aot-o0":
        return "%s -O0 %s -o /tmp/eshkol-sanitize-repro.bin" % (runner, source)
    if axis == "aot-o2":
        return "%s -O2 %s -o /tmp/eshkol-sanitize-repro.bin" % (runner, source)
    if phase == "run":
        return "/tmp/eshkol-sanitize-repro.bin"
    return "%s %s" % (axis, source)

with open(report_path, "w") as f:
    f.write("# Sanitizer Fuzz Report\n\n")
    f.write("- Started: `%s`\n" % started_at)
    f.write("- Finished: `%s`\n" % finished_at)
    f.write("- Build: `%s`\n" % rel(build_dir))
    f.write("- Corpus files: `%d`\n" % total_sources)
    f.write("- Run records: `%d`\n" % len(runs))
    f.write("- Axes: `-r`, `AOT -O0 compile/run`, `AOT -O2 compile/run`\n")
    f.write("- Generated differential fuzz: seed `%s`, count `%s`\n" % (seed, count))
    f.write("- ASAN_OPTIONS: `%s`\n" % asan_options)
    f.write("- UBSAN_OPTIONS: `%s`\n" % ubsan_options)
    f.write("- Retained sanitizer artifacts: `%s`\n" % rel(work_root))
    f.write("- ICC trace: `%s`\n\n" % rel(trace_file))

    f.write("## Teeth Check\n\n")
    f.write("- Status: `%s`\n" % self_status)
    f.write("- Summary: %s\n" % self_summary)
    f.write("\n")

    f.write("## Coverage\n\n")
    f.write("- Status: `%s`\n" % coverage_status)
    f.write("- Summary: %s\n" % coverage_summary)
    if run_counts:
        f.write("- Records by axis/phase/rc:\n")
        for (axis, phase, rc), n in sorted(run_counts.items()):
            f.write("  - `%s/%s rc=%s`: `%d`\n" % (axis, phase, rc, n))
    f.write("\n")

    f.write("## Findings\n\n")
    if not groups:
        f.write("No ASan/UBSan reports were observed in corpus logs.\n\n")
    else:
        f.write("| Task | Kind | Type | Hits | Top frame | First repro |\n")
        f.write("|---|---|---|---:|---|---|\n")
        for group in groups.values():
            sample = group["samples"][0] if group["samples"] else {}
            f.write(
                "| `%s` | `%s` | `%s` | %d | `%s` | `%s` |\n"
                % (
                    group["task_id"],
                    md(group["kind"]),
                    md(group["subtype"]),
                    group["hits"],
                    md(group["top_frame"]),
                    md(sample.get("source", "")),
                )
            )
        f.write("\n")

        for group in groups.values():
            sample = group["samples"][0] if group["samples"] else {}
            f.write("### %s\n\n" % group["task_id"])
            f.write("- Kind: `%s`\n" % group["kind"])
            f.write("- Type: `%s`\n" % group["subtype"])
            f.write("- Hits: `%d`\n" % group["hits"])
            f.write("- Top frame: `%s`\n" % group["top_frame"])
            if group["summary"]:
                f.write("- Summary: `%s`\n" % group["summary"])
            f.write("- First repro: `%s` via `%s/%s` rc `%s`\n" % (
                sample.get("source", ""),
                sample.get("axis", ""),
                sample.get("phase", ""),
                sample.get("rc", ""),
            ))
            f.write("- Repro command: `ASAN_OPTIONS='%s' UBSAN_OPTIONS='%s' %s`\n" % (
                asan_options,
                ubsan_options,
                command_for(sample),
            ))
            f.write("- First log: `%s`\n" % sample.get("log", ""))
            if group["samples"]:
                f.write("- Additional samples:\n")
                for s in group["samples"][1:]:
                    f.write("  - `%s` via `%s/%s`, log `%s`\n" % (
                        s.get("source", ""),
                        s.get("axis", ""),
                        s.get("phase", ""),
                        s.get("log", ""),
                    ))
            if group["snippets"]:
                f.write("\n```text\n%s\n```\n" % group["snippets"][0])
            f.write("\n")

if self_status != "PASS" or coverage_status != "PASS" or groups:
    sys.exit(1)
PY
}

echo "Sanitizer fuzz harness"
echo "  repo: $REPO_ROOT"
echo "  build: $BUILD_DIR_ABS"
echo "  report: $REPORT_ABS"
echo "  logs: $WORK_ROOT_ABS"
echo "  artifact cap: $(format_kb "$MAX_ARTIFACTS_KB")"
echo "  per-file cap: $(format_kb "$SWEEP_FILE_CAP_KB") (sweep children)"
if [ "$LIMIT" -gt 0 ]; then
    echo "  sweep mode: debug limit ($LIMIT files)"
elif [ "$FULL_SWEEP" = "1" ]; then
    echo "  sweep mode: full corpus"
else
    echo "  sweep mode: bounded gate ($GATE_LIMIT files; --full for entire corpus)"
fi
echo "  retained crash cap: $MAX_RETAINED_CRASHES"
echo

check_disk_budget "startup"
run_self_test
check_disk_budget "self-test"

if [ "$SELF_TEST_ONLY" -eq 1 ]; then
    FINISHED_AT="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    write_report "$FINISHED_AT" 0
    report_rc=$?
    check_disk_budget "report"
    cleanup_sanitizer_fuzz
    final_artifact_kb="$(artifact_usage_kb)"
    [ "$final_artifact_kb" -gt "$PEAK_ARTIFACT_KB" ] && PEAK_ARTIFACT_KB="$final_artifact_kb"
    echo "Artifact usage: final=$(format_kb "$final_artifact_kb") peak=$(format_kb "$PEAK_ARTIFACT_KB") cap=$(format_kb "$MAX_ARTIFACTS_KB")"
    exit "$report_rc"
fi

if [ "$SKIP_BUILD" -eq 0 ]; then
    CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}" scripts/build-sanitizer.sh asan+ubsan
fi

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_sanitizer_fuzz.sh: missing executable: $ESHKOL_RUN" >&2
    exit 2
fi

collect_corpus
check_disk_budget "corpus collection"
TOTAL_SOURCES=$(wc -l < "$CORPUS_LIST" | tr -d ' ')
echo "Corpus: $TOTAL_SOURCES unique .esk files"

# Resolve the effective sweep bound.
#   --limit N        debug override, always wins
#   --full / env     run the entire corpus
#   (default)        bounded gate sweep of GATE_LIMIT files
case "$GATE_LIMIT" in
    ''|*[!0-9]*)
        echo "run_sanitizer_fuzz.sh: --gate-limit must be a non-negative integer: $GATE_LIMIT" >&2
        exit 2
        ;;
esac
EFFECTIVE_LIMIT="$LIMIT"
if [ "$LIMIT" -gt 0 ]; then
    echo "Debug limit: first $LIMIT files"
elif [ "$FULL_SWEEP" = "1" ]; then
    echo "Full sweep: running the entire corpus (CI/mesh mode)"
else
    EFFECTIVE_LIMIT="$GATE_LIMIT"
    echo "Bounded gate sweep: first $EFFECTIVE_LIMIT files (use --full for the entire corpus)"
fi

# Kernel-enforced per-file size cap for the sweep and its children only. Set
# AFTER the (large) build so linking eshkol-run / libeshkol-static.a is not
# capped; bounds runaway per-case stdout/stderr and any stray core/temp file.
ulimit -f "$SWEEP_FILE_CAP_KB" 2>/dev/null || true

echo "Axes: -r, AOT -O0, AOT -O2"
echo

idx=0
while IFS= read -r src; do
    idx=$((idx + 1))
    if [ "$EFFECTIVE_LIMIT" -gt 0 ] && [ "$idx" -gt "$EFFECTIVE_LIMIT" ]; then
        break
    fi
    check_disk_budget "before corpus file $idx (${src#$REPO_ROOT/})"
    if [ $((idx % 25)) -eq 1 ]; then
        printf '  [%d/%d] %s\n' "$idx" "$TOTAL_SOURCES" "${src#$REPO_ROOT/}"
    fi
    run_r_axis "$src" "$idx"
    check_disk_budget "after corpus file $idx -r"
    run_aot_axis "$src" "$idx" "aot-o0" "-O0"
    check_disk_budget "after corpus file $idx AOT -O0"
    run_aot_axis "$src" "$idx" "aot-o2" "-O2"
    check_disk_budget "after corpus file $idx AOT -O2"
done < "$CORPUS_LIST"

if [ "$EFFECTIVE_LIMIT" -gt 0 ] && [ "$EFFECTIVE_LIMIT" -lt "$TOTAL_SOURCES" ]; then
    TOTAL_REPORTED="$EFFECTIVE_LIMIT"
else
    TOTAL_REPORTED="$TOTAL_SOURCES"
fi

FINISHED_AT="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
write_report "$FINISHED_AT" "$TOTAL_REPORTED"
report_rc=$?
check_disk_budget "report"
cleanup_sanitizer_fuzz
final_artifact_kb="$(artifact_usage_kb)"
[ "$final_artifact_kb" -gt "$PEAK_ARTIFACT_KB" ] && PEAK_ARTIFACT_KB="$final_artifact_kb"
echo "Artifact usage: final=$(format_kb "$final_artifact_kb") peak=$(format_kb "$PEAK_ARTIFACT_KB") cap=$(format_kb "$MAX_ARTIFACTS_KB")"
exit "$report_rc"
