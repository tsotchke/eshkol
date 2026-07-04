#!/usr/bin/env bash
# run_sanitizer_fuzz.sh - ASan/UBSan oracle over differential and test corpora.
#
# The differential harness checks semantic agreement. This harness is additive:
# it ignores ordinary compile/runtime failures unless a sanitizer emits a report.

set -u
cd "$(dirname "$0")/.."

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

usage() {
    cat <<'EOF'
Usage: scripts/run_sanitizer_fuzz.sh [options]

Builds or reuses build-asan-ubsan, runs every tests/**/*.esk input plus a
deterministic generated differential fuzz corpus through -r and AOT, and writes
SANITIZER_FUZZ_REPORT.md with ASan/UBSan reports deduped by top frame.

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
  --limit N                Run only the first N corpus files (debugging only).
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
        --limit) LIMIT="$2"; shift 2 ;;
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
GENERATED_DIR="$WORK_ROOT_ABS/generated-differential"
RUNS_TSV="$WORK_ROOT_ABS/runs.tsv"
SELF_TSV="$WORK_ROOT_ABS/selftest.tsv"
CORPUS_RAW="$WORK_ROOT_ABS/corpus.raw"
CORPUS_LIST="$WORK_ROOT_ABS/corpus.txt"
ESHKOL_RUN="$BUILD_DIR_ABS/eshkol-run"
TASK_DIR="$REPO_ROOT/.swarm/tasks"
STARTED_AT="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

mkdir -p "$LOG_DIR" "$BIN_DIR" "$GENERATED_DIR" "$(dirname "$TRACE_FILE_ABS")" "$TASK_DIR"
: > "$RUNS_TSV"
: > "$SELF_TSV"
: > "$TRACE_FILE_ABS"

if [ "$(uname -s)" = "Darwin" ]; then
    DEFAULT_ASAN_OPTIONS="detect_leaks=0:halt_on_error=1:abort_on_error=1:print_stacktrace=1:strict_string_checks=1:detect_stack_use_after_return=0:symbolize=1"
else
    DEFAULT_ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:abort_on_error=1:print_stacktrace=1:strict_string_checks=1:detect_stack_use_after_return=0:symbolize=1"
fi
DEFAULT_UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1:abort_on_error=1"
export ASAN_OPTIONS="${ASAN_OPTIONS:-$DEFAULT_ASAN_OPTIONS}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-$DEFAULT_UBSAN_OPTIONS}"

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

    "$cc_bin" -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
        -fno-omit-frame-pointer "$src" -o "$bin" >"$cout" 2>"$cerr"
    local crc=$?
    if [ "$crc" -ne 0 ] || [ ! -x "$bin" ]; then
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
    if grep -E 'ERROR: AddressSanitizer:|runtime error:|SUMMARY: UndefinedBehaviorSanitizer:' "$out" "$err" >/dev/null 2>&1; then
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
            --seed "$SEED" --count "$COUNT" --emit-only "$GENERATED_DIR"
        find "$GENERATED_DIR" -type f -name '*.esk' -print >> "$CORPUS_RAW"
    fi
    find "$REPO_ROOT/tests" -type f -name '*.esk' -print >> "$CORPUS_RAW"
    sort -u "$CORPUS_RAW" > "$CORPUS_LIST"
}

run_r_axis() {
    local src="$1" idx="$2"
    local h out err rc
    h="$(path_hash "$src")"
    out="$LOG_DIR/${idx}_r_${h}.out"
    err="$LOG_DIR/${idx}_r_${h}.err"
    run_guarded "$TIMEOUT_RUN" "$ESHKOL_RUN" -r "$src" >"$out" 2>"$err"
    rc=$?
    append_run "$src" "r" "run" "$rc" "$out" "$err"
}

run_aot_axis() {
    local src="$1" idx="$2" axis="$3" opt="$4"
    local h bin cout cerr out err crc rrc
    h="$(path_hash "$src")"
    bin="$BIN_DIR/${idx}_${axis}_${h}.bin"
    cout="$LOG_DIR/${idx}_${axis}_${h}.compile.out"
    cerr="$LOG_DIR/${idx}_${axis}_${h}.compile.err"
    out="$LOG_DIR/${idx}_${axis}_${h}.run.out"
    err="$LOG_DIR/${idx}_${axis}_${h}.run.err"

    run_guarded "$TIMEOUT_RUN" "$ESHKOL_RUN" "$opt" "$src" -o "$bin" >"$cout" 2>"$cerr"
    crc=$?
    append_run "$src" "$axis" "compile" "$crc" "$cout" "$cerr"
    if [ "$crc" -ne 0 ] || [ ! -x "$bin" ]; then
        return
    fi

    run_guarded "$TIMEOUT_RUN" "$bin" >"$out" 2>"$err"
    rrc=$?
    append_run "$src" "$axis" "run" "$rrc" "$out" "$err"
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
    f.write("- Raw logs: `%s`\n" % rel(work_root))
    f.write("- ICC trace: `%s`\n\n" % rel(trace_file))

    f.write("## Teeth Check\n\n")
    f.write("- Status: `%s`\n" % self_status)
    f.write("- Summary: %s\n" % self_summary)
    if self_data.get("source"):
        f.write("- Synthetic repro: `%s`\n" % rel(self_data["source"]))
    if self_data.get("stderr"):
        f.write("- Synthetic stderr: `%s`\n" % rel(self_data["stderr"]))
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
echo

run_self_test

if [ "$SELF_TEST_ONLY" -eq 1 ]; then
    FINISHED_AT="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    write_report "$FINISHED_AT" 0
    exit $?
fi

if [ "$SKIP_BUILD" -eq 0 ]; then
    CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}" scripts/build-sanitizer.sh asan+ubsan
fi

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_sanitizer_fuzz.sh: missing executable: $ESHKOL_RUN" >&2
    exit 2
fi

collect_corpus
TOTAL_SOURCES=$(wc -l < "$CORPUS_LIST" | tr -d ' ')
echo "Corpus: $TOTAL_SOURCES unique .esk files"
if [ "$LIMIT" -gt 0 ]; then
    echo "Debug limit: first $LIMIT files"
fi
echo "Axes: -r, AOT -O0, AOT -O2"
echo

idx=0
while IFS= read -r src; do
    idx=$((idx + 1))
    if [ "$LIMIT" -gt 0 ] && [ "$idx" -gt "$LIMIT" ]; then
        break
    fi
    if [ $((idx % 25)) -eq 1 ]; then
        printf '  [%d/%d] %s\n' "$idx" "$TOTAL_SOURCES" "${src#$REPO_ROOT/}"
    fi
    run_r_axis "$src" "$idx"
    run_aot_axis "$src" "$idx" "aot-o0" "-O0"
    run_aot_axis "$src" "$idx" "aot-o2" "-O2"
done < "$CORPUS_LIST"

if [ "$LIMIT" -gt 0 ] && [ "$LIMIT" -lt "$TOTAL_SOURCES" ]; then
    TOTAL_REPORTED="$LIMIT"
else
    TOTAL_REPORTED="$TOTAL_SOURCES"
fi

FINISHED_AT="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
write_report "$FINISHED_AT" "$TOTAL_REPORTED"
exit $?
