#!/usr/bin/env bash
# run_sdnc_oracle.sh — emit ICC-format runtime evidence for the SDNC paper
# reproducibility claim ("Self-Differentiating Neural Computer", tsotchke 2026).
#
# Three things land in scripts/icc_traces/sdnc_oracle.jsonl:
#
#   1. Per-program halt evidence — already produced by the reference C VM
#      via the eshkol_vm_halt event recognizer (runtime_evidence.py
#      registers each `flags.halt:true` step as a halt event keyed by
#      program name). 71 programs in the suite → 71 halt events.
#
#   2. Trace-agreement summary — one
#         {"kind":"sdnc_paper_suite","name":"trace_agreement",
#          "value":"PASS"|"FAIL", "snippet":"<n>/<total> agree"}
#      record per oracle invocation, derived from
#      scripts/paper/compare_traces.py's stdout.
#
#   3. Output-agreement summary — same shape but `name:output_agreement`.
#      The paper's headline claim is bit-identical OUTPUT agreement on
#      71/71 programs; trace agreement is the stronger fieldwise check.
#
# An oracle in .icc/completion-oracles.yaml then has a one-line criterion:
#   - runtime_event:
#       event_kinds: [sdnc_paper_suite]
#       event_names: [output_agreement]
#       event_values: ["PASS"]
#
# Usage:
#   bash scripts/run_sdnc_oracle.sh           # default (full suite)
#   bash scripts/run_sdnc_oracle.sh --quick   # skip heavy comparisons
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
OUT="$TRACE_DIR/sdnc_oracle.jsonl"
mkdir -p "$TRACE_DIR"
: "${OUT:?}"
: > "$OUT"

# Run the paper suite. It writes vm-traces.jsonl + transformer-traces.jsonl
# + a structured comparison-report.json under artifacts/paper/outputs/.
SUITE_LOG_TMP=$(mktemp "${TMPDIR:-/tmp}/eshkol-sdnc-suite.XXXXXX") || exit 1
cleanup() {
    [ -n "${SUITE_LOG_TMP:-}" ] && [ -f "$SUITE_LOG_TMP" ] && rm -f -- "$SUITE_LOG_TMP"
}
trap cleanup EXIT

if ! bash "$REPO_ROOT/scripts/paper/run_paper_suite.sh" "$@" >"$SUITE_LOG_TMP" 2>&1; then
    {
        printf '{"kind":"sdnc_paper_suite","name":"output_agreement","value":"FAIL","snippet":"paper suite exited non-zero","confidence":0.99}\n'
        printf '{"kind":"sdnc_paper_suite","name":"trace_agreement","value":"FAIL","snippet":"paper suite exited non-zero","confidence":0.99}\n'
        : "${OUT:?}"
    } >> "$OUT"
    cat "$SUITE_LOG_TMP" >&2
    echo "scripts/run_sdnc_oracle.sh: paper suite failed; FAIL events written" >&2
    exit 1
fi

# Parse the structured report. Stable schema:
#   {output_agreeing_programs: N, fully_agreeing_programs: N,
#    total_programs: N, status: "...", per_program: {...}}
REPORT="$REPO_ROOT/artifacts/paper/outputs/comparison-report.json"
if [ ! -f "$REPORT" ]; then
    {
        printf '{"kind":"sdnc_paper_suite","name":"output_agreement","value":"FAIL","snippet":"comparison-report.json not found","confidence":0.99}\n'
        printf '{"kind":"sdnc_paper_suite","name":"trace_agreement","value":"FAIL","snippet":"comparison-report.json not found","confidence":0.99}\n'
        : "${OUT:?}"
    } >> "$OUT"
    echo "scripts/run_sdnc_oracle.sh: missing $REPORT; FAIL events written" >&2
    exit 1
fi

python3 - "$REPORT" "$OUT" <<'PY'
import json, sys
report_path, out_path = sys.argv[1], sys.argv[2]
with open(report_path) as f:
    rep = json.load(f)
total   = rep.get("total_programs", 0)
out_pgm = rep.get("output_agreeing_programs", 0)
full    = rep.get("fully_agreeing_programs", 0)

def emit(name, agree, total, snippet_extra=""):
    verdict = "PASS" if (total > 0 and agree == total) else "FAIL"
    snippet = f"{agree}/{total} programs agree{snippet_extra}"
    rec = {
        "kind": "sdnc_paper_suite", "name": name,
        "value": verdict, "snippet": snippet, "confidence": 1.0,
    }
    return json.dumps(rec)

with open(out_path, "a") as f:
    f.write(emit("output_agreement", out_pgm, total, " on PRINT outputs (paper §4.4)") + "\n")
    f.write(emit("trace_agreement",  full,    total, " on full per-step state") + "\n")
    # Also surface a per-program halt summary so the oracle can spot
    # specific programs that regressed (the per_program dict has
    # {program: {agrees, fully_agrees, ...}} entries).
    pp = rep.get("per_program", {}) or {}
    for prog, info in pp.items():
        if not isinstance(info, dict):
            continue
        # Per-program PASS = every step agrees AND every step that produced
        # a PRINT output had its output match. The comparison-report.json
        # schema is {steps, agreeing_steps, disagreeing_steps,
        # output_steps, output_agreeing_steps, vm_outputs, tf_outputs}.
        steps             = info.get("steps", 0)
        agreeing_steps    = info.get("agreeing_steps", 0)
        output_steps      = info.get("output_steps", 0)
        output_agreeing   = info.get("output_agreeing_steps", 0)
        full_agreed = (steps > 0 and agreeing_steps == steps)
        out_agreed  = (output_steps == 0 or output_agreeing == output_steps)
        verdict = "PASS" if (full_agreed and out_agreed) else "FAIL"
        rec = {
            "kind": "sdnc_program",
            "name": prog,
            "value": verdict,
            "snippet": (
                f"steps={agreeing_steps}/{steps} "
                f"outputs={output_agreeing}/{output_steps}"
            ),
            "confidence": 0.99,
        }
        f.write(json.dumps(rec) + "\n")
PY

output_agree=$(python3 -c "import json; r=json.load(open('$REPORT')); print(f\"{r.get('output_agreeing_programs',0)}/{r.get('total_programs',0)}\")")
trace_agree=$(python3 -c "import json; r=json.load(open('$REPORT')); print(f\"{r.get('fully_agreeing_programs',0)}/{r.get('total_programs',0)}\")")

# Copy the per-step VM trace into the ICC trace dir so the
# eshkol_vm_step / eshkol_vm_halt events get ingested in the same
# `--trace-dir scripts/icc_traces` invocation.
if [ -f "$REPO_ROOT/artifacts/paper/outputs/vm-traces.jsonl" ]; then
    cp "$REPO_ROOT/artifacts/paper/outputs/vm-traces.jsonl" "$TRACE_DIR/sdnc_vm-traces.jsonl"
fi

echo "SDNC oracle trace written: $OUT (output=$output_agree trace=$trace_agree)"
