#!/usr/bin/env bash
# Machine-checked theorems behind the nLab conformance claims.
#
# Type-checks every proof under tests/formal/agda (Agda, --safe) and
# tests/formal/coq (Rocq), see docs/reference/math/formal-crosschecks.md.
#
# Result per prover: PASS when every file checks, FAIL when one does not,
# SKIP when the prover or its libraries are not installed on this host.
# A SKIP is never reported as PASS. Exit status: 0 when nothing FAILED and at
# least one prover ran; 1 on any FAIL; 2 when every prover was SKIPPED (or
# with --require when any prover was SKIPPED).
#
# Usage: scripts/run_formal_agda_gate.sh [--require] [--trace-dir DIR]
#   --require      treat SKIP as a gate failure (exit 2)
#   --trace-dir    write ICC runtime_event traces (default scripts/icc_traces)
#
# Environment: AGDA (default agda), AGDA_LIBRARY_FILE (default: the
# example-libraries file shipped next to Agda's data directory), COQC
# (default rocq c, then coqc).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGDA_DIR="$ROOT/tests/formal/agda"
COQ_DIR="$ROOT/tests/formal/coq"
TRACE_DIR="$ROOT/scripts/icc_traces"
REQUIRE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --require) REQUIRE=1 ;;
    --trace-dir) shift; TRACE_DIR="$1" ;;
    *) echo "usage: $0 [--require] [--trace-dir DIR]" >&2; exit 64 ;;
  esac
  shift
done
NICE="nice -n 15"

pass=0; fail=0; skip=0
results=()

record() { # kind name status detail
  results+=("$1|$2|$3|$4")
  printf '%s: [%s] %s %s\n' "$3" "$1" "$2" "$4"
}

# A proof file must not weaken the check: no postulates, no holes, no pragmas
# that switch off termination or positivity checking.
lint() {
  local f="$1"
  if grep -qE 'TERMINATING|NO_POSITIVITY_CHECK|NO_UNIVERSE_CHECK|--type-in-type|--allow-unsolved-metas' "$f" ||
     sed 's/--.*$//' "$f" | grep -qE '(^|[[:space:]])postulate([[:space:]]|$)|\{!|(^|[[:space:](])\?([[:space:])]|$)'; then
    return 1
  fi
  return 0
}
lint_coq() {
  local f="$1"
  if grep -nE '\b(Admitted|admit|Axiom|Parameter|Conjecture)\b' "$f" >/dev/null; then
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------- Agda
AGDA="${AGDA:-agda}"
if ! command -v "$AGDA" >/dev/null 2>&1; then
  record agda all SKIP "(agda not installed)"; skip=$((skip+1))
else
  LIBF="${AGDA_LIBRARY_FILE:-$("$AGDA" --print-agda-data-dir 2>/dev/null)/../example-libraries}"
  if [ ! -f "$LIBF" ] || ! grep -q 'cubical' "$LIBF" || ! grep -q 'standard-library' "$LIBF"; then
    record agda all SKIP "(cubical/standard-library not registered; set AGDA_LIBRARY_FILE)"; skip=$((skip+1))
  else
    ver="$("$AGDA" --version 2>/dev/null | head -1)"
    for f in "$AGDA_DIR"/*.agda; do
      name="$(basename "$f")"
      if ! head -1 "$f" | grep -q -- '--safe'; then
        record agda "$name" FAIL "(missing --safe)"; fail=$((fail+1)); continue
      fi
      if ! lint "$f"; then
        record agda "$name" FAIL "(postulate, hole or unchecked pragma)"; fail=$((fail+1)); continue
      fi
      out="$(cd "$AGDA_DIR" && $NICE "$AGDA" --library-file="$LIBF" "$name" 2>&1)"; rc=$?
      if [ $rc -eq 0 ] && ! printf '%s' "$out" | grep -qi 'unsolved'; then
        record agda "$name" PASS "($ver)"; pass=$((pass+1))
      else
        record agda "$name" FAIL "(exit $rc)"; fail=$((fail+1))
        printf '%s\n' "$out" | grep -v '^ *Checking\|^ *Loading\|^ *Finished' | head -20
      fi
    done
  fi
fi

# ---------------------------------------------------------------- Rocq
if compgen -G "$COQ_DIR/*.v" >/dev/null; then
  if command -v rocq >/dev/null 2>&1; then COQ=(rocq c); COQV="$(rocq --version 2>/dev/null | head -1)"
  elif command -v coqc >/dev/null 2>&1; then COQ=(coqc); COQV="$(coqc --version 2>/dev/null | head -1)"
  else COQ=(); fi
  if [ -n "${COQC:-}" ]; then COQ=("$COQC"); COQV="$("$COQC" --version 2>/dev/null | head -1)"; fi
  if [ ${#COQ[@]} -eq 0 ]; then
    record rocq all SKIP "(rocq/coqc not installed)"; skip=$((skip+1))
  else
    for f in "$COQ_DIR"/*.v; do
      name="$(basename "$f")"
      if ! lint_coq "$f"; then
        record rocq "$name" FAIL "(Admitted/Axiom present)"; fail=$((fail+1)); continue
      fi
      out="$(cd "$COQ_DIR" && $NICE "${COQ[@]}" -no-glob "$name" 2>&1)"; rc=$?
      if [ $rc -eq 0 ] && printf '%s' "$out" | grep -q 'Axioms:'; then
        record rocq "$name" FAIL "(a theorem depends on axioms)"; fail=$((fail+1))
        printf '%s\n' "$out" | grep -A5 'Axioms:' | head -20
      elif [ $rc -eq 0 ]; then
        record rocq "$name" PASS "($COQV)"; pass=$((pass+1))
      else
        record rocq "$name" FAIL "(exit $rc)"; fail=$((fail+1))
        printf '%s\n' "$out" | head -20
      fi
    done
  fi
fi

# certificate blocks must match their generators
for gen in "$COQ_DIR"/gen_*.py; do
  [ -f "$gen" ] || continue
  name="$(basename "$gen")"
  if ! command -v python3 >/dev/null 2>&1; then
    record certificates "$name" SKIP "(python3 not installed)"; skip=$((skip+1))
  elif $NICE python3 -I -B "$gen" --check >/dev/null 2>&1; then
    record certificates "$name" PASS "(generated block current)"; pass=$((pass+1))
  else
    record certificates "$name" FAIL "(generated block differs; rerun $name)"; fail=$((fail+1))
  fi
done

echo "formal gate: $pass passed, $fail failed, $skip skipped"

# ---------------------------------------------------------------- ICC trace
mkdir -p "$TRACE_DIR" 2>/dev/null && {
  tf="$TRACE_DIR/formal_agda_gate.jsonl"
  : > "$tf"
  for r in "${results[@]}"; do
    IFS='|' read -r kind name status detail <<<"$r"
    printf '{"kind":"runtime_event","event":"formal_proof_check","prover":"%s","file":"%s","status":"%s","producer":"scripts/run_formal_agda_gate.sh"}\n' \
      "$kind" "$name" "$status" >> "$tf"
  done
  printf '{"kind":"runtime_event","event":"formal_gate_summary","passed":%d,"failed":%d,"skipped":%d,"producer":"scripts/run_formal_agda_gate.sh"}\n' \
    "$pass" "$fail" "$skip" >> "$tf"
}

if [ $fail -gt 0 ]; then exit 1; fi
if [ $pass -eq 0 ]; then echo "SKIP: no prover available; nothing was checked"; exit 2; fi
if [ $REQUIRE -eq 1 ] && [ $skip -gt 0 ]; then exit 2; fi
exit 0
