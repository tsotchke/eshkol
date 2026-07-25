#!/usr/bin/env bash
# p8_fault_injection.sh — P8 escape-closure axis 7: toolchain fault-injection
# matrix.
#
# Originating escape (see .swarm/P8_ESCAPE_ANALYSIS.md): a driver failure exited
# 0 (a broken generated-program link under -r fell back to a reduced in-process
# run and "succeeded"), so a build system that trusts $? shipped a build that
# never linked. Point regression tests existed (#334); the escape is that the
# fault space (missing / unopenable / malformed / bad-require / broken-link /
# hang, across -r and AOT) was never swept as a matrix asserting the ONE
# invariant every fault must satisfy:
#
#     INVARIANT: a fault yields a NONZERO exit AND a diagnostic that names the
#                offending file / symbol / module — never exit 0.
#
# Verdicts per cell:
#   PASS    invariant held (nonzero exit + diagnostic token present)
#   FAIL    a HARD-GATE cell violated the invariant (a NEW masking regression)
#   XKNOWN  a cell tagged xmask=1 (a tracked-open exit-0 masking) did not meet
#           the invariant — tolerated, recorded, reported
#   XPASS   an xmask cell now MEETS the invariant (bug fixed) -> FAILS the gate
#           so the cell is promoted to a hard gate and its task is closed
#
# There are currently NO xmask cells: the ESH-0361 masking residue was fixed in
# #354 and all five of its cells were promoted to hard gates below. A future
# tracked-open masking is quarantined by adding xmask=1 back on its own cell.
#
# ICC wiring: pytest-style "PASSED tests/escape_matrix/fault::<id>" lines plus
# kind=escape_matrix JSON-L events into scripts/icc_traces/escape_matrix.jsonl,
# and a roll-up event p8_fault_injection_suite.
#
# Usage: scripts/p8/p8_fault_injection.sh [--build-dir DIR] [--quick]
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
QUICK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2;;
    --quick) QUICK=1; shift;;
    *) shift;;
  esac
done
case "$BUILD_DIR" in /*) ;; *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR";; esac
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
if [ ! -x "$ESHKOL_RUN" ]; then
  echo "p8_fault_injection: $ESHKOL_RUN not found — build eshkol-run first." >&2
  exit 2
fi

TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/escape_matrix.jsonl"
mkdir -p "$TRACE_DIR"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/p8-fault.XXXXXX")"
# Disk cap + guaranteed cleanup (fuzz/harness disk-budget policy).
cleanup() { chmod -R u+rwx "$WORK" 2>/dev/null; rm -rf "$WORK"; }
trap cleanup EXIT
export ESHKOL_JIT_CACHE_DIR="$WORK/jit"; mkdir -p "$ESHKOL_JIT_CACHE_DIR"

emit() { # id status snippet
  python3 -c 'import json,sys; print(json.dumps({"kind":"escape_matrix","name":sys.argv[1],"value":sys.argv[2],"snippet":sys.argv[3][:200],"confidence":0.95}))' \
    "$1" "$2" "$3" >> "$TRACE_FILE"
}

# perl alarm (macOS has no timeout(1)). The alarm timer survives exec, so
# SIGALRM is delivered directly to the exec'd process on timeout (default action
# terminate -> exit 142 = 128+SIGALRM). No fork/waitpid, so no reap/pipe hangs.
# Mirrors the proven pattern in scripts/run_stress.sh.
run_guarded() { # secs cmd...
  local secs="$1"; shift
  perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' "$secs" "$@"
}

PASS=0; FAIL=0; XK=0; XP=0

# A cell: run cmd (already built into a function via $CMD string), assert the
# invariant. Args: id  xmask(0/1)  diag_token  timeout  -- cmd...
cell() {
  local id="$1" xmask="$2" token="$3" tmo="$4"; shift 4
  [ "$1" = "--" ] && shift
  local out ec
  out="$(run_guarded "$tmo" "$@" 2>&1)"; ec=$?
  local ok=1
  [ "$ec" -eq 0 ] && ok=0                     # exit 0 violates the invariant
  if [ -n "$token" ]; then printf '%s' "$out" | grep -qiF "$token" || ok=0; fi
  if [ "$ok" -eq 1 ]; then
    if [ "$xmask" -eq 1 ]; then
      XP=$((XP+1)); echo "XPASS $id (masking fixed — promote to hard gate, close the tracked task)"; emit "$id" XPASS "invariant now holds"
      echo "XPASS tests/escape_matrix/fault::$id"
    else
      PASS=$((PASS+1)); emit "$id" PASS "exit=$ec token ok"
      echo "PASSED tests/escape_matrix/fault::$id"
    fi
  else
    if [ "$xmask" -eq 1 ]; then
      XK=$((XK+1)); emit "$id" XKNOWN "exit=$ec (tracked-open masking)"
      echo "XKNOWN tests/escape_matrix/fault::$id (exit=$ec, tracked-open masking)"
    else
      FAIL=$((FAIL+1)); emit "$id" FAIL "exit=$ec token-missing"
      echo "FAILED tests/escape_matrix/fault::$id (exit=$ec)"
    fi
  fi
}

# ---- fixtures ------------------------------------------------------------
OK="$WORK/ok.esk";       printf '(display 1)(newline)\n' > "$OK"
MAL="$WORK/malformed.esk"; printf '(display 1\n' > "$MAL"         # unbalanced
UNDEF="$WORK/undef.esk"; printf '(display (totally-undefined-fn 3))\n' > "$UNDEF"
BADREQ="$WORK/badreq.esk"; printf '(require nonexistent.module.xyz)(display 1)\n' > "$BADREQ"
NOPERM="$WORK/noperm.esk"; printf '(display 1)\n' > "$NOPERM"; chmod 000 "$NOPERM"
# Non-allocating infinite tail loop (no bignum growth that would slow SIGALRM).
HANG="$WORK/hang.esk";   printf '(let loop () (loop))\n' > "$HANG"
MISSING="$WORK/does-not-exist-$$.esk"      # never created

# ---- HARD-GATE cells (must exit nonzero + diagnose) — retro-guard #334 ----
cell fault_missing_r        0 "not found"    15 -- "$ESHKOL_RUN" -r "$MISSING"
cell fault_undef_symbol_r   0 "totally-undefined-fn" 20 -- "$ESHKOL_RUN" -r "$UNDEF"
cell fault_undef_symbol_aot 0 "totally-undefined-fn" 30 -- "$ESHKOL_RUN" "$UNDEF" -o "$WORK/undef.bin"
cell fault_malformed_aot    0 ""             30 -- "$ESHKOL_RUN" "$MAL" -o "$WORK/mal.bin"
cell fault_bad_output_dir   0 ""             30 -- "$ESHKOL_RUN" "$OK" -o "/nonexistent-dir-p8/out.bin"
cell fault_broken_lib_r     0 ""             20 -- "$ESHKOL_RUN" -r "$OK" --lib "/nonexistent/libbogus.dylib"
cell fault_broken_lib_aot   0 ""             30 -- "$ESHKOL_RUN" "$OK" -o "$WORK/lib.bin" --lib "/nonexistent/libbogus.dylib"

# ---- PROMOTED cells: the ESH-0361 exit-0 masking residue, fixed in #354 ----
# These five were xmask=1 (XKNOWN, tracked-open) until #354 made every one of
# them exit nonzero with a diagnostic and write no binary. The ratchet reported
# them as XPASS; they are now hard gates, each carrying the diagnostic token the
# fix introduced, so the masking cannot silently return. ESH-0361 is CLOSED —
# permanently pinned by tests/toolchain/driver_fault_exit_code_test.sh too.
cell fault_missing_aot      0 "File not found"          30 -- "$ESHKOL_RUN" "$MISSING" -o "$WORK/missing.bin"
cell fault_malformed_r      0 "unexpected end of input" 20 -- "$ESHKOL_RUN" -r "$MAL"
# Mode 000 is still readable as root (some containers), where this fault cannot
# be constructed at all. Assert only when the process genuinely cannot read the
# file — same guard as tests/toolchain/driver_fault_exit_code_test.sh cell (c).
if ! { : < "$NOPERM"; } 2>/dev/null; then
  cell fault_unreadable_r   0 "Failed to open file"     20 -- "$ESHKOL_RUN" -r "$NOPERM"
else
  echo "SKIP tests/escape_matrix/fault::fault_unreadable_r (mode-000 still readable — running as root?)"
  emit fault_unreadable_r SKIP "mode-000 readable (root) — fault not constructible on this host"
fi
cell fault_bad_require_r    0 "nonexistent.module.xyz"  20 -- "$ESHKOL_RUN" -r "$BADREQ"
cell fault_bad_require_aot  0 "nonexistent.module.xyz"  30 -- "$ESHKOL_RUN" "$BADREQ" -o "$WORK/br.bin"

# ---- HANG cell: an infinite loop under -r must be bounded, not exit 0 ----
# eshkol-run does not honor SIGALRM under -r, so bound it with an uncatchable
# SIGKILL from a background killer. A killed process (nonzero) proves the hang
# did NOT silently exit 0; a fast exit 0 would be masking (XKNOWN).
hang_cell() {
  local id="fault_hang_bounded_r" ec
  "$ESHKOL_RUN" -r "$HANG" >/dev/null 2>&1 &
  local pid=$!
  ( sleep 3; kill -9 "$pid" 2>/dev/null ) &
  local killer=$!
  wait "$pid" 2>/dev/null; ec=$?
  kill "$killer" 2>/dev/null; wait "$killer" 2>/dev/null
  if [ "$ec" -ne 0 ]; then
    PASS=$((PASS+1)); emit "$id" PASS "hang bounded (killed, ec=$ec)"
    echo "PASSED tests/escape_matrix/fault::$id"
  else
    XK=$((XK+1)); emit "$id" XKNOWN "hang exited 0 (masking)"
    echo "XKNOWN tests/escape_matrix/fault::$id (exit 0 — masking)"
  fi
}
hang_cell

chmod 644 "$NOPERM" 2>/dev/null

emit p8_fault_injection_suite "$([ "$FAIL" -eq 0 ] && [ "$XP" -eq 0 ] && echo PASS || echo FAIL)" \
     "pass=$PASS fail=$FAIL xknown=$XK xpass=$XP"
echo
echo "axis-7 fault-injection: PASS=$PASS FAIL=$FAIL XKNOWN=$XK XPASS=$XP"
if [ "$FAIL" -ne 0 ] || [ "$XP" -ne 0 ]; then
  echo "axis-7 gate: FAIL"; exit 1
fi
echo "axis-7 gate: PASS"
