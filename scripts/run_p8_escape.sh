#!/usr/bin/env bash
# run_p8_escape.sh — P8 "escape-closure" adversarial pillar orchestrator.
#
# The escape-closure pillar exists so that every externally-reported bug CLASS
# of the 2026-07 cycle would have been caught by our own framework FIRST. Each
# axis targets one escape (see docs/TESTING.md "P8 escape-closure pillar" and
# .swarm/P8_ESCAPE_ANALYSIS.md):
#
#   1 binding-form      AD point construction-form sweep (crash/silent-wrong)
#   2 indirection       AD callable reached direct/param/curried/let/2-level
#   3 arity-sweep       manifest-driven native-vs-VM parity ratchet
#   4 property-oracle    reference-free number/write round-trip + algebraic id
#   5 concurrency-fuzz   parallel-map scope-stack race corpus (vs serial oracle)
#   6 five-way-surface   doc<->manifest<->native<->VM<->provide agreement ratchet
#   7 fault-injection    toolchain exit-0-masking matrix
#   8 mem-profiles       workload-shaped flat-RSS invariant
#
# Axes 1/2/4/5 emit self-checking .esk corpora (shared scripts/p8/harness.py
# format) that THIS runner executes and classifies; axes 3/6/7/8 are
# self-contained scripts invoked here. All emit kind=escape_matrix JSON-L into
# scripts/icc_traces/escape_matrix.jsonl plus a roll-up p8_escape_matrix_green
# consumed by .icc/completion-oracles.yaml.
#
# Bounded / seeded / disk-capped with cleanup: corpora are generated into a
# per-run temp dir (removed on exit) with a fixed seed; --quick keeps the CI
# lane fast (< ~2 min), full mode (nightly) runs every axis at depth including
# the AOT and VM lanes.
#
# Usage: run_p8_escape.sh [--quick] [--full] [--build-dir DIR] [--axes LIST]
#   --quick        CI subset (JIT lane, sampled axis-3/axis-8)   [default]
#   --full         nightly: JIT+AOT+VM lanes, full axis-3, par_batch mem shape
#   --axes a,b,c   run only these axis numbers (default: all)
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
MODE="quick"; AXES="1,2,3,4,5,6,7,8"
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) MODE="quick"; shift;;
    --full)  MODE="full"; shift;;
    --build-dir) BUILD_DIR="$2"; shift 2;;
    --axes) AXES="$2"; shift 2;;
    *) shift;;
  esac
done
case "$BUILD_DIR" in /*) ;; *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR";; esac
export BUILD_DIR
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
VM_BIN="$BUILD_DIR/eshkol-vm-standalone-test"
[ -x "$ESHKOL_RUN" ] || { echo "run_p8_escape: $ESHKOL_RUN not found — build first." >&2; exit 2; }

TRACE_DIR="$REPO_ROOT/scripts/icc_traces"; mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/escape_matrix.jsonl"
: > "$TRACE_FILE"     # fresh evidence set each run

# Per-run isolation, and pin the binary under test. This suite shells out to
# eshkol-run for many minutes; a rebuild in the same worktree mid-run used to
# swap the compiler underneath it and produce verdicts (including crashes) that
# belong to no single build. Run against a private copy instead.
ESHKOL_TEST_ISOLATION_NO_TRAP=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/test_isolation.sh"
eshkol_test_isolation_init "p8-escape"
PINNED_BUILD_DIR="$(eshkol_test_pin_toolchain "$BUILD_DIR")"
ESHKOL_RUN="$PINNED_BUILD_DIR/eshkol-run"
[ -x "$PINNED_BUILD_DIR/eshkol-vm-standalone-test" ] \
    && VM_BIN="$PINNED_BUILD_DIR/eshkol-vm-standalone-test"
echo "run_p8_escape: pinned toolchain -> $PINNED_BUILD_DIR" >&2

WORK="$ESHKOL_TEST_TMPDIR/work"; mkdir -p "$WORK"
# Disk cap: a runaway generator must not fill the disk (fuzz disk-budget policy).
cleanup() { rm -rf "$WORK"; eshkol_test_isolation_cleanup; }
trap cleanup EXIT
export ESHKOL_JIT_CACHE_DIR="$WORK/jit"; mkdir -p "$ESHKOL_JIT_CACHE_DIR"
DISK_CAP_KB=$(( 512 * 1024 ))   # 512 MB corpus ceiling

emit() { python3 -c 'import json,sys;print(json.dumps({"kind":"escape_matrix","name":sys.argv[1],"value":sys.argv[2],"snippet":sys.argv[3][:200],"confidence":0.95}))' "$1" "$2" "$3" >> "$TRACE_FILE"; }
want() { case ",$AXES," in *",$1,"*) return 0;; *) return 1;; esac; }

FAILS=0
note_fail() { FAILS=$((FAILS+1)); }

# ---- run one self-checking .esk under a mode; echo verdict token -----------
# modes: jit | aot | vm.  Verdicts: PASS FAIL CRASH XKNOWN XPASS
run_selfcheck() { # file mode
  local f="$1" mode="$2" out rc decl npass nfail sums
  # grep -c prints 0 and exits 1 on no-match; capture the count, ignore status.
  local xcrash vmskip
  xcrash=$(grep -c 'P8-XCRASH' "$f" 2>/dev/null); xcrash=${xcrash:-0}
  vmskip=$(grep -c 'VM-SKIP' "$f" 2>/dev/null); vmskip=${vmskip:-0}
  decl=$(grep -m1 'CHECKS:' "$f" | grep -oE '[0-9]+' | head -1)
  case "$mode" in
    jit) out=$("$ESHKOL_RUN" -r "$f" 2>/dev/null); rc=$?;;
    aot) local bin="$WORK/a.out"; rm -f "$bin";
         if "$ESHKOL_RUN" "$f" -o "$bin" >/dev/null 2>&1 && [ -x "$bin" ]; then
           out=$("$bin" 2>/dev/null); rc=$?; rm -f "$bin";
         else out=""; rc=200; fi;;
    vm)  [ "$vmskip" -ne 0 ] && { echo SKIP; return; }
         local eskb="$WORK/a.eskb"; rm -f "$eskb";
         if "$ESHKOL_RUN" --profile hosted-vm --emit-eskb "$eskb" "$f" >/dev/null 2>&1 && [ -f "$eskb" ]; then
           out=$(ESHKOL_VM_NO_DISASM=1 "$VM_BIN" "$eskb" 2>/dev/null); rc=$?; rm -f "$eskb";
         else echo SKIP; return; fi;;
  esac
  npass=$(printf '%s' "$out" | grep -c '^PASS:')
  nfail=$(printf '%s' "$out" | grep -c '^FAIL:')
  sums=$(printf '%s' "$out" | grep -c '^SUMMARY')
  local clean=0
  [ "$rc" -eq 0 ] && [ "$sums" -ne 0 ] && [ "$nfail" -eq 0 ] && [ "$npass" = "$decl" ] && clean=1
  if [ "$xcrash" -ne 0 ]; then
    [ "$clean" -eq 1 ] && echo XPASS || echo XKNOWN
  else
    if [ "$clean" -eq 1 ]; then echo PASS
    elif [ "$rc" -ne 0 ] || [ "$sums" -eq 0 ]; then echo CRASH
    else echo FAIL; fi
  fi
}

# ---- drive a generated corpus dir across the mode set ---------------------
drive_corpus() { # axis_id label dir modes...
  local axis="$1" label="$2" dir="$3"; shift 3
  local modes="$*"
  local p=0 f=0 c=0 xk=0 xp=0 sk=0 total=0
  for esk in "$dir"/*.esk; do
    [ -e "$esk" ] || continue
    for m in $modes; do
      total=$((total+1))
      case "$(run_selfcheck "$esk" "$m")" in
        PASS) p=$((p+1));;
        FAIL) f=$((f+1)); echo "  FAIL  [$m] $(basename "$esk")"; echo "FAILED tests/escape_matrix/$axis::$(basename "$esk" .esk)::$m";;
        CRASH) c=$((c+1)); echo "  CRASH [$m] $(basename "$esk")"; echo "FAILED tests/escape_matrix/$axis::$(basename "$esk" .esk)::$m";;
        XKNOWN) xk=$((xk+1));;
        XPASS) xp=$((xp+1)); echo "  XPASS [$m] $(basename "$esk") (known-open cell now clean — promote to gate)";;
        SKIP) sk=$((sk+1));;
      esac
    done
  done
  local status="PASS"
  { [ "$f" -ne 0 ] || [ "$c" -ne 0 ] || [ "$xp" -ne 0 ]; } && status="FAIL"
  emit "p8_axis${axis}_${label}" "$status" "pass=$p fail=$f crash=$c xknown=$xk xpass=$xp skip=$sk total=$total"
  printf "  axis-%s %-16s pass=%d fail=%d crash=%d xknown=%d xpass=%d (modes: %s)\n" \
         "$axis" "$label" "$p" "$f" "$c" "$xk" "$xp" "$modes"
  [ "$status" = "PASS" ] || note_fail
}

disk_guard() { # dir
  local kb; kb=$(du -sk "$1" 2>/dev/null | awk '{print $1}')
  if [ -n "$kb" ] && [ "$kb" -gt "$DISK_CAP_KB" ]; then
    echo "run_p8_escape: corpus $1 exceeded ${DISK_CAP_KB}KB disk cap — aborting" >&2
    exit 3
  fi
}

echo "== P8 escape-closure pillar ($MODE) =="

if [ "$MODE" = "full" ]; then GEN_MODES="jit aot vm"; else GEN_MODES="jit"; fi

# ---------- axes 1 & 2: AD binding-form + indirection ----------------------
# The AD generator interleaves ad_bind_* (axis 1) and ad_indir_* (axis 2) files;
# split them so each axis reports independently.
if want 1 || want 2; then
  GD="$WORK/ad"; python3 scripts/p8/gen_ad_escape.py --out "$GD" --seed 8801 >/dev/null
  disk_guard "$GD"
  mkdir -p "$WORK/ad1" "$WORK/ad2"
  for f in "$WORK"/ad/ad_bind_*.esk; do [ -e "$f" ] && cp "$f" "$WORK/ad1/"; done
  for f in "$WORK"/ad/ad_indir_*.esk; do [ -e "$f" ] && cp "$f" "$WORK/ad2/"; done
  want 1 && drive_corpus 1 binding-form "$WORK/ad1" $GEN_MODES
  want 2 && drive_corpus 2 indirection  "$WORK/ad2" $( [ "$MODE" = full ] && echo "jit aot" || echo "jit" )
fi

# ---------- axis 4: property oracles ---------------------------------------
if want 4; then
  PD="$WORK/prop"; python3 scripts/p8/gen_property_oracles.py --out "$PD" --seed 8804 >/dev/null
  disk_guard "$PD"
  drive_corpus 4 property-oracle "$PD" $( [ "$MODE" = full ] && echo "jit aot vm" || echo "jit" )
fi

# ---------- axis 5: concurrency fuzz ---------------------------------------
if want 5; then
  CD="$WORK/conc"
  if [ "$MODE" = full ]; then SH=10; RP=20; else SH=6; RP=20; fi
  python3 scripts/p8/gen_concurrency_fuzz.py --out "$CD" --seed 8805 --repeats "$RP" --shapes "$SH" >/dev/null
  disk_guard "$CD"
  drive_corpus 5 concurrency-fuzz "$CD" jit
fi

# ---------- axis 3: manifest-driven arity sweep (native vs VM) -------------
if want 3 && [ -x "$VM_BIN" ]; then
  if [ "$MODE" = full ]; then A3="--full"; else A3="--sample 30 --seed 8803"; fi
  if python3 scripts/p8/p8_arity_sweep.py --native "$ESHKOL_RUN" --vm "$VM_BIN" \
       $A3 --trace "$TRACE_FILE" --workdir "$WORK/arity" --timeout 8 >/dev/null 2>&1; then
    echo "  axis-3 arity-sweep     PASS (native-vs-VM parity ratchet, no NEW divergence)"
  else
    echo "  axis-3 arity-sweep     FAIL (NEW native-vs-VM divergence — see trace)"; note_fail
  fi
elif want 3; then
  echo "  axis-3 arity-sweep     SKIP (no VM binary)"
fi

# ---------- axis 6: five-way surface agreement -----------------------------
if want 6; then
  if python3 scripts/p8/five_way_surface.py --repo-root "$REPO_ROOT" --trace "$TRACE_FILE" >/dev/null 2>&1; then
    echo "  axis-6 five-way        PASS (doc/manifest/native/VM/provide agreement, no NEW gap)"
  else
    echo "  axis-6 five-way        FAIL (NEW surface disagreement — see trace)"; note_fail
  fi
fi

# ---------- axis 7: toolchain fault injection ------------------------------
if want 7; then
  if bash scripts/p8/p8_fault_injection.sh --build-dir "$BUILD_DIR" $( [ "$MODE" = quick ] && echo --quick ) >/dev/null 2>&1; then
    echo "  axis-7 fault-injection PASS (no NEW exit-0 masking; #334 gates hold)"
  else
    echo "  axis-7 fault-injection FAIL (a hard-gate fault cell masked, or an XKNOWN was fixed)"; note_fail
  fi
fi

# ---------- axis 8: workload memory profiles -------------------------------
if want 8; then
  if bash scripts/p8/p8_mem_profiles.sh --build-dir "$BUILD_DIR" $( [ "$MODE" = quick ] && echo --quick ) >/dev/null 2>&1; then
    echo "  axis-8 mem-profiles    PASS (flat-RSS invariant across workload shapes)"
  else
    echo "  axis-8 mem-profiles    FAIL (RSS grew with work — reclamation regression)"; note_fail
  fi
fi

emit p8_escape_matrix_green "$([ "$FAILS" -eq 0 ] && echo PASS || echo FAIL)" "failed_axes=$FAILS mode=$MODE"
echo
echo "Trace written: $TRACE_FILE"
if [ "$FAILS" -ne 0 ]; then
  echo "P8 escape-closure gate: FAIL ($FAILS axis group(s) failed)"; exit 1
fi
echo "P8 escape-closure gate: PASS"
