#!/usr/bin/env bash
# Unified executable gate for Eshkol's Complete Grothendieckian & Beyond-Grothendieck
# Mathematics Programme (Pillars M0-M15 + Grothendieck Eras 1-3 + Beyond Frontiers).
# Runs all 22 stdlib suites in JIT mode, the Lean 4 certificate round-trip gate,
# and all 6 standalone Lean 4.12.0 Grothendieck formal proof suites.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ESHKOL_RUN="${ESHKOL_RUN:-${ROOT}/build/eshkol-run}"
LEAN_BIN="${LEAN_BIN:-$(command -v lean 2>/dev/null || echo "${HOME}/.elan/bin/lean")}"

if [ ! -x "${ESHKOL_RUN}" ]; then
  echo "ERROR: eshkol-run binary not found at ${ESHKOL_RUN} (set ESHKOL_RUN=...)" >&2
  exit 2
fi

SUITES=(
  tests/stdlib/operator_cstar_cp_test.esk
  tests/stdlib/topology_diagram_cellular_test.esk
  tests/stdlib/measure_markov_test.esk
  tests/stdlib/modal_logic_test.esk
  tests/stdlib/analysis_complex_variational_test.esk
  tests/stdlib/set_higher_algebra_test.esk
  tests/stdlib/scheme_grothendieck_test.esk
  tests/stdlib/geometry_em_test.esk
  tests/stdlib/quant_sde_test.esk
  tests/stdlib/qft_hep_test.esk
  tests/stdlib/hodge_cycles_periods_test.esk
  tests/stdlib/alg_group_diffeo_git_test.esk
  tests/stdlib/integrable_lax_hitchin_qybe_test.esk
  tests/stdlib/hodge_hitchin_lean4_test.esk
  tests/stdlib/grothendieck_unified_cross_pillar_test.esk
  tests/stdlib/grothendieck_functional_test.esk
  tests/stdlib/grothendieck_sga_motives_grr_test.esk
  tests/stdlib/grothendieck_pillars_six_ops_test.esk
  tests/stdlib/grothendieck_stacks_derivators_test.esk
  tests/stdlib/grothendieck_esquisse_tame_perfectoid_test.esk
  tests/stdlib/grothendieck_beyond_frontiers_test.esk
  tests/stdlib/grothendieck_all_eras_unified_test.esk
)

LEAN_PROOFS=(
  proofs/lean/EshkolGrothendieckEra1Functional.lean
  proofs/lean/EshkolGrothendieckEra2SGAMotivesGRR.lean
  proofs/lean/EshkolGrothendieckPillarsSixOps.lean
  proofs/lean/EshkolGrothendieckEra3StacksDerivators.lean
  proofs/lean/EshkolGrothendieckEra3EsquissePerfectoid.lean
  proofs/lean/EshkolGrothendieckBeyondFrontiers.lean
)

cd "${ROOT}"
passed=0
for s in "${SUITES[@]}"; do
  echo "=== [JIT] ${s} ==="
  "${ESHKOL_RUN}" -r -I "${ROOT}/lib" "${s}"
  passed=$((passed + 1))
done

echo "=== [Lean 4 + CertCheck Round-Trip] ==="
ESHKOL_RUN="${ESHKOL_RUN}" python3 tests/certificates/run_hodge_lean4_roundtrip.py

lean_passed=0
if [ -x "${LEAN_BIN}" ]; then
  for lp in "${LEAN_PROOFS[@]}"; do
    echo "=== [Lean 4.12] ${lp} ==="
    "${LEAN_BIN}" "${lp}"
    lean_passed=$((lean_passed + 1))
  done
else
  echo "ERROR: lean binary not found at ${LEAN_BIN}" >&2
  exit 3
fi

echo "PASS: math_programme_gate (${passed}/${#SUITES[@]} stdlib suites + Hodge Lean 4 round-trip + ${lean_passed}/${#LEAN_PROOFS[@]} Lean 4 proof suites)"
