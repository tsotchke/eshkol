#!/usr/bin/env bash
# Unified executable gate for Eshkol's 16-pillar Mathematics Programme (M0-M15).
# Runs all 15 stdlib suites in JIT mode and the Lean 4 certificate round-trip gate.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ESHKOL_RUN="${ESHKOL_RUN:-${ROOT}/build/eshkol-run}"

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

echo "PASS: math_programme_gate (${passed}/${#SUITES[@]} stdlib suites + Lean 4 round-trip)"
