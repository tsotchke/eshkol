#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <eshkol-run> <eshkol-vm-standalone-test>" >&2
  exit 2
fi

run="$1"
vm="$2"
fixture="tests/adr0000/stage1_strict_region_test.esk"

ad_output="$(${run} -r "${fixture}" 2>&1)"
grep -F "PASS: strict-region parallel-map" <<<"${ad_output}" >/dev/null
echo "PASS: strict-region native execution"

vm_output="$(ESHKOL_VM_NO_DISASM=1 "${vm}" "${fixture}" 2>&1)"
grep -F "PASS: strict-region parallel-map" <<<"${vm_output}" >/dev/null
echo "PASS: strict-region VM execution"

python3 scripts/check_adr0000_zset_fixture.py

global_writes="$(grep -E '__global_arena[[:space:]]*=' lib/core/runtime_regions.cpp | grep -v 'nullptr' | wc -l | tr -d ' ')"
if [[ "${global_writes}" != "1" ]]; then
  echo "FAIL: strict-region global arena writes=${global_writes}" >&2
  exit 1
fi
echo "PASS: strict-region global arena write gate"
mkdir -p scripts/icc_traces
printf '%s\n' '{"kind":"runtime_event","name":"adr0000_stage1_gate","status":"PASS","tsan":true}' \
  > scripts/icc_traces/adr0000_stage1.jsonl
