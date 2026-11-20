#!/usr/bin/env bash
# Run the nested-differentiation evidence profile (ADR-0027, ADR-0057):
# tests/ad/nested_depth_matrix_test.esk on the JIT, AOT and the VM.
set -euo pipefail
exec "$(dirname "$0")/run_ad_evidence_profile_gate.sh" nested "$@"
