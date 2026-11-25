#!/usr/bin/env bash
# Backwards-compatible P11 entry point for the shared profile launcher.
set -euo pipefail
exec "$(dirname "$0")/run_ad_evidence_profile_gate.sh" p11 "$@"
