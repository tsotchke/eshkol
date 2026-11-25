#!/usr/bin/env bash
# Run the P6 rational/bignum Taylor coefficient symbolic-oracle profile.
set -euo pipefail
exec "$(dirname "$0")/run_ad_evidence_profile_gate.sh" p6 "$@"
