#!/usr/bin/env bash
# One recipe shared by the complete smoke battery and the early release gate.
eshkol_release_invariant_probes() {
    probe abi_layout_pin "object header layout and guard symbol are pinned" \
        '"$(dirname "$ESHKOL_RUN")/abi_layout_pin_test"'
    probe abi_object_header_ratchet "no new object-header layout dependence" \
        'python3 "$REPO_ROOT/scripts/abi_header_inventory.py" check --repo "$REPO_ROOT"'
    probe closed_enum_dispatch_exhaustive "closed enum dispatch is exhaustive and armed" \
        'python3 "$REPO_ROOT/scripts/gate_exhaustive_dispatch.py"'
}
