#!/usr/bin/env bash
# One recipe shared by the complete smoke battery and the early release gate.
eshkol_release_invariant_probes() {
    probe abi_layout_pin "object header layout and guard symbol are pinned" \
        '"$(dirname "$ESHKOL_RUN")/abi_layout_pin_test"'
    probe abi_object_header_ratchet "no new object-header layout dependence" \
        'python3 "$REPO_ROOT/scripts/abi_header_inventory.py" check --repo "$REPO_ROOT"'
    probe closed_enum_dispatch_exhaustive "closed enum dispatch is exhaustive and armed" \
        'python3 "$REPO_ROOT/scripts/gate_exhaustive_dispatch.py"'
    probe ad_exactness_gate \
        'the no-finite-differences guarantee is enforced by a counter that can actually read nonzero: exact gradients report 0 FD evals, a real finite-difference backward reports exactly its perturbations and turns the shipped assertion #f (both engines), and matmul AD tape node counts stay within their ratchet with gradients exact' \
        'cd "$REPO_ROOT";
         out=$(BUILD_DIR="$BUILD_DIR" bash scripts/run_ad_exactness_gate.sh 2>&1) || exit 1;
         printf "%s" "$out" | grep -q "AD exactness gate: PASS"'
}
