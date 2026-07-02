# ICC-readable Eshkol contract-surface manifest.
# This file is documentation for ICC extraction, not an included build script.

add_test(NAME eshkol_runtime_contract_smoke COMMAND build/eshkol-run --version)
add_test(NAME eshkol_aot_contract_smoke COMMAND build/eshkol-run -o /tmp/eshkol-aot-smoke /tmp/eshkol-aot-smoke.esk)
add_test(NAME eshkol_noesis_link_contract_smoke COMMAND ./scripts/build.sh --release)
