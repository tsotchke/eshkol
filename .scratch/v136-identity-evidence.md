# v1.3.6 identity and evidence preparation

- Version identity is `1.3.6` in CMake and the public header; the release
  suffix is `1.3.6-evolve`.
- The frozen v1.3.5 record remains dated 2026-09-22. Public v1.3.6
  publication is a separate pending event with no date assigned; no release-ready
  or published verdict is asserted by this change.
- `.icc/completion-oracles.yaml` adds `v1.3.6-evolve`, carrying the strict
  ledger, oracle-integrity, freshness, changelog, disclosure, context and
  silent-wrong gates, plus JIT/AOT certified-enclosure containment evidence.
- Full CTest totals, VM-parity totals, integration SHA and publication URL
  stay pending until measured at the final cut. Required commands are:
  `cmake --build build && ctest --test-dir build --output-on-failure`,
  `BUILD_DIR=build scripts/run_vm_parity.sh`, and
  `python3 scripts/verify_v1_3_release_evidence.py --target v1.3.6-evolve
  --trace-dir <fresh-trace-dir>`.
