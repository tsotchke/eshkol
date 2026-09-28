# v1.3.6 release mechanics evidence

- Base: `2439e728263d6b7a8367af1b8fbb66873e6f2225` (PR #727 exact head).
- ICC source discovery: release readiness was hard-coded to `v1.3.5-evolve` in
  `scripts/run_v1_3_readiness.sh`, `.github/workflows/release.yml`, and
  `scripts/release_autopilot.py`; the frozen v1.3.5 oracle record remains unchanged.
- ICC status: repository artifacts present; isolated worktree used at
  `.worktrees/v136-release-mechanics`.
- Focused tests: `python3 -m unittest tests/toolchain/test_release_readiness_guard.py tests/toolchain/test_release_automation_workflow.py tests/toolchain/test_release_autopilot.py`
  -> 36 tests PASS.
- Syntax/config checks: `bash -n scripts/run_v1_3_readiness.sh`, Python compile,
  and workflow YAML parse PASS.
- ICC precommit: `icc pre-commit-check --repo eshkol --invoking-repo-path "$PWD" --allow-drift`
  -> PASS, 0 blockers.
- Negative controls cover wrong candidate target, wrong target in verdict, wrong
  SHA, dirty checkout, nonnumeric/wrong score, and pending release notes.
