# Eshkol — Complete ICC-led Review (2026-07-01)

Method: ICC campaign battery against the freshly re-indexed repo (`icc index --repo eshkol`,
drift cleared from 83 stale files) at master `bea8ba1c` (+ local eagle_train commit):
`production-audit`, `weakness-map`, canonical `readiness --target v1.3-evolve` (via
`scripts/run_v1_3_readiness.sh`, full smoke + trace), SICP full-book gate
(`scripts/run_sicp_smoke.sh`), `audit-patterns --preset shell-hardening`,
`find-stubbed-prod-paths`, `find-dead-code`, `find-contract-gaps`, `feature-inventory`,
`release-streams`, `test-coverage-audit`, plus live CI-lane verification.

## Verdict at a glance

| Surface | Verdict | Evidence |
|---|---|---|
| **v1.3-evolve readiness (canonical gate)** | **READY — score 100/100** | all smoke probes PASS (incl. AD input2 matmul/conv2d/batchnorm/layernorm/attention, agent-FFI AOT link, image-IO, keyword args, let-match, define-library, JIT cache invalidation, string R7RS edges); oracle `complete`; 0 contract gaps; 0 unmarked stubs |
| **CI (all platforms)** | **14/14 lanes SUCCESS** | linux x64+arm64 (lite/cuda/xla/asan), macOS x64+arm64 (lite/xla), **windows-arm64 lite/cuda/xla all green** — the last baseline-red platform is closed |
| **SICP full-book gate** (promoted to a v1.3 gate in #102) | **44/88 probes** — 22 required book systems not yet written | missing = the pending ESH-0029..0051 corpus (details below) |
| **production-audit** | `fail` on **hygiene only** | 6 risks: uncommitted local work (50 files), index-drift, 10 shell-hardening findings (test scripts), 2 justified-oversized files grew |
| **Contract gaps** | 0 | `find-contract-gaps` clean |
| **Stubbed production paths** | none dangerous | top classes are `external_backend_required` (GPU/XLA — legitimate) and `diagnostic_only` (benchmarks/logging); no stub/todo/unimplemented in prod paths |
| **Release streams** | 12/12 structurally `ready` | but all 12 skip *runtime* readiness (no scoped targets) — weakness #3 |

## 1. What is genuinely done (verified, not claimed)

- **Language correctness campaign landed.** The full arc from this cycle is merged:
  tensor type-guard (#79), conv2d NCHW unification (#80), with-region OOM 6.6GB→41MB (#81),
  numeric tower R7RS (#82), shared mutable capture (#83), **nested/higher-order AD (#84)**,
  first-class predicate booleans (#86), letrec instance isolation (#89), deep-CPS O0 cleanup
  (#93), runtime scalar nested gradients (#95), capability-denied signaling (#96 — the
  Noesis getenv follow-up ESH-0076), AD tensor metadata guards (#97), atomic object emission
  (#92 — the JIT-cache temp-.o race), object-emission watchdog (#99), error-string interning
  (#100). Plus the static-getenv link CTest (#88, ESH-0077).
- **Every platform lane is green** including windows-arm64 after the 3-layer unbreak
  (sincos shim → thunk convergence → small-model + FunctionSections + /OPT:REF,ICF
  dead-strip; matmul AOT object 248MB→53MB, compile 89s→18s as a side benefit).
- **The canonical release gate is real and passing**: readiness `v1.3-evolve` = ready/100
  with trace-aware oracle evidence, runnable via `scripts/run_v1_3_readiness.sh`.

## 2. The one big open front for v1.3: SICP full book (44/88)

PR #102 deliberately raised the bar from "SICP sampler" to **whole-book gate**. The 22
missing systems map 1:1 to ledger tasks ESH-0029..0051, each needing an `-r`+AOT probe:

| Book area | Tasks |
|---|---|
| ch2 completions | 2.2.4 picture-language painters (0038), 2.5 generic tower+coercion (0039), 2.5.3 polynomials (0040) |
| ch3 completions | 3.3.1 mutable pairs/cycles (0041), 3.3.4 circuits (0035), 3.3.5 constraints (0036), 3.4 concurrency/serializers (0037), 3.5 accelerated/power-series/integrator/random streams (0042–0045) |
| ch4 completions | 4.1.7 analyzing evaluator (0046), 4.1 derived forms (0047), 4.2 lazy evaluator (0029), 4.3 ambeval + nondet parser (0048–0049), 4.4 query system (0030) |
| ch5 completions | 5.1–5.2 stack ops + recursive machines (0050–0051), 5.3 GC model (0031), 5.4 explicit-control evaluator (0032), 5.5 SICP compiler (0033) |

These are *corpus-writing* tasks (the language substrate they need — first-class predicates,
closure isolation, deep CPS, streams — is now fixed), so they parallelize cleanly.

## 3. Weakness map (ICC ranked)

1. **(high, must) Adversarial eval scenarios** — the eval suite lacks failure-mode coverage
   (dirty worktrees, stale artifacts, model-server outage, disk pressure, failed gates).
2. **(high, must) Documentation truth** — of 8,404 extracted doc claims: 2,889 grounded,
   **3,020 unsupported**, 2,495 unresolved refs (9,896 unresolved references overall).
   Feature inventory agrees: 319/980 symbol clusters have **zero** doc coverage
   (`lib/backend/VM`, `core.memory`, `ad/tape`, `dnc/sdnc`, `agent/eagle`, GPU kernels…).
3. **(medium) Release streams lack runtime readiness targets** — all 12 streams pass
   structurally but skip live-evidence checks.

## 4. Code-health findings

- **Dead code:** 541 evidence-filtered candidates. Top: `vm_ts_classify_line`
  (vm_native.c:2553, **9.7KB**), `sys_ts_classify_line` (system_builtins.c:4204, 1.1KB),
  `init_readline` (eshkol-repl.cpp), `EshkolLLVMCodeGen::codegenVariableDefinition`
  (llvm_codegen.cpp:10692, 886 lines, orphaned), `parse_string_literal` (vm_parser.c),
  unused XLA reduce-gradient/codegen paths, `eshkol_tensor_svd`. A wire-or-delete sweep
  would shrink the 29K-line codegen and the VM meaningfully.
- **Shell-hardening:** 10 findings, 1 high (`cat > "$SRC" <<` heredoc in
  object_emit_phase_trace_test.sh) + 9 medium (unquoted-var redirects in test scripts).
  All in test/harness scripts, none in product code — low risk, cheap to fix.
- **Modularity:** the 2 justified monoliths (llvm_codegen.cpp, repl_jit.cpp) keep growing;
  production-audit asks for either extraction or updated justification evidence.
- **Python tooling untested:** 7 repo Python tools (check_wasm_imports, paper table gens,
  codegen_audit…) have 0 tests.

## 5. Hygiene (blocking `production-audit: pass`)

- ~50 uncommitted local files (the eagle_train/EAGLE FFI work, .swarm ledger updates,
  Noesis correspondence docs, `docs/platform/` contract-surface drafts, readiness script).
  These need commit-or-discard decisions; the eagle work (`lib/agent/c/agent_eagle_training.c`,
  `lib/agent/eagle.esk`, `examples/eagle_train.esk`, FFI test) looks commit-worthy.
- Re-index after committing (drift gate is strict: 1 modified file re-flags).

## 6. Open work registry (beyond SICP)

- **Open PRs:** #101 (types: skip body scans for external defines), #103 (SICP ch2–3 probes).
- **GPU campaign (delegated) still pending:** ESH-0022/0023 (critical: gpu-* builtins,
  AOT GPU dispatch), 0026/0027/0028 — tsotchke-chan's decomposition hasn't landed PRs yet;
  needs a follow-up nudge or local worktree agents.
- **AD residuals:** ESH-0071 (JIT gradient-recompile memoization — Noesis perf ask),
  ESH-0072 (in_progress: local-scalar capture PtrToInt).
- **Platform foundations:** ESH-0088 (AOT large-import-graph branch-island limits, active),
  native image-IO per-OS backends (0007–0009), event loop (0011), channels (0016),
  module privacy (0014), tooling triad (0013).
- **Noesis v1.5 gates:** ESH-0082–0085 (release gate + oracle, qLLM checkpoint loader,
  GeoRefine bundle gate, crypto symbol hygiene).
- **Known eagle-work finding:** vector-valued `(gradient fn vec)` throws
  `int64 from non-int-storage cell (type=6)` — scalar path exact, vector path broken;
  blocks the 7168-dim native EAGLE trainer (workaround: FFI linear_backward). Not yet a
  ledger task — should be filed as the next AD frontier.

## 7. Recommended order of attack

1. **Commit the local eagle/ledger work** (unblocks production-audit hygiene + reindex).
2. **File the vector-AD task** and fix it (or wire FFI linear_backward) — it gates the
   flagship native-training story.
3. **SICP full-book sprint**: fan out ESH-0029..0051 (corpus tasks, parallelizable) to
   close 44/88 → 88/88; that plus #101/#103 completes the v1.3 gate set.
4. **GPU campaign re-drive** (ESH-0022/0023 critical for v1.5).
5. **Doc-truth pass** on the 319 zero-coverage clusters + 3,020 unsupported claims
   (weakness-map task `documentation-truth-gate`).
6. **Dead-code sweep** (top ~15 items) + shell-hardening fixes + runtime readiness
   targets for the 12 release streams.
