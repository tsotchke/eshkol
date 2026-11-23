# November 13 socket review follow-up

The completed qualification below preserves raw suite failures and records separate targeted environment confirmations.

## Source and scope

The qualification base is `53220efc1638c77d108ab009d4a5e6939c488c7b`. The shared foundation source is `b682c40c4fbb4bb0944226ff2d49b0928c1fa785`, the freshly requalified PROCESS adapter is `d1814038877bc4fc416f588041382a34c59a83ba`, and the combined socket source is `8d99676f6467e362da34122bbe7f36d7e6e63907`.

The comparison control is `5d09f4aef21abe329d7dc6a6a5fcaae7a1e21839`: canonical integration **plus only the four reviewed #925 integration changes**, covering five files. It is not an unmodified-base build. The [control snapshot](https://github.com/Gabriel-Kahen/eshkol/commit/5d09f4aef21abe329d7dc6a6a5fcaae7a1e21839) is retained separately. Its cache publication retains the base's parallel initializer and the reviewed process-unique temporary files. #925's upstream integration remains maintainer work.

The shared foundation and combined core already accept PROCESS and use the same qualified registry file. The earlier PROCESS snapshot contained a redundant second PROCESS disjunct. The requalified PROCESS source adopts the exact shared/core registry bytes; its other 528 recorded native source hashes remain unchanged. Shared/core never lacked PROCESS acceptance. Each own-source and combined qualification retains its source identity.

Both full configured builds passed using Release, Clang 22.1.8/LLVM 21.1.8, agent FFI/tests enabled and GPU/BLAS/examples/Python disabled. Control uses Ninja; patched uses Unix Makefiles with compiler/stdlib prerequisites built before all targets. This is the recorded CPU configuration, not every optional platform configuration.

## Completed qualification

- **Shared foundation: 10/10 fresh Release CTests passed**, with Release assertions enabled specifically in the explicit-loop fixture. Standalone common controls also passed normally and under ASan/UBSan/leak detection.
- **Combined native socket/common/PROCESS qualification passed at exact source `8d99676...`**, normally and under ASan/UBSan/leak detection.
- [Hosted run 43493801710](https://github.com/Gabriel-Kahen/eshkol/actions/runs/43493801710) passed Linux epoll, Linux ASan/UBSan/leak, and macOS 15 arm64 kqueue at that same clean source. The Mac claim is native socket/common/PROCESS only.
- **25 runner controls passed.** Every ordinary JIT case measures its own fixture compilation. Controls verify the actual owned-child timeout changes from 45 seconds for a controlled 2-second compile to 110 seconds for a controlled 20-second compile, for JIT and AOT alike. Compiler-only measurements require a nonempty object and no error diagnostics. Bounds remain finite: compile 90 seconds, execution 45–180 seconds, peer lifetime 300 seconds, case CTest limit 300 seconds. These synthetic controls are not product execution.
- **33 Guile wrapper checks passed**, separately scoped as controlled-native wrapper tests.

## Corrections and public contract

| Follow-up | Implemented behavior |
|---|---|
| Empty negative poll after resource retirement | Actual managed-owner count gates the wait; terminal tickets do not make an empty loop wait indefinitely. Pending owners still retain the wake/progress path. |
| Spent explicit-loop wait | Cancellation takes precedence; an exhausted budget returns before watch registration and loop polling. The base's legacy wait policy remains present. |
| Module boundary | `cleanup_detach` is declared on the public lease header; the socket binding no longer includes a private registry header. |
| Generated handle page | The page matches the current base; unrelated generated metadata is deferred to integration. |
| Native temporary directories | All three standalone programs honor a short writable absolute TMPDIR, with the same documented short fallback and portable pathname budget. |
| Simultaneous cleanup errors | Primary descriptor error is retained while pathname responsibility remains independently retryable. Retry never repeats an attempted descriptor close; rollback still reports the residual path error separately. |
| Cleanup stage naming | `ESHKOL_CLEANUP_STAGE_*` names distinguish stages; old aliases retain their values. |
| Scoped-helper reference | Non-local exit preserves its exception/continuation control and does not report cleanup results. Explicit ticket forms guarantee inspection on those paths. |
| Retry checker role | OBSERVE denotes borrowing the same owner, not purity. Runtime result states change, but no ticket is consumed or acquired; cancellation is a protocol transition because it changes admissible socket operations. |

## Completed full registered-suite comparison

Runs were sequential at `ctest -j3`, `OMP_NUM_THREADS=2`, on 16 logical CPUs. The raw control result is **1239 PASS / 133 FAIL / 1 SKIP** (1373 tests); the raw combined core is **1298 PASS / 133 FAIL / 1 SKIP** (1432 tests). All 1373 common statuses match, all 59 added tests pass, and no common test changed outcome. Both recorded full builds passed. This is a comparative result, not a full-suite green claim.

Full failure stdout was reviewed: 131/133 pairs match after source/build/cache/temp/timestamp normalization. The two lexical inventory failures additionally report this work's PROCESS 1 + Unix 2 bytevector macro sites. The subsequent scoped registration is limited to those added sites; it does not rewrite the raw results. Nine captured nested-shorthand text lane logs also match, including their missing-stdlib failure.

All 23 current net/TLS tests passed. Current strict qualification produced 148 compiled PASS cases (40 positive, 105 negative, 3 rejections), 8 profile runtime executions and 28 handle runtime executions. Unix public lifecycle logs report 44 checks in each of JIT and AOT. Combined production PROCESS public CLI reports 23 checks in each mode, with no support-library argument and the real production agent archive; this is separate from the own-PROCESS native qualification.

Private matched Clang22 inventory scans completed. Actual checker runs on the COMPLETE original-control report and scoped-proposal core report retain exactly the same 157 unrelated lexical+semantic violations. Separately, 28 constructed report-level controls passed: PROCESS 4 / core 8 count registrations preserve existing metadata and historical provenance, reject extra/disappearing sites and constrain semantic decreases. The earlier 18 lexical violations are not the complete semantic count. This is not a full inventory pass.

Observed raw failure diagnostics, before targeted setup confirmations:

- ABI inventory baseline drift: 2
- missing libclang Python bindings: 1
- source resolver cannot find requested module: 14
- other shared baseline output/contract failure: 9
- source resolver cannot find stdlib: 55
- fixture precompiled stdlib object lacks embedded identity: 1
- compiler reports unknown stdlib function: 30
- missing PyYAML: 11
- VM source lane cannot resolve stdlib (confirmed in captured lane log): 1
- missing usable GNU/BSD time: 1
- Wasm missing callable tensor policy import: 3
- missing Playwright: 5

Targeted source-SDK confirmation completed separately using each source lib path. For each phase, 103 selected tests plus one automatic fixture prerequisite ran. Selected outcomes were 101 PASS and 2 FAIL; including the fixture gives 102 PASS/2 FAIL. The retained failures are `types_ext_gate` (record type/session arity diagnostics) and `ad_vm_exactness_gate` (inexact VM divergence/curl outputs); neither retains a missing-SDK diagnostic. These reruns do not alter the original full-suite failures.

Private PyYAML confirmation control: 10 PASS / 1 FAIL / 0 SKIP. Retained non-pass cases: `test_v1_3_release_evidence_recipe`.

Private PyYAML confirmation patched: 10 PASS / 1 FAIL / 0 SKIP. Retained non-pass cases: `test_v1_3_release_evidence_recipe`.

The remaining YAML CTest failure was a child-interpreter dependency: the test overwrites PYTHONPATH with its scripts directory, hiding the parent-only private PyYAML target. A separate private venv of the same Python 3.14.6 exposes pinned PyYAML 6.0.3 through site-packages. The exact remaining script then passed all 19 unit cases on EACH source. This is a direct narrow script confirmation, not another CTest run; the initial 10/11 CTest outcomes remain recorded.

Targeted YAML confirmation completed using private pinned PyYAML6.0.3 without an SDK-root override or system installation. Per-test outcomes and original-byte hashes are recorded separately in the manifest; raw YAML failures remain retained.

The absent system GNU time and out-of-tree workspace fixture were separately confirmed by narrow setup reruns on both sources. The aborted preflight without the LLVM loader path is excluded. The final branch amendment scope is documentation, scoped inventory metadata and a stale JIT comment; runtime qualification remains bound to 8d, rather than claiming every final production source byte is identical.

## Limits

The raw full suites remain red, and earlier failures can mask later behavior. Native Mac acceptance does not establish Mac public JIT/AOT, net/TLS or packaging. Windows/WASM refusal controls do not establish target execution. Raw machine logs stay local or on hosted CI; this summary and its [manifest](unix-socket-review-followup-2026-11-13.json) retain portable provenance and original-byte hashes. No maintainer approval or merge is claimed.

Fresh own-PROCESS four-case CTest and native normal/ASan/UBSan/leak qualification is recorded in the [PROCESS followup](subprocess-review-followup-2026-11-13.md); foundation qualification is in the [shared followup](shared-loop-leases-2026-11-13.md).

Final checks against the amended tree pass the strict source pins. The lexical inventory gate still reports exactly the control’s 18 unrelated file/class violation headings, with no PROCESS or Unix binding finding. The applied metadata matches the reviewed proposal exactly. A source comparison confirms that the only runtime/build/test file difference from the fully tested socket source is the stated JIT comment replacement.
