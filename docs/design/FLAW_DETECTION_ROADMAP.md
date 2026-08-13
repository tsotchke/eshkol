# Flaw-detection capability roadmap

Status: PROPOSED
Scope: v1.3.5 and v1.4
Written against: master `2ae3787f`, plus the fix lanes open at the time of writing (PR #420-#440)
Supersedes nothing. Composes with: the v1.3.5 evacuator flagship, the engine-parity campaign
(task #109), the ICC methodology task (#112), and the ICC behavioral-probe item (#120).

---

## 0. The direction this answers

> "i am happy the language is hardening, i think we need to expand our capabilities to detect
> flaws. this makes Eshkol truly trustworthy."

The v1.3.4 campaign hardened the language. It did so by finding **81 distinct ledgered
defects** (`.icc/silent-wrong-ledger.yaml` and the lanes stacked on it), of which the ICC
correctness chain found **zero**, the 58-probe ICC smoke harness found **zero of the 23
silent-wrong entries**, and the readiness oracle found **zero**. Almost every one was found by
a human reading code or running a program by hand.

That ratio is the subject of this document. Hardening is the work of fixing defects.
Trustworthiness is the property of being *able to find them* — repeatedly, mechanically, and
before a consumer does. This roadmap converts the campaign's accumulated escape analyses into
permanent detection infrastructure.

It is a design document and a set of prioritized build items. It fixes no defect and changes
no gate.

---

## 1. The trust thesis — what "truly trustworthy" means operationally

Eshkol is a compiler with an unusual trust surface: **two independent execution engines**
(native LLVM codegen and a bytecode VM), **four native compilation axes** (JIT-cached,
JIT-uncached, AOT-O0, AOT-O2), a **wasm** build, and an **automatic differentiation** layer
whose output is a number that no user can eyeball for correctness. A wrong derivative looks
exactly like a right one.

Trustworthiness is therefore not "the tests pass". It is a set of falsifiable operational
claims, each of which must be attached to a detector that would go red if the claim became
false.

### The five claims

**T1 — No silent wrong answers.**
The compiler never produces a wrong value with exit 0 and no diagnostic. A loud failure is
honest; a silent one is a lie told to a consumer. This is already the project's stated
release philosophy and is already machine-checked, one entry at a time, by
`scripts/gate_no_silent_wrong.py`. What is missing is not the gate but the *supply* of
entries: the gate can only block on defects somebody already found.

**T2 — The two engines mean the same thing.**
A program's meaning must not depend on which engine ran it. Today this is asserted for the
~68 curated `tests/vm_parity/corpus/` programs and for 12.21% of the language surface
(`scripts/run_engine_parity_coverage.py`, 136 of 1114 constructs with differential evidence,
floor 10.95%). For the other 87.79% nothing has ever required the engines to agree on a
value.

**T3 — A derivative is the derivative.**
An AD result must be exact, not plausible. This is the claim with the least human oversight
and the highest blast radius: SW-01, SW-03, SW-04, SW-05, SW-20, SW-21 and SW-33 were all
silent zeros or silently wrong gradients. The existing `scripts/run_ad_oracle.sh` (60/60) and
`scripts/run_ad_adversarial.sh` pin analytic values for the shapes somebody wrote down; none
of the seven above were among them.

**T4 — What the documentation says is what the binary does.**
A documented control that does nothing is a defect of the same severity as a wrong answer,
because a consumer relies on it identically. SW-10 shipped seven documented resource-limit
environment variables that were parsed and then consulted by nobody, for months.

**T5 — The gates that assert T1-T4 are themselves verified.**
A gate that grades green while measuring nothing is worse than no gate: it converts absence
of evidence into a positive claim. At the time of writing, `.icc/completion-oracles.yaml`
**does not parse on master** — broken by #429 at `2ae3787f`, so `icc readiness` silently fell
back to a generic 2-criterion oracle that always scores 100 (ledger entry PR-12; repair is
PR #436, unmerged). Every readiness number produced between those two commits is void.

### The trust equation

> Trust is not the number of tests that pass. It is the number of **defect classes** the
> project can detect, times the fraction of the surface each detector covers, times the
> fraction of the time the detector actually runs.

All three factors matter and the project is currently weak on all three:

| Factor | Current state | Evidence |
|---|---|---|
| Defect classes detectable | The correctness chain detects value and lexical disagreement only; behavioral probes do not exist | ICC correctness chain caught 0 of 62 ledgered defects |
| Surface fraction per detector | Execution coverage 100.00%; **comparison** coverage 12.21% | `run_language_coverage.sh` vs `run_engine_parity_coverage.py` |
| Frequency the detector runs | 14 of ~90 harnesses are reachable from any CI workflow | `.github/workflows/` grep, section D-13 |

### The doctrine this implies

Two rules follow, and both are already project practice in isolated places. This roadmap's
job is to make them universal.

1. **Retro-catch is the unit of detector value.** A detector is worth building only if it can
   be demonstrated to find a defect that was real at a known pre-fix SHA. Every fix PR in
   this campaign that added a detector proved it this way (#420, #426, #427, #428, #435,
   #438). A detector that has never been shown to catch anything is a hope, not a gate.

2. **Absence must be first-class.** Every probe in the 58-probe smoke harness asserts its own
   happy path — it certifies that something *works*. None asserts that a wrong answer is
   *impossible*. A defect that returns a wrong value with exit 0 is structurally invisible to
   a harness built entirely out of positive assertions. `no_open_silent_wrong` was the first
   criterion in the tree to make the absence of a defect class an oracle input. It should not
   be the last.

---

## 2. The escape record

What actually found the 81 ledgered defects:

| Source | Caught | Missed |
|---|---|---|
| ICC correctness chain (`contradictions` + `duplicate-implementations`) | 0 of 62 | all |
| ICC smoke harness (58 probes) | 0 of 23 SILENT-WRONG | 23 of 23 |
| ICC readiness / completion oracles | 0 of 23 | 23 of 23 |
| Parity / differential baselines | SW-13, SW-15..SW-18, SW-23, LE-07, all PARITY-RATCHET | every AD item, SW-06, SW-10, SW-11, SW-12 |
| `docs/KNOWN_ISSUES.md` (human record) | 13 entries | 10 entries |
| Direct human measurement | SW-06, SW-08, SW-10, SW-11, SW-12, LE-01, LE-04, DD-03, DD-04 | — |
| Fix-lane class-kill sweeps (a fix probing its own neighbourhood) | SW-25..SW-39, LE-09, LE-10, PR-12 | — |

The last row is the encouraging one, and it is the model for everything below. Once a lane
was required to probe the *class* rather than the filed reproducer, it produced defects at a
rate no standing gate matched. The `scripts/run_value_position_sweep.py` lane is the clearest
case: on its **first run** it found five previously unknown defects (SW-35, SW-36, SW-37,
SW-38, SW-39) across native rounding, `make-string`, `pow`, and the VM's `cadr`/`caddr`/
`string-length`/`string-ref`/`vref`.

Class-killing works. It is currently a human discipline applied per lane. The roadmap makes
it infrastructure.

---

## 3. The gap catalog

Sixteen gaps. Eleven were named in the maintainer's brief and are verified here against
sources; five (D-04, D-13, D-14, D-15, D-16) were found while verifying the other eleven.

Each gap states the defect class it lets through, the evidence, and — the part that matters —
**why the gates the project already owns cannot see it**. A gap is only a gap if it is
structural. If an existing gate would have caught it with more corpus, it is a corpus item,
not a roadmap item.

---

### D-01 — VALUE-POSITION BLINDNESS

**Class.** A builtin or stdlib procedure is correct in call position and defective when
referenced as a first-class value — passed to a higher-order procedure, stored, returned, or
read out of a list.

**Evidence.** LE-01 (builtins had no value representation at all: `Undefined variable:
string<?` for `(sort xs string<?)`, or a foreign-ABI function pointer the closure dispatcher
called with the wrong convention, SIGSEGV at `0x0`; closed by PR #427). SW-27 (a rest-arg
procedure referenced as a value lost its variadic flag, so `(h append '(1) '(2))` answered
`1` — the first argument — silently, exit 0; closed by #427). SW-31 (VM numeric predicates
carried an independent defect on the value route). SW-35 (`floor`/`ceiling`/`truncate`/
`round` as values returned a rational's **heap address** as a number, changing between runs).
SW-36, SW-37, SW-38, SW-39 (found by the sweep itself).

**Why the existing gates cannot see it.** Structural, three ways, all recorded in SW-27's
`missed_by` field:
- the differential corpus compares **execution axes against each other**, and a
  value-position defect is normally wrong the same way on all four axes, so they agree;
- vm-parity excuses native-only programs;
- the arity sweep, the surface-parity probe and the language-coverage floor all exercise
  builtins in **call position only**.

SW-27's escape analysis prescribed the counter-detector in its own words: *"add a
value-position axis to the HOF sweep, not a corpus program."*

**Status.** A counter-detector is **already being built** in the `fix/value-position-axis`
lane: `scripts/run_value_position_sweep.py` plus `tests/value_position/BASELINE.json`. This
roadmap does not duplicate it; it lands it, wires it, and extends it (see item B-01).

---

### D-02 — SHARED-DEFECT BLINDNESS (identical-everywhere-wrong)

**Class.** A defect that is wrong the *same way* on every axis and both engines. Differential
testing cannot see it by construction, because differential testing's oracle is agreement.

**Evidence.** SW-06 (`diff` on a quoted s-expression returned a fabricated `0` on **both**
engines; converted to a loud error by #423). SW-33/hygiene (`syntax-rules` has no referential
transparency; both engines answer `-5` where R7RS 4.3.2 requires `50` — the only one of 18
hygiene-matrix cells still wrong after #440, and wrong identically, so it blocks no parity
row). SW-27 (all four native axes wrong identically). SW-11 (`(the <type> expr)` is a runtime
no-op on both engines).

**Why the reference oracle did not catch these.** The project *has* an external ground-truth
oracle — `scripts/run_reference_differential.sh`, pillar P7a, diffing against chibi-scheme.
It missed them for four independent reasons, each verified:

1. **The corpus is 34 hand-written programs** (`tests/reference-diff/corpus/`), organised by
   R7RS chapter. None crosses two features. None uses a builtin in value position.
2. **There is no VM axis.** The harness runs `ref`, `esh-jit` and `esh-aot` only — zero
   occurrences of any VM invocation in the script. Every VM-side silent-wrong defect
   (SW-24, SW-25, SW-31, SW-34, SW-38, SW-39) is structurally outside its reach.
3. **It runs nowhere.** It is not invoked by any workflow in `.github/workflows/`, and not by
   `scripts/run_all_tests.sh`. It gates only through the ICC trace, which requires a human to
   run it and then run readiness. (This is D-13.)
4. **A lossy normalizer can erase the property under test.** This is documented in the tree:
   `scripts/p8/gen_property_oracles.py`'s own header records that the chibi differential
   missed a 6-significant-figure float-printing defect because its normalizer reformatted
   every float to `%.6g` on both sides before comparing. A differential is only ever as
   strong as the property it compares.

**Partial counter-detectors that exist.** Two, and both are the right shape:

- P8 axis 4 (`scripts/p8/gen_property_oracles.py`) generates reference-free property oracles
  in three families: `numrt` (number->string . string->number identity), `datart` (read/write
  round-trip), `alg` (exact algebraic identities). Three of the roughly ten families the
  campaign's defects would have needed.
- `scripts/gen_generative_corpus.py` (pillar P7c) already **generates** closed, printable,
  total, deterministic R7RS-small programs in two families — `diff` (cross-checked against
  chibi and every Eshkol oracle) and `meta` (self-checking metamorphic properties: apply
  equivalence, map ordering, commutativity, `reverse` involution, `length`/`append`
  homomorphism, let re-association, fold equivalence). Its own header states the motivating
  principle exactly: *"if our system does not constantly expose every single hidden bug then
  it has no coverage."*

So the generator exists, the property harness exists, and the metamorphic families exist. The
gap is (a) which property families are encoded, (b) the missing VM axis on the reference
oracle, and (c) D-13 — `run_generative_differential.py` is invoked by no workflow.

---

### D-03 — EXECUTION COVERAGE VERSUS COMPARISON COVERAGE

**Class.** Any construct that runs on both engines and returns a different value, where no
program has ever compared them.

**Evidence.** `scripts/run_language_coverage.sh` reports **1091 constructs, 100.00%**,
deficit 0, high-risk uncovered 0 — PASS. `scripts/run_engine_parity_coverage.py` reports
**136 of 1114 constructs with differential evidence (12.21%)**, floor 10.95% (PR #424; it was
113/1114 = 10.14% at `9f2da2ab`). Both report green. Executing a construct proves it runs;
comparing it proves it is right. The project measures the first exhaustively and the second
barely.

Ledger entry PR-10 records the consequence: SW-01, SW-02, SW-06 and SW-08 all live in the
87.79% with no differential evidence at all.

**Why the number is stuck where it is — and it is not corpus.** The VM emits per-construct
coverage markers only from `vm_language_coverage_native_dispatch` and `_named_call`, i.e.
from **builtin dispatch**. Special forms (`if`, `quote`, `let`, `cond`, `do`, `lambda`) are
compiled inline in `lib/backend/vm_compiler.c` and emit **no marker at all**, so they can
never earn differential credit no matter how many programs exercise them. Adding VM coverage
markers to special forms is a prerequisite for the number to move, and is the highest-value
follow-up named in the task #109 handoff.

**Second structural note.** Both parity probes grade **non-growth**, not correctness:
`engine_semantic_parity` passes at `10 divergent program(s), 0 new`, and
`surface_parity_probe` at `2506 probed / 318 divergences / 0 new`. A defect present before
the baseline is permanently invisible. A ratchet with no burn-down target is
indistinguishable from acceptance.

---

### D-04 — CROSSING BLINDNESS (the corpus is 1-wise; the defects are 2-wise) [NEW]

**Class.** A defect that exists only at the *interaction* of two features, each of which is
individually well covered.

**Evidence.** This is the single most repeated sentence in the campaign's escape analyses.

- PR #420: *"The differential corpus already had AD programs (`29_ad_derivative`,
  `30_ad_gradient`) and a shadowing program (`39_letrec_shadow`) — but nothing that
  **crossed** them, and the defect lived exactly at the crossing."* (AD capture reconstruction
  ignoring lexical scope.)
- PR #426: *"The corpus had AD programs and it had capture/shadowing programs, but it had no
  `hessian` at all, which is why four independent execution axes never disagreed about a
  program that could not be compiled on any of them."*
- SW-33 (min/max lane): a derivative taken **through** `min`/`max` is silently `0` on the VM.
  AD is covered. `min`/`max` are covered. The crossing was not.
- SW-34: a `do` loop variable captured **and** mutated by a body closure. `do` is covered;
  mutable capture is covered (`let` has had `needs_boxing()` since the beginning); the
  crossing was not — and neither were the `let*`, `letrec` and `letrec*` crossings, which
  #435 closed in the same family.

**Why the existing gates cannot see it.** Every corpus in the tree is a list of programs a
human thought to write. Nothing measures which *pairs* of constructs have been exercised
together, so "we have AD tests and we have shadowing tests" reads as coverage when the
product of the two is empty. Combinatorial interaction coverage is a measurable quantity and
the project measures no version of it.

---

### D-05 — TESTS THAT ANNOUNCE THEIR OWN FAILURE, AND REPRODUCERS THAT GO STALE

**Class.** (a) A committed test prints its own FAIL verdict while every gate stays green.
(b) A filed reproducer silently stops reproducing, training readers to discount the ones that
still do.

**Evidence (a).** SW-24: `tests/parser/test_function_shadowing.esk` contains
`(define + (lambda (a b) (* a b)))` and prints `FAIL: Expected 12` on the VM, out loud, for
months, on a green baseline. The VM answered `7` because its arithmetic opcode fast paths
dispatched on the head symbol alone. Closed by #429.

**Evidence (b).** DD-12: of the 38 filed reproducers under `tests/vm_parity/found/`, **13 now
AGREE and print the correct value — 34% stale**. Three of them (SW-15, SW-16, SW-17) were
carried into the ledger as open SILENT-WRONG entries before a re-run closed them. `found/` is
documented as evidence rather than gate input, so nothing re-runs it.

**Why the existing gates cannot see it.** This one is a near miss, which makes it precise.
**Twenty of the 46 `scripts/run_*_tests.sh` suites already scan for self-printed FAIL lines**
— `run_parser_tests.sh` even carries the comment *"programs print their own FAIL lines and
exit 0, so scan for"*. The scan exists. What does not exist is:
- the same scan on the **VM lane and the wasm lane** — `scripts/run_vm_parity.sh` grades by
  `cmp -s native.out vmsrc.out` alone, so two engines printing the *same* FAIL line pass, and
  a baselined divergence is never read; and
- any re-measurement of `found/` or of `ENGINE_PARITY_BASELINE.json`.

---

### D-06 — INVARIANTS THAT PASS ON EXISTENCE

**Class.** A gate whose severity and name promise a semantic guarantee while its
implementation checks only that a probe ran.

**Evidence.** `.icc/architecture-model.yaml:312`:

```yaml
  - id: INV-engine-semantic-parity
    kind: exercise
    incident: vm-parity-silent-wrong-answer
    severity: critical
    description: >
      Every corpus program must compute the SAME ANSWER under native LLVM and
      under the bytecode VM, and the differential construct-coverage fraction
      must hold its recorded floor.
    window_days: 30
```

The description states a threshold. The `kind` is `exercise`, which grades PASS on
`capability 'engine_semantic_parity' exercised within 30d (1 event(s))` — an existence check
that would pass identically at 1% coverage, or on an event whose payload said FAIL.
`INV-language-surface-exercise` has the same shape. Ledger entry PR-11.

**Why the existing gates cannot see it.** Nothing audits the *kind* of an invariant against
the strength of its description. ICC's `vacuous-assertions` detector is the right family
(checks satisfiable by construction) and currently reports 6 findings, 1 high — but it
operates on source assertions, not on the invariant registry.

---

### D-07 — SILENT GATE DEGRADATION

**Class.** A gate that quietly starts measuring less, while continuing to print a score.

**Evidence, live at the time of writing.** `.icc/completion-oracles.yaml` **does not parse on
master**:

```
yaml.parser.ParserError: while parsing a block mapping
  in ".icc/completion-oracles.yaml", line 217, column 9
expected <block end>, but found '<block mapping start>'
  in ".icc/completion-oracles.yaml", line 223, column 11
```

Introduced by #429 (`2ae3787f`), which added its `higher_order_shadowing_oracle` criterion but
omitted the preceding criterion's `action:` field and the new criterion's `- runtime_event:`
list-item opener. Ledger entry PR-12 records the effect: `icc readiness` **silently fell back
to a generic 2-criterion oracle that always scores 100**. The repaired file carries 41
criteria (PR #436, unmerged, 3-line diff).

So for the duration: a release gate designed around 41 criteria was grading 2, and reporting
ready.

**Why the existing gates cannot see it.** There is no schema or parse check over `.icc/*.yaml`
in any workflow, and `icc readiness` reports a score without reporting how many criteria
produced it. A criterion count that silently drops from 41 to 2 is the most important number
in the run and it is not printed as a delta anywhere.

---

### D-08 — BEHAVIORAL BLINDNESS IN ICC

**Class.** Every ICC correctness detector is a *static* detector — it reads symbols, text and
structure. None of them runs the compiler and looks at an answer.

**Evidence.** The correctness chain caught 0 of 62. Specifically:
- `contradictions` produced 30 findings, every one a `constant_disagreement` or
  `env_default_disagreement` on a build/tuning symbol. It flags that `ESHKOL_TIMEOUT_MS` has
  two different defaults and misses that **the variable does nothing** (SW-10). It reports
  nothing where `KNOWN_ISSUES.md:45` contradicts `tests/ad_oracle/README.md` (DD-04), where
  `KNOWN_ISSUES.md:379` contradicts the shipped `PARITY.tsv` (DD-03), or where
  `KNOWN_ISSUES.md:269` contradicts the VM's measured behaviour (LE-05).
- `duplicate-implementations` is a **lexical clone detector**. It ranked 25-36 entities,
  most of them in `tests/`, and did not flag SW-12 — two independent backward implementations
  of the same six tensor ops, in two files, that are not textually similar.

Structurally: ICC's built-in `correctness` chain has exactly **two** members —
`contradictions` and `duplicate-implementations` — both static, and their corroboration rate
on this repo is **0%** (they never flag the same entity). By contrast `engine-health` (four
members) corroborates at 34%, and path-scoped at 41%. The correctness chain is the thinnest
chain ICC ships and it is the one named for the property this project cares about most.

**Why this is the highest-leverage ICC gap.** The maintainer's stated intent is that ICC
serves algorithmic optimisation *and* flaw detection. It currently serves the first. The
missing detector class is behavioral probes: detectors whose evidence is *executed output*,
not source text. This is task #120's core.

**The scaffolding already exists in ICC and is unused here.** A behavioral detector does not
need to be built from nothing:

| ICC facility | What it does | What it would serve |
|---|---|---|
| `capture-baseline` / `verify-baseline` | Runs a command template with `{input}` across an input glob, stores stdout or its hash with `--canonicalize PATTERN=>REPL`, re-runs and diffs, **exit 2 on mismatch** | A ready-made golden differential harness; the missing engine for D-14's evidence-bearing ratchets |
| `numeric-baseline` / `numeric-diff` | Per-tensor capture and divergence check, exit 2 on divergence | T3 — AD gradients as a gated numeric baseline rather than hand-written analytic checks |
| `invariants --emit-trace` | Named properties as a ratchet whose predicates read the detectors' own JSON, so an invariant cannot drift from its measurement; emits `invariant_ratchet` events, so *a guard having run* is a fact on disk | D-06 and D-14; the registry is absent (task #112) which is why this is dead today |
| `execution-source-gate` | Every matched test event must record the execution source it claimed; missing provenance is NO_DATA plus a non-zero exit — a skip-disguised-as-a-pass detector | D-05 and D-06 directly |
| `find-stubbed-prod-paths` | Classifies paths including **`not_exercised_by_real_smoke`** | The "this probe certifies nothing" class |
| Detector I / J / K / L contract | ICC's lettered standing-detector series: persist a report, emit a standing-detector event the completion oracle gates on, appear on the `ops-status` board | The shape a new detector must take to be a first-class ICC citizen — the next one is **Detector M** |

The behavioral-probe plane is therefore less "build a new subsystem" and more "add lettered
detectors on the existing contract, and give the `correctness` chain semantic members".

---

### D-09 — DOC-CLAIM VERSUS BEHAVIOR

**Class.** A documented control, guarantee or limitation that the binary does not implement —
in either direction.

**Evidence.** SW-10: seven documented resource-limit variables (`ESHKOL_MAX_HEAP`,
`ESHKOL_TIMEOUT_MS`, `ESHKOL_MAX_STACK`, `ESHKOL_MAX_TENSOR_ELEMS`, `ESHKOL_MAX_STRING_LEN`,
`ESHKOL_ENFORCE_LIMITS`, `ESHKOL_LIMIT_WARNINGS`) parsed into the active configuration and
consulted by nobody; `ESHKOL_TIMEOUT_MS=500` printed *"Execution timeout: 500ms limit
exceeded"* and then ran the program to completion, exit 0, because nothing polled the
interrupt flag the watchdog set. Closed by PR #430. DD-10: `-D NAME[=VALUE]`, `-fPIC` and
`provide` are documented no-ops. LE-05: `KNOWN_ISSUES.md:269` says `(load "path.esk")`
"silently ignores a path literal" where the VM is in fact loud — the doc is wrong in the
*other* direction, which no doc-truth pass looking for over-claims would find.

**Why the existing gates cannot see it.** ICC's `doc-typed-claims` is the strongest doc
detector in the tree (17,251 claims, 151 wrong, 91 numeric) but it binds claims to *symbols
and numbers*, not to *behaviour*. The doc-example harness merged as #411 was built for this
and DD-07 records three defects that make it miss the files it checks (`EXPECT_RE` matches
only `=>`, `collect()` skips only blank lines between fences, neither runner sets
`ESHKOL_JIT_CACHE=0`). Nothing continuously executes a documented control and asserts the
documented effect.

---

### D-10 — STRUCT-LAYOUT AND ABI LIES

**Class.** Two declarations that claim to describe the same memory and do not; or a value
representation that misdescribes the shape of the thing it points at.

**Evidence.** SW-13: `lib/backend/vm_macro.c` casts between `MacroNode*` and the VM hub's
`Node*`. `sizeof(Node)` is **176**; `sizeof(MacroNode)` is **160**. `is_char` aliased the
children-array capacity, and `is_inexact`/`is_int`/`ival` were read **up to 16 bytes past the
allocation**. On 64-bit hosts those bytes happened to read zero; on wasm32 they did not, so
the browser build folded `(* 2 21)` from heap garbage and printed a garbage bignum — a wrong
answer shipping in the released web build while every native gate was green.

The ABI half of the same class: SW-27 (the value-position representation packed a plain arity
and no variadic bit, so the dispatcher called a rest-arg procedure as if fixed-arity) and
LE-01 (a raw `Function*` with a foreign ABI handed to the closure dispatcher).

**Why the existing gates cannot see it, verified.** There are **zero** `static_assert`s on
`sizeof` or `offsetof` anywhere in `lib/` or `inc/` (6 `static_assert`s total, none about
layout). `MacroNode` is *still* a separate struct declaration on master
(`lib/backend/vm_macro.c:41`); the `typedef struct Node MacroNode;` fix lives on the unmerged
#432. ICC's `odr-audit` detects same-name different-shape C-family typedefs (18 collisions, 2
high) but these two types have **different names**, so it is outside its rule.

The reproduction route is the roadmap item: #432 reproduced this **natively, with no wasm
toolchain**, by building the VM standalone under `-fsanitize=address`. That build does not
run in CI.

---

### D-11 — BUILD FRESHNESS

**Class.** A harness reporting a verdict that belongs to a binary other than the one under
test.

**Evidence.** Recorded twice in the campaign, and broadcast to all lanes as a standing trap:
*"Rebuild before believing any harness after a rebase"* — a stale binary produced a false
`engine_semantic_parity` FAIL on #418's new corpus file.

**Why the existing gates cannot see it, verified.** Freshness or fingerprint checks appear in
`scripts/run_icc_smoke.sh` and `scripts/gate_no_silent_wrong.py`, and in three remote/preflight
scripts. They appear **zero** times in `run_differential.sh`, `run_vm_parity.sh`,
`run_engine_parity_coverage.py` and `run_reference_differential.sh` — the four harnesses whose
verdicts the release depends on most.

**Precedent that already exists.** `scripts/run_p8_escape.sh` solved this properly and
documents why: *"This suite shells out to eshkol-run for many minutes; a rebuild in the same
worktree mid-run used to swap the compiler underneath it and produce verdicts (including
crashes) that belong to no single build. Run against a private copy instead."* That preamble
is the reusable artifact.

---

### D-12 — LEDGER DISCIPLINE AS CODE

**Class.** The ledger is the project's memory of what is wrong. Concurrent lanes corrupt it
in ways nothing detects.

**Evidence, all verified across branches.**

- **ID collision.** `SW-33` is allocated to **three different defects** across four lanes:
  `fix/native-rest-capture-and-do-loop` uses it for "native mutated + captured REST parameter
  does not write back"; `fix/numeric-tower-bignum-tail` and `fix/value-position-axis` both use
  it for "a derivative taken through min/max is silently 0 on the VM";
  `fix/syntax-rules-hygiene` uses it for "syntax-rules templates have no referential
  transparency". Three defects, one ID, no detector.
- **Entry loss on concurrent edit.** Commit `c5d6a672` on `fix/value-position-axis` is
  literally titled *"docs(ledger): restore PR-12, clobbered while closing SW-32"*. An entry
  was silently deleted by a merge and restored only because a human noticed.
- **Closure by assertion.** Guarded, but only partly: PR #421's grader prints entries
  `CLOSED WITHOUT A RE-MEASUREMENT SHA` (SW-06, SW-09, SW-09b, SW-22) so the weaker form stays
  visible. This is the right pattern and should be a hard failure after a grace period, not a
  printed note.
- **Merge state inverts the ledger.** At `9f2da2ab`, every defect the audit escalated was in
  the cut and every fix sat outside it on an unmerged branch. No criterion notices that a
  repo's known defects are documented on branches that have not merged.

**Why the existing gates cannot see it.** `scripts/gate_no_silent_wrong.py` validates the
schema of the ledger it is given. It cannot see a *second* ledger on another branch, an ID
used twice, or an entry that existed yesterday and does not exist today.

---

### D-13 — GATES THAT RUN NOWHERE [NEW]

**Class.** Not a detector gap — a *delivery* gap. It multiplies every other gap in this
catalog.

**Evidence, measured.** Across `.github/workflows/` (`ci.yml`, `adversarial-nightly.yml`,
`gpu-execution-gate.yml`, `identity-guard.yml`, `pages.yml`, `release.yml`), only 14 distinct
harness scripts are invoked at all. The following are invoked by **no workflow** and by
**`scripts/run_all_tests.sh`** neither:

| Harness | What it is |
|---|---|
| `run_differential.sh` | the four-axis native differential corpus (48 programs) |
| `run_reference_differential.sh` | the only external ground-truth oracle in the tree (P7a, chibi) |
| `run_engine_parity_coverage.py` | the differential construct-coverage number and its floor |
| `run_icc_smoke.sh` | the 58-probe release harness |
| `gate_no_silent_wrong.py` | the silent-wrong release gate |
| `run_metamorphic.sh` | metamorphic relations |
| `run_generative_differential.sh` | the generative differential oracle |
| `run_surface_parity.py` | the 2506-probe surface resolution parity |
| `run_sdnc_oracle.sh`, `run_metaprog_depth.sh`, `run_tensor_collection_depth.sh` | depth-parametric suites |
| `run_differential_fuzz.sh`, `run_sanitizer_fuzz.sh`, `run_edge_matrix.sh` | fuzz and edge-matrix |

Several of these `exit 0` by design and gate only through the ICC trace bundle, so the gate
fires only when a human runs the harness *and then* runs readiness in the same tree.

**Consequence.** The strongest detectors the project owns are the least frequently run. Every
build item below is worth a fraction of its value until this is fixed, which is why it is
priority one.

---

### D-14 — BASELINES THAT RECORD NO EVIDENCE [NEW]

**Class.** A ratchet file that records *that* something diverges without recording *what*, so
a real wrong answer is indistinguishable from cosmetic drift.

**Evidence.** PR-01: `tests/vm_parity/ENGINE_PARITY_BASELINE.json` stores **program paths
only** — no operation, no per-engine output. Nothing in the repo can reconstruct what diverges
without a rebuild and a re-run. When all 10 were re-run at `9f2da2ab`: **nine diverge only in
output formatting** (the VM's newline-per-call behaviour splits lines differently) and
**exactly one is a real value divergence** — `test_function_shadowing.esk`, which became SW-24.
The baseline conflated cosmetic drift with a wrong answer and hid the wrong answer among nine
harmless rows.

**Contrast, done right, on the in-flight `fix/value-position-axis` branch.**
`tests/value_position/BASELINE.json` requires
every entry to name the ledger ID it stands for and says so in its own `_comment`: *"Every
entry must carry a reason naming the ledger ID it stands for — an entry without one is
indistinguishable from hiding a bug. NEVER regenerate this file to turn a red gate green."*
That is the schema every ratchet in the tree should have.

---

### D-15 — SECOND IMPLEMENTATIONS OF ONE RULE [NEW]

**Class.** The same semantic rule implemented twice, in two places, that drift apart. This is
the single most common *root cause* in the campaign, and it has no detector.

**Evidence, from the campaign's own root-cause sections.**

| Defect | The two implementations |
|---|---|
| SW-32 | `min`/`max` carried a private **two**-tier ordering copy where `compare()` (behind `< > = <= >=`) runs **four** tiers. Both engines wrong, in opposite directions, for exact integers past 2^53 |
| SW-09 / #430 | The VM has **two** dispatch implementations (switch-based and threaded computed-goto). They disagreed about `OP_ADD` over i128, and about `ESHKOL_VM_MAX_INSN` — the computed-goto path every GCC/Clang build takes had **no instruction counter at all** |
| SW-12 | Six tensor ops carry **two independent backward implementations** (`lib/backend/tensor_backward.cpp` vs `lib/bridge/tensor_backward.cpp`) with no differential test between them |
| LE-01 / #427 | `codegenVariable` carried a hand-maintained cascade of per-builtin wrapper factories, **each re-implementing the operation a second time in IR** |
| SW-26 / #433 | `vector-ref`/`vector-set!`/`vector-length` type-checked `VAL_VECTOR` only, at **six** call sites across two dispatch loops |
| #420 / #426 | `resolveGradientCaptures` was corrected for ESH-0070 and **three** other call sites were missed; a **fourth** (`hessianJetPath`'s scalar arm) was missed again |
| #430 | `ESHKOL_MAX_STACK`: *"Two mechanisms, same default, no wire between them"* — codegen compared against a hard-coded `100000` unrelated to the configurable limit |

The fixes converged on one prescription, stated repeatedly: collapse to one implementation
and route every caller through it. #427: *"The authoritative call-site lowering **is** the
wrapper body, so a first-class reference cannot disagree with a direct call."* #426: *"Using
the shared resolver rather than a local reconstruction also means this site inherits the
corrections that were made to it ... instead of becoming a fourth copy of the same loop that
has to be fixed again later."*

**Why the existing gates cannot see it.** ICC's `duplicate-implementations` is lexical. None
of the pairs above are textually similar; several are in different languages (`.cpp` versus
`.c`) or different dispatch idioms. A semantic duplicate is exactly the dangerous kind, and it
is precisely the kind the detector cannot see.

---

### D-16 — ESCAPE ANALYSIS IS PROSE, NOT A GATE [NEW]

**Class.** The project already produces excellent escape analyses. Nothing records them,
nothing checks that the detector they prescribe was built, and nothing notices when one is
skipped.

**Evidence.** The doctrine already exists as a standing project rule: *every reported bug gets
an escape analysis — why did our own framework miss it — and that analysis leads to a new
generator axis or gate, not merely a point regression test*. The P8 escape-closure pillar
(`scripts/run_p8_escape.sh`) is that doctrine executed once and executed well: eight axes, each
one targeting a specific class of defect that had previously escaped, each documented in the
runner's own header with the escape it closes.

Since then it has been a per-PR courtesy. Every PR from #420 to #440 carries an escape
analysis in its **body**, several of which prescribe a detector in words. Exactly **one** of
those prescriptions became a gate (SW-27's "add a value-position axis", built as
`run_value_position_sweep.py`), and only because a human remembered it three PRs later. The
others are in GitHub comment text and nowhere else.

**Why the existing gates cannot see it.** PR bodies are not machine-readable input to
anything. There is no pull-request template requiring the analysis, no CI check requiring the
section, and no ledger of prescribed-but-unbuilt detectors.

---

## 4. Counter-detector designs

One row per gap. **Home** is where the detector lives: an ICC detector, a repo gate script, a
CI lane, or a corpus generator. **Cost** is engineer-days for a working, retro-catch-validated
first version: S = 1-2d, M = 3-5d, L = 6-12d, XL = a campaign.

| Gap | Counter-detector | Home | Cost | Seed artifact |
|---|---|---|---|---|
| D-01 | **Value-position axis.** One generated program per builtin evaluating the same call twice — call position and through a higher-order procedure — compared with `equal?` inside the program. Differential by construction: no hard-coded expectation, so it cannot pass by agreeing with a wrong answer. Extend to the VM axis, to `.esk` stdlib exports, and to `stored`/`returned`/`mapped` probes | repo gate + corpus generator | S (land) + M (extend) | `scripts/run_value_position_sweep.py` and `tests/value_position/BASELINE.json`, both on `fix/value-position-axis` |
| D-02 | **Property-oracle family expansion.** Add families to P8 axis 4 and to the P7c `meta` generator for the classes this campaign proved dangerous: `scope` (a binding must shadow a same-named global on every route: call, HOF, AD operand, `set!` target), `order` (min/max/sort must agree with the engine's own `<` on the same operands, in both operand orders), `exact` (the defining inequalities of `floor`/`ceiling`/`truncate`/`round`, checked not tabulated), `hygiene` (the 18-cell matrix as generated properties), `adcompose` (nested-differentiation identities). Separately: **add a VM axis to the reference differential**, and **audit every normalizer** for the property it erases | corpus generator + repo gate | M per family; M for the VM axis | `scripts/p8/gen_property_oracles.py` (3 families); `scripts/gen_generative_corpus.py` `meta` family (7 properties); PR #439's tests 63/64/65 are three of these hand-written; `scripts/run_reference_differential.sh` |
| D-02 | **Equivalence modulo inputs (EMI).** Take a program that runs green, mutate code that the observed execution never reaches (Orion) or that runs but cannot affect the printed result (Hermes), and require byte-identical output. This is the canonical answer to shared-defect blindness in the compiler-testing literature and it needs **no reference implementation and no expected value** — the original program is its own oracle. The generator this needs already exists | corpus generator | M | `scripts/gen_generative_corpus.py` (program generator) plus the per-construct coverage trace (which lines executed) already emitted by both engines |
| D-03 | **Comparison-coverage climb.** (i) Emit VM language-coverage markers from the special-form compile paths in `vm_compiler.c` — without this the fraction is capped. (ii) Convert the floor from "do not regress" to a **published rising schedule** with a date per step. (iii) Report `constructs_without_differential_evidence` as a first-class number in the readiness output | repo gate + oracle criterion | L | `scripts/run_engine_parity_coverage.py` (floor mechanism already built); task #109 |
| D-04 | **Pairwise crossing coverage.** Instrument the existing per-construct coverage records to emit *co-occurrence* pairs per program, then report 2-wise interaction coverage over a declared high-risk construct set (AD operators x binding forms x higher-order builtins x numeric tower x macro forms). A generator fills the empty cells. The measurement alone is valuable before any generator exists: it turns "we have AD tests and shadowing tests" into a number | repo gate, then corpus generator | M (measure) + L (generate) | the coverage trace format already emitted by both engines (`P`/`V` records); `scripts/gen_generative_corpus.py` is the generator to extend, not to replace |
| D-05 | **Self-verdict scanner, everywhere.** Extract the FAIL-line scan that 20 suites already carry into one shared helper, and apply it on **every** lane including VM, wasm and AOT. Any test whose stdout contains a self-reported failure is a gate failure regardless of exit status or cross-engine agreement. Plus: **re-run `tests/vm_parity/found/` inside the parity gate** and fail on a reproducer that no longer reproduces (it is either fixed — close the ledger entry — or the reproducer rotted) | repo gate | S | `scripts/run_parser_tests.sh` lines 83-88; DD-12 |
| D-06 | **Threshold-bearing invariants.** Add an invariant kind that grades a *payload field* against a bound, and convert `INV-engine-semantic-parity` and `INV-language-surface-exercise` to it. Then add a meta-check: any invariant whose `severity` is `critical` and whose `kind` is `exercise` is itself a finding | ICC detector + `.icc/architecture-model.yaml` | M | `kind: perturbation` and `kind: key-space-equality` already exist in the model; ICC `vacuous-assertions` is the adjacent detector |
| D-07 | **Gate self-verification.** (i) A CI job that parses and schema-validates every `.icc/*.yaml` on every PR — this is a ten-line job that would have caught PR-12 at review time. (ii) `icc readiness` must print `criterion_count` and its delta against the previous run, and **fail loudly** on a drop rather than scoring the remainder. (iii) A trace-bundle assertion that the criteria graded are the criteria declared | CI lane + ICC | S (i), M (ii-iii) | `icc readiness-diff` already computes per-criterion added/removed; PR #436 |
| D-08 | **The behavioral-probe plane for ICC.** A detector class whose evidence is executed output rather than source text, built as lettered detectors on the existing Detector I-L contract. Minimum viable set, each modelled on a defect this campaign found: (a) **declared-versus-observed** — for every documented env var / CLI flag, run a program with and without it and assert the documented difference (would have caught SW-10); (b) **implementation-pair differential** — given two symbols declared to implement one rule, run both on generated inputs and compare (would have caught SW-12 and SW-32); (c) **route-equivalence** — the same call reached by two lowering routes must agree (would have caught LE-01, SW-27, SW-35); (d) **claim-execution** — every doc code fence executed and its stated output asserted (D-09). Then **add the semantic members to ICC's `correctness` chain**, which today is two static detectors corroborating at 0% | ICC detector plane | XL (the plane) / M per probe class | task #120; `capture-baseline`/`verify-baseline`, `numeric-diff`, `execution-source-gate`, `invariants --emit-trace` and the Detector I-L contract are all shipped and unused here; `icc runtime-evidence` already ingests 4,437 events across 27 kinds; `.icc/architecture-model.yaml` `fidelity: runtime` invariants are the in-repo precedent |
| D-09 | **Executable documentation claims.** Every documented *control* (env var, flag, `provide`, `(the …)`) gets a claim record with a runnable assertion; the gate executes them all and fails on any claim whose asserted effect does not occur. Bidirectional: a doc that claims a limitation the binary does not have is equally a finding (LE-05) | repo gate + ICC `doc-typed-claims` extension | M | `scripts/doc_audit/` and the #411 harness (fix DD-07's three defects first); for the waiver discipline, the same shape as `.icc/silent-wrong-ledger.yaml` — an allowlisted claim carries an owner and an expiry, so it is scaffolding rather than a tolerance |
| D-10 | **Layout and ABI contracts.** (i) `static_assert` on `sizeof` and `offsetof` for every struct pair that is cast between — starting with `Node`/`MacroNode`, and generalised as a rule: a cast between two named struct types requires a layout contract in the same translation unit. (ii) An ICC detector for cross-type casts lacking such a contract (a natural sibling of `odr-audit`, which is name-based and therefore blind here). (iii) **An ASan VM-standalone lane in CI** — that is how #432 reproduced a wasm-only defect natively, in seconds | repo gate + ICC detector + CI lane | S (i), M (ii), S (iii) | #432's ASan reproduction recipe; `icc odr-audit` |
| D-11 | **Build fingerprint preamble.** One shared preamble that records the SHA, the build directory mtime and a hash of the binary under test into every emitted event, refuses to run against a binary older than the working tree, and pins a private copy for the duration of the run. Adopt in `run_differential.sh`, `run_vm_parity.sh`, `run_engine_parity_coverage.py`, `run_reference_differential.sh` first | repo gate (shared library) | S | `scripts/run_p8_escape.sh`'s pin-the-binary preamble, verbatim |
| D-12 | **Ledger protocol as code.** (i) **ID allocation**: IDs are reserved by appending to a single append-only allocation file on master before a lane uses one; the gate fails on a duplicate ID, and on a lane-local ID absent from the allocation file. (ii) **No-loss merge check**: the gate fails if the ledger on a PR branch has fewer entries than its merge base, unless each removal is justified in the PR body. (iii) **Expiry enforcement** already exists; add: closure without a `closed_at` SHA becomes a hard failure after a stated grace date. (iv) **Unmerged-fix awareness**: a criterion that reports how many ledger entries are closed only on unmerged branches | repo gate extension | S (i-iii), M (iv) | `scripts/gate_no_silent_wrong.py` (schema validation, fail-closed, and the `CLOSED WITHOUT A RE-MEASUREMENT SHA` report already exist) |
| D-13 | **Run the gates.** A `correctness-nightly` workflow invoking every trace-gated harness on a real runner, uploading the trace bundle, and running `icc readiness` on the result — so the trace-gated suites gate something. A `correctness-pr` subset (fast axes only) on every PR. Every harness that `exit 0`s by design must be paired here with the readiness step that reads its trace | CI lane | M | `adversarial-nightly.yml` is the working template (it already runs P8, TSan and packaging on a schedule) |
| D-14 | **Evidence-bearing ratchets.** A shared schema for every baseline file: per entry, the ledger ID it stands for, the per-engine normalized outputs (or their hashes), and a classification of `cosmetic` versus `value`. A `value` classification is never an acceptable baseline entry — it is a ledger entry. Migrate `ENGINE_PARITY_BASELINE.json`, `SURFACE_BASELINE.tsv`, `arity_parity_baseline.json`, `five_way_baseline.json`, `PARITY.tsv`, `EXCLUSIONS.tsv` | repo gate + schema | M | `tests/value_position/BASELINE.json`'s `_comment` is the schema statement; PR-01..PR-08 name the six files |
| D-15 | **Semantic-duplicate detector (ICC Detector M).** Two complementary halves: (a) a **declarative registry** — `.icc/implementation-pairs.yaml` naming every pair of symbols that implement one rule, with a required differential test per pair, gated; and (b) an ICC detector on the Detector I-L contract that *proposes* candidates for the registry by structure rather than text (two functions reachable from one dispatch name; two definitions of the same builtin id; a switch arm and a computed-goto arm of the same opcode; two `_backward` symbols for one op). (b) is research; (a) is buildable now and would have gated SW-12, SW-32, SW-09 and LE-01 | repo gate (a) + ICC detector (b) | M (a), L (b) | PR #433's `tensor_backward_dual_impl_gradcheck_test.cpp` is a working instance of (a) — one pair, native versus bridge, with central finite differences of an independently written third forward as the tie-breaking oracle. `duplicate-implementations` already ranks 80-99% divergent copies first, which is the right ranking on the wrong (lexical) similarity measure |
| D-16 | **Escape analysis as a gate.** (i) A PR template with a required `## Escape analysis` section answering *why did no gate catch this* and choosing one of: extended detector X, filed detector build-item Y, or justified N/A. (ii) A CI check that fails a PR touching `lib/`, `inc/` or `exe/` whose body lacks the section. (iii) `.icc/detector-backlog.yaml` — every prescribed-but-unbuilt detector, with the defect IDs that motivated it, gated the same way the silent-wrong ledger is: growth is allowed, silence is not | CI lane + repo gate | S (i-ii), S (iii) | `.icc/silent-wrong-ledger.yaml` (the gating pattern to copy: fail-closed, owner and expiry per entry, closure only by measurement); `scripts/run_p8_escape.sh` (an existing pillar whose axes are each named for the escape they close, i.e. the format an entry should take) |

---

## 5. Prioritized sequence

The ordering principle: **multipliers first** (things that make every other detector fire),
then **cheap high-yield** (things already 80% built), then **structural** (things that need
design), then **the plane** (ICC behavioral probes, which is a v1.4 campaign).

This sequence is designed to compose with, not compete against, the work already scheduled:
the **VM evacuator flagship** (SW-14, ruled the v1.3.5 flagship), the **engine-parity
campaign** (#109), the **ICC methodology registries** (#112), and the **ICC behavioral/perf
plane** (#119/#120 at v1.4 kickoff).

### v1.3.5 wave 1 — multipliers and the nearly-built (target: with the v1.3.5 cut)

| # | Item | Gap | Cost | Why now |
|---|---|---|---|---|
| **B-01** | **`correctness-nightly` + `correctness-pr` CI lanes.** Run the trace-gated harnesses on a real runner, upload the trace bundle, run readiness on it | D-13 | M | Multiplier on all sixteen. The differential corpus, the reference oracle, the parity-coverage floor, the silent-wrong gate and the 58-probe smoke harness currently fire only when a human runs them by hand |
| **B-02** | **Land the value-position sweep and wire it to the oracle.** Merge `run_value_position_sweep.py` + `tests/value_position/BASELINE.json`; add a `value_position_sweep_clean` criterion to `eshkol-compiler-readiness` | D-01 | S | Already built and already producing: five new defects (SW-35..SW-39) on its first run. The highest yield-per-day item in this document |
| **B-03** | **`.icc/*.yaml` parse-and-schema CI job, and criterion-count reporting in readiness.** Fail on a parse error; fail loudly on a criterion-count drop | D-07 | S | A release gate graded 2 of 41 criteria and printed ready. Ten-line job. Must land before the v1.3.5 readiness stamp means anything |
| **B-04** | **Self-verdict scanner on every lane, and `found/` re-measurement in the parity gate** | D-05 | S | SW-24 printed `FAIL` out loud for months; 34% of filed reproducers are stale. The scan already exists in 20 suites — this generalises it |
| **B-05** | **Build-fingerprint preamble in the four release-critical harnesses** | D-11 | S | Two false verdicts already attributed to this. The preamble already exists in `run_p8_escape.sh` |
| **B-06** | **Ledger protocol: ID allocation file, no-loss merge check, closure-without-SHA hard fail** | D-12 | S | `SW-33` currently names three different defects. With 5+ concurrent lanes this gets worse, not better |
| **B-07** | **Escape-analysis PR template + CI check + `.icc/detector-backlog.yaml`** | D-16 | S | This is the item that makes the roadmap self-extending. Costs almost nothing and captures every future escape analysis instead of losing it to PR comment text |
| **B-08** | **Layout contracts + ASan VM-standalone CI lane** | D-10 | S | Zero layout `static_assert`s exist today, and the `Node`/`MacroNode` aliasing is still on master. #432's ASan recipe reproduces a wasm-only wrong answer natively in seconds |

Wave 1 is eight items, seven of them S. It is deliberately shaped so that the expensive
v1.3.5 flagship (the VM evacuator) is not competing with it for design attention — wave 1 is
mostly wiring, not design, and the evacuator gets a direct benefit from B-01, B-05 and B-08.

### v1.3.5 wave 2 — structural (target: v1.3.5, may slip to v1.4 without blocking the tag)

| # | Item | Gap | Cost |
|---|---|---|---|
| B-09 | **Property-oracle families**: `scope`, `order`, `exact`, `hygiene`, `adcompose` added to P8 axis 4 as generators; PR #439's hand-written tests 63/64/65 promoted into them | D-02 | M per family |
| B-10 | **VM axis on the reference differential**, plus a normalizer audit (what property does each normalization erase?) | D-02 | M |
| B-10b | **EMI / equivalence-modulo-inputs axis** over the existing P7c generator: mutate unreached code (Orion) or result-irrelevant live code (Hermes) and require byte-identical output. No reference, no expected value | D-02 | M |
| B-11 | **VM special-form coverage markers** + a published rising floor schedule for differential construct coverage, and `constructs_without_differential_evidence` in the readiness output | D-03 | L |
| B-12 | **Evidence-bearing ratchet schema**, and migration of the six named baseline files | D-14 | M |
| B-13 | **`.icc/implementation-pairs.yaml`** registry with a mandatory differential test per pair; seed it with the seven pairs named in D-15 | D-15 | M |
| B-14 | **Threshold-bearing invariant kind**, and conversion of `INV-engine-semantic-parity` and `INV-language-surface-exercise` | D-06 | M |
| B-15 | **Executable documentation claims** for every documented control (fix DD-07's three harness defects first) | D-09 | M |

### v1.4 — the detector plane

| # | Item | Gap | Cost |
|---|---|---|---|
| B-16 | **Pairwise crossing coverage**: measure first (2-wise interaction coverage over the high-risk construct set), then generate into the empty cells | D-04 | M + L |
| B-17 | **ICC behavioral-probe plane** (task #120): declared-versus-observed, implementation-pair differential, route-equivalence, claim-execution | D-08 | XL |
| B-18 | **Semantic-duplicate candidate detector** in ICC — structural rather than lexical clone detection, proposing entries for B-13's registry | D-15 | L |
| B-19 | **ICC registries** (`.icc/invariants.yaml`, `program_state.yaml`, ADR store, default-off gate ledger) — task #112; these convert nine currently-dead ICC commands into live ones and are a prerequisite for B-17 | D-08 | L |

### What this sequence deliberately does not do

- **It does not propose a new fuzzer.** `run_differential_fuzz.sh`, `run_sanitizer_fuzz.sh`
  and `run_reader_fuzz.sh` exist. Their problem is D-13 (they run nowhere), not capability.
- **It does not propose replacing the ratchets with hard gates.** A ratchet with an
  evidence-bearing schema (B-12) and a burn-down target is a good instrument. A ratchet
  without them is acceptance in disguise.
- **It does not propose more corpus programs as a primary answer.** Every gap above is
  structural: more programs written by the same hands, exercising the same features one at a
  time, would not have found any of D-01, D-02, D-04 or D-15.

---

## 6. The standing escape-analysis protocol

This is the durable output of the roadmap. Everything in section 5 is a list of things to
build; this is the mechanism that keeps producing the list after they are built.

### The rule

> **Every PR that fixes a defect in `lib/`, `inc/` or `exe/` must answer, in its body: why did
> no gate catch this? And it must close that answer with one of three outcomes.**

The three permitted outcomes, in order of preference:

1. **EXTENDED** — a named existing detector was extended so it now catches the class. Cite the
   detector and the retro-catch: the detector, run at the pre-fix SHA, must go red.
2. **FILED** — the class needs a detector that does not exist. File it in
   `.icc/detector-backlog.yaml` with the defect IDs that motivate it, an owner and a target
   release. A filed item is a debt with a name, not a shrug.
3. **N/A** — with a stated reason. The only acceptable reasons are: the defect is not a class
   (a genuine one-off), or an existing detector *would* have caught it and did not run (in
   which case the finding is against D-13 and the PR says so).

"Nobody thought of it" is not an outcome. Every defect in this campaign was thought of by
somebody, eventually; the question the protocol asks is what will think of it next time.

### The format

The section is prose, but it must contain three facts, because these are the three that make
it actionable later:

- **The crossing.** What two things had to be true simultaneously? (`#420`: AD operator x a
  lambda capturing a parameter that shadows a global. `#426`: `hessian` x a capturing lambda.
  `SW-34`: `do` binding x mutable capture.) This is what makes D-04 buildable.
- **The blind axis.** Which oracle *structurally* could not have seen it? (Agreement-based —
  it was wrong identically everywhere. Call-position-only. Native-only. Non-growth ratchet.)
  This is what routes the finding to the right gap in section 3.
- **The retro-catch.** At the pre-fix SHA, does the detector you extended or propose go red?
  If it does not, it is not a detector for this defect.

### The enforcement

Three pieces, all in wave 1 (B-07):

1. A `.github/PULL_REQUEST_TEMPLATE.md` carrying a required `## Escape analysis` section with
   the three facts above as prompts.
2. A CI check fails a PR that touches `lib/`, `inc/` or `exe/` and whose body has no
   `## Escape analysis` section containing one of the three outcome keywords
   (`EXTENDED` / `FILED` / `N/A`).
3. `.icc/detector-backlog.yaml`, gated by a script modelled directly on
   `scripts/gate_no_silent_wrong.py`: fails closed on a missing or unparseable file, entries
   carry an owner and a target release, an entry past its target release without an extension
   is a finding. Growth is expected and healthy — a lane that files three detector items has
   done its job. Silence is the failure mode.

### Why this is worth the friction

The campaign already ran this protocol informally and it worked: the fix lanes that
class-killed rather than point-fixing produced SW-25 through SW-39, LE-09, LE-10 and PR-12 —
roughly a fifth of everything in the ledger — as a *side effect* of fixing something else.

The cost of the protocol is one paragraph per fix PR. The cost of not having it is visible in
this document: nineteen PRs' worth of excellent escape analyses, one of which became a gate.

---

## Appendix A — the detector inventory today

For reference, so the roadmap's claims about "what exists" can be checked.

**Repo gates that assert correctness (invoked by name in `.icc/completion-oracles.yaml`):**
`run_icc_smoke.sh` (58 probes), `gate_no_silent_wrong.py`, `run_differential.sh` (4 native
axes, 48 programs), `run_vm_parity.sh` (audit + 68-program corpus + out-of-subset + fatal),
`run_engine_parity_coverage.py` (differential construct coverage + floor),
`run_surface_parity.py` (2506 name-resolution probes), `run_language_coverage.sh` (1091
constructs), `run_ad_oracle.sh` (60 analytic checks), `run_ad_adversarial.sh`,
`run_reference_differential.sh` (chibi, 34 programs, no VM axis), `run_metamorphic.sh`,
`run_generative_differential.py`, `run_p8_escape.sh` (8 escape-closure axes),
`run_wasm_differential.sh`, plus the depth-parametric suites. 219 distinct oracle event names
across all targets; the readiness oracle carries 41 criteria when the file parses.

**ICC detectors relevant to correctness:** `gate-semantics-audit` (highest precision;
detects gates whose failure branch reads as success), `vacuous-assertions` (checks satisfiable
by construction), `odr-audit` (same-name different-shape C types), `duplicate-implementations`
(lexical clones), `contradictions` (symbol-value disagreement), `find-stubbed-prod-paths`,
`guard-liveness`, `cli-flag-audit`, `capability-exercise`, `doc-typed-claims`,
`architecture-verify` (9 invariants in `.icc/architecture-model.yaml`),
`invariants` (registry absent — task #112), `default-off-ledger` (107 gates, 0 ledgered),
`failure-signature-linkage` (passes on an empty set).

**Named ICC tool gaps carried forward.** Ten detector-precision gaps identified during the
v1.3.4 ICC audit passes (misattribution classes in `doc-typed-claims`, C++ qualified-name
head-patterns collapsing every `.cpp` to file scope, substring matches falsely clearing a
check); and one defect that silences a whole command — `quality-campaign` never binds
`oracle_status`, so any campaign stage declaring `oracle:` is permanently `pending` regardless
of what the completion oracle says. These are tracked against the ICC methodology task (#112),
which is a prerequisite for the behavioral-probe plane in B-17.

---

## Appendix B — prior art

None of the sixteen gaps is novel. Every one has an established technique behind it, and
naming the technique is what keeps a build item from being reinvented badly. Researched via
`icc research-search` per the standing web-search ruling; items ICC did not return are marked
so that the gap in the retrieval is on the record rather than hidden.

| Gap | Established technique | Canonical references |
|---|---|---|
| D-02, D-04 | **Random differential testing of compilers** — generate programs, run on multiple implementations, any disagreement is a bug | Yang, Chen, Eide, Regehr, *Finding and understanding bugs in C compilers*, PLDI 2011 (Csmith). Livinskii, Babokin, Regehr, *Random testing for C and C++ compilers with YARPGen*, OOPSLA 2020. Chen et al., *A Survey of Compiler Testing*, ACM CSUR 53(1), 2020 — the taxonomy this whole document sits inside |
| D-02 | **Equivalence modulo inputs** — the original program is its own oracle; no reference implementation needed. The direct answer to shared-defect blindness | Le, Afshari, Su, *Compiler validation via equivalence modulo inputs*, PLDI 2014 (Orion). Le, Sun, Su, *Finding deep compiler bugs via guided stochastic program mutation*, OOPSLA 2015 (Athena). Sun, Le, Su, *Finding compiler bugs via live code mutation*, OOPSLA 2016 (Hermes). **ICC did not return any of these** |
| D-02 | **Metamorphic testing** — assert relations between outputs of related inputs rather than absolute values | Donaldson, Lascu, *Metamorphic testing for (graphics) compilers*, MET@ICSE 2016. Donaldson, Evrard, Lascu, Thomson, *Automated testing of graphics shader compilers*, OOPSLA 2017. Li, *Haskell compiler testing automation based on equivalence-modulo-inputs*, 2019 — the closest functional-language precedent |
| D-02, D-04 | **Well-typed random term generation for functional languages** — the Scheme/Lisp shape of Csmith | Pałka, Claessen, Russo, Hughes, *Testing an optimising compiler by generating random lambda terms*, AST 2011. Fetscher, Claessen, Pałka, Hughes, Findler, *Making random judgments*, ESOP 2015. Klein et al., *Run your research: on the effectiveness of lightweight mechanization*, POPL 2012 (PLT Redex). `xsmith` (Racket) is the ecosystem's generator framework. **ICC returned none of these** — there is genuinely no Csmith-for-Scheme in the literature, which is why the in-tree `gen_generative_corpus.py` matters |
| D-02, D-09 | **Property-based testing** — properties plus shrinking, rather than examples | Claessen, Hughes, *QuickCheck*, ICFP 2000. MacIver, *Hypothesis* (targeted PBT and integrated shrinking). Lampropoulos et al., *Beginner's luck*, POPL 2017. Midtgaard, Møller, *QuickChecking static analysis properties*, STVR 2017 |
| D-03, D-15 | **Translation validation** — prove a compiled artifact equivalent to its source, per compilation, rather than proving the compiler | Pnueli, Siegel, Singerman, TACAS 1998. Necula, *Translation validation for an optimizing compiler*, PLDI 2000. Tristan, Leroy, PLDI 2009 / POPL 2010. **Alive / Alive2** (Lopes, Menendez, Nagarakatte, Regehr, PLDI 2015; Lopes et al., PLDI 2021) — bounded translation validation for LLVM IR, the most directly transplantable idea for a compiler with an LLVM backend |
| D-03, D-04 | **Test-suite adequacy by mutation** — measure whether the suite can *detect*, not whether it passes. The direct measure of the "trust equation" | DeMillo, Lipton, Sayward 1978. Jia, Harman, TSE 2011 (survey). Just et al., FSE 2014, *Are mutants a valid substitute for real faults?* **Mull** (LLVM-based mutation testing, arXiv 1908.01540) — usable on this codebase as-is |
| D-05, D-14 | **The oracle problem** — the general statement of why "it ran" is not "it is right" | Barr, Harman, McMinn, Shahbaz, Yoo, *The Oracle Problem in Software Testing: A Survey*, TSE 2015 |
| D-10 | **Silent data corruption detection** — cheap invariants and redundant computation for wrong answers that produce no fault | Dixit et al. (Meta), *Detecting silent data corruptions in the wild*, arXiv 2203.08989. Applicable by analogy: the detection strategy for "wrong with no signal" is the same whether the cause is a cosmic ray or a struct-layout lie |
| D-16 | **Test-case reduction and triage**, which is what makes an escape analysis cheap enough to require on every PR | Zeller, Hildebrandt, *Simplifying and isolating failure-inducing input*, TSE 2002 (ddmin). Regehr, Chen, Cuoq, Eide, *Test-case reduction for C compiler bugs*, PLDI 2012 (C-Reduce). Donaldson et al., PLDI 2021 (transformation-based reduction, almost free) |

**Operational note on using ICC for this.** `icc research-search` is a live multi-provider
literature search, not a curated store, and it is strongly sensitive to query phrasing:
concept-level queries ("multi-engine semantic parity", "translation validation") returned
under 30% relevance and considerable off-domain noise, while queries naming a tool or an
author (Csmith, CompCert, Regehr) returned the correct canonical literature immediately.
Query with proper nouns. Also record what it misses: the EMI family and the Scheme-side
generator literature above did not surface at all, and the two canonical surveys returned
only as untitled PDF links.

---

## Appendix C — measurement baselines

The numbers this roadmap should be graded against. Each is a real measurement at a named SHA,
not a target.

| Quantity | Value | Source |
|---|---|---|
| Distinct ledgered defect IDs (master `2ae3787f`) | 64 | `.icc/silent-wrong-ledger.yaml` |
| Distinct ledgered defect IDs across all lanes | 81 | all `refs/heads` |
| Ledgered defects found by the ICC correctness chain | 0 of 62 | the `missed_by:` field of `.icc/silent-wrong-ledger.yaml`, which names `icc-correctness-chain` on 43 entries |
| Language-surface execution coverage | 1091 / 1091 = 100.00% | `run_language_coverage.sh` |
| Engine differential construct coverage | 136 / 1114 = 12.21% (floor 10.95%) | `run_engine_parity_coverage.py`, PR #424 |
| Correctness harnesses reachable from CI | 14 of ~90 | `.github/workflows/` |
| Layout `static_assert`s in `lib/` + `inc/` | 0 | grep |
| Freshness guards in the 4 release-critical harnesses | 0 | grep |
| Stale reproducers in `tests/vm_parity/found/` | 13 of 38 = 34% | DD-12 |
| Baselined parity divergences that are real value divergences | 1 of 10 | PR-01, re-run at `9f2da2ab` |
| Readiness criteria graded while the oracle was unparseable | 2 of 41 | PR-12, PR #436 |
| Defects found by the value-position sweep on its first run | 5 | SW-35, SW-36, SW-37, SW-38, SW-39 |
| PRs #420-#440 carrying an escape analysis | ~19 | PR bodies |
| Escape analyses that became a standing gate | 1 | `run_value_position_sweep.py` |
