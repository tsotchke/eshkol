# Generative Multi-Oracle Differential Report (P7c)

> "If our system does not constantly expose every single hidden bug then it has
> no coverage." — the maintainer.

The hand-written reference-differential corpus (P7a, `scripts/gen_reference_
corpus.py`) is 34 fixed programs and, on current master, **34/34 AGREE** against
chibi — it no longer exposes anything. This pillar makes differential testing
**generative and multi-oracle**: it generates a large, deterministic family of
closed R7RS-small programs and cross-checks each across every installed
execution oracle. Any pairwise disagreement is an exposed bug.

See `tests/generative-diff/README.md` for the design; this document is the
**exposed-bug catalogue** with a minimal repro for each distinct defect. The
underlying defects are NOT fixed here — each is an open task for a separate
fixing agent. The harness itself ships green (regression mode against the
triaged baseline); it flips RED the moment a NEW divergence is generated.

## Resolved: the "34 vs 27" conflict

`docs/reports/REFERENCE_DIFFERENTIAL_REPORT.md` records **27/34 AGREE** (RED);
`CHANGELOG.md` claims **34/34 AGREE (100%)**. Re-running the actual harness on
this branch (master + this branch's non-functional additions):

```
$ scripts/run_reference_differential.sh
Reference : chibi-scheme 0.12.0 "magnesium"
Total     : 34    AGREE : 34    ESHKOL-DIVERGES : 0
Agreement rate: 100.0%    Gate : PASS
```

**The CHANGELOG is correct: 34/34 today.** The `REFERENCE_DIFFERENTIAL_REPORT.md`
"27/34" table is a **stale pre-fix snapshot** — the 7 divergences it lists
(ESH-0150..0156: apply-leading-args crash, multi-arg `vector-map`, `cond`/`case`
`=>`, `vector-copy`, vector quasiquote, `error-object?` family, `write` escaping)
plus ESH-0225 were all fixed and are documented as fixed in the CHANGELOG. The
report even carries a note that it is "superseded by this changelog entry." The
27/34 figure should be read only as history. This is precisely why a fixed corpus
is insufficient and a generative one is needed.

## Oracles

| Oracle | Command |
|---|---|
| `chibi`  | `chibi-scheme <prologue+body>` (external R7RS ground truth) |
| `jit`    | `build/eshkol-run -r prog.esk` |
| `aot-O0` | `build/eshkol-run -O0 prog.esk -o BIN && BIN` |
| `aot-O2` | `build/eshkol-run -O2 prog.esk -o BIN && BIN` |
| `vm`     | `eshkol-run --profile hosted-vm --emit-eskb X.eskb prog.esk` then `eshkol-vm-standalone-test X.eskb` |

## Generator

`scripts/gen_generative_corpus.py` — a deterministic (pure function of
`seed`,`count`) typed recursive generator over a meaningful R7RS-small subset:
exact integer arithmetic (incl. `quotient`/`remainder`/`modulo`, `gcd`/`lcm`,
bignum via bounded `expt`), exact rationals, inexact reals (`sqrt`/`floor`/
`round`/…), booleans and predicates, lists (`cons`/`append`/`reverse`/`map`/
`list-ref`/`list-tail`), higher-order `map`/`apply`/inline `filter`/`fold`,
`let`/`let*`/`letrec`/named-let, `cond`/`case`/`if`/`and`/`or`, closures, chars,
strings (`string-append`/`substring`/`string-upcase`/…), vectors
(`vector-map`/`list->vector`/`vector-copy`), and quasiquote. Every generated
program is **closed, printable, TOTAL** (structurally free of division by zero,
out-of-range indexing and `car` of `'()`) and **deterministic**. Two families:

* **diff** — typed value-printing probes, cross-checked across all oracles.
* **meta** — self-checking metamorphic properties (each must display `#t`):
  `(f x)==(apply f (list x))`, `(map f xs)` vs a hand-rolled left-to-right map,
  `+`/`*` commutativity, `reverse` involution, `length`/`append` homomorphism,
  `let`/`let*` re-association, fold-vs-`apply +`. A `#f` on any oracle — or a
  cross-oracle disagreement — is a bug even with **no reference installed**.

Ground-truth validation: all generated programs run cleanly on chibi (generator
is total) and every `meta` property evaluates to `#t` on chibi.

## Results summary

* **Native paths are clean.** Across every generated program checked, the four
  native/reference oracles — `chibi`, `jit`, `aot-O0`, `aot-O2` — **agree**.
  No `*_VS_CHIBI_MISMATCH`, no `AOT_O0_VS_O2_MISMATCH`, no `JIT_VS_AOT_MISMATCH`
  reproduced under re-verification. (Transient `AOT_O0_VS_O2_STATUS` events seen
  under heavy concurrent CPU load were **timeout artifacts**; they did not
  reproduce on a second run and are rejected by the harness's re-verify pass —
  see "False-positive control" below.) This is a strong positive result: the
  LLVM codegen, JIT/AOT paths and the new `-O2` default are mutually consistent
  and R7RS-conformant on the generated subset.

* **The bytecode VM silently diverges** on a broad, common slice of the
  language. Every divergence below is a **silent wrong answer**: the VM exits 0
  with **no** `ERROR`/`OVERFLOW`/diagnostic marker (the VM exits 0 even on fatal
  errors — `tests/vm_parity/found/error_exit_code_zero.esk`), so a wrong value
  with no warning is the dangerous case this pillar targets. These generalise
  the "27 VM-vs-native divergences" the maintainer flagged, now reproduced at
  generated scale and minimised to **7 distinct root-cause defect classes**.

## Exposed VM defects (the treasure) — minimal repros

Each was reduced to a one-line closed program. `chibi` and `jit` (and AOT) all
produce the correct value; the VM silently produces the wrong one.

### V1. Characters render as their integer code point under `display`
```scheme
(display #\I)          ; chibi/jit: I     vm: 73
(display (char-upcase #\a))  ; chibi/jit: A  vm: 65
```
The char type collapses to its code point when displayed.
(cf. `tests/vm_parity/found/char_type_collapsed.esk`.)

### V2. `list->vector` returns `#f`
```scheme
(display (list->vector (list 1 2 3)))   ; chibi/jit: #(1 2 3)   vm: #f
```
`(vector 1 2 3)` renders correctly, so it is `list->vector` specifically.

### V3. Large integers / bignums render as inexact floats
```scheme
(display (expt 2 40))   ; chibi/jit: 1099511627776   vm: 1.09951e+12
```
Exactness and full precision are lost in the VM's numeric printing/representation
for large integers.

### V4. Exact rationals collapse to inexact floats
```scheme
(display (/ 1 3))              ; chibi/jit: 1/3   vm: 0.333333
(display (+ (/ 1 3) (/ 1 6)))  ; chibi/jit: 1/2   vm: 0.5
```
The VM has no exact-rational representation; `/` produces a float.

### V5. `round` is not round-half-to-even
```scheme
(display (round 2.5))   ; chibi/jit: 2   vm: 3
```
R7RS `round` uses banker's rounding (2.5 → 2, 3.5 → 4); the VM rounds half up.

### V6. `equal?` on compound data returns `#f` (structural equality broken)
```scheme
(display (equal? (list 1 2 3) (list 1 2 3)))    ; chibi/jit: #t   vm: #f
(display (equal? (vector 1 2) (vector 1 2)))     ; chibi/jit: #t   vm: #f
```
`equal?` behaves like `eq?` (identity) on pairs and vectors; strings and numbers
compare correctly. This is the root cause of most `META_PROPERTY_FALSE` VM hits
(the `reverse`-involution and `map`-ordering properties compare freshly-built
lists with `equal?`). (cf. `tests/vm_parity/found/equal_eq_structural_false.esk`.)

### V7. Sibling binding-forms as operands corrupt each other
```scheme
(display (+ (let ((a 1) (b 2)) (+ a b))
            (let ((a 10) (b 20)) (+ a b))))   ; chibi/jit: 33   vm: 16
```
Two `let`/`let*` blocks used as sibling arguments to the same call clobber each
other's stack slots in the VM — each block evaluated **alone** is correct
(`(let ((a 1)(b 2)) (+ a b))` → 3), but as adjacent operands the result is
wrong. Independent of whether the two blocks share variable names. A frame /
slot-allocation defect in the VM compiler for nested binding forms in argument
position. (Related family: `consecutive_do_state_leak.esk`,
`define_after_do_corrupted.esk`.)

## False-positive control (why the native "clean" result is trustworthy)

A generative differential harness is only credible if it does not cry wolf.
Every divergence is **re-verified**: on the first sign of a disagreement the
harness re-runs the same program and keeps only the divergence **kinds** that
reproduce. Under the heavy concurrent build load on the development host
(load average ~13, several other agents compiling simultaneously) individual
`-O2` AOT compiles occasionally exceeded the timeout, which without re-verify
would surface as a spurious `AOT_O0_VS_O2_STATUS`. Re-verify rejects these; the
7 VM defects above are perfectly deterministic and survive every re-run.

## Reproduce

```sh
brew install chibi-scheme
cmake --build build --target eshkol-run stdlib eshkol-vm-standalone-test -j

# full discovery run (RED while any divergence remains — the philosophy)
scripts/run_generative_differential.py --seed 1234 --count 60

# regression tripwire against the triaged baseline (green now; RED on a NEW bug)
scripts/run_generative_differential.sh          # == --smoke --baseline <baseline>

# minimise a single VM defect by hand
printf '(display (equal? (list 1 2 3) (list 1 2 3)))(newline)\n' > /tmp/r.esk
build/eshkol-run -r /tmp/r.esk                                   # #t
build/eshkol-run --profile hosted-vm --emit-eskb /tmp/r.eskb /tmp/r.esk
build/eshkol-vm-standalone-test /tmp/r.eskb                      # #f
```

## ICC wiring

The harness writes `kind:"generative_differential"` JSON-L to
`scripts/icc_traces/generative_differential.jsonl`. The gate event
`generative_differential_oracle` is consumed by the new
`.icc/completion-oracles.yaml::generative-differential` oracle (severity high)
and by the `generative_differential_oracle` probe in `scripts/run_icc_smoke.sh`.
In regression mode both are PASS while every divergence is a known baseline
entry; a NEW cross-oracle divergence flips them RED.
