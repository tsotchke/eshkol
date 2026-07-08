# Noesis → Eshkol: `_eshkol_getenv` broken in the static-lib AOT path (2026-06-26)

**Severity: HIGH — blocks the production CLI (`bin/noesis`).** Regressed today; was
working earlier this session.

## Symptom
`(getenv "X")` returns `#f` even when X **is set** in the environment — but only in
binaries linked against `libeshkol-static.a` (the CMake `--emit-object` + c++ link
path). The interpreter and `eshkol-run -O2 -o` are fine.

```
;; ge.esk
(require stdlib) (display (getenv "NOESIS_TESTVAR")) (newline)
```
| build path | NOESIS_TESTVAR=hello → |
|---|---|
| `eshkol-run -r ge.esk`                    | **hello** ✅ |
| `eshkol-run -O2 -o gebin ge.esk`          | **hello** ✅ |
| CMake `--emit-object` + link libeshkol-static.a (= `bin/noesis`) | **#f / unset** ❌ |

Effect on Noesis: `bin/noesis <anything>` aborts at startup with
`Unhandled exception: curriculum: missing environment value` — because the bundled
`curriculum` module reads ~16 `NOESIS_CURRICULUM_*` policy vars at load, and every
read returns #f. Setting **all 16** vars does not help (getenv still returns #f),
which is what proves it's getenv, not missing policy. All 32 CLI smokes fail; the
interpreted gate suite (which uses `eshkol-run -r`) is unaffected — **23/30 gates
pass, geometry-substrate 10/10, W64 1000-cycle cert 8/8.**

## Localization
The emitted object references the runtime symbol, undefined, to be resolved from the
static lib:
```
$ nm ge.o | grep getenv
                 U _eshkol_getenv
```
So the object is correct; the broken implementation is **`_eshkol_getenv` inside
`libeshkol-static.a`** (today's 16:17 build at HEAD `60f05bfd`). `eshkol-run`'s own
embedded getenv works, so the static-lib copy has diverged/regressed.

## History
`make verify` (32/32 CLI smokes) **passed earlier this session** on a `bin/noesis`
built against an earlier `libeshkol-static.a`. It broke after the static lib was
rebuilt at 60f05bfd (CMakeLists.txt is currently modified in the tree; PRs #79–84 in
flight). A **clean `cmake --build build` of eshkol did not fix it** — so the
regression is in the source at this SHA, not a stale artifact.

## Ask
1. Fix `_eshkol_getenv` in `libeshkol-static.a` so static-linked AOT binaries read
   the real environment (parity with `eshkol-run`). Likely a regression from the
   in-flight merge; a hosted-native runtime symbol that got stubbed/short-circuited
   in the static archive.
2. Add a CTest that AOT-compiles a `getenv` probe **via the static-link path** (not
   just `eshkol-run -O2`) and asserts it reads a set var — the two AOT paths
   currently diverge and only the eshkol-run one is covered.

## Noesis-side follow-up (independent hardening, filing for our own tracker)
`curriculum` (and other strict-policy modules) read required env at MODULE TOP LEVEL,
so merely *loading* them in the CLI bundle can abort the whole binary. Even with a
correct getenv this means `bin/noesis version` requires 16 env vars. We will make
these reads lazy (evaluated at first use / construction, per the explicit-policy
intent) so the CLI is robust — but that is moot until #1, since getenv itself is
returning #f.
