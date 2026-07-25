# ESH-0361 — exit-0 masking in toolchain fault paths

Found by the P8 escape-closure **axis 7 (toolchain fault-injection)** suite on
master `5cb02c8a`. UNFIXED — recorded and reported, quarantined as XKNOWN in
`scripts/p8/p8_fault_injection.sh` so the suite stays green-able while tracked.
Each cell flips from XKNOWN to a hard gate (XPASS alarm) the moment it is fixed.

The `-r` (JIT) and AOT drivers exit **0** on several fault inputs, masking the
failure from any build system that checks `$?`. Contrast: the same driver
correctly exits nonzero for a *broken --lib link* (the #334 fix) and for a
*malformed source under AOT*, so the exit-code contract is inconsistent across
paths.

## Cells (all reproduce on master 5cb02c8a)

| # | invocation | prints | exit | verdict |
|---|------------|--------|------|---------|
| a | `eshkol-run MISSING.esk -o out.bin` (AOT) | `ERROR: File not found` | **0** | masks — **and emits a 5 MB out.bin** |
| b | `eshkol-run -r SYNTAX-ERR.esk` | `error: unexpected end of input in list` | **0** | masks a hard parse error |
| c | `eshkol-run -r UNREADABLE.esk` (chmod 000) | (nothing) | **0** | masks an unopenable file |
| d | `eshkol-run -r BAD-REQUIRE.esk` | runs the rest | **0** | ignores `(require missing.module)` (possibly lenient-by-design) |
| e | `eshkol-run BAD-REQUIRE.esk -o out` (AOT) | — | **0** | AOT ignores a missing require |

Cell **a** is the most harmful: a CI/build step sees exit 0 and a freshly
written binary, and ships an executable that never contained the user program.

Cells that CORRECTLY exit nonzero today (kept as hard gates in the suite, they
retro-guard #334): `--lib BOGUS` under `-r` and AOT, malformed source under AOT,
`-o` into a nonexistent directory, undefined top-level symbol, missing source
under `-r`.

## Minimal repro (cell a)

```sh
eshkol-run /path/does/not/exist.esk -o /tmp/out.bin ; echo "exit=$?"
# stderr: "     ERROR: File not found: /path/does/not/exist.esk"
# exit=0            <-- should be nonzero
# /tmp/out.bin      <-- 5 MB binary should NOT have been written
```
