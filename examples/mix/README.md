# Generate a standalone specialized program

The generator executes the self-interpreter specialization and its complete
normalizer as object-language Mix programs, then prints ordinary Eshkol source.
It emits `mix-generated(x) = x*x+4` and `mix-erased(x) = 0`. Both entries retain
an exact-integer input contract, including the erased read in the second.

From the repository root:

```sh
export ESHKOL_LIB_DIR="$PWD/lib"
export ESHKOL_PATH="$PWD/lib"
export ESHKOL_JIT_CACHE=0
build/eshkol-run --no-stdlib -I lib -L lib -O0 -r examples/mix/generate.esk > generated.esk
```

Append `(display (mix-generated 7)) (newline)` to the generated file. Execute it
with `build/eshkol-run -r generated.esk`, or compile it using
`build/eshkol-run -o generated generated.esk` and execute `./generated`.
The answer is 53. The resulting program needs neither Mix nor a runtime JIT.

`scripts/run_mix_gate.sh` performs this generation, compilation, independent
answer checking and integer-entry refusal on both JIT and AOT. It additionally
qualifies the raw F1/F2/F3 bytecode corpus and fourteen emitted F3 residuals.
See [the API and bounds](../../docs/reference/stdlib/mix.md).
