# Metamorphic/property oracle

P7c adds deterministic property-style tests whose oracle is an algebraic law,
not a hand-written expected value. `gen_metamorphic.py` emits static `.esk`
programs from an explicit xorshift64* PRNG seeded by case index; generated
programs use no runtime randomness.

## Covered law families

- `list-vector`: reverse/reverse identity, append length additivity, map
  fusion, append associativity, and list/vector roundtrip.
- `sorting`: sort idempotence, permutation preservation, and ordered output.
- `numeric`: exact commutativity, associativity, distributivity, rational
  numerator/denominator reduction, exponent addition, gcd/lcm, and
  `min + max = a + b`.
- `string-char`: string/list roundtrip, whole-substring identity, and exact
  number/string/number roundtrip.
- `roundtrip-control`: tracked write/read roundtrip gap, tensor save/load,
  call/cc identity, apply/map consistency, apply/sum consistency, and values
  roundtrip.

## Running

```sh
cmake --build build --target eshkol-run stdlib -j
scripts/run_metamorphic.sh
```

Useful variants:

```sh
scripts/run_metamorphic.sh --quick --no-aot
scripts/run_metamorphic.sh --regen --trials 64
```

The runner writes ICC evidence to `scripts/icc_traces/metamorphic.jsonl`.
Every generated file is run twice through JIT with cache disabled; the two
combined outputs and exit codes must match byte-for-byte. AOT is enabled by
default and can be skipped with `--no-aot`.

Known-open law violations print `XKNOWN:` and must name an `.swarm/tasks/`
ticket. XKNOWN findings are tolerated by the gate so the harness can land while
still keeping the violation visible in reports and ICC traces.
