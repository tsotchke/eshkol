# Learn a designated Hawkes program hole in the constructed SDNC

Status: review-ready. Selected MIT contribution by Richard Hoekstra.

Dependencies: the constructed SDNC at the pinned base. Prerequisite implementations are outside this patch. `SOURCE.toml` records exact identities.

```sh
python3 repro/sdnc-holes/run_sdnc_gate.py --results /tmp/sdnc-holes.json
```


`sdnc_holes.c` includes the actual upstream `weight_matrices.c` tool, constructs
its analytic 256-wide, six-layer interpreter and uses its matrix runner and
object-program AD. It does not train the random default `sdnc-program` model
or replace the interpreter with a host simulator.

The real target is the Hawkes x1 drift numerator at w=k/4:

```
72*x1' = k*(18*x1+9) + 24*x2 + 16 - 144*x1.
```

Only the integer operand k in [0,4] learns. Context operands are supplied
inputs; program opcodes, all other operands and every semantic weight remain
frozen. A host projected, clipped gradient update consumes the derivative
computed by matrix-executed object AD. It is not a gradient through semantic
weights or a surrogate-backward fine-tuning claim.

The training context, target k=3 and three held-out contexts are frozen before
updates. The held-out error never feeds optimization. Four updates take
k through 0,1,2,3 and retain 3. Held-out MSE goes 31833 -> 0. Independent
reference execution checks every matrix output; the object derivative equals
the analytic one. All 48,881,688 weight bytes remain identical. Four distinct
arithmetic/local/AD programs retain byte-identical outputs. Complete traces
cover 186 steps in 22 matrix runs, with zero native-delegated steps. This is a
bounded sketch-hole consumer and retention panel, not a general learning result
or full-ISA preservation proof. It requires no GPU or large training sweep.
