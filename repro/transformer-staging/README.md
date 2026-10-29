# Staging the original raw-token transformer source

A standalone MIT follow-up to Eshkol PR #855, pinned at
`3965b09e12d5c1e36afc2b04d27f4940e94bc999`. The original generic byte
parser/transformer and six residuals cover attention and an unfamiliar transfer
topology, each with inference, exact gradients and one internal update. Model
and token/target bytes remain dynamic. Numeric operations and refusal guards
execute inside the original lambda program.

`producer.lisp` replays annotated source staging without ASDF, RSH or a network
connection. It regenerates all six residual sources, checks their inclusion
in the closed compilation units, and independently rebuilds the exact seven
BLC bit strings. Complete ASCII and packed code are supplied. The frozen
Common Lisp stager runs with SBCL; Eshkol executes its output through the
existing `core.blc_bytes` observer. This is explicit host staging, not an
Eshkol-native producer, automatic annotation inference, packed-binary
specialization or a new Futamura-projection claim.

```sh
sbcl --script producer.lisp
python3 run_gate.py --engine all --eshkol-bin /path/to/eshkol-run \
  --source-root /path/to/eshkol --reference-bin /path/to/tromp/uni \
  --results fresh-results.json
```

Supply John Tromp's `uni.c` independently and compile it as the reference.
The gate uses four `-x` heap doublings for large residuals. Python 3.11+
needs no packages; Python 3.10 can use `tomli`. Every native process has
explicit address-space, stack, CPU and wall limits. The full corpus is
opt-in. `--case ID` selects a smaller diagnostic scope, recorded in the receipt.

Fourteen fixtures compare generic full frames and residual compact frames.
An independent rational forward-dual evaluator reconstructs features, logits,
losses, gradients, rounded updates and complete bytes, including malformed
input and numeric refusals. Programs never receive expected answers.

The earlier RSH experiment saved 7.91% of generic reducer work on a larger
44-case panel; valid attention inference saved 19.25%. Those are historical
reference-evaluator observations, not new Eshkol timings. This packet records
its own executions. Code can grow; producer cost is separate from execution.
No net latency, training or generalization claim follows.

`results.json` records qualification; `manifest.toml` identifies selected
sources and prerequisite modules. The MIT grant covers this packet only.
The PR targets `feat/v14-sblc` and requires the byte observer from PR #855.
Runtime qualification uses the pinned #855 source above. No production
compiler/runtime API changes are included.

Run `python3 self_test.py` to demonstrate that the actual readiness gate
rejects a successful backend returning incorrect data. Infrastructure failures
are distinguished from the expected rejection.
