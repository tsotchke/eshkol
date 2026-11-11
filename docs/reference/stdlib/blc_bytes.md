# Classical BLC byte observations

`(require core.blc_bytes)` provides one function:

```scheme
(blc-run-bytes program input work-limit heap-capacity output-limit)
```

`program` is a closed classical `core.blc` AST. `input` is a proper finite list
of integers 0..255. The function applies the program to its BLC8 encoding and
observes a finite list of eight-bit bytes using the existing Scott-pair/False
encoding. It returns:

```scheme
(status bytes work allocations beta cache-hits collections max-gc-live)
```

Only `normal` establishes a complete byte observation. Other statuses are
`work-bound`, `heap-bound`, `output-bound`, `input-bound`, `invalid-term`,
`invalid-input` and `invalid-output`. Returned bytes are only the completed
prefix on refusal. A refusal never claims divergence. Invalid resource
arguments raise an error: work must be a nonnegative integer, heap capacity
at least 16, and output limit an integer 0..65536.

```scheme
(blc-run-bytes blc-I '(0 1 255) 10000 1024 3)
;; (normal (0 1 255) ...counters...)
```

This deterministic call-by-need machine uses lexical environments, memoized
thunks and an explicit continuation stack. A copying collector updates all
machine and observer roots. Each invocation owns its mutable state. Immutable
code annotations avoid retaining caller environments for closed expressions;
input byte code is shared without changing the lambda program. Native tokens
serve only as output observation contexts. Program arithmetic, learning,
differentiation and code production execute within BLC.

The work counter charges machine transitions, syntax/input validation visits
and copied collector objects. It does not count every host instruction or
collector forwarding-table clear. It differs from strong beta steps and
Tromp graph steps. Allocations count pooled records; `max-gc-live` samples live
records after collections, rather than measuring process memory or peak live
objects. Two semispaces and a forwarding vector use nine vectors of
`heap-capacity` slots; syntax, input, output and host-runtime overhead are
additional. External wall/CPU/address-space limits remain necessary.

Input is capped at 65,536 bytes and source syntax at 65,536 visited nodes.
Syntax validation rejects open variables, extra constructor fields and
stochastic choices. The byte observer checks exactly eight bits and a proper
terminator for every output byte. A program that diverges or produces an
infinite stream cannot report `normal` within finite work/output bounds.

The API observes bytes and does not compute full normal forms. `core.blc`'s
existing tree normalizer remains available, and small-case tests compare its
answers with this machine. No sharing semantics are promised for SBLC choices.

Run the focused semantic, collector and refusal gate on JIT and AOT:

```sh
BUILD_DIR=build bash scripts/run_blc_bytes_gate.sh
```

Add `--corpus` for the slower [seven-program contribution](../../../examples/blc-corpus/README.md).
The corpus gate fails on any resource refusal, wrong byte or JIT/AOT receipt
difference. Its fixture-specific work/heap/output limits and process limits
are explicit. Full self-application is intentionally a slow acceptance case.
CTest runs the focused gate; configure `ESHKOL_BLC_CORPUS_TESTS=ON` to register
the additional corpus gate. Python 3.11+ is dependency free; 3.10 uses `tomli`.
