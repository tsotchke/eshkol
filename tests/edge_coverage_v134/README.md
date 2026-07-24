# v1.3.4 dynamic edge coverage

Generative, seeded, depth-parametric edge-case coverage for the language surface
the v1.3.4 wave added. This is the P2 (edge-matrix) + P6 (depth-parametric)
extension for the new surface, differentially verified across execution axes and
gated by the ICC release oracle.

Unlike `tests/edge_matrix/generated/` this corpus is **not committed**: it is a
pure function of `(seed, --max-depth, --counts)` and is regenerated into a
temp directory at run time (bounded + cleaned up, per the fuzz/harness
disk-budget rule). A CI divergence reproduces locally byte-for-byte with the
same seed.

## Pieces

| Piece | Path |
|---|---|
| Generator | [`scripts/gen_edge_v134.py`](../../scripts/gen_edge_v134.py) |
| Runner | [`scripts/run_edge_coverage_v134.sh`](../../scripts/run_edge_coverage_v134.sh) |
| ICC oracle | `.icc/completion-oracles.yaml` → `v1.3.4-edge-coverage` (per-family) + `v1.3-evolve` roll-up |
| ICC smoke probe | `scripts/run_icc_smoke.sh` → `edge_v134_dynamic_coverage` |
| Depth registry | `scripts/depth_coverage_registry.json` supplemental axes |

## Families

Each generated file is self-checking (`;; CHECKS: N` header, one `PASS: ` line
per assertion, `FAIL: ` on a wrong value); the runner classifies every file×mode
as PASS / ASSERT-FAIL / CRASH / COMPILE-ERR / HANG. Ground truth is computed in
the generator (Python) and embedded as the literal the Eshkol program is checked
against, or asserted as a metamorphic identity.

| Family (file prefix) | Surface | Axes | Execution axes |
|---|---|---|---|
| `nursery_` | nursery iter-scope, mutating define-loops / named-let ticks (ESH-0214e) | 6 barrier channels, escape-set size {1,4,16,64}, nested-loop depth 1..N (P6b) | JIT, AOT-O0, AOT-O2 |
| `parallel_` | `parallel-map` / `parallel-execute` capturing closures returning collections (#331) | n at pool threshold {1,4,15,16,17,64,500}, closure shapes, scope-op-heavy workers, nesting depth 1..N (P6f) | JIT, AOT |
| `gradient_` | exact gradient through a callable parameter + curried form (#330) | arity 1..5, list vs vector point, non-polynomial composition, in-loop repetition, composition depth 1..N (P6a) | JIT, AOT |
| `i128_` | native 128-bit integer (#314) | boundary constructors, wraparound at ±2^127, conversion edges, arithmetic-chain depth 1..N (P6d) | JIT, AOT, **VM (differential)** |
| `matmul_` | tensor / matmul | arange arities, matmul from reshape/arange vs reference product, multi-dim tensor-ref/tensor-set! | JIT, AOT-O0, AOT-O2 |
| `adtape_` | low-level ad-tape / ad-pow | fractional/negative/zero exponent, tape reuse, 1024-node growth boundary | **VM-only** (ad-pow / ad-tape-length are VM builtins) |
| `roundtrip_` | `number->string`∘`string->number` identity | subnormals, ±0.0, powers of two, 1e±308 | **staged** — see below |

### Staged (not yet gated)

* **`roundtrip`** asserts `number->string`∘`string->number` == identity. That
  requires the shortest-round-trip printer, which is not yet on master
  (`number->string` is still `%g`-lossy, so values needing >6 significant
  digits do not round-trip). The generator is ready; enable with
  `FAMILY=roundtrip scripts/run_edge_coverage_v134.sh` once the printer lands.
* **`matmul` on the VM**: the VM registers only flat arity-2 `tensor-ref`,
  arity-3 `tensor-set!`, arity-2 `reshape`, and arity-1 `arange`, so multi-dim
  tensor parity is native-only until the VM tensor surface lands. `matmul` IS
  gated on the native axes.

## Running

```sh
# full sweep, all applicable axes
scripts/run_edge_coverage_v134.sh

# one family (per-oracle gate)
FAMILY=i128 scripts/run_edge_coverage_v134.sh

# add optimization / VM axes, choose depth + seed
MODES="jit aot aot-O0 vm" MAX_DEPTH=6 SEED=20260723 scripts/run_edge_coverage_v134.sh
```

The full sweep (JIT + AOT-O0 + AOT-O2 + VM, depth 6) runs in well under a minute
on a 4-core slice. The ICC smoke probe runs a reduced depth-4 sweep.
