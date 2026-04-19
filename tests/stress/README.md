# Eshkol Stress Tests

Long-running / exhaustion tests that verify the overflow guards,
memory caps, and concurrency invariants hold under sustained load.
These intentionally run for minutes to hours — not part of the
default `run_all_tests.sh` suite; drive them from CI separately.

## Harnesses

| Suite | Duration | Purpose |
|---|---|---|
| `stress_parallel_at_scale.esk` | 1-5 min | `parallel-map` / `parallel-fold` at N=1M |
| `stress_alloc_loop.esk` | 10 min | Arena alloc/free cycling — leak detection |
| `stress_fd_exhaustion.esk` | 30 s | Subprocess spawn/destroy loop — fd cleanup |
| `stress_long_subprocess.esk` | 1 min | Chatty child w/ big output — pipe drain stability |

## Running

```
# Individual:
./build/eshkol-run -r tests/stress/stress_parallel_at_scale.esk

# Under ASan (rebuild sanitizer tree first):
bash scripts/build-sanitizer.sh asan
(cd build-asan && ./eshkol-run -r ../tests/stress/stress_parallel_at_scale.esk)

# Whole suite (long — intended for CI):
bash scripts/run_stress_tests.sh
```

## 24h soak (manual)

```
bash scripts/run_stress_tests.sh --hours 24 | tee 24h.log
```

The harness exits 0 only if every iteration succeeds and the RSS
stays within 2x the initial baseline. Any crash, hang beyond the
loop quantum, or unbounded RSS growth fails the run.
