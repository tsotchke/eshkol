# Frame-pool correspondence: native C against the Lean model

This test compares the native frame allocator (`lib/kernel/k2_mmu_frames.c`) with
`FramePool` in `proofs/lean/EshkolKernelK2.lean`. It runs random operations on native and
replays them on the model, which Lean evaluates directly.

```sh
CC=cc LEAN_BIN=<lean 4.12.0> ./run_frame_pool_correspondence.sh          # seeds 1 20261126 777
CC=cc LEAN_BIN=<lean 4.12.0> ./run_frame_pool_negative_controls.sh       # 3 controls
```

## Correspondence

| Native | Model |
|---|---|
| `k2_frame_pool_init(base, T, K)` | owners `i < K` kernel, the rest free, generations 1 |
| `k2_frame_alloc_user` returns frame `idx` | `allocUser idx`; the model must accept the index native chose |
| `k2_frame_alloc_user` refused | no index is accepted by `allocUser` |
| `k2_frame_free_user(handle)` | `freeUser idx gen` |
| `k2_frame_alloc_kernel` returns frame `idx` | `allocKernel idx`; same acceptance rule as `allocUser` |
| `k2_frame_alloc_kernel` refused | no index is accepted by `allocKernel` |

The model has no physical address. A native handle whose `phys_addr` does not match its
index therefore maps to an out-of-range model index. Native refuses such handles with
`STALE_HANDLE`. If it ever accepted one, the model would refuse and the test would report a
divergence.

After every operation the test makes two comparisons:

- **Operation:** the model's state after the step must equal native's: free count, allocated
  count, and the owner and generation of every frame.
- **Invariant:** native's state must satisfy `Conserved`, `Accounted`, and native's own
  `k2_frame_pool_invariants_hold`.

## Results (3 seeds × 2,000 traces × 60 operations)

- **Without `alloc_kernel`:** 360,000 operations, with no operation divergence and no
  invariant failure.
- **With `alloc_kernel`:** 360,000 operations, with no operation divergence and no
  invariant failure. Every trace stays in the model's domain to its end.
- **Statuses covered:** every refusal status the allocator returns without mappings:
  `OUT_OF_FRAMES`, `DOUBLE_FREE`, `STALE_HANDLE`, `KERNEL_FRAME_LOCKED` and `INVALID_ARG`.

Negative controls, run on the model's definitions:

| Mutant | Proofs | Correspondence |
|---|---|---|
| `alloc-forgets-owner` | REFUTED | diverges |
| `free-forgets-owner` | REFUTED | diverges |
| `alloc-skips-free-check` | REFUTED | **agrees** |

`alloc-skips-free-check` agrees because native chooses the frame itself, from its free list,
and never offers the model an index that is already allocated. The model's interface, in
which the caller names the index, is more general than native's. The check guards a case
that native's interface cannot produce. Only the `Accounted` proof sees it.

## Findings about the model

- **Runtime kernel allocation is now modelled.** `PoolOp.allocKernel` takes the frame native
  chose and gives it owner `kernel`. `pool_step_preserves_conservation` and
  `pool_step_preserves_accounted` cover it, so the frame theorems extend to traces that
  allocate kernel frames at run time.
- **Native's choice policy is not modelled.** Native allocates from a LIFO free list, lowest
  index first after initialisation. The model leaves the choice open, which is a sound
  abstraction for these properties.

## Limits

- **Mappings:** frames are never mapped, so `FRAME_STILL_MAPPED` is not exercised. Mapping
  is a separate part of the model (`UserAspace`) and of native.
- **Finite trace agreement, not a refinement proof.**
