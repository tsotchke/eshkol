# Compositor correspondence: native C against the Lean model

This test checks that the native compositor (`lib/core/os_k5_k6_services.c`) and
`proofs/lean/EshkolCompositorAligned.lean` behave the same. It starts both from the same
state and runs the same random operations. After every operation it compares the return
status, the surface table and the framebuffer. The Lean model is evaluated directly; it is
not transcribed into C.

```sh
CC=cc LEAN_BIN=<lean 4.12.0> ./run_compositor_correspondence.sh            # seeds 1 20261126 777
CC=cc LEAN_BIN=<lean 4.12.0> ./run_correspondence_negative_controls.sh     # 12 controls
```

## Abstraction relation

The model is one-dimensional; the native framebuffer has height 1.

- **Surfaces:** `Surface ⟨owner, xStart, xLen, writable⟩` corresponds to the native surface
  `(x = xStart, y = 0, w = xLen, h = 1)`, with rights `READ`, plus `WRITE` when writable.
- **Handles:** model handle `⟨id, badge⟩` corresponds to the native handle with slot `id`,
  the generation and capability id issued for that slot, and `owner_badge = badge`. Native
  never frees a slot, so slot and registration index coincide.
- **Forged handles:** a native handle whose (slot, generation, capability id) was never
  issued corresponds to a model id that names no surface. The traces include three such
  kinds: a failed registration's handle, an issued handle with its generation incremented,
  and one with its capability id changed. If native ever accepted one, the model would refuse
  it and the test would report a divergence; this difference cannot vanish inside the
  adapter.

## Results

- **Agreement:** 3 seeds × 2,000 traces × 60 operations; every trace agrees after every
  operation. Every native status occurs, including `NO_SLOTS` and `INVALID_ARG`.
- **Not a refinement proof:** this is finite trace agreement on the one-dimensional domain.

Negative controls: each of the 12 mutants of the model must diverge from native, except
`write-drop-conflict-check`. That check is redundant on every reachable state, because
registration already keeps writable surfaces of different owners apart. The correspondence
test cannot see it; only a theorem stated over all states can.

- **Proofs:** detect 10 of the 12 mutants.
- **This test:** detects 11 of the 12.
- **Together:** detect all 12.

## What the comparison showed about the specification

These are differences between `EshkolKernelPolicy.lean` and native:

- **Native refuses writes the model permits.** It refuses every write outside the caller's
  surface, past the framebuffer, or under a foreign or forged handle. The original model
  permits these, so the defect was in the model, not in the C code.
- **Native permits overlaps the model forbids.** Native refuses an overlap only between
  writable surfaces of different owners, which is the policy its header documents.
  `compositor_surface_spatial_disjointness`, which says that all registered surfaces are
  pairwise disjoint, therefore does not hold of the native compositor.
  `EshkolCompositorAligned` states the native policy.
- **The original model has no framebuffer extent.**

## Limits

- **One dimension:** row arithmetic in `write_rect` is exercised on row 0 only.
- **Write isolation, not confidentiality:** a read-only surface may overlap another owner's
  writable surface and observe what is written there.
- **No handle invalidation:** native has no operation that invalidates a handle, so no
  temporal property is stated.
