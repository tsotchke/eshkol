# Does popping a child restore its parent?

At the child arena's destruction boundary, the parent must again be the current allocation domain. Its frame must occupy the active stack prefix, and every slot beyond the depth must be empty. A fixture checks those properties in two workloads in one compiled Eshkol program. Each allocates in a child region and then continues using the parent.

From the repository root, run the model and the frozen observations with Python 3 and Lean 4.35.0 on PATH (or pass `--lean` with the binary path).

```sh
python3 verification/nested-regions/review.py
```

For a fresh native replay, set `ESHKOL_RELEASE` to the unpacked Linux x64 lite release. Its archive SHA-256 is `a1c467edf5ee519ab36d3f1a24df2eb2cba80211e303fec5be25a49f223d5f3b`. The host needs the release compiler's LLVM 21 shared libraries on its loader path, a C++20 compiler, GNU binutils and linkable BLAS with CBLAS, OpenSSL crypto, PNG, JPEG and WebP. On systems that split CBLAS into a separate shared library, append `--cblas /path/to/libcblas.so`. [Eshkol release archive](https://github.com/tsotchke/eshkol/releases/download/v1.3.6-evolve/eshkol-v1.3.6-evolve-linux-x64-lite.tar.gz)

```sh
python3 verification/nested-regions/review.py --release "$ESHKOL_RELEASE"
```

**What Lean checks.** In `proof/NestedRegions.lean`, `pop_layout` shows that clearing the child slot in a stack that fits its capacity gives the parent layout. `pop_child` assumes a valid layout, child ownership and a reclamation permit for the modeled survivors. It derives parent restoration, the child's absence from the resulting slots and separation of those survivors from the rewound extent. Both theorems use `propext` and `Quot.sound`. The driver checks the five frozen proof source digests, builds them and prints the axiom reports.

**What runs.** The native fixture compiles two applications of the BLC evaluator. It observes ten events per workload, including the active depth, all 64 physical slots, allocator route and arena extents. The parent cursor is 48 bytes before the child, 48 after its return and 96 after the next pair allocation. The two child cursors at destruction are 2,352 and 16,576 bytes. The normal forms are `0010` and `0000110`. The extra evaluator module lives under `fixture/` and is used only by this experiment.

**What fails.** For each child, Lean accepts the measured binding. It rejects three changes to its snapshot individually: selecting the fallback as current domain, dropping the parent from the active prefix, and retaining the child in the final inactive slot. That last case has the right domain and depth.

**Evidence boundary.** `observations.json` pins the historical source identities and measurements. Its source list describes the original recording. The packaged driver checks the compiler, runtime archive, standard library, modules and named headers before building the fresh workload. `native-receipt.json` records a Linux x86-64 replay with GCC 16.2.1, GNU ld 2.47, glibc 2.44 and Python 3.14.9. Its linked ELF digest differs from the historical one. Both sets of observations and decisions agree. The native observations cover two single-thread executions. Establishing a complete inventory of native survivors and proving that the runtime implements the Lean transition remain open.
