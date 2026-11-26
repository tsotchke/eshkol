# Rank-two matmul shape provenance

The inner-dimension equality is checked at the actual multiplication site. A diagnostic about the tensor previously bound there cannot authorize its replacement. In the alias fixture the first multiplication succeeds, another reference changes the right operand, and the next multiplication at the same site fails.

Run the suite from the repository root with the R0 checker and the corrected C FFI tensor constructors present.

```sh
sh verification/matmul/check.sh
```

Use Linux x86-64, Node 22, Python 3.10 or newer, Lean 4.35.0 including leanexport, GNU C and C++, and the Eshkol v1.3.6-evolve x64 lite distribution with matching LLVM 21 libraries. Put the compiler and Lean tools on PATH. The script finds the runtime archive beside the installed compiler.

| Experiment | What it checks |
|---|---|
| `replay.cjs` | Generate the dimension decisions in Lean. Compare all 256 decisions with the C++ helper and shipped WASM VM. Mutate the helper to compare the left axis with itself. |
| `shape-writers.py` | Check the pinned source inventory and replay the shape-view witness. On the observed native path a new header can point to the original axis storage while owning separate elements. Use `--list` to inspect lexical candidates. |
| `guard-replay.py` | Bind a proof to its site, compare the aliased native calls and follow both the inner-equality and size predicates into the emitted branch. |
| `publication-replay.py` | Compare the modeled ownership and dimension histories with six evacuation cases. A weakened relocation rule fails the preservation theorem. |
| `eligibility-replay.py` | Tie foreign exposure to the actual right operand and test eligibility at each invocation. Letting an unknown transfer retain authority breaks the revocation theorem. |

The inventory fingerprints the scanned native sources and shipped runtime surfaces. The C++ probe uses this checkout’s source with the release allocator. The emitted-IR check concerns the released native compiler. The models assume the described bindings and publication events. Production guard elimination requires an implementation that consumes those bindings and authorizations at the site.
