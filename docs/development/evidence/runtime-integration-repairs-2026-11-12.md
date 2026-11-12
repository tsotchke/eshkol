# Runtime integration repairs

The Taylor result-shape epoch output uses the stored 64-bit epoch type; refinement uses its explicit `<algorithm>` dependency; resident runtime units access the current memory context through the exported accessor. The JIT stdlib cache publishes completed objects from process-unique files, so concurrent writers never share one temporary name.

A fresh isolated Linux Release build of `eshkol-run` and `stdlib` passed. Five selected Taylor, refinement and resident JIT/AOT CTests passed. Three concurrent JIT writers executed against a cold private cache, published one valid object and left no temporary files. The [manifest](runtime-integration-repairs-2026-11-12.json) pins source and original local-log hashes. No raw machine logs are committed.

This branch contains only the four integration changes and this summary. It is based on canonical integration `e2c83bd48`; it contains no socket or subprocess implementation changes. macOS and full-release qualification are separate.
