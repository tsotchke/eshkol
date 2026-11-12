# Subprocess review revision

The PROCESS module has reviewed contracts for all 40 exports and a matching source pin. Scoped stream callbacks borrow a PROCESS_STREAM owner; cleanup IDs inside aggregate data are explicitly unsupported in strict mode. The module no longer bypasses contract checking through a stale pin.

The stream adapter uses the shared identity registry and retires only its own records. Failed watch removal preserves descriptors, the loop retain and lifetime lease until recovery; generic resource close refuses to retire PROCESS cleanup ownership. The [manifest](subprocess-review-revision-2026-11-12.json) records four fresh scoped CTests, native and sanitizer suites, checker syntax validation and the compiled PROCESS cases. The hosted native Linux/macOS socket workflow also executes the PROCESS source. This is not new public macOS JIT/AOT or package/release qualification.

This branch depends on the independent integration repairs and shared lease foundation. Historical production evidence keeps its original result/source hashes; raw machine logs are retained locally and summaries use portable placeholders.
