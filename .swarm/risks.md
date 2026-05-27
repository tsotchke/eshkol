# Eshkol Swarm Risks — v1.3-evolve

| Risk | Status | Mitigation |
|---|---|---|
| Parallel agents make conflicting design decisions about multi-shot `call/cc` or stack-frame metadata. | active | Lock both designs in `decisions.md` before dispatching tasks that touch them. |
| Per-platform native-media work diverges in API shape across macOS / Linux / Windows. | active | One spec (`ESH-0006`) defines the shared `image-read` / `image-write` / `image-resize` Eshkol-side contract; per-platform tasks must conform. |
| `(the-environment)` semantics drift between `eval`-with-env path and the captured-by-compile-time path. | active | One acceptance test verifies `(eval expr (the-environment))` is equivalent to evaluating `expr` in the surrounding `let` body. |
| Agents commit code that passes their local tests but breaks the ICC `v1.3-evolve` oracle. | active | Every task's `verification` runs `scripts/run_icc_smoke.sh` + the oracle check; nothing merges without the named probe at PASS. |
| Mesh node SSH / Tailscale outage interrupts a long-running parallel task. | active | All tasks are resumable. The claim file records the last-good commit; restart picks up from there. `bin/mesh status` health-check before dispatching. |
| Stack-copying `call/cc` interacts incorrectly with OALR region lifetimes. | active | Decision recorded: continuations pin live regions via refcount. Acceptance test exercises capture-after-region-pop. |
| AD tape state diverges across multi-shot continuation invocations. | active | Decision recorded: `call/cc` capture is forbidden inside `gradient` / `derivative` for v1.3. Runtime trap + clear error. |
| Persistent JIT cache poisoning (cached object from a now-buggy source tree). | active | Cache key includes content hash of every input source + every stdlib module + LLVM version. Mismatch triggers full recompile. |
| `tsotchke-chan ask` budget overrun on a long task. | active | Every task spec sets `--max-cost`. Codex-main monitors costs and pauses dispatch if a single task crosses 2× its budget. |
| Public-tree leakage of mesh node hostnames / IPs / Tailscale identifiers via swarm artifacts. | active | Mirror QGTL's hygiene policy: hostnames stay in `runtime/` (gitignored); swarm tasks reference abstract roles (`linux-x64`, `linux-arm64`, `darwin-arm64`, `windows-x64`). |
| Long-running codex-main runs out of disk on enki. | active | Enki disk monitored via `tsotchke-chan online`. Runtime artifacts > 7d are GC'd; events.jsonl rotates at 10 MB. |
