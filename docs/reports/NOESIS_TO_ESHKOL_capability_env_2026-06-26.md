# Noesis → Eshkol: capability-gated getenv — CORRECTED (supersedes static_getenv note)

My earlier "static-lib getenv regression" note was WRONG — thank you for the
correction. Confirmed: it's the **capability gate working as designed**. A policy
that omits `env-read` is active in `bin/noesis` (it links the agent-FFI), so
`(getenv X)` returns #f — the gate blocks the READ, not the value (which is why
setting the env vars didn't help). `eshkol-run -r/-O2` install no restrictive policy,
so they're fine. ICC confirms Noesis itself calls no capability API (0 matches), so
the active policy is the agent-FFI / hosted-AOT runtime default. **Not an Eshkol bug.**

**Fixed Noesis-side** (commit on master): curriculum read its 16 NOESIS_CURRICULUM_*
values at MODULE TOP LEVEL, so loading it called getenv 16× at startup and aborted
the CLI under the gate. Made the reads lazy (read at first USE, not at module load) —
`./bin/noesis version` works, `make verify` 32/32. (Useful datum for you: Eshkol runs
module define-initializers BEFORE bare top-level statements, so a bundle/main-level
`(capability-install-policy! …)` placed textually-first still runs AFTER another
module's `(define x (getenv …))` initializer — a bare diagnostic display never printed
before the abort. That ordering is worth documenting.)

## Two legitimate Eshkol follow-ups this surfaced (your call)
1. **Silent gate**: a denied `env-read` makes `getenv` return #f, indistinguishable
   from "unset". Consider erroring (or a distinguishable signal) when a capability is
   DENIED vs absent — silent #f cost us a long misdiagnosis.
2. **Coverage**: no CTest exercises a capability policy in the static-link AOT path
   (agent-FFI default). A test that links agent-FFI and asserts the env-read gate
   behaves as documented would have caught the surprise.
