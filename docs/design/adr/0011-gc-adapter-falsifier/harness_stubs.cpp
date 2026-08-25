/*
 * Falsifier F1 link stubs (ADR-0011).
 *
 * The arena allocator under test is lib/core/runtime_arena_core.cpp, linked
 * UNMODIFIED. It calls into two subsystems this harness deliberately does not
 * exercise: process-wide resource accounting / interrupt delivery
 * (inc/eshkol/core/resource_limits.h) and the parallel-worker predicate.
 * Linking the real ones would pull in the runtime's signal, timer, and
 * shutdown machinery for no experimental benefit.
 *
 * Both are orthogonal to the property under test. Neither influences bump
 * placement, block sizing, scope rewind, or teardown -- the accounting hooks
 * only tally bytes, and the worker predicate only selects a thread-local arena
 * this harness never installs. The measurements in ADR-0011 section 9 come
 * from arena_get_total_memory() and the OS resident-size counter, not from
 * anything here.
 */
#include <cstddef>
#include <cstdint>

extern "C" {

/* Process-wide byte accounting: tally only. True == "within budget". */
bool eshkol_track_allocation(size_t /*bytes*/)   { return true; }
void eshkol_track_deallocation(size_t /*bytes*/) {}

/* No limit is configured in the harness, so nothing is ever enforced. */
bool eshkol_limit_is_active(uint32_t /*which*/)  { return false; }
void eshkol_limit_enforce(int /*error*/, const char* /*detail*/) {}

/* The harness runs single-threaded and installs no worker arena. */
int arena_is_worker_thread(void) { return 0; }

}
