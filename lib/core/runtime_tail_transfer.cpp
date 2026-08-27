/**
 * @file runtime_tail_transfer.cpp
 * @brief Per-thread tail-transfer record: the state a general tail call hands
 *        to its driver loop (ADR-0000 item 3 / ADR-0006 section 3).
 *
 * WHY THIS EXISTS
 * ---------------
 * `musttail` gives O(1)-stack mutual tail recursion only when caller and callee
 * have byte-identical LLVM signatures, no argument points into the caller's
 * frame, and the target backend can lower an aggregate return that way. Several
 * call shapes fail one of those conditions and were therefore bounded, growing
 * one native frame per hop:
 *
 *   1. mutually recursive procedures with DIFFERENT ARITIES;
 *   2. tail calls through `guard` (musttail discards the frame and would skip
 *      the handler-stack pop that leaving a guard owes);
 *   3. every non-AArch64 target (LLVM 21 refuses aggregate-return musttail).
 *
 * The tail-transfer protocol removes the condition instead of the call. At a
 * transfer site the caller does NOT call the callee. It copies the evaluated
 * arguments into this record, records the callee's uniform entry, sets
 * `pending`, and RETURNS NORMALLY. Returning normally is the whole point:
 * every epilogue the frame owes -- the `guard` handler pop above all -- runs
 * exactly as it would on any other return. The nearest driver loop (emitted
 * into the public entry of every function that contains a transfer site) then
 * sees `pending`, clears it, and invokes the recorded uniform entry. One frame
 * is live per hop and it is reused, so the chain is O(1) in native stack
 * regardless of arity or target.
 *
 * WHY IT IS PER-THREAD
 * --------------------
 * `parallel-map` and friends run compiled closures on worker threads. A shared
 * record would let two workers overwrite each other's pending transfer -- a
 * silent wrong answer, the one failure class this work is not allowed to
 * introduce. The record is `thread_local`, so each worker drives its own chain.
 *
 * WHY THE ARGUMENT BUFFER IS OWNED HERE
 * -------------------------------------
 * ADR-0006 section 3: "no pointer into the discarded caller frame reaches the
 * callee." Arguments are COPIED into this buffer by value; the uniform entry
 * loads every one of them into SSA values before it calls the real body, so a
 * transfer performed by that body may overwrite the buffer safely.
 *
 * The struct itself lives in inc/eshkol/eshkol.h so that the code generator and
 * the runtime compute one set of field offsets from one declaration, the way
 * codegen already reads eshkol_continuation_state_t.
 */

#include "eshkol/eshkol.h"

extern "C" {

/**
 * @brief The calling thread's tail-transfer record.
 *
 * Zero-initialised, so `pending == 0` on a thread that has never transferred.
 * Never freed: static storage duration, one instance per thread.
 */
static thread_local eshkol_tail_transfer_t t_tail_transfer;

/**
 * @brief Address of the calling thread's tail-transfer record.
 *
 * Compiled code calls this once per activation that needs the record (a
 * transfer site, or a driver loop) and then addresses the fields directly, so
 * the protocol costs one call per participating activation rather than one
 * call per field access.
 *
 * @return Never NULL.
 */
eshkol_tail_transfer_t* eshkol_tail_transfer_slot(void) {
    return &t_tail_transfer;
}

}  // extern "C"
