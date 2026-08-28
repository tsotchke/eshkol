/**
 * @file vm_limits.c
 * @brief Bytecode VM execution-limit enforcement: the runaway-instruction
 *        guard and the cooperative timeout checkpoint.
 *
 * Split out of vm_run.c so the dispatch loop holds instruction semantics and
 * this file holds the policy that decides when a run must stop. The compile-
 * time ceilings this enforces at run time are declared in
 * inc/eshkol/backend/vm_limits.h; the counters live on the VM so they survive
 * the longjmp a continuation invoke performs.
 *
 */

/* ===== SW-10: VM runaway-instruction guard and timeout checkpoint =====
 *
 * `ESHKOL_VM_MAX_INSN` is documented as the VM's runaway-instruction guard,
 * default 10,000,000. Only the MSVC `switch` fallback below ever had a guard,
 * it was hard-coded to 10M with no environment override, and the computed-goto
 * path that every GCC/Clang build actually runs had none at all — so the two
 * dispatch implementations of one interpreter disagreed about when a program
 * is runaway. Both now share the counters below.
 *
 * Cost discipline: the per-instruction work is a single decrement-and-branch
 * on a local budget. Everything real — comparing the total against the cap,
 * polling for a timeout interrupt — happens once per VM_CHECK_INTERVAL
 * instructions inside vm_limits_checkpoint(), which is not inlined into the
 * dispatch path. The environment variable is read once per vm_run(), not once
 * per instruction. */
#define VM_CHECK_INTERVAL 4096u

/* VM-OWNED limit state.
 *
 * These deliberately do NOT reach into the hosted resource-limit layer. The VM
 * sources are freestanding-safe and are also compiled to WebAssembly, where
 * lib/core/resource_limits.cpp is not linked at all — an earlier cut of this
 * fix called eshkol_get_limits() straight from here and every WASM run died
 * with `missing function: eshkol_get_limits`, taking the whole
 * wasm_execute_diff_oracle corpus (65 programs) down with it.
 *
 * So the dependency runs the other way: hosted entry points RESOLVE the
 * configuration and PUSH it in via eshkol_vm_install_limits(). A build that
 * never calls that installer — WASM, or any freestanding profile — keeps the
 * compiled-in defaults below and links nothing extra. */
uint64_t g_eshkol_vm_max_insn = ESHKOL_VM_DEFAULT_MAX_INSN;
int      g_eshkol_vm_insn_limit_active = 0;   /* opt-in, like every ceiling */
int      g_eshkol_vm_enforce_hard_limits = 1;
void   (*g_eshkol_vm_poll_interrupt)(void) = 0;

/** Install the resolved limit configuration from a hosted entry point.
 *
 * @param max_insn   Instruction ceiling (0 = unlimited).
 * @param active     Non-zero if ESHKOL_VM_MAX_INSN was actually asked for.
 * @param enforce    Non-zero to terminate rather than merely record a breach.
 * @param poll       Cooperative timeout poll, or NULL for none. */
void eshkol_vm_install_limits(uint64_t max_insn, int active, int enforce,
                              void (*poll)(void)) {
    g_eshkol_vm_max_insn = max_insn;
    g_eshkol_vm_insn_limit_active = active;
    g_eshkol_vm_enforce_hard_limits = enforce;
    g_eshkol_vm_poll_interrupt = poll;
}

/** Periodic limit checkpoint. Returns 1 if the VM should keep running. */
static int vm_limits_checkpoint(VM* vm, uint64_t* executed, uint64_t max_insn) {
    *executed += VM_CHECK_INTERVAL;

    if (max_insn > 0 && *executed > max_insn && g_eshkol_vm_insn_limit_active) {
        char detail[128];
        snprintf(detail, sizeof(detail),
                 "bytecode VM executed %llu instructions (pc=%d)",
                 (unsigned long long)*executed, vm->pc);
        fflush(stdout);
        fprintf(stderr, "eshkol: fatal: %s, limit %llu, set by ESHKOL_VM_MAX_INSN\n",
                detail, (unsigned long long)max_insn);
        fflush(stderr);
        if (g_eshkol_vm_enforce_hard_limits) {
            _Exit(ESHKOL_EXIT_LIMIT_VM_INSN);
        }
        vm->error = 1;  /* advisory mode: stop this run, do not kill the process */
        return 0;
    }

    /* Same cooperative timeout poll the native engine does at loop back-edges;
     * the watchdog thread can only request the interrupt, someone has to act.
     * NULL in builds with no hosted runtime (WASM, freestanding). */
    if (g_eshkol_vm_poll_interrupt) g_eshkol_vm_poll_interrupt();
    return 1;
}
