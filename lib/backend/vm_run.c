/**
 * @brief Main bytecode interpreter loop: executes @p vm->code starting at
 *        @p vm->pc until OP_HALT, an error, or (on GCC/Clang) via
 *        computed-goto threaded dispatch — each opcode handler ends by
 *        jumping directly to the next handler through a label-address
 *        table, avoiding switch-statement bounds checks/indirect-jump
 *        overhead. Falls back to a plain `switch` dispatch loop on other
 *        compilers. Implements the full instruction set (stack/arithmetic/
 *        comparison ops, locals/upvalues, closures/calls/tail-calls,
 *        control flow, pairs/vectors/strings, native-call dispatch to the
 *        vm_*_dispatch() runtime modules, exception handling, and
 *        continuations) directly inline in this function body.
 */
/**
 * @brief Is @p v an EXACT number (integer, bignum, rational, i128)?
 *
 * R7RS exactness decides the division-by-zero policy, so OP_DIV needs it:
 * exact-by-exact-zero is a fatal "division by zero", while a single INEXACT
 * operand makes the whole operation IEEE-754 float division, which must
 * produce +nan.0 / ±inf.0 exactly as the native backend does. Anything
 * non-numeric answers 0 (inexact) so it cannot turn a float division into a
 * spurious error.
 */
static int vm_is_exact_number(Value v) {
    return v.type == VAL_INT || v.type == VAL_BIGNUM ||
           v.type == VAL_RATIONAL || v.type == VAL_I128;
}

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

static size_t vm_continuation_allocation_size(const VM* vm) {
    return sizeof(VmContinuation) +
        (size_t)vm->sp * sizeof(Value) +
        (size_t)vm->frame_count * sizeof(CallFrame) +
        (size_t)vm->n_winds * 2 * sizeof(Value) +
        (size_t)vm->n_parameter_bindings * 2 * sizeof(Value);
}

static void vm_capture_continuation_dynamic_state(VM* vm,
                                                  VmContinuation* cont) {
    char* cursor = (char*)cont->saved_frames +
        (size_t)vm->frame_count * sizeof(CallFrame);
    cont->n_winds = vm->n_winds;
    cont->n_parameter_bindings = vm->n_parameter_bindings;
    cont->saved_wind_befores = (Value*)cursor;
    cursor += (size_t)cont->n_winds * sizeof(Value);
    cont->saved_wind_afters = (Value*)cursor;
    cursor += (size_t)cont->n_winds * sizeof(Value);
    cont->saved_parameter_bindings = (Value*)cursor;
    cursor += (size_t)cont->n_parameter_bindings * sizeof(Value);
    cont->saved_parameter_values = (Value*)cursor;

    for (int i = 0; i < cont->n_winds; i++) {
        cont->saved_wind_befores[i] = vm->wind_stack[i].before;
        cont->saved_wind_afters[i] = vm->wind_stack[i].after;
    }
    for (int i = 0; i < cont->n_parameter_bindings; i++) {
        Value parameter_value = vm->parameter_bindings[i];
        VmParameter* parameter = vm_parameter_from_value(vm, parameter_value);
        cont->saved_parameter_bindings[i] = parameter_value;
        if (parameter) vm_param_ref(parameter, &cont->saved_parameter_values[i]);
        else cont->saved_parameter_values[i] = NIL_VAL;
    }
}

static void vm_restore_continuation_dynamic_state(VM* vm,
                                                  const VmContinuation* cont) {
    /* Parameter values live outside the VM execution stack.  Rebuild their
     * dynamic stacks from the continuation snapshot before resuming code;
     * merely restoring a binding depth would otherwise leave captured
     * parameterize extents pointing at the values of the abandoned path. */
    vm_unwind_parameter_bindings(vm, 0);

    for (int i = 0; i < cont->n_winds; i++) {
        vm->wind_stack[i].before = cont->saved_wind_befores[i];
        vm->wind_stack[i].after = cont->saved_wind_afters[i];
    }
    vm->n_winds = cont->n_winds;

    for (int i = 0; i < cont->n_parameter_bindings; i++) {
        Value parameter_value = cont->saved_parameter_bindings[i];
        VmParameter* parameter = vm_parameter_from_value(vm, parameter_value);
        if (!parameter) {
            vm->error = 1;
            return;
        }
        vm_param_push(&vm->heap.regions, parameter,
                      &cont->saved_parameter_values[i]);
        if (!vm_record_parameter_binding(vm, parameter_value)) return;
    }
}

void vm_run(VM* vm) {
    const int owns_native_escape = !vm->native_escape_ready;
    if (owns_native_escape) {
        vm->native_escape_ready = 1;
        if (setjmp(vm->native_escape_jmp) != 0) {
            /* A handled raise or continuation crossed one or more native C
             * helper frames.  Its handler/continuation already restored pc,
             * stack, frames, winds, parameters, and promise state; resume the
             * owning interpreter loop from that exact state. */
            vm->native_call_depth = 0;
            vm->halted = 0;
            vm->error = 0;
        }
    }
#if defined(__GNUC__) || defined(__clang__)
/* =========================================================================
 * Computed-goto (threaded) dispatch — GCC/Clang only.
 *
 * Each handler ends with DISPATCH() which fetches the next instruction and
 * jumps directly to its handler via the dispatch_table, eliminating the
 * switch overhead (no bounds check, no indirect branch through a jump table
 * generated by the compiler — just a single indirect goto).
 * ========================================================================= */

    static void* dispatch_table[OP_COUNT] = {
        [OP_NOP]           = &&lbl_NOP,
        [OP_CONST]         = &&lbl_CONST,
        [OP_NIL]           = &&lbl_NIL,
        [OP_TRUE]          = &&lbl_TRUE,
        [OP_FALSE]         = &&lbl_FALSE,
        [OP_POP]           = &&lbl_POP,
        [OP_DUP]           = &&lbl_DUP,
        [OP_ADD]           = &&lbl_ADD,
        [OP_SUB]           = &&lbl_SUB,
        [OP_MUL]           = &&lbl_MUL,
        [OP_DIV]           = &&lbl_DIV,
        [OP_MOD]           = &&lbl_MOD,
        [OP_NEG]           = &&lbl_NEG,
        [OP_ABS]           = &&lbl_ABS,
        [OP_EQ]            = &&lbl_EQ,
        [OP_LT]            = &&lbl_LT,
        [OP_GT]            = &&lbl_GT,
        [OP_LE]            = &&lbl_LE,
        [OP_GE]            = &&lbl_GE,
        [OP_NOT]           = &&lbl_NOT,
        [OP_GET_LOCAL]     = &&lbl_GET_LOCAL,
        [OP_SET_LOCAL]     = &&lbl_SET_LOCAL,
        [OP_GET_UPVALUE]   = &&lbl_GET_UPVALUE,
        [OP_SET_UPVALUE]   = &&lbl_SET_UPVALUE,
        [OP_CLOSURE]       = &&lbl_CLOSURE,
        [OP_CALL]          = &&lbl_CALL,
        [OP_TAIL_CALL]     = &&lbl_TAIL_CALL,
        [OP_RETURN]        = &&lbl_RETURN,
        [OP_JUMP]          = &&lbl_JUMP,
        [OP_JUMP_IF_FALSE] = &&lbl_JUMP_IF_FALSE,
        [OP_LOOP]          = &&lbl_LOOP,
        [OP_CONS]          = &&lbl_CONS,
        [OP_CAR]           = &&lbl_CAR,
        [OP_CDR]           = &&lbl_CDR,
        [OP_NULL_P]        = &&lbl_NULL_P,
        [OP_PRINT]         = &&lbl_PRINT,
        [OP_HALT]          = &&lbl_HALT,
        [OP_NATIVE_CALL]   = &&lbl_NATIVE_CALL,
        [OP_CLOSE_UPVALUE] = &&lbl_CLOSE_UPVALUE,
        [OP_VEC_CREATE]    = &&lbl_VEC_CREATE,
        [OP_VEC_REF]       = &&lbl_VEC_REF,
        [OP_VEC_SET]       = &&lbl_VEC_SET,
        [OP_VEC_LEN]       = &&lbl_VEC_LEN,
        [OP_STR_REF]       = &&lbl_STR_REF,
        [OP_STR_LEN]       = &&lbl_STR_LEN,
        [OP_PAIR_P]        = &&lbl_PAIR_P,
        [OP_NUM_P]         = &&lbl_NUM_P,
        [OP_STR_P]         = &&lbl_STR_P,
        [OP_BOOL_P]        = &&lbl_BOOL_P,
        [OP_PROC_P]        = &&lbl_PROC_P,
        [OP_VEC_P]         = &&lbl_VEC_P,
        [OP_SET_CAR]       = &&lbl_SET_CAR,
        [OP_SET_CDR]       = &&lbl_SET_CDR,
        [OP_POPN]          = &&lbl_POPN,
        [OP_OPEN_CLOSURE]  = &&lbl_NOP,
        [OP_CALLCC]        = &&lbl_CALLCC,
        [OP_INVOKE_CC]     = &&lbl_INVOKE_CC,
        [OP_PUSH_HANDLER]  = &&lbl_PUSH_HANDLER,
        [OP_POP_HANDLER]   = &&lbl_POP_HANDLER,
        [OP_GET_EXN]       = &&lbl_GET_EXN,
        [OP_PACK_REST]     = &&lbl_PACK_REST,
        [OP_WIND_PUSH]     = &&lbl_WIND_PUSH,
        [OP_WIND_POP]      = &&lbl_WIND_POP,
        [OP_VOID]          = &&lbl_VOID,
        [OP_LANGUAGE_COVERAGE] = &&lbl_LANGUAGE_COVERAGE,
        [OP_LANGUAGE_COVERAGE_CALL] = &&lbl_LANGUAGE_COVERAGE_CALL,
    };

    #define DISPATCH() do { \
        if (vm->halted || vm->error || vm->pc >= vm->code_len) goto vm_exit; \
        if (--vm_check_budget == 0) { \
            vm_check_budget = VM_CHECK_INTERVAL; \
            if (!vm_limits_checkpoint(vm, &vm_insns_executed, vm_max_insn)) \
                goto vm_exit; \
        } \
        instr = vm->code[vm->pc++]; \
        goto *dispatch_table[instr.op]; \
    } while(0)

    Instr instr;
    uint64_t vm_insns_executed = 0;
    const uint64_t vm_max_insn = g_eshkol_vm_max_insn;
    unsigned vm_check_budget = VM_CHECK_INTERVAL;
    DISPATCH();

    /* --- Constants & Stack --- */

    lbl_NOP:
        DISPATCH();

    lbl_CONST:
        if (instr.operand < 0 || instr.operand >= vm->n_constants) {
            fprintf(stderr, "INVALID CONSTANT INDEX %d\n", instr.operand);
            vm->error = 1; goto vm_exit;
        }
        /* A literal constant is never a tape node; clear any stale mapping on
         * its slot so reverse-mode AD treats it as an ad_const operand rather
         * than reusing a prior node index left in ad_node_map (only when a tape
         * is active — a NULL check keeps non-AD execution untouched). */
        if (vm->active_tape && vm->sp >= 0 && vm->sp < STACK_SIZE)
            vm->ad_node_map[vm->sp] = -1;
        vm_push(vm, vm->constants[instr.operand]);
        DISPATCH();

    lbl_NIL:   vm_push(vm, NIL_VAL);     DISPATCH();
    lbl_TRUE:  vm_push(vm, BOOL_VAL(1)); DISPATCH();
    lbl_FALSE: vm_push(vm, BOOL_VAL(0)); DISPATCH();
    lbl_POP:   vm_pop(vm);               DISPATCH();
    lbl_DUP:   vm_push(vm, vm_peek(vm, 0)); DISPATCH();

    /* --- Arithmetic ---
     *
     * Reverse-mode AD tracing: when vm->active_tape is set, binary/unary
     * operations record on the Wengert tape in addition to computing values.
     * ad_node_map[stack_slot] tracks which tape node corresponds to each
     * stack value (-1 = untracked). Untracked operands that interact with
     * tracked ones are promoted to ad_const nodes on the tape.
     */

#define VM_AD_BINARY(vm, a_sp, b_sp, tape_fn, result_val) do { \
    if ((vm)->active_tape) { \
        AdTape* _t = (AdTape*)(vm)->active_tape; \
        int _an = (vm)->ad_node_map[(a_sp)]; \
        int _bn = (vm)->ad_node_map[(b_sp)]; \
        if (_an != -1 || _bn != -1) { \
            if (_an == -1) _an = ad_const(_t, as_number((vm)->stack[(a_sp)])); \
            if (_bn == -1) _bn = ad_const(_t, as_number((vm)->stack[(b_sp)])); \
            (vm)->ad_node_map[(vm)->sp] = tape_fn(_t, _an, _bn); \
        } else { (vm)->ad_node_map[(vm)->sp] = -1; } \
    } else { (vm)->ad_node_map[(vm)->sp] = -1; } \
} while(0)

#define VM_AD_UNARY(vm, a_sp, tape_fn) do { \
    if ((vm)->active_tape) { \
        AdTape* _t = (AdTape*)(vm)->active_tape; \
        int _an = (vm)->ad_node_map[(a_sp)]; \
        if (_an != -1) { \
            (vm)->ad_node_map[(vm)->sp] = tape_fn(_t, _an); \
        } else { (vm)->ad_node_map[(vm)->sp] = -1; } \
    } else { (vm)->ad_node_map[(vm)->sp] = -1; } \
} while(0)

    lbl_ADD: { int b_sp = vm->sp - 1, a_sp = vm->sp - 2;
        Value b = vm_pop(vm), a = vm_pop(vm);
        /* SW-09: neither operand check below recognizes VAL_I128, so a
         * generic `+` over i128 values used to fall all the way through to
         * the double path, where as_number_vm() reads a heap-boxed i128 as
         * 0.0 — silently answering 0 with exit 0. The VM has no i128
         * opcodes (that is v1.3.5 scope); raise instead of fabricating a
         * result. The native engine already raises for the same case
         * (LE-03, "Type error in +: expected number, vector, or tensor"). */
        if (a.type == VAL_I128 || b.type == VAL_I128) {
            vm_raise_error_msg(vm,
                "+: i128 arithmetic is not supported on the VM (no i128 opcodes "
                "are implemented in the bytecode interpreter); use the native "
                "backend");
            DISPATCH();
        }
        if (a.type == VAL_HYPER_DUAL || b.type == VAL_HYPER_DUAL) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 1905); }
        else if (a.type == VAL_DUAL || b.type == VAL_DUAL) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 373); }
        else if (a.type == VAL_RATIONAL || b.type == VAL_RATIONAL) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 331); }
        else if (a.type == VAL_COMPLEX || b.type == VAL_COMPLEX) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 307); }
        else if (vm_either_bignum(a, b)) { vm->ad_node_map[vm->sp] = -1; vm_bignum_arith(vm, a, b, '+'); }
        else if (a.type == VAL_INT && b.type == VAL_INT) { int64_t r; VM_AD_BINARY(vm, a_sp, b_sp, ad_add, 0);
            if (__builtin_add_overflow(a.as.i, b.as.i, &r)) vm_bignum_arith(vm, a, b, '+'); else vm_push(vm, INT_VAL(r)); }
        else { VM_AD_BINARY(vm, a_sp, b_sp, ad_add, 0);
            vm_push(vm, number_val_contagious(a, b, as_number_vm(vm, a) + as_number_vm(vm, b))); } DISPATCH(); }
    lbl_SUB: { int b_sp = vm->sp - 1, a_sp = vm->sp - 2;
        Value b = vm_pop(vm), a = vm_pop(vm);
        /* SW-09b: same family as lbl_ADD's guard — every arithmetic/
         * comparison opcode that falls through to as_number_vm() misreads
         * a heap-boxed VAL_I128 as 0.0. */
        if (a.type == VAL_I128 || b.type == VAL_I128) {
            vm_raise_error_msg(vm,
                "-: i128 arithmetic is not supported on the VM (no i128 opcodes "
                "are implemented in the bytecode interpreter); use the native "
                "backend");
            DISPATCH();
        }
        if (a.type == VAL_HYPER_DUAL || b.type == VAL_HYPER_DUAL) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 1906); }
        else if (a.type == VAL_DUAL || b.type == VAL_DUAL) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 374); }
        else if (a.type == VAL_RATIONAL || b.type == VAL_RATIONAL) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 332); }
        else if (a.type == VAL_COMPLEX || b.type == VAL_COMPLEX) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 308); }
        else if (vm_either_bignum(a, b)) { vm->ad_node_map[vm->sp] = -1; vm_bignum_arith(vm, a, b, '-'); }
        else if (a.type == VAL_INT && b.type == VAL_INT) { int64_t r; VM_AD_BINARY(vm, a_sp, b_sp, ad_sub, 0);
            if (__builtin_sub_overflow(a.as.i, b.as.i, &r)) vm_bignum_arith(vm, a, b, '-'); else vm_push(vm, INT_VAL(r)); }
        else { VM_AD_BINARY(vm, a_sp, b_sp, ad_sub, 0);
            vm_push(vm, number_val_contagious(a, b, as_number_vm(vm, a) - as_number_vm(vm, b))); } DISPATCH(); }
    lbl_MUL: { int b_sp = vm->sp - 1, a_sp = vm->sp - 2;
        Value b = vm_pop(vm), a = vm_pop(vm);
        /* SW-09b: see lbl_ADD/lbl_SUB. */
        if (a.type == VAL_I128 || b.type == VAL_I128) {
            vm_raise_error_msg(vm,
                "*: i128 arithmetic is not supported on the VM (no i128 opcodes "
                "are implemented in the bytecode interpreter); use the native "
                "backend");
            DISPATCH();
        }
        if (a.type == VAL_HYPER_DUAL || b.type == VAL_HYPER_DUAL) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 1907); }
        else if (a.type == VAL_DUAL || b.type == VAL_DUAL) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 375); }
        else if (a.type == VAL_RATIONAL || b.type == VAL_RATIONAL) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 333); }
        else if (a.type == VAL_COMPLEX || b.type == VAL_COMPLEX) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 309); }
        else if (vm_either_bignum(a, b)) { vm->ad_node_map[vm->sp] = -1; vm_bignum_arith(vm, a, b, '*'); }
        else if (a.type == VAL_INT && b.type == VAL_INT) { int64_t r; VM_AD_BINARY(vm, a_sp, b_sp, ad_mul, 0);
            if (__builtin_mul_overflow(a.as.i, b.as.i, &r)) vm_bignum_arith(vm, a, b, '*'); else vm_push(vm, INT_VAL(r)); }
        else { VM_AD_BINARY(vm, a_sp, b_sp, ad_mul, 0);
            vm_push(vm, number_val_contagious(a, b, as_number_vm(vm, a) * as_number_vm(vm, b))); } DISPATCH(); }
    lbl_DIV: { int b_sp = vm->sp - 1, a_sp = vm->sp - 2;
        Value b = vm_pop(vm), a = vm_pop(vm);
        /* SW-09b: see lbl_ADD/lbl_SUB/lbl_MUL. */
        if (a.type == VAL_I128 || b.type == VAL_I128) {
            vm_raise_error_msg(vm,
                "/: i128 arithmetic is not supported on the VM (no i128 opcodes "
                "are implemented in the bytecode interpreter); use the native "
                "backend");
            DISPATCH();
        }
        if (a.type == VAL_HYPER_DUAL || b.type == VAL_HYPER_DUAL) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 1908); }
        else if (a.type == VAL_DUAL || b.type == VAL_DUAL) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 376); }
        else if (a.type == VAL_RATIONAL || b.type == VAL_RATIONAL) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 334); }
        else if (a.type == VAL_COMPLEX || b.type == VAL_COMPLEX) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 310); }
        else if (a.type == VAL_INT && b.type == VAL_INT) {
            /* exact/exact -> exact result (R7RS): native 334 (rational div)
             * reduces the fraction and collapses denom==1 back to an integer,
             * so (/ 1 3) yields 1/3 and (/ 6 3) yields 2 rather than the
             * inexact float the double path produced. */
            if (b.as.i == 0) { fprintf(stderr, "DIVIDE BY ZERO\n"); vm->error = 1; goto vm_exit; }
            vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 334);
        }
        /* A bignum operand must reach the bignum domain: as_number() reads a
         * heap pointer's .as.i and answers 0.0, so falling through to the
         * double path below made every bignum division silently produce 0. */
        else if (vm_either_bignum(a, b)) { vm->ad_node_map[vm->sp] = -1; vm_bignum_arith(vm, a, b, '/'); if (vm->error) goto vm_exit; }
        else {
        double bd = as_number_vm(vm, b);
        /* Only EXACT-by-exact-zero is an error.  With any inexact operand this
         * is IEEE-754 division and must yield +nan.0 / ±inf.0 like native —
         * erroring here aborted the run and dropped every later top-level
         * form (tests/vm_parity/corpus/37_float_div_zero.esk). */
        if (bd == 0 && vm_is_exact_number(a) && vm_is_exact_number(b)) {
            fprintf(stderr, "DIVIDE BY ZERO\n"); vm->error = 1; goto vm_exit; }
        VM_AD_BINARY(vm, a_sp, b_sp, ad_div, 0);
        vm_push(vm, number_val_contagious(a, b, as_number_vm(vm, a) / bd)); } DISPATCH(); }
    lbl_MOD: {
        Value b = vm_pop(vm), a = vm_pop(vm);
        /* SW-09b: see lbl_ADD. modulo's double path (fmod) reads a
         * heap-boxed VAL_I128 as 0.0 exactly like the other arithmetic ops. */
        if (a.type == VAL_I128 || b.type == VAL_I128) {
            vm_raise_error_msg(vm,
                "modulo: i128 arithmetic is not supported on the VM (no i128 "
                "opcodes are implemented in the bytecode interpreter); use the "
                "native backend");
            DISPATCH();
        }
        if (vm_either_bignum(a, b)) { vm->ad_node_map[vm->sp] = -1; vm_bignum_arith(vm, a, b, 'm'); DISPATCH(); }
        if (a.type == VAL_INT && b.type == VAL_INT) {
            if (b.as.i == 0) { fprintf(stderr, "MODULO BY ZERO\n"); vm->error = 1; goto vm_exit; }
            int64_t r = a.as.i % b.as.i; if (r != 0 && ((r ^ b.as.i) < 0)) r += b.as.i;
            vm_push(vm, INT_VAL(r)); DISPATCH();
        }
        double bd = as_number_vm(vm, b);
        if (bd == 0) { fprintf(stderr, "MODULO BY ZERO\n"); vm->error = 1; goto vm_exit; }
        double r = fmod(as_number_vm(vm, a), bd);
        if (r != 0 && ((r > 0) != (bd > 0))) r += bd;
        vm_push(vm, number_val_contagious(a, b, r));
        DISPATCH();
    }
    lbl_NEG: { int a_sp = vm->sp - 1; Value a = vm_pop(vm);
        /* SW-09b: see lbl_ADD. Unary negate has the same fall-through-to-
         * double shape as the binary ops. */
        if (a.type == VAL_I128) {
            vm_raise_error_msg(vm,
                "-: i128 arithmetic is not supported on the VM (no i128 opcodes "
                "are implemented in the bytecode interpreter); use the native "
                "backend");
            DISPATCH();
        }
        if (a.type == VAL_HYPER_DUAL) { vm_push(vm, a); vm_dispatch_native(vm, 1909); }
        else if (a.type == VAL_DUAL) { vm_push(vm, a); vm_dispatch_native(vm, 384); }
        /* A rational must negate in the rational domain: falling through to the
         * double path read the heap pointer as 0.0, so (- 1/3) answered -0. */
        else if (a.type == VAL_RATIONAL) { vm_push(vm, a); vm_dispatch_native(vm, 335); }
        else if (a.type == VAL_BIGNUM) { vm->ad_node_map[vm->sp] = -1; vm_push_bignum_norm(vm, bignum_neg(&vm->heap.regions, (VmBignum*)vm->heap.objects[a.as.ptr]->opaque.ptr)); }
        else if (a.type == VAL_INT) { VM_AD_UNARY(vm, a_sp, ad_neg);
            if (a.as.i == INT64_MIN) vm_push_bignum_norm(vm, bignum_neg(&vm->heap.regions, bignum_from_int64(&vm->heap.regions, a.as.i)));
            else vm_push(vm, INT_VAL(-a.as.i)); }
        else { VM_AD_UNARY(vm, a_sp, ad_neg); vm_push(vm, number_val_contagious1(a, -as_number_vm(vm, a))); } DISPATCH(); }
    lbl_ABS: { int a_sp = vm->sp - 1; Value a = vm_pop(vm);
        /* SW-09b: see lbl_NEG. */
        if (a.type == VAL_I128) {
            vm_raise_error_msg(vm,
                "abs: i128 arithmetic is not supported on the VM (no i128 "
                "opcodes are implemented in the bytecode interpreter); use the "
                "native backend");
            DISPATCH();
        }
        if (a.type == VAL_HYPER_DUAL) { vm_push(vm, a); vm_dispatch_native(vm, 1916); }
        else if (a.type == VAL_DUAL) { vm_push(vm, a); vm_dispatch_native(vm, 383); }
        /* See lbl_NEG: (abs 1/3) answered 0 through the double path. */
        else if (a.type == VAL_RATIONAL) { vm_push(vm, a); vm_dispatch_native(vm, 336); }
        else if (a.type == VAL_BIGNUM) { vm->ad_node_map[vm->sp] = -1; vm_push_bignum_norm(vm, bignum_abs_val(&vm->heap.regions, (VmBignum*)vm->heap.objects[a.as.ptr]->opaque.ptr)); }
        else if (a.type == VAL_INT) { VM_AD_UNARY(vm, a_sp, ad_abs);
            if (a.as.i == INT64_MIN) vm_push_bignum_norm(vm, bignum_abs_val(&vm->heap.regions, bignum_from_int64(&vm->heap.regions, a.as.i)));
            else vm_push(vm, INT_VAL(a.as.i < 0 ? -a.as.i : a.as.i)); }
        else { VM_AD_UNARY(vm, a_sp, ad_abs); vm_push(vm, number_val_contagious1(a, fabs(as_number_vm(vm, a)))); } DISPATCH(); }

    /* --- Comparison --- */

    lbl_EQ: { Value b = vm_pop(vm), a = vm_pop(vm);
        /* SW-09b: generic comparison over i128 has the identical bug shape
         * as generic arithmetic — as_number_vm() reads a heap-boxed
         * VAL_I128 as 0.0, so e.g. (= (i128 5) (i128 5)) silently answered
         * #t via 0.0==0.0 regardless of the real values. The dedicated
         * `i128-*?` comparison surface (KNOWN_ISSUES.md) is unaffected. */
        if (a.type == VAL_I128 || b.type == VAL_I128) {
            vm_raise_error_msg(vm,
                "=: i128 comparison is not supported on the VM (no i128 opcodes "
                "are implemented in the bytecode interpreter); use i128=? or "
                "the native backend");
            DISPATCH();
        }
        if (vm_either_bignum(a, b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm, a, b) == 0)); DISPATCH(); }
        if (a.type == VAL_INT && b.type == VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i == b.as.i)); DISPATCH(); }
        vm_push(vm, BOOL_VAL(as_number_vm(vm, a) == as_number_vm(vm, b))); DISPATCH(); }
    lbl_LT: { Value b = vm_pop(vm), a = vm_pop(vm);
        /* SW-09b: see lbl_EQ. */
        if (a.type == VAL_I128 || b.type == VAL_I128) {
            vm_raise_error_msg(vm,
                "<: i128 comparison is not supported on the VM (no i128 opcodes "
                "are implemented in the bytecode interpreter); use i128<? or "
                "the native backend");
            DISPATCH();
        }
        if (vm_either_bignum(a, b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm, a, b) <  0)); DISPATCH(); }
        if (a.type == VAL_INT && b.type == VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i <  b.as.i)); DISPATCH(); }
        vm_push(vm, BOOL_VAL(as_number_vm(vm, a) <  as_number_vm(vm, b))); DISPATCH(); }
    lbl_GT: { Value b = vm_pop(vm), a = vm_pop(vm);
        /* SW-09b: see lbl_EQ. */
        if (a.type == VAL_I128 || b.type == VAL_I128) {
            vm_raise_error_msg(vm,
                ">: i128 comparison is not supported on the VM (no i128 opcodes "
                "are implemented in the bytecode interpreter); use i128>? or "
                "the native backend");
            DISPATCH();
        }
        if (vm_either_bignum(a, b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm, a, b) >  0)); DISPATCH(); }
        if (a.type == VAL_INT && b.type == VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i >  b.as.i)); DISPATCH(); }
        vm_push(vm, BOOL_VAL(as_number_vm(vm, a) >  as_number_vm(vm, b))); DISPATCH(); }
    lbl_LE: { Value b = vm_pop(vm), a = vm_pop(vm);
        /* SW-09b: see lbl_EQ. */
        if (a.type == VAL_I128 || b.type == VAL_I128) {
            vm_raise_error_msg(vm,
                "<=: i128 comparison is not supported on the VM (no i128 opcodes "
                "are implemented in the bytecode interpreter); use i128<=? or "
                "the native backend");
            DISPATCH();
        }
        if (vm_either_bignum(a, b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm, a, b) <= 0)); DISPATCH(); }
        if (a.type == VAL_INT && b.type == VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i <= b.as.i)); DISPATCH(); }
        vm_push(vm, BOOL_VAL(as_number_vm(vm, a) <= as_number_vm(vm, b))); DISPATCH(); }
    lbl_GE: { Value b = vm_pop(vm), a = vm_pop(vm);
        /* SW-09b: see lbl_EQ. */
        if (a.type == VAL_I128 || b.type == VAL_I128) {
            vm_raise_error_msg(vm,
                ">=: i128 comparison is not supported on the VM (no i128 opcodes "
                "are implemented in the bytecode interpreter); use i128>=? or "
                "the native backend");
            DISPATCH();
        }
        if (vm_either_bignum(a, b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm, a, b) >= 0)); DISPATCH(); }
        if (a.type == VAL_INT && b.type == VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i >= b.as.i)); DISPATCH(); }
        vm_push(vm, BOOL_VAL(as_number_vm(vm, a) >= as_number_vm(vm, b))); DISPATCH(); }
    lbl_NOT: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(!is_truthy(a))); DISPATCH(); }

    /* --- Variables --- */

    lbl_GET_LOCAL: {
        int src = vm->fp + instr.operand;
        if (src < 0 || src >= STACK_SIZE) {
            fprintf(stderr, "GET_LOCAL: index %d out of bounds [0, %d)\n", src, STACK_SIZE);
            vm->error = 1; goto vm_exit;
        }
        if (vm->sp >= 0 && vm->sp < STACK_SIZE)
            vm->ad_node_map[vm->sp] = vm->ad_node_map[src];
        vm_push(vm, vm->stack[src]);
        DISPATCH(); }
    lbl_SET_LOCAL: {
        int dst = vm->fp + instr.operand;
        if (dst < 0 || dst >= STACK_SIZE) {
            fprintf(stderr, "SET_LOCAL: index %d out of bounds [0, %d)\n", dst, STACK_SIZE);
            vm->error = 1; goto vm_exit;
        }
        if (vm->sp > 0 && vm->sp <= STACK_SIZE)
            vm->ad_node_map[dst] = vm->ad_node_map[vm->sp - 1];
        vm->stack[dst] = vm_peek(vm, 0);
        vm_pop(vm);
        DISPATCH(); }
    lbl_GET_UPVALUE: {
        Value closure_val = vm->stack[vm->fp - 1];
        if (closure_val.type == VAL_CLOSURE) {
            HeapObject* cl = vm->heap.objects[closure_val.as.ptr];
            if (instr.operand >= 0 && instr.operand < cl->closure.n_upvalues) {
                int32_t open_slot = cl->closure.open_slots[instr.operand];
                if (open_slot >= 0 && open_slot < STACK_SIZE)
                    vm_push(vm, vm->stack[open_slot]);
                else
                    vm_push(vm, cl->closure.upvalues[instr.operand]);
            } else {
                fprintf(stderr, "UPVALUE INDEX OUT OF BOUNDS\n");
                vm_push(vm, NIL_VAL);
            }
        } else {
            vm_push(vm, NIL_VAL);
        }
        DISPATCH();
    }
    lbl_SET_UPVALUE: {
        Value closure_val = vm->stack[vm->fp - 1];
        if (closure_val.type == VAL_CLOSURE) {
            HeapObject* cl = vm->heap.objects[closure_val.as.ptr];
            if (instr.operand >= 0 && instr.operand < cl->closure.n_upvalues) {
                int32_t open_slot = cl->closure.open_slots[instr.operand];
                if (open_slot >= 0 && open_slot < STACK_SIZE)
                    vm->stack[open_slot] = vm_peek(vm, 0);
                else
                    cl->closure.upvalues[instr.operand] = vm_peek(vm, 0);
            } else {
                fprintf(stderr, "UPVALUE INDEX OUT OF BOUNDS\n");
            }
        }
        vm_pop(vm);
        DISPATCH();
    }

    /* --- Closures --- */

    lbl_CLOSURE: {
        int const_idx = instr.operand & 0xFFFF;
        int n_upvalues = (instr.operand >> 16) & 0xFF;
        if (n_upvalues > 16) n_upvalues = 16;
        Value func_const = vm->constants[const_idx];
        int32_t func_pc = (int32_t)func_const.as.i;
        /* Arity packed by the compiler in bits 32..40 of the func-PC constant
         * (bit 40 = present flag); low 32 bits are the PC, so PC re-basing on
         * inlining/ESKB load leaves the arity untouched. */
        int32_t clo_arity = ((func_const.as.i >> 40) & 1)
            ? (int32_t)((func_const.as.i >> 32) & 0xFF) : -1;
        int32_t ptr = heap_alloc(&vm->heap);
        if (ptr < 0) { vm->error = 1; goto vm_exit; }
        vm->heap.objects[ptr]->type = HEAP_CLOSURE;
        vm->heap.objects[ptr]->closure.func_pc = func_pc;
        vm->heap.objects[ptr]->closure.arity = clo_arity;
        vm->heap.objects[ptr]->closure.n_upvalues = n_upvalues;
        for (int i = 0; i < 16; i++)
            vm->heap.objects[ptr]->closure.open_slots[i] = -1;
        for (int i = n_upvalues - 1; i >= 0; i--) {
            vm->heap.objects[ptr]->closure.upvalues[i] = vm_pop(vm);
        }
        vm_push(vm, CLOSURE_VAL(ptr));
        DISPATCH();
    }

    /* --- Function call --- */

    lbl_CALL: {
        int argc = instr.operand;
        Value func = vm->stack[vm->sp - 1 - argc];

        vm_language_coverage_named_call(vm, func);

        if (func.type == VAL_PARAMETER_OBJ) {
            Value result = vm_parameter_invoke(vm, func, &vm->stack[vm->sp - argc], argc);
            vm->sp -= argc + 1;
            vm_push(vm, result);
            DISPATCH();
        }

        /* Continuation invocation: (k value) */
        if (func.type == VAL_CONTINUATION && argc >= 1) {
            Value val = vm->stack[vm->sp - 1];
            VmContinuation* cont = (VmContinuation*)vm->heap.objects[func.as.ptr]->opaque.ptr;
            if (cont) {
                while (vm->n_winds > cont->n_winds) {
                    vm->n_winds--;
                    Value after = vm->wind_stack[vm->n_winds].after;
                    vm_run_wind_after(vm, after);
                }
                vm_promise_eval_unwind_to(vm, cont->promise_mark);
                if (cont->sp > STACK_SIZE || cont->frame_count > MAX_FRAMES) { vm->error = 1; goto vm_exit; }
                vm_restore_continuation_dynamic_state(vm, cont);
                if (vm->error) goto vm_exit;
                memcpy(vm->stack, cont->saved_stack, cont->sp * sizeof(Value));
                memcpy(vm->frames, cont->saved_frames, cont->frame_count * sizeof(CallFrame));
                vm->sp = cont->sp; vm->fp = cont->fp;
                vm->frame_count = cont->frame_count;
                vm->n_handlers = cont->n_handlers;
                vm->pc = cont->pc;
                vm_push(vm, val);
                vm_escape_native_control(vm);
                DISPATCH();
            }
        }

        if (func.type != VAL_CLOSURE) {
            fprintf(stderr,
                    "ERROR: calling non-function at pc=%d argc=%d type=%d\n",
                    vm->pc - 1, argc, (int)func.type);
            vm->error = 1; goto vm_exit;
        }

        HeapObject* cl = vm->heap.objects[func.as.ptr];

        if (vm->frame_count >= MAX_FRAMES) { fprintf(stderr, "FRAME OVERFLOW\n"); vm->error = 1; goto vm_exit; }
        vm->frames[vm->frame_count].return_pc = vm->pc;
        vm->frames[vm->frame_count].return_fp = vm->fp;
        vm->frames[vm->frame_count].func_pc = cl->closure.func_pc;
        vm->frame_count++;

        vm->fp = vm->sp - argc;
        vm->pc = cl->closure.func_pc;
        DISPATCH();
    }

    lbl_TAIL_CALL: {
        int argc = instr.operand;
        Value func = vm->stack[vm->sp - 1 - argc];
        vm_language_coverage_named_call(vm, func);
        if (func.type == VAL_PARAMETER_OBJ) {
            Value result = vm_parameter_invoke(vm, func, &vm->stack[vm->sp - argc], argc);
            if (vm->frame_count <= 0) {
                vm->sp = 0;
                vm_push(vm, result);
                vm->halted = 1;
                goto vm_exit;
            }
            vm->frame_count--;
            if (vm->frames[vm->frame_count].return_pc == -1) {
                vm->sp = 0;
                vm_push(vm, result);
                vm->halted = 1;
                goto vm_exit;
            }
            vm->sp = vm->fp - 1;
            vm->fp = vm->frames[vm->frame_count].return_fp;
            vm->pc = vm->frames[vm->frame_count].return_pc;
            vm_push(vm, result);
            DISPATCH();
        }
        /* Continuation invocation in tail position */
        if (func.type == VAL_CONTINUATION && argc >= 1) {
            Value val = vm->stack[vm->sp - 1];
            VmContinuation* cont = (VmContinuation*)vm->heap.objects[func.as.ptr]->opaque.ptr;
            if (cont) {
                while (vm->n_winds > cont->n_winds) { vm->n_winds--; vm_run_wind_after(vm, vm->wind_stack[vm->n_winds].after); }
                vm_promise_eval_unwind_to(vm, cont->promise_mark);
                if (cont->sp > STACK_SIZE || cont->frame_count > MAX_FRAMES) { vm->error = 1; goto vm_exit; }
                vm_restore_continuation_dynamic_state(vm, cont);
                if (vm->error) goto vm_exit;
                memcpy(vm->stack, cont->saved_stack, cont->sp * sizeof(Value)); memcpy(vm->frames, cont->saved_frames, cont->frame_count * sizeof(CallFrame));
                vm->sp = cont->sp; vm->fp = cont->fp; vm->frame_count = cont->frame_count; vm->n_handlers = cont->n_handlers; vm->pc = cont->pc;
                vm_push(vm, val);
                vm_escape_native_control(vm);
                DISPATCH();
            }
        }
        if (func.type != VAL_CLOSURE) { vm->error = 1; goto vm_exit; }
        HeapObject* cl = vm->heap.objects[func.as.ptr];

        for (int i = 0; i < argc; i++) {
            vm->stack[vm->fp + i] = vm->stack[vm->sp - argc + i];
        }
        vm->sp = vm->fp + argc;
        /* Update closure slot so GET_UPVALUE sees the NEW closure's upvalues */
        vm->stack[vm->fp - 1] = func;
        vm->pc = cl->closure.func_pc;
        DISPATCH();
    }

    lbl_RETURN: {
        Value result = vm_pop(vm);
        if (vm->frame_count <= 0) {
            vm_push(vm, result);
            vm->halted = 1;
            goto vm_exit;
        }
        vm->frame_count--;
        /* Check for native-call sentinel */
        if (vm->frames[vm->frame_count].return_pc == -1) {
            vm_push(vm, result);
            vm->halted = 1;
            goto vm_exit;
        }
        vm->sp = vm->fp - 1;
        vm->fp = vm->frames[vm->frame_count].return_fp;
        vm->pc = vm->frames[vm->frame_count].return_pc;
        vm_push(vm, result);
        DISPATCH();
    }

    /* --- Control Flow --- */

    lbl_JUMP:
        vm->pc = instr.operand;
        DISPATCH();
    lbl_JUMP_IF_FALSE: {
        Value cond = vm_pop(vm);
        if (!is_truthy(cond)) vm->pc = instr.operand;
        DISPATCH();
    }
    lbl_LOOP:
        vm->pc = instr.operand;
        DISPATCH();

    /* --- Pairs --- */

    lbl_CONS: {
        Value car = vm_pop(vm), cdr = vm_pop(vm);
        int32_t ptr = heap_alloc(&vm->heap);
        if (ptr < 0) { vm->error = 1; goto vm_exit; }
        vm->heap.objects[ptr]->type = HEAP_CONS;
        vm->heap.objects[ptr]->cons.car = car;
        vm->heap.objects[ptr]->cons.cdr = cdr;
        vm_push(vm, PAIR_VAL(ptr));
        DISPATCH();
    }
    lbl_CAR: {
        Value pair = vm_pop(vm);
        if (pair.type != VAL_PAIR) { fprintf(stderr, "CAR on non-pair\n"); vm->error = 1; goto vm_exit; }
        vm_push(vm, vm->heap.objects[pair.as.ptr]->cons.car);
        DISPATCH();
    }
    lbl_CDR: {
        Value pair = vm_pop(vm);
        if (pair.type != VAL_PAIR) { fprintf(stderr, "CDR on non-pair\n"); vm->error = 1; goto vm_exit; }
        vm_push(vm, vm->heap.objects[pair.as.ptr]->cons.cdr);
        DISPATCH();
    }
    lbl_NULL_P: {
        Value v = vm_pop(vm);
        vm_push(vm, BOOL_VAL(v.type == VAL_NIL));
        DISPATCH();
    }

    /* --- I/O --- */

    lbl_PRINT: {
        Value v = vm_pop(vm);
        if (v.type != VAL_VOID) {
            print_value(vm, v);
            printf("\n"); fflush(stdout);
            if (vm->n_outputs < 256) vm->outputs[vm->n_outputs++] = v;
        }
        DISPATCH();
    }

    lbl_VOID:
        vm_push(vm, (Value){.type = VAL_VOID});
        DISPATCH();

    lbl_LANGUAGE_COVERAGE:
        DISPATCH();

    lbl_LANGUAGE_COVERAGE_CALL:
        vm->language_coverage_call_hash = (uint32_t)instr.operand;
        vm->language_coverage_call_pc = vm->pc;
        DISPATCH();

    lbl_HALT:
        vm->halted = 1;
        goto vm_exit;

    lbl_NATIVE_CALL: {
        vm_language_coverage_native_dispatch(vm, instr.operand);
        vm_dispatch_native(vm, instr.operand);
        DISPATCH();
    }

    lbl_CLOSE_UPVALUE: {
        /* Patch the TOS closure's upvalue[operand] to point to the closure itself */
        Value cl_val = vm_peek(vm, 0);
        if (cl_val.type == VAL_CLOSURE) {
            HeapObject* cl = vm->heap.objects[cl_val.as.ptr];
            if (instr.operand >= 0 && instr.operand < cl->closure.n_upvalues)
                cl->closure.upvalues[instr.operand] = cl_val;
        }
        DISPATCH();
    }

    lbl_VEC_CREATE: {
        int count = instr.operand;
        int32_t ptr = heap_alloc(&vm->heap);
        if (ptr < 0) { vm->error = 1; goto vm_exit; }
        vm->heap.objects[ptr]->type = HEAP_VECTOR;
        VmVector* vec = (VmVector*)vm_alloc(&vm->heap.regions, sizeof(VmVector));
        if (!vec) { vm->error = 1; goto vm_exit; }
        vec->len = count;
        vec->cap = count;
        vec->items = (Value*)vm_alloc(&vm->heap.regions, count * sizeof(Value));
        if (!vec->items && count > 0) { vm->error = 1; goto vm_exit; }
        for (int i = count - 1; i >= 0; i--) vec->items[i] = vm_pop(vm);
        vm->heap.objects[ptr]->opaque.ptr = vec;
        vm_push(vm, (Value){.type = VAL_VECTOR, .as.ptr = ptr});
        DISPATCH();
    }

    /* The threaded (computed-goto) bodies below and the switch-based fallback
     * further down are the two halves of the same interpreter; the inline
     * vector/string accessor fast paths must enforce the same catchable
     * out-of-range contract as the native codegen in BOTH.  See
     * vm_raise_error_msg() in vm_native.c. */
    lbl_VEC_REF: {
        Value idx = vm_pop(vm), vec_val = vm_pop(vm);
        if (vec_val.type != VAL_VECTOR) { vm_push(vm, NIL_VAL); DISPATCH(); }
        VmVector* vec = (VmVector*)vm->heap.objects[vec_val.as.ptr]->opaque.ptr;
        int i = (int)as_number(idx);
        if (!vec || i < 0 || i >= vec->len) {
            vm_raise_error_msg(vm, "vector-ref: index out of bounds");
            DISPATCH();
        }
        vm_push(vm, vec->items[i]);
        DISPATCH();
    }

    lbl_VEC_SET: {
        Value val = vm_pop(vm), idx = vm_pop(vm), vec_val = vm_pop(vm);
        if (vec_val.type == VAL_VECTOR) {
            VmVector* vec = (VmVector*)vm->heap.objects[vec_val.as.ptr]->opaque.ptr;
            int i = (int)as_number(idx);
            if (!vec || i < 0 || i >= vec->len) {
                vm_raise_error_msg(vm, "vector-set!: index out of bounds");
                DISPATCH();
            }
            vec->items[i] = val;
        }
        vm_push(vm, NIL_VAL);
        DISPATCH();
    }

    lbl_VEC_LEN: {
        Value vec_val = vm_pop(vm);
        if (vec_val.type == VAL_VECTOR) {
            VmVector* vec = (VmVector*)vm->heap.objects[vec_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(vec ? vec->len : 0));
        } else vm_push(vm, INT_VAL(0));
        DISPATCH();
    }

    lbl_STR_REF: {
        Value idx = vm_pop(vm), str_val = vm_pop(vm);
        if (str_val.type == VAL_STRING) {
            VmString* s = (VmString*)vm->heap.objects[str_val.as.ptr]->opaque.ptr;
            int i = (int)as_number(idx);
            if (!s || i < 0 || i >= s->byte_len) {
                vm_raise_error_msg(vm, "string-ref: index out of bounds");
                DISPATCH();
            }
            /* R7RS string-ref returns a character, not its integer code. */
            vm_push(vm, (Value){.type = VAL_CHAR, .as.i = (unsigned char)s->data[i]});
        } else vm_push(vm, (Value){.type = VAL_CHAR, .as.i = 0});
        DISPATCH();
    }

    lbl_STR_LEN: {
        Value str_val = vm_pop(vm);
        if (str_val.type == VAL_STRING) {
            VmString* s = (VmString*)vm->heap.objects[str_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(s ? s->byte_len : 0));
        } else vm_push(vm, INT_VAL(0));
        DISPATCH();
    }

    lbl_PAIR_P:  { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_PAIR)); DISPATCH(); }
    lbl_NUM_P:   { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_INT || v.type == VAL_FLOAT)); DISPATCH(); }
    lbl_STR_P:   { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_STRING)); DISPATCH(); }
    lbl_BOOL_P:  { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_BOOL)); DISPATCH(); }
    lbl_PROC_P:  { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_CLOSURE)); DISPATCH(); }
    lbl_VEC_P:   { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_VECTOR)); DISPATCH(); }

    lbl_SET_CAR: {
        Value val = vm_pop(vm), pair = vm_pop(vm);
        if (pair.type == VAL_PAIR) vm->heap.objects[pair.as.ptr]->cons.car = val;
        vm_push(vm, NIL_VAL);
        DISPATCH();
    }
    lbl_SET_CDR: {
        Value val = vm_pop(vm), pair = vm_pop(vm);
        if (pair.type == VAL_PAIR) vm->heap.objects[pair.as.ptr]->cons.cdr = val;
        vm_push(vm, NIL_VAL);
        DISPATCH();
    }

    lbl_POPN: {
        int n = instr.operand;
        if (n > 0 && vm->sp > n) {
            Value top = vm->stack[vm->sp - 1];
            vm->sp -= n;
            vm->stack[vm->sp - 1] = top;
        }
        DISPATCH();
    }

    lbl_CALLCC: {
        Value proc = vm_pop(vm);
        if (proc.type != VAL_CLOSURE) { vm_push(vm, NIL_VAL); DISPATCH(); }
        /* Validate bounds before capture */
        if (vm->sp > STACK_SIZE || vm->frame_count > MAX_FRAMES) { vm->error = 1; goto vm_exit; }
        int32_t cont_ptr = heap_alloc(&vm->heap);
        if (cont_ptr < 0) { vm->error = 1; goto vm_exit; }
        vm->heap.objects[cont_ptr]->type = HEAP_CONTINUATION;
        /* Store: pc, fp, sp, frame_count, n_handlers, n_winds + copy of stack + frames */
        VmContinuation* cont = (VmContinuation*)vm_alloc(&vm->heap.regions,
            vm_continuation_allocation_size(vm));
        if (!cont) { vm->error = 1; goto vm_exit; }
        cont->pc = vm->pc; cont->fp = vm->fp; cont->sp = vm->sp;
        cont->frame_count = vm->frame_count;
        cont->n_handlers = vm->n_handlers;
        cont->promise_mark = vm->promise_eval_head;
        cont->saved_stack = (Value*)((char*)cont + sizeof(VmContinuation));
        cont->saved_frames = (CallFrame*)((char*)cont->saved_stack + vm->sp * sizeof(Value));
        memcpy(cont->saved_stack, vm->stack, vm->sp * sizeof(Value));
        memcpy(cont->saved_frames, vm->frames, vm->frame_count * sizeof(CallFrame));
        vm_capture_continuation_dynamic_state(vm, cont);
        vm->heap.objects[cont_ptr]->opaque.ptr = cont;
        /* Create continuation closure: a special closure that invokes OP_INVOKE_CC */
        Value cont_val = (Value){.type = VAL_CONTINUATION, .as.ptr = cont_ptr};
        /* Call proc(continuation) */
        vm_push(vm, proc);
        vm_push(vm, cont_val);
        /* Set up call frame for proc(k) */
        HeapObject* cl_cc = vm->heap.objects[proc.as.ptr];
        if (vm->frame_count >= MAX_FRAMES) { vm->error = 1; goto vm_exit; }
        vm->frames[vm->frame_count].return_pc = vm->pc;
        vm->frames[vm->frame_count].return_fp = vm->fp;
        vm->frames[vm->frame_count].func_pc = cl_cc->closure.func_pc;
        vm->frame_count++;
        vm->fp = vm->sp - 1; /* 1 arg: the continuation */
        vm->pc = cl_cc->closure.func_pc;
        DISPATCH();
    }

    lbl_PUSH_HANDLER: {
        if (vm->n_handlers >= 16) { fprintf(stderr, "HANDLER STACK OVERFLOW\n"); vm->error = 1; goto vm_exit; }
        vm->handler_stack[vm->n_handlers].pc = instr.operand;
        vm->handler_stack[vm->n_handlers].sp = vm->sp;
        vm->handler_stack[vm->n_handlers].fp = vm->fp;
        vm->handler_stack[vm->n_handlers].frame_count = vm->frame_count;
        vm->handler_stack[vm->n_handlers].n_winds = vm->n_winds;
        vm->handler_stack[vm->n_handlers].n_parameter_bindings = vm->n_parameter_bindings;
        vm->handler_stack[vm->n_handlers].promise_mark = vm->promise_eval_head;
        vm->handler_stack[vm->n_handlers].region_handle_mark = eshkol_region_handle_seq_mark();  /* #341 */
        vm->n_handlers++;
        DISPATCH();
    }

    lbl_POP_HANDLER: {
        if (vm->n_handlers > 0) vm->n_handlers--;
        DISPATCH();
    }

    lbl_GET_EXN: {
        vm_push(vm, vm->current_exception);
        DISPATCH();
    }

    lbl_INVOKE_CC: {
        /* Invoke a captured continuation with a value */
        Value val = vm_pop(vm);
        Value cont_val = vm_pop(vm);
        if (cont_val.type == VAL_CONTINUATION) {
            VmContinuation* cont = (VmContinuation*)vm->heap.objects[cont_val.as.ptr]->opaque.ptr;
            if (cont) {
                /* Unwind dynamic-wind after-thunks */
                while (vm->n_winds > cont->n_winds) {
                    vm->n_winds--;
                    Value after = vm->wind_stack[vm->n_winds].after;
                    vm_run_wind_after(vm, after);
                }
                vm_promise_eval_unwind_to(vm, cont->promise_mark);
                /* Restore saved state (with bounds validation) */
                if (cont->sp > STACK_SIZE || cont->frame_count > MAX_FRAMES) { vm->error = 1; goto vm_exit; }
                vm_restore_continuation_dynamic_state(vm, cont);
                if (vm->error) goto vm_exit;
                memcpy(vm->stack, cont->saved_stack, cont->sp * sizeof(Value));
                memcpy(vm->frames, cont->saved_frames, cont->frame_count * sizeof(CallFrame));
                vm->sp = cont->sp; vm->fp = cont->fp;
                vm->frame_count = cont->frame_count;
                vm->n_handlers = cont->n_handlers;
                vm->pc = cont->pc;
                vm_push(vm, val);
                vm_escape_native_control(vm);
            }
        }
        DISPATCH();
    }

    lbl_PACK_REST: {
        int n_fixed = instr.operand;
        int n_args = vm->sp - vm->fp;
        Value list = NIL_VAL;
        for (int i = n_args - 1; i >= n_fixed; i--) {
            Value item = vm->stack[vm->fp + i];
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; goto vm_exit; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = item;
            vm->heap.objects[p]->cons.cdr = list;
            list = PAIR_VAL(p);
        }
        vm->sp = vm->fp + n_fixed;
        vm_push(vm, list);
        DISPATCH();
    }

    lbl_WIND_PUSH: {
        Value after = vm_pop(vm);
        if (vm->n_winds >= 32) { fprintf(stderr, "WIND STACK OVERFLOW\n"); vm->error = 1; goto vm_exit; }
        /* The compiler has already called before and leaves only the after
         * thunk on the operand stack.  Keep before as metadata for future
         * continuation re-entry support; consuming a second stack value here
         * previously corrupted the surrounding dynamic extent. */
        vm->wind_stack[vm->n_winds].before = NIL_VAL;
        vm->wind_stack[vm->n_winds].after = after;
        vm->n_winds++;
        DISPATCH();
    }

    lbl_WIND_POP: {
        /* Normal-path after invocation is emitted explicitly by the
         * compiler.  This opcode only removes the exceptional-exit guard. */
        if (vm->n_winds > 0) vm->n_winds--;
        DISPATCH();
    }

vm_exit:
    #undef DISPATCH
    if (owns_native_escape) vm->native_escape_ready = 0;

#else
/* =========================================================================
 * Fallback: standard switch dispatch for non-GCC/Clang compilers (MSVC etc.)
 * ========================================================================= */

    /* Same guard and the same configurable ceiling as the computed-goto path
     * above, so the two dispatch implementations of this one interpreter agree
     * about when a program has run away. */
    uint64_t vm_insns_executed = 0;
    const uint64_t vm_max_insn = g_eshkol_vm_max_insn;
    unsigned vm_check_budget = VM_CHECK_INTERVAL;
    while (!vm->halted && !vm->error && vm->pc < vm->code_len) {
        if (--vm_check_budget == 0) {
            vm_check_budget = VM_CHECK_INTERVAL;
            if (!vm_limits_checkpoint(vm, &vm_insns_executed, vm_max_insn)) break;
        }
        Instr instr = vm->code[vm->pc++];

        switch (instr.op) {
        case OP_NOP: break;

        case OP_CONST:
            if (instr.operand < 0 || instr.operand >= vm->n_constants) {
                fprintf(stderr, "INVALID CONSTANT INDEX %d\n", instr.operand);
                vm->error = 1; break;
            }
            vm_push(vm, vm->constants[instr.operand]);
            break;

        case OP_NIL:   vm_push(vm, NIL_VAL); break;
        case OP_TRUE:  vm_push(vm, BOOL_VAL(1)); break;
        case OP_FALSE: vm_push(vm, BOOL_VAL(0)); break;
        case OP_POP:   vm_pop(vm); break;
        case OP_DUP:   vm_push(vm, vm_peek(vm, 0)); break;

        /* Arithmetic */
        case OP_ADD: { Value b = vm_pop(vm), a = vm_pop(vm);
            /* SW-09: see the identical guard in lbl_ADD above — this switch-
             * based loop is the non-computed-goto twin of the same opcode
             * and must reject i128 operands the same way, not silently
             * coerce them to 0.0 via as_number_vm(). */
            if (a.type == VAL_I128 || b.type == VAL_I128) {
                vm_raise_error_msg(vm,
                    "+: i128 arithmetic is not supported on the VM (no i128 opcodes "
                    "are implemented in the bytecode interpreter); use the native "
                    "backend");
                break;
            }
            if (a.type==VAL_HYPER_DUAL||b.type==VAL_HYPER_DUAL) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,1905); }
            else if (a.type==VAL_DUAL||b.type==VAL_DUAL) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,373); }
            else if (a.type==VAL_RATIONAL||b.type==VAL_RATIONAL) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,331); }
            else if (a.type==VAL_COMPLEX||b.type==VAL_COMPLEX) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,307); }
            else if (vm_either_bignum(a,b)) vm_bignum_arith(vm,a,b,'+');
            else if (a.type==VAL_INT && b.type==VAL_INT) { int64_t r; if (__builtin_add_overflow(a.as.i,b.as.i,&r)) vm_bignum_arith(vm,a,b,'+'); else vm_push(vm, INT_VAL(r)); }
            else vm_push(vm, number_val_contagious(a, b, as_number_vm(vm,a) + as_number_vm(vm,b))); break; }
        case OP_SUB: { Value b = vm_pop(vm), a = vm_pop(vm);
            /* SW-09b: switch-based twin of lbl_SUB. */
            if (a.type == VAL_I128 || b.type == VAL_I128) {
                vm_raise_error_msg(vm,
                    "-: i128 arithmetic is not supported on the VM (no i128 opcodes "
                    "are implemented in the bytecode interpreter); use the native "
                    "backend");
                break;
            }
            if (a.type==VAL_HYPER_DUAL||b.type==VAL_HYPER_DUAL) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,1906); }
            else if (a.type==VAL_DUAL||b.type==VAL_DUAL) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,374); }
            else if (a.type==VAL_RATIONAL||b.type==VAL_RATIONAL) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,332); }
            else if (a.type==VAL_COMPLEX||b.type==VAL_COMPLEX) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,308); }
            else if (vm_either_bignum(a,b)) vm_bignum_arith(vm,a,b,'-');
            else if (a.type==VAL_INT && b.type==VAL_INT) { int64_t r; if (__builtin_sub_overflow(a.as.i,b.as.i,&r)) vm_bignum_arith(vm,a,b,'-'); else vm_push(vm, INT_VAL(r)); }
            else vm_push(vm, number_val_contagious(a, b, as_number_vm(vm,a) - as_number_vm(vm,b))); break; }
        case OP_MUL: { Value b = vm_pop(vm), a = vm_pop(vm);
            /* SW-09b: switch-based twin of lbl_MUL. */
            if (a.type == VAL_I128 || b.type == VAL_I128) {
                vm_raise_error_msg(vm,
                    "*: i128 arithmetic is not supported on the VM (no i128 opcodes "
                    "are implemented in the bytecode interpreter); use the native "
                    "backend");
                break;
            }
            if (a.type==VAL_HYPER_DUAL||b.type==VAL_HYPER_DUAL) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,1907); }
            else if (a.type==VAL_DUAL||b.type==VAL_DUAL) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,375); }
            else if (a.type==VAL_RATIONAL||b.type==VAL_RATIONAL) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,333); }
            else if (a.type==VAL_COMPLEX||b.type==VAL_COMPLEX) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,309); }
            else if (vm_either_bignum(a,b)) vm_bignum_arith(vm,a,b,'*');
            else if (a.type==VAL_INT && b.type==VAL_INT) { int64_t r; if (__builtin_mul_overflow(a.as.i,b.as.i,&r)) vm_bignum_arith(vm,a,b,'*'); else vm_push(vm, INT_VAL(r)); }
            else vm_push(vm, number_val_contagious(a, b, as_number_vm(vm,a) * as_number_vm(vm,b))); break; }
        case OP_DIV: { Value b = vm_pop(vm), a = vm_pop(vm);
            /* SW-09b: switch-based twin of lbl_DIV. */
            if (a.type == VAL_I128 || b.type == VAL_I128) {
                vm_raise_error_msg(vm,
                    "/: i128 arithmetic is not supported on the VM (no i128 opcodes "
                    "are implemented in the bytecode interpreter); use the native "
                    "backend");
                break;
            }
            if (a.type==VAL_HYPER_DUAL||b.type==VAL_HYPER_DUAL) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,1908); }
            else if (a.type==VAL_DUAL||b.type==VAL_DUAL) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,376); }
            else if (a.type==VAL_RATIONAL||b.type==VAL_RATIONAL) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,334); }
            else if (a.type==VAL_COMPLEX||b.type==VAL_COMPLEX) { vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,310); }
            else if (a.type==VAL_INT && b.type==VAL_INT) {
                /* exact/exact → exact result (R7RS): native 334 (rational div)
                 * reduces the fraction and collapses denom==1 back to an
                 * integer, so (/ 1 3) yields 1/3 and (/ 6 3) yields 2 rather
                 * than the inexact float the double path produced. */
                if (b.as.i == 0) { fprintf(stderr, "DIVIDE BY ZERO\n"); vm->error = 1; break; }
                vm_push(vm,a); vm_push(vm,b); vm_dispatch_native(vm,334);
            }
            /* See the threaded-dispatch OP_DIV above: bignums need the bignum
             * domain, and only EXACT-by-exact-zero is an error. */
            else if (vm_either_bignum(a,b)) { vm_bignum_arith(vm,a,b,'/'); }
            else { double bd = as_number_vm(vm,b);
            if (bd == 0 && vm_is_exact_number(a) && vm_is_exact_number(b)) {
                fprintf(stderr, "DIVIDE BY ZERO\n"); vm->error = 1; break; }
            vm_push(vm, number_val_contagious(a, b, as_number_vm(vm,a) / bd)); } break; }
        case OP_MOD: {
            Value b = vm_pop(vm), a = vm_pop(vm);
            /* SW-09b: switch-based twin of lbl_MOD. */
            if (a.type == VAL_I128 || b.type == VAL_I128) {
                vm_raise_error_msg(vm,
                    "modulo: i128 arithmetic is not supported on the VM (no i128 "
                    "opcodes are implemented in the bytecode interpreter); use the "
                    "native backend");
                break;
            }
            if (vm_either_bignum(a, b)) { vm_bignum_arith(vm, a, b, 'm'); break; }
            if (a.type == VAL_INT && b.type == VAL_INT) {
                if (b.as.i == 0) { fprintf(stderr, "MODULO BY ZERO\n"); vm->error = 1; break; }
                int64_t r = a.as.i % b.as.i; if (r != 0 && ((r ^ b.as.i) < 0)) r += b.as.i;
                vm_push(vm, INT_VAL(r)); break;
            }
            double bd = as_number_vm(vm, b);
            if (bd == 0) { fprintf(stderr, "MODULO BY ZERO\n"); vm->error = 1; break; }
            double r = fmod(as_number_vm(vm, a), bd);
            if (r != 0 && ((r > 0) != (bd > 0))) r += bd;
            vm_push(vm, number_val_contagious(a, b, r));
            break;
        }
        case OP_NEG: { Value a = vm_pop(vm);
            /* SW-09b: switch-based twin of lbl_NEG. */
            if (a.type == VAL_I128) {
                vm_raise_error_msg(vm,
                    "-: i128 arithmetic is not supported on the VM (no i128 opcodes "
                    "are implemented in the bytecode interpreter); use the native "
                    "backend");
                break;
            }
            /* See the threaded lbl_NEG: a rational needs the rational domain;
             * the double path below reads its heap pointer as 0.0. */
            if (a.type == VAL_RATIONAL) { vm_push(vm, a); vm_dispatch_native(vm, 335); break; }
            if (a.type == VAL_BIGNUM) { vm_push_bignum_norm(vm, bignum_neg(&vm->heap.regions, (VmBignum*)vm->heap.objects[a.as.ptr]->opaque.ptr)); break; }
            if (a.type == VAL_INT && a.as.i != INT64_MIN) { vm_push(vm, INT_VAL(-a.as.i)); break; }
            if (a.type == VAL_INT) { vm_push_bignum_norm(vm, bignum_neg(&vm->heap.regions, bignum_from_int64(&vm->heap.regions, a.as.i))); break; }
            vm_push(vm, number_val_contagious1(a, -as_number_vm(vm, a))); break; }
        case OP_ABS: { Value a = vm_pop(vm);
            /* SW-09b: switch-based twin of lbl_ABS. */
            if (a.type == VAL_I128) {
                vm_raise_error_msg(vm,
                    "abs: i128 arithmetic is not supported on the VM (no i128 "
                    "opcodes are implemented in the bytecode interpreter); use the "
                    "native backend");
                break;
            }
            if (a.type == VAL_RATIONAL) { vm_push(vm, a); vm_dispatch_native(vm, 336); break; }
            if (a.type == VAL_BIGNUM) { vm_push_bignum_norm(vm, bignum_abs_val(&vm->heap.regions, (VmBignum*)vm->heap.objects[a.as.ptr]->opaque.ptr)); break; }
            if (a.type == VAL_INT && a.as.i != INT64_MIN) { vm_push(vm, INT_VAL(a.as.i < 0 ? -a.as.i : a.as.i)); break; }
            if (a.type == VAL_INT) { vm_push_bignum_norm(vm, bignum_abs_val(&vm->heap.regions, bignum_from_int64(&vm->heap.regions, a.as.i))); break; }
            vm_push(vm, number_val_contagious1(a, fabs(as_number_vm(vm, a)))); break; }

        /* Comparison — push proper booleans */
        case OP_EQ: { Value b = vm_pop(vm), a = vm_pop(vm);
            /* SW-09b: switch-based twin of lbl_EQ. */
            if (a.type == VAL_I128 || b.type == VAL_I128) {
                vm_raise_error_msg(vm,
                    "=: i128 comparison is not supported on the VM (no i128 opcodes "
                    "are implemented in the bytecode interpreter); use i128=? or "
                    "the native backend");
                break;
            }
            if (vm_either_bignum(a,b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm,a,b) == 0)); break; }
            if (a.type==VAL_INT && b.type==VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i == b.as.i)); break; }
            vm_push(vm, BOOL_VAL(as_number_vm(vm,a) == as_number_vm(vm,b))); break; }
        case OP_LT: { Value b = vm_pop(vm), a = vm_pop(vm);
            /* SW-09b: switch-based twin of lbl_LT. */
            if (a.type == VAL_I128 || b.type == VAL_I128) {
                vm_raise_error_msg(vm,
                    "<: i128 comparison is not supported on the VM (no i128 opcodes "
                    "are implemented in the bytecode interpreter); use i128<? or "
                    "the native backend");
                break;
            }
            if (vm_either_bignum(a,b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm,a,b) <  0)); break; }
            if (a.type==VAL_INT && b.type==VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i <  b.as.i)); break; }
            vm_push(vm, BOOL_VAL(as_number_vm(vm,a) <  as_number_vm(vm,b))); break; }
        case OP_GT: { Value b = vm_pop(vm), a = vm_pop(vm);
            /* SW-09b: switch-based twin of lbl_GT. */
            if (a.type == VAL_I128 || b.type == VAL_I128) {
                vm_raise_error_msg(vm,
                    ">: i128 comparison is not supported on the VM (no i128 opcodes "
                    "are implemented in the bytecode interpreter); use i128>? or "
                    "the native backend");
                break;
            }
            if (vm_either_bignum(a,b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm,a,b) >  0)); break; }
            if (a.type==VAL_INT && b.type==VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i >  b.as.i)); break; }
            vm_push(vm, BOOL_VAL(as_number_vm(vm,a) >  as_number_vm(vm,b))); break; }
        case OP_LE: { Value b = vm_pop(vm), a = vm_pop(vm);
            /* SW-09b: switch-based twin of lbl_LE. */
            if (a.type == VAL_I128 || b.type == VAL_I128) {
                vm_raise_error_msg(vm,
                    "<=: i128 comparison is not supported on the VM (no i128 opcodes "
                    "are implemented in the bytecode interpreter); use i128<=? or "
                    "the native backend");
                break;
            }
            if (vm_either_bignum(a,b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm,a,b) <= 0)); break; }
            if (a.type==VAL_INT && b.type==VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i <= b.as.i)); break; }
            vm_push(vm, BOOL_VAL(as_number_vm(vm,a) <= as_number_vm(vm,b))); break; }
        case OP_GE: { Value b = vm_pop(vm), a = vm_pop(vm);
            /* SW-09b: switch-based twin of lbl_GE. */
            if (a.type == VAL_I128 || b.type == VAL_I128) {
                vm_raise_error_msg(vm,
                    ">=: i128 comparison is not supported on the VM (no i128 opcodes "
                    "are implemented in the bytecode interpreter); use i128>=? or "
                    "the native backend");
                break;
            }
            if (vm_either_bignum(a,b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm,a,b) >= 0)); break; }
            if (a.type==VAL_INT && b.type==VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i >= b.as.i)); break; }
            vm_push(vm, BOOL_VAL(as_number_vm(vm,a) >= as_number_vm(vm,b))); break; }
        case OP_NOT: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(!is_truthy(a))); break; }

        /* Variables */
        case OP_GET_LOCAL: {
            int src = vm->fp + instr.operand;
            if (src >= 0 && src < STACK_SIZE && vm->sp < STACK_SIZE)
                vm->ad_node_map[vm->sp] = vm->ad_node_map[src];
            vm_push(vm, vm->stack[src]);
            break; }
        case OP_SET_LOCAL:
            vm->stack[vm->fp + instr.operand] = vm_peek(vm, 0);
            vm_pop(vm);
            break;
        case OP_GET_UPVALUE: {
            Value closure_val = vm->stack[vm->fp - 1]; /* closure is just below frame */
            if (closure_val.type == VAL_CLOSURE) {
                HeapObject* cl = vm->heap.objects[closure_val.as.ptr];
                if (instr.operand >= 0 && instr.operand < cl->closure.n_upvalues) {
                    int32_t open_slot = cl->closure.open_slots[instr.operand];
                    if (open_slot >= 0 && open_slot < STACK_SIZE)
                        vm_push(vm, vm->stack[open_slot]);
                    else
                        vm_push(vm, cl->closure.upvalues[instr.operand]);
                } else {
                    fprintf(stderr, "UPVALUE INDEX OUT OF BOUNDS\n");
                    vm_push(vm, NIL_VAL);
                }
            } else {
                vm_push(vm, NIL_VAL);
            }
            break;
        }
        case OP_SET_UPVALUE: {
            Value closure_val = vm->stack[vm->fp - 1];
            if (closure_val.type == VAL_CLOSURE) {
                HeapObject* cl = vm->heap.objects[closure_val.as.ptr];
                if (instr.operand >= 0 && instr.operand < cl->closure.n_upvalues) {
                    int32_t open_slot = cl->closure.open_slots[instr.operand];
                    if (open_slot >= 0 && open_slot < STACK_SIZE)
                        vm->stack[open_slot] = vm_peek(vm, 0);
                    else
                        cl->closure.upvalues[instr.operand] = vm_peek(vm, 0);
                } else {
                    fprintf(stderr, "UPVALUE INDEX OUT OF BOUNDS\n");
                }
            }
            vm_pop(vm);
            break;
        }

        /* Closures */
        case OP_CLOSURE: {
            /* Operand: low 16 bits = constant pool index, bits 16-23 = n_upvalues */
            int const_idx = instr.operand & 0xFFFF;
            int n_upvalues = (instr.operand >> 16) & 0xFF;
            if (n_upvalues > 16) n_upvalues = 16;
            Value func_const = vm->constants[const_idx];
            int32_t func_pc = (int32_t)func_const.as.i;
            int32_t clo_arity = ((func_const.as.i >> 40) & 1)
                ? (int32_t)((func_const.as.i >> 32) & 0xFF) : -1;
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_CLOSURE;
            vm->heap.objects[ptr]->closure.func_pc = func_pc;
            vm->heap.objects[ptr]->closure.arity = clo_arity;
            vm->heap.objects[ptr]->closure.n_upvalues = n_upvalues;
            for (int i = 0; i < 16; i++)
                vm->heap.objects[ptr]->closure.open_slots[i] = -1;
            /* Pop upvalues from stack (pushed before CLOSURE, in reverse order) */
            for (int i = n_upvalues - 1; i >= 0; i--) {
                vm->heap.objects[ptr]->closure.upvalues[i] = vm_pop(vm);
            }
            vm_push(vm, CLOSURE_VAL(ptr));
            break;
        }

        /* Function call */
        case OP_CALL: {
            int argc = instr.operand;
            Value func = vm->stack[vm->sp - 1 - argc]; /* function is below args */

            vm_language_coverage_named_call(vm, func);

            if (func.type == VAL_PARAMETER_OBJ) {
                Value result = vm_parameter_invoke(vm, func,
                    &vm->stack[vm->sp - argc], argc);
                vm->sp -= argc + 1;
                vm_push(vm, result);
                break;
            }

            /* Continuation invocation: (k value) */
            if (func.type == VAL_CONTINUATION && argc >= 1) {
                Value val = vm->stack[vm->sp - 1];
                VmContinuation* cont = (VmContinuation*)vm->heap.objects[func.as.ptr]->opaque.ptr;
                if (cont) {
                    while (vm->n_winds > cont->n_winds) { vm->n_winds--; vm_run_wind_after(vm, vm->wind_stack[vm->n_winds].after); }
                    vm_promise_eval_unwind_to(vm, cont->promise_mark);
                    if (cont->sp > STACK_SIZE || cont->frame_count > MAX_FRAMES) { vm->error = 1; break; }
                    vm_restore_continuation_dynamic_state(vm, cont);
                    if (vm->error) break;
                    memcpy(vm->stack, cont->saved_stack, cont->sp * sizeof(Value)); memcpy(vm->frames, cont->saved_frames, cont->frame_count * sizeof(CallFrame));
                    vm->sp = cont->sp; vm->fp = cont->fp; vm->frame_count = cont->frame_count; vm->n_handlers = cont->n_handlers; vm->pc = cont->pc;
                    vm_push(vm, val);
                    vm_escape_native_control(vm);
                }
                break;
            }

            if (func.type != VAL_CLOSURE) {
                fprintf(stderr,
                        "ERROR: calling non-function at pc=%d argc=%d type=%d\n",
                        vm->pc - 1, argc, (int)func.type);
                vm->error = 1; break;
            }

            HeapObject* cl = vm->heap.objects[func.as.ptr];

            /* Save call frame */
            if (vm->frame_count >= MAX_FRAMES) { fprintf(stderr, "FRAME OVERFLOW\n"); vm->error = 1; break; }
            vm->frames[vm->frame_count].return_pc = vm->pc;
            vm->frames[vm->frame_count].return_fp = vm->fp;
            vm->frames[vm->frame_count].func_pc = cl->closure.func_pc;
            vm->frame_count++;

            /* Set up new frame: func sits at sp-argc-1, args at sp-argc..sp-1 */
            vm->fp = vm->sp - argc;
            vm->pc = cl->closure.func_pc;
            break;
        }

        case OP_TAIL_CALL: {
            int argc = instr.operand;
            Value func = vm->stack[vm->sp - 1 - argc];
            vm_language_coverage_named_call(vm, func);
            if (func.type == VAL_PARAMETER_OBJ) {
                Value result = vm_parameter_invoke(vm, func,
                    &vm->stack[vm->sp - argc], argc);
                if (vm->frame_count <= 0) {
                    vm->sp = 0;
                    vm_push(vm, result);
                    vm->halted = 1;
                    break;
                }
                vm->frame_count--;
                if (vm->frames[vm->frame_count].return_pc == -1) {
                    vm->sp = 0;
                    vm_push(vm, result);
                    vm->halted = 1;
                    break;
                }
                vm->sp = vm->fp - 1;
                vm->fp = vm->frames[vm->frame_count].return_fp;
                vm->pc = vm->frames[vm->frame_count].return_pc;
                vm_push(vm, result);
                break;
            }
            if (func.type == VAL_CONTINUATION && argc >= 1) {
                Value val = vm->stack[vm->sp - 1];
                VmContinuation* cont = (VmContinuation*)
                    vm->heap.objects[func.as.ptr]->opaque.ptr;
                if (cont) {
                    while (vm->n_winds > cont->n_winds) {
                        vm->n_winds--;
                        vm_run_wind_after(
                            vm, vm->wind_stack[vm->n_winds].after);
                    }
                    vm_promise_eval_unwind_to(vm, cont->promise_mark);
                    if (cont->sp > STACK_SIZE ||
                        cont->frame_count > MAX_FRAMES) {
                        vm->error = 1;
                        break;
                    }
                    vm_restore_continuation_dynamic_state(vm, cont);
                    if (vm->error) break;
                    memcpy(vm->stack, cont->saved_stack,
                           cont->sp * sizeof(Value));
                    memcpy(vm->frames, cont->saved_frames,
                           cont->frame_count * sizeof(CallFrame));
                    vm->sp = cont->sp;
                    vm->fp = cont->fp;
                    vm->frame_count = cont->frame_count;
                    vm->n_handlers = cont->n_handlers;
                    vm->pc = cont->pc;
                    vm_push(vm, val);
                    vm_escape_native_control(vm);
                }
                break;
            }
            if (func.type != VAL_CLOSURE) { vm->error = 1; break; }
            HeapObject* cl = vm->heap.objects[func.as.ptr];

            /* Move args to current frame position (reuse frame) */
            for (int i = 0; i < argc; i++) {
                vm->stack[vm->fp + i] = vm->stack[vm->sp - argc + i];
            }
            vm->sp = vm->fp + argc;
            /* Update closure slot so GET_UPVALUE sees the NEW closure's upvalues */
            vm->stack[vm->fp - 1] = func;
            vm->pc = cl->closure.func_pc;
            break;
        }

        case OP_RETURN: {
            Value result = vm_pop(vm);
            if (vm->frame_count <= 0) {
                /* Top-level return */
                vm_push(vm, result);
                vm->halted = 1;
                break;
            }
            vm->frame_count--;
            /* Check for native-call sentinel */
            if (vm->frames[vm->frame_count].return_pc == -1) {
                vm_push(vm, result);
                vm->halted = 1;
                break;
            }
            vm->sp = vm->fp - 1; /* discard frame + function slot */
            vm->fp = vm->frames[vm->frame_count].return_fp;
            vm->pc = vm->frames[vm->frame_count].return_pc;
            vm_push(vm, result);
            break;
        }

        /* Control Flow */
        case OP_JUMP:
            vm->pc = instr.operand;
            break;
        case OP_JUMP_IF_FALSE: {
            Value cond = vm_pop(vm);
            if (!is_truthy(cond)) vm->pc = instr.operand;
            break;
        }
        case OP_LOOP:
            vm->pc = instr.operand;
            break;

        /* Pairs */
        case OP_CONS: {
            Value car = vm_pop(vm), cdr = vm_pop(vm);  /* TOS=car, SOS=cdr */
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_CONS;
            vm->heap.objects[ptr]->cons.car = car;
            vm->heap.objects[ptr]->cons.cdr = cdr;
            vm_push(vm, PAIR_VAL(ptr));
            break;
        }
        case OP_CAR: {
            Value pair = vm_pop(vm);
            if (pair.type != VAL_PAIR) { fprintf(stderr, "CAR on non-pair\n"); vm->error = 1; break; }
            vm_push(vm, vm->heap.objects[pair.as.ptr]->cons.car);
            break;
        }
        case OP_CDR: {
            Value pair = vm_pop(vm);
            if (pair.type != VAL_PAIR) { fprintf(stderr, "CDR on non-pair\n"); vm->error = 1; break; }
            vm_push(vm, vm->heap.objects[pair.as.ptr]->cons.cdr);
            break;
        }
        case OP_NULL_P: {
            Value v = vm_pop(vm);
            vm_push(vm, BOOL_VAL(v.type == VAL_NIL));
            break;
        }

        /* I/O */
        case OP_PRINT: {
            Value v = vm_pop(vm);
            if (v.type != VAL_VOID) {
                print_value(vm, v);
                printf("\n");
                if (vm->n_outputs < 256) vm->outputs[vm->n_outputs++] = v;
            }
            break;
        }

        case OP_VOID:
            vm_push(vm, (Value){.type = VAL_VOID});
            break;

        case OP_LANGUAGE_COVERAGE:
            break;

        case OP_LANGUAGE_COVERAGE_CALL:
            vm->language_coverage_call_hash = (uint32_t)instr.operand;
            vm->language_coverage_call_pc = vm->pc;
            break;

        case OP_HALT:
            vm->halted = 1;
            break;

        case OP_NATIVE_CALL: {
            vm_language_coverage_native_dispatch(vm, instr.operand);
            vm_dispatch_native(vm, instr.operand);
            break;
        }

        case OP_CLOSE_UPVALUE: {
            Value cl_val = vm_peek(vm, 0);
            if (cl_val.type == VAL_CLOSURE) {
                HeapObject* cl = vm->heap.objects[cl_val.as.ptr];
                if (instr.operand >= 0 && instr.operand < cl->closure.n_upvalues)
                    cl->closure.upvalues[instr.operand] = cl_val;
            }
            break;
        }

        case OP_VEC_CREATE: {
            int count = instr.operand;
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_VECTOR;
            VmVector* vec = (VmVector*)vm_alloc(&vm->heap.regions, sizeof(VmVector));
            if (!vec) { vm->error = 1; break; }
            vec->len = count; vec->cap = count;
            vec->items = (Value*)vm_alloc(&vm->heap.regions, count * sizeof(Value));
            if (!vec->items && count > 0) { vm->error = 1; break; }
            for (int i = count - 1; i >= 0; i--) vec->items[i] = vm_pop(vm);
            vm->heap.objects[ptr]->opaque.ptr = vec;
            vm_push(vm, (Value){.type = VAL_VECTOR, .as.ptr = ptr});
            break;
        }

        /* OP_VEC_REF / OP_VEC_SET / OP_STR_REF are the inline fast paths the
         * VM compiler emits for direct (vector-ref v i) / (vector-set! v i x) /
         * (string-ref s i) calls; the native-call handlers (vm_native.c cases
         * 219/220/551) serve the indirect/higher-order calls.  Both must
         * enforce the same catchable out-of-range contract as the native
         * codegen — see vm_raise_error_msg() in vm_native.c. */
        case OP_VEC_REF: {
            Value idx = vm_pop(vm), vec_val = vm_pop(vm);
            if (vec_val.type != VAL_VECTOR) { vm_push(vm, NIL_VAL); break; }
            VmVector* vec = (VmVector*)vm->heap.objects[vec_val.as.ptr]->opaque.ptr;
            int i = (int)as_number(idx);
            if (!vec || i < 0 || i >= vec->len) {
                vm_raise_error_msg(vm, "vector-ref: index out of bounds");
                break;
            }
            vm_push(vm, vec->items[i]);
            break;
        }

        case OP_VEC_SET: {
            Value val = vm_pop(vm), idx = vm_pop(vm), vec_val = vm_pop(vm);
            if (vec_val.type == VAL_VECTOR) {
                VmVector* vec = (VmVector*)vm->heap.objects[vec_val.as.ptr]->opaque.ptr;
                int i = (int)as_number(idx);
                if (!vec || i < 0 || i >= vec->len) {
                    vm_raise_error_msg(vm, "vector-set!: index out of bounds");
                    break;
                }
                vec->items[i] = val;
            }
            vm_push(vm, NIL_VAL);
            break;
        }

        case OP_VEC_LEN: {
            Value vec_val = vm_pop(vm);
            if (vec_val.type == VAL_VECTOR) {
                VmVector* vec = (VmVector*)vm->heap.objects[vec_val.as.ptr]->opaque.ptr;
                vm_push(vm, INT_VAL(vec ? vec->len : 0));
            } else vm_push(vm, INT_VAL(0));
            break;
        }

        case OP_STR_REF: {
            Value idx = vm_pop(vm), str_val = vm_pop(vm);
            if (str_val.type == VAL_STRING) {
                VmString* s = (VmString*)vm->heap.objects[str_val.as.ptr]->opaque.ptr;
                int i = (int)as_number(idx);
                if (!s || i < 0 || i >= s->byte_len) {
                    vm_raise_error_msg(vm, "string-ref: index out of bounds");
                    break;
                }
                /* R7RS string-ref returns a character, not its integer code. */
                vm_push(vm, (Value){.type = VAL_CHAR, .as.i = (unsigned char)s->data[i]});
            } else vm_push(vm, (Value){.type = VAL_CHAR, .as.i = 0});
            break;
        }

        case OP_STR_LEN: {
            Value str_val = vm_pop(vm);
            if (str_val.type == VAL_STRING) {
                VmString* s = (VmString*)vm->heap.objects[str_val.as.ptr]->opaque.ptr;
                vm_push(vm, INT_VAL(s ? s->byte_len : 0));
            } else vm_push(vm, INT_VAL(0));
            break;
        }

        case OP_PAIR_P: { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_PAIR)); break; }
        case OP_NUM_P:  { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_INT || v.type == VAL_FLOAT)); break; }
        case OP_STR_P:  { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_STRING)); break; }
        case OP_BOOL_P: { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_BOOL)); break; }
        case OP_PROC_P: { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_CLOSURE)); break; }
        case OP_VEC_P:  { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_VECTOR)); break; }

        case OP_SET_CAR: {
            Value val = vm_pop(vm), pair = vm_pop(vm);
            if (pair.type == VAL_PAIR) vm->heap.objects[pair.as.ptr]->cons.car = val;
            vm_push(vm, NIL_VAL); break;
        }
        case OP_SET_CDR: {
            Value val = vm_pop(vm), pair = vm_pop(vm);
            if (pair.type == VAL_PAIR) vm->heap.objects[pair.as.ptr]->cons.cdr = val;
            vm_push(vm, NIL_VAL); break;
        }

        case OP_POPN: {
            int n = instr.operand;
            if (n > 0 && vm->sp > n) {
                Value top = vm->stack[vm->sp - 1];
                vm->sp -= n;
                vm->stack[vm->sp - 1] = top;
            }
            break;
        }

        case OP_CALLCC: {
            /* Switch fallback: same logic as computed-goto lbl_CALLCC */
            Value proc = vm_pop(vm);
            if (proc.type != VAL_CLOSURE) { vm_push(vm, NIL_VAL); break; }
            int32_t cont_ptr = heap_alloc(&vm->heap);
            if (cont_ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[cont_ptr]->type = HEAP_CONTINUATION;
            VmContinuation* cont = (VmContinuation*)vm_alloc(&vm->heap.regions,
                vm_continuation_allocation_size(vm));
            if (!cont) { vm->error = 1; break; }
            cont->pc = vm->pc; cont->fp = vm->fp; cont->sp = vm->sp;
            cont->frame_count = vm->frame_count;
            cont->n_handlers = vm->n_handlers;
            cont->promise_mark = vm->promise_eval_head;
            cont->saved_stack = (Value*)((char*)cont + sizeof(VmContinuation));
            cont->saved_frames = (CallFrame*)((char*)cont->saved_stack + vm->sp * sizeof(Value));
            memcpy(cont->saved_stack, vm->stack, vm->sp * sizeof(Value));
            memcpy(cont->saved_frames, vm->frames, vm->frame_count * sizeof(CallFrame));
            vm_capture_continuation_dynamic_state(vm, cont);
            vm->heap.objects[cont_ptr]->opaque.ptr = cont;
            Value cont_val = (Value){.type = VAL_CONTINUATION, .as.ptr = cont_ptr};
            vm_push(vm, proc); vm_push(vm, cont_val);
            HeapObject* cl_cc = vm->heap.objects[proc.as.ptr];
            if (vm->frame_count >= MAX_FRAMES) { vm->error = 1; break; }
            vm->frames[vm->frame_count].return_pc = vm->pc;
            vm->frames[vm->frame_count].return_fp = vm->fp;
            vm->frames[vm->frame_count].func_pc = cl_cc->closure.func_pc;
            vm->frame_count++;
            vm->fp = vm->sp - 1; vm->pc = cl_cc->closure.func_pc;
            break;
        }
        case OP_INVOKE_CC: {
            Value val = vm_pop(vm); Value cont_val = vm_pop(vm);
            if (cont_val.type == VAL_CONTINUATION) {
                VmContinuation* cont = (VmContinuation*)vm->heap.objects[cont_val.as.ptr]->opaque.ptr;
                if (cont) {
                    while (vm->n_winds > cont->n_winds) { vm->n_winds--; vm_run_wind_after(vm, vm->wind_stack[vm->n_winds].after); }
                    vm_promise_eval_unwind_to(vm, cont->promise_mark);
                    if (cont->sp > STACK_SIZE || cont->frame_count > MAX_FRAMES) { vm->error = 1; break; }
                    vm_restore_continuation_dynamic_state(vm, cont);
                    if (vm->error) break;
                    memcpy(vm->stack, cont->saved_stack, cont->sp * sizeof(Value)); memcpy(vm->frames, cont->saved_frames, cont->frame_count * sizeof(CallFrame));
                    vm->sp = cont->sp; vm->fp = cont->fp; vm->frame_count = cont->frame_count; vm->n_handlers = cont->n_handlers; vm->pc = cont->pc;
                    vm_push(vm, val);
                    vm_escape_native_control(vm);
                }
            }
            break;
        }
        case OP_OPEN_CLOSURE: break;
        case OP_PUSH_HANDLER: {
            if (vm->n_handlers >= 16) { fprintf(stderr, "HANDLER STACK OVERFLOW\n"); vm->error = 1; break; }
            vm->handler_stack[vm->n_handlers].pc = instr.operand;
            vm->handler_stack[vm->n_handlers].sp = vm->sp;
            vm->handler_stack[vm->n_handlers].fp = vm->fp;
            vm->handler_stack[vm->n_handlers].frame_count = vm->frame_count;
            vm->handler_stack[vm->n_handlers].n_winds = vm->n_winds;
            vm->handler_stack[vm->n_handlers].n_parameter_bindings = vm->n_parameter_bindings;
            vm->handler_stack[vm->n_handlers].promise_mark = vm->promise_eval_head;
            vm->handler_stack[vm->n_handlers].region_handle_mark = eshkol_region_handle_seq_mark();  /* #341 */
            vm->n_handlers++;
            break;
        }
        case OP_POP_HANDLER: { if (vm->n_handlers > 0) vm->n_handlers--; break; }
        case OP_GET_EXN: { vm_push(vm, vm->current_exception); break; }
        case OP_PACK_REST: {
            int n_fixed = instr.operand;
            int n_args = vm->sp - vm->fp;
            Value list = NIL_VAL;
            for (int i = n_args - 1; i >= n_fixed; i--) {
                Value item = vm->stack[vm->fp + i];
                int32_t p = heap_alloc(&vm->heap);
                if (p < 0) { vm->error = 1; break; }
                vm->heap.objects[p]->type = HEAP_CONS;
                vm->heap.objects[p]->cons.car = item;
                vm->heap.objects[p]->cons.cdr = list;
                list = PAIR_VAL(p);
            }
            vm->sp = vm->fp + n_fixed;
            vm_push(vm, list);
            break;
        }
        case OP_WIND_PUSH: {
            Value after = vm_pop(vm);
            if (vm->n_winds < 32) { vm->wind_stack[vm->n_winds].before = NIL_VAL; vm->wind_stack[vm->n_winds].after = after; vm->n_winds++; }
            break;
        }
        case OP_WIND_POP: {
            if (vm->n_winds > 0) vm->n_winds--;
            break;
        }
        /* OP_CLOSE_UPVALUE handled at line 720 — no duplicate */

        default:
            fprintf(stderr, "UNKNOWN OPCODE %d\n", instr.op);
            vm->error = 1;
            break;
        }
    }
    if (owns_native_escape) vm->native_escape_ready = 0;
#endif
}

/*******************************************************************************
 * Test Programs
 ******************************************************************************/

/** @brief Mnemonic names for the first 38 base opcodes, indexed by opcode
 *         value, used by the test-program bytecode disassembler/printer
 *         below (extended opcodes beyond OP_NATIVE_CALL are not covered). */
static const char* opnames[] = {
    "NOP","CONST","NIL","TRUE","FALSE","POP","DUP",
    "ADD","SUB","MUL","DIV","MOD","NEG","ABS",
    "EQ","LT","GT","LE","GE","NOT",
    "GETL","SETL","GETUP","SETUP",
    "CLOSURE","CALL","TCALL","RET",
    "JUMP","JIF","LOOP",
    "CONS","CAR","CDR","NULLP",
    "PRINT","HALT","NATIVE"
};

/** @brief Append one bytecode instruction (@p op, @p operand) to @p vm's
 *         fixed-size (4096-instruction) test-program code buffer. */
static void emit(VM* vm, uint8_t op, int32_t operand) {
    if (vm->code_len >= 4096) return;
    vm->code[vm->code_len++] = (Instr){op, operand};
}

/** @brief Allocate and vm_init() a fresh VM instance with a 4096-instruction
 *         code buffer, for use by hand-assembled test programs (see
 *         vm_tests.c). */
VM* vm_create(void) {
    VM* vm = (VM*)calloc(1, sizeof(VM));
    if (!vm) return NULL;
    vm_init(vm);
    vm->code = (Instr*)calloc(4096, sizeof(Instr));
    if (!vm->code) { free(vm); return NULL; }
    return vm;
}
/** @brief Release all resources owned by @p vm (open regex handles,
 *         dlopen'd libraries, the heap's arena, and the code buffer) and
 *         free @p vm itself. */
void vm_free(VM* vm) {
    vm_regex_free_all(vm);
    vm_dlopen_close_all(vm);
    heap_destroy(&vm->heap);
    free(vm->code);
    free(vm->constants);
    vm->constants = NULL;
    vm->const_cap = 0;
    free(vm);
}
