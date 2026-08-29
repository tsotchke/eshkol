/**
 * @file vm_frame.c
 * @brief Call-frame and closure management for the bytecode VM: upvalue
 *        access, closure construction, and the return sequence.
 *
 * Split out of vm_run.c, and shared by both dispatch mechanisms for the same
 * reason as vm_ops.c: these bodies were identical in the computed-goto loop
 * and in the `switch` fallback, so they are defined once and called twice.
 * OP_CALL and OP_TAIL_CALL stay inline in the dispatch loop — their two
 * copies are not equivalent today (they differ in what happens when a
 * continuation object carries a null payload), and unifying them would be a
 * behaviour change rather than a move.
 *
 */

static void vm_exec_get_upvalue(VM* vm, int32_t operand) {
    Value closure_val = vm->stack[vm->fp - 1];
    if (closure_val.type == VAL_CLOSURE) {
        HeapObject* cl = vm->heap.objects[closure_val.as.ptr];
        if (operand >= 0 && operand < cl->closure.n_upvalues) {
            int32_t open_slot = cl->closure.open_slots[operand];
            if (open_slot >= 0 && open_slot < STACK_SIZE)
                vm_push(vm, vm->stack[open_slot]);
            else
                vm_push(vm, cl->closure.upvalues[operand]);
        } else {
            fprintf(stderr, "UPVALUE INDEX OUT OF BOUNDS\n");
            vm_push(vm, NIL_VAL);
        }
    } else {
        vm_push(vm, NIL_VAL);
    }
}

static void vm_exec_set_upvalue(VM* vm, int32_t operand) {
    Value closure_val = vm->stack[vm->fp - 1];
    if (closure_val.type == VAL_CLOSURE) {
        HeapObject* cl = vm->heap.objects[closure_val.as.ptr];
        if (operand >= 0 && operand < cl->closure.n_upvalues) {
            int32_t open_slot = cl->closure.open_slots[operand];
            if (open_slot >= 0 && open_slot < STACK_SIZE)
                vm->stack[open_slot] = vm_peek(vm, 0);
            else
                cl->closure.upvalues[operand] = vm_peek(vm, 0);
        } else {
            fprintf(stderr, "UPVALUE INDEX OUT OF BOUNDS\n");
        }
    }
    vm_pop(vm);
}

static void vm_exec_close_upvalue(VM* vm, int32_t operand) {
    /* Patch the TOS closure's upvalue[operand] to point to the closure itself */
    Value cl_val = vm_peek(vm, 0);
    if (cl_val.type == VAL_CLOSURE) {
        HeapObject* cl = vm->heap.objects[cl_val.as.ptr];
        if (operand >= 0 && operand < cl->closure.n_upvalues)
            cl->closure.upvalues[operand] = cl_val;
    }
}

static void vm_exec_closure(VM* vm, int32_t operand) {
    int const_idx = operand & 0xFFFF;
    int n_upvalues = (operand >> 16) & 0xFF;
    /* n_upvalues is the count the COMPILER pushed onto the operand stack
     * to feed this closure (func.n_upvalues, bounded at compile time by
     * MAX_UPVALUES). It must never exceed the runtime closure's array
     * capacity: the two are the same constant (ESHKOL_VM_MAX_CLOSURE_
     * UPVALUES, see vm_limits.h), so this can only fire on a corrupted
     * or build-mismatched .eskb. Previously this silently clamped to a
     * hardcoded 16 and then popped only the clamped count — leaving the
     * excess already-pushed values stranded on the stack, which
     * desynced every stack-slot offset the compiler had computed for
     * the rest of the program. A too-small limit must fail loudly
     * instead of running on with a corrupted stack. */
    if (n_upvalues > ESHKOL_VM_MAX_CLOSURE_UPVALUES) {
        fprintf(stderr,
                "ERROR: OP_CLOSURE upvalue count %d exceeds runtime capacity %d "
                "(pc=%d) — refusing to run a program with a corrupted or "
                "build-mismatched closure encoding\n",
                n_upvalues, ESHKOL_VM_MAX_CLOSURE_UPVALUES, vm->pc - 1);
        vm->error = 1; return;
    }
    Value func_const = vm->constants[const_idx];
    int32_t func_pc = (int32_t)func_const.as.i;
    /* Arity packed by the compiler in bits 32..40 of the func-PC constant
     * (bit 40 = present flag); low 32 bits are the PC, so PC re-basing on
     * inlining/ESKB load leaves the arity untouched. */
    int32_t clo_arity = ((func_const.as.i >> 40) & 1)
        ? (int32_t)((func_const.as.i >> 32) & 0xFF) : -1;
    int32_t ptr = heap_alloc(&vm->heap);
    if (ptr < 0) { vm->error = 1; return; }
    vm->heap.objects[ptr]->type = HEAP_CLOSURE;
    vm->heap.objects[ptr]->closure.func_pc = func_pc;
    vm->heap.objects[ptr]->closure.arity = clo_arity;
    vm->heap.objects[ptr]->closure.n_upvalues = n_upvalues;
    for (int i = 0; i < ESHKOL_VM_MAX_CLOSURE_UPVALUES; i++)
        vm->heap.objects[ptr]->closure.open_slots[i] = -1;
    for (int i = n_upvalues - 1; i >= 0; i--) {
        vm->heap.objects[ptr]->closure.upvalues[i] = vm_pop(vm);
    }
    vm_push(vm, CLOSURE_VAL(ptr));
}

static void vm_exec_return(VM* vm) {
    Value result = vm_pop(vm);
    // A tail transfer intentionally leaves its guard handlers live so a
    // re-raise reaches the next logical activation. Retire only handlers
    // explicitly marked by that transfer and owned by this frame generation;
    // frame_count alone is not an identity and can match an enclosing frame.
    vm_pop_tail_retained_handlers(vm);
    if (vm->frame_count <= 0) {
        vm_push(vm, result);
        vm->halted = 1;
        return;
    }
    vm->frame_count--;
    /* Check for native-call sentinel */
    if (vm->frames[vm->frame_count].return_pc == -1) {
        vm_push(vm, result);
        vm->halted = 1;
        return;
    }
    vm->sp = vm->fp - 1;
    vm->fp = vm->frames[vm->frame_count].return_fp;
    vm->pc = vm->frames[vm->frame_count].return_pc;
    vm_push(vm, result);
}
