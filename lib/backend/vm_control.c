/**
 * @file vm_control.c
 * @brief Non-local control for the bytecode VM: continuation capture and
 *        resumption, dynamic-wind rerooting, and the exception-handler stack.
 *
 * Split out of vm_run.c: the dispatch loop decides WHICH opcode runs, this
 * file owns the VM state a `call/cc`, a `dynamic-wind` or a `guard` has to
 * save and put back. The capture/restore pair is the delicate part — R7RS
 * `call/cc` captures the control state and not the store (SW-52), so a
 * continuation snapshots only the operand slots above vm->global_top.
 *
 */

/* Slots [0, vm->global_top) are top-level bindings — the store. R7RS
 * `call/cc` captures the control state, not the store, so a continuation
 * snapshots only [global_top, sp) and a re-entry leaves the bindings below
 * holding whatever the program has since `set!` into them. Capturing them
 * too is what made every mutation between capture and re-entry silently
 * revert (SW-52). */
static int vm_continuation_stack_span(const VM* vm) {
    int span = vm->sp - vm->global_top;
    return span > 0 ? span : 0;
}

static size_t vm_continuation_allocation_size(const VM* vm) {
    return sizeof(VmContinuation) +
        (size_t)vm_continuation_stack_span(vm) * sizeof(Value) +
        (size_t)vm->frame_count * sizeof(CallFrame) +
        (size_t)vm->n_winds * 2 * sizeof(Value) +
        (size_t)vm->n_parameter_bindings * 2 * sizeof(Value);
}

/* Snapshot the control stack (operands above the store boundary) and the call
 * frames. Must run before vm_capture_continuation_dynamic_state(), which lays
 * its own arrays out after saved_frames. */
static void vm_capture_continuation_stack(VM* vm, VmContinuation* cont) {
    int span = vm_continuation_stack_span(vm);
    cont->stack_base = vm->global_top;
    cont->saved_stack = (Value*)((char*)cont + sizeof(VmContinuation));
    cont->saved_frames =
        (CallFrame*)((char*)cont->saved_stack + (size_t)span * sizeof(Value));
    memcpy(cont->saved_stack, vm->stack + cont->stack_base,
           (size_t)span * sizeof(Value));
    memcpy(cont->saved_frames, vm->frames,
           (size_t)vm->frame_count * sizeof(CallFrame));
}

static void vm_capture_continuation_dynamic_state(VM* vm,
                                                  VmContinuation* cont) {
    char* cursor = (char*)cont->saved_frames +
        (size_t)vm->frame_count * sizeof(CallFrame);
    cont->n_winds = vm->n_winds;
    cont->n_parameter_bindings = vm->n_parameter_bindings;
    cont->n_region_brackets = vm->n_region_brackets;
    /* Stage-1 evacuator: a captured continuation can resurrect a stack state
     * from inside a region body, and re-entering a region whose arena was
     * released is not something Stage-1 supports. So every region open at
     * capture time is PINNED — it will be promoted whole rather than freed.
     * The cost is reclamation in the rare call/cc-inside-with-region case, and
     * it is paid in the direction of a leak, never a dangling index. */
    if (vm->heap.regions.depth > 0)
        heap_region_pin_all(&vm->heap, "a continuation was captured inside a region");
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

/* Restore the control half of a captured continuation: the operand slots at
 * or above the *current* store boundary, plus the call frames.
 *
 * The store boundary can only have risen since capture (OP_GLOBAL_MARK is
 * monotonic), so bindings established after the capture keep their current
 * values rather than being rolled back to the snapshot — top-level `define`
 * and `set!` are store effects and R7RS re-entry does not undo them.
 *
 * The one shape this representation cannot express is a top-level binding
 * whose slot sits *above* the resumed stack top: restoring the snapshot would
 * have to put operands where a live binding now lives. The VM keeps the store
 * and the operand stack in one array, so there is no correct answer here —
 * it fails LOUDLY rather than resuming onto a corrupted store.
 *
 * Returns 1 on success, 0 with vm->error set on failure.
 */
static int vm_restore_continuation_stack(VM* vm, const VmContinuation* cont) {
    int base = vm->global_top;
    if (base < cont->stack_base) base = cont->stack_base;   /* defensive */
    if (base > cont->sp) {
        fprintf(stderr,
                "ERROR: cannot resume this continuation — %d top-level "
                "binding slot(s) were established after it was captured, "
                "above its saved stack top (%d). Resuming would overwrite "
                "live bindings with stale operands.\n"
                "  This is a representation limit of the bytecode VM, which "
                "stores top-level bindings in operand-stack slots. Move the "
                "affected top-level define(s) above the call/cc, or run this "
                "program on the native backend.\n",
                base - cont->sp, cont->sp);
        vm->error = 1;
        return 0;
    }
    memcpy(vm->stack + base,
           cont->saved_stack + (base - cont->stack_base),
           (size_t)(cont->sp - base) * sizeof(Value));
    memcpy(vm->frames, cont->saved_frames,
           (size_t)cont->frame_count * sizeof(CallFrame));
    return 1;
}

/** @brief Identity of two wind-stack thunk values (same type, same payload). */
static int vm_wind_value_same(Value a, Value b) {
    if (a.type != b.type) return 0;
    return a.as.i == b.as.i;
}

/**
 * @brief Move the wind stack to the continuation's saved extent, running the
 *        thunks the transfer crosses (R7RS 6.10 rerooting).
 *
 * Unwinding alone is only correct for an escape, where the target extent is an
 * ancestor of the current one. Re-entering a continuation captured inside a
 * `dynamic-wind` whose extent has since been left has to run that extent's
 * `before` thunk again on the way back in, or the body resumes with its setup
 * undone. Wind entries are pushed in dynamic order, so a shared prefix of the
 * two stacks is a shared dynamic extent and nothing there needs to run.
 */
static void vm_reroot_winds(VM* vm, const VmContinuation* cont) {
    int limit = vm->n_winds < cont->n_winds ? vm->n_winds : cont->n_winds;
    int common = 0;
    while (common < limit
           && vm_wind_value_same(vm->wind_stack[common].before,
                                 cont->saved_wind_befores[common])
           && vm_wind_value_same(vm->wind_stack[common].after,
                                 cont->saved_wind_afters[common])) {
        common++;
    }

    /* Leave: innermost `after` first. */
    while (vm->n_winds > common) {
        vm->n_winds--;
        vm_run_wind_after(vm, vm->wind_stack[vm->n_winds].after);
    }
    /* Enter: outermost `before` first. Publish each entry before running its
     * thunk so a continuation captured inside a `before` sees a coherent
     * stack. Parameter objects are re-established by the parameter-binding
     * replay in vm_restore_continuation_dynamic_state(), not here. */
    for (int i = common; i < cont->n_winds; i++) {
        vm->wind_stack[i].before = cont->saved_wind_befores[i];
        vm->wind_stack[i].after  = cont->saved_wind_afters[i];
        vm->n_winds = i + 1;
        if (cont->saved_wind_befores[i].type == VAL_CLOSURE)
            vm_run_wind_after(vm, cont->saved_wind_befores[i]);
    }
    vm->n_winds = cont->n_winds;
}

static void vm_restore_continuation_dynamic_state(VM* vm,
                                                  const VmContinuation* cont) {
    /* Parameter values live outside the VM execution stack.  Rebuild their
     * dynamic stacks from the continuation snapshot before resuming code;
     * merely restoring a binding depth would otherwise leave captured
     * parameterize extents pointing at the values of the abandoned path. */
    vm_unwind_parameter_bindings(vm, 0);

    /* Stage-1 evacuator: close every `with-region` the transfer is jumping out
     * of, the counterpart of native's eshkol_region_unwind_for_continuation().
     * The regions are pinned first, so nothing the abandoned path allocated is
     * freed — the continuation's value may live anywhere in it and, unlike a
     * raise, there is no single in-flight slot to promote. */
    vm_region_bracket_unwind_pinned(vm, cont->n_region_brackets);

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

/* Resume a captured continuation with @p val: reroot the wind stack, unwind
 * the promise/parameter/region state the transfer leaves, put the saved
 * control state back, and escape any native C frames the invoke crossed.
 *
 * This body used to be copied four times — OP_CALL, OP_TAIL_CALL and
 * OP_INVOKE_CC, each once per dispatch mechanism. Failure is reported the way
 * it always was: vm->error is set and the caller's next dispatch step leaves
 * the loop. */
/* Build the value frame delivered by a continuation invocation before the
 * saved control stack is restored. The argument slots belong to the active
 * invocation and vm_restore_continuation_stack() may overwrite them. */
static int vm_continuation_result(VM* vm, const Value* args, int argc,
                                  Value* out) {
    if (!vm || !args || argc < 0 || argc > STACK_SIZE) return 0;
    if (argc == 1) {
        *out = args[0];
        return 1;
    }
    int32_t ptr = heap_alloc(&vm->heap);
    if (ptr < 0) return 0;
    VmVector* values = (VmVector*)vm_alloc(&vm->heap.regions,
                                           sizeof(VmVector));
    if (!values) return 0;
    values->len = argc;
    values->cap = argc;
    values->items = (Value*)vm_alloc(&vm->heap.regions,
                                     (size_t)(argc > 0 ? argc : 1) * sizeof(Value));
    if (!values->items) return 0;
    memcpy(values->items, args, (size_t)argc * sizeof(Value));
    vm->heap.objects[ptr]->type = HEAP_MULTI_VALUE;
    vm->heap.objects[ptr]->opaque.ptr = values;
    *out = (Value){.type = VAL_MULTI_VALUE, .as.ptr = ptr};
    return 1;
}

static void vm_continuation_resume(VM* vm, VmContinuation* cont, Value val) {
    vm_reroot_winds(vm, cont);
    vm_promise_eval_unwind_to(vm, cont->promise_mark);
    if (cont->sp > STACK_SIZE || cont->frame_count > MAX_FRAMES) { vm->error = 1; return; }
    vm_restore_continuation_dynamic_state(vm, cont);
    if (vm->error) return;
    if (!vm_restore_continuation_stack(vm, cont)) return;
    vm->sp = cont->sp; vm->fp = cont->fp;
    vm->frame_count = cont->frame_count;
    vm->n_handlers = cont->n_handlers;
    vm->pc = cont->pc;
    vm_push(vm, val);
    vm_escape_native_control(vm);
}

static void vm_exec_invoke_cc(VM* vm) {
    /* Invoke a captured continuation with a value */
    Value val = vm_pop(vm);
    Value cont_val = vm_pop(vm);
    if (cont_val.type == VAL_CONTINUATION) {
        VmContinuation* cont = (VmContinuation*)vm->heap.objects[cont_val.as.ptr]->opaque.ptr;
        if (cont) vm_continuation_resume(vm, cont, val);
    }
}

static void vm_exec_push_handler(VM* vm, int32_t operand) {
    if (vm->n_handlers >= 16) { fprintf(stderr, "HANDLER STACK OVERFLOW\n"); vm->error = 1; return; }
    vm->handler_stack[vm->n_handlers].pc = operand;
    vm->handler_stack[vm->n_handlers].sp = vm->sp;
    vm->handler_stack[vm->n_handlers].fp = vm->fp;
    vm->handler_stack[vm->n_handlers].frame_count = vm->frame_count;
    vm->handler_stack[vm->n_handlers].n_winds = vm->n_winds;
    vm->handler_stack[vm->n_handlers].n_parameter_bindings = vm->n_parameter_bindings;
    vm->handler_stack[vm->n_handlers].promise_mark = vm->promise_eval_head;
    vm->handler_stack[vm->n_handlers].region_handle_mark = eshkol_region_handle_seq_mark();  /* #341 */
    vm->handler_stack[vm->n_handlers].region_bracket_mark = vm->n_region_brackets;
    vm->n_handlers++;
}
