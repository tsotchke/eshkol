/**
 * @file vm_ops.c
 * @brief Value-opcode bodies shared by both of the VM's dispatch
 *        implementations: the numeric-operand guards, the comparison
 *        opcodes, the pair opcodes, the operand-stack shuffle, and the
 *        vector opcodes.
 *
 * vm_run.c implements one interpreter through two dispatch mechanisms — a
 * computed-goto threaded loop on GCC/Clang and a `switch` fallback elsewhere.
 * Every body below was duplicated once per mechanism; each is now defined
 * once here and called from both, so a fix to an opcode cannot land in one
 * dispatch path and miss the other. Opcodes whose two copies are NOT
 * equivalent today are deliberately left inline in vm_run.c rather than
 * unified here, because unifying them would change behaviour.
 *
 * Signalling convention, unchanged from the inline bodies: a handler reports
 * failure by setting vm->error (or vm->halted) and returning; the caller's
 * next dispatch step observes the flag and leaves the loop, exactly as the
 * `goto vm_exit` / `break` it replaced did.
 *
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

/** Return non-zero for every scalar tag accepted by the arithmetic opcodes. */
static int vm_is_arithmetic_number(Value v) {
    return v.type == VAL_INT || v.type == VAL_FLOAT ||
           v.type == VAL_BIGNUM || v.type == VAL_RATIONAL ||
           v.type == VAL_COMPLEX || v.type == VAL_DUAL ||
           v.type == VAL_HYPER_DUAL || v.type == VAL_I128;
}

/** Raise a catchable type error before an opcode reaches as_number_vm(). */
static int vm_require_arithmetic_numbers(VM* vm, Value a, Value b,
                                         const char* op) {
    char message[96];
    if (vm_is_arithmetic_number(a) && vm_is_arithmetic_number(b)) return 1;
    snprintf(message, sizeof(message), "%s: expected numeric operands", op);
    vm_raise_error_msg(vm, message);
    return 0;
}

static void vm_exec_eq(VM* vm) {
    Value b = vm_pop(vm), a = vm_pop(vm);
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
        return;
    }
    if (vm_either_exact_wide(a, b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm, a, b) == 0)); return; }
    if (a.type == VAL_INT && b.type == VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i == b.as.i)); return; }
    vm_push(vm, BOOL_VAL(as_number_vm(vm, a) == as_number_vm(vm, b)));
}

static void vm_exec_lt(VM* vm) {
    Value b = vm_pop(vm), a = vm_pop(vm);
    /* SW-09b: see vm_exec_eq(). */
    if (a.type == VAL_I128 || b.type == VAL_I128) {
        vm_raise_error_msg(vm,
            "<: i128 comparison is not supported on the VM (no i128 opcodes "
            "are implemented in the bytecode interpreter); use i128<? or "
            "the native backend");
        return;
    }
    if (vm_either_exact_wide(a, b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm, a, b) <  0)); return; }
    if (a.type == VAL_INT && b.type == VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i <  b.as.i)); return; }
    vm_push(vm, BOOL_VAL(as_number_vm(vm, a) <  as_number_vm(vm, b)));
}

static void vm_exec_gt(VM* vm) {
    Value b = vm_pop(vm), a = vm_pop(vm);
    /* SW-09b: see vm_exec_eq(). */
    if (a.type == VAL_I128 || b.type == VAL_I128) {
        vm_raise_error_msg(vm,
            ">: i128 comparison is not supported on the VM (no i128 opcodes "
            "are implemented in the bytecode interpreter); use i128>? or "
            "the native backend");
        return;
    }
    if (vm_either_exact_wide(a, b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm, a, b) >  0)); return; }
    if (a.type == VAL_INT && b.type == VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i >  b.as.i)); return; }
    vm_push(vm, BOOL_VAL(as_number_vm(vm, a) >  as_number_vm(vm, b)));
}

static void vm_exec_le(VM* vm) {
    Value b = vm_pop(vm), a = vm_pop(vm);
    /* SW-09b: see vm_exec_eq(). */
    if (a.type == VAL_I128 || b.type == VAL_I128) {
        vm_raise_error_msg(vm,
            "<=: i128 comparison is not supported on the VM (no i128 opcodes "
            "are implemented in the bytecode interpreter); use i128<=? or "
            "the native backend");
        return;
    }
    if (vm_either_exact_wide(a, b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm, a, b) <= 0)); return; }
    if (a.type == VAL_INT && b.type == VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i <= b.as.i)); return; }
    vm_push(vm, BOOL_VAL(as_number_vm(vm, a) <= as_number_vm(vm, b)));
}

static void vm_exec_ge(VM* vm) {
    Value b = vm_pop(vm), a = vm_pop(vm);
    /* SW-09b: see vm_exec_eq(). */
    if (a.type == VAL_I128 || b.type == VAL_I128) {
        vm_raise_error_msg(vm,
            ">=: i128 comparison is not supported on the VM (no i128 opcodes "
            "are implemented in the bytecode interpreter); use i128>=? or "
            "the native backend");
        return;
    }
    if (vm_either_exact_wide(a, b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm, a, b) >= 0)); return; }
    if (a.type == VAL_INT && b.type == VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i >= b.as.i)); return; }
    vm_push(vm, BOOL_VAL(as_number_vm(vm, a) >= as_number_vm(vm, b)));
}

static void vm_exec_cons(VM* vm) {
    Value car = vm_pop(vm), cdr = vm_pop(vm);
    int32_t ptr = heap_alloc(&vm->heap);
    if (ptr < 0) { vm->error = 1; return; }
    vm->heap.objects[ptr]->type = HEAP_CONS;
    vm->heap.objects[ptr]->cons.car = car;
    vm->heap.objects[ptr]->cons.cdr = cdr;
    vm_push(vm, PAIR_VAL(ptr));
}

static void vm_exec_car(VM* vm) {
    Value pair = vm_pop(vm);
    if (pair.type != VAL_PAIR) {
        vm_raise_error_msg(vm, "car: argument is not a pair");
        return;
    }
    vm_push(vm, vm->heap.objects[pair.as.ptr]->cons.car);
}

static void vm_exec_cdr(VM* vm) {
    Value pair = vm_pop(vm);
    if (pair.type != VAL_PAIR) {
        vm_raise_error_msg(vm, "cdr: argument is not a pair");
        return;
    }
    vm_push(vm, vm->heap.objects[pair.as.ptr]->cons.cdr);
}

static void vm_exec_set_car(VM* vm) {
    Value val = vm_pop(vm), pair = vm_pop(vm);
    if (pair.type == VAL_PAIR) vm->heap.objects[pair.as.ptr]->cons.car = val;
    vm_push(vm, NIL_VAL);
}

static void vm_exec_set_cdr(VM* vm) {
    Value val = vm_pop(vm), pair = vm_pop(vm);
    if (pair.type == VAL_PAIR) vm->heap.objects[pair.as.ptr]->cons.cdr = val;
    vm_push(vm, NIL_VAL);
}

static void vm_exec_popn(VM* vm, int32_t operand) {
    int n = operand;
    if (n > 0 && vm->sp > n) {
        Value top = vm->stack[vm->sp - 1];
        vm->sp -= n;
        vm->stack[vm->sp - 1] = top;
    }
}

static void vm_exec_vec_create(VM* vm, int32_t operand) {
    int count = operand;
    int32_t ptr = heap_alloc(&vm->heap);
    if (ptr < 0) { vm->error = 1; return; }
    vm->heap.objects[ptr]->type = HEAP_VECTOR;
    VmVector* vec = (VmVector*)vm_alloc(&vm->heap.regions, sizeof(VmVector));
    if (!vec) { vm->error = 1; return; }
    vec->len = count;
    vec->cap = count;
    vec->items = (Value*)vm_alloc(&vm->heap.regions, count * sizeof(Value));
    if (!vec->items && count > 0) { vm->error = 1; return; }
    for (int i = count - 1; i >= 0; i--) vec->items[i] = vm_pop(vm);
    vm->heap.objects[ptr]->opaque.ptr = vec;
    vm_push(vm, (Value){.type = VAL_VECTOR, .as.ptr = ptr});
}

static void vm_exec_vec_ref(VM* vm) {
    Value idx = vm_pop(vm), vec_val = vm_pop(vm);
    if (vec_val.type == VAL_TENSOR) {
        /* SW-26: e.g. (vector-ref (fg-marginal fg 0) 0). */
        vm_vecref_tensor_path(vm, vec_val, idx);
        return;
    }
    if (vec_val.type != VAL_VECTOR) { vm_push(vm, NIL_VAL); return; }
    VmVector* vec = (VmVector*)vm->heap.objects[vec_val.as.ptr]->opaque.ptr;
    int i = (int)as_number(idx);
    if (!vec || i < 0 || i >= vec->len) {
        vm_raise_error_msg(vm, "vector-ref: index out of bounds");
        return;
    }
    vm_push(vm, vec->items[i]);
}

static void vm_exec_vec_set(VM* vm) {
    Value val = vm_pop(vm), idx = vm_pop(vm), vec_val = vm_pop(vm);
    if (vec_val.type == VAL_VECTOR) {
        VmVector* vec = (VmVector*)vm->heap.objects[vec_val.as.ptr]->opaque.ptr;
        int i = (int)as_number(idx);
        if (!vec || i < 0 || i >= vec->len) {
            vm_raise_error_msg(vm, "vector-set!: index out of bounds");
            return;
        }
        vec->items[i] = val;
    } else if (vec_val.type == VAL_TENSOR) {
        /* SW-26 sibling gap. */
        if (!vm_vecset_tensor_path(vm, vec_val, idx, val)) return;
    }
    vm_push(vm, NIL_VAL);
}

static void vm_exec_vec_len(VM* vm) {
    Value vec_val = vm_pop(vm);
    if (vec_val.type == VAL_VECTOR) {
        VmVector* vec = (VmVector*)vm->heap.objects[vec_val.as.ptr]->opaque.ptr;
        vm_push(vm, INT_VAL(vec ? vec->len : 0));
    } else if (vec_val.type == VAL_TENSOR) {
        /* SW-26 sibling gap. */
        vm_push(vm, INT_VAL(vm_veclen_tensor_path(vm, vec_val)));
    } else vm_push(vm, INT_VAL(0));
}
