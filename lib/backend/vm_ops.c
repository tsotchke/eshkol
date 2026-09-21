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

/* Reverse-mode AD tracing for a binary arithmetic operation.
 *
 * When vm->active_tape is set, binary operations record on the Wengert tape in
 * addition to computing values. ad_node_map[stack_slot] tracks which tape node
 * corresponds to each stack value (-1 = untracked). Untracked operands that
 * interact with tracked ones are promoted to ad_const nodes on the tape. */
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

/**
 * @brief The four arithmetic operators, `+ - * /`, on the two values on top
 *        of the operand stack. Pops both and pushes the result.
 *
 * This is the ONLY implementation. OP_ADD/OP_SUB/OP_MUL/OP_DIV call it from
 * both dispatch loops, and so do the first-class procedures `+ - * /` (natives
 * 142-145, which the variadic prelude folds with and which `apply`, `map`,
 * `fold-left`, `reduce` and a procedure-valued variable reach).
 *
 * There used to be three copies. The first-class natives kept their own list
 * of operand types (complex, then rational, then bignum, each added after a
 * defect), which never gained the dual number, the hyper-dual, the i128 or the
 * operand check, and never recorded on the reverse tape. So
 * `(derivative (lambda (a) (fold-left + 0.0 (list (* a 1.0) (* a 2.0)))) 2.0)`
 * was 0 on the VM, as was every gradient through `apply +` (SW-183). The
 * switch-dispatch twin had no reverse-tape recording.
 *
 * Operand order is fixed: complex (which lifts a real carrier, ADR-0025), then
 * dual carriers (hyper-dual, then dual), exact domains (rational), bignum,
 * fixnum and inexact. Failure sets
 * vm->error, per this file's convention.
 */
static void vm_op_arith(VM* vm, char op) {
    int b_sp = vm->sp - 1, a_sp = vm->sp - 2;
    Value b = vm_pop(vm), a = vm_pop(vm);
    const char* name;
    int i128_id, hyper_id, dual_id, rational_id, complex_id;
    int (*tape_fn)(AdTape*, int, int);
    switch (op) {
        case '+': name = "+"; i128_id = 2103; hyper_id = 1905; dual_id = 373; rational_id = 331; complex_id = 307; tape_fn = ad_add; break;
        case '-': name = "-"; i128_id = 2104; hyper_id = 1906; dual_id = 374; rational_id = 332; complex_id = 308; tape_fn = ad_sub; break;
        case '*': name = "*"; i128_id = 2105; hyper_id = 1907; dual_id = 375; rational_id = 333; complex_id = 309; tape_fn = ad_mul; break;
        case '/': name = "/"; i128_id = 2106; hyper_id = 1908; dual_id = 376; rational_id = 334; complex_id = 310; tape_fn = ad_div; break;
        default:
            fprintf(stderr, "vm_op_arith: unknown operator '%c'\n", op);
            vm->error = 1;
            return;
    }
    if (!vm_require_arithmetic_numbers(vm, a, b, name)) return;
    /* SW-09: a heap-boxed i128 read by as_number_vm() is 0.0; route it through
     * the shared i128 kernel so fixed-width wrap semantics agree with native. */
    if (a.type == VAL_I128 || b.type == VAL_I128) {
        vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, i128_id); return;
    }
    /* A complex operand takes the complex path first: its natives lift a real
     * carrier to a complex with a tangent (ADR-0025), where the dual path
     * would read the complex as a real number. */
    if (a.type == VAL_COMPLEX || b.type == VAL_COMPLEX) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, complex_id); }
    else if (a.type == VAL_HYPER_DUAL || b.type == VAL_HYPER_DUAL) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, hyper_id); }
    else if (a.type == VAL_DUAL || b.type == VAL_DUAL)       { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, dual_id); }
    else if (a.type == VAL_RATIONAL || b.type == VAL_RATIONAL) { vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, rational_id); }
    else if (op == '/' && a.type == VAL_INT && b.type == VAL_INT) {
        /* exact/exact -> exact result (R7RS): native 334 (rational div) reduces
         * the fraction and collapses denom==1 back to an integer, so (/ 1 3) is
         * 1/3 and (/ 6 3) is 2 rather than an inexact float. */
        if (b.as.i == 0) { fprintf(stderr, "DIVIDE BY ZERO\n"); vm->error = 1; return; }
        vm_push(vm, a); vm_push(vm, b); vm_dispatch_native(vm, 334);
    }
    /* A bignum must reach the bignum domain: as_number() reads a heap pointer's
     * .as.i and answers 0.0, so the double path would silently produce 0. */
    else if (vm_either_bignum(a, b)) { vm->ad_node_map[vm->sp] = -1; vm_bignum_arith(vm, a, b, op); }
    else if (a.type == VAL_INT && b.type == VAL_INT) {
        int64_t r; int overflow;
        VM_AD_BINARY(vm, a_sp, b_sp, tape_fn, 0);
        if (op == '+')      overflow = __builtin_add_overflow(a.as.i, b.as.i, &r);
        else if (op == '-') overflow = __builtin_sub_overflow(a.as.i, b.as.i, &r);
        else                overflow = __builtin_mul_overflow(a.as.i, b.as.i, &r);
        if (overflow) vm_bignum_arith(vm, a, b, op); else vm_push(vm, INT_VAL(r));
    } else {
        double av = as_number_vm(vm, a), bv = as_number_vm(vm, b), r;
        if (op == '/') {
            /* Only EXACT-by-exact-zero is an error. With any inexact operand
             * this is IEEE-754 division and yields +nan.0 / +-inf.0 like native
             * (tests/vm_parity/corpus/37_float_div_zero.esk). */
            if (bv == 0 && vm_is_exact_number(a) && vm_is_exact_number(b)) {
                fprintf(stderr, "DIVIDE BY ZERO\n"); vm->error = 1; return;
            }
            r = av / bv;
        } else {
            r = (op == '+') ? av + bv : (op == '-') ? av - bv : av * bv;
        }
        VM_AD_BINARY(vm, a_sp, b_sp, tape_fn, 0);
        vm_push(vm, number_val_contagious(a, b, r));
    }
}


/* vm_either_ad_carrier() lives in vm_native.c, next to vm_either_exact_wide()
 * and vm_bignum_compare_vals() which it must be checked ahead of; vm_ops.c is
 * #included after vm_native.c (see eshkol_vm.c) so it is already in scope
 * here. See its doc comment there for the full SW-158 rationale. */

static void vm_exec_eq(VM* vm) {
    Value b = vm_pop(vm), a = vm_pop(vm);
    /* Generic comparison over i128 uses the shared fixed-width kernel rather
     * than as_number_vm(), which cannot inspect the boxed 128-bit payload. */
    if (a.type == VAL_I128 || b.type == VAL_I128) {
        vm_push(vm, a); vm_push(vm, b);
        vm_dispatch_native(vm, 2112);
        return;
    }
    if (vm_either_ad_carrier(a, b)) { vm_push(vm, BOOL_VAL(as_number_vm(vm, a) == as_number_vm(vm, b))); return; }
    if (vm_either_exact_wide(a, b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm, a, b) == 0)); return; }
    if (a.type == VAL_INT && b.type == VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i == b.as.i)); return; }
    vm_push(vm, BOOL_VAL(as_number_vm(vm, a) == as_number_vm(vm, b)));
}

static void vm_exec_lt(VM* vm) {
    Value b = vm_pop(vm), a = vm_pop(vm);
    if (!vm_require_arithmetic_numbers(vm, a, b, "<")) return;
    /* SW-09b: see vm_exec_eq(). */
    if (a.type == VAL_I128 || b.type == VAL_I128) {
        vm_push(vm, a); vm_push(vm, b);
        vm_dispatch_native(vm, 2113);
        return;
    }
    if (vm_either_ad_carrier(a, b)) { vm_push(vm, BOOL_VAL(as_number_vm(vm, a) <  as_number_vm(vm, b))); return; }
    if (vm_either_exact_wide(a, b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm, a, b) <  0)); return; }
    if (a.type == VAL_INT && b.type == VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i <  b.as.i)); return; }
    vm_push(vm, BOOL_VAL(as_number_vm(vm, a) <  as_number_vm(vm, b)));
}

static void vm_exec_gt(VM* vm) {
    Value b = vm_pop(vm), a = vm_pop(vm);
    if (!vm_require_arithmetic_numbers(vm, a, b, ">")) return;
    /* SW-09b: see vm_exec_eq(). */
    if (a.type == VAL_I128 || b.type == VAL_I128) {
        vm_push(vm, a); vm_push(vm, b);
        vm_dispatch_native(vm, 2114);
        return;
    }
    if (vm_either_ad_carrier(a, b)) { vm_push(vm, BOOL_VAL(as_number_vm(vm, a) >  as_number_vm(vm, b))); return; }
    if (vm_either_exact_wide(a, b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm, a, b) >  0)); return; }
    if (a.type == VAL_INT && b.type == VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i >  b.as.i)); return; }
    vm_push(vm, BOOL_VAL(as_number_vm(vm, a) >  as_number_vm(vm, b)));
}

static void vm_exec_le(VM* vm) {
    Value b = vm_pop(vm), a = vm_pop(vm);
    if (!vm_require_arithmetic_numbers(vm, a, b, "<=")) return;
    /* SW-09b: see vm_exec_eq(). */
    if (a.type == VAL_I128 || b.type == VAL_I128) {
        vm_push(vm, a); vm_push(vm, b);
        vm_dispatch_native(vm, 2115);
        return;
    }
    if (vm_either_ad_carrier(a, b)) { vm_push(vm, BOOL_VAL(as_number_vm(vm, a) <= as_number_vm(vm, b))); return; }
    if (vm_either_exact_wide(a, b)) { vm_push(vm, BOOL_VAL(vm_bignum_compare_vals(vm, a, b) <= 0)); return; }
    if (a.type == VAL_INT && b.type == VAL_INT) { vm_push(vm, BOOL_VAL(a.as.i <= b.as.i)); return; }
    vm_push(vm, BOOL_VAL(as_number_vm(vm, a) <= as_number_vm(vm, b)));
}

static void vm_exec_ge(VM* vm) {
    Value b = vm_pop(vm), a = vm_pop(vm);
    if (!vm_require_arithmetic_numbers(vm, a, b, ">=")) return;
    /* SW-09b: see vm_exec_eq(). */
    if (a.type == VAL_I128 || b.type == VAL_I128) {
        vm_push(vm, a); vm_push(vm, b);
        vm_dispatch_native(vm, 2116);
        return;
    }
    if (vm_either_ad_carrier(a, b)) { vm_push(vm, BOOL_VAL(as_number_vm(vm, a) >= as_number_vm(vm, b))); return; }
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
    vm_push(vm, (Value){.type = VAL_VOID});  /* ADR-0024: unspecified */
}

static void vm_exec_set_cdr(VM* vm) {
    Value val = vm_pop(vm), pair = vm_pop(vm);
    if (pair.type == VAL_PAIR) vm->heap.objects[pair.as.ptr]->cons.cdr = val;
    vm_push(vm, (Value){.type = VAL_VOID});  /* ADR-0024: unspecified */
}

static void vm_exec_popn(VM* vm, int32_t operand) {
    int n = operand;
    if (n > 0 && vm->sp > n) {
        /* The n slots beneath the result are this scope's locals; a closure
         * still reading one of them keeps its last value (SW-190). */
        if (vm->n_open_uvs > 0) vm_close_open_upvalues_from(vm, vm->sp - 1 - n);
        Value top = vm->stack[vm->sp - 1];
        vm->sp -= n;
        vm->stack[vm->sp - 1] = top;
    }
}

static void vm_exec_vec_create(VM* vm, int32_t operand) {
    int count = operand;
    if (count < 0) {
        vm_raise_error_msg(vm, "vector: size is outside the representable range");
        return;
    }
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
    vm_push(vm, (Value){.type = VAL_VOID});  /* ADR-0024: unspecified */
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
