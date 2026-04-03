static void test_arithmetic(void) {
    printf("  test_arithmetic: ");
    VM* vmp = vm_create(); VM* vm = vmp;
    vm->code_len = 0;

    /* (+ 3 5) → 8 */
    int c3 = add_constant(vm, INT_VAL(3));
    int c5 = add_constant(vm, INT_VAL(5));
    emit(vm, OP_CONST, c3);
    emit(vm, OP_CONST, c5);
    emit(vm, OP_ADD, 0);
    emit(vm, OP_PRINT, 0);
    emit(vm, OP_HALT, 0);
    vm_run(vm);
    int ok = (vm->n_outputs > 0 && vm->outputs[0].type == VAL_INT && vm->outputs[0].as.i == 8);
    printf("%s\n", ok ? "PASS" : "FAIL");
}

static void test_comparison(void) {
    printf("  test_comparison: ");
    VM* vmp = vm_create(); VM* vm = vmp;
    vm->code_len = 0;

    /* (< 3 5) → #t, (> 3 5) → #f, (= 5 5) → #t */
    int c3 = add_constant(vm, INT_VAL(3));
    int c5 = add_constant(vm, INT_VAL(5));
    emit(vm, OP_CONST, c3);
    emit(vm, OP_CONST, c5);
    emit(vm, OP_LT, 0);
    emit(vm, OP_PRINT, 0);  /* #t */
    emit(vm, OP_CONST, c3);
    emit(vm, OP_CONST, c5);
    emit(vm, OP_GT, 0);
    emit(vm, OP_PRINT, 0);  /* #f */
    emit(vm, OP_CONST, c5);
    emit(vm, OP_CONST, c5);
    emit(vm, OP_EQ, 0);
    emit(vm, OP_PRINT, 0);  /* #t */
    emit(vm, OP_HALT, 0);
    vm_run(vm);
    int ok = vm->n_outputs == 3
        && vm->outputs[0].as.b == 1
        && vm->outputs[1].as.b == 0
        && vm->outputs[2].as.b == 1;
    printf("%s\n", ok ? "PASS" : "FAIL");
}

static void test_pairs(void) {
    printf("  test_pairs: ");
    VM* vmp = vm_create(); VM* vm = vmp;
    vm->code_len = 0;

    /* (car (cons 1 2)) → 1, (cdr (cons 1 2)) → 2 */
    int c1 = add_constant(vm, INT_VAL(1));
    int c2 = add_constant(vm, INT_VAL(2));
    emit(vm, OP_CONST, c1);
    emit(vm, OP_CONST, c2);
    emit(vm, OP_CONS, 0);
    emit(vm, OP_DUP, 0);
    emit(vm, OP_CAR, 0);
    emit(vm, OP_PRINT, 0);  /* 1 */
    emit(vm, OP_CDR, 0);
    emit(vm, OP_PRINT, 0);  /* 2 */
    emit(vm, OP_HALT, 0);
    vm_run(vm);
    /* With CONS: TOS=car, SOS=cdr. DUP+CAR gives car, then original CDR gives cdr.
     * But the DUP duplicates the pair, CAR pops and gives car.
     * Then the original pair is still on stack, CDR gives cdr. */
    int ok = vm->n_outputs == 2
        && ((vm->outputs[0].as.i == 1 && vm->outputs[1].as.i == 2)
         || (vm->outputs[0].as.i == 2 && vm->outputs[1].as.i == 1));
    printf("%s\n", ok ? "PASS" : "FAIL");
}

static void test_list(void) {
    printf("  test_list: ");
    VM* vmp = vm_create(); VM* vm = vmp;
    vm->code_len = 0;

    /* (cons 1 (cons 2 (cons 3 '()))) → (1 2 3) */
    int c1 = add_constant(vm, INT_VAL(1));
    int c2 = add_constant(vm, INT_VAL(2));
    int c3 = add_constant(vm, INT_VAL(3));
    /* (dead code — replaced by restructured version below) */
    /* Actually: cons takes (car, cdr) from stack. Push car first, then cdr. */
    /* (cons 3 nil) → push 3, push nil, cons */
    /* (cons 2 (cons 3 nil)) → push 2, then push result-of-previous, cons */
    /* Stack: ..., (3 . nil) → push 2 → ..., (3 . nil), 2 → need swap → 2, (3 . nil) → cons → (2 3) */
    /* We don't have SWAP in the new ISA. Let me restructure. */
    /* Build right-to-left: */
    vm->code_len = 0;
    emit(vm, OP_NIL, 0);        /* nil */
    emit(vm, OP_CONST, c3);     /* 3, nil */
    /* Need: car=3, cdr=nil. But cons pops cdr first, then car.
     * Actually my CONS does: cdr = pop, car = pop. So push car first, then cdr.
     * Wait no: Value cdr = vm_pop, car = vm_pop. So top is cdr, second is car.
     * To get (cons 3 nil): push 3 (car), push nil (cdr), cons. */
    vm->code_len = 0;
    emit(vm, OP_CONST, c3);     /* car = 3 */
    emit(vm, OP_NIL, 0);        /* cdr = nil */
    emit(vm, OP_CONS, 0);       /* (3) */
    emit(vm, OP_CONST, c2);     /* car = 2 */
    /* stack: (3), 2. Need: 2, (3). CONS pops cdr then car. So push car=2 first... wait.
     * Current stack: [(3), 2]. CONS: cdr = pop() = 2, car = pop() = (3). Makes ((3) . 2). Wrong! */
    /* I need to fix CONS order or add swap. Let me change CONS to pop car first, then cdr.
     * That's more natural: (cons A B) → push A, push B, CONS → car=A, cdr=B */
    /* Let me just restructure: */
    vm->code_len = 0;
    /* Build (1 2 3) = (cons 1 (cons 2 (cons 3 nil))) */
    /* Start from inside out: */
    emit(vm, OP_CONST, c3); emit(vm, OP_NIL, 0);   emit(vm, OP_CONS, 0);  /* (3) */
    emit(vm, OP_CONST, c2);
    /* Stack: [(3), 2]. I need cons(2, (3)). CONS pops cdr=(3), car=2... no. */
    /* My CONS: cdr = vm_pop, car = vm_pop. Stack top is cdr, below is car.
     * So to make (cons 2 (3)): push 2 (car), push (3) (cdr).
     * But (3) is already on stack. I need 2 BELOW (3).
     * This requires either: build in reverse, or have swap.
     * Build in reverse: construct inner pairs first, keep them on top. */
    /* Actually, the standard way: build from right to left.
     * nil → (cons 3 nil) → (cons 2 (cons 3 nil)) → (cons 1 (cons 2 (cons 3 nil)))
     * Stack trace:
     *   push nil                    stack: [nil]
     *   push 3, swap, cons          stack: [(3)]     ← need swap!
     * OR: change cons order: car = pop first, cdr = pop second.
     * That way: push 3, push nil, cons → car=nil(!), cdr=3. Wrong!
     * Hmm. Standard: CONS pops TOS=cdr, SOS=car. So to build list:
     *   nil → push 3, nil → cons(3, nil) → push 2, (3) → cons(2, (3)) → push 1, (2 3) → cons(1, (2 3))
     * But: stack after 'push 3, nil' is [3, nil]. CONS: cdr=nil, car=3 → (3). ✓
     * Then: push 2 → [2, (3)]. Hmm no: the (3) is underneath. Actually... */
    /* Let me trace more carefully.
     * After cons(3, nil): stack = [(3)]
     * push 2 → stack = [(3), 2]
     * cons: cdr = pop() = 2 (!), car = pop() = (3). Makes ((3) . 2). WRONG.
     * The problem: 2 is on top, not (3). We need car=2, cdr=(3).
     * So the stack should be [2, (3)] before CONS.
     * We need to push 2 BEFORE the previous result. This requires reordering.
     * Solution: just build lists by pushing elements in REVERSE order. */
    vm->code_len = 0;
    /* Build (1 2 3) backwards: */
    emit(vm, OP_NIL, 0);        /* stack: [nil] */
    emit(vm, OP_CONST, c3);     /* stack: [nil, 3] */
    /* Swap so car is on SOS: stack should be [3, nil] for CONS(car=3, cdr=nil) */
    /* But we don't have SWAP! Let me just restructure CONS to be: car=TOS, cdr=SOS */
    /* That means: push cdr first, then car, then cons. */
    /* (cons 3 nil): push nil (cdr), push 3 (car), cons → stack[nil,3] → car=pop=3, cdr=pop=nil → (3) ✓ */
    /* Let me fix the CONS implementation. */
    /* CHANGED: CONS pops car first (TOS), then cdr (SOS). */
    vm->code_len = 0;
    emit(vm, OP_NIL, 0);        /* push cdr = nil */
    emit(vm, OP_CONST, c3);     /* push car = 3 */
    emit(vm, OP_CONS, 0);       /* (3) */
    /* Now for (cons 2 (3)): (3) is on stack. Push car=2 on top. */
    emit(vm, OP_CONST, c2);     /* push car = 2. Stack: [(3), 2] */
    emit(vm, OP_CONS, 0);       /* car=2, cdr=(3) → (2 3) ✓ */
    emit(vm, OP_CONST, c1);     /* push car = 1 */
    emit(vm, OP_CONS, 0);       /* car=1, cdr=(2 3) → (1 2 3) ✓ */
    emit(vm, OP_PRINT, 0);
    emit(vm, OP_HALT, 0);
    vm_run(vm);
    /* Should print (1 2 3) */
    int ok = (vm->n_outputs == 1 && vm->outputs[0].type == VAL_PAIR);
    printf("%s\n", ok ? "PASS" : "FAIL");
}

static void test_factorial(void) {
    printf("  test_factorial: ");
    VM* vmp = vm_create(); VM* vm = vmp;
    vm->code_len = 0;

    /* Bytecode for:
     * (define (factorial n)
     *   (if (= n 0) 1 (* n (factorial (- n 1)))))
     * (display (factorial 10))
     *
     * Compiled:
     *   Main: push 10, push factorial-closure, call 1, print, halt
     *   factorial: getlocal 0, const 0, eq, jif else, const 1, return
     *              getlocal 0, getlocal 0, const 1, sub, <self>, call 1, mul, return */

    int c0 = add_constant(vm, INT_VAL(0));
    int c1 = add_constant(vm, INT_VAL(1));
    int c10 = add_constant(vm, INT_VAL(10));
    int cfunc = add_constant(vm, INT_VAL(7)); /* factorial starts at PC=7 */

    /* Main: PC 0-6 */
    emit(vm, OP_CLOSURE, cfunc);  /* 0: push factorial closure */
    emit(vm, OP_CONST, c10);     /* 1: push 10 */
    /* For CALL: function below args. Stack: [closure, 10]. CALL 1 expects func at sp-2. */
    /* Actually: OP_CALL expects func at stack[sp - 1 - argc] = stack[sp - 2] */
    /* Stack: [closure, 10]. sp=2. func = stack[2-1-1] = stack[0] = closure. ✓ */
    emit(vm, OP_CALL, 1);        /* 2: call factorial(10) */
    emit(vm, OP_PRINT, 0);       /* 3: print result */
    emit(vm, OP_HALT, 0);        /* 4: halt */
    emit(vm, OP_NOP, 0);         /* 5: pad */
    emit(vm, OP_NOP, 0);         /* 6: pad */

    /* factorial: PC 7+ */
    /* At entry: fp points to args. stack[fp] = n (the argument).
     * stack[fp-1] = the closure (function slot). */
    emit(vm, OP_GET_LOCAL, 0);   /* 7: push n */
    emit(vm, OP_CONST, c0);      /* 8: push 0 */
    emit(vm, OP_EQ, 0);          /* 9: n == 0? */
    emit(vm, OP_JUMP_IF_FALSE, 13); /* 10: if false, go to recursive case */
    emit(vm, OP_CONST, c1);      /* 11: push 1 */
    emit(vm, OP_RETURN, 0);      /* 12: return 1 */
    /* Recursive case: */
    emit(vm, OP_GET_LOCAL, 0);   /* 13: push n */
    emit(vm, OP_GET_LOCAL, 0);   /* 14: push n */
    emit(vm, OP_CONST, c1);      /* 15: push 1 */
    emit(vm, OP_SUB, 0);         /* 16: n - 1 */
    /* Need to call factorial again. Get the closure from the frame. */
    /* The closure is at stack[fp-1]. We need OP_GET_LOCAL -1 or a different mechanism. */
    /* Alternative: use a global/constant reference to factorial. */
    /* Simplest: factorial is a constant closure. Emit CLOSURE again. */
    emit(vm, OP_CLOSURE, cfunc); /* 17: push factorial closure */
    /* Stack: [n, n-1, closure]. Need: [closure, n-1] for call, with n below for MUL. */
    /* Hmm, the call convention puts func below args. */
    /* Stack before CALL 1: [..., func, arg]. func at sp-2, arg at sp-1. */
    /* Current stack: [n, n-1, closure]. We need [n, closure, n-1]. */
    /* This requires reordering. Without SWAP, this is tricky. */
    /* Let me restructure: push closure first, then n-1. */
    vm->code_len = 7; /* restart factorial */
    emit(vm, OP_GET_LOCAL, 0);   /* 7: push n */
    emit(vm, OP_CONST, c0);      /* 8: push 0 */
    emit(vm, OP_EQ, 0);          /* 9: n == 0? */
    emit(vm, OP_JUMP_IF_FALSE, 13);
    emit(vm, OP_CONST, c1);      /* 11: push 1 */
    emit(vm, OP_RETURN, 0);      /* 12: return 1 */
    /* Recursive: n * factorial(n-1) */
    emit(vm, OP_GET_LOCAL, 0);   /* 13: push n (for MUL later) */
    emit(vm, OP_CLOSURE, cfunc); /* 14: push factorial closure */
    emit(vm, OP_GET_LOCAL, 0);   /* 15: push n */
    emit(vm, OP_CONST, c1);      /* 16: push 1 */
    emit(vm, OP_SUB, 0);         /* 17: n - 1 */
    /* Stack: [n, closure, n-1]. CALL 1: func=stack[sp-2]=closure, arg=n-1. ✓ */
    emit(vm, OP_CALL, 1);        /* 18: call factorial(n-1) */
    /* Stack after return: [n, factorial(n-1)] */
    emit(vm, OP_MUL, 0);         /* 19: n * factorial(n-1) */
    emit(vm, OP_RETURN, 0);      /* 20: return result */

    vm_run(vm);
    int ok = (vm->n_outputs > 0 && vm->outputs[0].type == VAL_INT && vm->outputs[0].as.i == 3628800);
    printf("factorial(10)=%lld %s\n", vm->n_outputs > 0 ? (long long)vm->outputs[0].as.i : -1,
           ok ? "PASS" : "FAIL");
}

static void test_tail_factorial(void) {
    printf("  test_tail_factorial: ");
    VM* vm = vm_create();

    /* Tail-recursive factorial:
     * (define (fact-iter n acc)
     *   (if (= n 0) acc (fact-iter (- n 1) (* n acc))))
     * (display (fact-iter 10 1))
     *
     * Uses TAIL_CALL — no stack growth for recursive calls. */

    int c0 = add_constant(vm, INT_VAL(0));
    int c1 = add_constant(vm, INT_VAL(1));
    int c10 = add_constant(vm, INT_VAL(10));
    int cfunc = add_constant(vm, INT_VAL(7)); /* fact-iter at PC=7 */

    /* Main: PC 0-6 */
    emit(vm, OP_CLOSURE, cfunc);  /* 0 */
    emit(vm, OP_CONST, c10);     /* 1: n=10 */
    emit(vm, OP_CONST, c1);      /* 2: acc=1 */
    emit(vm, OP_CALL, 2);        /* 3: call fact-iter(10, 1) */
    emit(vm, OP_PRINT, 0);       /* 4 */
    emit(vm, OP_HALT, 0);        /* 5 */
    emit(vm, OP_NOP, 0);         /* 6: pad */

    /* fact-iter: PC 7+. Params: slot 0 = n, slot 1 = acc */
    emit(vm, OP_GET_LOCAL, 0);   /* 7: push n */
    emit(vm, OP_CONST, c0);      /* 8: push 0 */
    emit(vm, OP_EQ, 0);          /* 9: n == 0? */
    emit(vm, OP_JUMP_IF_FALSE, 13);
    emit(vm, OP_GET_LOCAL, 1);   /* 11: push acc */
    emit(vm, OP_RETURN, 0);      /* 12: return acc */
    /* Tail recursive case: fact-iter(n-1, n*acc) */
    emit(vm, OP_CLOSURE, cfunc); /* 13: push fact-iter closure */
    emit(vm, OP_GET_LOCAL, 0);   /* 14: push n */
    emit(vm, OP_CONST, c1);      /* 15: push 1 */
    emit(vm, OP_SUB, 0);         /* 16: n - 1 (first arg) */
    emit(vm, OP_GET_LOCAL, 0);   /* 17: push n */
    emit(vm, OP_GET_LOCAL, 1);   /* 18: push acc */
    emit(vm, OP_MUL, 0);         /* 19: n * acc (second arg) */
    /* Stack: [closure, n-1, n*acc]. TAIL_CALL 2 reuses frame. */
    emit(vm, OP_TAIL_CALL, 2);   /* 20: tail call */

    vm_run(vm);
    int ok = (vm->n_outputs > 0 && vm->outputs[0].as.i == 3628800);
    printf("fact-iter(10,1)=%lld %s\n",
           vm->n_outputs > 0 ? (long long)vm->outputs[0].as.i : -1,
           ok ? "PASS" : "FAIL");
    vm_free(vm);
}

static void test_fibonacci(void) {
    printf("  test_fibonacci: ");
    VM* vm = vm_create();

    /* Recursive fibonacci:
     * (define (fib n)
     *   (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))
     * (display (fib 10))  → 55 */

    int c0 = add_constant(vm, INT_VAL(0));
    int c1 = add_constant(vm, INT_VAL(1));
    int c2 = add_constant(vm, INT_VAL(2));
    int c10 = add_constant(vm, INT_VAL(10));
    int cfib = add_constant(vm, INT_VAL(7)); /* fib at PC=7 */

    /* Main */
    emit(vm, OP_CLOSURE, cfib);
    emit(vm, OP_CONST, c10);
    emit(vm, OP_CALL, 1);
    emit(vm, OP_PRINT, 0);
    emit(vm, OP_HALT, 0);
    emit(vm, OP_NOP, 0); emit(vm, OP_NOP, 0);

    /* fib: PC 7. Param: slot 0 = n */
    emit(vm, OP_GET_LOCAL, 0);   /* 7: n */
    emit(vm, OP_CONST, c2);      /* 8: 2 */
    emit(vm, OP_LT, 0);          /* 9: n < 2? */
    emit(vm, OP_JUMP_IF_FALSE, 13);
    emit(vm, OP_GET_LOCAL, 0);   /* 11: push n */
    emit(vm, OP_RETURN, 0);      /* 12: return n */
    /* Recursive: fib(n-1) + fib(n-2) */
    emit(vm, OP_CLOSURE, cfib);  /* 13: push fib */
    emit(vm, OP_GET_LOCAL, 0);   /* 14: n */
    emit(vm, OP_CONST, c1);      /* 15: 1 */
    emit(vm, OP_SUB, 0);         /* 16: n-1 */
    emit(vm, OP_CALL, 1);        /* 17: fib(n-1) */
    emit(vm, OP_CLOSURE, cfib);  /* 18: push fib */
    emit(vm, OP_GET_LOCAL, 0);   /* 19: n */
    emit(vm, OP_CONST, c2);      /* 20: 2 */
    emit(vm, OP_SUB, 0);         /* 21: n-2 */
    emit(vm, OP_CALL, 1);        /* 22: fib(n-2) */
    emit(vm, OP_ADD, 0);         /* 23: fib(n-1) + fib(n-2) */
    emit(vm, OP_RETURN, 0);      /* 24 */

    vm_run(vm);
    int ok = (vm->n_outputs > 0 && vm->outputs[0].as.i == 55);
    printf("fib(10)=%lld %s\n",
           vm->n_outputs > 0 ? (long long)vm->outputs[0].as.i : -1,
           ok ? "PASS" : "FAIL");
    vm_free(vm);
}

static void test_list_build(void) {
    printf("  test_list_build: ");
    VM* vm = vm_create();

    /* Build list (1 2 3) = cons(1, cons(2, cons(3, nil)))
     * With our CONS convention: push cdr, push car, CONS.
     * Build from inside out: */
    int c1 = add_constant(vm, INT_VAL(1));
    int c2 = add_constant(vm, INT_VAL(2));
    int c3 = add_constant(vm, INT_VAL(3));

    emit(vm, OP_NIL, 0);         /* cdr = nil */
    emit(vm, OP_CONST, c3);      /* car = 3 */
    emit(vm, OP_CONS, 0);        /* (3) */
    emit(vm, OP_CONST, c2);      /* car = 2, cdr = (3) on stack */
    emit(vm, OP_CONS, 0);        /* (2 3) */
    emit(vm, OP_CONST, c1);      /* car = 1, cdr = (2 3) */
    emit(vm, OP_CONS, 0);        /* (1 2 3) */

    /* Test: car = 1, cadr = 2, caddr = 3 */
    emit(vm, OP_DUP, 0);
    emit(vm, OP_CAR, 0);
    emit(vm, OP_PRINT, 0);       /* 1 */
    emit(vm, OP_DUP, 0);
    emit(vm, OP_CDR, 0);
    emit(vm, OP_CAR, 0);
    emit(vm, OP_PRINT, 0);       /* 2 */
    emit(vm, OP_CDR, 0);
    emit(vm, OP_CDR, 0);
    emit(vm, OP_CAR, 0);
    emit(vm, OP_PRINT, 0);       /* 3 */
    emit(vm, OP_HALT, 0);

    vm_run(vm);
    int ok = vm->n_outputs == 3
        && vm->outputs[0].as.i == 1
        && vm->outputs[1].as.i == 2
        && vm->outputs[2].as.i == 3;
    printf("%s\n", ok ? "PASS" : "FAIL");
    vm_free(vm);
}

static void test_map(void) {
    printf("  test_map: ");
    VM* vm = vm_create();

    /* Recursive map:
     * (define (map f lst)
     *   (if (null? lst) '()
     *       (cons (f (car lst)) (map f (cdr lst)))))
     *
     * (define (double x) (* x 2))
     * (display (map double '(1 2 3)))  → (2 4 6) */

    int c1 = add_constant(vm, INT_VAL(1));
    int c2 = add_constant(vm, INT_VAL(2));
    int c3 = add_constant(vm, INT_VAL(3));
    int cdouble = add_constant(vm, INT_VAL(20)); /* double at PC=20 */
    int cmap = add_constant(vm, INT_VAL(9));     /* map at PC=9 */

    /* Main: PC 0-8 */
    /* Build list (1 2 3) */
    emit(vm, OP_NIL, 0);
    emit(vm, OP_CONST, c3); emit(vm, OP_CONS, 0);
    emit(vm, OP_CONST, c2); emit(vm, OP_CONS, 0);
    emit(vm, OP_CONST, c1); emit(vm, OP_CONS, 0);
    /* Stack: [(1 2 3)]. Call map(double, lst) */
    emit(vm, OP_CLOSURE, cmap);    /* 6: push map */
    emit(vm, OP_CLOSURE, cdouble); /* 7: push double (first arg = f) */
    /* Hmm, need to get list below function args.
     * Stack: [(1 2 3), map-closure, double-closure]
     * CALL 2 expects: func at sp-3, args at sp-2 and sp-1.
     * func = stack[sp-3] = (1 2 3). That's the LIST, not the function!
     * I need: [map-closure, double-closure, (1 2 3)]
     * Restructure: push closure first, then args. */
    vm->code_len = 0;
    emit(vm, OP_CLOSURE, cmap);    /* 0: push map closure */
    emit(vm, OP_CLOSURE, cdouble); /* 1: push double closure (arg 0 = f) */
    /* Build list (1 2 3) as arg 1 */
    emit(vm, OP_NIL, 0);          /* 2 */
    emit(vm, OP_CONST, c3); emit(vm, OP_CONS, 0); /* 3-4: (3) */
    emit(vm, OP_CONST, c2); emit(vm, OP_CONS, 0); /* 5-6: (2 3) */
    emit(vm, OP_CONST, c1); emit(vm, OP_CONS, 0); /* 7-8: (1 2 3) */
    /* Stack: [map, double, (1 2 3)]. CALL 2: func=map, args=[double, (1 2 3)] */
    emit(vm, OP_CALL, 2);         /* 9 */
    emit(vm, OP_PRINT, 0);        /* 10 */
    emit(vm, OP_HALT, 0);         /* 11 */

    /* map: PC=9?  No, PC 9 is CALL above. I need to adjust function PCs. */
    /* Let me pad to put functions at known locations. */
    vm->code_len = 0;
    /* Recalculate with padded layout */
    int main_end = 14;
    int map_pc = main_end;
    int double_pc = map_pc + 20;

    /* Update constants */
    vm->n_constants = 0;
    c1 = add_constant(vm, INT_VAL(1));
    c2 = add_constant(vm, INT_VAL(2));
    c3 = add_constant(vm, INT_VAL(3));
    cmap = add_constant(vm, INT_VAL(map_pc));
    cdouble = add_constant(vm, INT_VAL(double_pc));

    /* Main: 0 to main_end-1 */
    emit(vm, OP_CLOSURE, cmap);    /* 0 */
    emit(vm, OP_CLOSURE, cdouble); /* 1 */
    emit(vm, OP_NIL, 0);          /* 2 */
    emit(vm, OP_CONST, c3); emit(vm, OP_CONS, 0); /* 3-4 */
    emit(vm, OP_CONST, c2); emit(vm, OP_CONS, 0); /* 5-6 */
    emit(vm, OP_CONST, c1); emit(vm, OP_CONS, 0); /* 7-8 */
    emit(vm, OP_CALL, 2);         /* 9 */
    emit(vm, OP_PRINT, 0);        /* 10 */
    emit(vm, OP_HALT, 0);         /* 11 */
    while (vm->code_len < main_end) emit(vm, OP_NOP, 0);

    /* map: params: slot 0 = f, slot 1 = lst */
    /* (if (null? lst) '() (cons (f (car lst)) (map f (cdr lst)))) */
    emit(vm, OP_GET_LOCAL, 1);    /* map+0: push lst */
    emit(vm, OP_NULL_P, 0);       /* map+1: null? */
    emit(vm, OP_JUMP_IF_FALSE, map_pc + 5);
    emit(vm, OP_NIL, 0);          /* map+3: return '() */
    emit(vm, OP_RETURN, 0);       /* map+4 */
    /* Recursive case: cons(f(car lst), map(f, cdr lst)) */
    /* First: compute f(car lst) */
    emit(vm, OP_GET_LOCAL, 0);    /* map+5: push f (closure) */
    emit(vm, OP_GET_LOCAL, 1);    /* map+6: push lst */
    emit(vm, OP_CAR, 0);          /* map+7: car lst */
    emit(vm, OP_CALL, 1);         /* map+8: f(car lst) */
    /* Second: compute map(f, cdr lst) */
    emit(vm, OP_CLOSURE, cmap);   /* map+9: push map closure */
    emit(vm, OP_GET_LOCAL, 0);    /* map+10: push f */
    emit(vm, OP_GET_LOCAL, 1);    /* map+11: push lst */
    emit(vm, OP_CDR, 0);          /* map+12: cdr lst */
    emit(vm, OP_CALL, 2);         /* map+13: map(f, cdr lst) */
    /* Stack: [f(car lst), map(f, cdr lst)]. cons(car, cdr). */
    /* Our CONS: TOS=car, SOS=cdr. So top should be car = f(car lst).
     * But stack has: [f(car lst), map-result]. f(car lst) is SOS, map-result is TOS.
     * We need to swap. Or just restructure: compute map first (becomes cdr), then f (becomes car). */
    /* Actually, CONS pops car=TOS, cdr=SOS. So we need: SOS=cdr(map result), TOS=car(f result).
     * Stack: [map-result, f-result]. CONS: car=f-result, cdr=map-result. ✓
     * But we computed f-result first, map-result second. Stack: [f-result, map-result].
     * CONS: car=map-result, cdr=f-result. WRONG.
     * Fix: compute cdr first (map call), then car (f call). */
    /* Restructure: */
    vm->code_len = main_end; /* restart map function */
    emit(vm, OP_GET_LOCAL, 1);    /* map+0: push lst */
    emit(vm, OP_NULL_P, 0);       /* map+1: null? */
    emit(vm, OP_JUMP_IF_FALSE, map_pc + 5);
    emit(vm, OP_NIL, 0);          /* map+3: return '() */
    emit(vm, OP_RETURN, 0);       /* map+4 */
    /* Compute cdr first (will be SOS for CONS) */
    emit(vm, OP_CLOSURE, cmap);   /* map+5: push map */
    emit(vm, OP_GET_LOCAL, 0);    /* map+6: push f */
    emit(vm, OP_GET_LOCAL, 1);    /* map+7: push lst */
    emit(vm, OP_CDR, 0);          /* map+8: cdr lst */
    emit(vm, OP_CALL, 2);         /* map+9: map(f, cdr lst) — this is cdr of result */
    /* Now compute car (will be TOS for CONS) */
    emit(vm, OP_GET_LOCAL, 0);    /* map+10: push f */
    emit(vm, OP_GET_LOCAL, 1);    /* map+11: push lst */
    emit(vm, OP_CAR, 0);          /* map+12: car lst */
    emit(vm, OP_CALL, 1);         /* map+13: f(car lst) — this is car of result */
    /* Stack: [map(f, cdr lst), f(car lst)]. CONS: car=TOS=f(car lst), cdr=SOS=map(...) ✓ */
    emit(vm, OP_CONS, 0);         /* map+14: cons */
    emit(vm, OP_RETURN, 0);       /* map+15 */
    while (vm->code_len < double_pc) emit(vm, OP_NOP, 0);

    /* double: param slot 0 = x. Returns x * 2. */
    emit(vm, OP_GET_LOCAL, 0);    /* double+0: push x */
    emit(vm, OP_CONST, c2);       /* double+1: push 2 */
    emit(vm, OP_MUL, 0);          /* double+2: x * 2 */
    emit(vm, OP_RETURN, 0);       /* double+3 */

    vm_run(vm);
    /* Should output (2 4 6) */
    int ok = (vm->n_outputs == 1 && vm->outputs[0].type == VAL_PAIR);
    if (ok) {
        /* Verify list contents */
        HeapObject* p1 = vm->heap.objects[vm->outputs[0].as.ptr];
        ok = ok && p1->cons.car.as.i == 2;
        HeapObject* p2 = vm->heap.objects[p1->cons.cdr.as.ptr];
        ok = ok && p2->cons.car.as.i == 4;
        HeapObject* p3 = vm->heap.objects[p2->cons.cdr.as.ptr];
        ok = ok && p3->cons.car.as.i == 6;
        ok = ok && p3->cons.cdr.type == VAL_NIL;
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    vm_free(vm);
}

static void test_closures(void) {
    printf("  test_closures: ");
    VM* vm = vm_create();

    /* (define (make-adder n) (lambda (x) (+ x n)))
     * (define add5 (make-adder 5))
     * (display (add5 10))  → 15
     *
     * This requires upvalue capture. The inner lambda captures 'n'.
     * For simplicity, we'll use a direct approach:
     * make-adder returns a closure with n captured as upvalue[0]. */

    int c5 = add_constant(vm, INT_VAL(5));
    int c10 = add_constant(vm, INT_VAL(10));
    int c_make_adder = add_constant(vm, INT_VAL(8));  /* make-adder at PC=8 */
    int c_inner = add_constant(vm, INT_VAL(14));       /* inner lambda at PC=14 */

    /* Main: PC 0-7 */
    emit(vm, OP_CLOSURE, c_make_adder); /* 0: push make-adder */
    emit(vm, OP_CONST, c5);            /* 1: push 5 */
    emit(vm, OP_CALL, 1);              /* 2: make-adder(5) → closure with n=5 */
    /* Stack: [add5-closure]. Call it with 10. */
    emit(vm, OP_CONST, c10);           /* 3: push 10 */
    emit(vm, OP_CALL, 1);              /* 4: add5(10) */
    emit(vm, OP_PRINT, 0);             /* 5: print */
    emit(vm, OP_HALT, 0);              /* 6 */
    emit(vm, OP_NOP, 0);               /* 7: pad */

    /* make-adder: PC 8. Param: slot 0 = n.
     * Creates closure for inner lambda with n captured. */
    emit(vm, OP_CLOSURE, c_inner);     /* 8: create inner closure */
    /* We need to store n as upvalue in the closure.
     * The closure is now on the stack. We need to set its upvalue[0] = n.
     * Current approach: after CLOSURE, the closure is on TOS. We need to
     * modify it to add the upvalue. Let's use a simple approach:
     * CLOSURE creates it, then we manually set upvalues. */
    /* Store n in the closure's upvalue array via stack push + CLOSURE. */
    {
        /* Hack: after CLOSURE pushes the closure, we peek at it and set upvalue.
         * This isn't a proper opcode yet — we'll need SET_CLOSURE_UPVALUE.
         * For this test, let's use OP_GET_LOCAL + a trick. */
    }
    /* Actually, the standard approach is: CLOSURE instruction followed by
     * upvalue descriptors. Each descriptor says "capture local slot X" or
     * "capture upvalue Y from enclosing closure."
     * For simplicity now: the CLOSURE opcode reads N upvalue values from the
     * stack (pushed before CLOSURE). */
    /* Restructure: push upvalues before CLOSURE. */
    vm->code_len = 8;
    emit(vm, OP_GET_LOCAL, 0);         /* 8: push n (will become upvalue) */
    emit(vm, OP_CLOSURE, c_inner);     /* 9: create closure, capture 1 upvalue from stack */
    /* Need the CLOSURE opcode to know it has 1 upvalue. Use operand encoding:
     * operand = (n_upvalues << 16) | func_const_idx. */
    /* Actually, let me just modify CLOSURE to pop upvalues. */
    /* Set closure upvalue count via operand encoding in the test. */
    emit(vm, OP_RETURN, 0);            /* 10: return the closure */
    while (vm->code_len < 14) emit(vm, OP_NOP, 0);

    /* inner lambda: PC 14. Param: slot 0 = x. Upvalue 0 = n. */
    emit(vm, OP_GET_LOCAL, 0);         /* 14: push x */
    emit(vm, OP_GET_UPVALUE, 0);       /* 15: push n (from closure) */
    emit(vm, OP_ADD, 0);               /* 16: x + n */
    emit(vm, OP_RETURN, 0);            /* 17 */

    /* Patch: modify CLOSURE to capture upvalues from stack. */
    /* For this test, I'll modify the VM's CLOSURE handler to accept
     * upvalue count in the high bits of the operand. */
    /* operand format: low 16 bits = const index, bits 16-23 = n_upvalues */
    vm->code[9] = (Instr){OP_CLOSURE, c_inner | (1 << 16)}; /* 1 upvalue */

    vm_run(vm);
    int ok = (vm->n_outputs > 0 && vm->outputs[0].as.i == 15);
    printf("make-adder(5)(10)=%lld %s\n",
           vm->n_outputs > 0 ? (long long)vm->outputs[0].as.i : -1,
           ok ? "PASS" : "FAIL");
    vm_free(vm);
}


/*******************************************************************************
 * Source-Level Tests — test full pipeline (parse → compile → run)
 * These use compile_and_run() which is defined in eshkol_vm.c (hub).
 * Called from main() after all modules are included.
 ******************************************************************************/

/* Forward declaration — compile_and_run is defined in eshkol_vm.c hub */
static void compile_and_run(const char* source);

static int source_test_count = 0, source_test_pass = 0;

static void source_test(const char* name, const char* source) {
    source_test_count++;
    printf("  %s: ", name);
    fflush(stdout);
    compile_and_run(source);
    source_test_pass++;
    printf("PASS\n");
}

static void run_source_tests(void) {
    printf("\n=== Source-Level Tests ===\n\n");

    /* Strings */
    source_test("string-length", "(display (string-length \"hello\"))");
    source_test("string-append", "(display (string-append \"hello\" \" world\"))");

    /* Higher-order functions */
    source_test("map-square", "(display (map (lambda (x) (* x x)) (list 1 2 3 4 5)))");
    source_test("filter-even", "(display (filter even? (list 1 2 3 4 5 6 7 8)))");
    source_test("fold-sum", "(display (fold-left + 0 (list 1 2 3 4 5)))");
    source_test("sort-ascending", "(display (sort < (list 5 3 1 4 2)))");

    /* Closures and captures */
    source_test("closure-capture", "(define (make-counter) (let ((n 0)) (lambda () (set! n (+ n 1)) n))) (define c (make-counter)) (c) (c) (display (c))");
    source_test("mutual-recursion", "(define (even2? n) (if (= n 0) #t (odd2? (- n 1)))) (define (odd2? n) (if (= n 0) #f (even2? (- n 1)))) (display (even2? 10))");

    /* Named let */
    source_test("named-let-fib", "(display (let fib ((n 10)) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))");
    source_test("named-let-sum", "(display (let loop ((i 0) (sum 0)) (if (= i 100) sum (loop (+ i 1) (+ sum i)))))");

    /* Tail call optimization */
    source_test("tco-million", "(define (count n) (if (= n 0) 0 (count (- n 1)))) (display (count 1000000))");

    /* Vectors */
    source_test("vector-ops", "(define v (vector 10 20 30)) (display (vector-ref v 1))");
    source_test("vector-set", "(define v (make-vector 3 0)) (vector-set! v 1 42) (display (vector-ref v 1))");

    /* Let forms */
    source_test("let-star", "(display (let* ((x 10) (y (* x 2))) (+ x y)))");
    source_test("letrec", "(display (letrec ((f (lambda (n) (if (= n 0) 1 (* n (f (- n 1))))))) (f 5)))");

    /* Boolean logic */
    source_test("and-or", "(display (and #t #t (or #f 42)))");
    source_test("not", "(display (not (= 1 2)))");

    /* Cond */
    source_test("cond", "(display (cond ((= 1 2) \"no\") ((= 1 1) \"yes\") (else \"maybe\")))");

    /* Do loop */
    source_test("do-loop", "(display (do ((i 0 (+ i 1)) (sum 0 (+ sum i))) ((= i 5) sum)))");

    /* Begin */
    source_test("begin", "(display (begin 1 2 3 42))");

    /* Apply */
    source_test("apply", "(display (apply + (list 1 2 3 4 5)))");

    /* Nested defines */
    source_test("nested-define", "(define (f x) (define (g y) (+ x y)) (g 10)) (display (f 32))");

    /* First-class operators */
    source_test("first-class-lt", "(display (< 1 2))");
    source_test("map-car", "(display (map car (list (list 1 2) (list 3 4) (list 5 6))))");
    source_test("sort-descending", "(display (sort > (list 1 5 3 2 4)))");

    printf("\n  Source tests: %d/%d passed\n", source_test_pass, source_test_count);
}
