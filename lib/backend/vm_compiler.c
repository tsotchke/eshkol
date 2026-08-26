static void compile_expr_impl(FuncChunk* c, Node* node, int tail);
static void compile_expr(FuncChunk* c, Node* node, int tail);

/* Element count above which a `#(...)` / `(vector ...)` literal is built by
 * allocate-then-fill (constant operand-stack depth) instead of by pushing every
 * element and running OP_VEC_CREATE. Below it the direct form is emitted, which
 * is one instruction per element and comfortably inside the operand stack;
 * above it the literal's size must not be limited by the stack, so the fill
 * form is used. */
#ifndef VM_VEC_LITERAL_STACK_CHUNK
#define VM_VEC_LITERAL_STACK_CHUNK 256
#endif

/**
 * @brief Compile node->children[first..last] as a sequence of operands left
 *        on the stack, registering each pushed result as an anonymous local
 *        so that c->n_locals keeps tracking the true compile-time stack depth
 *        above fp.
 *
 * Binding forms (let, let-star, letrec) allocate their local stack slots
 * from c->n_locals. Inline opcode forms (arithmetic, comparisons, vector-ref)
 * push operand values onto the stack without registering them; if a later
 * operand is itself a binding form it would then allocate slots that alias an
 * earlier operand still sitting on the stack, silently corrupting the result
 * (the "sibling-let corruption" defect: (+ (let ...) (let ...)) -> wrong sum).
 * The generic function-call path already tracks operands this way via
 * __call_arg__ locals; this helper applies the same invariant to the inline
 * opcode forms. The caller emits the combining opcode (which consumes the
 * operands) and then restores c->n_locals to its saved entry value.
 */
/* Pack a function's declared fixed arity into bits 32..40 of its func-PC
 * constant (bit 40 = present flag).  The PC occupies only the low 32 bits, so
 * nested-closure PC re-basing (which adds a small offset to the low word) and
 * ESKB reload leave the arity intact; OP_CLOSURE unpacks it into
 * closure.arity, which vm_closure_arity() reports to `gradient`. */
#define VM_PACK_FUNC_ARITY(pc, arity) \
    ((int64_t)(uint32_t)(pc) | (1LL << 40) | (((int64_t)((arity) & 0xFF)) << 32))

/* ── R7RS §5.3.1 TOP-LEVEL REDEFINITION ─────────────────────────────────────
 *
 * "At the top level of a program, a definition
 *      (define <variable> <expression>)
 *  has essentially the same effect as the assignment expression
 *      (set! <variable> <expression>)
 *  if <variable> is bound to a non-syntax value."
 *
 * The VM binds each top-level define to a fresh stack slot, and resolve_local()
 * scans backwards, so a *later* reference already found the later definition.
 * What it could not do is update an *earlier* one: a procedure defined before
 * the redefinition captured the first definition's slot as an upvalue, so
 * calling it after the redefinition still ran the old body.
 *
 * Top-level captures are already by reference — compile_form_define() and
 * compile_form_lambda_2() convert every is_local upvalue of a top-level
 * closure into an open (stack-slot) upvalue via native call 151, precisely so
 * a later `set!` is visible. A redefinition is a `set!`, so assigning to the
 * existing slot instead of adding a second one gives R7RS semantics for free.
 *
 * This is restricted to names the *user program* defines more than once
 * (registered below from a pre-scan of the source). Applying it to any name
 * that merely already resolves would also capture the builtin preamble and
 * the Scheme prelude, whose slots are pre-registered before user code is
 * compiled: a user `(define (car x) ...)` would then rebind the slot the
 * prelude's own `map`/`fold` read through, letting user code break the
 * standard library. Native keeps the stdlib in a separately compiled object
 * where that cannot happen, so matching it here also preserves parity.
 */
#define VM_MAX_REDEFINED_NAMES 128
static char g_vm_redefined_names[VM_MAX_REDEFINED_NAMES][128];
static int  g_vm_n_redefined = 0;

/** @brief Forget every registered redefined top-level name. */
static void vm_clear_redefined_toplevel_names(void) {
    g_vm_n_redefined = 0;
}

/** @brief Register @p name as defined more than once at the program's top
 *         level (idempotent; silently ignored past the table capacity, which
 *         only costs the old stale-binding behaviour). */
static void vm_add_redefined_toplevel_name(const char* name) {
    if (!name || !name[0]) return;
    for (int i = 0; i < g_vm_n_redefined; i++)
        if (strcmp(g_vm_redefined_names[i], name) == 0) return;
    if (g_vm_n_redefined >= VM_MAX_REDEFINED_NAMES) return;
    strncpy(g_vm_redefined_names[g_vm_n_redefined], name, 127);
    g_vm_redefined_names[g_vm_n_redefined][127] = 0;
    g_vm_n_redefined++;
}

/** @return 1 if @p name is defined more than once at the top level of the
 *          program being compiled. */
static int vm_is_redefined_toplevel_name(const char* name) {
    if (!name || !name[0]) return 0;
    for (int i = 0; i < g_vm_n_redefined; i++)
        if (strcmp(g_vm_redefined_names[i], name) == 0) return 1;
    return 0;
}

/** @return the name a top-level `define` form binds — `x` for both
 *          `(define x v)` and `(define (x . args) body)` — or NULL if @p e is
 *          not a define form. */
static const char* vm_define_bound_name(Node* e) {
    if (!e || e->type != N_LIST || e->n_children < 2) return NULL;
    if (e->children[0]->type != N_SYMBOL) return NULL;
    if (strcmp(e->children[0]->symbol, "define") != 0) return NULL;
    Node* target = e->children[1];
    if (target->type == N_SYMBOL) return target->symbol;
    if (target->type == N_LIST && target->n_children >= 1 &&
        target->children[0]->type == N_SYMBOL) {
        return target->children[0]->symbol;
    }
    return NULL;
}

/** @brief Register every name that @p n top-level @p forms define more than
 *         once (R7RS §5.3.1). Replaces any previous registration. */
static void vm_register_redefined_from_forms(Node** forms, int n) {
    vm_clear_redefined_toplevel_names();
    if (!forms) return;

    char seen[VM_MAX_REDEFINED_NAMES * 4][128];
    int n_seen = 0;
    for (int i = 0; i < n; i++) {
        const char* name = vm_define_bound_name(forms[i]);
        if (!name) continue;
        int already = 0;
        for (int s = 0; s < n_seen; s++)
            if (strcmp(seen[s], name) == 0) { already = 1; break; }
        if (already) {
            vm_add_redefined_toplevel_name(name);
        } else if (n_seen < (int)(sizeof(seen) / sizeof(seen[0]))) {
            strncpy(seen[n_seen], name, 127);
            seen[n_seen][127] = 0;
            n_seen++;
        }
    }
}

/** @brief Same as vm_register_redefined_from_forms() for a driver that has no
 *         parsed top-level array — parses @p source into a throw-away AST just
 *         to count the definitions. The parser has no registration side
 *         effects (macros are expanded during compilation, not parsing), so
 *         reading the source twice is safe. */
static void vm_prescan_redefined_toplevel_names(const char* source) {
    vm_clear_redefined_toplevel_names();
    if (!source) return;

    const char* saved_src = src_ptr;
    src_ptr = source;

    Node* forms[VM_MAX_REDEFINED_NAMES * 8];
    int n_forms = 0;
    while (n_forms < (int)(sizeof(forms) / sizeof(forms[0]))) {
        skip_ws();
        if (!*src_ptr) break;
        Node* expr = parse_sexp();
        if (!expr) break;
        forms[n_forms++] = expr;
    }

    vm_register_redefined_from_forms(forms, n_forms);

    for (int i = 0; i < n_forms; i++) free_node(forms[i]);
    src_ptr = saved_src;
}

/* First stack slot that belongs to the user program: every slot below it was
 * created by emit_builtin_preamble() or by the Scheme prelude. A redefinition
 * may only assign to a location the user program itself created — otherwise the
 * FIRST user definition of a name that happens to collide with a builtin (and
 * is then redefined) would overwrite the preamble's own closure slot, letting
 * user code rebind what the prelude's map/fold read through. Native keeps its
 * stdlib in a separately compiled object where that cannot happen; the
 * watermark keeps the VM matching it. */
static int g_vm_user_locals_base = 0;

/** @brief Mark @p n_locals as the boundary between prelude and user slots. */
static void vm_set_user_locals_base(int n_locals) {
    g_vm_user_locals_base = n_locals > 0 ? n_locals : 0;
}

/* The group-compilation driver in compile_and_run() pre-registers a NIL local
 * per member of a mutually-recursive define group before compiling any of
 * them, so inside a group `resolve_local` finds a slot that no definition has
 * run yet. Suppress the redefinition rule for that window. */
static int g_vm_predeclared_group_depth = 0;

/** @return the existing top-level slot a redefinition of @p name must assign
 *          to, or -1 when this define should create a new binding.
 *
 * A heap-boxed target (set!-mutated *and* captured, so its slot holds a
 * 1-element vector) is declined: assigning into the box means emitting
 * GET_LOCAL/CONST before the value, which would leave two untracked values
 * under the value's own compile-time stack accounting. Such a name would have
 * to be redefined *and* set!-mutated *and* captured; declining leaves it on
 * the previous behaviour rather than risking mis-tracked slots. */
static int vm_redefinition_target_slot(FuncChunk* c, const char* name) {
    if (!c || c->enclosing != NULL) return -1;          /* top level only */
    if (g_vm_predeclared_group_depth > 0) return -1;
    if (!vm_is_redefined_toplevel_name(name)) return -1;

    int slot = resolve_local(c, name);
    if (slot < 0) return -1;
    if (slot < g_vm_user_locals_base) return -1;   /* prelude/builtin location */
    for (int li = c->n_locals - 1; li >= 0; li--) {
        if (c->locals[li].slot == slot && c->locals[li].boxed) return -1;
    }
    return slot;
}

/** @return 1 when @p name is bound by USER code somewhere in scope — a
 *          parameter or let/lambda local of any chunk on the scope chain, or
 *          a root-chunk slot at or above the user-locals watermark.
 *
 * SW-24 (ESH-0070 class): the builtin-procedure fast paths in compile_expr
 * (arithmetic/comparison opcodes, car/cdr chains, display, …) dispatch on
 * the head SYMBOL alone, so `(define + (lambda (a b) (* a b))) (+ 3 4)`
 * still emitted OP_ADD and printed 7. A user binding must shadow the fast
 * path; the preamble/prelude's own bindings (root slots BELOW the
 * watermark) must not, because the fast paths intentionally implement
 * exactly those. A watermark of 0 means user code has not started yet (the
 * prelude itself is being compiled), so nothing counts as a user rebinding.
 */
static int vm_head_user_rebound(FuncChunk* c, const char* name) {
    for (FuncChunk* p = c; p; p = p->enclosing) {
        int slot = resolve_local(p, name);
        if (slot >= 0) {
            if (p->enclosing == NULL)
                return g_vm_user_locals_base > 0 && slot >= g_vm_user_locals_base;
            return 1;
        }
    }
    return 0;
}

static void compile_operands_tracked(FuncChunk* c, Node* node, int first, int last) {
    for (int i = first; i <= last; i++) {
        compile_expr(c, node->children[i], 0);
        add_local(c, "__operand__");
    }
}

/** @brief Emit bytecode that constructs a quoted-symbol value: packs
 *         @p symbol's bytes into 8-byte constant chunks and passes them to
 *         native call 101 (symbol construction). Shared by compile_quote()
 *         and compile_quasiquote() paths that need to quote a bare
 *         symbol. */
static void compile_symbol_literal(FuncChunk* c, const char* symbol) {
    int len = symbol ? (int)strlen(symbol) : 0;
    int n_packs = (len + 7) / 8;
    chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(len)));
    for (int p = 0; p < n_packs; p++) {
        int64_t pack = 0;
        for (int b = 0; b < 8 && p * 8 + b < len; b++)
            pack |= ((int64_t)(unsigned char)symbol[p * 8 + b]) << (b * 8);
        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(pack)));
    }
    chunk_emit(c, OP_NATIVE_CALL, 101);
}

/**
 * @brief Compile a `(quasiquote node)` template: `(unquote x)` compiles
 *        @c x normally; atoms compile as literal constants (numbers,
 *        quoted symbols via compile_symbol_literal(), strings, booleans);
 *        lists are rebuilt right-to-left via OP_CONS, splicing
 *        `(unquote-splicing x)` elements in via a native "append" call
 *        (native id 73) instead of consing.
 */
/** @brief True when @p n is an `(unquote-splicing x)` form. */
static int vm_qq_is_splice(Node* n) {
    return n && n->type == N_LIST && n->n_children == 2 &&
           n->children[0]->type == N_SYMBOL &&
           strcmp(n->children[0]->symbol, "unquote-splicing") == 0;
}

static void compile_quasiquote(FuncChunk* c, Node* node) {
    if (!node) { chunk_emit(c, OP_NIL, 0); return; }

    /* (unquote x) -> compile x normally */
    if (node->type == N_LIST && node->n_children == 2 &&
        node->children[0]->type == N_SYMBOL && strcmp(node->children[0]->symbol, "unquote") == 0) {
        compile_expr(c, node->children[1], 0);
        return;
    }

    /* Atom: number */
    if (node->type == N_NUMBER) {
        int ci = chunk_add_const(c, node->is_int ? INT_VAL(node->ival)
            : (node->numval == (int64_t)node->numval ? INT_VAL((int64_t)node->numval) : FLOAT_VAL(node->numval)));
        if (ci >= 0) chunk_emit(c, OP_CONST, ci);
        return;
    }
    /* Atom: symbol — quote as string */
    if (node->type == N_SYMBOL) {
        compile_symbol_literal(c, node->symbol);
        return;
    }
    /* Atom: string */
    if (node->type == N_STRING) {
        compile_expr(c, node, 0);
        return;
    }
    /* Atom: boolean */
    if (node->type == N_BOOL) {
        chunk_emit(c, node->numval ? OP_TRUE : OP_FALSE, 0);
        return;
    }

    /* List: accumulate LEFT TO RIGHT in segments.
     *
     * This used to build right-to-left, consing literals onto an accumulator
     * and calling `append` for each `,@`. But native call 73 pops its SECOND
     * operand from the top of stack (`b = pop, a = pop; result = a ++ b`),
     * and the accumulator already sat underneath — so every splice computed
     * `acc ++ spliced` and the spliced elements landed at the END of the
     * list regardless of where they were written:
     *
     *   `(1 ,@xs 5 6)   with xs = (2 3 4)   =>  (1 5 6 2 3 4)   not (1 2 3 4 5 6)
     *   `(,@a ,@b)                          =>  (p q x y)       not (x y p q)
     *
     * Accumulating left to right makes the operand order come out right with
     * no new opcode: each maximal run of literal elements is built into its
     * own segment list and appended to the accumulator, and each `,@` is
     * appended in the position it was written. `append` returns its second
     * argument unchanged when the first is '(), so the initial empty
     * accumulator costs nothing. */
    if (node->type == N_LIST) {
        chunk_emit(c, OP_NIL, 0); /* accumulator = '() */
        int i = 0;
        while (i < node->n_children) {
            Node* elem = node->children[i];
            if (vm_qq_is_splice(elem)) {
                compile_expr(c, elem->children[1], 0);  /* TOS = spliced list */
                chunk_emit(c, OP_NATIVE_CALL, 73);      /* acc ++ spliced */
                i++;
                continue;
            }
            /* Maximal run of non-spliced elements [i, j). */
            int j = i;
            while (j < node->n_children && !vm_qq_is_splice(node->children[j])) j++;
            chunk_emit(c, OP_NIL, 0);                   /* segment tail */
            for (int k = j - 1; k >= i; k--) {
                compile_quasiquote(c, node->children[k]);
                chunk_emit(c, OP_CONS, 0);
            }
            chunk_emit(c, OP_NATIVE_CALL, 73);          /* acc ++ segment */
            i = j;
        }
        return;
    }

    /* Fallback: emit nil */
    chunk_emit(c, OP_NIL, 0);
}

/* compile_depth tracked in global context — not in CompilerContext struct
 * because it's transient per-compilation, not persistent state */
static int compile_depth = 0;

/** @brief Recursion-depth-guarded wrapper around compile_expr_impl():
 *         bumps/checks compile_depth (erroring past 1000 nested
 *         expressions) around the actual compilation call. */
static void compile_expr(FuncChunk* c, Node* node, int tail) {
    compile_depth++;
    if (compile_depth > 1000) { fprintf(stderr, "ERROR: expression nesting too deep (>1000)\n"); compile_depth--; return; }
    compile_expr_impl(c, node, tail);
    compile_depth--;
}

/** @brief Map a call head symbol (`+`/`add`/`tensor-add`/etc. and its
 *         sub/mul/div counterparts) to the corresponding GPU-dispatchable
 *         element-wise tensor-op native call ID, or -1 if @p fn isn't one
 *         of the recognized aliases. */
static int gpu_elementwise_native_id(Node* fn) {
    if (is_sym(fn, "+") || is_sym(fn, "add") || is_sym(fn, "add2") || is_sym(fn, "tensor-add")) return 441;
    if (is_sym(fn, "-") || is_sym(fn, "sub") || is_sym(fn, "sub2") || is_sym(fn, "tensor-sub")) return 442;
    if (is_sym(fn, "*") || is_sym(fn, "mul") || is_sym(fn, "mul2") || is_sym(fn, "tensor-mul")) return 443;
    if (is_sym(fn, "/") || is_sym(fn, "div") || is_sym(fn, "div2") || is_sym(fn, "tensor-div")) return 444;
    return -1;
}

/** @brief Map a call head symbol (`+`/`sum`/`tensor-sum`/etc. and its
 *         mean/max/min counterparts) to the corresponding GPU-dispatchable
 *         tensor-reduce native call ID, or -1 if @p fn isn't one of the
 *         recognized aliases. */
static int gpu_reduce_native_id(Node* fn) {
    if (is_sym(fn, "+") || is_sym(fn, "sum") || is_sym(fn, "tensor-sum") || is_sym(fn, "_tensor-reduce-sum")) return 457;
    if (is_sym(fn, "mean") || is_sym(fn, "tensor-mean") || is_sym(fn, "_tensor-reduce-mean")) return 458;
    if (is_sym(fn, "max") || is_sym(fn, "tensor-max") || is_sym(fn, "_tensor-reduce-max")) return 459;
    if (is_sym(fn, "min") || is_sym(fn, "tensor-min") || is_sym(fn, "_tensor-reduce-min")) return 460;
    return -1;
}


/* ═══ Extracted compilation sub-functions ═══ */

/** @brief Compile a `(cond clause...)` special form: chains
 *         JUMP_IF_FALSE tests through each `(test body...)` clause
 *         (with `else` always taken) to the matching consequent, whose
 *         last body expression is compiled in tail position when @p tail
 *         is set. Non-last body expressions are compiled and popped
 *         (implicit begin). */
static void compile_form_cond(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    int end_patches[64];
    int n_patches = 0;
    for (int i = 1; i < node->n_children; i++) {
        Node* clause = node->children[i];
        if (clause->type != N_LIST || clause->n_children < 1) continue;
        /* R7RS 4.2.1: a `(<test>)` clause with NO body evaluates to the TEST
         * VALUE when the test is truthy, and falls through when it is not.
         * These clauses used to be skipped outright by an `n_children < 2`
         * guard, so `(cond ((+ 1 2)) (else 'fail))` returned `fail` on the VM
         * and `3` natively — the whole point of the idiom, `(cond ((assoc k
         * alist)) (else ...))`, silently produced the wrong branch. */
        if (clause->n_children == 1 && !is_sym(clause->children[0], "else")) {
            compile_expr(c, clause->children[0], 0);   /* test -> TOS */
            chunk_emit(c, OP_DUP, 0);                  /* keep a copy       */
            int jfall = placeholder(c);                /* pops the copy     */
            if (n_patches < 64) end_patches[n_patches++] = placeholder(c);
            patch(c, jfall, OP_JUMP_IF_FALSE, c->code_len);
            chunk_emit(c, OP_POP, 0);                  /* falsy: discard it  */
            continue;
        }
        if (clause->n_children < 2) continue;
        if (is_sym(clause->children[0], "else")) {
            /* else clause — always taken */
            for (int j = 1; j < clause->n_children; j++) {
                if (j < clause->n_children - 1) { compile_expr(c, clause->children[j], 0); chunk_emit(c, OP_POP, 0); }
                else compile_expr(c, clause->children[j], tail);
            }
            break;
        }
        /* Test → if false, jump to next clause */
        compile_expr(c, clause->children[0], 0);
        int jnext = placeholder(c);
        /* Body */
        for (int j = 1; j < clause->n_children; j++) {
            if (j < clause->n_children - 1) { compile_expr(c, clause->children[j], 0); chunk_emit(c, OP_POP, 0); }
            else compile_expr(c, clause->children[j], tail);
        }
        if (n_patches < 64) end_patches[n_patches++] = placeholder(c); /* jump to end */
        patch(c, jnext, OP_JUMP_IF_FALSE, c->code_len);
    }
    /* Patch all end jumps */
    for (int i = 0; i < n_patches; i++) patch(c, end_patches[i], OP_JUMP, c->code_len);
    return;
}

/** @brief Compile a `(case key-expr ((datum...) body...)... [(else
 *         body...)])` special form: evaluates the key once, then for each
 *         clause tests it (via OP_EQ against each quoted datum) and
 *         branches to the matching clause's body, with `else` always
 *         taken as a fallback. */
static void compile_form_case(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    compile_expr(c, node->children[1], 0); /* evaluate key expression → TOS */
    int end_patches_c[64]; int n_patches_c = 0;
    for (int i = 2; i < node->n_children; i++) {
        Node* clause = node->children[i];
        if (clause->type != N_LIST || clause->n_children < 2) continue;
        if (is_sym(clause->children[0], "else")) {
            chunk_emit(c, OP_POP, 0); /* discard key */
            for (int j = 1; j < clause->n_children; j++) {
                if (j < clause->n_children - 1) { compile_expr(c, clause->children[j], 0); chunk_emit(c, OP_POP, 0); }
                else compile_expr(c, clause->children[j], tail);
            }
            break;
        }
        /* ((val1 val2 ...) body ...) */
        Node* vals = clause->children[0];
        if (vals->type != N_LIST) continue;
        /* Test key against each val: DUP, EQ, if true → jump to body */
        int body_patches[16]; int n_bp = 0;
        for (int v = 0; v < vals->n_children; v++) {
            chunk_emit(c, OP_DUP, 0);
            compile_quote(c, vals->children[v]);
            chunk_emit(c, OP_EQ, 0);
            /* If true, jump to body */
            if (n_bp < 16) body_patches[n_bp++] = c->code_len;
            chunk_emit(c, OP_JUMP_IF_FALSE, 0); /* placeholder: if false, continue */
            /* Match! Jump to body code */
            int jbody = c->code_len;
            chunk_emit(c, OP_JUMP, 0); /* placeholder: jump to body */
            /* Patch the JIF to skip the JUMP (continue testing) */
            patch(c, body_patches[n_bp-1], OP_JUMP_IF_FALSE, c->code_len);
            body_patches[n_bp-1] = jbody; /* reuse slot for body jump */
        }
        /* No val matched — jump to next clause */
        int jnext = c->code_len;
        chunk_emit(c, OP_JUMP, 0);
        /* Body code (reached by any matching val's jump) */
        for (int bp = 0; bp < n_bp; bp++)
            patch(c, body_patches[bp], OP_JUMP, c->code_len);
        chunk_emit(c, OP_POP, 0); /* discard key */
        for (int j = 1; j < clause->n_children; j++) {
            if (j < clause->n_children - 1) { compile_expr(c, clause->children[j], 0); chunk_emit(c, OP_POP, 0); }
            else compile_expr(c, clause->children[j], tail);
        }
        if (n_patches_c < 64) end_patches_c[n_patches_c++] = c->code_len;
        chunk_emit(c, OP_JUMP, 0);
        /* Patch jnext to after body */
        patch(c, jnext, OP_JUMP, c->code_len);
    }
    for (int i = 0; i < n_patches_c; i++) patch(c, end_patches_c[i], OP_JUMP, c->code_len);
    return;
}

/**
 * @brief Reports whether @p name is a library this compilation unit has
 *        already defined with `define-library`.
 *
 * R7RS-small 5.6.1: a `define-library` form defines its library for the forms
 * that follow it, so an import of such a name must resolve from the unit and
 * never from the filesystem.  Being *compiled* is what makes a library
 * resolvable, which is also what makes an import written above its
 * `define-library` correctly fail to find it.
 */
/* A compile-time defect the VM must not run past.
 *
 * The VM's ordinary diagnostics are warnings (an unbound name is reported and
 * the program still runs), which is exactly how an `import` written above its
 * `define-library` could produce a program that ran and answered while the
 * native lanes refused to compile it at all.  This flag is the fail-closed
 * signal for that class: both drivers check it after compiling and refuse to
 * execute or to emit bytecode. */
static int g_vm_compile_failed = 0;

/**
 * @brief Reports a fatal compile-time defect and arms the fail-closed flag.
 *
 * @param message  the defect, one line.
 * @param detail   optional second line naming the rule that was broken;
 *                 may be NULL.
 */
static void vm_compile_error(const char* message, const char* detail) {
    fprintf(stderr, "ERROR: %s\n", message ? message : "compilation failed");
    if (detail && *detail) fprintf(stderr, "ERROR:   %s\n", detail);
    g_vm_compile_failed = 1;
}

/** @return nonzero when this compilation hit a fatal defect. */
static int vm_compile_failed(void) { return g_vm_compile_failed; }

/** @brief Clears the fail-closed flag at the start of a compilation. */
static void vm_clear_compile_failure(void) { g_vm_compile_failed = 0; }

/* Libraries this compilation unit defines somewhere BELOW the point currently
 * being compiled.  Populated before compilation begins, and emptied name by
 * name as each `define-library` is reached, so whatever is still listed when
 * an import fails is precisely the set of libraries that exist in this unit
 * but are written after the import — the forward reference R7RS-small 5.6.1
 * forbids.  Mirrors library_registry::planUnit() on the native side. */
static char g_vm_planned_libraries[64][128];
static int g_vm_n_planned_libraries = 0;

static void vm_clear_planned_libraries(void) { g_vm_n_planned_libraries = 0; }

static void vm_add_planned_library(const char* name) {
    if (!name || !*name || g_vm_n_planned_libraries >= 64) return;
    for (int i = 0; i < g_vm_n_planned_libraries; i++)
        if (strcmp(g_vm_planned_libraries[i], name) == 0) return;
    strncpy(g_vm_planned_libraries[g_vm_n_planned_libraries], name, 127);
    g_vm_planned_libraries[g_vm_n_planned_libraries][127] = '\0';
    g_vm_n_planned_libraries++;
}

static int vm_library_planned_later(const char* name) {
    if (!name) return 0;
    for (int i = 0; i < g_vm_n_planned_libraries; i++)
        if (strcmp(g_vm_planned_libraries[i], name) == 0) return 1;
    return 0;
}

static void vm_drop_planned_library(const char* name) {
    for (int i = 0; i < g_vm_n_planned_libraries; i++) {
        if (strcmp(g_vm_planned_libraries[i], name) != 0) continue;
        for (int j = i + 1; j < g_vm_n_planned_libraries; j++)
            memcpy(g_vm_planned_libraries[j - 1], g_vm_planned_libraries[j], 128);
        g_vm_n_planned_libraries--;
        return;
    }
}

static int vm_unit_library_index(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < g_compiler_ctx.n_unit_libraries; i++) {
        if (strcmp(g_compiler_ctx.unit_libraries[i].name, name) == 0) return i;
    }
    return -1;
}

static int vm_unit_library_defined(const char* name) {
    return vm_unit_library_index(name) >= 0;
}

/**
 * @brief Records @p name as a library defined by this compilation unit,
 *        together with the union of its `export` clauses.
 *
 * A repeated `define-library` for the same name re-establishes it, matching
 * the native registry's last-definition-wins rule.
 */
static void vm_unit_library_define(const char* name,
                                   const char exports[][128], int n_exports) {
    if (!name || !*name) return;
    int idx = vm_unit_library_index(name);
    if (idx < 0) {
        if (g_compiler_ctx.n_unit_libraries >= 64) return;
        idx = g_compiler_ctx.n_unit_libraries++;
        strncpy(g_compiler_ctx.unit_libraries[idx].name, name, 127);
        g_compiler_ctx.unit_libraries[idx].name[127] = '\0';
    }
    g_compiler_ctx.unit_libraries[idx].n_exports = 0;
    for (int i = 0; i < n_exports && i < 64; i++) {
        strncpy(g_compiler_ctx.unit_libraries[idx].exports[i], exports[i], 127);
        g_compiler_ctx.unit_libraries[idx].exports[i][127] = '\0';
        g_compiler_ctx.unit_libraries[idx].n_exports++;
    }
}

/**
 * @brief Joins an R7RS library-name datum into the dotted module name the
 *        rest of the module machinery speaks.
 *
 * `(smoke v1_3)` becomes `smoke.v1_3`, matching what the native front end's
 * parse_r7rs_library_name() produces, so both back ends resolve the same
 * library to the same file (`smoke/v1_3.esk`) when it is not in this unit.
 * R7RS permits exact non-negative integers as name components, so an integer
 * literal is joined by its written value.
 *
 * @return 1 on success, 0 if @p datum is not a well-formed library name.
 */
static int vm_library_name_from_datum(const Node* datum, char* out, size_t out_size) {
    if (!datum || !out || out_size == 0) return 0;
    if (datum->type != N_LIST || datum->n_children < 1) return 0;
    out[0] = '\0';
    size_t used = 0;
    for (int i = 0; i < datum->n_children; i++) {
        const Node* part = datum->children[i];
        char piece[128];
        if (part->type == N_SYMBOL) {
            snprintf(piece, sizeof(piece), "%s", part->symbol);
        } else if (part->type == N_NUMBER && part->is_int && part->ival >= 0) {
            snprintf(piece, sizeof(piece), "%lld", (long long)part->ival);
        } else {
            return 0;
        }
        size_t need = strlen(piece) + (i > 0 ? 1 : 0);
        if (used + need + 1 > out_size) return 0;
        if (i > 0) out[used++] = '.';
        memcpy(out + used, piece, strlen(piece));
        used += strlen(piece);
        out[used] = '\0';
    }
    return used > 0;
}

/**
 * @brief Notes every `define-library` in @p forms that has not been compiled
 *        yet, so an import above one can be reported as the forward reference
 *        it is rather than as a missing file.
 */
static void vm_plan_unit_libraries(Node* const* forms, int n_forms) {
    vm_clear_planned_libraries();
    if (!forms) return;
    for (int i = 0; i < n_forms; i++) {
        const Node* f = forms[i];
        if (!f || f->type != N_LIST || f->n_children < 2) continue;
        if (f->children[0]->type != N_SYMBOL) continue;
        if (strcmp(f->children[0]->symbol, "define-library") != 0) continue;
        char name[256];
        if (!vm_library_name_from_datum(f->children[1], name, sizeof(name))) continue;
        if (vm_unit_library_defined(name)) continue;
        vm_add_planned_library(name);
    }
}

/**
 * @brief vm_plan_unit_libraries() for a driver with no parsed top-level array:
 *        parses @p source into throw-away forms just to note the library names.
 *
 * The parser has no registration side effects, so reading the source twice is
 * safe — the same argument vm_prescan_redefined_toplevel_names() relies on.
 */
static void vm_prescan_unit_libraries(const char* source) {
    vm_clear_planned_libraries();
    if (!source) return;

    const char* saved_src = src_ptr;
    src_ptr = source;

    Node* forms[VM_MAX_REDEFINED_NAMES * 8];
    int n_forms = 0;
    while (n_forms < (int)(sizeof(forms) / sizeof(forms[0]))) {
        skip_ws();
        if (!*src_ptr) break;
        Node* expr = parse_sexp();
        if (!expr) break;
        forms[n_forms++] = expr;
    }

    vm_plan_unit_libraries(forms, n_forms);

    for (int i = 0; i < n_forms; i++) free_node(forms[i]);
    src_ptr = saved_src;
}

/**
 * @brief Peels the R7RS import-set wrappers off @p set and returns the
 *        library-name datum underneath.
 *
 * `(only (m) a)`, `(except (m) a)`, `(prefix (m) p:)` and `(rename (m) (a b))`
 * all name a library in their second position; anything else IS the library
 * name.  The VM has no per-module visibility boundary — every top-level
 * binding is a global of the one unit — so the filters themselves are not
 * lowered here; what matters for resolution is which library is named.
 */
static const Node* vm_import_set_library_datum(const Node* set) {
    if (!set || set->type != N_LIST || set->n_children < 1) return set;
    const Node* head = set->children[0];
    if (head->type == N_SYMBOL && set->n_children >= 2 &&
        (strcmp(head->symbol, "only") == 0 || strcmp(head->symbol, "except") == 0 ||
         strcmp(head->symbol, "prefix") == 0 || strcmp(head->symbol, "rename") == 0)) {
        return vm_import_set_library_datum(set->children[1]);
    }
    return set;
}

/**
 * @brief Loads and compiles the source file backing dotted module @p mod_name.
 *
 * Shared by `(require m)` and `(import (m …))` so the two forms cannot drift
 * apart in what they resolve or how often they load it.  Emits no balancing
 * value of its own: the caller owns the stack contract described on
 * compile_form_require().
 */
static void vm_compile_module_by_name(FuncChunk* c, const char* mod_name) {
    if (!mod_name || !*mod_name) return;

    /* Track already-loaded modules to avoid double-loading */
    for (int i = 0; i < g_compiler_ctx.n_loaded; i++) {
        if (strcmp(g_compiler_ctx.loaded_modules[i], mod_name) == 0) return;
    }
    if (g_compiler_ctx.n_loaded < 64)
        strncpy(g_compiler_ctx.loaded_modules[g_compiler_ctx.n_loaded++], mod_name, 127);

    /* stdlib is the prelude — builtins already available */
    if (strcmp(mod_name, "stdlib") == 0) return;

    /* Build file path: module.name → lib/module/name.esk */
    char path[512];
    snprintf(path, sizeof(path), "lib/");
    int pi = 4;
    for (const char* p = mod_name; *p && pi < 500; p++) {
        path[pi++] = (*p == '.') ? '/' : *p;
    }
    path[pi] = '\0';
    strncat(path, ".esk", sizeof(path) - pi - 1);

#ifdef ESHKOL_VM_NO_DISASM
    /* WASM mode: no filesystem access. Prelude builtins already available. */
    return;
#else
    /* Read and parse the file */
    FILE* mf = fopen(path, "r");
    if (!mf) {
        /* Try alternative path: replace ALL dots with slashes */
        char alt[512];
        snprintf(alt, sizeof(alt), "%s.esk", mod_name);
        for (char* p = alt; *p; p++) if (*p == '.') *p = '/';
        mf = fopen(alt, "r");
    }
    if (mf) {
        fseek(mf, 0, SEEK_END);
        long len = ftell(mf);
        fseek(mf, 0, SEEK_SET);
        char* src = (char*)malloc(len + 1);
        if (src) {
            fread(src, 1, len, mf);
            src[len] = '\0';
            fclose(mf);
            /* Parse and compile all top-level forms.
             *
             * Under the SAME stack discipline the unit's own top level uses
             * (eshkol_vm.c): a form that bound nothing left one value behind,
             * and dropping the POP here desynchronized `n_locals` from the
             * real stack depth for every module containing a non-defining
             * top-level form — every later local in that module, and in the
             * importing unit, then addressed the wrong slot. */
            const char* saved_src = src_ptr;
            src_ptr = src;
            while (1) {
                skip_ws();
                if (!*src_ptr) break;
                Node* expr = parse_sexp();
                if (!expr) break;
                int before = c->n_locals;
                compile_expr(c, expr, 0);
                if (c->n_locals == before) chunk_emit(c, OP_POP, 0);
                free_node(expr);
            }
            src_ptr = saved_src;
            free(src);
        } else {
            fclose(mf);
        }
    }
    /* If file not found, silently continue (builtins always available) */
#endif
}

/**
 * @brief Compile a `(require module.name)` form: resolves the dotted
 *        module name to a `lib/module/name.esk` source path, and if not
 *        already loaded (tracked in the compiler context to avoid
 *        double-loading) and not the always-available `stdlib` prelude,
 *        reads and compiles that file's top-level forms inline (no-op
 *        under WASM, which has no filesystem access).
 */
static void compile_form_require(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    /* A `(require …)` form is an expression and MUST leave exactly one value on
     * the stack when it adds no top-level binding: the top-level (and function
     * body) compilers emit an OP_POP after any expression that grew no local,
     * so a require that emits nothing would have that POP discard a *live*
     * value — silently shifting every subsequent binding down one slot (the
     * classic "define after (require stdlib) is unbound" corruption).  The
     * no-op fast paths below therefore push an explicit OP_NIL placeholder that
     * the caller's POP balances; the file-loading path adds real bindings, so a
     * placeholder is emitted there only if it happened to add none. */
    int locals_at_start = c->n_locals;
    if (node->n_children >= 2 && node->children[1]->type == N_SYMBOL) {
        /* A `require` may also name a library this unit defines: the two
         * module styles share one namespace, so `(require m)` after
         * `(define-library (m) …)` must not go to disk either. */
        if (!vm_unit_library_defined(node->children[1]->symbol)) {
            vm_compile_module_by_name(c, node->children[1]->symbol);
        }
    }
    /* Balance the caller's POP when the require added no binding of its own
     * (empty/absent module, or a malformed require). */
    if (c->n_locals == locals_at_start) chunk_emit(c, OP_NIL, 0);
    return;
}

/** @brief Emits `(define <alias> <target>)` through the ordinary define path. */
static void vm_emit_import_alias(FuncChunk* c, const char* alias, const char* target) {
    if (!alias || !*alias || !target || !*target || strcmp(alias, target) == 0) return;
    Node* def = make_call_node("define");
    if (!def) return;
    Node* lhs = make_symbol_node(alias);
    Node* rhs = make_symbol_node(target);
    if (!lhs || !rhs) { free_node(lhs); free_node(rhs); free_node(def); return; }
    add_child(def, lhs);
    add_child(def, rhs);
    int before = c->n_locals;
    compile_expr(c, def, 0);
    if (c->n_locals == before) chunk_emit(c, OP_POP, 0);
    free_node(def);
}

/**
 * @brief Resolves one R7RS import set of a SAME-UNIT library, emitting the
 *        aliases it asks for and reporting the names it makes visible.
 *
 * The VM has one global top level and therefore no visibility boundary, so
 * `only` and `except` need no code — they only narrow which names a `prefix`
 * further out is applied to.  `rename` and `prefix` DO need code: they are
 * the import sets that introduce new names, which the native front end also
 * lowers to plain `define` aliases.
 *
 * @return the number of names @p visible received, or -1 when the set does
 *   not bottom out in a library this compilation unit defines (in which case
 *   nothing was emitted and the caller falls back to the module search path).
 */
static int vm_resolve_unit_import_set(FuncChunk* c, const Node* set,
                                      char visible[][128], int max_visible) {
    if (!set || set->type != N_LIST || set->n_children < 1) return -1;
    const Node* head = set->children[0];

    if (head->type == N_SYMBOL && set->n_children >= 2 &&
        (strcmp(head->symbol, "only") == 0 || strcmp(head->symbol, "except") == 0 ||
         strcmp(head->symbol, "prefix") == 0 || strcmp(head->symbol, "rename") == 0)) {
        int n = vm_resolve_unit_import_set(c, set->children[1], visible, max_visible);
        if (n < 0) return -1;

        if (strcmp(head->symbol, "only") == 0) {
            int kept = 0;
            for (int i = 0; i < n; i++) {
                for (int a = 2; a < set->n_children; a++) {
                    if (set->children[a]->type == N_SYMBOL &&
                        strcmp(set->children[a]->symbol, visible[i]) == 0) {
                        if (kept != i) strncpy(visible[kept], visible[i], 127);
                        kept++;
                        break;
                    }
                }
            }
            return kept;
        }
        if (strcmp(head->symbol, "except") == 0) {
            int kept = 0;
            for (int i = 0; i < n; i++) {
                int excluded = 0;
                for (int a = 2; a < set->n_children; a++) {
                    if (set->children[a]->type == N_SYMBOL &&
                        strcmp(set->children[a]->symbol, visible[i]) == 0) { excluded = 1; break; }
                }
                if (excluded) continue;
                if (kept != i) strncpy(visible[kept], visible[i], 127);
                kept++;
            }
            return kept;
        }
        if (strcmp(head->symbol, "rename") == 0) {
            for (int a = 2; a < set->n_children; a++) {
                const Node* pair = set->children[a];
                if (pair->type != N_LIST || pair->n_children != 2) continue;
                if (pair->children[0]->type != N_SYMBOL ||
                    pair->children[1]->type != N_SYMBOL) continue;
                const char* from = pair->children[0]->symbol;
                const char* to = pair->children[1]->symbol;
                for (int i = 0; i < n; i++) {
                    if (strcmp(visible[i], from) != 0) continue;
                    vm_emit_import_alias(c, to, from);
                    strncpy(visible[i], to, 127);
                    visible[i][127] = '\0';
                    break;
                }
            }
            return n;
        }
        /* prefix: the last element is the prefix symbol */
        const Node* prefix = set->children[set->n_children - 1];
        if (prefix->type != N_SYMBOL) return n;
        for (int i = 0; i < n; i++) {
            char alias[128];
            snprintf(alias, sizeof(alias), "%s%s", prefix->symbol, visible[i]);
            vm_emit_import_alias(c, alias, visible[i]);
            strncpy(visible[i], alias, 127);
            visible[i][127] = '\0';
        }
        return n;
    }

    /* Base case: a library name datum. */
    char name[256];
    if (!vm_library_name_from_datum(set, name, sizeof(name))) return -1;
    int idx = vm_unit_library_index(name);
    if (idx < 0) return -1;
    int n = g_compiler_ctx.unit_libraries[idx].n_exports;
    if (n > max_visible) n = max_visible;
    for (int i = 0; i < n; i++) {
        strncpy(visible[i], g_compiler_ctx.unit_libraries[idx].exports[i], 127);
        visible[i][127] = '\0';
    }
    return n;
}

/**
 * @brief Compile an R7RS `(import <import-set> …)` form.
 *
 * Resolution order matches the native front end (lib/frontend/library_registry.h):
 *   1. libraries this compilation unit defined earlier with `define-library`,
 *   2. the module search path.
 * A same-unit library's bindings are already top-level globals of this unit —
 * `define-library` spliced its body in — so importing one loads nothing; only
 * the aliases an import set asks for are emitted.
 */
static void compile_form_import(FuncChunk* c, Node* node, int tail) {
    (void)tail;
    int locals_at_start = c->n_locals;
    for (int i = 1; i < node->n_children; i++) {
        char visible[64][128];
        if (vm_resolve_unit_import_set(c, node->children[i], visible, 64) >= 0) continue;
        const Node* lib = vm_import_set_library_datum(node->children[i]);
        char name[256];
        if (!vm_library_name_from_datum(lib, name, sizeof(name))) continue;
        if (vm_library_planned_later(name)) {
            /* The library exists in this unit but is written BELOW the import.
             * R7RS-small 5.6.1 defines a library by its `define-library` form,
             * so nothing is defined until that form has been read. Refusing
             * here is what keeps the VM from running a program both native
             * lanes reject — its body would otherwise still be spliced in
             * below, and the import would look like it had worked. */
            char msg[512];
            snprintf(msg, sizeof(msg),
                     "library '%s' is imported above its define-library form", name);
            vm_compile_error(msg,
                             "A library must be defined before it is imported "
                             "(R7RS-small 5.6.1); move the define-library above "
                             "the import, or put the library in its own file.");
            continue;
        }
        vm_compile_module_by_name(c, name);
    }
    if (c->n_locals == locals_at_start) chunk_emit(c, OP_NIL, 0);
}

/**
 * @brief Compile an R7RS `(define-library (name …) <library-declaration> …)` form.
 *
 * The library's `begin` bodies are spliced into the unit exactly where the
 * form appears — the VM has one global top level, so that IS the library's
 * effect — and the library name is recorded afterwards, so it becomes
 * importable by the forms that follow and stays unresolvable to the forms
 * above it (R7RS-small 5.6.1).  `export` needs no code for the same reason
 * `provide` needs none: there is no visibility boundary to enforce.  An
 * `import` clause is resolved with the same order a top-level import uses,
 * before the name of the library being defined is registered, so a library
 * cannot import itself into existence.
 */
static void compile_form_define_library(FuncChunk* c, Node* node, int tail) {
    (void)tail;
    int locals_at_start = c->n_locals;

    char library_name[256];
    int named = vm_library_name_from_datum(node->children[1], library_name,
                                           sizeof(library_name));

    char exports[64][128];
    int n_exports = 0;

    for (int i = 2; i < node->n_children; i++) {
        Node* clause = node->children[i];
        if (!clause || clause->type != N_LIST || clause->n_children < 1) continue;
        Node* clause_head = clause->children[0];
        if (clause_head->type != N_SYMBOL) continue;

        if (strcmp(clause_head->symbol, "export") == 0) {
            /* No code: the VM has no visibility boundary, so an export names
             * a binding that is already a global of this unit.  The list is
             * still recorded — it is the surface a later `(prefix …)` import
             * of this library builds its aliases over, and R7RS allows more
             * than one export clause, whose union is the library's surface. */
            for (int e = 1; e < clause->n_children && n_exports < 64; e++) {
                if (clause->children[e]->type != N_SYMBOL) continue;
                strncpy(exports[n_exports], clause->children[e]->symbol, 127);
                exports[n_exports][127] = '\0';
                n_exports++;
            }
            continue;
        }
        if (strcmp(clause_head->symbol, "import") == 0) {
            /* Resolve, then discard the placeholder compile_form_import()
             * leaves for a caller that would POP it — this clause is not an
             * expression position. */
            int before = c->n_locals;
            compile_form_import(c, clause, 0);
            if (c->n_locals == before) chunk_emit(c, OP_POP, 0);
            continue;
        }
        if (strcmp(clause_head->symbol, "begin") == 0) {
            /* Same stack discipline the top-level driver applies: a body form
             * that binds nothing leaves one value that has to be dropped, and
             * one that binds occupies its slot and must not be. */
            for (int b = 1; b < clause->n_children; b++) {
                int before = c->n_locals;
                compile_expr(c, clause->children[b], 0);
                if (c->n_locals == before) chunk_emit(c, OP_POP, 0);
            }
            continue;
        }
        if (strcmp(clause_head->symbol, "include") == 0 ||
            strcmp(clause_head->symbol, "include-ci") == 0) {
            /* `include` is already a VM form (it splices a file's forms into
             * the current unit); a library declaration means exactly that. */
            int before = c->n_locals;
            compile_expr(c, clause, 0);
            if (c->n_locals == before) chunk_emit(c, OP_POP, 0);
            continue;
        }
        /* Any other declaration (cond-expand, include-library-declarations)
         * is not part of the supported subset; ignore it rather than emit
         * bytecode whose meaning we cannot justify. */
    }

    if (named) {
        vm_unit_library_define(library_name, exports, n_exports);
        vm_drop_planned_library(library_name);
    }

    if (c->n_locals == locals_at_start) chunk_emit(c, OP_NIL, 0);
}

/**
 * @brief Compile a `(define-record-type name (constructor field...) pred?
 *        (field accessor [mutator])...)` special form. Records are
 *        represented as tagged vectors (element 0 = the type name string,
 *        packed the same way compile_symbol_literal() does; elements
 *        1..N = field values). Compiles a small standalone closure for
 *        the constructor, the predicate (currently a simplified `vector?`
 *        check rather than a full type-tag comparison), and each field's
 *        accessor/mutator, inlining each closure's bytecode into @p c's
 *        chunk (remapping its local constant-pool indices and internal
 *        jump targets) and binding it as a local.
 */
static void compile_form_define_record_type(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    const char* type_name = node->children[1]->symbol;
    (void)type_name; /* used conceptually as type tag */
    Node* ctor = node->children[2]; /* (constructor f1 f2 ...) */
    const char* pred_name = node->children[3]->symbol;

    /* --- Constructor --- */
    if (ctor->type == N_LIST && ctor->n_children >= 1) {
        const char* ctor_name = ctor->children[0]->symbol;
        int n_fields = ctor->n_children - 1;

        /* Compile constructor as a closure that creates a tagged vector */
        FuncChunk func; chunk_init_arrays(&func);
        func.enclosing = c;
        func.param_count = n_fields;
        for (int i = 0; i < n_fields; i++)
            add_local(&func, ctor->children[i + 1]->symbol);

        /* Body: push type tag (as symbol), then all fields, create vector */
        /* Use type_name as a string constant for the tag */
        int len = (int)strlen(node->children[1]->symbol);
        int n_packs = (len + 7) / 8;
        chunk_emit(&func, OP_CONST, chunk_add_const(&func, INT_VAL(len)));
        for (int p = 0; p < n_packs; p++) {
            int64_t pack = 0;
            for (int b = 0; b < 8 && p * 8 + b < len; b++) {
                pack |= ((int64_t)(unsigned char)node->children[1]->symbol[p * 8 + b]) << (b * 8);
            }
            chunk_emit(&func, OP_CONST, chunk_add_const(&func, INT_VAL(pack)));
        }
        chunk_emit(&func, OP_NATIVE_CALL, 100); /* build-string-from-packed */
        for (int i = 0; i < n_fields; i++)
            chunk_emit(&func, OP_GET_LOCAL, i);
        chunk_emit(&func, OP_VEC_CREATE, n_fields + 1); /* +1 for type tag */
        chunk_emit(&func, OP_RETURN, 0);

        /* Inline func body into parent chunk */
        int cfunc = chunk_add_const(c, INT_VAL(0));
        int jover = placeholder(c);
        int func_start = c->code_len;
        c->constants[cfunc].as.i = func_start;
        int const_map[4096];
        for (int i = 0; i < func.n_constants; i++)
            const_map[i] = chunk_add_const(c, func.constants[i]);
        for (int i = 0; i < func.code_len; i++) {
            Instr fi = func.code[i];
            if (fi.op == OP_CONST) fi.operand = const_map[fi.operand];
            if (fi.op == OP_JUMP || fi.op == OP_JUMP_IF_FALSE || fi.op == OP_LOOP || fi.op == OP_PUSH_HANDLER)
                fi.operand += func_start;
            chunk_emit_instr(c, fi);
        }
        patch(c, jover, OP_JUMP, c->code_len);
        chunk_emit(c, OP_CLOSURE, cfunc);
        add_local(c, ctor_name);
        chunk_free_arrays(&func);
    }

    /* --- Predicate --- */
    {
        FuncChunk func; chunk_init_arrays(&func);
        func.enclosing = c;
        func.param_count = 1;
        add_local(&func, "v");
        /* Check: (and (vector? v) (> (vector-length v) 0) (equal? (vector-ref v 0) type-name)) */
        chunk_emit(&func, OP_GET_LOCAL, 0);
        chunk_emit(&func, OP_VEC_P, 0);
        chunk_emit(&func, OP_RETURN, 0); /* simplified: just vector? check */

        int cfunc = chunk_add_const(c, INT_VAL(0));
        int jover = placeholder(c);
        int func_start = c->code_len;
        c->constants[cfunc].as.i = func_start;
        int const_map[4096];
        for (int i = 0; i < func.n_constants; i++)
            const_map[i] = chunk_add_const(c, func.constants[i]);
        for (int i = 0; i < func.code_len; i++) {
            Instr fi = func.code[i];
            if (fi.op == OP_CONST) fi.operand = const_map[fi.operand];
            chunk_emit_instr(c, fi);
        }
        patch(c, jover, OP_JUMP, c->code_len);
        chunk_emit(c, OP_CLOSURE, cfunc);
        add_local(c, pred_name);
        chunk_free_arrays(&func);
    }

    /* --- Accessors (and optional mutators) --- */
    for (int i = 4; i < node->n_children; i++) {
        Node* field_spec = node->children[i];
        if (field_spec->type != N_LIST || field_spec->n_children < 2) continue;
        int field_idx = i - 4 + 1; /* +1 because index 0 is the type tag */

        /* Accessor */
        {
            const char* acc_name = field_spec->children[1]->symbol;
            FuncChunk func; chunk_init_arrays(&func);
            func.enclosing = c;
            func.param_count = 1;
            add_local(&func, "v");
            chunk_emit(&func, OP_GET_LOCAL, 0);
            chunk_emit(&func, OP_CONST, chunk_add_const(&func, INT_VAL(field_idx)));
            chunk_emit(&func, OP_VEC_REF, 0);
            chunk_emit(&func, OP_RETURN, 0);

            int cfunc = chunk_add_const(c, INT_VAL(0));
            int jover = placeholder(c);
            int func_start = c->code_len;
            c->constants[cfunc].as.i = func_start;
            int const_map[4096];
            for (int i2 = 0; i2 < func.n_constants; i2++)
                const_map[i2] = chunk_add_const(c, func.constants[i2]);
            for (int i2 = 0; i2 < func.code_len; i2++) {
                Instr fi = func.code[i2];
                if (fi.op == OP_CONST) fi.operand = const_map[fi.operand];
                chunk_emit_instr(c, fi);
            }
            patch(c, jover, OP_JUMP, c->code_len);
            chunk_emit(c, OP_CLOSURE, cfunc);
            add_local(c, acc_name);
            chunk_free_arrays(&func);
        }

        /* Mutator (optional, at children[2]) */
        if (field_spec->n_children >= 3) {
            const char* mut_name = field_spec->children[2]->symbol;
            FuncChunk func; chunk_init_arrays(&func);
            func.enclosing = c;
            func.param_count = 2;
            add_local(&func, "v");
            add_local(&func, "val");
            chunk_emit(&func, OP_GET_LOCAL, 0);   /* vector */
            chunk_emit(&func, OP_CONST, chunk_add_const(&func, INT_VAL(field_idx)));
            chunk_emit(&func, OP_GET_LOCAL, 1);   /* new value */
            chunk_emit(&func, OP_VEC_SET, 0);
            chunk_emit(&func, OP_RETURN, 0);

            int cfunc = chunk_add_const(c, INT_VAL(0));
            int jover = placeholder(c);
            int func_start = c->code_len;
            c->constants[cfunc].as.i = func_start;
            int const_map[4096];
            for (int i2 = 0; i2 < func.n_constants; i2++)
                const_map[i2] = chunk_add_const(c, func.constants[i2]);
            for (int i2 = 0; i2 < func.code_len; i2++) {
                Instr fi = func.code[i2];
                if (fi.op == OP_CONST) fi.operand = const_map[fi.operand];
                chunk_emit_instr(c, fi);
            }
            patch(c, jover, OP_JUMP, c->code_len);
            chunk_emit(c, OP_CLOSURE, cfunc);
            add_local(c, mut_name);
            chunk_free_arrays(&func);
        }
    }
    return;
}

/** @brief Compile a `(parameterize ((param value)...) body...)` special
 *         form.  Parameter and value expressions are first evaluated into
 *         locals, then every converter is run exactly once before any
 *         dynamic binding is installed.  This mirrors the native lowering:
 *         a converter that raises cannot leave a partially-bound parameter
 *         stack, and cleanup never re-evaluates a parameter expression. */
static void compile_form_parameterize(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head;
    Node* bindings = node->children[1];
    int saved_locals = c->n_locals;
    int parameter_slots[64];
    int raw_value_slots[64];
    int converted_value_slots[64];
    int n_bindings = 0;

    /* Preserve the normal left-to-right evaluation of binding expressions. */
    for (int i = 0; i < bindings->n_children && n_bindings < 64; i++) {
        Node* binding = bindings->children[i];
        if (binding->type != N_LIST || binding->n_children != 2) continue;
        compile_expr(c, binding->children[0], 0);
        parameter_slots[n_bindings] = add_local(c, "__parameterize_parameter__");
        compile_expr(c, binding->children[1], 0);
        raw_value_slots[n_bindings] = add_local(c, "__parameterize_raw_value__");
        n_bindings++;
    }

    /* Converters run after all binding expressions but before the first push.
     * Native 705 is deliberately separate from 702 so it is impossible for
     * a binding to be converted twice. */
    for (int i = 0; i < n_bindings; i++) {
        chunk_emit(c, OP_GET_LOCAL, parameter_slots[i]);
        chunk_emit(c, OP_GET_LOCAL, raw_value_slots[i]);
        chunk_emit(c, OP_NATIVE_CALL, 705); /* parameter-convert */
        converted_value_slots[i] = add_local(c, "__parameterize_value__");
    }

    /* Enter every binding.  Native 702 records a parameter cleanup entry in
     * the VM wind stack, so exceptions and continuation escapes pop in LIFO
     * order before enclosing dynamic-wind after thunks run. */
    for (int i = 0; i < n_bindings; i++) {
        chunk_emit(c, OP_GET_LOCAL, parameter_slots[i]);
        chunk_emit(c, OP_GET_LOCAL, converted_value_slots[i]);
        chunk_emit(c, OP_NATIVE_CALL, 702); /* parameterize-push */
        chunk_emit(c, OP_POP, 0);
    }

    /* A tail call would bypass the mandatory pop sequence, so a bound body
     * is never compiled in tail position. */
    for (int i = 2; i < node->n_children; i++) {
        if (i > 2) chunk_emit(c, OP_POP, 0);
        compile_expr(c, node->children[i],
                     n_bindings == 0 && tail && i == node->n_children - 1);
    }

    /* Pop each stored parameter in reverse order without re-evaluation. */
    for (int i = n_bindings - 1; i >= 0; i--) {
        chunk_emit(c, OP_GET_LOCAL, parameter_slots[i]);
        chunk_emit(c, OP_NATIVE_CALL, 703); /* parameterize-pop */
        chunk_emit(c, OP_POP, 0);
    }

    int n_locals = c->n_locals - saved_locals;
    if (n_locals > 0) chunk_emit(c, OP_POPN, n_locals);
    c->n_locals = saved_locals;
    return;
}

/** @brief Emit an exact/minimum arity check for a values packet in
 *         @p result_slot.  Native 656 accepts `(packet min max)`, where -1
 *         means no upper bound, and returns the packet unchanged. */
static void compile_validate_values_arity(FuncChunk* c, int result_slot,
                                          int min_count, int max_count) {
    chunk_emit(c, OP_GET_LOCAL, result_slot);
    chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(min_count)));
    chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(max_count)));
    chunk_emit(c, OP_NATIVE_CALL, 656);
    chunk_emit(c, OP_POP, 0);
}

/**
 * @brief Bind a producer result already stored in @p result_slot to R7RS
 *        values formals.  A symbol captures all values as a proper list;
 *        a fixed list requires an exact count; dotted formals require at
 *        least the fixed prefix and bind the remainder as a proper list.
 */
static void compile_bind_values_formals(FuncChunk* c, int result_slot,
                                        Node* formals) {
    if (!formals) return;
    if (formals->type == N_SYMBOL) {
        chunk_emit(c, OP_GET_LOCAL, result_slot);
        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
        chunk_emit(c, OP_NATIVE_CALL, 655);
        add_local(c, formals->symbol);
        return;
    }
    if (formals->type != N_LIST) {
        fprintf(stderr, "ERROR: values formals must be a symbol or list\n");
        return;
    }

    int dot = -1;
    for (int i = 0; i < formals->n_children; i++) {
        if (is_sym(formals->children[i], ".")) { dot = i; break; }
    }
    int fixed = dot >= 0 ? dot : formals->n_children;
    Node* rest = NULL;
    if (dot >= 0) {
        if (dot + 2 != formals->n_children ||
            formals->children[dot + 1]->type != N_SYMBOL) {
            fprintf(stderr, "ERROR: malformed dotted values formals\n");
            return;
        }
        rest = formals->children[dot + 1];
    }

    compile_validate_values_arity(c, result_slot, fixed, rest ? -1 : fixed);
    for (int i = 0; i < fixed; i++) {
        if (formals->children[i]->type != N_SYMBOL) {
            fprintf(stderr, "ERROR: values formal must be an identifier\n");
            continue;
        }
        chunk_emit(c, OP_GET_LOCAL, result_slot);
        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(i)));
        chunk_emit(c, OP_NATIVE_CALL, 651);
        add_local(c, formals->children[i]->symbol);
    }
    if (rest) {
        chunk_emit(c, OP_GET_LOCAL, result_slot);
        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(fixed)));
        chunk_emit(c, OP_NATIVE_CALL, 655);
        add_local(c, rest->symbol);
    }
}

/** @brief Compile `(define-values formals producer)` with a single producer
 *         evaluation and full fixed/rest/zero-value semantics. */
static void compile_form_define_values(FuncChunk* c, Node* node) {
    compile_expr(c, node->children[2], 0);
    int result_slot = add_local(c, "__define_values_result__");
    compile_bind_values_formals(c, result_slot, node->children[1]);
}

/**
 * @brief Compile `let-values` or `let*-values`.  `let-values` evaluates every
 *        producer before introducing any binding; `let*-values` introduces
 *        each binding before compiling the next producer.  Both retain the
 *        producer packets as scoped temporaries and remove all temporaries and
 *        bindings with one OP_POPN after the body result is produced.
 */
static void compile_form_let_values(FuncChunk* c, Node* node, int tail,
                                    int sequential) {
    Node* bindings_list = node->children[1];
    int saved_locals = c->n_locals;
    int result_slots[64];
    Node* formals[64];
    int n_bindings = 0;

    for (int b = 0; b < bindings_list->n_children && n_bindings < 64; b++) {
        Node* binding = bindings_list->children[b];
        if (binding->type != N_LIST || binding->n_children != 2) continue;
        compile_expr(c, binding->children[1], 0);
        result_slots[n_bindings] = add_local(c, "__let_values_result__");
        formals[n_bindings] = binding->children[0];
        n_bindings++;
        if (sequential)
            compile_bind_values_formals(c, result_slots[n_bindings - 1],
                                        formals[n_bindings - 1]);
    }
    if (!sequential) {
        for (int i = 0; i < n_bindings; i++)
            compile_bind_values_formals(c, result_slots[i], formals[i]);
    }

    int scoped_locals = c->n_locals - saved_locals;
    for (int i = 2; i < node->n_children; i++) {
        if (i > 2) chunk_emit(c, OP_POP, 0);
        compile_expr(c, node->children[i],
                     scoped_locals == 0 && tail && i == node->n_children - 1);
    }
    if (scoped_locals > 0) chunk_emit(c, OP_POPN, scoped_locals);
    c->n_locals = saved_locals;
}

/**
 * @brief Compile a `(with-exception-handler handler thunk)` special form:
 *        PUSH_HANDLER around a call to the 0-arg @p thunk, POP_HANDLER on
 *        normal exit; on exception, calls @p handler with the exception
 *        (read from the VM's current_exn register via OP_GET_EXN) as a
 *        regular (never tail) call, so the handler keeps its own frame
 *        for upvalue access (e.g. a captured call/cc continuation).
 */
static void compile_form_with_exception_handler(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    int handler_patch = c->code_len;
    chunk_emit(c, OP_PUSH_HANDLER, 0);

    /* Call thunk (0-arg function) */
    compile_expr(c, node->children[2], 0);
    chunk_emit(c, OP_CALL, 0);

    /* Normal exit */
    chunk_emit(c, OP_POP_HANDLER, 0);
    int end_patch = c->code_len;
    chunk_emit(c, OP_JUMP, 0);

    /* Exception handler: exn is in current_exn VM register.
     * Call handler(exn). NEVER tail-call — the handler may need
     * the enclosing frame for upvalue access (e.g., call/cc's k). */
    patch(c, handler_patch, OP_PUSH_HANDLER, c->code_len);
    compile_expr(c, node->children[1], 0); /* push handler closure */
    chunk_emit(c, OP_GET_EXN, 0);           /* push exn from VM register */
    chunk_emit(c, OP_CALL, 1);

    patch(c, end_patch, OP_JUMP, c->code_len);
    return;
}

/**
 * @brief Compile an R7RS `(guard (var clause...) body...)` special form.
 *        The handler is compiled as its own closure taking the exception
 *        value as its sole parameter (own frame, so let/define/nested
 *        exprs inside it get self-consistent local slot numbering); see
 *        the detailed PUSH_HANDLER/POP_HANDLER bytecode layout comment
 *        below. Clauses are tried like `cond`, with a bare `(var ...)`
 *        (no clauses) falling back to compiling just the guard's own tail
 *        expression.
 */
static void compile_form_guard(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    Node* clause_list = node->children[1]; /* (var (test handler) ...) */
    if (clause_list->type != N_LIST || clause_list->n_children < 1) {
        compile_expr(c, node->children[node->n_children - 1], tail);
        return;
    }
    /* CORRECT ARCHITECTURE: the guard handler is compiled as a closure
     * that takes the exception value as its sole parameter. This gives it
     * its own call frame with a known fp, so let/define/nested expressions
     * inside the handler have self-consistent local slot numbering.
     *
     * Compilation:
     *   PUSH_HANDLER handler_addr
     *   <body>
     *   POP_HANDLER
     *   JUMP end
     * handler_addr:
     *   GET_EXN                    ; push exception from VM register
     *   CLOSURE handler_func       ; push handler closure (takes 1 param: exn)
     *   ; swap so stack = [closure, exn] for CALL 1
     *   ; actually: push closure first, then GET_EXN
     *   CALL 1                     ; call handler_closure(exn)
     *   JUMP end
     *
     * handler_func body: (exn is local 0)
     *   compile clause tests and bodies with exn as a normal local parameter
     */
    char* exn_name = clause_list->children[0]->symbol;
    int saved_locals = c->n_locals;

    /* Emit PUSH_HANDLER */
    int handler_patch = c->code_len;
    chunk_emit(c, OP_PUSH_HANDLER, 0);

    /* Compile body expressions */
    for (int i = 2; i < node->n_children; i++) {
        if (i < node->n_children - 1) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_POP, 0); }
        else compile_expr(c, node->children[i], 0);
    }

    /* Normal exit */
    chunk_emit(c, OP_POP_HANDLER, 0);
    int end_patch = c->code_len;
    chunk_emit(c, OP_JUMP, 0);

    /* Compile handler as a closure with exn as parameter 0 */
    FuncChunk handler_func; chunk_init_arrays(&handler_func);
    handler_func.enclosing = c;
    handler_func.param_count = 1;
    add_local(&handler_func, exn_name); /* exn is local 0 */

    /* Compile clauses inside the handler function */
    int hf_end_patches[32]; int hf_n_end = 0;
    for (int ci = 1; ci < clause_list->n_children; ci++) {
        Node* clause = clause_list->children[ci];
        if (clause->type != N_LIST || clause->n_children < 1) continue;
        if (clause->children[0]->type == N_SYMBOL && strcmp(clause->children[0]->symbol, "else") == 0) {
            for (int j = 1; j < clause->n_children; j++) {
                if (j < clause->n_children - 1) { compile_expr(&handler_func, clause->children[j], 0); chunk_emit(&handler_func, OP_POP, 0); }
                else compile_expr(&handler_func, clause->children[j], 1);
            }
            chunk_emit(&handler_func, OP_RETURN, 0);
            break;
        }
        compile_expr(&handler_func, clause->children[0], 0);
        int jnext = handler_func.code_len;
        chunk_emit(&handler_func, OP_JUMP_IF_FALSE, 0);
        for (int j = 1; j < clause->n_children; j++) {
            if (j < clause->n_children - 1) { compile_expr(&handler_func, clause->children[j], 0); chunk_emit(&handler_func, OP_POP, 0); }
            else compile_expr(&handler_func, clause->children[j], 1);
        }
        chunk_emit(&handler_func, OP_RETURN, 0);
        patch(&handler_func, jnext, OP_JUMP_IF_FALSE, handler_func.code_len);
    }
    /* If no clause matched: re-raise */
    chunk_emit(&handler_func, OP_GET_LOCAL, 0); /* push exn */
    chunk_emit(&handler_func, OP_NATIVE_CALL, 130); /* re-raise */
    chunk_emit(&handler_func, OP_RETURN, 0);

    /* Inline handler function code into parent chunk */
    int const_map_h[4096];
    for (int i = 0; i < handler_func.n_constants; i++)
        const_map_h[i] = chunk_add_const(c, handler_func.constants[i]);
    int hfunc_const = chunk_add_const(c, INT_VAL(0)); /* placeholder */

    /* Handler dispatch code: CLOSURE + CALL */
    patch(c, handler_patch, OP_PUSH_HANDLER, c->code_len);
    int hjover = c->code_len;
    chunk_emit(c, OP_JUMP, 0); /* jump over inlined handler body */

    int hfunc_pc = c->code_len;
    c->constants[hfunc_const].as.i = hfunc_pc;

    /* Copy handler function code with remapping */
    for (int i = 0; i < handler_func.code_len; i++) {
        Instr fi = handler_func.code[i];
        if (fi.op == OP_CONST) fi.operand = const_map_h[fi.operand];
        if (fi.op == OP_JUMP || fi.op == OP_JUMP_IF_FALSE || fi.op == OP_LOOP || fi.op == OP_PUSH_HANDLER)
            fi.operand += hfunc_pc;
        if (fi.op == OP_CLOSURE) {
            int ci2 = fi.operand & 0xFFFF;
            int nu2 = (fi.operand >> 16) & 0xFF;
            fi.operand = const_map_h[ci2] | (nu2 << 16);
        }
        chunk_emit_instr(c, fi);
    }

    patch(c, hjover, OP_JUMP, c->code_len);

    /* Emit: push handler closure, push exn, CALL 1 */
    int n_hf_upvals = handler_func.n_upvalues;
    for (int i = 0; i < n_hf_upvals; i++)
        chunk_emit(c, handler_func.upvalues[i].is_local ? OP_GET_LOCAL : OP_GET_UPVALUE,
                   handler_func.upvalues[i].enclosing_slot);
    chunk_emit(c, OP_CLOSURE, hfunc_const | (n_hf_upvals << 16));
    chunk_emit(c, OP_GET_EXN, 0);
    chunk_emit(c, OP_CALL, 1);

    /* end label */
    patch(c, end_patch, OP_JUMP, c->code_len);

    chunk_free_arrays(&handler_func);
    c->n_locals = saved_locals;
    return;
}

/** @brief Compile `(dynamic-wind before thunk after)`: calls @p before,
 *         pushes @p after onto the VM's wind stack (OP_WIND_PUSH) so
 *         non-local exits (continuations, exceptions) still run it, calls
 *         @p thunk, pops the wind entry (OP_WIND_POP), then calls @p after
 *         on the normal-exit path too — leaving @p thunk's result as TOS
 *         after @p after's own (discarded) result is popped. */
static void compile_form_dynamic_wind(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    /* Evaluate `before` once and keep the closure: the wind entry has to
     * record it so a continuation re-entering this extent can run it again
     * (R7RS rerooting). Recording only `after` made re-entry resume the body
     * with its setup undone. */
    compile_expr(c, node->children[1], 0);
    chunk_emit(c, OP_DUP, 0);
    chunk_emit(c, OP_CALL, 0);        /* call before() ... */
    chunk_emit(c, OP_POP, 0);         /* ... and discard its result */

    /* Push [before, after] onto the wind stack */
    compile_expr(c, node->children[3], 0);
    chunk_emit(c, OP_WIND_PUSH, 0);

    /* Call thunk() */
    compile_expr(c, node->children[2], 0);
    chunk_emit(c, OP_CALL, 0);

    /* Pop wind stack */
    chunk_emit(c, OP_WIND_POP, 0);

    /* Call after() (normal exit) */
    compile_expr(c, node->children[3], 0);
    chunk_emit(c, OP_CALL, 0);
    chunk_emit(c, OP_POP, 0);
    /* thunk result is below after result on stack.
     * After POP of after_result, thunk_result is TOS. */
    return;
}

/** @brief Compile `(delay expr)` / `(delay-force expr)`: builds
 *         a zero-argument thunk closure computing @p expr, inlines its
 *         bytecode into @p c, and packages it as a tagged 3-slot promise
 *         `[state thunk cached]`. State 0 is ordinary delay, state 2 is the
 *         iterative delay-force trampoline, and state 1 is memoized. */
static void compile_form_delay(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    int force_through = is_sym(head, "delay-force");
    (void)tail;
    FuncChunk func; chunk_init_arrays(&func);
    func.enclosing = c;
    func.param_count = 0;
    compile_expr(&func, node->children[1], 1);
    chunk_emit(&func, OP_RETURN, 0);

    int cfunc = chunk_add_const(c, INT_VAL(0));
    int jover = placeholder(c);
    int func_start = c->code_len;
    c->constants[cfunc].as.i = func_start;

    int const_map[MAX_CONSTS];
    for (int i = 0; i < func.n_constants; ++i)
        const_map[i] = chunk_add_const(c, func.constants[i]);
    for (int i = 0; i < func.code_len; ++i) {
        if (func.code[i].op == OP_CLOSURE) {
            int child_ci = func.code[i].operand & 0xFFFF;
            c->constants[const_map[child_ci]].as.i += func_start;
        }
    }
    for (int i = 0; i < func.code_len; ++i) {
        Instr instr = func.code[i];
        if (instr.op == OP_CONST) instr.operand = const_map[instr.operand];
        if (instr.op == OP_JUMP || instr.op == OP_JUMP_IF_FALSE ||
            instr.op == OP_LOOP || instr.op == OP_PUSH_HANDLER)
            instr.operand += func_start;
        if (instr.op == OP_CLOSURE) {
            int child_ci = instr.operand & 0xFFFF;
            int child_upvalues = (instr.operand >> 16) & 0xFF;
            instr.operand = const_map[child_ci] | (child_upvalues << 16);
        }
        chunk_emit_instr(c, instr);
    }
    patch(c, jover, OP_JUMP, c->code_len);

    /* Stack: state, captured thunk, cached-null. */
    chunk_emit(c, OP_CONST,
               chunk_add_const(c, INT_VAL(force_through ? 2 : 0)));
    int n_upvalues = func.n_upvalues;
    for (int i = 0; i < n_upvalues; ++i) {
        chunk_emit(c, func.upvalues[i].is_local ? OP_GET_LOCAL : OP_GET_UPVALUE,
                   func.upvalues[i].enclosing_slot);
    }
    chunk_emit(c, OP_CLOSURE, cfunc | (n_upvalues << 16));

    /* Preserve mutable top-level captures and relayed enclosing upvalues in
     * exactly the same way as an ordinary lambda closure. */
    if (c->enclosing == NULL) {
        for (int i = 0; i < n_upvalues; ++i) {
            if (!func.upvalues[i].is_local) continue;
            chunk_emit(c, OP_DUP, 0);
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(i)));
            chunk_emit(c, OP_CONST,
                       chunk_add_const(c, INT_VAL(func.upvalues[i].enclosing_slot)));
            chunk_emit(c, OP_NATIVE_CALL, 151);
            chunk_emit(c, OP_POP, 0);
        }
    } else {
        for (int i = 0; i < n_upvalues; ++i) {
            if (func.upvalues[i].is_local) continue;
            chunk_emit(c, OP_DUP, 0);
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(i)));
            chunk_emit(c, OP_CONST,
                       chunk_add_const(c, INT_VAL(func.upvalues[i].enclosing_slot)));
            chunk_emit(c, OP_NATIVE_CALL, 252);
            chunk_emit(c, OP_POP, 0);
        }
    }

    chunk_emit(c, OP_NIL, 0);
    chunk_emit(c, OP_VEC_CREATE, 3);
    chunk_emit(c, OP_NATIVE_CALL, VM_NATIVE_PROMISE_CREATE);
    chunk_free_arrays(&func);
    return;
}

/** @brief Compile a `(let ((var val)...) body...)` special form:
 *        evaluates each binding's value in the outer scope, boxing it in
 *        a 1-element vector (needs_boxing()) when it's both `set!`-mutated
 *        and captured by a nested lambda, then compiles the body with
 *        those locals in scope. */
static void compile_form_let(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    int saved_locals = c->n_locals;
    c->scope_depth++;

    /* Collect body nodes for scanning */
    Node* body_nodes[64];
    int n_bodies = 0;
    for (int i = 2; i < node->n_children && n_bodies < 64; i++)
        body_nodes[n_bodies++] = node->children[i];

    Node* bindings = node->children[1];
    for (int i = 0; i < bindings->n_children; i++) {
        Node* b = bindings->children[i];
        if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
            const char* vname = b->children[0]->symbol;
            int box = needs_boxing(body_nodes, n_bodies, vname);
            compile_expr(c, b->children[1], 0);
            if (box) {
                /* Wrap value in a 1-element vector (box) */
                chunk_emit(c, OP_VEC_CREATE, 1);
            }
            int slot = add_local(c, vname);
            if (box) {
                /* Mark this local as boxed */
                c->locals[c->n_locals - 1].boxed = 1;
            }
        }
    }
    int n_let_locals = c->n_locals - saved_locals;

    /* Compile body — don't use tail position if locals need cleanup */
    int body_tail = (n_let_locals > 0) ? 0 : tail;
    for (int i = 2; i < node->n_children; i++) {
        if (i < node->n_children - 1) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_POP, 0); }
        else compile_expr(c, node->children[i], body_tail);
    }

    /* Scope cleanup: remove let-bound locals, keep body result. */
    if (n_let_locals > 0) {
        chunk_emit(c, OP_POPN, n_let_locals);
    }
    c->n_locals = saved_locals;
    c->scope_depth--;
    return;
}

/** @brief Compile `(let* ((var val)...) body...)`: unlike compile_form_let(),
 *         each binding's value expression is compiled with the
 *         previously-bound locals already in scope (sequential, not
 *         parallel, binding).
 *
 *         Mutable-capture boxing (the SW-25 family) applies here exactly as
 *         it does to plain `let`, with one difference forced by sequential
 *         binding: a `let*` binding is in scope for every LATER initializer
 *         as well as for the body, so a nested lambda in a later initializer
 *         captures it too. The scan therefore covers the remaining
 *         initializers together with the body. */
static void compile_form_let_star(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    int saved_locals = c->n_locals;
    c->scope_depth++;
    Node* bindings = node->children[1];
    for (int i = 0; i < bindings->n_children; i++) {
        Node* b = bindings->children[i];
        if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
            const char* vname = b->children[0]->symbol;
            Node* scope_nodes[64];
            int n_scope = 0;
            for (int j = i + 1; j < bindings->n_children && n_scope < 64; j++)
                scope_nodes[n_scope++] = bindings->children[j];
            for (int j = 2; j < node->n_children && n_scope < 64; j++)
                scope_nodes[n_scope++] = node->children[j];
            int box = needs_boxing(scope_nodes, n_scope, vname);
            compile_expr(c, b->children[1], 0);
            if (box) chunk_emit(c, OP_VEC_CREATE, 1);
            add_local(c, vname);
            if (box) c->locals[c->n_locals - 1].boxed = 1;
        }
    }
    int n_let_locals = c->n_locals - saved_locals;
    int body_tail = (n_let_locals > 0) ? 0 : tail;
    for (int i = 2; i < node->n_children; i++) {
        if (i < node->n_children - 1) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_POP, 0); }
        else compile_expr(c, node->children[i], body_tail);
    }
    if (n_let_locals > 0) chunk_emit(c, OP_POPN, n_let_locals);
    c->n_locals = saved_locals;
    c->scope_depth--;
    return;
}

/**
 * @brief Compile `(letrec ((var val)...) body...)`: pushes NIL
 *        placeholders and registers all binding names as locals first (so
 *        mutually-recursive references resolve), then compiles and
 *        SET_LOCALs each initializer, then — critically — converts each
 *        newly-bound closure's upvalues from captured-by-value to open
 *        (by-reference, via native call 131 open_upvalues) so its
 *        GET_UPVALUE reads see the other letrec bindings' final values
 *        rather than the placeholder NILs captured at closure-creation
 *        time.
 */
static void compile_form_letrec(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    int saved_locals = c->n_locals;
    c->scope_depth++;
    Node* bindings = node->children[1];
    int n_bindings = 0;

    /* Scope of every letrec binding: all initializers plus the body (the
     * bindings are mutually visible). Used for the SW-25-family mutable
     * capture scan below. */
    Node* scope_nodes[64];
    int n_scope = 0;
    for (int i = 0; i < bindings->n_children && n_scope < 64; i++)
        scope_nodes[n_scope++] = bindings->children[i];
    for (int i = 2; i < node->n_children && n_scope < 64; i++)
        scope_nodes[n_scope++] = node->children[i];

    /* 1. Push placeholders and register names. A binding that is both
     * `set!`-mutated and captured by a nested lambda gets its heap box HERE,
     * before any initializer runs, so every closure created by an initializer
     * or by the body captures the SAME box (the box pointer is already final
     * at capture time; only its contents change). Without the box such a
     * binding was captured by value and an inner `set!` mutated a private
     * copy — the same defect SW-25 records for parameters. */
    for (int i = 0; i < bindings->n_children; i++) {
        Node* b = bindings->children[i];
        if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
            int box = needs_boxing(scope_nodes, n_scope, b->children[0]->symbol);
            chunk_emit(c, OP_NIL, 0);
            if (box) chunk_emit(c, OP_VEC_CREATE, 1);
            add_local(c, b->children[0]->symbol);
            if (box) c->locals[c->n_locals - 1].boxed = 1;
            n_bindings++;
        }
    }
    int n_let_locals = c->n_locals - saved_locals;

    /* 2. Compile each initializer and store it: a plain SET_LOCAL for an
     * ordinary binding, a VEC_SET into the box for a boxed one. */
    for (int i = 0; i < bindings->n_children; i++) {
        Node* b = bindings->children[i];
        if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
            int slot = resolve_local(c, b->children[0]->symbol);
            int boxed = 0;
            for (int li = c->n_locals - 1; li >= 0; li--)
                if (c->locals[li].slot == slot && c->locals[li].boxed) { boxed = 1; break; }
            if (slot >= 0 && boxed) {
                chunk_emit(c, OP_GET_LOCAL, slot);                      /* box    */
                chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));/* index  */
                compile_expr(c, b->children[1], 0);                     /* value  */
                chunk_emit(c, OP_VEC_SET, 0);
                chunk_emit(c, OP_POP, 0);   /* VEC_SET pushes NIL; discard it */
            } else {
                compile_expr(c, b->children[1], 0);
                if (slot >= 0) chunk_emit(c, OP_SET_LOCAL, slot);
            }
        }
    }

    /* 3. Patch closures: convert captured-by-value upvalues to open (by-reference).
     * After SET_LOCAL, each closure is at its stack slot. For each closure,
     * we use NATIVE_CALL 131 to convert its upvalues to open slot references.
     * This way GET_UPVALUE reads the CURRENT stack value (not the captured NIL). */
    for (int i = 0; i < n_bindings; i++) {
        int slot_i = saved_locals + i;
        /* For each upvalue in this closure, set it to open with the
         * enclosing stack slot. The upvalues reference OTHER letrec bindings. */
        chunk_emit(c, OP_GET_LOCAL, slot_i);     /* push closure */
        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(n_bindings)));
        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(saved_locals)));
        chunk_emit(c, OP_NATIVE_CALL, 131);       /* open_upvalues(closure, count, base_slot) */
        chunk_emit(c, OP_POP, 0);                 /* discard result */
    }

    /* Body — if there are locals to clean up, don't compile in tail position
     * (TAIL_CALL would skip the POPN cleanup) */
    int body_tail = (n_let_locals > 0) ? 0 : tail;
    for (int i = 2; i < node->n_children; i++) {
        if (i < node->n_children - 1) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_POP, 0); }
        else compile_expr(c, node->children[i], body_tail);
    }
    if (n_let_locals > 0) chunk_emit(c, OP_POPN, n_let_locals);
    c->n_locals = saved_locals;
    c->scope_depth--;
    return;
}

/** @brief Compile `(letrec* ((var val)...) body...)`: like
 *         compile_form_letrec() (NIL placeholders + SET_LOCAL
 *         initializers) but without the open-upvalue patching step —
 *         letrec*'s sequential (rather than "all closures see final
 *         values") semantics don't need it. Mutable-capture boxing applies
 *         identically (SW-25 family). */
static void compile_form_letrec_star(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    int saved_locals = c->n_locals;
    c->scope_depth++;
    Node* bindings = node->children[1];

    Node* scope_nodes[64];
    int n_scope = 0;
    for (int i = 0; i < bindings->n_children && n_scope < 64; i++)
        scope_nodes[n_scope++] = bindings->children[i];
    for (int i = 2; i < node->n_children && n_scope < 64; i++)
        scope_nodes[n_scope++] = node->children[i];

    for (int i = 0; i < bindings->n_children; i++) {
        Node* b = bindings->children[i];
        if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
            int box = needs_boxing(scope_nodes, n_scope, b->children[0]->symbol);
            chunk_emit(c, OP_NIL, 0);
            if (box) chunk_emit(c, OP_VEC_CREATE, 1);
            add_local(c, b->children[0]->symbol);
            if (box) c->locals[c->n_locals - 1].boxed = 1;
        }
    }
    int n_let_locals = c->n_locals - saved_locals;
    for (int i = 0; i < bindings->n_children; i++) {
        Node* b = bindings->children[i];
        if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
            int slot = resolve_local(c, b->children[0]->symbol);
            int boxed = 0;
            for (int li = c->n_locals - 1; li >= 0; li--)
                if (c->locals[li].slot == slot && c->locals[li].boxed) { boxed = 1; break; }
            if (slot >= 0 && boxed) {
                chunk_emit(c, OP_GET_LOCAL, slot);
                chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
                compile_expr(c, b->children[1], 0);
                chunk_emit(c, OP_VEC_SET, 0);
                chunk_emit(c, OP_POP, 0);
            } else {
                compile_expr(c, b->children[1], 0);
                if (slot >= 0) chunk_emit(c, OP_SET_LOCAL, slot);
            }
        }
    }
    {
        int body_tail = (n_let_locals > 0) ? 0 : tail;
        for (int i = 2; i < node->n_children; i++) {
            if (i < node->n_children - 1) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_POP, 0); }
            else compile_expr(c, node->children[i], body_tail);
        }
    }
    if (n_let_locals > 0) chunk_emit(c, OP_POPN, n_let_locals);
    c->n_locals = saved_locals;
    c->scope_depth--;
    return;
}

/**
 * @brief Compile a top-level or internal `(define name value)` or
 *        `(define (name params... [. rest]) body...)` form. The simple
 *        variable case just compiles the value and binds it as a new
 *        local. The function case reserves the function's local slot
 *        first (so recursive calls can capture it as an upvalue),
 *        compiles the body into a separate FuncChunk (handling a dotted
 *        rest parameter via OP_PACK_REST), then inlines that chunk's code
 *        into @p c and emits an OP_CLOSURE over it.
 */
/** @brief Resolve a formal-parameter node to its bound name, tolerating a
 *         type-annotated `(name : type)` list form (the name is its first
 *         child); a bare symbol is returned verbatim, anything else as "". */
static const char* param_name(Node* p) {
    if (!p) return "";
    if (p->type == N_SYMBOL) return p->symbol;
    if (p->type == N_LIST && p->n_children >= 1 && p->children[0]->type == N_SYMBOL)
        return p->children[0]->symbol;
    return "";
}

/**
 * @brief Heap-box every parameter of the just-opened function chunk @p func
 *        that is both `set!`-mutated and captured by a nested lambda (SW-25).
 *
 * THE DEFECT THIS CLOSES. `compile_form_let()` has always run needs_boxing()
 * over its bindings, so a `let`-bound variable that an inner lambda mutates
 * lives in a shared 1-element vector and every closure sees the same cell.
 * PARAMETERS never got that treatment: compile_form_define(),
 * compile_form_lambda() and compile_form_lambda_2() called add_local() for
 * each parameter and went straight to the body. A parameter was therefore
 * captured BY VALUE — OP_CLOSURE copies the enclosing slot into the closure's
 * upvalue array — so `(define (f n) ((lambda () (set! n (+ n 1)))) n)` mutated
 * the closure's private copy and answered 5 where native answers 6, with no
 * diagnostic.
 *
 * The open-slot relay (native calls 151/252) is NOT an alternative here: it
 * aliases an absolute VM stack slot and is deliberately restricted to the
 * top-level chunk, because a function's frame is destroyed on return and the
 * alias would outlive it (see compile_form_lambda_2()). Boxing is the
 * mechanism that works at every depth, and is what native's mutable-capture
 * lowering does.
 *
 * WHY THE WRAP IS EMITTED, NOT THE BINDING REWRITTEN. A `let` boxes at the
 * binding site because it computes the value itself; a parameter's value is
 * placed in the frame by the CALLER, so the box has to be installed at
 * function entry instead: read the incoming argument, wrap it in a 1-element
 * vector, store it back into the same slot. Every later read/write of that
 * name compiles through the boxed paths in compile_expr()/compile_form_set()
 * because the Local is marked `boxed`, and a nested closure capturing it
 * copies the BOX pointer — which is exactly the sharing that was missing.
 *
 * MUST BE CALLED after all parameters (including a rest parameter and its
 * OP_PACK_REST) are in place and before any body expression is compiled:
 * OP_PACK_REST reads the raw argument window `sp - fp`, which the wrap would
 * otherwise disturb.
 *
 * @param func       The callee chunk whose locals are exactly its parameters.
 * @param body_nodes The function's body expressions.
 * @param n_bodies   Number of entries in @p body_nodes.
 */
static void vm_box_mutable_captured_params(FuncChunk* func, Node* body_nodes[],
                                           int n_bodies) {
    if (n_bodies <= 0) return;
    for (int li = 0; li < func->n_locals; li++) {
        if (func->locals[li].boxed || !func->locals[li].name) continue;
        if (!needs_boxing(body_nodes, n_bodies, func->locals[li].name)) continue;
        int slot = func->locals[li].slot;
        chunk_emit(func, OP_GET_LOCAL, slot);   /* push the incoming argument */
        chunk_emit(func, OP_VEC_CREATE, 1);     /* wrap it in a 1-element box */
        chunk_emit(func, OP_SET_LOCAL, slot);   /* store the box back (pops)   */
        func->locals[li].boxed = 1;
    }
}

/**
 * @brief Collect the body expressions of a function form into @p out for the
 *        capture/mutation scan, returning how many were collected.
 */
static int vm_collect_body_nodes(Node* node, int body_start, Node** out, int max) {
    int n = 0;
    for (int i = body_start; i < node->n_children && n < max; i++)
        out[n++] = node->children[i];
    return n;
}

static void compile_form_define(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    if (node->children[1]->type == N_SYMBOL) {
        /* Simple variable definition */
        int redef_slot = vm_redefinition_target_slot(c, node->children[1]->symbol);
        compile_expr(c, node->children[2], 0);
        if (redef_slot >= 0) {
            /* R7RS §5.3.1: assign to the name's existing location, so every
             * closure that already captured that slot sees the new value.
             * SET_LOCAL consumes the value; push NIL in its place so the
             * caller's "this define added no local, POP its result"
             * bookkeeping still balances. */
            chunk_emit(c, OP_SET_LOCAL, redef_slot);
            chunk_emit(c, OP_NIL, 0);
            return;
        }
        add_local(c, node->children[1]->symbol);
        return;
    }
    if (node->children[1]->type == N_LIST && node->children[1]->n_children >= 1) {
        /* Function definition: (define (name params...) [: rettype] body) */
        Node* sig = node->children[1];
        char* fname = sig->children[0]->symbol;

        /* R7RS §5.3.1: a redefinition reuses the name's existing location
         * rather than binding a new one, so the body below also resolves the
         * name to that slot. */
        int redef_slot = vm_redefinition_target_slot(c, fname);
        int func_slot = redef_slot >= 0 ? redef_slot : add_local(c, fname);

        /* Compile function body into a separate chunk.
         * The body can reference fname via GET_UPVALUE which will be captured
         * from the enclosing scope's func_slot. */
        FuncChunk func; chunk_init_arrays(&func);
        func.enclosing = c;

        /* Check for dot notation in params: (name x y . rest). A bare `.`
         * marks the variadic tail; the R7RS 7.1.1 vertical-line spelling
         * `|.|` is an ordinary parameter NAMED "." and must not trigger
         * this (is_verbatim), matching every other dot-delimiter site. */
        int has_rest = 0, fixed_params = sig->n_children - 1;
        for (int i = 1; i < sig->n_children; i++) {
            if (sig->children[i]->type == N_SYMBOL && !sig->children[i]->is_verbatim &&
                strcmp(sig->children[i]->symbol, ".") == 0) {
                has_rest = 1;
                fixed_params = i - 1;
                break;
            }
        }
        func.param_count = has_rest ? 255 : fixed_params;

        /* Add fixed parameters as locals.  Each parameter may be a bare symbol
         * or a type-annotated (name : type) list — bind the name in both. */
        for (int i = 1; i <= fixed_params; i++)
            add_local(&func, param_name(sig->children[i]));
        /* Add rest parameter if present */
        if (has_rest && fixed_params + 2 < sig->n_children) {
            add_local(&func, param_name(sig->children[fixed_params + 2])); /* name after dot */
            chunk_emit(&func, OP_PACK_REST, fixed_params);
        }

        /* Skip a `: rettype` return-type annotation between the signature and
         * the body (e.g. (define (f x) : real (+ x 1))). */
        int body_start = 2;
        if (node->n_children >= 4 && node->children[2]->type == N_SYMBOL
            && strcmp(node->children[2]->symbol, ":") == 0)
            body_start = 4;

        /* SW-25: a parameter that is `set!`-mutated AND captured by a nested
         * lambda must be shared through a heap box, exactly as a `let` binding
         * is; otherwise the closure mutates a private copy. */
        {
            Node* body_nodes[64];
            int n_bodies = vm_collect_body_nodes(node, body_start, body_nodes, 64);
            vm_box_mutable_captured_params(&func, body_nodes, n_bodies);
        }

        /* Compile body expressions */
        for (int i = body_start; i < node->n_children; i++) {
            int is_last = (i == node->n_children - 1);
            compile_expr(&func, node->children[i], is_last);
            if (!is_last) chunk_emit(&func, OP_POP, 0);
        }
        chunk_emit(&func, OP_RETURN, 0);

        /* Emit function code at end of current chunk, record its PC */
        int func_pc = c->code_len + 2; /* +2 for CLOSURE + NOP below */
        /* Map child constants to parent indices */
        int const_map[4096];
        for (int i = 0; i < func.n_constants; i++) {
            const_map[i] = chunk_add_const(c, func.constants[i]);
        }
        int cfunc = chunk_add_const(c, INT_VAL(0)); /* placeholder for func PC */

        int jover = placeholder(c);
        int actual_func_pc = c->code_len;
        c->constants[cfunc].as.i = VM_PACK_FUNC_ARITY(actual_func_pc, func.param_count);

        /* Adjust nested function PC constants: any constant in the child
         * that was used as a CLOSURE operand contains a PC relative to the
         * child chunk. After inlining, it needs to be offset by actual_func_pc. */
        for (int i = 0; i < func.code_len; i++) {
            if (func.code[i].op == OP_CLOSURE) {
                int ci = func.code[i].operand & 0xFFFF;
                int parent_ci = const_map[ci];
                /* The constant holds a PC relative to child chunk start.
                 * Adjust to be relative to parent chunk start. */
                c->constants[parent_ci].as.i += actual_func_pc;
            }
        }

        /* Copy function body with proper remapping */
        for (int i = 0; i < func.code_len; i++) {
            Instr fi = func.code[i];
            if (fi.op == OP_CONST) fi.operand = const_map[fi.operand];
            if (fi.op == OP_JUMP || fi.op == OP_JUMP_IF_FALSE || fi.op == OP_LOOP || fi.op == OP_PUSH_HANDLER)
                fi.operand += actual_func_pc;
            if (fi.op == OP_CLOSURE) {
                int ci = fi.operand & 0xFFFF;
                int nu = (fi.operand >> 16) & 0xFF;
                fi.operand = const_map[ci] | (nu << 16);
            }
            chunk_emit_instr(c, fi);
        }
        if (c->enclosing == NULL && func.n_upvalues == 0 && strcmp(fname, "main") != 0) {
            chunk_add_entry(c, fname, func.param_count, func.n_locals,
                            func.n_upvalues, actual_func_pc, func.code_len);
        }

        /* Patch jump over function body */
        patch(c, jover, OP_JUMP, c->code_len);

        /* Emit CLOSURE instruction for the defined function.
         * For self-recursion: the closure captures itself from func_slot.
         * We push func_slot's value (currently NIL) as upvalue,
         * then create closure, then patch func_slot to point to the closure. */
        /* Emit upvalue captures for CLOSURE.
         * The function body compiled into `func` may reference:
         *   - Its own name (self-reference for recursion) → upvalue index determined by func.upvalues
         *   - Other enclosing locals (fold, etc.) → also in func.upvalues
         * Push each upvalue value from the enclosing scope, then CLOSURE captures them. */
        int n_upvals = func.n_upvalues;
        int self_uv_idx = -1;

        for (int i = 0; i < n_upvals; i++) {
            if (strcmp(func.upvalues[i].name, fname) == 0) {
                /* Self-reference: push NIL placeholder (will be patched) */
                chunk_emit(c, OP_NIL, 0);
                self_uv_idx = func.upvalues[i].index;
            } else {
                /* Capture from enclosing scope (local or upvalue) */
                chunk_emit(c, func.upvalues[i].is_local ? OP_GET_LOCAL : OP_GET_UPVALUE,
                           func.upvalues[i].enclosing_slot);
            }
        }

        chunk_emit(c, OP_CLOSURE, cfunc | (n_upvals << 16));
        if (self_uv_idx >= 0) {
            chunk_emit(c, OP_CLOSE_UPVALUE, self_uv_idx);  /* patch self-ref */
        }
        /* Convert local upvalues to open (stack slot references)
         * for top-level defines only (where enclosing scope persists forever). */
        if (c->enclosing == NULL) {
            for (int i = 0; i < n_upvals; i++) {
                if (i == self_uv_idx) continue;
                if (!func.upvalues[i].is_local) continue;
                chunk_emit(c, OP_DUP, 0);
                chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(i)));
                chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(func.upvalues[i].enclosing_slot)));
                chunk_emit(c, OP_NATIVE_CALL, 151);
                chunk_emit(c, OP_POP, 0);
                /* Record for group re-patching (mutual recursion) */
                if (g_repatch_func_slots && g_n_repatch < 256) {
                    g_repatch_func_slots[g_n_repatch] = func_slot;
                    g_repatch_uv_indices[g_n_repatch] = i;
                    g_repatch_enc_slots[g_n_repatch] = func.upvalues[i].enclosing_slot;
                    g_n_repatch++;
                }
            }
        }
        if (redef_slot >= 0) {
            /* R7RS §5.3.1: store the new procedure into the name's existing
             * location. Must come after the open-upvalue conversion above,
             * which needs the closure on the stack top. */
            chunk_emit(c, OP_SET_LOCAL, redef_slot);
            chunk_emit(c, OP_NIL, 0);
        }
        chunk_free_arrays(&func);
        return;
    }
}

/**
 * @brief Compile `(set! name value)`: for an unboxed local, a direct
 *        OP_SET_LOCAL; for a boxed local (mutated + captured — see
 *        needs_boxing()), a VEC_SET into its 1-element box vector. When
 *        @p name isn't a local in the current scope, walks the enclosing
 *        FuncChunk chain to find it, threading upvalue registrations
 *        through every intermediate scope (mirroring how a read reference
 *        would resolve it) and emitting OP_SET_UPVALUE or a boxed VEC_SET
 *        through the upvalue. Warns to stderr if the name can't be
 *        resolved anywhere. Always pushes NIL as the (unspecified) result.
 */
static void compile_form_set_bang(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    const char* var_name = node->children[1]->symbol;
    int slot = resolve_local(c, var_name);

    /* Check if the target variable is boxed */
    int is_boxed = 0;
    if (slot >= 0) {
        for (int li = c->n_locals - 1; li >= 0; li--) {
            if (c->locals[li].slot == slot && c->locals[li].boxed) { is_boxed = 1; break; }
        }
    }

    if (slot >= 0 && is_boxed) {
        /* Boxed local: emit GET_LOCAL(box), CONST(0), compile(value), VEC_SET.
         *
         * OP_VEC_SET pops its three operands and PUSHES NIL, but the shared
         * tail below already pushes the NIL that `set!` evaluates to. Without
         * this POP the boxed paths left TWO values where every caller pops
         * one, so each execution stranded one operand-stack value. Inside a
         * frame that is about to RETURN the surplus was absorbed and invisible;
         * in a bytecode loop compiled into the SAME chunk — a `do` body, whose
         * statements are followed by exactly one OP_POP each — it accumulated
         * once per iteration until the operand stack overflowed. */
        chunk_emit(c, OP_GET_LOCAL, slot);  /* push box (vector) */
        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0))); /* index 0 */
        compile_expr(c, node->children[2], 0); /* compile new value */
        chunk_emit(c, OP_VEC_SET, 0);       /* box[0] = value */
        chunk_emit(c, OP_POP, 0);           /* discard VEC_SET's NIL */
    } else if (slot >= 0) {
        /* Unboxed local: direct SET_LOCAL */
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_SET_LOCAL, slot);
    } else {
        /* Try upvalue resolution for outer-scope mutation */
        const char* name = node->children[1]->symbol;
        FuncChunk* chain[32]; int depth = 0;
        for (FuncChunk* p = c; p && depth < 32; p = p->enclosing)
            chain[depth++] = p;
        int found = 0;
        /* Nearest enclosing scope first — see the read path's note. Walking
         * outermost-first made `set!` on a shadowed name assign the TOP-LEVEL
         * binding rather than the nearer one the reference reads. */
        for (int d = 1; d < depth && !found; d++) {
            int enc_slot = resolve_local(chain[d], name);
            if (enc_slot >= 0) {
                /* Check if the source variable is boxed */
                int var_boxed = 0;
                for (int li = chain[d]->n_locals - 1; li >= 0; li--) {
                    if (chain[d]->locals[li].slot == enc_slot && chain[d]->locals[li].boxed) {
                        var_boxed = 1; break;
                    }
                }

                int prev_slot = enc_slot;
                int prev_is_local = 1;
                for (int level = d - 1; level >= 0; level--) {
                    FuncChunk* fc = chain[level];
                    int uv_idx = -1;
                    for (int i = 0; i < fc->n_upvalues; i++) {
                        if (strcmp(fc->upvalues[i].name, name) == 0) {
                            uv_idx = fc->upvalues[i].index; break;
                        }
                    }
                    if (uv_idx < 0 && fc->n_upvalues < MAX_UPVALUES) {
                        uv_idx = fc->n_upvalues;
                        fc->upvalues[fc->n_upvalues].name = strdup(name);
                        fc->upvalues[fc->n_upvalues].enclosing_slot = prev_slot;
                        fc->upvalues[fc->n_upvalues].index = uv_idx;
                        fc->upvalues[fc->n_upvalues].is_local = prev_is_local;
                        fc->upvalues[fc->n_upvalues].boxed = var_boxed;
                        fc->n_upvalues++;
                    } else if (uv_idx < 0) {
                        /* MAX_UPVALUES exhausted: this scope already relays
                         * MAX_UPVALUES distinct free variables through to its
                         * closures and cannot add `name`. Silently continuing
                         * left uv_idx (and every later use of it as an
                         * enclosing_slot/operand) at -1, which is exactly the
                         * fixed-limit-corrupts-silently shape this rule
                         * exists to prevent — fail the compile instead. */
                        char msg[256];
                        snprintf(msg, sizeof(msg),
                                 "closure exceeds the %d-upvalue capture limit (variable '%s')",
                                 MAX_UPVALUES, name);
                        vm_compile_error(msg,
                                         "a single lexical scope may capture at most "
                                         "MAX_UPVALUES distinct free variables into its "
                                         "nested closures; split the procedure into "
                                         "smaller ones so fewer are captured together.");
                    }
                    prev_slot = uv_idx;
                    prev_is_local = 0;
                }
                int final_uv = -1;
                for (int i = 0; i < c->n_upvalues; i++) {
                    if (strcmp(c->upvalues[i].name, name) == 0) {
                        final_uv = c->upvalues[i].index; break;
                    }
                }
                if (final_uv >= 0) {
                    if (var_boxed) {
                        /* Boxed upvalue: GET_UPVALUE(box), CONST 0, value,
                         * VEC_SET — then discard VEC_SET's NIL, for the same
                         * reason as the boxed-local path above. */
                        chunk_emit(c, OP_GET_UPVALUE, final_uv);
                        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
                        compile_expr(c, node->children[2], 0);
                        chunk_emit(c, OP_VEC_SET, 0);
                        chunk_emit(c, OP_POP, 0);
                    } else {
                        compile_expr(c, node->children[2], 0);
                        chunk_emit(c, OP_SET_UPVALUE, final_uv);
                    }
                    found = 1;
                }
            }
        }
        if (!found) fprintf(stderr, "WARNING: set! on undefined variable '%s'\n", name);
    }
    /* set! returns void — push NIL */
    chunk_emit(c, OP_NIL, 0);
    return;
}

/**
 * @brief Compile `(do ((var init [step])...) (test result...) body...)`.
 *
 * Emits: bind the loop variables, then loop — evaluate the exit test; if true,
 * evaluate the result expressions (last one in tail position when the scope
 * needs no cleanup) and leave the loop; if false, run the body, evaluate ALL
 * step expressions before storing any of them (so each step sees the pre-step
 * values of the others — R7RS 4.2.4 parallel update), then loop back.
 *
 * THREE DEFECTS THIS REPLACES.
 *
 *  1. THE LOOP VARIABLES WERE NEVER POPPED (LE-10, and a silent-wrong case
 *     wider than the one that entry was filed for). `compile_form_let()` ends
 *     with `OP_POPN n_let_locals`; this function ended by decrementing
 *     `c->n_locals` — a COMPILE-TIME bookkeeping change with no runtime
 *     effect — so every `do` form stranded one operand-stack value per loop
 *     variable. Top-level bindings live in stack slots the compiler hands out
 *     by counting (`add_local`), and the top-level driver emits exactly one
 *     `OP_POP` per expression that grew no local, so the stranded values shift
 *     every later slot. This is the same corruption documented on
 *     compile_form_require() and compile_form_with_region(), reached a third
 *     way. It needed no closure and no capture to bite:
 *
 *         (do ((i 0 (+ i 1))) ((= i 2)) 1)
 *         (define q 42)
 *         (display q)          ; VM 2, native 42 — q read the stranded `i`
 *
 *     Two sequential `do` loops miscompiled the second one the same way
 *     (VM 13, native 40). The loud shapes LE-10 records — "VM heap object
 *     limit reached", "calling non-function", and hangs — are the same
 *     stranding landing on a slot whose junk happens to be a loop counter or
 *     a callee.
 *
 *  2. THE BODY AND STEPS WERE COMPILED TWICE. The function used to emit a
 *     complete first version of the loop, then "discard" it with
 *     `c->code_len = loop_top` and re-emit — leftover scaffolding from a
 *     debugging session, its reasoning still in the source as commentary about
 *     JUMP_IF_FALSE polarity. Rewinding `code_len` un-emits CODE and nothing
 *     else: the discarded pass had already appended to the constant pool,
 *     registered upvalues, written closure PC constants pointing into the
 *     region about to be overwritten, and pushed entries onto the global
 *     forward-reference repatch arrays. Compiling a body for its side effects
 *     and throwing the code away is not a discard; the correct structure is
 *     simply emitted once below.
 *
 *  3. A MUTATED-AND-CAPTURED LOOP VARIABLE WAS CAPTURED BY VALUE (SW-34).
 *     `compile_form_let()` has always run `needs_boxing()` over its bindings;
 *     this function bound its variables with a bare `add_local()`, so
 *     `OP_CLOSURE` copied the variable's VALUE into the closure's upvalue
 *     array and an inner `set!` mutated a private copy:
 *
 *         (define (f n)
 *           (do ((i 0 (+ i 1)) (a 0)) ((= i n) a)
 *               ((lambda () (set! a (+ a 1))))))
 *         (f 5)                ; VM 0, native 5
 *
 *     The same loop at TOP LEVEL was correct, which is the tell: the top-level
 *     chunk has the open-slot relay (native calls 151/252) aliasing an
 *     absolute VM stack slot, and that relay is deliberately restricted to the
 *     root chunk because a function's frame dies on return. Boxing is the
 *     mechanism that works at every depth — the same fix, and the same
 *     `needs_boxing()` predicate, that the `let` binding form already used.
 *
 * A do variable is in scope for the test, the results, the body AND every step
 * expression, so the capture scan covers all four; a variable captured only by
 * a step or a result is boxed just the same.
 */
static void compile_form_do(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head;
    int saved_locals = c->n_locals;
    c->scope_depth++;
    Node* vars = node->children[1];
    Node* test = node->children[2];

    /* Everything a do variable is in scope for, for needs_boxing(). */
    Node* scope_nodes[64];
    int n_scope = 0;
    if (test && test->type == N_LIST)
        for (int i = 0; i < test->n_children && n_scope < 64; i++)
            scope_nodes[n_scope++] = test->children[i];
    for (int i = 3; i < node->n_children && n_scope < 64; i++)
        scope_nodes[n_scope++] = node->children[i];
    for (int i = 0; i < vars->n_children && n_scope < 64; i++) {
        Node* b = vars->children[i];
        if (b->type == N_LIST && b->n_children >= 3) scope_nodes[n_scope++] = b->children[2];
    }

    /* Bind the loop variables, boxing the ones a nested lambda both captures
     * and `set!`s so every closure shares one cell (SW-34). */
    for (int i = 0; i < vars->n_children; i++) {
        Node* b = vars->children[i];
        if (b->type == N_LIST && b->n_children >= 2 && b->children[0]->type == N_SYMBOL) {
            const char* vname = b->children[0]->symbol;
            int box = needs_boxing(scope_nodes, n_scope, vname);
            compile_expr(c, b->children[1], 0);
            if (box) chunk_emit(c, OP_VEC_CREATE, 1);
            add_local(c, vname);
            if (box) c->locals[c->n_locals - 1].boxed = 1;
        }
    }
    int n_do_locals = c->n_locals - saved_locals;

    if (test && test->type == N_LIST && test->n_children >= 1) {
        int loop_top = c->code_len;

        /* Exit test. TRUE means leave the loop, so the body is the FALSE arm. */
        compile_expr(c, test->children[0], 0);
        int jbody = placeholder(c);           /* JUMP_IF_FALSE -> body */

        /* Exit arm: the result sequence. Not in tail position when locals
         * still have to be popped — a TAIL_CALL would skip the OP_POPN. */
        int result_tail = (n_do_locals > 0) ? 0 : tail;
        if (test->n_children >= 2) {
            for (int i = 1; i < test->n_children; i++) {
                int is_last = (i == test->n_children - 1);
                compile_expr(c, test->children[i], is_last ? result_tail : 0);
                if (!is_last) chunk_emit(c, OP_POP, 0);
            }
        } else {
            chunk_emit(c, OP_NIL, 0);
        }
        int jexit = placeholder(c);           /* JUMP -> past body+step */

        /* Body. */
        patch(c, jbody, OP_JUMP_IF_FALSE, c->code_len);
        for (int i = 3; i < node->n_children; i++) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_POP, 0);
        }

        /* Steps: evaluate every step expression BEFORE storing any of them, so
         * each one sees the pre-step values (R7RS parallel update). The values
         * are held in anonymous locals rather than being consumed straight off
         * the stack, because a boxed variable is written with VEC_SET, which
         * needs its box and index UNDER the value — an order a bare push/store
         * pair cannot produce. The placeholder name is unspellable as a Scheme
         * symbol, so resolve_local() can never match it. */
        int n_steps = 0;
        for (int i = 0; i < vars->n_children; i++) {
            Node* b = vars->children[i];
            if (b->type == N_LIST && b->n_children >= 3) {
                compile_expr(c, b->children[2], 0);
                add_local(c, " do step");
                n_steps++;
            }
        }
        int step_base = c->n_locals - n_steps;
        int step_i = 0;
        for (int i = 0; i < vars->n_children; i++) {
            Node* b = vars->children[i];
            if (!(b->type == N_LIST && b->n_children >= 3)) continue;
            int tmp_slot = c->locals[step_base + step_i].slot;
            step_i++;
            int slot = resolve_local(c, b->children[0]->symbol);
            if (slot < 0) continue;
            int boxed = 0;
            for (int li = c->n_locals - 1; li >= 0; li--)
                if (c->locals[li].slot == slot && c->locals[li].boxed) { boxed = 1; break; }
            if (boxed) {
                chunk_emit(c, OP_GET_LOCAL, slot);                       /* box   */
                chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0))); /* index */
                chunk_emit(c, OP_GET_LOCAL, tmp_slot);                   /* value */
                chunk_emit(c, OP_VEC_SET, 0);
                chunk_emit(c, OP_POP, 0);       /* VEC_SET pushes NIL */
            } else {
                chunk_emit(c, OP_GET_LOCAL, tmp_slot);
                chunk_emit(c, OP_SET_LOCAL, slot);
            }
        }
        for (int i = 0; i < n_steps; i++) chunk_emit(c, OP_POP, 0);
        c->n_locals -= n_steps;

        chunk_emit(c, OP_LOOP, loop_top);
        patch(c, jexit, OP_JUMP, c->code_len);
    } else {
        /* Malformed `(do (bindings))` with no test clause: still leave exactly
         * one value, so the caller's balancing OP_POP has something to take. */
        chunk_emit(c, OP_NIL, 0);
    }

    /* Drop the loop variables, keeping the result on top (LE-10). */
    if (n_do_locals > 0) chunk_emit(c, OP_POPN, n_do_locals);
    c->n_locals = saved_locals;
    c->scope_depth--;
    return;
}

/**
 * @brief Compile `(lambda rest-symbol body...)` (the fully-variadic form,
 *        a bare symbol instead of a parameter list): all call arguments
 *        are packed into a single list bound to that symbol (OP_PACK_REST
 *        0 at entry), then compiles the body into a separate FuncChunk,
 *        inlines it into @p c (remapping constant-pool indices, internal
 *        jump targets, and nested closures' PC constants), and emits the
 *        upvalue-capturing OP_CLOSURE.
 */
static void compile_form_lambda(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    /* Variadic: all arguments collected into a single list parameter */
    FuncChunk func; chunk_init_arrays(&func);
    func.enclosing = c;
    func.param_count = 255; /* sentinel: variadic, use PACK_REST at entry */
    add_local(&func, node->children[1]->symbol); /* rest list at local 0 */
    /* Emit PACK_REST 0 at function entry: pack ALL args into list at local 0 */
    chunk_emit(&func, OP_PACK_REST, 0);

    /* SW-25: box the rest parameter when the body both mutates and captures
     * it (must follow OP_PACK_REST, which reads the raw argument window). */
    {
        Node* body_nodes[64];
        int n_bodies = vm_collect_body_nodes(node, 2, body_nodes, 64);
        vm_box_mutable_captured_params(&func, body_nodes, n_bodies);
    }

    for (int i = 2; i < node->n_children; i++) {
        int is_last = (i == node->n_children - 1);
        compile_expr(&func, node->children[i], is_last);
        if (!is_last) chunk_emit(&func, OP_POP, 0);
    }
    chunk_emit(&func, OP_RETURN, 0);

    int cfunc = chunk_add_const(c, INT_VAL(0));
    int jover = placeholder(c);
    int func_start = c->code_len;
    c->constants[cfunc].as.i = func_start;

    int const_map2[MAX_CONSTS];
    for (int i = 0; i < func.n_constants; i++)
        const_map2[i] = chunk_add_const(c, func.constants[i]);
    for (int i = 0; i < func.code_len; i++) {
        if (func.code[i].op == OP_CLOSURE) {
            int ci = func.code[i].operand & 0xFFFF;
            int parent_ci = const_map2[ci];
            c->constants[parent_ci].as.i += func_start;
        }
    }
    for (int i = 0; i < func.code_len; i++) {
        Instr fi = func.code[i];
        if (fi.op == OP_CONST) fi.operand = const_map2[fi.operand];
        if (fi.op == OP_JUMP || fi.op == OP_JUMP_IF_FALSE || fi.op == OP_LOOP || fi.op == OP_PUSH_HANDLER)
            fi.operand += func_start;
        if (fi.op == OP_CLOSURE) {
            int ci = fi.operand & 0xFFFF;
            int nu = (fi.operand >> 16) & 0xFF;
            fi.operand = const_map2[ci] | (nu << 16);
        }
        chunk_emit_instr(c, fi);
    }
    patch(c, jover, OP_JUMP, c->code_len);
    int n_upvals = func.n_upvalues;
    for (int i = 0; i < n_upvals; i++) {
        chunk_emit(c, func.upvalues[i].is_local ? OP_GET_LOCAL : OP_GET_UPVALUE,
                   func.upvalues[i].enclosing_slot);
    }
    chunk_emit(c, OP_CLOSURE, cfunc | (n_upvals << 16));
    /* Same open-slot conversion compile_form_lambda_2() performs — a
     * fully-variadic lambda must see enclosing `set!`s (and have its own
     * `set!`s be visible) through a live reference, not a stale by-value
     * capture. */
    if (c->enclosing == NULL) {
        for (int i = 0; i < n_upvals; i++) {
            if (!func.upvalues[i].is_local) continue;
            chunk_emit(c, OP_DUP, 0);
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(i)));
            chunk_emit(c, OP_CONST,
                       chunk_add_const(c, INT_VAL(func.upvalues[i].enclosing_slot)));
            chunk_emit(c, OP_NATIVE_CALL, 151);
            chunk_emit(c, OP_POP, 0);
        }
    } else {
        for (int i = 0; i < n_upvals; i++) {
            if (func.upvalues[i].is_local) continue;
            chunk_emit(c, OP_DUP, 0);
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(i)));
            chunk_emit(c, OP_CONST,
                       chunk_add_const(c, INT_VAL(func.upvalues[i].enclosing_slot)));
            chunk_emit(c, OP_NATIVE_CALL, 252);
            chunk_emit(c, OP_POP, 0);
        }
    }
    chunk_free_arrays(&func);
    return;
}

/**
 * @brief Compile `(lambda (params... [. rest]) body...)`, the standard
 *        fixed/dotted-parameter-list form (compile_form_lambda() handles
 *        the fully-variadic bare-symbol form). Compiles the body into a
 *        separate FuncChunk, packing any dotted rest parameter via
 *        OP_PACK_REST, inlines it into @p c the same way
 *        compile_form_lambda() does, and — for closures defined at the
 *        top level with locally-captured (is_local) upvalues — additionally
 *        converts each such upvalue to an open (by-reference) slot via
 *        native call 151/252 so later `set!`s on the captured variable are
 *        visible to the closure.
 */
static void compile_form_lambda_2(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    Node* params = node->children[1];
    FuncChunk func; chunk_init_arrays(&func);
    func.enclosing = c;

    /* Check for dot notation: (x y . rest). A bare `.` marks the variadic
     * tail; `|.|` (R7RS 7.1.1 vertical-line spelling) is an ordinary
     * parameter NAMED "." and must not trigger this (is_verbatim). */
    int has_rest = 0;
    int fixed_params = params->n_children;
    for (int i = 0; i < params->n_children; i++) {
        if (params->children[i]->type == N_SYMBOL && !params->children[i]->is_verbatim &&
            strcmp(params->children[i]->symbol, ".") == 0) {
            has_rest = 1;
            fixed_params = i; /* params before the dot */
            break;
        }
    }
    func.param_count = fixed_params;

    for (int i = 0; i < fixed_params; i++)
        add_local(&func, params->children[i]->symbol);
    if (has_rest && fixed_params + 2 <= params->n_children) {
        /* Rest parameter name is after the dot */
        add_local(&func, params->children[fixed_params + 1]->symbol);
        /* At function entry: pack extra args from fp+fixed_params to sp into list */
        chunk_emit(&func, OP_PACK_REST, fixed_params);
        func.param_count = 255; /* sentinel: variadic */
    }

    /* SW-25: box parameters that the body both `set!`s and captures. */
    {
        Node* body_nodes[64];
        int n_bodies = vm_collect_body_nodes(node, 2, body_nodes, 64);
        vm_box_mutable_captured_params(&func, body_nodes, n_bodies);
    }

    for (int i = 2; i < node->n_children; i++) {
        int is_last = (i == node->n_children - 1);
        compile_expr(&func, node->children[i], is_last);
        if (!is_last) chunk_emit(&func, OP_POP, 0);
    }
    chunk_emit(&func, OP_RETURN, 0);

    /* Emit: JUMP over lambda body, then body, then CLOSURE */
    int cfunc = chunk_add_const(c, INT_VAL(0));
    int jover = placeholder(c);
    int func_start = c->code_len;
    c->constants[cfunc].as.i = VM_PACK_FUNC_ARITY(func_start, func.param_count);

    int const_map2[MAX_CONSTS];
    for (int i = 0; i < func.n_constants; i++)
        const_map2[i] = chunk_add_const(c, func.constants[i]);

    /* Adjust nested CLOSURE PC constants */
    for (int i = 0; i < func.code_len; i++) {
        if (func.code[i].op == OP_CLOSURE) {
            int ci = func.code[i].operand & 0xFFFF;
            int parent_ci = const_map2[ci];
            c->constants[parent_ci].as.i += func_start;
        }
    }

    for (int i = 0; i < func.code_len; i++) {
        Instr fi = func.code[i];
        if (fi.op == OP_CONST) fi.operand = const_map2[fi.operand];
        if (fi.op == OP_JUMP || fi.op == OP_JUMP_IF_FALSE || fi.op == OP_LOOP || fi.op == OP_PUSH_HANDLER)
            fi.operand += func_start;
        if (fi.op == OP_CLOSURE) {
            int ci = fi.operand & 0xFFFF;
            int nu = (fi.operand >> 16) & 0xFF;
            fi.operand = const_map2[ci] | (nu << 16);
        }
        chunk_emit_instr(c, fi);
    }
    patch(c, jover, OP_JUMP, c->code_len);

    /* Push upvalue captures from enclosing scope before creating closure */
    int n_upvals = func.n_upvalues;
    for (int i = 0; i < n_upvals; i++) {
        chunk_emit(c, func.upvalues[i].is_local ? OP_GET_LOCAL : OP_GET_UPVALUE,
                   func.upvalues[i].enclosing_slot);
    }
    chunk_emit(c, OP_CLOSURE, cfunc | (n_upvals << 16));
    /* Convert upvalues to open slots for set! mutation visibility.
     * For is_local upvalues at top level: use NATIVE_CALL 151 (direct open slot).
     * For non-local upvalues: use NATIVE_CALL 252 to propagate parent's open slot. */
    if (c->enclosing == NULL) {
        for (int i = 0; i < n_upvals; i++) {
            if (!func.upvalues[i].is_local) continue;
            chunk_emit(c, OP_DUP, 0);
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(i)));
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(func.upvalues[i].enclosing_slot)));
            chunk_emit(c, OP_NATIVE_CALL, 151);
            chunk_emit(c, OP_POP, 0);
        }
    } else {
        /* Inside a function: only propagate open slots from parent.
         * DON'T create new open slots for local captures (the function's
         * stack frame will be destroyed on return, making them invalid). */
        for (int i = 0; i < n_upvals; i++) {
            if (!func.upvalues[i].is_local) {
                /* Captured from parent's upvalue — propagate parent's open slot if any */
                chunk_emit(c, OP_DUP, 0);
                chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(i)));
                chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(func.upvalues[i].enclosing_slot)));
                chunk_emit(c, OP_NATIVE_CALL, 252);
                chunk_emit(c, OP_POP, 0);
            }
        }
    }
    chunk_free_arrays(&func);
    return;
}

/**
 * @brief Is @p n the reader's representation of a quoted symbol, i.e. `'sym`
 *        (equivalently the list `(quote sym)`)?
 *
 * The VM reader lowers `'x` to the two-element list `(quote x)` (vm_parser.c),
 * so this is the shape a with-region region NAME arrives in.
 */
static int is_quoted_symbol(Node* n) {
    return n && n->type == N_LIST && n->n_children == 2 &&
           is_sym(n->children[0], "quote") &&
           n->children[1] && n->children[1]->type == N_SYMBOL;
}

/**
 * @brief Compile `(with-region [spec] body ...)` — the VM lowering of the OALR
 *        lexically-scoped region form.
 *
 * TWO DEFECTS THIS REPLACES. The previous lowering was
 *
 *     for (int i = 1; i < node->n_children; i++)
 *         compile_expr(c, node->children[i], tail && i == last);
 *
 * which got the form wrong in two independent ways:
 *
 *  1. NO `OP_POP` FOR NON-FINAL BODY EXPRESSIONS. Every body expression leaves
 *     its value on the operand stack, so a multi-expression body stranded one
 *     value per non-final expression. This is NOT benign on the VM: top-level
 *     bindings live in stack slots that the compiler hands out by counting
 *     (`add_local`), and the top-level driver emits exactly ONE `OP_POP` per
 *     expression that grew no local (eshkol_vm.c). Strand N extra values and
 *     every subsequent top-level `define` is assigned a slot that is already
 *     occupied by junk — the same slot-shift corruption documented on
 *     compile_form_require() above, in the opposite direction. Lowered exactly
 *     like `begin` now: discard all but the last.
 *
 *  2. THE REGION SPECIFIER WAS COMPILED AS AN EXPRESSION. The documented
 *     surface syntax (docs/reference/runtime/memory-model.md) is
 *
 *         (with-region body ...)              ; anonymous
 *         (with-region 'name body ...)        ; named
 *         (with-region ('name size) body ...) ; named + size hint in bytes
 *
 *     The specifier is DECLARATIVE — it names and sizes the arena. Compiling
 *     it as an expression made `'name` a stray value push (see defect 1), and
 *     made `('name size)` a CALL of the symbol `name` with argument `size`,
 *     i.e. a hard "not a function" runtime error on a documented spelling.
 *     The specifier is now recognised and skipped, as the native front end
 *     does (lib/frontend/parser.cpp, ESHKOL_WITH_REGION_OP).
 *
 * HOW THE BODY IS BRACKETED. Native emits
 * region_create/region_push/eshkol_region_enter around the body and tears it
 * down through the single shared teardown primitive `eshkol_region_unwind_to()`
 * (llvm_codegen.cpp codegenWithRegion), which promotes the body result one
 * region level out before the arena dies. The VM now emits the counterpart of
 * exactly that shape:
 *
 *     OP_CONST <size-hint or 0>
 *     OP_NATIVE_CALL 2213     ; heap_region_push + bracket mark
 *     OP_POP                  ; 2213's unspecified value
 *     <body ...>              ; non-final expressions each followed by OP_POP
 *     OP_NATIVE_CALL 2214     ; vm_region_bracket_unwind_to
 *
 * Two properties of that sequence are load-bearing.
 *
 * The body result is left ON the operand stack across 2214. The Stage-1
 * evacuator (lib/backend/vm_region_evac.c) decides what escapes by marking
 * from the VM root set, and the operand stack is a root — so "the result is
 * promoted" is not a special case in the teardown, it is the ordinary
 * consequence of the result still being reachable. Popping it first would
 * hand the evacuator a value nothing points at.
 *
 * The final body expression is compiled NON-TAIL even when the form itself is
 * in tail position. A tail call would jump away from the frame and skip 2214,
 * leaving the region open forever; native has the same constraint for the same
 * reason (its unwind must run before the arena dies). The cost is one lost
 * tail-call opportunity per region body, and it is paid deliberately.
 *
 * The user-reachable handle surface (`region-open` / `region-close`, native
 * 2210-2212) is NOT part of this: it remains bookkeeping-only on the VM, sharing
 * the native handle protocol with reclaim = 0 (tests/vm_parity/PARITY.tsv).
 * Promoting it to a reclaiming close is Stage-2.
 */
static void compile_form_with_region(FuncChunk* c, Node* node, int tail) {
    /* Recognise the optional region specifier. `'name` and `('name size)` are
     * specifiers; anything else is the first body expression.
     *
     * The native front end distinguishes the reader's `'name` sugar from an
     * explicitly written `(quote name)` (its tokenizer sees TOKEN_QUOTE), and
     * treats only the former as a specifier. The VM reader collapses both to
     * `(quote name)`, so a with-region whose SOLE body expression is literally
     * `(quote name)` is read here as "specifier, empty body" — a degenerate
     * form with no use (its value is a symbol and its body allocates nothing).
     * That one undocumented spelling is therefore the ONE place this form
     * diverges from native (native yields the symbol, the VM the empty-body
     * diagnostic plus `()`); it is filed as a verified divergence in
     * tests/vm_parity/found/with_region_explicit_quote_body_vm.esk and on the
     * op:WITH_REGION row of tests/vm_parity/PARITY.tsv. Every DOCUMENTED
     * spelling agrees on both substrates (corpus/with_region_lowering.esk). */
    (void)tail;   /* see the comment above: a region body is never a tail call */

    int body_start = 1;
    int64_t size_hint = 0;
    Node* spec = node->children[1];
    if (is_quoted_symbol(spec)) {
        body_start = 2;                              /* (with-region 'name …) */
    } else if (spec && spec->type == N_LIST && spec->n_children >= 1 &&
               spec->n_children <= 2 && is_quoted_symbol(spec->children[0])) {
        body_start = 2;                       /* (with-region ('name size) …) */
        /* The size hint is the arena tuning knob documented in
         * docs/reference/runtime/memory-model.md: a region whose whole step
         * fits in one block keeps peak RSS flat instead of walking it upward
         * through geometric block doubling. It is a hint only — nothing about
         * the form's value or effects depends on it. */
        Node* sz = (spec->n_children == 2) ? spec->children[1] : NULL;
        if (sz && sz->type == N_NUMBER) {
            double v = sz->is_int ? (double)sz->ival : sz->numval;
            if (v > 0 && v < 1e12) size_hint = (int64_t)v;
        }
    }

    if (body_start >= node->n_children) {
        /* Native rejects this at parse time ("with-region requires at least
         * one body expression"). Report it the same way and still leave
         * exactly one value on the stack: an expression that emits nothing
         * would have the caller's balancing OP_POP discard a live value. */
        fprintf(stderr, "ERROR: with-region requires at least one body expression\n");
        chunk_emit(c, OP_NIL, 0);
        return;
    }

    int ci = chunk_add_const(c, INT_VAL(size_hint));
    if (ci >= 0) chunk_emit(c, OP_CONST, ci); else chunk_emit(c, OP_NIL, 0);
    chunk_emit(c, OP_NATIVE_CALL, VM_NATIVE_REGION_EVAC_PUSH);
    chunk_emit(c, OP_POP, 0);

    for (int i = body_start; i < node->n_children; i++) {
        if (i < node->n_children - 1) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_POP, 0);
        } else {
            compile_expr(c, node->children[i], 0);
        }
    }

    chunk_emit(c, OP_NATIVE_CALL, VM_NATIVE_REGION_EVAC_POP);
}

/**
 * @brief Core expression compiler: dispatches on Node @p node's type/head
 *        symbol and emits the corresponding bytecode into chunk @p c.
 *
 * Handles macro expansion (checked first), literals (number/string/bool),
 * variable references (resolving local slots, walking the enclosing
 * FuncChunk chain to build Lox-style upvalue-capture relay chains for
 * outer-scope variables, unboxing `set!`-mutated+captured locals/upvalues
 * read through their 1-element box vector, and the special guard
 * exception-variable slot -99 -> OP_GET_EXN), function application (with
 * tail calls when @p tail is set), and the built-in primitive operators
 * (arithmetic, comparisons, cons/car/cdr, vector/string ops, call/cc,
 * etc.). Most substantial special forms (cond/case/let family/define/
 * set!/do/lambda/guard/dynamic-wind/delay/parameterize/let-values/
 * with-exception-handler/define-record-type/require/with-region) are
 * delegated to the
 * dedicated compile_form_*() functions above; `if` and the arithmetic/
 * comparison primitives remain compiled inline here.
 * @p tail indicates whether @p node is in tail position, enabling
 * OP_TAIL_CALL instead of OP_CALL for the final call in a function body.
 */
static void compile_expr_impl(FuncChunk* c, Node* node, int tail) {
    if (!node) return;

    /* Check for macro expansion — must come before all other dispatch */
    if (node->type == N_LIST && node->n_children > 0 &&
        node->children[0]->type == N_SYMBOL) {
        VmMacro* macro = vm_macro_lookup(node->children[0]->symbol);
        if (macro) {
            MacroNode* expanded = vm_macro_expand((const MacroNode*)node);
            if (expanded && expanded != (MacroNode*)node) {
                compile_expr(c, (Node*)expanded, tail);
                /* Note: expanded node leaked — acceptable for compiler lifetime */
                return;
            }
        }
    }

    if (node->type == N_NUMBER) {
        double v = node->numval;
        if (node->is_char) {
            /* Character literal (#\x): push the codepoint, then tag it as a
             * VAL_CHAR at runtime so display/char? distinguish it from an int
             * (native 228). Using an int constant + native call keeps the ESKB
             * constant pool format unchanged. */
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL((int64_t)v)));
            chunk_emit(c, OP_NATIVE_CALL, 228);
        } else if (node->is_int)
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(node->ival)));
        else if (!node->is_inexact && v == (int64_t)v && fabs(v) < 1e15)
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL((int64_t)v)));
        else
            chunk_emit(c, OP_CONST, chunk_add_const(c, FLOAT_VAL(v)));
        return;
    }

    if (node->type == N_BOOL) {
        chunk_emit(c, node->numval ? OP_TRUE : OP_FALSE, 0);
        return;
    }

    /* String literal — encode as a constant with embedded string data.
     * We use a special convention: the constant's .as.ptr field stores
     * a negative index into a string table. At runtime, OP_CONST for
     * a string constant allocates it on the heap.
     * Simpler approach: use OP_NATIVE_CALL 56 with string ID. */
    if (node->type == N_STRING) {
        /* String literal → emit packed char data + NATIVE_CALL 100 to build heap string.
         * Pack up to 8 chars per int64 constant, push them, then call build-string. */
        int len = (int)strlen(node->symbol);
        int n_packs = (len + 7) / 8;
        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(len)));
        for (int p = 0; p < n_packs; p++) {
            int64_t pack = 0;
            for (int b = 0; b < 8 && p * 8 + b < len; b++) {
                pack |= ((int64_t)(unsigned char)node->symbol[p * 8 + b]) << (b * 8);
            }
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(pack)));
        }
        chunk_emit(c, OP_NATIVE_CALL, 100); /* build-string-from-packed */
        return;
    }

    if (node->type == N_SYMBOL) {
        if (strcmp(node->symbol, "#t") == 0) { chunk_emit(c, OP_TRUE, 0); return; }
        if (strcmp(node->symbol, "#f") == 0) { chunk_emit(c, OP_FALSE, 0); return; }
        /* Variable lookup: local → enclosing (upvalue) → error */
        int slot = resolve_local(c, node->symbol);
        if (slot == -99) {
            /* Special: guard exception variable → use OP_GET_EXN */
            chunk_emit(c, OP_GET_EXN, 0);
            return;
        }
        if (slot >= 0) {
            chunk_emit(c, OP_GET_LOCAL, slot);
            /* If boxed, unbox: the local holds a vector, read element 0 */
            for (int li = c->n_locals - 1; li >= 0; li--) {
                if (c->locals[li].slot == slot && c->locals[li].boxed) {
                    chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
                    chunk_emit(c, OP_VEC_REF, 0);
                    break;
                }
            }
            return;
        }
        /* Check enclosing scopes for upvalue (walk entire scope chain).
         * If the variable is found N levels up, each intermediate level
         * must also capture it as an upvalue (relay chain).
         * This implements Lox-style upvalue chains. */
        {
            /* Build the chain of FuncChunks from current to root */
            FuncChunk* chain[32];
            int depth = 0;
            for (FuncChunk* p = c; p && depth < 32; p = p->enclosing)
                chain[depth++] = p;

            /* Search from the NEAREST enclosing scope outward. chain[0] is the
             * current chunk and chain[depth-1] is `main`, and the VM binds every
             * top-level define to a stack slot in `main` (see resolve_local's
             * note), so walking outermost-first resolved a free variable to the
             * TOP-LEVEL binding whenever one shared its name — the inverse of
             * lexical scoping. `(define qg 100.0) (define (g qg) ((lambda (t)
             * (* qg t)) 2.0))` returned 200 instead of 3, silently, because the
             * lambda captured main's `qg` rather than g's parameter. Nearest
             * binding wins, as R7RS 4.1.1 requires. */
            for (int d = 1; d < depth; d++) {
                int enc_slot = resolve_local(chain[d], node->symbol);
                if (enc_slot >= 0) {
                    /* Found at level d. Check if it's boxed at the source. */
                    int var_boxed = 0;
                    for (int li = chain[d]->n_locals - 1; li >= 0; li--) {
                        if (chain[d]->locals[li].slot == enc_slot && chain[d]->locals[li].boxed) {
                            var_boxed = 1; break;
                        }
                    }

                    /* Ensure each level from d-1 down to 0 captures this as an upvalue. */
                    int prev_slot = enc_slot;
                    int prev_is_local = 1;

                    for (int level = d - 1; level >= 0; level--) {
                        FuncChunk* fc = chain[level];
                        int uv_idx = -1;
                        for (int i = 0; i < fc->n_upvalues; i++) {
                            if (strcmp(fc->upvalues[i].name, node->symbol) == 0) {
                                uv_idx = fc->upvalues[i].index;
                                break;
                            }
                        }
                        if (uv_idx < 0 && fc->n_upvalues < MAX_UPVALUES) {
                            uv_idx = fc->n_upvalues;
                            fc->upvalues[fc->n_upvalues].name = strdup(node->symbol);
                            fc->upvalues[fc->n_upvalues].enclosing_slot = prev_slot;
                            fc->upvalues[fc->n_upvalues].index = uv_idx;
                            fc->upvalues[fc->n_upvalues].is_local = prev_is_local;
                            fc->upvalues[fc->n_upvalues].boxed = var_boxed;
                            fc->n_upvalues++;
                        } else if (uv_idx < 0) {
                            /* MAX_UPVALUES exhausted — see the identical
                             * branch in compile_form_set_bang() for why this
                             * must fail the compile rather than continue with
                             * uv_idx (and its downstream enclosing_slot uses)
                             * left at -1. */
                            char msg[256];
                            snprintf(msg, sizeof(msg),
                                     "closure exceeds the %d-upvalue capture limit (variable '%s')",
                                     MAX_UPVALUES, node->symbol);
                            vm_compile_error(msg,
                                             "a single lexical scope may capture at most "
                                             "MAX_UPVALUES distinct free variables into its "
                                             "nested closures; split the procedure into "
                                             "smaller ones so fewer are captured together.");
                        }
                        prev_slot = uv_idx;
                        prev_is_local = 0;
                    }

                    /* Emit GET_UPVALUE for the innermost (current) scope */
                    int final_uv = -1;
                    int final_boxed = 0;
                    for (int i = 0; i < c->n_upvalues; i++) {
                        if (strcmp(c->upvalues[i].name, node->symbol) == 0) {
                            final_uv = c->upvalues[i].index;
                            final_boxed = c->upvalues[i].boxed;
                            break;
                        }
                    }
                    if (final_uv >= 0) {
                        chunk_emit(c, OP_GET_UPVALUE, final_uv);
                        /* Unbox if the captured variable is boxed */
                        if (final_boxed) {
                            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
                            chunk_emit(c, OP_VEC_REF, 0);
                        }
                        return;
                    }
                }
            }
        }
        if (node->symbol[0] == '?') {
            compile_symbol_literal(c, node->symbol);
        } else {
            fprintf(stderr, "WARNING: undefined variable '%s'\n", node->symbol);
            chunk_emit(c, OP_NIL, 0);
        }
        return;
    }

    if (node->type != N_LIST || node->n_children == 0) { chunk_emit(c, OP_NIL, 0); return; }

    Node* head = node->children[0];

    /* SW-24 (ESH-0070 class): every fast path below this point dispatches on
     * the head SYMBOL alone, so a user binding that shadows a builtin name —
     * `(define + (lambda (a b) (* a b)))`, `(let ((car ...)) ...)` — was
     * silently bypassed and the opcode/native fast path ran instead
     * ((+ 3 4) printed 7). If the head symbol resolves to a binding USER
     * code created (see vm_head_user_rebound; preamble/prelude root slots
     * below the watermark do not count), compile the form as a plain call
     * through that binding — the same code the generic fallback at the end
     * of this function emits. Macros were already dispatched above, so
     * user-defined syntax is unaffected. */
    if (head->type == N_SYMBOL && head->symbol &&
        vm_head_user_rebound(c, head->symbol)) {
        int argc = node->n_children - 1;
        int saved_locals = c->n_locals;
        compile_expr(c, head, 0);  /* push the (rebound) function */
        add_local(c, "__call_func__");
        for (int i = 1; i < node->n_children; i++) {
            compile_expr(c, node->children[i], 0);
            add_local(c, "__call_arg__");
        }
        if (vm_language_coverage_compilation_enabled()) {
            chunk_emit(c, OP_LANGUAGE_COVERAGE_CALL,
                       (int)vm_language_coverage_name_hash(head->symbol));
        }
        if (tail)
            chunk_emit(c, OP_TAIL_CALL, argc);
        else
            chunk_emit(c, OP_CALL, argc);
        c->n_locals = saved_locals; /* CALL consumed func+args */
        return;
    }

    if (is_sym(head, "gpu-matmul") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 440);
        return;
    }
    if (is_sym(head, "gpu-elementwise") && node->n_children == 4) {
        int native_id = gpu_elementwise_native_id(node->children[1]);
        if (native_id >= 0) {
            compile_expr(c, node->children[2], 0);
            compile_expr(c, node->children[3], 0);
            chunk_emit(c, OP_NATIVE_CALL, native_id);
            return;
        }
    }
    if (is_sym(head, "gpu-reduce") && node->n_children == 3) {
        int native_id = gpu_reduce_native_id(node->children[1]);
        if (native_id >= 0) {
            compile_expr(c, node->children[2], 0);
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(-1)));
            chunk_emit(c, OP_NATIVE_CALL, native_id);
            return;
        }
    }
    if (is_sym(head, "gpu-softmax") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 463);
        return;
    }
    if (is_sym(head, "gpu-transpose") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 416);
        return;
    }
    /* ESH-0226: (reshape tensor d1 d2 ...) — variadic trailing dims (rank >= 2).
     * Mirrors the LLVM special form (tensor_codegen.h: "Reshape tensor:
     * (reshape tensor new-dims...)"); reshape's BUILTINS-table entry is a
     * fixed 2-arg (tensor, shape) closure, so a 3+-arg call like
     * (reshape M 2 2) must have its trailing dims packed into a shape list
     * at compile time instead of going through the generic fixed-arity call
     * path (which would silently drop the extra argument). The 2-arg form
     * — (reshape tensor shape-list-or-scalar) — is untouched and still goes
     * through normal variable lookup. */
    if ((is_sym(head, "reshape") || is_sym(head, "tensor-reshape")) && node->n_children >= 4) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NIL, 0);
        for (int i = node->n_children - 1; i >= 2; i--) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_CONS, 0);
        }
        chunk_emit(c, OP_NATIVE_CALL, 415);
        return;
    }

    /* #322: (arange stop) | (arange start stop) | (arange start stop step).
     * The native handler (case 419) pops exactly (start, stop, step) in that
     * stack order — bottom→top — matching the LLVM path
     * (tensor_creation_codegen.cpp::arange):
     *   (arange n)              -> arange(0, n, 1)      [0, 1, ..., n-1]
     *   (arange start stop)     -> arange(start, stop, 1)
     *   (arange start stop step)
     * arange's BUILTINS-table entry is fixed-arity-1, so the 1- and 3-arg
     * spellings cannot go through the generic builtin closure: that closure
     * loads a single local and then the 3-pop native handler reads two stale
     * stack slots for `start`/`stop`, yielding a bogus 1-element/empty tensor
     * (the reshape/arange rejection in #322 — a malformed arange result is not
     * a valid matmul operand). Emit the defaulted operands explicitly so all
     * three arities are deterministic and match native. */
    if (is_sym(head, "arange") && node->n_children >= 2 && node->n_children <= 4) {
        if (node->n_children == 2) {
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));   /* start */
            compile_expr(c, node->children[1], 0);                     /* stop  */
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(1)));   /* step  */
        } else if (node->n_children == 3) {
            compile_expr(c, node->children[1], 0);                     /* start */
            compile_expr(c, node->children[2], 0);                     /* stop  */
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(1)));   /* step  */
        } else { /* node->n_children == 4 */
            compile_expr(c, node->children[1], 0);                     /* start */
            compile_expr(c, node->children[2], 0);                     /* stop  */
            compile_expr(c, node->children[3], 0);                     /* step  */
        }
        chunk_emit(c, OP_NATIVE_CALL, 419);
        return;
    }

    /* (bytevector-copy bv start [end]) — the R7RS optional-range spellings.
     * bytevector-copy's BUILTINS-table entry is a fixed 1-arg closure, so the
     * range arguments were silently dropped and the call returned a *full*
     * copy where the native codegen raises on an out-of-range range. Emit the
     * dedicated range natives instead (vm_native.c cases 2203/2204), which
     * carry the same bounds contract as the codegen. */
    if (is_sym(head, "bytevector-copy") &&
        (node->n_children == 3 || node->n_children == 4)) {
        compile_expr(c, node->children[1], 0);          /* bv    */
        compile_expr(c, node->children[2], 0);          /* start */
        if (node->n_children == 4) {
            compile_expr(c, node->children[3], 0);      /* end   */
            chunk_emit(c, OP_NATIVE_CALL, 2204);
        } else {
            chunk_emit(c, OP_NATIVE_CALL, 2203);
        }
        return;
    }

    /* (bytevector-copy! to at from start [end]) — same story for the mutating
     * form, whose BUILTINS entry is a fixed 3-arg closure (vm_native.c cases
     * 2205/2206). */
    if (is_sym(head, "bytevector-copy!") &&
        (node->n_children == 5 || node->n_children == 6)) {
        compile_expr(c, node->children[1], 0);          /* to    */
        compile_expr(c, node->children[2], 0);          /* at    */
        compile_expr(c, node->children[3], 0);          /* from  */
        compile_expr(c, node->children[4], 0);          /* start */
        if (node->n_children == 6) {
            compile_expr(c, node->children[5], 0);      /* end   */
            chunk_emit(c, OP_NATIVE_CALL, 2206);
        } else {
            chunk_emit(c, OP_NATIVE_CALL, 2205);
        }
        return;
    }

    /* (tensor a b c ...) — the variadic tensor constructor.
     *
     * `tensor` is a VARIADIC special form natively (leading exact-integer dims
     * then product(dims) values, or a bare list of values, or one rectangular
     * nested collection), but the VM's BUILTINS table aliased the name to
     * make-tensor's fixed 2-arg native, so every call read only its first two
     * arguments: (tensor 2 2 1.0 2.0 3.0 4.0) built #(2 2) and a nested list
     * built (). Pack the arguments into a list (exactly like the tensor-ref
     * special form below) and let native 473 apply the real rule. */
    if (is_sym(head, "tensor") && node->n_children >= 2) {
        chunk_emit(c, OP_NIL, 0);
        for (int i = node->n_children - 1; i >= 1; i--) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_CONS, 0);
        }
        chunk_emit(c, OP_NATIVE_CALL, 473);
        return;
    }

    /* #322: (tensor-ref t i j ...) — multi-dim element read with the indices
     * spelled as separate trailing args (the native idiom, matched by the LLVM
     * path). tensor-ref's BUILTINS-table entry is a fixed 2-arg (tensor, index)
     * closure, so (tensor-ref C 0 1) otherwise loads only the first index and
     * the extra dims are silently dropped — a flat access that returns the
     * wrong element. The native handler (case 411) already accepts a *list*
     * index for multi-dim access, so pack the trailing dims into a shape list
     * at compile time (exactly like the reshape special form). The 2-arg forms
     * — (tensor-ref C flat) and (tensor-ref C (list i j)) — are untouched. */
    if ((is_sym(head, "tensor-ref") || is_sym(head, "tensor-get")) && node->n_children >= 4) {
        compile_expr(c, node->children[1], 0);          /* the tensor */
        chunk_emit(c, OP_NIL, 0);
        for (int i = node->n_children - 1; i >= 2; i--) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_CONS, 0);                   /* -> (i j ...) */
        }
        chunk_emit(c, OP_NATIVE_CALL, 411);
        return;
    }

    /* #322: (tensor-set! t i j ... v) — multi-dim element write with the
     * indices spelled as separate trailing args. tensor-set!'s BUILTINS-table
     * entry is a fixed 3-arg (tensor, index, value) closure, so
     * (tensor-set! A 0 1 v) otherwise binds `value` to the *second index* and
     * drops the real value, silently no-op'ing the multi-dim write. The native
     * handler (case 412) accepts a list index, so pack the middle dims into a
     * shape list and keep the trailing value. Stack order for case 412 is
     * (tensor, index-list, value) bottom→top. The 3-arg forms —
     * (tensor-set! A flat v) and (tensor-set! A (list i j) v) — are untouched. */
    if (is_sym(head, "tensor-set!") && node->n_children >= 5) {
        compile_expr(c, node->children[1], 0);          /* the tensor */
        chunk_emit(c, OP_NIL, 0);
        for (int i = node->n_children - 2; i >= 2; i--) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_CONS, 0);                   /* -> (i j ...) */
        }
        compile_expr(c, node->children[node->n_children - 1], 0);  /* value */
        chunk_emit(c, OP_NATIVE_CALL, 412);
        return;
    }

    /* ── Constant Folding ── */
    /* If all operands are compile-time constants, evaluate at compile time */
    if (node->type == N_LIST && node->n_children >= 3) {
        if (head->type == N_SYMBOL) {
            /* The fold's numeric domain is decided by the literals' EXACTNESS
             * TAGS (parser: is_int = exact int64 literal, is_inexact = written
             * in float syntax), never by the folded value's shape.  Folding
             * `(- 2.0 1.0)` to the exact integer 1 because 1.0 happens to be
             * integral let the following `/` divide exactly and produce 1/3
             * where R7RS 6.2.2 requires 0.3333333333333333. */
            int all_const = 1, all_int = 1, any_inexact = 0;
            for (int i = 1; i < node->n_children; i++) {
                if (node->children[i]->type != N_NUMBER) { all_const = 0; break; }
                if (!node->children[i]->is_int) all_int = 0;
                if (node->children[i]->is_inexact) any_inexact = 1;
            }
            int is_add = strcmp(head->symbol, "+") == 0;
            int is_sub = strcmp(head->symbol, "-") == 0;
            int is_mul = strcmp(head->symbol, "*") == 0;
            /* Exact-integer fold with overflow detection. On overflow, DON'T
             * fold — fall through to the runtime reduce loops, whose ops now
             * promote to bignum (folding to a double here would silently make
             * e.g. (* 9223372036854775807 2) inexact). */
            if (all_const && all_int && (is_add || is_sub || is_mul)) {
                int64_t acc = is_mul ? 1 : 0;
                int overflow = 0;
                if (is_add) {
                    for (int i = 1; i < node->n_children; i++)
                        if (__builtin_add_overflow(acc, node->children[i]->ival, &acc)) { overflow = 1; break; }
                } else if (is_sub) {
                    acc = node->children[1]->ival;
                    for (int i = 2; i < node->n_children; i++)
                        if (__builtin_sub_overflow(acc, node->children[i]->ival, &acc)) { overflow = 1; break; }
                } else { /* is_mul */
                    for (int i = 1; i < node->n_children; i++)
                        if (__builtin_mul_overflow(acc, node->children[i]->ival, &acc)) { overflow = 1; break; }
                }
                if (!overflow) {
                    int ci = chunk_add_const(c, INT_VAL(acc));
                    if (ci >= 0) chunk_emit(c, OP_CONST, ci);
                    return;
                }
                /* overflow → leave to runtime */
            } else if (all_const && any_inexact) {
                double result = 0;
                int folded = 0;
                if (is_add) {
                    result = 0;
                    for (int i = 1; i < node->n_children; i++) result += node->children[i]->numval;
                    folded = 1;
                } else if (is_sub && node->n_children >= 2) {
                    result = node->children[1]->numval;
                    for (int i = 2; i < node->n_children; i++) result -= node->children[i]->numval;
                    folded = 1;
                } else if (is_mul) {
                    result = 1;
                    for (int i = 1; i < node->n_children; i++) result *= node->children[i]->numval;
                    folded = 1;
                }
                /* NOTE: `/` is deliberately NOT folded here. Folding it at
                 * compile time collapsed exact/exact division to an inexact
                 * float ((/ 1 3) -> 0.333…), and there is no exactness flag on
                 * N_NUMBER to tell 6 from 6.0. Leaving division to the runtime
                 * lets it produce an exact rational (or int), or a float, based
                 * on the actual operand types. */
                if (folded) {
                    /* At least one operand was written in inexact syntax, so
                     * the folded constant is INEXACT — unconditionally.  The
                     * old value-shape test emitted INT_VAL whenever the result
                     * was integral, which is how (- 2.0 1.0) reached the
                     * runtime as the exact 1 and made (/ 1 3) exact; it also
                     * flattened a folded -0.0 to the exact 0. */
                    int ci = chunk_add_const(c, FLOAT_VAL(result));
                    if (ci >= 0) chunk_emit(c, OP_CONST, ci);
                    return;
                }
            }
            /* all_const but neither all-exact-int64 nor any-inexact: the
             * operands are exact literals this fold cannot represent (an
             * integer literal wider than int64).  Folding them through a
             * double would silently make an exact result inexact, so leave
             * them to the runtime, whose ops promote to the bignum domain. */
        }
    }

    /* (+ a b ...), (- a b), (* a b ...), (/ a b)
     *
     * The reduce loops keep exactly one accumulator value on the stack while
     * the next operand is compiled. That accumulator must be tracked as an
     * anonymous local so a binding form in a later operand doesn't alias its
     * slot (sibling-let corruption); see compile_operands_tracked(). */
    if (is_sym(head, "+")) {
        int saved = c->n_locals;
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) {
            add_local(c, "__operand__");                    /* accumulator on stack */
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_ADD, 0);
            c->n_locals = saved;                            /* ADD collapsed acc+operand → 1 */
        }
        return;
    }
    if (is_sym(head, "-")) {
        if (node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NEG, 0); return; }
        int saved = c->n_locals;
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) {
            add_local(c, "__operand__");
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_SUB, 0);
            c->n_locals = saved;
        }
        return;
    }
    if (is_sym(head, "*")) {
        int saved = c->n_locals;
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) {
            add_local(c, "__operand__");
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_MUL, 0);
            c->n_locals = saved;
        }
        return;
    }
    if (is_sym(head, "/")) {
        int saved = c->n_locals;
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) {
            add_local(c, "__operand__");
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_DIV, 0);
            c->n_locals = saved;
        }
        return;
    }

    /* Comparisons — push proper booleans */
    if (is_sym(head, "=") && node->n_children == 3) { int s = c->n_locals; compile_operands_tracked(c, node, 1, 2); c->n_locals = s; chunk_emit(c, OP_EQ, 0); return; }
    if (is_sym(head, "<") && node->n_children == 3) { int s = c->n_locals; compile_operands_tracked(c, node, 1, 2); c->n_locals = s; chunk_emit(c, OP_LT, 0); return; }
    if (is_sym(head, ">") && node->n_children == 3) { int s = c->n_locals; compile_operands_tracked(c, node, 1, 2); c->n_locals = s; chunk_emit(c, OP_GT, 0); return; }
    if (is_sym(head, "<=") && node->n_children == 3) { int s = c->n_locals; compile_operands_tracked(c, node, 1, 2); c->n_locals = s; chunk_emit(c, OP_LE, 0); return; }
    if (is_sym(head, ">=") && node->n_children == 3) { int s = c->n_locals; compile_operands_tracked(c, node, 1, 2); c->n_locals = s; chunk_emit(c, OP_GE, 0); return; }
    if (is_sym(head, "not") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NOT, 0); return; }
    if (is_sym(head, "zero?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0))); chunk_emit(c, OP_EQ, 0); return; }
    /* Core type predicates — always available as opcodes (not closures) */
    if (is_sym(head, "null?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NULL_P, 0); return; }
    if (is_sym(head, "pair?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_PAIR_P, 0); return; }
    if (is_sym(head, "number?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NUM_P, 0); return; }
    if (is_sym(head, "string?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_STR_P, 0); return; }
    if (is_sym(head, "boolean?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_BOOL_P, 0); return; }
    if (is_sym(head, "procedure?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_PROC_P, 0); return; }
    if (is_sym(head, "vector?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_VEC_P, 0); return; }

    /* display is a core opcode — always available, not a closure.
     * OP_PRINT pops the value. We push NIL as the return value so
     * the stack accounting is correct in begin/sequence contexts. */
    if (is_sym(head, "display") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_PRINT, 0);
        chunk_emit(c, OP_VOID, 0);  /* push unspecified return value */
        return;
    }
    /* (display value port) — the explicit-port form. Without this, `display`
     * has no 2-argument shape at all: the arity-1 case above only matches
     * n_children==2, so a 2-argument call fell through to the arity-1
     * BUILTINS preamble closure invoked with an extra argument -- an arity
     * mismatch that silently wrote to stdout instead of `port` and left the
     * evaluation stack one slot off, corrupting anything the call result fed
     * into. Native 2226 (_display2) is vm_write_value_port() with
     * write_syntax=0 -- the same routine _write2/618 already uses, just
     * without quotes/bars -- so both forms share one port-writing
     * implementation. Operand order matches _write2: value pushed first,
     * then port, so native pops port then value. */
    if (is_sym(head, "display") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 2226);
        return;
    }
    /* Type predicates that need VM opcodes (not closures — these check types at opcode level) */
    /* SW-31: integer? is NOT number?. Aliasing it to OP_NUM_P made
     * (integer? 5.5) answer #t and (integer? <bignum>) answer #f. It lowers to
     * its own native (1717) which asks whether the value has no fractional
     * part, across the whole tower. */
    if (is_sym(head, "integer?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 1717); return; }

    /* abs and modulo are opcodes, not native calls — keep as special cases.
     * NOTE: remainder must NOT map to OP_MOD. OP_MOD implements R7RS `modulo`
     * (floor semantics, sign of divisor); `remainder` needs truncating
     * semantics (sign of dividend). It resolves through the native table
     * (id 37) like `quotient` (id 38), which computes ia%ib correctly. */
    if (is_sym(head, "abs") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_ABS, 0); return; }
    if (is_sym(head, "modulo") && node->n_children == 3) { int s = c->n_locals; compile_operands_tracked(c, node, 1, 2); c->n_locals = s; chunk_emit(c, OP_MOD, 0); return; }

    /* All other builtins (sin, cos, sqrt, even?, odd?, floor, ceiling, round, expt, min, max,
     * positive?, negative?, number->string, string-append, string=?, newline, length, etc.)
     * are first-class closures defined in the preamble. They resolve via normal variable lookup
     * and are called via the standard CALL mechanism. No special-casing needed. */

    /* R7RS vector constructor / reader literal.  Numeric vectors remain
     * vectors; tensors have their own `(tensor ...)` constructor.  Treating
     * `#(1 2 3)` as a tensor made vector?, vector-length, call-with-values,
     * and every vector consumer silently disagree with the native backend. */
    if (is_sym(head, "vector")) {
        int n_elems = node->n_children - 1;
        /* OP_VEC_CREATE consumes its elements off the operand stack, so the
         * direct form needs n_elems stack slots — which made a literal's member
         * count a function of ESHKOL_VM_STACK_SIZE (a #(...) of a few thousand
         * numbers died with "STACK OVERFLOW" partway through pushing it).
         * A literal's size must be governed by the literal, so past a
         * threshold the vector is allocated once and filled slot by slot at
         * CONSTANT stack depth: the element count is then bounded only by the
         * growable code and constant arrays, not by the operand stack. */
        if (n_elems > VM_VEC_LITERAL_STACK_CHUNK) {
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(n_elems)));
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
            chunk_emit(c, OP_NATIVE_CALL, 218);   /* make-vector(n, fill) */
            int saved_locals = c->n_locals;
            add_local(c, "__vec_literal__");
            for (int i = 1; i < node->n_children; i++) {
                chunk_emit(c, OP_DUP, 0);                                   /* vec */
                chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(i - 1)));/* idx */
                int elem_locals = c->n_locals;
                add_local(c, "__vec_literal_idx__");
                compile_expr(c, node->children[i], 0);                      /* val */
                c->n_locals = elem_locals;
                chunk_emit(c, OP_VEC_SET, 0);   /* pops val, idx, vec; pushes nil */
                chunk_emit(c, OP_POP, 0);       /* drop the nil */
            }
            c->n_locals = saved_locals;
            return;
        }
        for (int i = 1; i < node->n_children; i++) compile_expr(c, node->children[i], 0);
        chunk_emit(c, OP_VEC_CREATE, n_elems);
        return;
    }
    if (is_sym(head, "make-vector") && node->n_children >= 2) {
        /* (make-vector n) or (make-vector n fill) — emit via NATIVE or direct */
        int s = c->n_locals;
        compile_expr(c, node->children[1], 0);
        if (node->n_children >= 3) { add_local(c, "__operand__"); compile_expr(c, node->children[2], 0); c->n_locals = s; }
        else chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
        /* make-vector: n and fill are on stack, dispatch to runtime native */
        chunk_emit(c, OP_NATIVE_CALL, 218);
        return;
    }
    if (is_sym(head, "vector-ref") && node->n_children == 3) { int s = c->n_locals; compile_operands_tracked(c, node, 1, 2); c->n_locals = s; chunk_emit(c, OP_VEC_REF, 0); return; }
    if (is_sym(head, "vector-set!") && node->n_children == 4) { int s = c->n_locals; compile_operands_tracked(c, node, 1, 3); c->n_locals = s; chunk_emit(c, OP_VEC_SET, 0); return; }
    if (is_sym(head, "vector-length") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_VEC_LEN, 0); return; }

    /* Mutation */
    if (is_sym(head, "set-car!") && node->n_children == 3) { int s = c->n_locals; compile_operands_tracked(c, node, 1, 2); c->n_locals = s; chunk_emit(c, OP_SET_CAR, 0); return; }
    if (is_sym(head, "set-cdr!") && node->n_children == 3) { int s = c->n_locals; compile_operands_tracked(c, node, 1, 2); c->n_locals = s; chunk_emit(c, OP_SET_CDR, 0); return; }

    /* String operations via opcodes (these ARE opcodes, not native calls) */
    if (is_sym(head, "string-length") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_STR_LEN, 0);
        return;
    }
    if (is_sym(head, "string-ref") && node->n_children == 3) {
        int s = c->n_locals;
        compile_operands_tracked(c, node, 1, 2);
        c->n_locals = s;
        chunk_emit(c, OP_STR_REF, 0);
        return;
    }
    /* Rational literal: (exact-rational num denom) — generated by parser for 1/3 syntax */
    if (is_sym(head, "exact-rational") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 330);
        return;
    }
    /* substring, string->number, string-upcase, string-downcase, string-copy,
     * string-fill!: all in BUILTINS table — go through normal variable lookup.
     * string-length and string-ref use dedicated opcodes (OP_STR_LEN, OP_STR_REF)
     * above and are kept for performance. */

    /* Compound list accessors: cadr, cdar, cddr, caar */
    if (is_sym(head, "cadr") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0);
        return;
    }
    if (is_sym(head, "cdar") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0);
        return;
    }
    if (is_sym(head, "cddr") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0);
        return;
    }
    if (is_sym(head, "caar") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0);
        return;
    }
    if (is_sym(head, "caddr") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0);
        return;
    }
    /* Remaining 3-level cXXXr forms */
    if (is_sym(head, "caaar") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "caadr") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "cadar") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "cdaar") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0); return; }
    if (is_sym(head, "cdadr") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0); return; }
    if (is_sym(head, "cddar") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); return; }
    if (is_sym(head, "cdddr") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); return; }
    /* 4-level cXXXXr forms (16 total) */
    if (is_sym(head, "caaaar") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "caaadr") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "caadar") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "caaddr") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "cadaar") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "cadadr") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "caddar") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "cadddr") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "cdaaar") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0); return; }
    if (is_sym(head, "cdaadr") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0); return; }
    if (is_sym(head, "cdadar") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0); return; }
    if (is_sym(head, "cdaddr") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0); return; }
    if (is_sym(head, "cddaar") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); return; }
    if (is_sym(head, "cddadr") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); return; }
    if (is_sym(head, "cdddar") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); return; }
    if (is_sym(head, "cddddr") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); return; }
    /* first through fifth */
    if (is_sym(head, "first") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "second") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "third") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "fourth") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "fifth") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); return; }

    /* (cond (test1 expr1) (test2 expr2) ... (else exprN)) */
    if (is_sym(head, "cond") && node->n_children >= 2) { compile_form_cond(c, node, tail); return; }

    /* (case expr ((val ...) body ...) ... (else body ...))
     * Compiles as: evaluate key, then for each clause: DUP key, test each val,
     * if any matches jump to body, else next clause. */
    if (is_sym(head, "case") && node->n_children >= 3) { compile_form_case(c, node, tail); return; }

    /* (when test body...) — one-armed if */
    if (is_sym(head, "when") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0);
        int jf = placeholder(c);
        for (int i = 2; i < node->n_children; i++) {
            compile_expr(c, node->children[i], 0);
            if (i < node->n_children - 1) chunk_emit(c, OP_POP, 0);
        }
        patch(c, jf, OP_JUMP_IF_FALSE, c->code_len);
        return;
    }

    /* (unless test body...) — negated when */
    if (is_sym(head, "unless") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NOT, 0);
        int jf = placeholder(c);
        for (int i = 2; i < node->n_children; i++) {
            compile_expr(c, node->children[i], 0);
            if (i < node->n_children - 1) chunk_emit(c, OP_POP, 0);
        }
        patch(c, jf, OP_JUMP_IF_FALSE, c->code_len);
        return;
    }

    /* (require module.name) — load and compile the module */
    if (is_sym(head, "require")) { compile_form_require(c, node, tail); return; }
    /* (define-library (name …) <declaration> …) — R7RS-small 5.6.1 */
    if (is_sym(head, "define-library") && node->n_children >= 2 &&
        node->children[1]->type == N_LIST) {
        compile_form_define_library(c, node, tail); return;
    }
    /* (import <import-set> …) — resolves against this unit's libraries first */
    if (is_sym(head, "import") && node->n_children >= 2) {
        compile_form_import(c, node, tail); return;
    }
    /* (provide name ...) / (export name ...) — no-op: all symbols are visible.
     * The OP_NIL is not decorative: the top-level and body compilers POP after
     * any form that bound nothing, so returning without pushing would make
     * that POP discard a live value and shift every later binding down a slot
     * (the same corruption compile_form_require() guards against). */
    if (is_sym(head, "provide") || is_sym(head, "export")) {
        chunk_emit(c, OP_NIL, 0);
        return;
    }

    /* (define-syntax name (syntax-rules (literals...) (pattern template) ...))
     *
     * The OP_NIL is not decorative — same discipline as (provide)/(export)
     * directly above.  define-syntax binds no runtime local, and every
     * top-level and body driver POPs after a form that bound nothing, so
     * returning without pushing makes that POP discard a LIVE value and
     * shift every later local down one slot.  That was the real mechanism
     * behind SW-30's VM column: the template's `let` wrote slot k while the
     * body read slot k+1, so (hyg 1) printed 1 instead of 101 and the user's
     * same-named top-level variable read back as (). */
    if (is_sym(head, "define-syntax") && node->n_children >= 3) {
        vm_macro_define_syntax((const MacroNode*)node);
        chunk_emit(c, OP_NIL, 0);
        return;
    }

    /* (define-record-type name (constructor field...) pred (field accessor [mutator]) ...) */
    if (is_sym(head, "define-record-type") && node->n_children >= 4) { compile_form_define_record_type(c, node, tail); return; }

    /* (make-parameter default [converter]) — this is a VM special lowering
     * because the builtin preamble is fixed-arity while R7RS permits the
     * optional converter.  Native 700 applies that converter to the default
     * and keeps it for every later parameterize binding. */
    if (is_sym(head, "make-parameter") &&
        (node->n_children == 2 || node->n_children == 3)) {
        compile_expr(c, node->children[1], 0);
        if (node->n_children == 3) compile_expr(c, node->children[2], 0);
        else chunk_emit(c, OP_NIL, 0);
        chunk_emit(c, OP_NATIVE_CALL, 700);
        return;
    }

    /* (parameterize ((param1 val1) (param2 val2) ...) body ...) */
    if (is_sym(head, "parameterize") && node->n_children >= 3) { compile_form_parameterize(c, node, tail); return; }

    /* (let-values (((x y ...) producer) ...) body ...) */
    if (is_sym(head, "let-values") && node->n_children >= 3) { compile_form_let_values(c, node, tail, 0); return; }
    if (is_sym(head, "let*-values") && node->n_children >= 3) { compile_form_let_values(c, node, tail, 1); return; }

    /* (with-exception-handler handler thunk) — call thunk with handler installed.
     * Uses OP_GET_EXN to access exception from VM register. */
    if (is_sym(head, "with-exception-handler") && node->n_children == 3) { compile_form_with_exception_handler(c, node, tail); return; }

    /* (call/cc proc) or (call-with-current-continuation proc) */
    if ((is_sym(head, "call/cc") || is_sym(head, "call-with-current-continuation")) && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_CALLCC, 0);
        return;
    }

    /* (raise expr) — throw exception */
    if (is_sym(head, "raise") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 130); /* native raise */
        return;
    }

    /* (guard (var (test handler) ...) body ...) — exception handler
     * R7RS: (guard (exn ((test) handler) ...) body ...)
     * Compiled as:
     *   PUSH_HANDLER handler_addr
     *   <body>
     *   POP_HANDLER
     *   JUMP end
     * handler_addr:          ; exception value on TOS
     *   SET_LOCAL exn_slot   ; bind exception to var
     *   <cond-like clause dispatch>
     * end:
     */
    if (is_sym(head, "guard") && node->n_children >= 3) { compile_form_guard(c, node, tail); return; }

    /* (apply f args-list) — call f with list as arguments */
    if (is_sym(head, "apply") && node->n_children == 3) {
        /* Handled via NATIVE_CALL 70 which unpacks the list at runtime */
        compile_expr(c, node->children[1], 0); /* push f */
        compile_expr(c, node->children[2], 0); /* push args list */
        chunk_emit(c, OP_NATIVE_CALL, 70); /* apply: takes f and args-list from stack */
        return;
    }

    /* (values expr1 expr2 ...) — multiple return values.
     * Simplified: pack into a vector. Single value = return as-is. */
    if (is_sym(head, "values") && node->n_children >= 1) {
        if (node->n_children == 2) {
            compile_expr(c, node->children[1], tail);
        } else {
            for (int i = 1; i < node->n_children; i++)
                compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_CONST,
                       chunk_add_const(c, INT_VAL(node->n_children - 1)));
            chunk_emit(c, OP_NATIVE_CALL, 650);
        }
        return;
    }

    /* (call-with-values producer consumer)
     * Call producer(), then unpack its result and call consumer with the values.
     * If result is a vector (from multi-value `values`), unpack it.
     * Otherwise, call consumer with the single result. */
    if (is_sym(head, "call-with-values") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0); /* push producer */
        chunk_emit(c, OP_CALL, 0);              /* call producer() → result */
        compile_expr(c, node->children[2], 0); /* push consumer */
        /* Stack: [result, consumer]. Use apply to unpack. */
        /* Native 251: call-with-values-apply(result, consumer) */
        chunk_emit(c, OP_NATIVE_CALL, 251);
        return;
    }

    /* (dynamic-wind before thunk after)
     * R7RS: call before(), register after on wind stack, call thunk(),
     * pop wind stack, call after(). If a continuation escapes through
     * this dynamic-wind, the after thunk is called during unwinding. */
    if (is_sym(head, "dynamic-wind") && node->n_children == 4) { compile_form_dynamic_wind(c, node, tail); return; }

    /* (delay expr) → create a promise: #(#f <thunk>)
     * The thunk is a nullary closure wrapping expr. */
    if ((is_sym(head, "delay") || is_sym(head, "delay-force")) &&
        node->n_children == 2) {
        compile_form_delay(c, node, tail);
        return;
    }

    /* (force promise) → force a promise (evaluate thunk if not yet forced) */
    if (is_sym(head, "force") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); /* push promise */
        chunk_emit(c, OP_NATIVE_CALL, 132);     /* native force */
        return;
    }

    /* (parallel-execute thunk ...) → build a thunk list for native 624.
     * The builtin table only supports fixed arities; keep this variadic form
     * as a compiler surface while native 624 handles scheduling/fallbacks. */
    if (is_sym(head, "parallel-execute") && node->n_children >= 1) {
        chunk_emit(c, OP_NIL, 0);
        for (int i = node->n_children - 1; i >= 1; i--) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_CONS, 0);
        }
        chunk_emit(c, OP_NATIVE_CALL, 624);
        return;
    }

    /* (make-promise val) / (promise? x) */
    if (is_sym(head, "make-promise") && node->n_children == 2) {
        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(1)));
        chunk_emit(c, OP_NIL, 0);
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_VEC_CREATE, 3);
        chunk_emit(c, OP_NATIVE_CALL, VM_NATIVE_PROMISE_CREATE);
        return;
    }
    if (is_sym(head, "promise?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, VM_NATIVE_PROMISE_P);
        return;
    }

    /* atan, string-append, max, min: variadic wrappers defined in prelude;
     * they go through normal variable lookup — no compiler special-casing needed */

    /* case-lambda: dispatch on argument count */
    if (is_sym(head, "case-lambda") && node->n_children >= 2) {
        /* Transform: (case-lambda ((x) body1) ((x y) body2) ...)
         * → (lambda args (cond ((= (length args) 1) (apply (lambda (x) body1) args))
         *                       ((= (length args) 2) (apply (lambda (x y) body2) args))
         *                       ...))
         * Simplified: compile first matching clause inline */
        /* For now: compile the first clause as a regular lambda */
        Node* clause = node->children[1]; /* first clause */
        if (clause->type == N_LIST && clause->n_children >= 2) {
            Node* params = clause->children[0];
            /* Build a lambda node: (lambda params body...) */
            Node* lam = make_node(N_LIST);
            Node* sym = make_node(N_SYMBOL); strncpy(sym->symbol, "lambda", 127);
            add_child(lam, sym);
            add_child(lam, params);
            for (int i = 1; i < clause->n_children; i++)
                add_child(lam, clause->children[i]);
            compile_expr(c, lam, tail);
            /* Don't free children since they're shared with the original node */
            lam->n_children = 0;
            free(lam->children); lam->children = NULL;
            free(lam); free(sym);
        }
        return;
    }

    /* memq, memv, assv, string-fill!, string-copy, eq?, eqv?, equal?, quotient:
     * all in BUILTINS table — go through normal variable lookup */

    /* Pair operations */
    if (is_sym(head, "cons") && node->n_children == 3) {
        int s = c->n_locals;
        compile_expr(c, node->children[2], 0); /* cdr first (SOS) */
        add_local(c, "__operand__");
        compile_expr(c, node->children[1], 0); /* car second (TOS) */
        c->n_locals = s;
        chunk_emit(c, OP_CONS, 0); return;
    }
    if (is_sym(head, "car") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "cdr") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); return; }
    if (is_sym(head, "list")) {
        /* (list a b c) → cons(a, cons(b, cons(c, nil))) */
        chunk_emit(c, OP_NIL, 0);
        for (int i = node->n_children - 1; i >= 1; i--) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_CONS, 0);
        }
        return;
    }

    /* (display expr) */
    if (is_sym(head, "display") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_PRINT, 0);
        return;
    }

    /* (if cond then else) */
    if (is_sym(head, "if") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0);
        int jf = placeholder(c);
        compile_expr(c, node->children[2], tail);
        if (node->n_children >= 4) {
            int jend = placeholder(c);
            patch(c, jf, OP_JUMP_IF_FALSE, c->code_len);
            compile_expr(c, node->children[3], tail);
            patch(c, jend, OP_JUMP, c->code_len);
        } else {
            /* One-armed `if`: the false path must still leave a value. Every
             * consumer of an expression's result assumes exactly one — the
             * `begin` sequencer, the operand-tracking helpers, and the
             * top-level driver, all of which emit an unconditional OP_POP.
             * Falling straight through to the merge point pushed nothing, so
             * that POP removed a slot the expression never pushed, leaving sp
             * one below the live top. Nothing faulted: reads go by absolute
             * slot index, so the damage only surfaced when the next push
             * overwrote the top-level binding sitting just above sp, silently
             * replacing a live variable with an unrelated value. */
            int jend = placeholder(c);
            patch(c, jf, OP_JUMP_IF_FALSE, c->code_len);
            chunk_emit(c, OP_VOID, 0);
            patch(c, jend, OP_JUMP, c->code_len);
        }
        return;
    }

    /* (begin e1 e2 ...) */
    if (is_sym(head, "begin")) {
        for (int i = 1; i < node->n_children; i++) {
            if (i < node->n_children - 1) {
                compile_expr(c, node->children[i], 0);
                chunk_emit(c, OP_POP, 0);
            } else {
                compile_expr(c, node->children[i], tail);
            }
        }
        return;
    }

    /* (let ((var val) ...) body) */
    /* Named let: (let name ((var init) ...) body ...)
     * Compiles as: (letrec ((name (lambda (vars...) body...))) (name inits...)) */
    if (is_sym(head, "let") && node->n_children >= 4
        && node->children[1]->type == N_SYMBOL
        && node->children[2]->type == N_LIST) {
        char* loop_name = node->children[1]->symbol;
        Node* bindings = node->children[2];
        int saved_locals = c->n_locals;
        c->scope_depth++;

        /* Compile as letrec with a single binding: the loop function */
        /* Push NIL placeholder for the loop function */
        chunk_emit(c, OP_NIL, 0);
        int loop_slot = add_local(c, loop_name);

        /* Compile the loop function body */
        FuncChunk func; chunk_init_arrays(&func);
        func.enclosing = c;
        func.param_count = bindings->n_children;
        for (int i = 0; i < bindings->n_children; i++) {
            Node* b = bindings->children[i];
            if (b->type == N_LIST && b->n_children >= 1)
                add_local(&func, b->children[0]->symbol);
        }
        for (int i = 3; i < node->n_children; i++) {
            int is_last = (i == node->n_children - 1);
            compile_expr(&func, node->children[i], is_last);
            if (!is_last) chunk_emit(&func, OP_POP, 0);
        }
        chunk_emit(&func, OP_RETURN, 0);

        /* Inline function code */
        int const_map_nl[4096];
        for (int i = 0; i < func.n_constants; i++)
            const_map_nl[i] = chunk_add_const(c, func.constants[i]);
        int cfunc = chunk_add_const(c, INT_VAL(0));
        int jover = placeholder(c);
        int func_pc = c->code_len;
        c->constants[cfunc].as.i = func_pc;

        /* Adjust nested CLOSURE PC constants: any lambda compiled inside the loop
         * body has its func_pc stored as a constant in func's constant pool. When we
         * inline func into c, those PC values must be offset by func_pc (same fix that
         * compile_form_lambda_2 applies at lines 1313-1319). */
        for (int i = 0; i < func.code_len; i++) {
            if (func.code[i].op == OP_CLOSURE) {
                int ci = func.code[i].operand & 0xFFFF;
                int parent_ci = const_map_nl[ci];
                c->constants[parent_ci].as.i += func_pc;
            }
        }

        for (int i = 0; i < func.code_len; i++) {
            Instr fi = func.code[i];
            if (fi.op == OP_CONST) fi.operand = const_map_nl[fi.operand];
            if (fi.op == OP_JUMP || fi.op == OP_JUMP_IF_FALSE || fi.op == OP_LOOP || fi.op == OP_PUSH_HANDLER)
                fi.operand += func_pc;
            if (fi.op == OP_CLOSURE) {
                int ci2 = fi.operand & 0xFFFF;
                int nu2 = (fi.operand >> 16) & 0xFF;
                fi.operand = const_map_nl[ci2] | (nu2 << 16);
            }
            chunk_emit_instr(c, fi);
        }
        patch(c, jover, OP_JUMP, c->code_len);

        /* Create closure with self-reference upvalue */
        int n_upvals = func.n_upvalues;
        int self_uv_idx = -1;
        for (int i = 0; i < n_upvals; i++) {
            if (strcmp(func.upvalues[i].name, loop_name) == 0) {
                chunk_emit(c, OP_NIL, 0);
                self_uv_idx = func.upvalues[i].index;
            } else {
                chunk_emit(c, func.upvalues[i].is_local ? OP_GET_LOCAL : OP_GET_UPVALUE,
                           func.upvalues[i].enclosing_slot);
            }
        }
        chunk_emit(c, OP_CLOSURE, cfunc | (n_upvals << 16));
        if (self_uv_idx >= 0) chunk_emit(c, OP_CLOSE_UPVALUE, self_uv_idx);

        /* Finish the lambda lowering: convert captured upvalues to open
         * (by-reference) slots exactly as compile_form_lambda_2() and
         * compile_form_define() do for their closures.  Without this the loop
         * closure holds by-VALUE copies, so a `set!` of an enclosing variable
         * from the loop body writes the copy and the write vanishes when the
         * loop returns.  Top level: native 151 opens a direct reference to the
         * (permanent) top-level frame slot.  Nested: native 252 relays the
         * enclosing closure's own open slot, which is the only reference that
         * stays valid once this frame is gone.
         *
         * Unlike a general lambda (compile_form_lambda_2, which must not open
         * slots into a frame its closure can outlive), a named let's closure is
         * called synchronously and discarded before the enclosing frame
         * returns, so opening a slot into a NESTED frame is sound here — that
         * is the only way `(define (f n) (let loop (...) (set! n ...)) n)` can
         * see the loop's writes.  The one escape route, leaking the loop
         * closure out of the named let, is closed by native 152 below. */
        int opened_frame_slot = 0;
        for (int i = 0; i < n_upvals; i++) {
            if (strcmp(func.upvalues[i].name, loop_name) == 0) continue;
            int direct = func.upvalues[i].is_local;
            if (direct && c->enclosing != NULL) opened_frame_slot = 1;
            chunk_emit(c, OP_DUP, 0);
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(i)));
            chunk_emit(c, OP_CONST,
                       chunk_add_const(c, INT_VAL(func.upvalues[i].enclosing_slot)));
            chunk_emit(c, OP_NATIVE_CALL, direct ? 151 : 252);
            chunk_emit(c, OP_POP, 0);
        }

        /* Store closure in loop_slot */
        chunk_emit(c, OP_SET_LOCAL, loop_slot);

        /* Note: self-reference upvalue is already patched by OP_CLOSE_UPVALUE
         * at line above (after OP_CLOSURE). No additional open_upvalues needed. */

        /* Call the loop function with initial values */
        chunk_emit(c, OP_GET_LOCAL, loop_slot);
        for (int i = 0; i < bindings->n_children; i++) {
            Node* b = bindings->children[i];
            if (b->type == N_LIST && b->n_children >= 2)
                compile_expr(c, b->children[1], 0);
            else
                chunk_emit(c, OP_NIL, 0);
        }
        int body_tail = 1 > 0 ? 0 : tail; /* don't tail-call — need POPN cleanup */
        chunk_emit(c, body_tail ? OP_TAIL_CALL : OP_CALL, bindings->n_children);

        /* The loop is finished, so every by-reference slot it opened into THIS
         * frame must be closed (native 152) before the frame can go away.  A
         * loop closure that never escaped is unreachable from here on; one that
         * did escape keeps the values it last saw instead of a dangling alias
         * into a frame that is about to be reused. */
        if (opened_frame_slot) {
            chunk_emit(c, OP_GET_LOCAL, loop_slot);
            chunk_emit(c, OP_NATIVE_CALL, 152);
            chunk_emit(c, OP_POP, 0);
        }

        /* Cleanup */
        chunk_free_arrays(&func);
        chunk_emit(c, OP_POPN, 1); /* remove loop function slot */
        c->n_locals = saved_locals;
        c->scope_depth--;
        return;
    }

    /* (let ((var val) ...) body) — compile using stack locals.
     * Variables that are both captured by closures AND mutated via set!
     * are heap-boxed: stored in a 1-element vector so all closures share state. */
    if (is_sym(head, "let") && node->n_children >= 3 && node->children[1]->type == N_LIST) { compile_form_let(c, node, tail); return; }

    /* (let* ((var val) ...) body) — sequential bindings */
    if (is_sym(head, "let*") && node->n_children >= 3 && node->children[1]->type == N_LIST) { compile_form_let_star(c, node, tail); return; }

    /* (letrec ((var val) ...) body) — recursive bindings with open upvalues.
     *
     * Letrec semantics: all bindings are visible to all initializers.
     * Implementation:
     * 1. Push NIL placeholders for all bindings
     * 2. Compile each initializer (lambdas capture open upvalue refs to stack slots)
     * 3. SET_LOCAL each initializer result to its slot
     * 4. Now all closures' open upvalues point to the correct stack slots
     * 5. When a closure reads GET_UPVALUE, it reads the current stack value (open ref)
     *
     * The key: compile_expr for the lambda creates a closure. The closure's upvalues
     * capture VALUES from the stack (which are NIL at creation time). We need them
     * to capture REFERENCES instead.
     *
     * Simplest correct approach: after creating all closures and SET_LOCAL'ing them,
     * use NATIVE_CALL to patch each closure's upvalue to read from the stack slot.
     * Or: use OP_CLOSE_UPVALUE to patch each closure's upvalue after all are defined. */
    if (is_sym(head, "letrec") && node->n_children >= 3 && node->children[1]->type == N_LIST) { compile_form_letrec(c, node, tail); return; }

    /* (letrec* ((var val) ...) body) — sequential recursive (R7RS) */
    if (is_sym(head, "letrec*") && node->n_children >= 3 && node->children[1]->type == N_LIST) { compile_form_letrec_star(c, node, tail); return; }

    /* (define name value) or (define (name params...) body) */
    if (is_sym(head, "define") && node->n_children >= 3) { compile_form_define(c, node, tail); return; }

    /* (set! name value) */
    if (is_sym(head, "set!") && node->n_children == 3 && node->children[1]->type == N_SYMBOL) { compile_form_set_bang(c, node, tail); return; }

    /* (do ((var init step) ...) (test result) body ...) */
    if (is_sym(head, "do") && node->n_children >= 3) { compile_form_do(c, node, tail); return; }

    /* (and e1 e2 ...) — short circuit */
    if (is_sym(head, "and") && node->n_children >= 2) {
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) {
            chunk_emit(c, OP_DUP, 0);
            int jf = placeholder(c);
            chunk_emit(c, OP_POP, 0);
            compile_expr(c, node->children[i], 0);
            patch(c, jf, OP_JUMP_IF_FALSE, c->code_len);
        }
        return;
    }

    /* (or e1 e2 ...) — short circuit */
    if (is_sym(head, "or") && node->n_children >= 2) {
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) {
            chunk_emit(c, OP_DUP, 0);
            chunk_emit(c, OP_NOT, 0);
            int jf = placeholder(c);
            chunk_emit(c, OP_POP, 0);
            compile_expr(c, node->children[i], 0);
            patch(c, jf, OP_JUMP_IF_FALSE, c->code_len);
        }
        return;
    }

    /* (lambda (params...) body) */
    /* (lambda args body) — all args as a list */
    if (is_sym(head, "lambda") && node->n_children >= 3 && node->children[1]->type == N_SYMBOL) { compile_form_lambda(c, node, tail); return; }

    /* (lambda (x y . rest) body) or (lambda (x y) body) — standard and variadic */
    if (is_sym(head, "lambda") && node->n_children >= 3 && node->children[1]->type == N_LIST) { compile_form_lambda_2(c, node, tail); return; }

    /* (quote datum) — compile arbitrary quoted data to cons cells */
    if (is_sym(head, "quote") && node->n_children == 2) {
        compile_quote(c, node->children[1]);
        return;
    }

    /* (quasiquote datum) — compile with unquote/unquote-splicing support */
    if (is_sym(head, "quasiquote") && node->n_children == 2) {
        compile_quasiquote(c, node->children[1]);
        return;
    }

    /* New tensor dtype constructor syntax:
     *   (make-tensor dims fill :dtype 'f32)
     *
     * The generic builtins table is fixed-arity, so the optional dtype spelling
     * needs a VM compiler lowering while the two-argument form remains a normal
     * first-class builtin call. */
    if (is_sym(head, "make-tensor") && node->n_children == 5 &&
        node->children[3]->type == N_SYMBOL &&
        (strcmp(node->children[3]->symbol, ":dtype") == 0 ||
         strcmp(node->children[3]->symbol, "#:dtype") == 0 ||
         strcmp(node->children[3]->symbol, "dtype") == 0)) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[4], 0);
        chunk_emit(c, OP_NATIVE_CALL, 423);
        return;
    }

    /* =========================================================================
     * All functions from here to the syntax-forms block below are handled
     * via the BUILTINS table (emit_builtin_preamble creates first-class
     * closures for each). The compiler must NOT intercept them — user-defined
     * functions with the same name would otherwise be silently bypassed.
     *
     * Any function in the BUILTINS[] array in eshkol_vm.c is accessible via
     * normal global variable lookup and should NOT appear in this compiler.
     * ========================================================================= */
    /* Section removed: all BUILTINS-table functions go through variable lookup */

    /***************************************************************************
     * Syntax forms: let-syntax, letrec-syntax, define-values, syntax-error,
     * include, include-ci, OALR forms, with-region, define-type
     ***************************************************************************/

    /* -- let-syntax -- */
    if (is_sym(head, "let-syntax") && node->n_children >= 3) {
        Node* bindings = node->children[1];
        int saved = g_n_macros;
        for (int i = 0; i < bindings->n_children; i++)
            vm_macro_define_syntax((const MacroNode*)bindings->children[i]);
        for (int i = 2; i < node->n_children; i++)
            compile_expr(c, node->children[i], tail && i == node->n_children - 1);
        g_n_macros = saved;
        return;
    }

    /* -- letrec-syntax -- */
    if (is_sym(head, "letrec-syntax") && node->n_children >= 3) {
        Node* bindings = node->children[1];
        int saved = g_n_macros;
        for (int i = 0; i < bindings->n_children; i++)
            vm_macro_define_syntax((const MacroNode*)bindings->children[i]);
        for (int i = 2; i < node->n_children; i++)
            compile_expr(c, node->children[i], tail && i == node->n_children - 1);
        g_n_macros = saved;
        return;
    }

    /* -- define-values -- */
    if (is_sym(head, "define-values") && node->n_children >= 3) {
        compile_form_define_values(c, node);
        return;
    }

    /* -- syntax-error -- */
    if (is_sym(head, "syntax-error")) {
        if (node->n_children >= 2)
            fprintf(stderr, "SYNTAX ERROR: %s\n",
                    node->children[1]->type == N_STRING ? node->children[1]->symbol : "unknown");
        return;
    }

    /* -- include / include-ci -- */
    if ((is_sym(head, "include") || is_sym(head, "include-ci")) && node->n_children >= 2) {
#ifdef ESHKOL_VM_NO_DISASM
        /* WASM: no filesystem, skip include */
        return;
#else
        const char* path = node->children[1]->symbol;
        int fold_case = is_sym(head, "include-ci");
        FILE* incf = fopen(path, "r");
        if (incf) {
            fseek(incf, 0, SEEK_END); long len = ftell(incf); fseek(incf, 0, SEEK_SET);
            char* src = (char*)malloc(len + 1);
            if (src) {
                fread(src, 1, len, incf); src[len] = 0; fclose(incf);
                const char* saved = src_ptr;
                int saved_fold_case = g_compiler_ctx.fold_case_symbols;
                src_ptr = src;
                g_compiler_ctx.fold_case_symbols = fold_case;
                while (1) { skip_ws(); if (!*src_ptr) break; Node* e = parse_sexp(); if (!e) break; compile_expr(c, e, 0); free_node(e); }
                src_ptr = saved;
                g_compiler_ctx.fold_case_symbols = saved_fold_case;
                free(src);
            } else fclose(incf);
        }
        return;
#endif
    }

    /* -- OALR forms (pass-through: ownership enforced at compile-time, not runtime) -- */
    if (is_sym(head, "owned") && node->n_children == 2) { compile_expr(c, node->children[1], tail); return; }
    if (is_sym(head, "move") && node->n_children == 2) { compile_expr(c, node->children[1], tail); return; }
    if (is_sym(head, "borrow") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0); /* the borrowed value */
        for (int i = 2; i < node->n_children; i++)
            compile_expr(c, node->children[i], tail && i == node->n_children - 1);
        return;
    }
    if (is_sym(head, "shared") && node->n_children == 2) { compile_expr(c, node->children[1], tail); return; }
    if (is_sym(head, "weak-ref") && node->n_children == 2) { compile_expr(c, node->children[1], tail); return; }

    /* -- with-region -- */
    if (is_sym(head, "with-region") && node->n_children >= 2) {
        compile_form_with_region(c, node, tail);
        return;
    }

    /* -- define-type (type alias: compile-time only, no runtime effect) -- */
    if (is_sym(head, "define-type")) { return; }


    if (is_sym(head, "vref") && node->n_children == 3) {
        int s = c->n_locals;
        compile_operands_tracked(c, node, 1, 2);
        c->n_locals = s;
        chunk_emit(c, OP_VEC_REF, 0); return;
    }

    /* SW-31: real? lowers to native 1697 (the whole tower minus COMPLEX)
     * instead of aliasing the number? opcode, which omitted RATIONAL/BIGNUM. */
    if (is_sym(head, "real?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 1697); return;
    }

    /* `derivative` curried form (ESH-0369): `(derivative f)` with no point is
     * the documented spelling that returns f' as a first-class procedure. The
     * VM reaches `derivative` as native call id 393, which pops exactly
     * (f, x) — so the curried form used to pop garbage off the operand stack
     * and bind a non-callable, and applying it failed with "calling
     * non-function". Lower it the same way `gradient` lowers its curry (see
     * below): synthesize
     *     (lambda (__dx__) (derivative <f> __dx__))
     * from existing forms, so no builtin is added (the builtin count, and
     * therefore the top-level slot layout, is unchanged) and the curried form
     * reaches native 393 with exactly the same (f, x) as the direct form —
     * hence identical values.
     *
     * FIRST order only. Differentiating the resulting closure again is nested
     * differentiation, which the VM's flat single-perturbation dual cannot
     * represent; native call 393 now RAISES for a dual-valued point rather
     * than fabricating 0 (see vm_native.c case 393), so the unsupported case
     * is loud. Tracked as a native-only row in tests/vm_parity/PARITY.tsv. */
    if (is_sym(head, "derivative") && node->n_children == 2) {
        Node* inner = make_call_node("derivative");
        add_child(inner, node->children[1]);
        add_child(inner, make_symbol_node("__dx__"));
        Node* params = make_node(N_LIST);          /* (__dx__) — one FIXED param */
        add_child(params, make_symbol_node("__dx__"));
        Node* lam = make_call_node("lambda");
        add_child(lam, params);
        add_child(lam, inner);
        compile_expr(c, lam, tail);
        /* Synthetic wrapper nodes intentionally leak (bounded, one per curried
         * occurrence); children[1] is shared with `node` and is freed with the
         * top-level AST, so it must NOT be freed here — same ownership rule as
         * the `gradient` curry below. */
        return;
    }

    /* `gradient` special form: currying + point-spreading, lowered so the
     * callable stays on the operand stack of the native gradient primitive
     * (call id 750) rather than being routed through an intermediate Scheme
     * wrapper frame — a first-class/named loss reached through `f` is then
     * differentiated by the closure bridge exactly as the native path does.
     * Forms:
     *   (gradient f point)  -> native 750 directly (f, point on the stack)
     *   (gradient f)        -> a curry closure that, applied to a single
     *                          collection passes it whole, to a single scalar
     *                          stays scalar, and to N scalars gathers a vector
     *   (gradient f x y …)  -> (gradient f (vector x y …))
     * The curry closure is synthesized from existing forms (no dedicated
     * builtin, so the builtin count — and thus top-level slot layout — is
     * unchanged), and the curried ((gradient f) …) form is that closure
     * applied to its arguments; every surface form reaches native 750 with the
     * same (f, point), so direct/wrapped/curried agree bit-for-bit. */
    if (is_sym(head, "gradient") && node->n_children >= 2) {
        if (node->n_children == 3) {
            int s = c->n_locals;
            compile_operands_tracked(c, node, 1, 2); /* push f, then point */
            c->n_locals = s;
            chunk_emit(c, OP_NATIVE_CALL, 750);
            return;
        }
        Node* fexpr = node->children[1];
        if (node->n_children == 2) {
            /* (lambda __ga__
             *   (if (if (pair? __ga__) (null? (cdr __ga__)) #f)
             *       (gradient <f> (car __ga__))
             *       (gradient <f> (list->vector __ga__)))) */
            Node* single = make_call_node("if");
            {   Node* isp = make_call_node("pair?");
                add_child(isp, make_symbol_node("__ga__"));
                Node* isn = make_call_node("null?");
                Node* cdrn = make_call_node("cdr");
                add_child(cdrn, make_symbol_node("__ga__"));
                add_child(isn, cdrn);
                add_child(single, isp);
                add_child(single, isn);
                Node* f_false = make_node(N_BOOL); f_false->numval = 0;
                add_child(single, f_false);
            }
            Node* whole = make_call_node("gradient");
            add_child(whole, fexpr);
            {   Node* carn = make_call_node("car");
                add_child(carn, make_symbol_node("__ga__"));
                add_child(whole, carn); }
            Node* spread = make_call_node("gradient");
            add_child(spread, fexpr);
            {   Node* l2v = make_call_node("list->vector");
                add_child(l2v, make_symbol_node("__ga__"));
                add_child(spread, l2v); }
            Node* dispatch = make_call_node("if");
            add_child(dispatch, single);
            add_child(dispatch, whole);
            add_child(dispatch, spread);
            Node* lam = make_call_node("lambda");
            add_child(lam, make_symbol_node("__ga__"));
            add_child(lam, dispatch);
            compile_expr(c, lam, tail);
            /* Synthetic wrapper nodes intentionally leak (bounded, one per
             * curried occurrence); `fexpr` is shared with `node` and freed with
             * the top-level AST, so it must NOT be freed here. */
            return;
        }
        /* n_children >= 4: (gradient f a b …) -> (gradient f (vector a b …)). */
        Node* vec = make_call_node("vector");
        for (int i = 2; i < node->n_children; i++) add_child(vec, node->children[i]);
        Node* grad = make_call_node("gradient");
        add_child(grad, fexpr);
        add_child(grad, vec);
        compile_expr(c, grad, tail);
        return;
    }

    /* Function call: (f arg1 arg2 ...)
     * Register each pushed value as an anonymous local so n_locals tracks
     * the actual stack depth. This prevents let/letrec inside arguments
     * from allocating slots that conflict with operand stack values. */
    if (head->type == N_SYMBOL || head->type == N_LIST) {
        int argc = node->n_children - 1;
        int saved_locals = c->n_locals;
        compile_expr(c, head, 0);  /* push function */
        add_local(c, "__call_func__");
        for (int i = 1; i < node->n_children; i++) {
            compile_expr(c, node->children[i], 0);
            add_local(c, "__call_arg__");
        }
        if (head->type == N_SYMBOL && vm_language_coverage_compilation_enabled()) {
            chunk_emit(c, OP_LANGUAGE_COVERAGE_CALL,
                       (int)vm_language_coverage_name_hash(head->symbol));
        }
        if (tail)
            chunk_emit(c, OP_TAIL_CALL, argc);
        else
            chunk_emit(c, OP_CALL, argc);
        c->n_locals = saved_locals; /* CALL consumed func+args, restore n_locals */
        return;
    }

    fprintf(stderr, "WARNING: unhandled: %s\n", head->type == N_SYMBOL ? head->symbol : "(?)");
    chunk_emit(c, OP_NIL, 0);
}


/* ── Peephole Optimization (from compiler) ── */
