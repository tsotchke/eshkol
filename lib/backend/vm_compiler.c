static void compile_expr_impl(FuncChunk* c, Node* node, int tail);

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
        int ci = chunk_add_const(c, node->numval == (int64_t)node->numval ? INT_VAL((int64_t)node->numval) : FLOAT_VAL(node->numval));
        if (ci >= 0) chunk_emit(c, OP_CONST, ci);
        return;
    }
    /* Atom: symbol — quote as string */
    if (node->type == N_SYMBOL) {
        int len = (int)strlen(node->symbol);
        int n_packs = (len + 7) / 8;
        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(len)));
        for (int p = 0; p < n_packs; p++) {
            int64_t pack = 0;
            for (int b = 0; b < 8 && p * 8 + b < len; b++)
                pack |= ((int64_t)(unsigned char)node->symbol[p * 8 + b]) << (b * 8);
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(pack)));
        }
        chunk_emit(c, OP_NATIVE_CALL, 100);
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

    /* List: build from right to left using cons */
    if (node->type == N_LIST) {
        chunk_emit(c, OP_NIL, 0); /* start with empty list */
        for (int i = node->n_children - 1; i >= 0; i--) {
            Node* elem = node->children[i];
            /* Check for unquote-splicing */
            if (elem->type == N_LIST && elem->n_children == 2 &&
                elem->children[0]->type == N_SYMBOL &&
                strcmp(elem->children[0]->symbol, "unquote-splicing") == 0) {
                /* Compile the spliced expression */
                compile_expr(c, elem->children[1], 0);
                /* Append to accumulator: (append spliced acc) */
                chunk_emit(c, OP_NATIVE_CALL, 73); /* append */
            } else {
                compile_quasiquote(c, elem);
                chunk_emit(c, OP_CONS, 0);
            }
        }
        return;
    }

    /* Fallback: emit nil */
    chunk_emit(c, OP_NIL, 0);
}

/* compile_depth tracked in global context — not in CompilerContext struct
 * because it's transient per-compilation, not persistent state */
static int compile_depth = 0;

static void compile_expr(FuncChunk* c, Node* node, int tail) {
    compile_depth++;
    if (compile_depth > 1000) { fprintf(stderr, "ERROR: expression nesting too deep (>1000)\n"); compile_depth--; return; }
    compile_expr_impl(c, node, tail);
    compile_depth--;
}


/* ═══ Extracted compilation sub-functions ═══ */

static void compile_form_cond(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    int end_patches[64];
    int n_patches = 0;
    for (int i = 1; i < node->n_children; i++) {
        Node* clause = node->children[i];
        if (clause->type != N_LIST || clause->n_children < 2) continue;
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

static void compile_form_require(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    if (node->n_children >= 2 && node->children[1]->type == N_SYMBOL) {
        const char* mod_name = node->children[1]->symbol;
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
#endif
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
                /* Parse and compile all top-level forms */
                const char* saved_src = src_ptr;
                src_ptr = src;
                while (1) {
                    skip_ws();
                    if (!*src_ptr) break;
                    Node* expr = parse_sexp();
                    if (!expr) break;
                    compile_expr(c, expr, 0);
                    free_node(expr);
                }
                src_ptr = saved_src;
                free(src);
            } else {
                fclose(mf);
            }
        }
        /* If file not found, silently continue (builtins always available) */
    }
    return;
}

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

static void compile_form_parameterize(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    Node* bindings = node->children[1];
    int n_bindings = bindings->n_children;

    /* Push each parameter binding */
    for (int i = 0; i < n_bindings; i++) {
        if (bindings->children[i]->type == N_LIST &&
            bindings->children[i]->n_children == 2) {
            compile_expr(c, bindings->children[i]->children[0], 0); /* param */
            compile_expr(c, bindings->children[i]->children[1], 0); /* new value */
            chunk_emit(c, OP_NATIVE_CALL, 702); /* parameterize-push */
            chunk_emit(c, OP_POP, 0); /* discard void result */
        }
    }

    /* Compile body */
    for (int i = 2; i < node->n_children; i++) {
        if (i > 2) chunk_emit(c, OP_POP, 0);
        compile_expr(c, node->children[i], tail && i == node->n_children - 1);
    }

    /* Pop each binding in reverse order for proper unwinding */
    for (int i = n_bindings - 1; i >= 0; i--) {
        if (bindings->children[i]->type == N_LIST &&
            bindings->children[i]->n_children >= 1) {
            compile_expr(c, bindings->children[i]->children[0], 0); /* param */
            chunk_emit(c, OP_NATIVE_CALL, 703); /* parameterize-pop */
            chunk_emit(c, OP_POP, 0);
        }
    }
    return;
}

static void compile_form_let_values(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    Node* bindings_list = node->children[1];
    int saved_locals = c->n_locals;

    for (int b = 0; b < bindings_list->n_children; b++) {
        Node* binding = bindings_list->children[b];
        if (binding->type != N_LIST || binding->n_children != 2) continue;
        Node* formals = binding->children[0]; /* (x y ...) or single var */
        Node* producer = binding->children[1];

        /* Compile the producer expression */
        compile_expr(c, producer, 0);

        if (formals->type == N_LIST) {
            /* Multiple return values — bind first to result, rest get nil */
            if (formals->n_children > 0)
                add_local(c, formals->children[0]->symbol);
            for (int i = 1; i < formals->n_children; i++) {
                chunk_emit(c, OP_NIL, 0);
                add_local(c, formals->children[i]->symbol);
            }
        } else if (formals->type == N_SYMBOL) {
            /* Single variable */
            add_local(c, formals->symbol);
        }
    }

    /* Compile body expressions */
    for (int i = 2; i < node->n_children; i++) {
        if (i > 2) chunk_emit(c, OP_POP, 0);
        compile_expr(c, node->children[i], tail && i == node->n_children - 1);
    }

    /* Clean up scope: pop bindings below result */
    int n_bound = c->n_locals - saved_locals;
    if (n_bound > 0)
        chunk_emit(c, OP_POPN, n_bound);
    c->n_locals = saved_locals;
    return;
}

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

static void compile_form_dynamic_wind(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    /* Call before() */
    compile_expr(c, node->children[1], 0);
    chunk_emit(c, OP_CALL, 0);
    chunk_emit(c, OP_POP, 0);

    /* Push after thunk onto wind stack */
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

static void compile_form_delay(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    {
        /* Save current chunk state, compile a sub-function */
        FuncChunk func; chunk_init_arrays(&func);
        func.enclosing = c;
        func.param_count = 0;
        compile_expr(&func, node->children[1], 1); /* compile expr as body */
        chunk_emit(&func, OP_RETURN, 0);
        /* Inline the function code */
        int jover = c->code_len;
        chunk_emit(c, OP_JUMP, 0);
        int cfunc = c->n_constants;
        chunk_add_const(c, INT_VAL(c->code_len));
        for (int i = 0; i < func.code_len; i++) {
            Instr fi = func.code[i];
            if (fi.op == OP_CONST) fi.operand = chunk_add_const(c, func.constants[fi.operand]);
            if (fi.op == OP_JUMP || fi.op == OP_JUMP_IF_FALSE || fi.op == OP_LOOP || fi.op == OP_PUSH_HANDLER)
                fi.operand += c->code_len;
            chunk_emit(c, fi.op, fi.operand);
        }
        patch(c, jover, OP_JUMP, c->code_len);
        /* Stack: push #f, push closure, create vector */
        chunk_emit(c, OP_FALSE, 0);
        chunk_emit(c, OP_CLOSURE, cfunc);
        chunk_emit(c, OP_VEC_CREATE, 2); /* #(#f thunk) */
        chunk_free_arrays(&func);
    }
    return;
}

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

static void compile_form_let_star(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    int saved_locals = c->n_locals;
    c->scope_depth++;
    Node* bindings = node->children[1];
    for (int i = 0; i < bindings->n_children; i++) {
        Node* b = bindings->children[i];
        if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
            compile_expr(c, b->children[1], 0);
            add_local(c, b->children[0]->symbol);
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

static void compile_form_letrec(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    int saved_locals = c->n_locals;
    c->scope_depth++;
    Node* bindings = node->children[1];
    int n_bindings = 0;

    /* 1. Push NIL placeholders and register names */
    for (int i = 0; i < bindings->n_children; i++) {
        Node* b = bindings->children[i];
        if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
            chunk_emit(c, OP_NIL, 0);
            add_local(c, b->children[0]->symbol);
            n_bindings++;
        }
    }
    int n_let_locals = c->n_locals - saved_locals;

    /* 2. Compile each initializer and SET_LOCAL */
    for (int i = 0; i < bindings->n_children; i++) {
        Node* b = bindings->children[i];
        if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
            compile_expr(c, b->children[1], 0);
            int slot = resolve_local(c, b->children[0]->symbol);
            if (slot >= 0) chunk_emit(c, OP_SET_LOCAL, slot);
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

static void compile_form_letrec_star(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    int saved_locals = c->n_locals;
    c->scope_depth++;
    Node* bindings = node->children[1];
    for (int i = 0; i < bindings->n_children; i++) {
        Node* b = bindings->children[i];
        if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
            chunk_emit(c, OP_NIL, 0);
            add_local(c, b->children[0]->symbol);
        }
    }
    int n_let_locals = c->n_locals - saved_locals;
    for (int i = 0; i < bindings->n_children; i++) {
        Node* b = bindings->children[i];
        if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
            compile_expr(c, b->children[1], 0);
            int slot = resolve_local(c, b->children[0]->symbol);
            if (slot >= 0) chunk_emit(c, OP_SET_LOCAL, slot);
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

static void compile_form_define(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    if (node->children[1]->type == N_SYMBOL) {
        /* Simple variable definition */
        compile_expr(c, node->children[2], 0);
        add_local(c, node->children[1]->symbol);
        return;
    }
    if (node->children[1]->type == N_LIST && node->children[1]->n_children >= 1) {
        /* Function definition: (define (name params...) body) */
        Node* sig = node->children[1];
        char* fname = sig->children[0]->symbol;

        int func_slot = add_local(c, fname);
        int pre_registered = 0;

        /* Compile function body into a separate chunk.
         * The body can reference fname via GET_UPVALUE which will be captured
         * from the enclosing scope's func_slot. */
        FuncChunk func; chunk_init_arrays(&func);
        func.enclosing = c;

        /* Check for dot notation in params: (name x y . rest) */
        int has_rest = 0, fixed_params = sig->n_children - 1;
        for (int i = 1; i < sig->n_children; i++) {
            if (sig->children[i]->type == N_SYMBOL && strcmp(sig->children[i]->symbol, ".") == 0) {
                has_rest = 1;
                fixed_params = i - 1;
                break;
            }
        }
        func.param_count = has_rest ? 255 : fixed_params;

        /* Add fixed parameters as locals */
        for (int i = 1; i <= fixed_params; i++)
            add_local(&func, sig->children[i]->symbol);
        /* Add rest parameter if present */
        if (has_rest && fixed_params + 2 < sig->n_children) {
            add_local(&func, sig->children[fixed_params + 2]->symbol); /* name after dot */
            chunk_emit(&func, OP_PACK_REST, fixed_params);
        }

        /* Compile body expressions */
        for (int i = 2; i < node->n_children; i++) {
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
        c->constants[cfunc].as.i = actual_func_pc;

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
            }
        }
        chunk_free_arrays(&func);
        return;
    }
}

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
        /* Boxed local: emit GET_LOCAL(box), CONST(0), compile(value), VEC_SET */
        chunk_emit(c, OP_GET_LOCAL, slot);  /* push box (vector) */
        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0))); /* index 0 */
        compile_expr(c, node->children[2], 0); /* compile new value */
        chunk_emit(c, OP_VEC_SET, 0);       /* box[0] = value */
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
        for (int d = depth - 1; d >= 1 && !found; d--) {
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
                        /* Boxed upvalue: GET_UPVALUE(box), CONST 0, value, VEC_SET */
                        chunk_emit(c, OP_GET_UPVALUE, final_uv);
                        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
                        compile_expr(c, node->children[2], 0);
                        chunk_emit(c, OP_VEC_SET, 0);
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

static void compile_form_do(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    c->scope_depth++;
    Node* vars = node->children[1];
    Node* test = node->children[2];

    /* Initialize loop variables */
    for (int i = 0; i < vars->n_children; i++) {
        Node* b = vars->children[i];
        if (b->type == N_LIST && b->n_children >= 2 && b->children[0]->type == N_SYMBOL) {
            compile_expr(c, b->children[1], 0);
            add_local(c, b->children[0]->symbol);
        }
    }

    /* Loop top */
    int loop_top = c->code_len;

    /* Test */
    if (test->type == N_LIST && test->n_children >= 1) {
        compile_expr(c, test->children[0], 0);
        int jexit = placeholder(c);

        /* Body (if any) */
        for (int i = 3; i < node->n_children; i++) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_POP, 0);
        }

        /* Step — evaluate ALL step expressions, then store (parallel) */
        int step_count = 0;
        for (int i = 0; i < vars->n_children; i++) {
            Node* b = vars->children[i];
            if (b->type == N_LIST && b->n_children >= 3) {
                compile_expr(c, b->children[2], 0);
                step_count++;
            }
        }
        /* Store in reverse order */
        for (int i = vars->n_children - 1; i >= 0; i--) {
            Node* b = vars->children[i];
            if (b->type == N_LIST && b->n_children >= 3) {
                int slot = resolve_local(c, b->children[0]->symbol);
                if (slot >= 0) chunk_emit(c, OP_SET_LOCAL, slot);
            }
        }

        /* Loop back */
        chunk_emit(c, OP_LOOP, loop_top);

        /* Exit: evaluate result expression */
        patch(c, jexit, OP_JUMP_IF_FALSE, c->code_len - 1);
        /* Wait — JUMP_IF_FALSE jumps when false. The test is the EXIT condition.
         * When test is TRUE → exit. When FALSE → continue loop.
         * So: if test is true → DON'T jump (fall through to exit).
         *     if test is false → jump back to loop.
         * Need: JUMP_IF_FALSE → loop_body, then after body+step → LOOP back.
         * After LOOP: exit point. */
        /* Actually restructure: test → if FALSE, do body+step+loop. If TRUE, exit. */
        /* Current: test → jexit (JUMP_IF_FALSE to ???). Body. Step. LOOP.
         * jexit should point to AFTER the LOOP (the exit point).
         * But JUMP_IF_FALSE jumps when FALSE. If test is FALSE → continue loop body.
         * If test is TRUE → skip to exit.
         * So JUMP_IF_FALSE should jump PAST the exit... no.
         *
         * Let me use: NOT test → JUMP_IF_FALSE to exit. */
        /* Restart: */
        c->code_len = loop_top; /* redo from loop top */
        compile_expr(c, test->children[0], 0);
        /* test is TRUE when loop should exit */
        int jbody = placeholder(c); /* JUMP_IF_FALSE → body (continue loop) */
        /* Exit: result */
        if (test->n_children >= 2)
            compile_expr(c, test->children[1], tail);
        else
            chunk_emit(c, OP_NIL, 0);
        int jexit2 = placeholder(c); /* JUMP over body+step */

        /* Body */
        int body_start = c->code_len;
        patch(c, jbody, OP_JUMP_IF_FALSE, body_start);

        for (int i = 3; i < node->n_children; i++) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_POP, 0);
        }

        /* Step */
        step_count = 0;
        for (int i = 0; i < vars->n_children; i++) {
            Node* b = vars->children[i];
            if (b->type == N_LIST && b->n_children >= 3) {
                compile_expr(c, b->children[2], 0);
                step_count++;
            }
        }
        for (int i = vars->n_children - 1; i >= 0; i--) {
            Node* b = vars->children[i];
            if (b->type == N_LIST && b->n_children >= 3) {
                int slot = resolve_local(c, b->children[0]->symbol);
                if (slot >= 0) chunk_emit(c, OP_SET_LOCAL, slot);
            }
        }

        chunk_emit(c, OP_LOOP, loop_top);
        patch(c, jexit2, OP_JUMP, c->code_len);
    }

    /* Pop locals */
    while (c->n_locals > 0 && c->locals[c->n_locals-1].depth == c->scope_depth)
        c->n_locals--;
    c->scope_depth--;
    return;
}

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
    chunk_free_arrays(&func);
    return;
}

static void compile_form_lambda_2(FuncChunk* c, Node* node, int tail) {
    Node* head = node->children[0];
    (void)head; (void)tail;
    Node* params = node->children[1];
    FuncChunk func; chunk_init_arrays(&func);
    func.enclosing = c;

    /* Check for dot notation: (x y . rest) */
    int has_rest = 0;
    int fixed_params = params->n_children;
    for (int i = 0; i < params->n_children; i++) {
        if (params->children[i]->type == N_SYMBOL && strcmp(params->children[i]->symbol, ".") == 0) {
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
    c->constants[cfunc].as.i = func_start;

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
        if (v == (int64_t)v && fabs(v) < 1e15)
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

            /* Search from the outermost scope inward */
            for (int d = depth - 1; d >= 1; d--) {
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
        fprintf(stderr, "WARNING: undefined variable '%s'\n", node->symbol);
        chunk_emit(c, OP_NIL, 0);
        return;
    }

    if (node->type != N_LIST || node->n_children == 0) { chunk_emit(c, OP_NIL, 0); return; }

    Node* head = node->children[0];

    /* ── Constant Folding ── */
    /* If all operands are compile-time constants, evaluate at compile time */
    if (node->type == N_LIST && node->n_children >= 3) {
        if (head->type == N_SYMBOL) {
            int all_const = 1;
            for (int i = 1; i < node->n_children; i++) {
                if (node->children[i]->type != N_NUMBER) { all_const = 0; break; }
            }
            if (all_const) {
                double result = 0;
                int folded = 0;
                if (strcmp(head->symbol, "+") == 0) {
                    result = 0;
                    for (int i = 1; i < node->n_children; i++) result += node->children[i]->numval;
                    folded = 1;
                } else if (strcmp(head->symbol, "-") == 0 && node->n_children >= 2) {
                    result = node->children[1]->numval;
                    for (int i = 2; i < node->n_children; i++) result -= node->children[i]->numval;
                    folded = 1;
                } else if (strcmp(head->symbol, "*") == 0) {
                    result = 1;
                    for (int i = 1; i < node->n_children; i++) result *= node->children[i]->numval;
                    folded = 1;
                } else if (strcmp(head->symbol, "/") == 0 && node->n_children == 3 && node->children[2]->numval != 0) {
                    result = node->children[1]->numval / node->children[2]->numval;
                    folded = 1;
                }
                if (folded) {
                    int ci = chunk_add_const(c, result == (int64_t)result && fabs(result) < 1e15
                        ? INT_VAL((int64_t)result) : FLOAT_VAL(result));
                    if (ci >= 0) chunk_emit(c, OP_CONST, ci);
                    return;
                }
            }
        }
    }

    /* (+ a b ...), (- a b), (* a b ...), (/ a b) */
    if (is_sym(head, "+")) {
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_ADD, 0); }
        return;
    }
    if (is_sym(head, "-")) {
        if (node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NEG, 0); return; }
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_SUB, 0); }
        return;
    }
    if (is_sym(head, "*")) {
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_MUL, 0); }
        return;
    }
    if (is_sym(head, "/")) {
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_DIV, 0); }
        return;
    }

    /* Comparisons — push proper booleans */
    if (is_sym(head, "=") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_EQ, 0); return; }
    if (is_sym(head, "<") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_LT, 0); return; }
    if (is_sym(head, ">") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_GT, 0); return; }
    if (is_sym(head, "<=") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_LE, 0); return; }
    if (is_sym(head, ">=") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_GE, 0); return; }
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
        chunk_emit(c, OP_NIL, 0);  /* push return value (void → NIL) */
        return;
    }
    /* Type predicates that need VM opcodes (not closures — these check types at opcode level) */
    if (is_sym(head, "integer?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NUM_P, 0); return; }

    /* abs and modulo are opcodes, not native calls — keep as special cases */
    if (is_sym(head, "abs") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_ABS, 0); return; }
    if (is_sym(head, "modulo") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_MOD, 0); return; }
    if (is_sym(head, "remainder") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_MOD, 0); return; }

    /* All other builtins (sin, cos, sqrt, even?, odd?, floor, ceiling, round, expt, min, max,
     * positive?, negative?, number->string, string-append, string=?, newline, length, etc.)
     * are first-class closures defined in the preamble. They resolve via normal variable lookup
     * and are called via the standard CALL mechanism. No special-casing needed. */

    /* Vector operations */
    if (is_sym(head, "vector")) {
        for (int i = 1; i < node->n_children; i++) compile_expr(c, node->children[i], 0);
        chunk_emit(c, OP_VEC_CREATE, node->n_children - 1);
        return;
    }
    if (is_sym(head, "make-vector") && node->n_children >= 2) {
        /* (make-vector n) or (make-vector n fill) — emit via NATIVE or direct */
        compile_expr(c, node->children[1], 0);
        if (node->n_children >= 3) compile_expr(c, node->children[2], 0);
        else chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
        /* make-vector: n and fill are on stack, dispatch to runtime native */
        chunk_emit(c, OP_NATIVE_CALL, 218);
        return;
    }
    if (is_sym(head, "vector-ref") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_VEC_REF, 0); return; }
    if (is_sym(head, "vector-set!") && node->n_children == 4) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); compile_expr(c, node->children[3], 0); chunk_emit(c, OP_VEC_SET, 0); return; }
    if (is_sym(head, "vector-length") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_VEC_LEN, 0); return; }

    /* Mutation */
    if (is_sym(head, "set-car!") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_SET_CAR, 0); return; }
    if (is_sym(head, "set-cdr!") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_SET_CDR, 0); return; }

    /* String operations via opcodes (these ARE opcodes, not native calls) */
    if (is_sym(head, "string-length") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_STR_LEN, 0);
        return;
    }
    if (is_sym(head, "string-ref") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
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
    if (is_sym(head, "substring") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 553);
        return;
    }
    if (is_sym(head, "string->number") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 563);
        return;
    }
    if (is_sym(head, "string-upcase") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 557);
        return;
    }
    if (is_sym(head, "string-downcase") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 558);
        return;
    }
    /* All other string operations (string=?, newline, etc.)
     * are first-class closures from the preamble. */

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
    /* first through fifth */
    if (is_sym(head, "first") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "second") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "third") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); return; }

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
    /* (provide name ...) — no-op: all symbols are visible */
    if (is_sym(head, "provide")) {
        return;
    }

    /* (define-syntax name (syntax-rules (literals...) (pattern template) ...)) */
    if (is_sym(head, "define-syntax") && node->n_children >= 3) {
        vm_macro_define_syntax((const MacroNode*)node);
        return;
    }

    /* (define-record-type name (constructor field...) pred (field accessor [mutator]) ...) */
    if (is_sym(head, "define-record-type") && node->n_children >= 4) { compile_form_define_record_type(c, node, tail); return; }

    /* (parameterize ((param1 val1) (param2 val2) ...) body ...) */
    if (is_sym(head, "parameterize") && node->n_children >= 3) { compile_form_parameterize(c, node, tail); return; }

    /* (let-values (((x y ...) producer) ...) body ...) */
    if (is_sym(head, "let-values") && node->n_children >= 3) { compile_form_let_values(c, node, tail); return; }

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
    if (is_sym(head, "values") && node->n_children >= 2) {
        if (node->n_children == 2) {
            compile_expr(c, node->children[1], tail);
        } else {
            for (int i = 1; i < node->n_children; i++)
                compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_VEC_CREATE, node->n_children - 1);
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
    if (is_sym(head, "delay") && node->n_children == 2) { compile_form_delay(c, node, tail); return; }

    /* (force promise) → force a promise (evaluate thunk if not yet forced) */
    if (is_sym(head, "force") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); /* push promise */
        chunk_emit(c, OP_NATIVE_CALL, 132);     /* native force */
        return;
    }

    /* (make-promise val) / (promise? x) */
    if (is_sym(head, "promise?") && node->n_children == 2) {
        /* A promise is a vector of length 2 with first element being bool */
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_VEC_P, 0); /* rough check: is it a vector? */
        return;
    }

    /* (atan y) or (atan y x) — 1 or 2 args */
    if (is_sym(head, "atan")) {
        if (node->n_children == 2) {
            compile_expr(c, node->children[1], 0);
            chunk_emit(c, OP_NATIVE_CALL, 31); /* 1-arg atan */
        } else if (node->n_children == 3) {
            compile_expr(c, node->children[1], 0);
            compile_expr(c, node->children[2], 0);
            chunk_emit(c, OP_NATIVE_CALL, 250); /* 2-arg atan2 */
        }
        return;
    }

    /* Variadic string-append: chain 2-arg NATIVE_CALL 54 calls */
    if (is_sym(head, "string-append") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_NATIVE_CALL, 554); /* 2-arg string-append */
        }
        return;
    }

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

    /* memq: identity-based list search */
    if (is_sym(head, "memq") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 139);
        return;
    }
    if (is_sym(head, "memv") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 137); /* same as member for our types */
        return;
    }
    if (is_sym(head, "assv") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 138); /* same as assoc for our types */
        return;
    }

    /* string-fill!: not commonly used but required for R7RS */
    if (is_sym(head, "string-fill!") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 556); /* reuse make-string pattern */
        return;
    }

    /* string-copy */
    if (is_sym(head, "string-copy") && node->n_children >= 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 566);
        return;
    }

    /* Equality predicates */
    if (is_sym(head, "eq?") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 133); /* eq?: identity/pointer equality */
        return;
    }
    if (is_sym(head, "eqv?") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 133); /* eqv? same as eq? for our types */
        return;
    }
    if (is_sym(head, "equal?") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 134); /* equal?: deep structural equality */
        return;
    }

    /* length is now a first-class closure from the preamble.
     * quotient can be defined in terms of floor and / as a preamble builtin too.
     * For now, keep quotient as a special case using opcodes. */
    if (is_sym(head, "quotient") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_DIV, 0);
        /* Floor the result */
        chunk_emit(c, OP_NATIVE_CALL, 26);
        return;
    }

    /* Pair operations */
    if (is_sym(head, "cons") && node->n_children == 3) {
        compile_expr(c, node->children[2], 0); /* cdr first (SOS) */
        compile_expr(c, node->children[1], 0); /* car second (TOS) */
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
            patch(c, jf, OP_JUMP_IF_FALSE, c->code_len);
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

    /***************************************************************************
     * Complex number builtins (native IDs 300-319)
     ***************************************************************************/
    if (is_sym(head, "make-rectangular") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 300);
        return;
    }
    if (is_sym(head, "make-polar") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 301);
        return;
    }
    if (is_sym(head, "real-part") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 302);
        return;
    }
    if (is_sym(head, "imag-part") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 303);
        return;
    }
    if (is_sym(head, "magnitude") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 304);
        return;
    }
    if (is_sym(head, "angle") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 305);
        return;
    }
    if (is_sym(head, "conjugate") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 306);
        return;
    }
    if (is_sym(head, "complex?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 317);
        return;
    }

    /***************************************************************************
     * Rational number builtins (native IDs 330-349)
     ***************************************************************************/
    if (is_sym(head, "numerator") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 331);
        return;
    }
    if (is_sym(head, "denominator") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 332);
        return;
    }
    /* Variadic max/min: chain 2-arg native calls */
    if (is_sym(head, "max") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_NATIVE_CALL, 34); /* 2-arg max */
        }
        return;
    }
    if (is_sym(head, "min") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_NATIVE_CALL, 33); /* 2-arg min */
        }
        return;
    }

    if (is_sym(head, "exact->inexact") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 213);
        return;
    }
    if (is_sym(head, "inexact->exact") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 214);
        return;
    }
    if (is_sym(head, "rationalize") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 345);
        return;
    }

    /***************************************************************************
     * Automatic differentiation builtins (native IDs 370-399)
     ***************************************************************************/
    if (is_sym(head, "make-dual") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 370);
        return;
    }
    if (is_sym(head, "dual-primal") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 371);
        return;
    }
    if (is_sym(head, "dual-tangent") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 372);
        return;
    }
    if (is_sym(head, "dual?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 383);
        return;
    }
    if (is_sym(head, "gradient") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 750);
        return;
    }
    if (is_sym(head, "derivative") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 393);
        return;
    }
    if (is_sym(head, "jacobian") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 751);
        return;
    }
    if (is_sym(head, "hessian") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 752);
        return;
    }

    /***************************************************************************
     * Tensor builtins (native IDs 410-469)
     ***************************************************************************/
    if (is_sym(head, "make-tensor") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 410);
        return;
    }
    if (is_sym(head, "tensor-shape") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 413);
        return;
    }
    if (is_sym(head, "tensor-reshape") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 414);
        return;
    }
    if (is_sym(head, "tensor-transpose") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 415);
        return;
    }
    if (is_sym(head, "zeros") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 417);
        return;
    }
    if (is_sym(head, "ones") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 418);
        return;
    }
    if (is_sym(head, "matmul") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 440);
        return;
    }
    if (is_sym(head, "softmax") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 463);
        return;
    }

    /***************************************************************************
     * Consciousness Engine builtins (native IDs 500-549)
     ***************************************************************************/
    if (is_sym(head, "logic-var?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 501);
        return;
    }
    if (is_sym(head, "unify") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 502);
        return;
    }
    if (is_sym(head, "walk") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 503);
        return;
    }
    if (is_sym(head, "make-substitution") && node->n_children == 1) {
        chunk_emit(c, OP_NATIVE_CALL, 505);
        return;
    }
    if (is_sym(head, "substitution?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 506);
        return;
    }
    if (is_sym(head, "make-fact") && node->n_children >= 2) {
        for (int i = 1; i < node->n_children; i++)
            compile_expr(c, node->children[i], 0);
        chunk_emit(c, OP_NATIVE_CALL, 507);
        return;
    }
    if (is_sym(head, "fact?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 508);
        return;
    }
    if (is_sym(head, "make-kb") && node->n_children == 1) {
        chunk_emit(c, OP_NATIVE_CALL, 509);
        return;
    }
    if (is_sym(head, "kb?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 510);
        return;
    }
    if (is_sym(head, "kb-assert!") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 511);
        return;
    }
    if (is_sym(head, "kb-query") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 512);
        return;
    }
    if (is_sym(head, "make-factor-graph") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 520);
        return;
    }
    if (is_sym(head, "factor-graph?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 521);
        return;
    }
    if (is_sym(head, "fg-add-factor!") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 522);
        return;
    }
    if (is_sym(head, "fg-infer!") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 523);
        return;
    }
    if (is_sym(head, "free-energy") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 525);
        return;
    }
    if (is_sym(head, "expected-free-energy") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 526);
        return;
    }
    if (is_sym(head, "make-workspace") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 540);
        return;
    }
    if (is_sym(head, "workspace?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 541);
        return;
    }
    if (is_sym(head, "ws-register!") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 542);
        return;
    }
    if (is_sym(head, "ws-step!") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 543);
        return;
    }

    /***************************************************************************
     * I/O builtins (native IDs 580-602)
     ***************************************************************************/
    if (is_sym(head, "open-input-file") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 580);
        return;
    }
    if (is_sym(head, "open-output-file") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 581);
        return;
    }
    if (is_sym(head, "close-port") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 582);
        return;
    }
    if (is_sym(head, "read-char") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 583);
        return;
    }
    if (is_sym(head, "read-line") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 585);
        return;
    }
    if (is_sym(head, "write-string") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 587);
        return;
    }
    if (is_sym(head, "eof-object?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 592);
        return;
    }
    if (is_sym(head, "open-input-string") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 596);
        return;
    }
    if (is_sym(head, "open-output-string") && node->n_children == 1) {
        chunk_emit(c, OP_NATIVE_CALL, 597);
        return;
    }
    if (is_sym(head, "get-output-string") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 598);
        return;
    }
    if (is_sym(head, "file-exists?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 599);
        return;
    }

    /***************************************************************************
     * Hash table builtins (native IDs 660-670)
     ***************************************************************************/
    if (is_sym(head, "make-hash-table") && node->n_children == 1) {
        chunk_emit(c, OP_NATIVE_CALL, 660);
        return;
    }
    if (is_sym(head, "hash-ref") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        if (node->n_children >= 4)
            compile_expr(c, node->children[3], 0);
        else
            chunk_emit(c, OP_NIL, 0); /* default */
        chunk_emit(c, OP_NATIVE_CALL, 661);
        return;
    }
    if (is_sym(head, "hash-set!") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 662);
        return;
    }
    if (is_sym(head, "hash-has-key?") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 663);
        return;
    }
    if (is_sym(head, "hash-remove!") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 664);
        return;
    }
    if (is_sym(head, "hash-keys") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 665);
        return;
    }
    if (is_sym(head, "hash-values") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 666);
        return;
    }
    if (is_sym(head, "hash-count") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 667);
        return;
    }
    if (is_sym(head, "hash-table?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 670);
        return;
    }

    /***************************************************************************
     * Error object builtins (native IDs 710-714)
     ***************************************************************************/
    if (is_sym(head, "error-object?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 711);
        return;
    }
    if (is_sym(head, "error-object-message") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 712);
        return;
    }
    if (is_sym(head, "error-object-irritants") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 713);
        return;
    }

    /***************************************************************************
     * Missing tensor ops
     ***************************************************************************/
    if (is_sym(head, "reshape") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0); /* tensor */
        /* Build shape list from remaining args */
        chunk_emit(c, OP_NIL, 0);
        for (int i = node->n_children - 1; i >= 2; i--) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_CONS, 0);
        }
        chunk_emit(c, OP_NATIVE_CALL, 414); /* reshape */
        return;
    }
    if (is_sym(head, "tensor-get") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0); /* tensor */
        chunk_emit(c, OP_NIL, 0);
        for (int i = node->n_children - 1; i >= 2; i--) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_CONS, 0);
        }
        chunk_emit(c, OP_NATIVE_CALL, 411); /* tensor-ref */
        return;
    }
    if (is_sym(head, "arange") && node->n_children >= 2) {
        for (int i = 1; i < node->n_children; i++)
            compile_expr(c, node->children[i], 0);
        /* Pad missing args: (arange n) → (arange n 0 1), (arange n m) → (arange n m 1) */
        if (node->n_children == 2) {
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(1)));
        }
        if (node->n_children == 3)
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(1)));
        chunk_emit(c, OP_NATIVE_CALL, 419);
        return;
    }

    /***************************************************************************
     * Missing neural net ops
     ***************************************************************************/
    if (is_sym(head, "relu") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 462);
        return;
    }
    if (is_sym(head, "sigmoid") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 464);
        return;
    }
    if (is_sym(head, "dropout") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 470); return; }
    if (is_sym(head, "conv2d") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 465);
        return;
    }
    if (is_sym(head, "batch-norm") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 464);
        return;
    }
    if (is_sym(head, "mse-loss") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 459);
        return;
    }
    if (is_sym(head, "cross-entropy-loss") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 460);
        return;
    }

    /***************************************************************************
     * Missing AD ops
     ***************************************************************************/
    if (is_sym(head, "divergence") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 395);
        return;
    }
    if (is_sym(head, "curl") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 396);
        return;
    }
    if (is_sym(head, "laplacian") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 397);
        return;
    }

    /***************************************************************************
     * Missing inference ops
     ***************************************************************************/
    if (is_sym(head, "fg-update-cpt!") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 524);
        return;
    }

    if (is_sym(head, "fg-observe!") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);  /* fg */
        compile_expr(c, node->children[2], 0);  /* var_id */
        compile_expr(c, node->children[3], 0);  /* observed_state */
        chunk_emit(c, OP_NATIVE_CALL, 527);
        return;
    }

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
        compile_expr(c, node->children[2], 0);
        Node* formals = node->children[1];
        if (formals->type == N_LIST) {
            add_local(c, formals->children[0]->symbol);
            for (int i = 1; i < formals->n_children; i++) {
                chunk_emit(c, OP_NIL, 0);
                add_local(c, formals->children[i]->symbol);
            }
        }
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
        FILE* incf = fopen(path, "r");
        if (incf) {
            fseek(incf, 0, SEEK_END); long len = ftell(incf); fseek(incf, 0, SEEK_SET);
            char* src = (char*)malloc(len + 1);
            if (src) {
                fread(src, 1, len, incf); src[len] = 0; fclose(incf);
                const char* saved = src_ptr; src_ptr = src;
                while (1) { skip_ws(); if (!*src_ptr) break; Node* e = parse_sexp(); if (!e) break; compile_expr(c, e, 0); free_node(e); }
                src_ptr = saved; free(src);
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
        for (int i = 1; i < node->n_children; i++)
            compile_expr(c, node->children[i], tail && i == node->n_children - 1);
        return;
    }

    /* -- define-type (type alias: compile-time only, no runtime effect) -- */
    if (is_sym(head, "define-type")) { return; }

    /***************************************************************************
     * Eshkol shorthand builtins
     ***************************************************************************/
    if (is_sym(head, "vref") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_VEC_REF, 0); return;
    }
    if (is_sym(head, "diff") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 393); return;
    }
    if (is_sym(head, "tensor") && node->n_children >= 2) {
        compile_expr(c, node->children[1], 0);
        if (node->n_children >= 3) compile_expr(c, node->children[2], 0);
        else chunk_emit(c, OP_CONST, chunk_add_const(c, FLOAT_VAL(0)));
        chunk_emit(c, OP_NATIVE_CALL, 410); return;
    }
    if (is_sym(head, "pow") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 32); return;
    }
    if (is_sym(head, "type-of") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 740); return;
    }
    if (is_sym(head, "sign") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 743); return;
    }

    /***************************************************************************
     * Missing type predicates
     ***************************************************************************/
    if (is_sym(head, "real?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NUM_P, 0); return;
    }
    if (is_sym(head, "rational?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 740); return;
    }
    if (is_sym(head, "tensor?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 740); return;
    }
    if (is_sym(head, "port?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 730); return;
    }
    if (is_sym(head, "input-port?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 728); return;
    }
    if (is_sym(head, "output-port?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 729); return;
    }

    /***************************************************************************
     * Missing math: cosh, sinh, tanh
     ***************************************************************************/
    if (is_sym(head, "cosh") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 720); return;
    }
    if (is_sym(head, "sinh") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 721); return;
    }
    if (is_sym(head, "tanh") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 722); return;
    }

    /***************************************************************************
     * Missing I/O: write-char, write-line, read
     ***************************************************************************/
    if (is_sym(head, "write-char") && node->n_children >= 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 586); return;
    }
    if (is_sym(head, "write-line") && node->n_children >= 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 726); return;
    }
    if (is_sym(head, "read") && node->n_children <= 2) {
        if (node->n_children == 2) compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 588); return;
    }

    /***************************************************************************
     * Missing tensor ops: tensor-ref, tensor-sum, tensor-mean, tensor-dot,
     * transpose, flatten, linspace, eye
     ***************************************************************************/
    if (is_sym(head, "tensor-ref") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NIL, 0);
        for (int i = node->n_children - 1; i >= 2; i--) {
            compile_expr(c, node->children[i], 0); chunk_emit(c, OP_CONS, 0);
        }
        chunk_emit(c, OP_NATIVE_CALL, 411); return;
    }
    if (is_sym(head, "tensor-sum") && node->n_children >= 2) {
        compile_expr(c, node->children[1], 0);
        if (node->n_children >= 3) compile_expr(c, node->children[2], 0);
        else chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
        chunk_emit(c, OP_NATIVE_CALL, 445); return;
    }
    if (is_sym(head, "tensor-mean") && node->n_children >= 2) {
        compile_expr(c, node->children[1], 0);
        if (node->n_children >= 3) compile_expr(c, node->children[2], 0);
        else chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
        chunk_emit(c, OP_NATIVE_CALL, 446); return;
    }
    if (is_sym(head, "tensor-dot") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 449); return;
    }
    if (is_sym(head, "transpose") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 415); return;
    }
    if (is_sym(head, "flatten") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 416); return;
    }
    if (is_sym(head, "linspace") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 746); return;
    }
    if (is_sym(head, "eye") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 745); return;
    }

    /***************************************************************************
     * Missing hash: hash-clear!
     ***************************************************************************/
    if (is_sym(head, "hash-clear!") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 668); return;
    }

    /***************************************************************************
     * gcd / lcm
     ***************************************************************************/
    if (is_sym(head, "gcd") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 224); return;
    }
    if (is_sym(head, "lcm") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 225); return;
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
