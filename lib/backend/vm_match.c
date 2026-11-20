/* Match lowering for the source bytecode compiler. Every pattern takes a
 * subject slot and leaves one boolean. Clause bindings live in private slots
 * until the complete pattern has succeeded. */
/* Pending jumps form a chain through their operands. Only an emitted jump
 * joins the chain, so every link names an instruction inside the chunk. */
static void match_jump(FuncChunk* c, int* chain, uint8_t op) {
    int site = c->code_len;
    chunk_emit(c, op, *chain);
    if (c->code_len > site) *chain = site;
}

static void match_join(FuncChunk* c, int chain) {
    while (chain >= 0 && chain < c->code_len) {
        Instr jump = c->code[chain];
        patch(c, chain, jump.op, c->code_len);
        chain = jump.operand;
    }
}

static int match_binding(FuncChunk* c, int first, const char* name) {
    for (int i = first; i < c->n_locals; ++i)
        if (strcmp(c->locals[i].name, name) == 0) return i;
    return -1;
}

static void match_reserve_bindings(FuncChunk* c, Node* p, int first) {
    if (p->type == N_SYMBOL && !is_sym(p, "_")) {
        if (match_binding(c, first, p->symbol) < 0) {
            chunk_emit(c, OP_NIL, 0);
            add_temporary_local(c, p->symbol);
        }
    } else if (p->type == N_LIST && p->n_children) {
        Node* head = p->children[0];
        if (is_sym(head, "?") && p->n_children == 3)
            match_reserve_bindings(c, p->children[2], first);
        if (is_sym(head, "cons") || is_sym(head, "list") || is_sym(head, "or"))
            for (int i = 1; i < p->n_children; ++i)
                match_reserve_bindings(c, p->children[i], first);
    }
}

static void match_store_binding(FuncChunk* c, Node* name, int value, int first) {
    if (is_sym(name, "_")) return;
    int slot = match_binding(c, first, name->symbol);
    if (slot < 0) {
        vm_compile_error("match expects a binding identifier", NULL);
        return;
    }
    chunk_emit(c, OP_GET_LOCAL, value);
    chunk_emit(c, OP_SET_LOCAL, slot);
}

static void compile_match_pattern(FuncChunk* c, Node* p, int value, int first) {
    if (p->type == N_SYMBOL) {
        match_store_binding(c, p, value, first);
        chunk_emit(c, OP_TRUE, 0);
        return;
    }
    if (p->type != N_LIST || p->n_children == 0 || p->is_quote_sugar || p->is_numeric_literal) {
        chunk_emit(c, OP_GET_LOCAL, value);
        add_temporary_local(c, "");
        if (p->is_quote_sugar) compile_quote(c, p->children[1]);
        else compile_expr(c, p, 0);
        chunk_emit(c, OP_NATIVE_CALL, VM_NATIVE_MATCH_LITERAL);
        c->n_locals--;
        return;
    }
    Node* head = p->children[0];
    if (is_sym(head, "?") && (p->n_children == 2 || p->n_children == 3)) {
        Node arg = {.type = N_SYMBOL, .symbol = "match-subject",
                    .macro_value_owner = c->binding_env_id, .macro_value_slot = value};
        Node* parts[] = {p->children[1], &arg};
        Node call = {.type = N_LIST, .children = parts, .n_children = 2};
        compile_expr(c, &call, 0);
        if (p->n_children == 3) match_store_binding(c, p->children[2], value, first);
        return;
    }
    if (is_sym(head, "or")) {
        int success = -1;
        for (int i = 1; i < p->n_children; ++i) {
            compile_match_pattern(c, p->children[i], value, first);
            int next = placeholder(c);
            match_jump(c, &success, OP_JUMP);
            patch(c, next, OP_JUMP_IF_FALSE, c->code_len);
        }
        chunk_emit(c, OP_FALSE, 0);
        int end = placeholder(c);
        match_join(c, success);
        chunk_emit(c, OP_TRUE, 0);
        patch(c, end, OP_JUMP, c->code_len);
        return;
    }
    int pair = is_sym(head, "cons");
    if ((!pair && !is_sym(head, "list")) || (pair && p->n_children != 3)) {
        vm_compile_error("unknown or malformed match pattern", head->symbol);
        chunk_emit(c, OP_FALSE, 0);
        return;
    }

    /* Walk a proper list to its end, or split a single pair at its tail.
     * All failure edges have just the cursor left above the clause slots. */
    int failed = -1;
    chunk_emit(c, OP_GET_LOCAL, value);
    int cursor = add_temporary_local(c, "");
    for (int i = 1; i < p->n_children; ++i) {
        if (pair && i == 2) {
            compile_match_pattern(c, p->children[i], cursor, first);
        } else {
            chunk_emit(c, OP_GET_LOCAL, cursor);
            chunk_emit(c, OP_PAIR_P, 0);
            match_jump(c, &failed, OP_JUMP_IF_FALSE);
            chunk_emit(c, OP_GET_LOCAL, cursor);
            chunk_emit(c, OP_CAR, 0);
            int element = add_temporary_local(c, "");
            compile_match_pattern(c, p->children[i], element, first);
            chunk_emit(c, OP_POPN, 1);
            c->n_locals--;
        }
        match_jump(c, &failed, OP_JUMP_IF_FALSE);
        if (!pair || i == 1) {
            chunk_emit(c, OP_GET_LOCAL, cursor);
            chunk_emit(c, OP_CDR, 0);
            chunk_emit(c, OP_SET_LOCAL, cursor);
        }
    }
    if (pair) chunk_emit(c, OP_TRUE, 0);
    else {
        chunk_emit(c, OP_GET_LOCAL, cursor);
        chunk_emit(c, OP_NULL_P, 0);
    }
    int end = placeholder(c);
    match_join(c, failed);
    chunk_emit(c, OP_FALSE, 0);
    patch(c, end, OP_JUMP, c->code_len);
    chunk_emit(c, OP_POPN, 1);
    c->n_locals--;
}

static void compile_form_match(FuncChunk* c, Node* form, int tail) {
    if (form->n_children == 2) {
        fprintf(stderr, "WARNING: match requires at least one clause\n");
        chunk_emit(c, OP_NIL, 0);
        return;
    }
    int saved = c->n_locals;
    compile_expr(c, form->children[1], 0);
    int subject = add_temporary_local(c, "");
    int done = -1;
    int has_catchall = 0;
    c->scope_depth++;
    for (int i = 2; i < form->n_children; ++i) {
        Node* clause = form->children[i];
        if (clause->type != N_LIST || !clause->n_children) {
            vm_compile_error("match expects a clause with a pattern", NULL);
            break;
        }
        if (clause->children[0]->type == N_SYMBOL) has_catchall = 1;
        int first = c->n_locals;
        match_reserve_bindings(c, clause->children[0], first);
        int binding_count = c->n_locals - first;
        compile_match_pattern(c, clause->children[0], subject, first);
        int next = placeholder(c);
        for (int j = first; j < first + binding_count; ++j) {
            c->locals[j].compiler_temporary = 0;
            if (needs_local_boxing(clause->children + 1, clause->n_children - 1,
                                   c->locals[j].name)) {
                chunk_emit(c, OP_GET_LOCAL, j);
                chunk_emit(c, OP_VEC_CREATE, 1);
                chunk_emit(c, OP_SET_LOCAL, j);
                c->locals[j].boxed = 1;
            }
        }
        if (clause->n_children == 1) chunk_emit(c, OP_NIL, 0);
        vm_compile_scope_body(c, clause, 1, saved, tail);
        match_jump(c, &done, OP_JUMP);
        patch(c, next, OP_JUMP_IF_FALSE, c->code_len);
        for (int j = 0; j < binding_count; ++j) chunk_emit(c, OP_POP, 0);
        c->n_locals = first;
    }
    c->scope_depth--;
    c->n_locals = saved;
    chunk_emit(c, OP_POP, 0);
    if (has_catchall) {
        chunk_emit(c, OP_NIL, 0);
    } else {
        const char message[] = "match: no clause matched — non-exhaustive pattern match";
        fprintf(stderr, "WARNING: match expression has no catch-all clause (_ or else) — "
                        "will raise runtime error if no pattern matches\n");
        compile_packed_string_literal(c, message, sizeof message - 1);
        chunk_emit(c, OP_NATIVE_CALL, 237); /* error */
    }
    match_join(c, done);
}
