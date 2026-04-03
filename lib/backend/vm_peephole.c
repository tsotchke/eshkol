static void peephole_optimize(FuncChunk* c) {
    int changed = 1;
    while (changed) {
        changed = 0;
        for (int i = 0; i < c->code_len - 1; i++) {
            /* Pattern: CONST 0 + ADD → remove both (identity) */
            if (c->code[i].op == OP_CONST && c->code[i+1].op == OP_ADD) {
                Value v = c->constants[c->code[i].operand];
                if (v.type == VAL_INT && v.as.i == 0) {
                    c->code[i].op = OP_NOP; c->code[i].operand = 0;
                    c->code[i+1].op = OP_NOP; c->code[i+1].operand = 0;
                    changed = 1;
                }
            }
            /* Pattern: CONST 1 + MUL → remove both (identity) */
            if (c->code[i].op == OP_CONST && c->code[i+1].op == OP_MUL) {
                Value v = c->constants[c->code[i].operand];
                if (v.type == VAL_INT && v.as.i == 1) {
                    c->code[i].op = OP_NOP; c->code[i].operand = 0;
                    c->code[i+1].op = OP_NOP; c->code[i+1].operand = 0;
                    changed = 1;
                }
            }
            /* Pattern: CONST 0 + MUL → replace with CONST 0 (always zero) */
            if (c->code[i].op == OP_CONST && c->code[i+1].op == OP_MUL) {
                Value v = c->constants[c->code[i].operand];
                if (v.type == VAL_INT && v.as.i == 0) {
                    /* Drop the other operand, keep CONST 0 */
                    c->code[i+1].op = OP_NOP; c->code[i+1].operand = 0;
                    /* But we also need to drop the value below — this is tricky for a stack machine.
                     * Skip this optimization for safety. */
                    c->code[i+1].op = OP_MUL; /* undo */
                }
            }
            /* Pattern: NOT + NOT → remove both (double negation) */
            if (c->code[i].op == OP_NOT && c->code[i+1].op == OP_NOT) {
                c->code[i].op = OP_NOP; c->code[i].operand = 0;
                c->code[i+1].op = OP_NOP; c->code[i+1].operand = 0;
                changed = 1;
            }
            /* Pattern: NEG + NEG → remove both (double negation) */
            if (c->code[i].op == OP_NEG && c->code[i+1].op == OP_NEG) {
                c->code[i].op = OP_NOP; c->code[i].operand = 0;
                c->code[i+1].op = OP_NOP; c->code[i+1].operand = 0;
                changed = 1;
            }
            /* Pattern: DUP + POP → remove both */
            if (c->code[i].op == OP_DUP && c->code[i+1].op == OP_POP) {
                c->code[i].op = OP_NOP; c->code[i].operand = 0;
                c->code[i+1].op = OP_NOP; c->code[i+1].operand = 0;
                changed = 1;
            }
        }
    }

    /* Count eliminated NOPs for metrics */
    int n_nops = 0;
    for (int i = 0; i < c->code_len; i++) {
        if (c->code[i].op == OP_NOP) n_nops++;
    }
    if (n_nops > 0) {
        printf("  [peephole] eliminated %d instructions\n", n_nops);
    }
    /* Note: we leave NOPs in place rather than compacting, because compacting
     * requires fixing all jump targets. The VM handles NOPs at near-zero cost. */
}
