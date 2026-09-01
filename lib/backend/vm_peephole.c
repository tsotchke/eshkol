/**
 * @brief Run a fixed-point pass of local bytecode peephole optimizations
 *        over chunk @p c: eliminates `CONST 0 + ADD` and `CONST 1 + MUL`
 *        identities, `NOT + NOT` / `NEG + NEG` double-negation pairs, and
 *        `DUP + POP` pairs, replacing eliminated instructions with OP_NOP
 *        in place (never compacted, since compaction would require fixing
 *        up jump targets — the VM treats NOP as near-zero cost). Prints a
 *        summary of eliminated instruction count.
 */
static void peephole_optimize(FuncChunk* c) {
    /* ── Basic-block entry map ────────────────────────────────────────────
     * Every pattern below rewrites the PAIR (i, i+1) on the assumption that
     * the only way to reach i+1 is by falling through i.  If anything can
     * BRANCH TO i+1, the second instruction runs without the first, and
     * deleting both deletes an operation that was never an identity.
     *
     * That is a real silent-wrong, not a theoretical one.  `(+ x (if p a 0))`
     * compiles to
     *
     *     CONST x ; <test> ; JIF Lelse ; <a> ; JUMP Lend ; Lelse: CONST 0 ; Lend: ADD
     *
     * where `CONST 0` is the ELSE ARM's value and `ADD` is the OUTER `+`.
     * The unguarded "CONST 0 + ADD is an identity" rule matched that pair
     * across the Lend label and NOP'd both, so the outer addition vanished
     * whenever the THEN arm was taken: the VM printed 5 for
     * `(+ 1 (if #t 5 0))` where native prints 6, with exit 0 and no
     * diagnostic.  `CONST 1 + MUL` and `DUP + POP` have the same shape.
     *
     * Marking every branch target — and every closure entry PC, which the
     * compiler stores in the low 32 bits of the OP_CLOSURE constant — keeps
     * the rewrite inside one basic block.  Marking is conservative: an
     * unknown entry simply blocks an optimization. */
    unsigned char* is_entry = (unsigned char*)calloc((size_t)c->code_len + 1, 1);
    if (!is_entry) return;   /* out of memory: skip optimizing, never miscompile */
    for (int i = 0; i < c->code_len; i++) {
        int t = -1;
        switch (c->code[i].op) {
        case OP_JUMP: case OP_JUMP_IF_FALSE: case OP_LOOP: case OP_PUSH_HANDLER:
            t = c->code[i].operand;
            break;
        case OP_CLOSURE: {
            /* Body entry PC lives in the referenced constant's low 32 bits
             * (bits 32.. carry the packed arity). */
            int ci = c->code[i].operand & 0xFFFF;
            if (ci >= 0 && ci < c->n_constants && c->constants[ci].type == VAL_INT)
                t = (int)(int32_t)(c->constants[ci].as.i & 0xFFFFFFFF);
            break;
        }
        default: break;
        }
        if (t >= 0 && t <= c->code_len) is_entry[t] = 1;
    }
    /* Every compiled function body is entered at its recorded offset. */
    for (int e = 0; e < c->n_entries; e++) {
        int t = c->entries[e].code_offset;
        if (t >= 0 && t <= c->code_len) is_entry[t] = 1;
    }

    int changed = 1;
    while (changed) {
        changed = 0;
        for (int i = 0; i < c->code_len - 1; i++) {
            /* Never splice across a basic-block boundary (see above). */
            if (is_entry[i + 1]) continue;
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

    free(is_entry);

    /* Count eliminated NOPs for metrics */
    int n_nops = 0;
    for (int i = 0; i < c->code_len; i++) {
        if (c->code[i].op == OP_NOP) n_nops++;
    }
    /* Compiler diagnostic, not program output: it belongs on stderr and must
     * honour the same disassembly switch as the rest of the VM's chatter.
     * Printed on STDOUT it prefixed the program's own output, so
     * tests/parser/edge_cases_test.esk diverged from native at character 0 —
     * a parity failure caused purely by a debug line. */
#ifndef ESHKOL_VM_NO_DISASM
    if (n_nops > 0 && !getenv("ESHKOL_VM_NO_DISASM")) {
        fprintf(stderr, "  [peephole] eliminated %d instructions\n", n_nops);
    }
#else
    (void)n_nops;
#endif
    /* Note: we leave NOPs in place rather than compacting, because compacting
     * requires fixing all jump targets. The VM handles NOPs at near-zero cost. */
}
