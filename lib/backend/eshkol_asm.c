/**
 * @file eshkol_asm.c
 * @brief Eshkol bytecode assembler — generates ESKB binary for weight matrix execution.
 *
 * Produces direct bytecode (no prelude, no closures) that the weight matrix
 * compiler can execute through actual W@x+b matrix multiplication.
 *
 * Usage: ./eshkol_asm [program_name]
 *   Programs: add, factorial, fibonacci, sum, complex
 *   Output: /tmp/eshkol_<name>.bc (ESKB format)
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Opcodes (match eshkol_compiler.c) */
enum {
    OP_NOP=0, OP_CONST=1, OP_NIL=2, OP_TRUE=3, OP_FALSE=4, OP_POP=5, OP_DUP=6,
    OP_ADD=7, OP_SUB=8, OP_MUL=9, OP_DIV=10, OP_MOD=11, OP_NEG=12, OP_ABS=13,
    OP_EQ=14, OP_LT=15, OP_GT=16, OP_LE=17, OP_GE=18, OP_NOT=19,
    OP_GET_LOCAL=20, OP_SET_LOCAL=21,
    OP_CALL=25, OP_RETURN=27,
    OP_JUMP=28, OP_JUMP_IF_FALSE=29, OP_LOOP=30,
    OP_CONS=31, OP_CAR=32, OP_CDR=33, OP_NULL_P=34,
    OP_PRINT=35, OP_HALT=36,
};

typedef struct { uint8_t op; int32_t operand; } Instr;

static void write_eskb(const char* path, const Instr* prog, int n_instr) {
    FILE* f = fopen(path, "wb");
    if (!f) { printf("ERROR: cannot open %s\n", path); return; }
    uint32_t magic = 0x45534B42; /* "ESKB" */
    uint32_t ni = n_instr, nc = 0;
    fwrite(&magic, 4, 1, f);
    fwrite(&ni, 4, 1, f);
    fwrite(&nc, 4, 1, f);  /* 0 constants — all values are immediate */
    for (int i = 0; i < n_instr; i++) {
        fwrite(&prog[i].op, 1, 1, f);
        fwrite(&prog[i].operand, 4, 1, f);
    }
    fclose(f);
    printf("Wrote %d instructions to %s\n", n_instr, path);
}

int main(int argc, char** argv) {
    const char* name = argc > 1 ? argv[1] : "all";
    printf("=== Eshkol Bytecode Assembler ===\n\n");

    /* (display (+ 3 5)) → 8 */
    if (strcmp(name, "add") == 0 || strcmp(name, "all") == 0) {
        Instr p[] = {
            {OP_CONST,3},{OP_CONST,5},{OP_ADD,0},{OP_PRINT,0},{OP_HALT,0}
        };
        write_eskb("/tmp/eshkol_add.bc", p, 5);
    }

    /* Iterative factorial(5) = 120 */
    if (strcmp(name, "factorial") == 0 || strcmp(name, "all") == 0) {
        Instr p[] = {
            {OP_CONST,1},{OP_SET_LOCAL,0},       /* result = 1 */
            {OP_CONST,5},{OP_SET_LOCAL,1},       /* counter = 5 */
            {OP_GET_LOCAL,1},{OP_JUMP_IF_FALSE,15},
            {OP_GET_LOCAL,0},{OP_GET_LOCAL,1},{OP_MUL,0},{OP_SET_LOCAL,0},
            {OP_GET_LOCAL,1},{OP_CONST,1},{OP_SUB,0},{OP_SET_LOCAL,1},
            {OP_JUMP,4},
            {OP_GET_LOCAL,0},{OP_PRINT,0},{OP_HALT,0},
        };
        write_eskb("/tmp/eshkol_factorial.bc", p, 18);
    }

    /* Iterative fibonacci(7) = 13 */
    if (strcmp(name, "fibonacci") == 0 || strcmp(name, "all") == 0) {
        Instr p[] = {
            {OP_CONST,0},{OP_SET_LOCAL,0},       /* a = 0 */
            {OP_CONST,1},{OP_SET_LOCAL,1},       /* b = 1 */
            {OP_CONST,7},{OP_SET_LOCAL,2},       /* n = 7 */
            {OP_GET_LOCAL,2},{OP_JUMP_IF_FALSE,19},
            {OP_GET_LOCAL,0},{OP_GET_LOCAL,1},{OP_ADD,0},
            {OP_GET_LOCAL,1},{OP_SET_LOCAL,0},
            {OP_SET_LOCAL,1},
            {OP_GET_LOCAL,2},{OP_CONST,1},{OP_SUB,0},{OP_SET_LOCAL,2},
            {OP_JUMP,6},
            {OP_GET_LOCAL,0},{OP_PRINT,0},{OP_HALT,0},
        };
        write_eskb("/tmp/eshkol_fibonacci.bc", p, 22);
    }

    /* Recursive factorial(5) = 120 via CALL/RETURN */
    if (strcmp(name, "rec_factorial") == 0 || strcmp(name, "all") == 0) {
        Instr p[] = {
            {OP_CONST,5},{OP_CONST,5},{OP_CALL,1},{OP_PRINT,0},{OP_HALT,0},
            /* fact (pc=5): */
            {OP_GET_LOCAL,0},{OP_CONST,0},{OP_EQ,0},{OP_JUMP_IF_FALSE,11},
            {OP_CONST,1},{OP_RETURN,0},
            {OP_GET_LOCAL,0},
            {OP_GET_LOCAL,0},{OP_CONST,1},{OP_SUB,0},{OP_CONST,5},{OP_CALL,1},
            {OP_MUL,0},{OP_RETURN,0},
        };
        write_eskb("/tmp/eshkol_rec_factorial.bc", p, 19);
    }

    /* Recursive fibonacci(7) = 13 via CALL/RETURN */
    if (strcmp(name, "rec_fibonacci") == 0 || strcmp(name, "all") == 0) {
        Instr p[] = {
            {OP_CONST,7},{OP_CONST,5},{OP_CALL,1},{OP_PRINT,0},{OP_HALT,0},
            /* fib (pc=5): */
            {OP_GET_LOCAL,0},{OP_CONST,1},{OP_LE,0},{OP_JUMP_IF_FALSE,11},
            {OP_GET_LOCAL,0},{OP_RETURN,0},
            {OP_GET_LOCAL,0},{OP_CONST,1},{OP_SUB,0},{OP_CONST,5},{OP_CALL,1},
            {OP_GET_LOCAL,0},{OP_CONST,2},{OP_SUB,0},{OP_CONST,5},{OP_CALL,1},
            {OP_ADD,0},{OP_RETURN,0},
        };
        write_eskb("/tmp/eshkol_rec_fibonacci.bc", p, 23);
    }

    /* Complex: (car (cons (+ 10 20) 40)) = 30 */
    if (strcmp(name, "complex") == 0 || strcmp(name, "all") == 0) {
        Instr p[] = {
            {OP_CONST,10},{OP_CONST,20},{OP_ADD,0},   /* 30 */
            {OP_CONST,40},{OP_CONS,0},                 /* (cons 30 40) */
            {OP_CAR,0},{OP_PRINT,0},{OP_HALT,0},      /* car → 30 */
        };
        write_eskb("/tmp/eshkol_complex.bc", p, 8);
    }

    printf("\nDone. Run with: ESHKOL_BC=/tmp/eshkol_<name>.bc /tmp/weight_matrices_v3\n");
    return 0;
}
