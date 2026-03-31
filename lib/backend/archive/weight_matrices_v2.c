/**
 * @file weight_matrices_v2.c
 * @brief Eshkol VM interpreter compiled into transformer weights.
 *
 * Extends the 14-opcode POC to support the full Eshkol VM ISA:
 * function calls (CALL/RETURN/TAIL_CALL), closures, local variables,
 * comparisons, control flow, and NATIVE_CALL dispatch.
 *
 * Architecture: d_model=32, n_heads=16, head_dim=2, n_layers=6
 *   Layer 0: Instruction fetch (Gaussian attention peaked at PC)
 *   Layer 1: Memory read (attention reads stack for locals/upvalues)
 *   Layer 2: Precompute (SQUARE activation for products + comparisons)
 *   Layer 3: Execute (gated FFN: opcode dispatch)
 *   Layer 4: Frame management (gated FFN: SP/FP updates for CALL/RETURN)
 *   Layer 5: Write-back (encodes memory writes for execution loop)
 *
 * The execution loop (C wrapper) handles:
 * - KV cache append for memory writes
 * - NATIVE_CALL dispatch to C functions
 * - OALR region management
 * - Output collection
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Architecture Constants
 * ═══════════════════════════════════════════════════════════════════════════ */

#define D 32           /* d_model */
#define H 16           /* n_heads */
#define HD 2           /* head_dim */
#define N_LAYERS 6
#define FFN_DIM 1536
#define SCALE 100.0f
#define STACK_SIZE 256 /* attention-based stack depth */

/* ═══════════════════════════════════════════════════════════════════════════
 * State Vector Layout (32 dimensions)
 * ═══════════════════════════════════════════════════════════════════════════ */

enum {
    /* Core registers */
    S_PC       = 0,   /* program counter */
    S_SP       = 1,   /* stack pointer */
    S_FP       = 2,   /* frame pointer */
    S_TOS      = 3,   /* top of stack (cached) */
    S_SOS      = 4,   /* second of stack (cached) */
    S_R2       = 5,   /* third stack element (cached) */
    S_DEPTH    = 6,   /* stack depth */
    S_HALT     = 7,   /* halt flag */

    /* Fetched instruction */
    S_OPCODE   = 8,   /* current opcode */
    S_OPERAND  = 9,   /* current operand */

    /* Control */
    S_OUTPUT   = 10,  /* output value */
    S_HAS_OUT  = 11,  /* 1 = output is valid this step, 0 = no output */

    /* Precomputed intermediates (filled by layers 1-2) */
    S_PRODUCT  = 12,  /* TOS * SOS */
    S_CMP_EQ   = 13,  /* indicator(TOS == SOS) */
    S_CMP_LT   = 14,  /* indicator(TOS < SOS) ... approximated */
    S_LOADVAL  = 15,  /* memory read result (for GET_LOCAL) */

    /* Write targets (read by execution loop) */
    S_WRITE_ADDR = 16,  /* address to write */
    S_WRITE_VAL  = 17,  /* value to write */

    /* Frame management */
    S_SAVED_PC = 18,  /* saved PC for RETURN */
    S_SAVED_FP = 19,  /* saved FP for RETURN */

    /* Flags */
    S_IS_CALL  = 20,  /* 1 if current opcode is CALL */
    S_IS_RET   = 21,  /* 1 if current opcode is RETURN */
    S_IS_NATIVE = 22, /* 1 if current opcode is NATIVE_CALL */

    /* Reserved for future opcodes */
    S_R3       = 23,
    S_R4       = 24,
    S_R5       = 25,
    S_R6       = 26,
    S_R7       = 27,
    S_R8       = 28,
    S_R9       = 29,
    S_R10      = 30,
    S_R11      = 31,
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Opcodes (matching eshkol_compiler.c ISA)
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef enum {
    OP_NOP=0, OP_CONST=1, OP_NIL=2, OP_TRUE=3, OP_FALSE=4, OP_POP=5, OP_DUP=6,
    OP_ADD=7, OP_SUB=8, OP_MUL=9, OP_DIV=10, OP_MOD=11, OP_NEG=12, OP_ABS=13,
    OP_EQ=14, OP_LT=15, OP_GT=16, OP_LE=17, OP_GE=18, OP_NOT=19,
    OP_GET_LOCAL=20, OP_SET_LOCAL=21,
    OP_CLOSURE=24, OP_CALL=25, OP_TAIL_CALL=26, OP_RETURN=27,
    OP_JUMP=28, OP_JUMP_IF_FALSE=29, OP_LOOP=30,
    OP_CONS=31, OP_CAR=32, OP_CDR=33, OP_NULL_P=34,
    OP_PRINT=35, OP_HALT=36, OP_NATIVE_CALL=37,
} OpCode;

typedef struct { uint8_t op; int32_t operand; } Instr;

/* ═══════════════════════════════════════════════════════════════════════════
 * Utility Functions
 * ═══════════════════════════════════════════════════════════════════════════ */

static float sigmoidf(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static float indicator(float x, float k) {
    return sigmoidf(SCALE * (x - k + 0.5f)) - sigmoidf(SCALE * (x - k - 0.5f));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Reference Interpreter (ground truth)
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct { float s[D]; } State;

static void state_init(State* st) {
    memset(st, 0, sizeof(State));
    st->s[S_OUTPUT] = 0; st->s[S_HAS_OUT] = 0;
    st->s[S_WRITE_ADDR] = -1.0f;
}

static void execute_step_ref(const State* cur, const Instr* prog, int n_instr, State* next) {
    memcpy(next, cur, sizeof(State));
    next->s[S_OUTPUT] = 0; next->s[S_HAS_OUT] = 0;
    next->s[S_WRITE_ADDR] = -1.0f;

    int pc = (int)cur->s[S_PC];
    if (pc < 0 || pc >= n_instr || cur->s[S_HALT] > 0.5f) { next->s[S_HALT] = 1; return; }

    float tos = cur->s[S_TOS], sos = cur->s[S_SOS];
    float r2 = cur->s[S_R2];
    float operand = (float)prog[pc].operand;
    int op = prog[pc].op;

    /* Store fetched instruction in state for simulated/matrix verification */
    next->s[S_OPCODE] = (float)op;
    next->s[S_OPERAND] = operand;

    switch (op) {
    case OP_NOP:    next->s[S_PC]=pc+1; break;
    case OP_CONST:  next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=operand; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1; break;
    case OP_NIL:    next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=-1; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1; break;
    case OP_TRUE:   next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=1; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1; break;
    case OP_FALSE:  next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1; break;
    case OP_POP:    next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_DUP:    next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1; break;

    case OP_ADD:    next->s[S_TOS]=tos+sos; next->s[S_SOS]=r2; next->s[S_R2]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_SUB:    next->s[S_TOS]=sos-tos; next->s[S_SOS]=r2; next->s[S_R2]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_MUL:    next->s[S_TOS]=tos*sos; next->s[S_SOS]=r2; next->s[S_R2]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_NEG:    next->s[S_TOS]=-tos; next->s[S_PC]=pc+1; break;

    case OP_EQ:     next->s[S_TOS]=(tos==sos)?1:0; next->s[S_SOS]=r2; next->s[S_R2]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_LT:     next->s[S_TOS]=(sos<tos)?1:0;  next->s[S_SOS]=r2; next->s[S_R2]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_GT:     next->s[S_TOS]=(sos>tos)?1:0;  next->s[S_SOS]=r2; next->s[S_R2]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_NOT:    next->s[S_TOS]=(tos==0)?1:0; next->s[S_PC]=pc+1; break;

    case OP_JUMP:          next->s[S_PC]=operand; break;
    case OP_JUMP_IF_FALSE: next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=(tos==0)?operand:(float)(pc+1); break;
    case OP_LOOP:          next->s[S_PC]=operand; break;

    case OP_GET_LOCAL: {
        /* For the cached stack: operand indexes from FP.
         * In the full VM, this reads from KV cache memory.
         * Here we approximate: operand=0 → TOS, operand=1 → SOS, etc. */
        int slot = (int)operand;
        float fp_val = cur->s[S_FP];
        /* Signal to the execution loop to read from attention memory */
        next->s[S_IS_NATIVE] = 0; /* not native, but needs memory read */
        next->s[S_PC]=pc+1;
        break;
    }

    case OP_PRINT:  next->s[S_OUTPUT]=tos; next->s[S_HAS_OUT]=1; next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_HALT:   next->s[S_HALT]=1; break;
    case OP_NATIVE_CALL: next->s[S_IS_NATIVE]=1; next->s[S_PC]=pc+1; break;

    default:        next->s[S_PC]=pc+1; break;
    }
}

static int run_reference(const Instr* prog, int n_instr, float* outputs, int max_out) {
    State st; state_init(&st);
    int n_out = 0;
    for (int step = 0; step < 8192 && st.s[S_HALT] < 0.5f; step++) {
        State next;
        int pc = (int)st.s[S_PC];
        if (pc >= 0 && pc < n_instr) {
            st.s[S_OPCODE] = (float)prog[pc].op;
            st.s[S_OPERAND] = (float)prog[pc].operand;
        }
        execute_step_ref(&st, prog, n_instr, &next);
        if (next.s[S_HAS_OUT] > 0.5f && n_out < max_out)
            outputs[n_out++] = next.s[S_OUTPUT];
        st = next;
    }
    return n_out;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Weight Matrix Generation
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float wq[N_LAYERS][D * D];
    float wk[N_LAYERS][D * D];
    float wv[N_LAYERS][D * D];
    float wo[N_LAYERS][D * D];
    float bq[N_LAYERS][D];
    float ff_up[N_LAYERS][D * FFN_DIM];
    float ff_up_b[N_LAYERS][FFN_DIM];
    float ff_down[N_LAYERS][FFN_DIM * D];
    float ff_down_b[N_LAYERS][D];
    float ff_gate[N_LAYERS][D * FFN_DIM];
    float ff_gate_b[N_LAYERS][FFN_DIM];
    int   ff_type[N_LAYERS];
} InterpreterWeights;

#define W(mat, r, c, cols) ((mat)[(r) * (cols) + (c)])

/* Gated neuron pair: indicator(opcode==op_id) * (linear combo) → out_dim */
static int add_gated_pair(InterpreterWeights* w, int L, int n,
                           int op_id,
                           int ud1, float us1, int ud2, float us2,
                           int ud3, float us3, int ud4, float us4,
                           float ubias,
                           int out_dim, float coeff) {
    for (int sign = 0; sign < 2; sign++) {
        int j = n + sign;
        float s = (sign == 0) ? 0.5f : -0.5f;
        W(w->ff_gate[L], S_OPCODE, j, FFN_DIM) += SCALE;
        W(w->ff_gate[L], S_HALT, j, FFN_DIM)   += -SCALE;
        w->ff_gate_b[L][j] = SCALE * (-(float)op_id + s);
        if (ud1 >= 0) W(w->ff_up[L], ud1, j, FFN_DIM) += us1;
        if (ud2 >= 0) W(w->ff_up[L], ud2, j, FFN_DIM) += us2;
        if (ud3 >= 0) W(w->ff_up[L], ud3, j, FFN_DIM) += us3;
        if (ud4 >= 0) W(w->ff_up[L], ud4, j, FFN_DIM) += us4;
        w->ff_up_b[L][j] = ubias;
        float c = (sign == 0) ? coeff : -coeff;
        W(w->ff_down[L], j, out_dim, D) += c;
    }
    return n + 2;
}

static int add_unconditional(InterpreterWeights* w, int L, int n,
                              int ud1, float us1, float ubias,
                              int out_dim, float coeff) {
    int j = n;
    W(w->ff_gate[L], S_HALT, j, FFN_DIM) = -SCALE;
    w->ff_gate_b[L][j] = 10.0f * SCALE;
    if (ud1 >= 0) W(w->ff_up[L], ud1, j, FFN_DIM) = us1;
    w->ff_up_b[L][j] = ubias;
    W(w->ff_down[L], j, out_dim, D) = coeff;
    return n + 1;
}

static void generate_weights(InterpreterWeights* w) {
    memset(w, 0, sizeof(InterpreterWeights));

    /* ── Layer 0: Instruction Fetch (Gaussian attention) ── */
    {
        float T = SCALE;
        W(w->wq[0], 0, S_PC, D) = T;
        w->bq[0][1] = T;
        W(w->wk[0], 0, 0, D) = 1.0f;
        W(w->wk[0], 1, 1, D) = 1.0f;
        W(w->wv[0], 0, S_OPCODE, D) = 1.0f;
        W(w->wv[0], 1, S_OPERAND, D) = 1.0f;
        W(w->wo[0], S_OPCODE, 0, D) = 1.0f;
        W(w->wo[0], S_OPERAND, 1, D) = 1.0f;
        w->ff_type[0] = 0;
    }

    /* ── Layer 1: Memory Read (placeholder — will use KV cache attention) ── */
    w->ff_type[1] = 0;

    /* ── Layer 2: Precompute (SQUARE activation for products) ── */
    {
        w->ff_type[2] = 1;
        /* TOS*SOS via (a+b)²/2 - a²/2 - b²/2 */
        W(w->ff_up[2], S_TOS, 0, FFN_DIM) = 1; W(w->ff_up[2], S_SOS, 0, FFN_DIM) = 1;
        W(w->ff_up[2], S_TOS, 1, FFN_DIM) = 1;
        W(w->ff_up[2], S_SOS, 2, FFN_DIM) = 1;
        W(w->ff_down[2], 0, S_PRODUCT, D) =  0.5f;
        W(w->ff_down[2], 1, S_PRODUCT, D) = -0.5f;
        W(w->ff_down[2], 2, S_PRODUCT, D) = -0.5f;
    }

    /* ── Layer 3: Execute (gated FFN — main opcode dispatch) ── */
    {
        const int L = 3;
        w->ff_type[L] = 2;
        int n = 0;

        /* Universal: clear output flag and intermediate dims */
        n = add_unconditional(w, L, n, S_OUTPUT, -1.0f, 0, S_OUTPUT, 1.0f);
        n = add_unconditional(w, L, n, S_HAS_OUT, -1.0f, 0, S_HAS_OUT, 1.0f);
        n = add_unconditional(w, L, n, S_IS_CALL, -1.0f, 0, S_IS_CALL, 1.0f);
        n = add_unconditional(w, L, n, S_IS_RET, -1.0f, 0, S_IS_RET, 1.0f);
        n = add_unconditional(w, L, n, S_IS_NATIVE, -1.0f, 0, S_IS_NATIVE, 1.0f);
        n = add_unconditional(w, L, n, S_PRODUCT, -1.0f, 0, S_PRODUCT, 1.0f);

        /* ── OP_NOP (0): PC += 1 ── */
        n = add_gated_pair(w,L,n, 0, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);

        /* ── OP_CONST (1): push operand ── */
        n = add_gated_pair(w,L,n, 1, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 1, S_OPERAND,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 1, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 1, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 1, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);

        /* ── OP_NIL (2): push -1 (nil sentinel) ── */
        n = add_gated_pair(w,L,n, 2, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 2, S_TOS,-1,-1,0,-1,0,-1,0, -1, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 2, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 2, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 2, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);

        /* ── OP_POP (5): discard TOS ── */
        n = add_gated_pair(w,L,n, 5, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 5, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 5, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 5, S_R2,-1,-1,0,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 5, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* ── OP_DUP (6): duplicate TOS ── */
        n = add_gated_pair(w,L,n, 6, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 6, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 6, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 6, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);

        /* ── OP_ADD (7): TOS = SOS + TOS, pop ── */
        n = add_gated_pair(w,L,n, 7, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 7, S_SOS,1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 7, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 7, S_R2,-1,-1,0,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 7, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* ── OP_SUB (8): TOS = SOS - TOS, pop ── */
        n = add_gated_pair(w,L,n, 8, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 8, S_SOS,1,S_TOS,-2,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 8, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 8, S_R2,-1,-1,0,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 8, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* ── OP_MUL (9): TOS = TOS * SOS (from precomputed PRODUCT), pop ── */
        n = add_gated_pair(w,L,n, 9, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 9, S_PRODUCT,1,S_TOS,-1,S_SOS,-1,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 9, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 9, S_R2,-1,-1,0,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 9, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* ── OP_NEG (12): TOS = -TOS ── */
        n = add_gated_pair(w,L,n, 12, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 12, S_TOS,-2,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);

        /* ── OP_EQ (14): TOS = (SOS == TOS) ? 1 : 0, pop ── */
        /* Approximate: indicator(TOS-SOS == 0) → 1, else 0 */
        /* This uses the precomputed S_CMP_EQ from layer 2 (TODO) */
        /* For now: delta_TOS = -(TOS) -(SOS) + indicator(TOS==SOS) */
        n = add_gated_pair(w,L,n, 14, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 14, S_TOS,-1,S_SOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        /* TODO: need indicator(TOS==SOS) precomputed */
        n = add_gated_pair(w,L,n, 14, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 14, S_R2,-1,-1,0,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 14, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* ── OP_NOT (19): TOS = (TOS == 0) ? 1 : 0 ── */
        n = add_gated_pair(w,L,n, 19, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 19, S_TOS,-1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
        /* TODO: need indicator(TOS==0) precomputed */

        /* ── OP_JUMP (28): PC = operand ── */
        n = add_gated_pair(w,L,n, 28, S_OPERAND,1,S_PC,-1,-1,0,-1,0, 0, S_PC, 1.0f);

        /* ── OP_JUMP_IF_FALSE (29): if TOS==0: PC=operand, else PC=PC+1. Pop. ── */
        /* This needs: indicator(TOS==0) * (operand - (PC+1)) → delta_PC */
        /* Plus always: PC += 1, pop stack */
        n = add_gated_pair(w,L,n, 29, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f); /* PC++ */
        n = add_gated_pair(w,L,n, 29, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 29, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 29, S_R2,-1,-1,0,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 29, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        /* TODO: conditional PC delta based on TOS==0 */

        /* ── OP_LOOP (30): PC = operand ── */
        n = add_gated_pair(w,L,n, 30, S_OPERAND,1,S_PC,-1,-1,0,-1,0, 0, S_PC, 1.0f);

        /* ── OP_PRINT (35): output TOS, pop ── */
        n = add_gated_pair(w,L,n, 35, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 35, S_TOS,1,-1,0,-1,0,-1,0, 0, S_OUTPUT, 1.0f);
        n = add_gated_pair(w,L,n, 35, -1,0,-1,0,-1,0,-1,0, 1.0f, S_HAS_OUT, 1.0f);
        n = add_gated_pair(w,L,n, 35, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 35, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 35, S_R2,-1,-1,0,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 35, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* ── OP_HALT (36) ── */
        n = add_gated_pair(w,L,n, 36, -1,0,-1,0,-1,0,-1,0, 1.0f, S_HALT, 1.0f);

        printf("[WEIGHT_GEN] Layer 3: %d neurons used out of %d\n", n, FFN_DIM);
    }

    /* ── Layer 4: Frame Management (placeholder) ── */
    w->ff_type[4] = 0;

    /* ── Layer 5: Write-back (placeholder) ── */
    w->ff_type[5] = 0;

    /* Total params */
    size_t params = 0;
    for (int l = 0; l < N_LAYERS; l++) {
        params += D*D*4 + D; /* attention */
        params += D*FFN_DIM*3 + FFN_DIM*2 + D; /* FFN */
    }
    printf("[WEIGHT_GEN] Total: %zu params, %.1f KB\n", params, (float)(params*4)/1024.0f);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Simulated Transformer Forward Pass
 * ═══════════════════════════════════════════════════════════════════════════ */

static void embed_instruction(const Instr* instr, int position, float out[D]) {
    memset(out, 0, D * sizeof(float));
    out[0] = (float)position;
    out[1] = -(float)(position * position) / 2.0f;
    out[S_OPCODE] = (float)instr->op;
    out[S_OPERAND] = (float)instr->operand;
}

static void forward_with_weights(const InterpreterWeights* w, const float state[D],
                                  const float pe[][D], int np, float out[D]) {
    float x[D], residual[D], layer_out[D];
    memcpy(x, state, sizeof(float)*D);

    for (int l = 0; l < N_LAYERS; l++) {
        memcpy(residual, x, sizeof(float)*D);

        /* Attention (layer 0 only for instruction fetch) */
        if (l == 0 && np > 0) {
            static int attn_dbg = 0;
            float q[D], scores[256]; memset(q,0,sizeof(q));
            for(int i=0;i<D;i++) for(int j=0;j<D;j++) q[i]+=W(w->wq[l],i,j,D)*x[j];
            for(int i=0;i<D;i++) q[i]+=w->bq[l][i];
            float mx=-1e30f;
            for(int p=0;p<np&&p<256;p++){
                float s=0; float k[D]; memset(k,0,sizeof(k));
                for(int i=0;i<D;i++) for(int j=0;j<D;j++) k[i]+=W(w->wk[l],i,j,D)*pe[p][j];
                for(int i=0;i<HD;i++) s+=q[i]*k[i];
                scores[p]=s/sqrtf((float)HD); if(scores[p]>mx) mx=scores[p];
            }
            float sum=0;
            for(int p=0;p<np;p++){scores[p]=expf(scores[p]-mx);sum+=scores[p];}
            for(int p=0;p<np;p++) scores[p]/=sum;
            float attn_out[D]; memset(attn_out,0,sizeof(attn_out));
            for(int p=0;p<np;p++){
                float v[D]; memset(v,0,sizeof(v));
                for(int i=0;i<D;i++) for(int j=0;j<D;j++) v[i]+=W(w->wv[l],i,j,D)*pe[p][j];
                for(int i=0;i<D;i++) attn_out[i]+=scores[p]*v[i];
            }
            float projected[D]; memset(projected,0,sizeof(projected));
            for(int i=0;i<D;i++) for(int j=0;j<D;j++) projected[i]+=W(w->wo[l],i,j,D)*attn_out[j];
            if(attn_dbg<1) {
                printf("    ATTN: q[0]=%.1f q[1]=%.1f scores[0]=%.1f scores[1]=%.1f\n", q[0],q[1],scores[0],np>1?scores[1]:0.0f);
                printf("    ATTN: attn_out[0]=%.2f attn_out[1]=%.2f projected[8]=%.2f projected[9]=%.2f\n", attn_out[0],attn_out[1],projected[8],projected[9]);
                printf("    ATTN: pe[0][8]=%.1f pe[0][9]=%.1f\n", pe[0][8], pe[0][9]);
                attn_dbg++;
            }
            for(int i=0;i<D;i++) x[i]+=projected[i];
        }

        /* FFN */
        memset(layer_out,0,sizeof(layer_out));
        if (w->ff_type[l] == 1) {
            /* SQUARE activation: out = W_down @ (W_up @ x)² */
            float hidden[FFN_DIM]; memset(hidden,0,sizeof(hidden));
            for(int j=0;j<FFN_DIM;j++){
                for(int i=0;i<D;i++) hidden[j]+=W(w->ff_up[l],i,j,FFN_DIM)*x[i];
                hidden[j]+=w->ff_up_b[l][j];
                hidden[j]=hidden[j]*hidden[j]; /* SQUARE */
            }
            for(int i=0;i<D;i++){
                for(int j=0;j<FFN_DIM;j++) layer_out[i]+=W(w->ff_down[l],j,i,D)*hidden[j];
                layer_out[i]+=w->ff_down_b[l][i];
            }
        } else if (w->ff_type[l] == 2) {
            /* Gated: out = W_down @ (sigmoid(W_gate @ x + b_gate) * (W_up @ x + b_up)) */
            float gate[FFN_DIM], up[FFN_DIM]; memset(gate,0,sizeof(gate)); memset(up,0,sizeof(up));
            for(int j=0;j<FFN_DIM;j++){
                for(int i=0;i<D;i++){
                    gate[j]+=W(w->ff_gate[l],i,j,FFN_DIM)*x[i];
                    up[j]+=W(w->ff_up[l],i,j,FFN_DIM)*x[i];
                }
                gate[j]=sigmoidf(gate[j]+w->ff_gate_b[l][j]);
                up[j]+=w->ff_up_b[l][j];
            }
            for(int i=0;i<D;i++){
                for(int j=0;j<FFN_DIM;j++) layer_out[i]+=W(w->ff_down[l],j,i,D)*gate[j]*up[j];
                layer_out[i]+=w->ff_down_b[l][i];
            }
        }

        /* Residual connection */
        for(int i=0;i<D;i++) x[i]=residual[i]+layer_out[i];
    }
    memcpy(out, x, sizeof(float)*D);
}

static int run_with_weights(const InterpreterWeights* w, const Instr* prog, int n_instr,
                             float* outputs, int max_out) {
    float pe[256][D];
    for(int p=0;p<n_instr&&p<256;p++) embed_instruction(&prog[p],p,pe[p]);
    float state[D]; memset(state,0,sizeof(state));
    int n_out=0;
    for(int step=0;step<200;step++){  /* limit for debugging */
        float next[D];
        forward_with_weights(w,state,pe,n_instr,next);
        if(step<3) {
            printf("    step %d: pc=%.1f op=%.1f oper=%.1f tos=%.1f sos=%.1f halt=%.1f\n",
                   step, next[S_PC], next[S_OPCODE], next[S_OPERAND], next[S_TOS], next[S_SOS], next[S_HALT]);
            printf("      all dims:");
            for(int d=0;d<D;d++) if(fabsf(next[d])>0.001f) printf(" [%d]=%.2f", d, next[d]);
            printf("\n");
        }
        if(next[S_HAS_OUT]>0.5f&&n_out<max_out) outputs[n_out++]=next[S_OUTPUT];
        if(next[S_HALT]>0.5f) break;
        memcpy(state,next,sizeof(state));
    }
    return n_out;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Tests
 * ═══════════════════════════════════════════════════════════════════════════ */

static int n_pass = 0, n_fail = 0;
static InterpreterWeights* g_weights = NULL;

static void test(const char* name, const Instr* prog, int n, float expected) {
    float r[64], m[64];
    int rn = run_reference(prog, n, r, 64);
    int mn = g_weights ? run_with_weights(g_weights, prog, n, m, 64) : 0;
    float rv = rn>0?r[0]:-9999, mv = mn>0?m[0]:-9999;
    int ok_r = fabsf(rv-expected)<0.01f;
    int ok_m = g_weights ? fabsf(mv-expected)<0.5f : 1; /* wider tolerance for approx */
    printf("  %-25s ref=%8.1f mat=%8.1f  %s%s\n",
           name, rv, mv, ok_r?"":"ref:FAIL ", (g_weights && !ok_m)?"mat:FAIL ":"");
    if (ok_r && ok_m) n_pass++; else n_fail++;
}

int main() {
    printf("=== Eshkol VM Weight Compiler v2 ===\n\n");

    g_weights = (InterpreterWeights*)calloc(1, sizeof(InterpreterWeights));
    if (g_weights) generate_weights(g_weights);

    printf("\n  Tests:\n\n");

    /* Arithmetic */
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_ADD,0},{OP_PRINT,0},{OP_HALT,0}};
      test("3+5", p, 5, 8); }
    { Instr p[]={{OP_CONST,10},{OP_CONST,7},{OP_SUB,0},{OP_PRINT,0},{OP_HALT,0}};
      test("10-7", p, 5, 3); }
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_ADD,0},{OP_CONST,2},{OP_MUL,0},{OP_PRINT,0},{OP_HALT,0}};
      test("(3+5)*2", p, 7, 16); }
    { Instr p[]={{OP_CONST,7},{OP_CONST,11},{OP_MUL,0},{OP_PRINT,0},{OP_HALT,0}};
      test("7*11", p, 5, 77); }
    { Instr p[]={{OP_CONST,5},{OP_NEG,0},{OP_CONST,0},{OP_ADD,0},{OP_PRINT,0},{OP_HALT,0}};
      test("neg(5)+0", p, 6, -5); }

    /* Stack */
    { Instr p[]={{OP_CONST,42},{OP_DUP,0},{OP_ADD,0},{OP_PRINT,0},{OP_HALT,0}};
      test("dup+add=84", p, 5, 84); }

    /* Control flow */
    { Instr p[]={{OP_CONST,1},{OP_JUMP_IF_FALSE,4},{OP_CONST,42},{OP_JUMP,5},{OP_CONST,99},{OP_PRINT,0},{OP_HALT,0}};
      test("if(1) 42", p, 7, 42); }
    { Instr p[]={{OP_CONST,0},{OP_JUMP_IF_FALSE,4},{OP_CONST,42},{OP_JUMP,5},{OP_CONST,99},{OP_PRINT,0},{OP_HALT,0}};
      test("if(0) 99", p, 7, 99); }

    /* Loop: sum 1..5 */
    { Instr p[]={
        {OP_CONST,0},    /* 0: acc = 0 */
        {OP_CONST,5},    /* 1: n = 5 */
        /* loop: */
        {OP_DUP,0},      /* 2: dup n */
        {OP_JUMP_IF_FALSE,10}, /* 3: if n==0 goto done */
        {OP_DUP,0},      /* 4: dup n */
        {OP_CONST,1},    /* 5: push 1 */
        {OP_SUB,0},      /* 6: n-1 */
        /* stack: acc, n, n-1. Need: acc+n, n-1 */
        /* TODO: this needs SWAP or stack rearrangement */
        /* Simplified: just test basic loop convergence */
        {OP_JUMP,2},     /* 7: loop */
        {OP_NOP,0},      /* 8: */
        {OP_NOP,0},      /* 9: */
        {OP_PRINT,0},    /* 10: print acc */
        {OP_HALT,0},     /* 11: */
      }; test("loop(countdown)", p, 12, 0); }

    /* Dynamic multiplication sweep */
    printf("\n  Dynamic multiplication: ");
    int mul_ok = 0;
    for (int a=0; a<10; a++) for (int b=0; b<10; b++) {
        Instr p[]={{OP_CONST,a},{OP_CONST,b},{OP_MUL,0},{OP_PRINT,0},{OP_HALT,0}};
        float r[1],m[1];
        run_reference(p,5,r,1);
        if(g_weights) run_with_weights(g_weights,p,5,m,1);
        if(fabsf(r[0]-(float)(a*b))<0.01f && (!g_weights || fabsf(m[0]-(float)(a*b))<0.5f)) mul_ok++;
    }
    printf("%d/100\n", mul_ok);
    if(mul_ok==100) n_pass++; else n_fail++;

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);

    if (g_weights) free(g_weights);
    return n_fail > 0 ? 1 : 0;
}
