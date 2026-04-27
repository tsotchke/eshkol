/**
 * @file weight_matrices.c
 * @brief Universal stack machine interpreter compiled into transformer weights.
 *
 * Generates explicit weight matrices that encode a universal interpreter.
 * When loaded into qLLM, the transformer executes ANY program fed as tokens.
 *
 * Three execution modes verified against each other:
 *   1. Reference interpreter (direct C switch statement)
 *   2. Simulated transformer (C functions mirroring weight computation)
 *   3. Matrix-based forward pass (actual W @ x + b through generic matmul)
 *
 * Architecture: d_model=16, n_heads=8, head_dim=2, n_layers=4
 *   Layer 0: Instruction fetch (Gaussian attention peaked at PC)
 *   Layer 1: Product precompute (SQUARE activation for TOS*SOS)
 *   Layer 2: Preprocessing (gated FFN for address resolution)
 *   Layer 3: Execution (gated FFN for opcode dispatch)
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define D 22   /* 14 state + 8 intermediate (product, loadval, store_flags[4], nz_flag, spare) */
#define H 11   /* d_model / head_dim */
#define HD 2
#define N_LAYERS 4
#define MEM_SIZE 4
#define N_OPCODES 14
#define FFN_DIM 256
#define SCALE 100.0f

typedef enum {
    OP_NOP=0, OP_CONST=1, OP_ADD=2, OP_SUB=3, OP_MUL=4, OP_DUP=5,
    OP_SWAP=6, OP_DROP=7, OP_LOAD=8, OP_STORE=9, OP_JUMP=10,
    OP_JUMP_IF=11, OP_OUTPUT=12, OP_HALT=13
} OpCode;

typedef struct { OpCode op; int operand; } Instr;

enum {
    S_PC=0, S_TOS=1, S_SOS=2, S_R2=3, S_R3=4, S_DEPTH=5,
    S_OUTPUT=6, S_HALT=7, S_MEM0=8, S_MEM1=9, S_MEM2=10, S_MEM3=11,
    S_OPCODE=12, S_OPERAND=13, S_PRODUCT=14, S_LOADVAL=15,
    S_STORED0=16, S_STORED1=17, S_STORED2=18, S_STORED3=19,
    S_ZOPER=20, S_ZPC1=21  /* zero_oper = ind(TOS==0)*oper, zero_pc1 = ind(TOS==0)*(PC+1) */
};

/*******************************************************************************
 * Utility Functions
 ******************************************************************************/

static float sigmoidf(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static float indicator(float x, float k) {
    return sigmoidf(SCALE * (x - k + 0.5f)) - sigmoidf(SCALE * (x - k - 0.5f));
}

static void embed_instruction(const Instr* instr, int position, float out[D]) {
    memset(out, 0, D * sizeof(float));
    out[0]  = (float)position;
    out[1]  = -(float)(position * position) / 2.0f;
    out[S_OPCODE]  = (float)instr->op;
    out[S_OPERAND] = (float)instr->operand;
}

/*******************************************************************************
 * 1. Reference Interpreter (direct C, ground truth)
 ******************************************************************************/

typedef struct { float s[D]; } State;

static void state_init(State* st) {
    memset(st, 0, sizeof(State));
    st->s[S_OUTPUT] = -1.0f;
}

static void execute_step(const State* cur, const Instr* prog, int n_instr, State* next) {
    memcpy(next, cur, sizeof(State));
    next->s[S_OUTPUT] = -1.0f;
    next->s[S_PRODUCT] = 0; next->s[S_LOADVAL] = 0;

    int pc = (int)cur->s[S_PC];
    if (pc < 0 || pc >= n_instr || cur->s[S_HALT] > 0.5f) { next->s[S_HALT] = 1; return; }

    float tos = cur->s[S_TOS], sos = cur->s[S_SOS];
    float r2 = cur->s[S_R2], r3 = cur->s[S_R3];
    float operand = (float)prog[pc].operand;
    int addr;

    switch (prog[pc].op) {
    case OP_NOP:    next->s[S_PC]=pc+1; break;
    case OP_CONST:  next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=operand; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1; break;
    case OP_ADD:    next->s[S_TOS]=tos+sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_SUB:    next->s[S_TOS]=sos-tos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_MUL:    next->s[S_TOS]=tos*sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_DUP:    next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1; break;
    case OP_SWAP:   next->s[S_TOS]=sos; next->s[S_SOS]=tos; next->s[S_PC]=pc+1; break;
    case OP_DROP:   next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_LOAD:   addr=(int)tos; if(addr>=0&&addr<MEM_SIZE) next->s[S_TOS]=cur->s[S_MEM0+addr]; next->s[S_PC]=pc+1; break;
    case OP_STORE:  addr=(int)sos; if(addr>=0&&addr<MEM_SIZE) next->s[S_MEM0+addr]=tos; next->s[S_TOS]=r2; next->s[S_SOS]=r3; next->s[S_R2]=0; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-2; next->s[S_PC]=pc+1; break;
    case OP_JUMP:   next->s[S_PC]=operand; break;
    case OP_JUMP_IF: next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=(tos!=0)?operand:(float)(pc+1); break;
    case OP_OUTPUT: next->s[S_OUTPUT]=tos; next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_HALT:   next->s[S_HALT]=1; break;
    default:        next->s[S_PC]=pc+1; break;
    }
}

static int run_reference(const Instr* prog, int n_instr, float* outputs, int max_out) {
    State trace[8192];
    state_init(&trace[0]);
    int t = 0, n_out = 0;
    while (t < 8191 && trace[t].s[S_HALT] < 0.5f) {
        int pc = (int)trace[t].s[S_PC];
        if (pc >= 0 && pc < n_instr) {
            trace[t].s[S_OPCODE] = (float)prog[pc].op;
            trace[t].s[S_OPERAND] = (float)prog[pc].operand;
        }
        execute_step(&trace[t], prog, n_instr, &trace[t+1]);
        if (trace[t+1].s[S_OUTPUT] >= 0 && n_out < max_out)
            outputs[n_out++] = trace[t+1].s[S_OUTPUT];
        t++;
    }
    return n_out;
}

/*******************************************************************************
 * 2. Simulated Transformer (C functions, verified correct)
 ******************************************************************************/

static void layer0_attention(const float x[D], const float pe[][D], int np, float out[D]) {
    memset(out, 0, D * sizeof(float));
    float T = SCALE;
    float scores[256]; float mx = -1e30f;
    for (int p = 0; p < np && p < 256; p++) {
        scores[p] = (x[S_PC]*T*pe[p][0] + T*pe[p][1]) / sqrtf(2.0f);
        if (scores[p] > mx) mx = scores[p];
    }
    float sum = 0;
    for (int p = 0; p < np; p++) { scores[p] = expf(scores[p]-mx); sum += scores[p]; }
    for (int p = 0; p < np; p++) scores[p] /= sum;
    float v0=0, v1=0;
    for (int p = 0; p < np; p++) { v0 += scores[p]*pe[p][S_OPCODE]; v1 += scores[p]*pe[p][S_OPERAND]; }
    out[S_OPCODE] = v0; out[S_OPERAND] = v1;
}

static void layer1_ffn(const float x[D], float out[D]) {
    memset(out, 0, D*sizeof(float));
    float a = x[S_TOS], b = x[S_SOS];
    out[S_PRODUCT] = 0.5f*(a+b)*(a+b) - 0.5f*a*a - 0.5f*b*b;
}

static void layer2_ffn(const float x[D], float out[D]) {
    memset(out, 0, D*sizeof(float));
    /* LOAD address resolution: indicator(TOS==a) * mem[a] → LOADVAL */
    for (int a = 0; a < MEM_SIZE; a++)
        out[S_LOADVAL] += indicator(x[S_TOS], (float)a) * x[S_MEM0+a];
    /* STORE deltas: indicator(SOS==a) * (TOS - mem[a]) → STORED0+a */
    for (int a = 0; a < MEM_SIZE; a++)
        out[S_STORED0+a] = indicator(x[S_SOS], (float)a) * (x[S_TOS] - x[S_MEM0+a]);
    /* JUMP_IF zero-case: indicator(TOS==0) * operand, indicator(TOS==0) * (PC+1) */
    float iz = indicator(x[S_TOS], 0.0f);
    out[S_ZOPER] = iz * x[S_OPERAND];
    out[S_ZPC1]  = iz * (x[S_PC] + 1.0f);
}

static void layer3_ffn(const float x[D], float out[D]) {
    memset(out, 0, D*sizeof(float));
    float op=x[S_OPCODE], oper=x[S_OPERAND], tos=x[S_TOS], sos=x[S_SOS];
    float r2=x[S_R2], r3=x[S_R3], product=x[S_PRODUCT], lv=x[S_LOADVAL];
    float alive = 1.0f - sigmoidf(SCALE*(x[S_HALT]-0.5f));

    /* Universal: clear output and intermediates */
    out[S_OUTPUT] = -1.0f - x[S_OUTPUT];
    /* Clear all intermediate dims (12-21) */
    for (int i = S_OPCODE; i <= S_ZPC1; i++) out[i] = -x[i];

    float g;
    /* NOP */   g=indicator(op,0)*alive; out[S_PC]+=g;
    /* CONST */ g=indicator(op,1)*alive; out[S_PC]+=g; out[S_TOS]+=g*(oper-tos); out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
    /* ADD */   g=indicator(op,2)*alive; out[S_PC]+=g; out[S_TOS]+=g*sos; out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* SUB */   g=indicator(op,3)*alive; out[S_PC]+=g; out[S_TOS]+=g*(sos-2*tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* MUL */   g=indicator(op,4)*alive; out[S_PC]+=g; out[S_TOS]+=g*(product-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* DUP */   g=indicator(op,5)*alive; out[S_PC]+=g; out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
    /* SWAP */  g=indicator(op,6)*alive; out[S_PC]+=g; out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(tos-sos);
    /* DROP */  g=indicator(op,7)*alive; out[S_PC]+=g; out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* LOAD */  g=indicator(op,8)*alive; out[S_PC]+=g; out[S_TOS]+=g*(lv-tos);
    /* STORE */ g=indicator(op,9)*alive; out[S_PC]+=g;
                for(int a=0;a<MEM_SIZE;a++) out[S_MEM0+a]+=g*x[S_STORED0+a]; /* precomputed store deltas */
                out[S_TOS]+=g*(r2-tos); out[S_SOS]+=g*(r3-sos); out[S_R2]+=g*(-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-2);
    /* JUMP */  g=indicator(op,10)*alive; out[S_PC]+=g*(oper-x[S_PC]);
    /* JIF */   g=indicator(op,11)*alive;
                { /* new_PC = operand - zero_oper + zero_pc1 (precomputed by layer 2)
                   * delta[PC] = operand - zero_oper + zero_pc1 - PC */
                  out[S_PC]+=g*(oper - x[S_ZOPER] + x[S_ZPC1] - x[S_PC]); }
                out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* OUTPUT */g=indicator(op,12)*alive; out[S_PC]+=g; out[S_OUTPUT]+=g*(tos+1);
                out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* HALT */  g=indicator(op,13)*alive; out[S_HALT]+=g;
}

static int run_simulated(const Instr* prog, int n_instr, float* outputs, int max_out) {
    float pe[256][D];
    for (int p = 0; p < n_instr && p < 256; p++) embed_instruction(&prog[p], p, pe[p]);
    float state[D]; memset(state, 0, sizeof(state)); state[S_OUTPUT] = -1;
    int n_out = 0;
    for (int step = 0; step < 8192; step++) {
        float x[D]; memcpy(x, state, sizeof(x));
        float tmp[D];
        layer0_attention(x, pe, n_instr, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
        layer1_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
        layer2_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
        layer3_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
        if (x[S_OUTPUT] >= -0.5f && n_out < max_out) outputs[n_out++] = x[S_OUTPUT];
        if (x[S_HALT] > 0.5f) break;
        memcpy(state, x, sizeof(state));
    }
    return n_out;
}

/*******************************************************************************
 * 3. Explicit Weight Matrices + Matrix-Based Forward Pass
 ******************************************************************************/

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
    int   ff_type[N_LAYERS];  /* 0=noop, 1=standard+square, 2=gated+sigmoid */
} InterpreterWeights;

/* W[row][col] stored as W[row * cols + col] */
#define W(mat, r, c, cols) ((mat)[(r) * (cols) + (c)])

/* Add a gated neuron pair for layer 3 opcode dispatch.
 * Each pair implements: indicator(opcode==op_id) * alive * (up_value) → out_dim
 * where up_value is a linear combination of state dims.
 *
 * gate_pos = sigmoid(S*(opcode - op_id + 0.5) - S*halt)
 * gate_neg = sigmoid(S*(opcode - op_id - 0.5) - S*halt)
 * W_down routes (gate_pos*up - gate_neg*up) → out_dim
 */
static int add_gated_pair(InterpreterWeights* w, int L, int n,
                           int op_id,
                           /* up projection: up to 4 (dim, scale) pairs + bias */
                           int ud1, float us1, int ud2, float us2,
                           int ud3, float us3, int ud4, float us4,
                           float ubias,
                           /* output */
                           int out_dim, float coeff) {
    for (int sign = 0; sign < 2; sign++) {
        int j = n + sign;
        float s = (sign == 0) ? 0.5f : -0.5f;

        /* Gate: sigmoid(S*(opcode - op_id ± 0.5) - S*halt) */
        W(w->ff_gate[L], S_OPCODE, j, FFN_DIM) += SCALE;
        W(w->ff_gate[L], S_HALT, j, FFN_DIM)   += -SCALE;
        w->ff_gate_b[L][j] = SCALE * (-(float)op_id + s);

        /* Up projection */
        if (ud1 >= 0) W(w->ff_up[L], ud1, j, FFN_DIM) += us1;
        if (ud2 >= 0) W(w->ff_up[L], ud2, j, FFN_DIM) += us2;
        if (ud3 >= 0) W(w->ff_up[L], ud3, j, FFN_DIM) += us3;
        if (ud4 >= 0) W(w->ff_up[L], ud4, j, FFN_DIM) += us4;
        w->ff_up_b[L][j] = ubias;

        /* Down projection */
        float c = (sign == 0) ? coeff : -coeff;
        W(w->ff_down[L], j, out_dim, D) += c;
    }
    return n + 2;
}

/* Shorthand for unconditional (always-on) neuron */
static int add_unconditional(InterpreterWeights* w, int L, int n,
                              int ud1, float us1, float ubias,
                              int out_dim, float coeff) {
    int j = n;
    /* Gate: sigmoid(big_positive - S*halt) ≈ alive */
    W(w->ff_gate[L], S_HALT, j, FFN_DIM) = -SCALE;
    w->ff_gate_b[L][j] = 10.0f * SCALE;
    if (ud1 >= 0) W(w->ff_up[L], ud1, j, FFN_DIM) = us1;
    w->ff_up_b[L][j] = ubias;
    W(w->ff_down[L], j, out_dim, D) = coeff;
    return n + 1;
}

static void generate_weights(InterpreterWeights* w) {
    memset(w, 0, sizeof(InterpreterWeights));

    /* ── Layer 0: Attention (instruction fetch) ── */
    {
        float T = SCALE;
        W(w->wq[0], 0, S_PC, D) = T;        /* q[0] = PC * T */
        w->bq[0][1] = T;                      /* q[1] = T (constant) */
        W(w->wk[0], 0, 0, D) = 1.0f;         /* k[0] = embed[0] = position */
        W(w->wk[0], 1, 1, D) = 1.0f;         /* k[1] = embed[1] = -pos²/2 */
        W(w->wv[0], 0, S_OPCODE, D) = 1.0f;  /* v[0] = opcode */
        W(w->wv[0], 1, S_OPERAND, D) = 1.0f; /* v[1] = operand */
        W(w->wo[0], S_OPCODE, 0, D) = 1.0f;  /* out[12] = head0[0] */
        W(w->wo[0], S_OPERAND, 1, D) = 1.0f; /* out[13] = head0[1] */
        w->ff_type[0] = 0; /* no-op FFN */
    }

    /* ── Layer 1: Product FFN (SQUARE activation) ── */
    {
        w->ff_type[1] = 1;
        W(w->ff_up[1], S_TOS, 0, FFN_DIM) = 1; W(w->ff_up[1], S_SOS, 0, FFN_DIM) = 1; /* n0: TOS+SOS */
        W(w->ff_up[1], S_TOS, 1, FFN_DIM) = 1;                                          /* n1: TOS */
        W(w->ff_up[1], S_SOS, 2, FFN_DIM) = 1;                                          /* n2: SOS */
        W(w->ff_down[1], 0, S_PRODUCT, D) =  0.5f;
        W(w->ff_down[1], 1, S_PRODUCT, D) = -0.5f;
        W(w->ff_down[1], 2, S_PRODUCT, D) = -0.5f;
    }

    /* ── Layer 2: Preprocessing (gated FFN) ── */
    {
        w->ff_type[2] = 2;
        int n = 0;

        /* LOAD address resolution: indicator(TOS==a) * mem[a] → LOADVAL */
        for (int a = 0; a < MEM_SIZE; a++) {
            W(w->ff_gate[2], S_TOS, n, FFN_DIM) = SCALE;
            w->ff_gate_b[2][n] = SCALE * (-(float)a + 0.5f);
            W(w->ff_up[2], S_MEM0+a, n, FFN_DIM) = 1.0f;
            W(w->ff_down[2], n, S_LOADVAL, D) = 1.0f;
            n++;
            W(w->ff_gate[2], S_TOS, n, FFN_DIM) = SCALE;
            w->ff_gate_b[2][n] = SCALE * (-(float)a - 0.5f);
            W(w->ff_up[2], S_MEM0+a, n, FFN_DIM) = 1.0f;
            W(w->ff_down[2], n, S_LOADVAL, D) = -1.0f;
            n++;
        }

        /* STORE deltas: indicator(SOS==a) * (TOS - mem[a]) → S_STORED0+a */
        for (int a = 0; a < MEM_SIZE; a++) {
            /* Pos: gate = sigmoid(S*(SOS - a + 0.5)), up = TOS - mem[a] */
            W(w->ff_gate[2], S_SOS, n, FFN_DIM) = SCALE;
            w->ff_gate_b[2][n] = SCALE * (-(float)a + 0.5f);
            W(w->ff_up[2], S_TOS, n, FFN_DIM) = 1.0f;
            W(w->ff_up[2], S_MEM0+a, n, FFN_DIM) = -1.0f;
            W(w->ff_down[2], n, S_STORED0+a, D) = 1.0f;
            n++;
            /* Neg: gate = sigmoid(S*(SOS - a - 0.5)), up = TOS - mem[a] */
            W(w->ff_gate[2], S_SOS, n, FFN_DIM) = SCALE;
            w->ff_gate_b[2][n] = SCALE * (-(float)a - 0.5f);
            W(w->ff_up[2], S_TOS, n, FFN_DIM) = 1.0f;
            W(w->ff_up[2], S_MEM0+a, n, FFN_DIM) = -1.0f;
            W(w->ff_down[2], n, S_STORED0+a, D) = -1.0f;
            n++;
        }

        /* JUMP_IF zero case: indicator(TOS==0) * operand → S_ZOPER */
        {
            /* Pos: sigmoid(S*(TOS + 0.5)) * operand */
            W(w->ff_gate[2], S_TOS, n, FFN_DIM) = SCALE;
            w->ff_gate_b[2][n] = SCALE * 0.5f;
            W(w->ff_up[2], S_OPERAND, n, FFN_DIM) = 1.0f;
            W(w->ff_down[2], n, S_ZOPER, D) = 1.0f;
            n++;
            /* Neg: sigmoid(S*(TOS - 0.5)) * operand */
            W(w->ff_gate[2], S_TOS, n, FFN_DIM) = SCALE;
            w->ff_gate_b[2][n] = SCALE * (-0.5f);
            W(w->ff_up[2], S_OPERAND, n, FFN_DIM) = 1.0f;
            W(w->ff_down[2], n, S_ZOPER, D) = -1.0f;
            n++;
        }

        /* JUMP_IF zero case: indicator(TOS==0) * (PC+1) → S_ZPC1 */
        {
            /* Pos: sigmoid(S*(TOS + 0.5)) * (PC + 1) */
            W(w->ff_gate[2], S_TOS, n, FFN_DIM) = SCALE;
            w->ff_gate_b[2][n] = SCALE * 0.5f;
            W(w->ff_up[2], S_PC, n, FFN_DIM) = 1.0f;
            w->ff_up_b[2][n] = 1.0f;  /* +1 for PC+1 */
            W(w->ff_down[2], n, S_ZPC1, D) = 1.0f;
            n++;
            /* Neg: sigmoid(S*(TOS - 0.5)) * (PC + 1) */
            W(w->ff_gate[2], S_TOS, n, FFN_DIM) = SCALE;
            w->ff_gate_b[2][n] = SCALE * (-0.5f);
            W(w->ff_up[2], S_PC, n, FFN_DIM) = 1.0f;
            w->ff_up_b[2][n] = 1.0f;
            W(w->ff_down[2], n, S_ZPC1, D) = -1.0f;
            n++;
        }

        printf("[WEIGHT_GEN] Layer 2: %d neurons\n", n);
    }

    /* ── Layer 3: Execution (gated FFN) ── */
    {
        const int L = 3;
        w->ff_type[L] = 2;
        int n = 0;

        /* Universal: clear output to -1 */
        n = add_unconditional(w, L, n, S_OUTPUT, -1.0f, -1.0f, S_OUTPUT, 1.0f);
        /* Universal: clear ALL intermediate dims (12-21) */
        for (int d = S_OPCODE; d <= S_ZPC1; d++) {
            n = add_unconditional(w, L, n, d, -1.0f, 0, d, 1.0f);
        }

        /* OP_NOP (0): PC += 1 */
        n = add_gated_pair(w,L,n, 0, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);

        /* OP_CONST (1): TOS=oper, SOS=TOS, R2=SOS, R3=R2, depth++, PC++ */
        n = add_gated_pair(w,L,n, 1, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 1, S_OPERAND,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 1, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 1, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 1, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 1, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);

        /* OP_ADD (2): TOS=TOS+SOS (delta=+SOS), shift up, depth--, PC++ */
        n = add_gated_pair(w,L,n, 2, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 2, S_SOS,1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 2, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 2, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 2, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 2, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_SUB (3): TOS=SOS-TOS (delta=SOS-2*TOS), shift up */
        n = add_gated_pair(w,L,n, 3, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 3, S_SOS,1,S_TOS,-2,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 3, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 3, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 3, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 3, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_MUL (4): TOS=TOS*SOS=product (delta=product-TOS), shift up */
        n = add_gated_pair(w,L,n, 4, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 4, S_PRODUCT,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 4, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 4, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 4, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 4, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_DUP (5): push TOS copy, shift down */
        n = add_gated_pair(w,L,n, 5, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 5, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 5, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 5, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 5, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);

        /* OP_SWAP (6): swap TOS and SOS */
        n = add_gated_pair(w,L,n, 6, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 6, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 6, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);

        /* OP_DROP (7): pop TOS, shift up */
        n = add_gated_pair(w,L,n, 7, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 7, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 7, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 7, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 7, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 7, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_LOAD (8): TOS = loadval (delta = loadval - TOS), PC++ */
        n = add_gated_pair(w,L,n, 8, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 8, S_LOADVAL,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);

        /* OP_STORE (9): mem[SOS]=TOS, pop both, shift up, PC++ */
        n = add_gated_pair(w,L,n, 9, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        /* Memory write: use precomputed store_delta[a] from layer 2.
         * store_delta[a] = indicator(SOS==a) * (TOS - mem[a]), already in dim S_STORED0+a.
         * Layer 3 just gates by opcode==9: indicator(op==9) * store_delta[a] → mem[a] */
        for (int a = 0; a < MEM_SIZE; a++) {
            n = add_gated_pair(w,L,n, 9, S_STORED0+a,1,-1,0,-1,0,-1,0, 0, S_MEM0+a, 1.0f);
        }
        /* STORE register shifts */
        n = add_gated_pair(w,L,n, 9, S_R2,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 9, S_R3,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 9, S_R2,-1,-1,0,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 9, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 9, -1,0,-1,0,-1,0,-1,0, -2.0f, S_DEPTH, 1.0f);

        /* OP_JUMP (10): PC = operand (delta = operand - PC) */
        n = add_gated_pair(w,L,n, 10, S_OPERAND,1,S_PC,-1,-1,0,-1,0, 0, S_PC, 1.0f);

        /* OP_JUMP_IF (11): complex — use simulated path for now */
        /* Pop + conditional PC. Encoding the nonzero test (TOS != 0) in a single
         * gate neuron pair requires nesting, which we handle in the simulated path. */
        /* Register shifts for the pop: */
        n = add_gated_pair(w,L,n, 11, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 11, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 11, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 11, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 11, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        /* JUMP_IF PC update: delta[PC] = operand - zero_oper + zero_pc1 - PC
         * = (1-ind(TOS==0))*operand + ind(TOS==0)*(PC+1) - PC
         * All precomputed by layer 2, now linear in state dims. */
        n = add_gated_pair(w,L,n, 11,
                           S_OPERAND, 1, S_ZOPER, -1, S_ZPC1, 1, S_PC, -1,
                           0, S_PC, 1.0f);

        /* OP_OUTPUT (12): output = TOS, pop, shift up, PC++ */
        n = add_gated_pair(w,L,n, 12, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 12, S_TOS,1,-1,0,-1,0,-1,0, 1.0f, S_OUTPUT, 1.0f);
        n = add_gated_pair(w,L,n, 12, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 12, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 12, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 12, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 12, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_HALT (13): halt = 1 */
        n = add_gated_pair(w,L,n, 13, -1,0,-1,0,-1,0,-1,0, 1.0f, S_HALT, 1.0f);

        printf("[WEIGHT_GEN] Layer 3: %d neurons used out of %d\n", n, FFN_DIM);
    }

    printf("[WEIGHT_GEN] Weights: %zu params, %.1f KB\n",
           sizeof(InterpreterWeights)/sizeof(float),
           sizeof(InterpreterWeights)/1024.0f);
}

/* ── Matrix-based forward pass ── */

static void matvec_t(const float* x, const float* W, float* out, int rows, int cols) {
    memset(out, 0, cols * sizeof(float));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out[j] += x[i] * W[i * cols + j];
}

static void forward_with_weights(const InterpreterWeights* w,
                                  const float state[D],
                                  const float pe[][D], int np,
                                  float next[D]) {
    float x[D]; memcpy(x, state, sizeof(float)*D);

    for (int L = 0; L < N_LAYERS; L++) {
        /* Attention */
        float ao[D]; memset(ao, 0, sizeof(ao));
        if (L == 0) {
            float Q[D]; memset(Q, 0, sizeof(Q));
            for (int i=0;i<D;i++) for(int j=0;j<D;j++) Q[i]+=w->wq[L][i*D+j]*x[j];
            for (int i=0;i<D;i++) Q[i]+=w->bq[L][i];

            float scores[256]; float mx=-1e30f;
            float Va[256][D];
            for (int p=0; p<np&&p<256; p++) {
                float K[D]; memset(K,0,sizeof(K));
                memset(Va[p],0,sizeof(Va[p]));
                for(int i=0;i<D;i++) for(int j=0;j<D;j++) { K[i]+=w->wk[L][i*D+j]*pe[p][j]; Va[p][i]+=w->wv[L][i*D+j]*pe[p][j]; }
                scores[p]=(Q[0]*K[0]+Q[1]*K[1])/sqrtf((float)HD);
                if(scores[p]>mx) mx=scores[p];
            }
            float sum=0;
            for(int p=0;p<np;p++){scores[p]=expf(scores[p]-mx);sum+=scores[p];}
            for(int p=0;p<np;p++) scores[p]/=sum;
            float hout[D]; memset(hout,0,sizeof(hout));
            for(int p=0;p<np;p++) for(int d=0;d<HD;d++) hout[d]+=scores[p]*Va[p][d];
            for(int i=0;i<D;i++) for(int j=0;j<D;j++) ao[i]+=w->wo[L][i*D+j]*hout[j];
        }
        for(int i=0;i<D;i++) x[i]+=ao[i];

        /* FFN */
        float fo[D]; memset(fo,0,sizeof(fo));
        if (w->ff_type[L]==1) {
            float h[FFN_DIM];
            matvec_t(x, w->ff_up[L], h, D, FFN_DIM);
            for(int i=0;i<FFN_DIM;i++) h[i]+=w->ff_up_b[L][i];
            for(int i=0;i<FFN_DIM;i++) h[i]*=h[i]; /* SQUARE */
            matvec_t(h, w->ff_down[L], fo, FFN_DIM, D);
            for(int i=0;i<D;i++) fo[i]+=w->ff_down_b[L][i];
        } else if (w->ff_type[L]==2) {
            float gate[FFN_DIM], up[FFN_DIM], h[FFN_DIM];
            matvec_t(x, w->ff_gate[L], gate, D, FFN_DIM);
            for(int i=0;i<FFN_DIM;i++) gate[i]=sigmoidf(gate[i]+w->ff_gate_b[L][i]);
            matvec_t(x, w->ff_up[L], up, D, FFN_DIM);
            for(int i=0;i<FFN_DIM;i++) up[i]+=w->ff_up_b[L][i];
            for(int i=0;i<FFN_DIM;i++) h[i]=gate[i]*up[i];
            matvec_t(h, w->ff_down[L], fo, FFN_DIM, D);
            for(int i=0;i<D;i++) fo[i]+=w->ff_down_b[L][i];
        }
        for(int i=0;i<D;i++) x[i]+=fo[i];
    }
    memcpy(next, x, sizeof(float)*D);
}

static int run_with_weights(const InterpreterWeights* w,
                             const Instr* prog, int n_instr,
                             float* outputs, int max_out) {
    float pe[256][D];
    for(int p=0;p<n_instr&&p<256;p++) embed_instruction(&prog[p],p,pe[p]);
    float state[D]; memset(state,0,sizeof(state)); state[S_OUTPUT]=-1;
    int n_out=0;
    for(int step=0;step<8192;step++){
        float next[D];
        forward_with_weights(w,state,pe,n_instr,next);
        if(next[S_OUTPUT]>=-0.5f&&n_out<max_out) outputs[n_out++]=next[S_OUTPUT];
        if(next[S_HALT]>0.5f) break;
        memcpy(state,next,sizeof(state));
    }
    return n_out;
}

/*******************************************************************************
 * Tests: 3-way comparison (reference vs simulated vs matrix-based)
 ******************************************************************************/

/*******************************************************************************
 * Binary Weight Export (for qLLM loading)
 ******************************************************************************/

/* Export weights as a structured binary file.
 * Header: magic, version, d_model, n_layers, ffn_dim, n_heads, head_dim
 * Then: per-layer attention weights, per-layer FFN weights (with type tags) */
static void export_weights_binary(const InterpreterWeights* w, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { printf("ERROR: cannot open %s\n", path); return; }

    /* Header */
    uint32_t magic = 0x514C4D57; /* "QLMW" */
    uint32_t version = 1;
    uint32_t d = D, nl = N_LAYERS, fd = FFN_DIM, nh = H, hd = HD;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&d, 4, 1, f);
    fwrite(&nl, 4, 1, f);
    fwrite(&fd, 4, 1, f);
    fwrite(&nh, 4, 1, f);
    fwrite(&hd, 4, 1, f);

    /* Write all weight arrays */
    fwrite(w, sizeof(InterpreterWeights), 1, f);

    fclose(f);
    printf("[EXPORT] Wrote %zu bytes to %s\n", sizeof(InterpreterWeights) + 28, path);
}

/*******************************************************************************
 * Tests: 3-way comparison (reference vs simulated vs matrix-based)
 ******************************************************************************/

static int n_pass = 0, n_fail = 0;
static InterpreterWeights* g_weights = NULL;

static void test(const char* name, const Instr* prog, int n, float expected) {
    float r[64], s[64], m[64];
    int rn = run_reference(prog, n, r, 64);
    int sn = run_simulated(prog, n, s, 64);
    int mn = g_weights ? run_with_weights(g_weights, prog, n, m, 64) : 0;

    float rv = rn>0?r[0]:-9999, sv = sn>0?s[0]:-9999, mv = mn>0?m[0]:-9999;
    int ok_r = fabsf(rv-expected)<0.01f;
    int ok_s = fabsf(sv-expected)<0.01f;
    int ok_m = g_weights ? fabsf(mv-expected)<0.01f : 1;

    printf("  %-20s ref=%7.1f sim=%7.1f mat=%7.1f  %s%s%s\n",
           name, rv, sv, mv,
           ok_r?"":"ref:FAIL ", ok_s?"":"sim:FAIL ",
           (g_weights && !ok_m)?"mat:FAIL ":"");

    if (ok_r && ok_s && ok_m) n_pass++; else n_fail++;
}

int main() {
    printf("=== Universal Interpreter Weight Compiler ===\n\n");

    /* Generate explicit weight matrices */
    g_weights = (InterpreterWeights*)calloc(1, sizeof(InterpreterWeights));
    if (g_weights) generate_weights(g_weights);

    printf("\n  Tests (ref=reference, sim=simulated, mat=matrix-based):\n\n");

    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_ADD,0},{OP_OUTPUT,0},{OP_HALT,0}};
      test("3+5", p, 5, 8); }
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_ADD,0},{OP_CONST,2},{OP_MUL,0},{OP_OUTPUT,0},{OP_HALT,0}};
      test("(3+5)*2", p, 7, 16); }
    { Instr p[]={{OP_CONST,10},{OP_CONST,7},{OP_SUB,0},{OP_OUTPUT,0},{OP_HALT,0}};
      test("10-7", p, 5, 3); }
    { Instr p[]={{OP_CONST,0},{OP_CONST,42},{OP_STORE,0},{OP_CONST,0},{OP_LOAD,0},{OP_OUTPUT,0},{OP_HALT,0}};
      test("mem[0]=42", p, 7, 42); }
    { Instr p[]={
        {OP_CONST,0},{OP_CONST,0},{OP_STORE,0},
        {OP_CONST,1},{OP_CONST,5},{OP_STORE,0},
        {OP_CONST,1},{OP_LOAD,0},{OP_DUP,0},{OP_JUMP_IF,12},{OP_DROP,0},{OP_JUMP,26},
        {OP_CONST,0},{OP_LOAD,0},{OP_ADD,0},{OP_CONST,0},{OP_SWAP,0},{OP_STORE,0},
        {OP_CONST,1},{OP_LOAD,0},{OP_CONST,1},{OP_SUB,0},{OP_CONST,1},{OP_SWAP,0},{OP_STORE,0},
        {OP_JUMP,6},{OP_CONST,0},{OP_LOAD,0},{OP_OUTPUT,0},{OP_HALT,0},
      }; test("sum(1..5)", p, 30, 15); }
    { Instr p[]={
        {OP_CONST,0},{OP_CONST,1},{OP_STORE,0},
        {OP_CONST,1},{OP_CONST,5},{OP_STORE,0},
        {OP_CONST,1},{OP_LOAD,0},{OP_DUP,0},{OP_JUMP_IF,12},{OP_DROP,0},{OP_JUMP,26},
        {OP_CONST,0},{OP_LOAD,0},{OP_MUL,0},{OP_CONST,0},{OP_SWAP,0},{OP_STORE,0},
        {OP_CONST,1},{OP_LOAD,0},{OP_CONST,1},{OP_SUB,0},{OP_CONST,1},{OP_SWAP,0},{OP_STORE,0},
        {OP_JUMP,6},{OP_CONST,0},{OP_LOAD,0},{OP_OUTPUT,0},{OP_HALT,0},
      }; test("5!", p, 30, 120); }
    { Instr p[]={
        {OP_CONST,0},{OP_CONST,0},{OP_STORE,0},
        {OP_CONST,1},{OP_CONST,1},{OP_STORE,0},
        {OP_CONST,2},{OP_CONST,7},{OP_STORE,0},
        {OP_CONST,2},{OP_LOAD,0},{OP_DUP,0},{OP_JUMP_IF,16},{OP_DROP,0},{OP_JUMP,39},{OP_NOP,0},
        {OP_DROP,0},
        {OP_CONST,0},{OP_LOAD,0},{OP_CONST,1},{OP_LOAD,0},{OP_ADD,0},
        {OP_CONST,1},{OP_LOAD,0},{OP_CONST,0},{OP_SWAP,0},{OP_STORE,0},
        {OP_CONST,1},{OP_SWAP,0},{OP_STORE,0},
        {OP_CONST,2},{OP_LOAD,0},{OP_CONST,1},{OP_SUB,0},{OP_CONST,2},{OP_SWAP,0},{OP_STORE,0},
        {OP_JUMP,9},
        {OP_CONST,0},{OP_LOAD,0},{OP_OUTPUT,0},{OP_HALT,0},
      }; test("fib(7)", p, 43, 13); }
    { Instr p[]={
        {OP_CONST,2},{OP_CONST,3},{OP_MUL,0},
        {OP_CONST,4},{OP_CONST,5},{OP_MUL,0},
        {OP_ADD,0},{OP_OUTPUT,0},{OP_HALT,0},
      }; test("(2*3)+(4*5)", p, 9, 26); }
    { Instr p[]={{OP_CONST,7},{OP_CONST,11},{OP_MUL,0},{OP_OUTPUT,0},{OP_HALT,0}};
      test("7*11", p, 5, 77); }

    /* Dynamic mul tests */
    printf("\n  Dynamic multiplication: ");
    int mul_ok = 0;
    for (int a=0; a<10; a++) for (int b=0; b<10; b++) {
        Instr p[]={{OP_CONST,a},{OP_CONST,b},{OP_MUL,0},{OP_OUTPUT,0},{OP_HALT,0}};
        float r[1],s[1],m[1];
        run_reference(p,5,r,1); run_simulated(p,5,s,1);
        if(g_weights) run_with_weights(g_weights,p,5,m,1);
        if(fabsf(r[0]-(float)(a*b))<0.01f && fabsf(s[0]-(float)(a*b))<0.01f
           && (!g_weights || fabsf(m[0]-(float)(a*b))<0.01f)) mul_ok++;
    }
    printf("%d/100\n", mul_ok);
    if(mul_ok==100) n_pass++; else n_fail++;

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);

    /* Export weights */
    if (g_weights && n_fail == 0) {
        export_weights_binary(g_weights, "/tmp/interpreter_weights.bin");
    }

    if (g_weights) free(g_weights);
    return n_fail > 0 ? 1 : 0;
}
