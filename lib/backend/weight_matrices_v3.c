/**
 * @file weight_matrices_v3.c
 * @brief Universal Eshkol VM interpreter compiled into transformer weights.
 *
 * Incremental expansion from v1 (14 opcodes, d_model=22) to full ISA.
 * Opcode numbering matches eshkol_compiler.c canonical enum.
 *
 * Three execution modes verified against each other:
 *   1. Reference interpreter (direct C switch)
 *   2. Simulated transformer (C functions mirroring weight computation)
 *   3. Matrix-based forward pass (actual W @ x + b through generic matmul)
 *
 * Architecture: d_model=32, n_heads=16, head_dim=2, n_layers=5, FFN_DIM=512
 *   Layer 0: Instruction fetch (Gaussian attention peaked at PC)
 *   Layer 1: Product precompute (SQUARE activation for TOS*SOS)
 *   Layer 2: Preprocessing (gated FFN for address resolution, comparisons)
 *   Layer 3: Execution (gated FFN for opcode dispatch)
 *   Layer 4: Frame management (gated FFN for CALL/RET, currently no-op)
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define D 32
#define H 16
#define HD 2
#define N_LAYERS 5
#define MEM_SIZE 4
#define FFN_DIM 512
#define SCALE 100.0f

/* Opcodes — canonical numbering from eshkol_compiler.c */
typedef enum {
    OP_NOP=0, OP_CONST=1, OP_NIL=2, OP_TRUE=3, OP_FALSE=4, OP_POP=5, OP_DUP=6,
    OP_ADD=7, OP_SUB=8, OP_MUL=9, OP_DIV=10, OP_MOD=11, OP_NEG=12, OP_ABS=13,
    OP_EQ=14, OP_LT=15, OP_GT=16, OP_LE=17, OP_GE=18, OP_NOT=19,
    OP_GET_LOCAL=20, OP_SET_LOCAL=21, OP_GET_UPVALUE=22, OP_SET_UPVALUE=23,
    OP_CLOSURE=24, OP_CALL=25, OP_TAIL_CALL=26, OP_RETURN=27,
    OP_JUMP=28, OP_JUMP_IF_FALSE=29, OP_LOOP=30,
    OP_CONS=31, OP_CAR=32, OP_CDR=33, OP_NULL_P=34,
    OP_PRINT=35, OP_HALT=36, OP_NATIVE_CALL=37,
    OP_CLOSE_UPVALUE=38,
    OP_VEC_CREATE=39, OP_VEC_REF=40, OP_VEC_SET=41, OP_VEC_LEN=42,
    OP_STR_REF=43, OP_STR_LEN=44,
    OP_PAIR_P=45, OP_NUM_P=46, OP_STR_P=47, OP_BOOL_P=48, OP_PROC_P=49, OP_VEC_P=50,
    OP_SET_CAR=51, OP_SET_CDR=52, OP_POPN=53,
    OP_OPEN_CLOSURE=54, OP_CALLCC=55, OP_INVOKE_CC=56,
    OP_PUSH_HANDLER=57, OP_POP_HANDLER=58, OP_GET_EXN=59,
    OP_PACK_REST=60, OP_WIND_PUSH=61, OP_WIND_POP=62,
    OP_COUNT=63
} OpCode;

typedef struct { OpCode op; int operand; } Instr;

/* State vector layout (d_model=32) */
enum {
    /* Permanent state (0-15) — persist across steps */
    S_PC=0, S_TOS=1, S_SOS=2, S_R2=3, S_R3=4, S_DEPTH=5,
    S_OUTPUT=6, S_HALT=7,
    S_MEM0=8, S_MEM1=9, S_MEM2=10, S_MEM3=11,
    S_SP=12, S_FP=13, S_HAS_OUT=14, S_SPARE=15,

    /* Intermediate / transient (16-31) — cleared every cycle by Layer 3 */
    S_OPCODE=16, S_OPERAND=17,
    S_PRODUCT=18, S_LOADVAL=19,
    S_STORED0=20, S_STORED1=21, S_STORED2=22, S_STORED3=23,
    S_ZOPER=24, S_ZPC1=25,
    S_CMP_EQ=26, S_CMP_LT=27,
    S_IS_CALL=28, S_IS_RET=29, S_IS_NATIVE=30,
    S_ABS_DELTA=31
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
    out[0]  = (float)position;                        /* key[0] = position */
    out[1]  = -(float)(position * position) / 2.0f;   /* key[1] = -pos²/2 */
    out[S_OPCODE]  = (float)instr->op;                /* value[0] = opcode */
    out[S_OPERAND] = (float)instr->operand;           /* value[1] = operand */
}

/* Forward declarations for exec loop */
typedef struct {
    float return_pc;
    float saved_mem[MEM_SIZE];
    float saved_tos, saved_sos, saved_r2, saved_r3, saved_depth;
} CallFrame;
static CallFrame g_frames[64];
static int g_frame_count = 0;
static void exec_loop_postprocess(float x[D], const Instr* prog, int n_instr);

/* Simple heap for CONS/CAR/CDR (pairs stored as consecutive float pairs) */
#define HEAP_SIZE 4096
static float g_heap[HEAP_SIZE];
static int g_heap_ptr = 0;

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
    next->s[S_HAS_OUT] = 0;
    /* Clear intermediates */
    for (int i = S_OPCODE; i < D; i++) next->s[i] = 0;

    int pc = (int)cur->s[S_PC];
    if (pc < 0 || pc >= n_instr || cur->s[S_HALT] > 0.5f) { next->s[S_HALT] = 1; return; }

    float tos = cur->s[S_TOS], sos = cur->s[S_SOS];
    float r2 = cur->s[S_R2], r3 = cur->s[S_R3];
    float operand = (float)prog[pc].operand;
    int addr;

    switch (prog[pc].op) {
    case OP_NOP:    next->s[S_PC]=pc+1; break;
    case OP_CONST:  next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=operand; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1; break;
    case OP_NIL:    next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=-1; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1; break;
    case OP_TRUE:   next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=1; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1; break;
    case OP_FALSE:  next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1; break;
    case OP_ADD:    next->s[S_TOS]=tos+sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_SUB:    next->s[S_TOS]=sos-tos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_MUL:    next->s[S_TOS]=tos*sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_NEG:    next->s[S_TOS]=-tos; next->s[S_PC]=pc+1; break;
    case OP_ABS:    next->s[S_TOS]=fabsf(tos); next->s[S_PC]=pc+1; break;
    case OP_EQ:     next->s[S_TOS]=(tos==sos)?1.0f:0.0f; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_LT:     next->s[S_TOS]=(sos<tos)?1.0f:0.0f; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_GT:     next->s[S_TOS]=(sos>tos)?1.0f:0.0f; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_LE:     next->s[S_TOS]=(sos<=tos)?1.0f:0.0f; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_GE:     next->s[S_TOS]=(sos>=tos)?1.0f:0.0f; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_DIV:    next->s[S_TOS]=(tos!=0)?sos/tos:0; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_MOD:    next->s[S_TOS]=(tos!=0)?fmodf(sos,tos):0; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_NOT:    next->s[S_TOS]=(tos==0)?1.0f:0.0f; next->s[S_PC]=pc+1; break;
    case OP_POP:    next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_DUP:    next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1; break;
    case OP_GET_LOCAL:
        addr=(int)operand;
        if(addr>=0&&addr<MEM_SIZE) {
            next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos;
            next->s[S_TOS]=cur->s[S_MEM0+addr]; next->s[S_DEPTH]=cur->s[S_DEPTH]+1;
        }
        next->s[S_PC]=pc+1; break;
    case OP_SET_LOCAL:
        addr=(int)operand;
        if(addr>=0&&addr<MEM_SIZE) next->s[S_MEM0+addr]=tos;
        next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1;
        next->s[S_PC]=pc+1; break;
    case OP_CALL:   /* Set IS_CALL for exec loop to handle frame management */
        next->s[S_IS_CALL]=1; next->s[S_PC]=pc+1; break;
    case OP_RETURN: /* Set IS_RET for exec loop to handle frame restore */
        next->s[S_IS_RET]=1; next->s[S_PC]=pc+1; break;
    case OP_JUMP:   next->s[S_PC]=operand; break;
    case OP_JUMP_IF_FALSE:
        next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1;
        next->s[S_PC]=(tos==0)?operand:(float)(pc+1); break;
    case OP_LOOP:   next->s[S_PC]=operand; break;
    case OP_PRINT:  next->s[S_OUTPUT]=tos; next->s[S_HAS_OUT]=1; next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1; break;
    case OP_HALT:   next->s[S_HALT]=1; break;
    /* All remaining opcodes delegate to exec loop via IS_NATIVE */
    case OP_CONS: case OP_CAR: case OP_CDR: case OP_NULL_P:
    case OP_NATIVE_CALL: case OP_TAIL_CALL:
    case OP_CLOSURE: case OP_GET_UPVALUE: case OP_SET_UPVALUE: case OP_CLOSE_UPVALUE:
    case OP_VEC_CREATE: case OP_VEC_REF: case OP_VEC_SET: case OP_VEC_LEN:
    case OP_STR_REF: case OP_STR_LEN:
    case OP_PAIR_P: case OP_NUM_P: case OP_STR_P: case OP_BOOL_P: case OP_PROC_P: case OP_VEC_P:
    case OP_SET_CAR: case OP_SET_CDR: case OP_POPN:
    case OP_OPEN_CLOSURE: case OP_CALLCC: case OP_INVOKE_CC:
    case OP_PUSH_HANDLER: case OP_POP_HANDLER: case OP_GET_EXN:
    case OP_PACK_REST: case OP_WIND_PUSH: case OP_WIND_POP:
        next->s[S_IS_NATIVE]=1; next->s[S_PC]=pc+1; break;
    default:        next->s[S_PC]=pc+1; break;
    }
}

static int run_reference(const Instr* prog, int n_instr, float* outputs, int max_out) {
    State trace[8192];
    state_init(&trace[0]);
    g_frame_count = 0; g_heap_ptr = 0;
    int t = 0, n_out = 0;
    while (t < 8191 && trace[t].s[S_HALT] < 0.5f) {
        int pc = (int)trace[t].s[S_PC];
        if (pc >= 0 && pc < n_instr) {
            trace[t].s[S_OPCODE] = (float)prog[pc].op;
            trace[t].s[S_OPERAND] = (float)prog[pc].operand;
        }
        execute_step(&trace[t], prog, n_instr, &trace[t+1]);
        exec_loop_postprocess(trace[t+1].s, prog, n_instr);
        if (trace[t+1].s[S_HAS_OUT] > 0.5f && n_out < max_out)
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
    /* GET_LOCAL address resolution: indicator(OPERAND==a) * mem[a] → LOADVAL */
    for (int a = 0; a < MEM_SIZE; a++)
        out[S_LOADVAL] += indicator(x[S_OPERAND], (float)a) * x[S_MEM0+a];
    /* SET_LOCAL store deltas: indicator(OPERAND==a) * (TOS - mem[a]) → STORED0+a */
    for (int a = 0; a < MEM_SIZE; a++)
        out[S_STORED0+a] = indicator(x[S_OPERAND], (float)a) * (x[S_TOS] - x[S_MEM0+a]);
    /* JUMP_IF_FALSE: precompute indicator(TOS==0) * operand and * (PC+1)
     * JUMP_IF_FALSE jumps to operand when TOS==0, falls through when TOS!=0.
     * new_PC = (TOS==0) ? operand : (PC+1)
     *        = (PC+1) + indicator(TOS,0)*(operand - (PC+1))
     * delta[PC] = 1 + ZOPER - ZPC1
     * where ZOPER = indicator(TOS,0)*operand, ZPC1 = indicator(TOS,0)*(PC+1) */
    float iz = indicator(x[S_TOS], 0.0f);
    out[S_ZOPER] = iz * x[S_OPERAND];
    out[S_ZPC1]  = iz * (x[S_PC] + 1.0f);
    /* Comparison precomputes */
    out[S_CMP_EQ] = indicator(x[S_TOS] - x[S_SOS], 0.0f);
    out[S_CMP_LT] = sigmoidf(SCALE * (x[S_TOS] - x[S_SOS] - 0.5f));
    /* ABS precompute: indicator(TOS < 0) * (-2*TOS) */
    out[S_ABS_DELTA] = sigmoidf(SCALE * (-x[S_TOS] - 0.5f)) * (-2.0f * x[S_TOS]);
}

static void layer3_ffn(const float x[D], float out[D]) {
    memset(out, 0, D*sizeof(float));
    float op=x[S_OPCODE], oper=x[S_OPERAND], tos=x[S_TOS], sos=x[S_SOS];
    float r2=x[S_R2], r3=x[S_R3], product=x[S_PRODUCT], lv=x[S_LOADVAL];
    float alive = 1.0f - sigmoidf(SCALE*(x[S_HALT]-0.5f));

    /* Universal: clear output and HAS_OUT */
    out[S_OUTPUT] = -1.0f - x[S_OUTPUT];
    out[S_HAS_OUT] = -x[S_HAS_OUT];
    /* Universal: clear ALL intermediate dims (16-31) */
    for (int i = S_OPCODE; i < D; i++) out[i] += -x[i];

    float g;
    /* OP_NOP (0) */  g=indicator(op,0)*alive; out[S_PC]+=g;
    /* OP_CONST (1) */g=indicator(op,1)*alive; out[S_PC]+=g; out[S_TOS]+=g*(oper-tos); out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
    /* OP_NIL (2) */  g=indicator(op,2)*alive; out[S_PC]+=g; out[S_TOS]+=g*(-1-tos); out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
    /* OP_TRUE (3) */ g=indicator(op,3)*alive; out[S_PC]+=g; out[S_TOS]+=g*(1-tos); out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
    /* OP_FALSE (4) */g=indicator(op,4)*alive; out[S_PC]+=g; out[S_TOS]+=g*(0-tos); out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
    /* OP_POP (5) */  g=indicator(op,5)*alive; out[S_PC]+=g; out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* OP_DUP (6) */  g=indicator(op,6)*alive; out[S_PC]+=g; out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
    /* OP_ADD (7) */  g=indicator(op,7)*alive; out[S_PC]+=g; out[S_TOS]+=g*sos; out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* OP_SUB (8) */  g=indicator(op,8)*alive; out[S_PC]+=g; out[S_TOS]+=g*(sos-2*tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* OP_MUL (9) */  g=indicator(op,9)*alive; out[S_PC]+=g; out[S_TOS]+=g*(product-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* OP_NEG (12) */ g=indicator(op,12)*alive; out[S_PC]+=g; out[S_TOS]+=g*(-2*tos);
    /* OP_ABS (13) */ g=indicator(op,13)*alive; out[S_PC]+=g; out[S_TOS]+=g*x[S_ABS_DELTA];
    /* OP_EQ (14) */  g=indicator(op,14)*alive; out[S_PC]+=g; out[S_TOS]+=g*(x[S_CMP_EQ]-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* OP_LT (15): SOS < TOS → CMP_LT precomputed as sigmoid(SCALE*(TOS-SOS-0.5)) */
    g=indicator(op,15)*alive; out[S_PC]+=g; out[S_TOS]+=g*(x[S_CMP_LT]-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* OP_GT (16): SOS > TOS → 1 - CMP_LT - CMP_EQ */
    g=indicator(op,16)*alive; out[S_PC]+=g; out[S_TOS]+=g*(1.0f-x[S_CMP_LT]-x[S_CMP_EQ]-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* OP_LE (17): SOS <= TOS → CMP_LT + CMP_EQ */
    g=indicator(op,17)*alive; out[S_PC]+=g; out[S_TOS]+=g*(x[S_CMP_LT]+x[S_CMP_EQ]-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* OP_GE (18): SOS >= TOS → 1 - CMP_LT */
    g=indicator(op,18)*alive; out[S_PC]+=g; out[S_TOS]+=g*(1.0f-x[S_CMP_LT]-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* OP_DIV (10): delegate to exec loop — set IS_NATIVE, keep TOS/SOS for exec loop */
    g=indicator(op,10)*alive; out[S_PC]+=g; out[S_IS_NATIVE]+=g;
    /* OP_MOD (11): same pattern as DIV */
    g=indicator(op,11)*alive; out[S_PC]+=g; out[S_IS_NATIVE]+=g;
    /* OP_NOT (19): uses ZOPER trick — encode NOT with operand=1 so ZOPER = indicator(TOS,0)*1 */
    g=indicator(op,19)*alive; out[S_PC]+=g; out[S_TOS]+=g*(x[S_ZOPER]-tos);
    /* OP_GET_LOCAL (20): push mem[operand] — LOADVAL precomputed from OPERAND in layer 2 */
    g=indicator(op,20)*alive;
    out[S_PC]+=g; out[S_TOS]+=g*(lv-tos); out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
    /* OP_SET_LOCAL (21): mem[operand]=TOS, pop — uses precomputed STORED deltas from layer 2 */
    g=indicator(op,21)*alive; {
        out[S_PC]+=g;
        for (int a = 0; a < MEM_SIZE; a++) out[S_MEM0+a] += g * x[S_STORED0+a];
        out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    }
    /* OP_CONS (31): delegate to exec loop */
    g=indicator(op,31)*alive; out[S_PC]+=g; out[S_IS_NATIVE]+=g;
    /* OP_CAR (32): delegate to exec loop */
    g=indicator(op,32)*alive; out[S_PC]+=g; out[S_IS_NATIVE]+=g;
    /* OP_CDR (33): delegate to exec loop */
    g=indicator(op,33)*alive; out[S_PC]+=g; out[S_IS_NATIVE]+=g;
    /* OP_NULL_P (34): delegate to exec loop */
    g=indicator(op,34)*alive; out[S_PC]+=g; out[S_IS_NATIVE]+=g;
    /* OP_GET_UPVALUE (22): delegate */
    g=indicator(op,22)*alive; out[S_PC]+=g; out[S_IS_NATIVE]+=g;
    /* OP_SET_UPVALUE (23): delegate */
    g=indicator(op,23)*alive; out[S_PC]+=g; out[S_IS_NATIVE]+=g;
    /* OP_CLOSURE (24): delegate */
    g=indicator(op,24)*alive; out[S_PC]+=g; out[S_IS_NATIVE]+=g;
    /* OP_TAIL_CALL (26): delegate */
    g=indicator(op,26)*alive; out[S_PC]+=g; out[S_IS_NATIVE]+=g;
    /* OP_NATIVE_CALL (37): delegate */
    g=indicator(op,37)*alive; out[S_PC]+=g; out[S_IS_NATIVE]+=g;
    /* Remaining delegated opcodes (38-62): all set IS_NATIVE + PC++ */
    for (int opc = 38; opc <= 62; opc++) {
        g=indicator(op,(float)opc)*alive; out[S_PC]+=g; out[S_IS_NATIVE]+=g;
    }
    /* OP_CALL (25): set IS_CALL flag for exec loop, PC++ */
    g=indicator(op,25)*alive; out[S_PC]+=g; out[S_IS_CALL]+=g;
    /* OP_RETURN (27): set IS_RET flag for exec loop, PC++ */
    g=indicator(op,27)*alive; out[S_PC]+=g; out[S_IS_RET]+=g;
    /* OP_JUMP (28) */
    g=indicator(op,28)*alive; out[S_PC]+=g*(oper-x[S_PC]);
    /* OP_JUMP_IF_FALSE (29): if TOS==0 goto operand, else PC+1. Pop TOS.
     * delta[PC] = 1 + ZOPER - ZPC1 (inverted from v1's JUMP_IF) */
    g=indicator(op,29)*alive;
    out[S_PC]+=g*(1.0f + x[S_ZOPER] - x[S_ZPC1]);
    out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* OP_LOOP (30): unconditional backward jump */
    g=indicator(op,30)*alive; out[S_PC]+=g*(oper-x[S_PC]);
    /* OP_PRINT (35): output = TOS, pop */
    g=indicator(op,35)*alive; out[S_PC]+=g; out[S_OUTPUT]+=g*(tos+1); out[S_HAS_OUT]+=g;
    out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    /* OP_HALT (36) */
    g=indicator(op,36)*alive; out[S_HALT]+=g;
}

/* Post-process: handle IS_NATIVE, IS_CALL, IS_RET flags.
 * The transformer set flags and PC++. The exec loop performs the actual operations. */
static void exec_loop_postprocess(float x[D], const Instr* prog, int n_instr) {
    /* IS_NATIVE: DIV, MOD, etc. */
    if (x[S_IS_NATIVE] > 0.5f) {
        int pc = (int)roundf(x[S_PC]) - 1;
        if (pc >= 0 && pc < n_instr) {
            int opcode = prog[pc].op;
            float tos = x[S_TOS], sos = x[S_SOS];
            float r2 = x[S_R2], r3 = x[S_R3];
            if (opcode == OP_DIV) {
                x[S_TOS] = (tos != 0) ? sos / tos : 0;
                x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                x[S_DEPTH] -= 1;
            } else if (opcode == OP_MOD) {
                x[S_TOS] = (tos != 0) ? fmodf(sos, tos) : 0;
                x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                x[S_DEPTH] -= 1;
            } else if (opcode == OP_CONS) {
                /* CONS: allocate pair on heap. TOS=cdr, SOS=car → push heap ptr */
                if (g_heap_ptr + 2 <= HEAP_SIZE) {
                    int ptr = g_heap_ptr;
                    g_heap[g_heap_ptr++] = sos;  /* car */
                    g_heap[g_heap_ptr++] = tos;  /* cdr */
                    x[S_TOS] = (float)ptr;       /* pair reference = heap index */
                    x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                    x[S_DEPTH] -= 1;
                }
            } else if (opcode == OP_CAR) {
                /* CAR: TOS is pair ptr → replace with car */
                int ptr = (int)tos;
                if (ptr >= 0 && ptr + 1 < HEAP_SIZE)
                    x[S_TOS] = g_heap[ptr];      /* car */
            } else if (opcode == OP_CDR) {
                /* CDR: TOS is pair ptr → replace with cdr */
                int ptr = (int)tos;
                if (ptr >= 0 && ptr + 1 < HEAP_SIZE)
                    x[S_TOS] = g_heap[ptr + 1];  /* cdr */
            } else if (opcode == OP_NULL_P) {
                /* NULL_P: TOS → 1 if nil (-1), else 0 */
                x[S_TOS] = (tos == -1.0f) ? 1.0f : 0.0f;
            } else if (opcode == OP_SET_CAR) {
                /* SET_CAR: TOS=val, SOS=pair → mutate car */
                int ptr = (int)sos;
                if (ptr >= 0 && ptr < HEAP_SIZE) g_heap[ptr] = tos;
                x[S_TOS] = r2; x[S_SOS] = r3; x[S_R2] = 0; x[S_R3] = 0;
                x[S_DEPTH] -= 2;
            } else if (opcode == OP_SET_CDR) {
                /* SET_CDR: TOS=val, SOS=pair → mutate cdr */
                int ptr = (int)sos;
                if (ptr >= 0 && ptr + 1 < HEAP_SIZE) g_heap[ptr + 1] = tos;
                x[S_TOS] = r2; x[S_SOS] = r3; x[S_R2] = 0; x[S_R3] = 0;
                x[S_DEPTH] -= 2;
            } else if (opcode == OP_PAIR_P) {
                /* PAIR_P: TOS → 1 if heap pair ptr, else 0 */
                x[S_TOS] = (tos >= 0 && (int)tos + 1 < g_heap_ptr) ? 1.0f : 0.0f;
            } else if (opcode == OP_NUM_P) {
                /* NUM_P: numbers are non-negative, non-pair floats */
                x[S_TOS] = 1.0f; /* simplified: all values are numbers in this VM */
            } else if (opcode == OP_POPN) {
                /* POPN: pop N values below TOS, keeping TOS */
                int count = (int)(pc >= 0 && pc < n_instr ? prog[pc].operand : 0);
                /* simplified: just adjust depth */
                x[S_DEPTH] -= count;
            } else if (opcode == OP_TAIL_CALL) {
                /* TAIL_CALL: like CALL but reuses current frame (no frame push) */
                int argc = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                float func_pc = tos;
                float args[4] = {sos, r2, r3, 0};
                for (int i = 0; i < MEM_SIZE; i++) x[S_MEM0+i] = 0;
                for (int i = 0; i < argc && i < MEM_SIZE; i++)
                    x[S_MEM0+i] = args[i];
                x[S_PC] = func_pc;
                x[S_TOS] = 0; x[S_SOS] = 0; x[S_R2] = 0; x[S_R3] = 0;
                x[S_DEPTH] -= (1 + argc);
            }
            /* ── Closures ── */
            else if (opcode == OP_CLOSURE) {
                /* CLOSURE: operand = func_pc. Allocate closure on heap: [func_pc, n_upvals, upval0, ...] */
                int func_entry = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                if (g_heap_ptr + 2 <= HEAP_SIZE) {
                    int cptr = g_heap_ptr;
                    g_heap[g_heap_ptr++] = (float)func_entry;  /* entry point */
                    g_heap[g_heap_ptr++] = 0;                  /* n_upvals (set by OPEN_CLOSURE) */
                    /* Push closure ptr, shift down */
                    x[S_R3] = x[S_R2]; x[S_R2] = x[S_SOS]; x[S_SOS] = x[S_TOS];
                    x[S_TOS] = (float)cptr;
                    x[S_DEPTH] += 1;
                }
            } else if (opcode == OP_GET_UPVALUE) {
                /* GET_UPVALUE: operand = upvalue index. Read from current closure's upvalue array. */
                /* Simplified: treat as GET_LOCAL for now (closure upvalues stored in MEM) */
                int idx = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                if (idx >= 0 && idx < MEM_SIZE) {
                    x[S_R3] = x[S_R2]; x[S_R2] = x[S_SOS]; x[S_SOS] = x[S_TOS];
                    x[S_TOS] = x[S_MEM0 + idx];
                    x[S_DEPTH] += 1;
                }
            } else if (opcode == OP_SET_UPVALUE) {
                /* SET_UPVALUE: operand = index. Store TOS to upvalue slot, pop. */
                int idx = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                if (idx >= 0 && idx < MEM_SIZE) x[S_MEM0 + idx] = tos;
                x[S_TOS] = sos; x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                x[S_DEPTH] -= 1;
            } else if (opcode == OP_CLOSE_UPVALUE) {
                /* CLOSE_UPVALUE: no-op in this simplified VM (upvalues are in MEM) */
            } else if (opcode == OP_OPEN_CLOSURE) {
                /* OPEN_CLOSURE: no-op in simplified VM */
            }
            /* ── Vectors ── */
            else if (opcode == OP_VEC_CREATE) {
                /* VEC_CREATE: operand = count. Pop count values, create vector on heap. */
                int count = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                if (g_heap_ptr + count + 1 <= HEAP_SIZE) {
                    int vptr = g_heap_ptr;
                    g_heap[g_heap_ptr++] = (float)count;  /* length */
                    /* Pop values from stack into vector (TOS is last element) */
                    float vals[4] = {tos, sos, r2, r3};
                    for (int i = 0; i < count && i < 4; i++)
                        g_heap[g_heap_ptr++] = vals[count - 1 - i];
                    /* Adjust stack: pop count values, push vector ptr */
                    x[S_TOS] = (float)vptr;
                    x[S_SOS] = 0; x[S_R2] = 0; x[S_R3] = 0;
                    x[S_DEPTH] -= (count - 1); /* pop count, push 1 */
                }
            } else if (opcode == OP_VEC_REF) {
                /* VEC_REF: TOS=index, SOS=vector_ptr → push vector[index] */
                int vptr = (int)sos;
                int idx = (int)tos;
                if (vptr >= 0 && vptr < HEAP_SIZE) {
                    int len = (int)g_heap[vptr];
                    if (idx >= 0 && idx < len)
                        x[S_TOS] = g_heap[vptr + 1 + idx];
                }
                x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                x[S_DEPTH] -= 1;
            } else if (opcode == OP_VEC_SET) {
                /* VEC_SET: TOS=value, SOS=index, R2=vector_ptr → mutate */
                int vptr = (int)r2;
                int idx = (int)sos;
                if (vptr >= 0 && vptr < HEAP_SIZE) {
                    int len = (int)g_heap[vptr];
                    if (idx >= 0 && idx < len)
                        g_heap[vptr + 1 + idx] = tos;
                }
                x[S_TOS] = r3; x[S_SOS] = 0; x[S_R2] = 0; x[S_R3] = 0;
                x[S_DEPTH] -= 3;
            } else if (opcode == OP_VEC_LEN) {
                /* VEC_LEN: TOS=vector_ptr → push length */
                int vptr = (int)tos;
                if (vptr >= 0 && vptr < HEAP_SIZE)
                    x[S_TOS] = g_heap[vptr];
            }
            /* ── Strings (simplified: stored as vectors of char codes) ── */
            else if (opcode == OP_STR_REF) {
                /* STR_REF: TOS=index, SOS=string_ptr → char at index */
                int sptr = (int)sos;
                int idx = (int)tos;
                if (sptr >= 0 && sptr < HEAP_SIZE) {
                    int len = (int)g_heap[sptr];
                    if (idx >= 0 && idx < len)
                        x[S_TOS] = g_heap[sptr + 1 + idx];
                }
                x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                x[S_DEPTH] -= 1;
            } else if (opcode == OP_STR_LEN) {
                /* STR_LEN: TOS=string_ptr → push length */
                int sptr = (int)tos;
                if (sptr >= 0 && sptr < HEAP_SIZE)
                    x[S_TOS] = g_heap[sptr];
            }
            /* ── Type predicates ── */
            else if (opcode == OP_STR_P || opcode == OP_BOOL_P ||
                     opcode == OP_PROC_P || opcode == OP_VEC_P) {
                /* Simplified type predicates — always return 0 (no type tags in float VM) */
                x[S_TOS] = 0.0f;
            }
            /* ── Exceptions ── */
            else if (opcode == OP_PUSH_HANDLER) {
                /* PUSH_HANDLER: operand = handler_pc. Save handler info. */
                /* Simplified: store handler_pc in a global */
            } else if (opcode == OP_POP_HANDLER) {
                /* POP_HANDLER: remove topmost exception handler */
            } else if (opcode == OP_GET_EXN) {
                /* GET_EXN: push current exception value (0 if none) */
                x[S_R3] = x[S_R2]; x[S_R2] = x[S_SOS]; x[S_SOS] = x[S_TOS];
                x[S_TOS] = 0; /* no exception value in simplified VM */
                x[S_DEPTH] += 1;
            }
            /* ── Continuations ── */
            else if (opcode == OP_CALLCC) {
                /* CALLCC: simplified — just call the function with a dummy continuation */
                /* In the full VM, this captures the full continuation. */
                /* For now, treat like CALL with a continuation object = -2 sentinel */
                float func_pc_cc = tos;
                if (g_frame_count < 64) {
                    CallFrame* f = &g_frames[g_frame_count];
                    f->return_pc = x[S_PC];
                    for (int i = 0; i < MEM_SIZE; i++) f->saved_mem[i] = x[S_MEM0+i];
                    f->saved_tos = r2; f->saved_sos = r3;
                    f->saved_r2 = 0; f->saved_r3 = 0;
                    f->saved_depth = x[S_DEPTH] - 2;
                    g_frame_count++;
                }
                x[S_MEM0] = -2.0f; /* continuation sentinel as arg */
                x[S_PC] = func_pc_cc;
                x[S_TOS] = 0; x[S_SOS] = 0; x[S_R2] = 0; x[S_R3] = 0;
                x[S_DEPTH] = 0;
            } else if (opcode == OP_INVOKE_CC) {
                /* INVOKE_CC: invoke continuation with value. Like RETURN. */
                float retval_cc = tos;
                if (g_frame_count > 0) {
                    g_frame_count--;
                    CallFrame* f = &g_frames[g_frame_count];
                    x[S_PC] = f->return_pc;
                    for (int i = 0; i < MEM_SIZE; i++) x[S_MEM0+i] = f->saved_mem[i];
                    x[S_TOS] = retval_cc;
                    x[S_SOS] = f->saved_tos; x[S_R2] = f->saved_sos;
                    x[S_R3] = f->saved_r2;
                    x[S_DEPTH] = f->saved_depth + 1;
                }
            }
            /* ── Variadic / dynamic-wind ── */
            else if (opcode == OP_PACK_REST) {
                /* PACK_REST: operand=n_fixed. Cons remaining args into list. */
                /* Simplified: no-op (variadic args handled by compiler) */
            } else if (opcode == OP_WIND_PUSH || opcode == OP_WIND_POP) {
                /* Dynamic wind: no-op in simplified VM */
            }
        }
        x[S_IS_NATIVE] = 0;
    }

    /* IS_CALL: push call frame, jump to function entry
     * Convention: CALL operand = argc
     * Stack before CALL: [func_pc, arg0, arg1, ..., argN-1, ...]
     *   TOS = func_pc (function entry address)
     *   SOS..R3 = args (pushed before the function ref)
     * After CALL: save return PC and caller's memory, set PC to func_pc,
     *   store args in MEM slots, clear stack cache. */
    if (x[S_IS_CALL] > 0.5f) {
        int pc = (int)roundf(x[S_PC]) - 1;
        int argc = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
        float func_pc = x[S_TOS];
        /* If func_pc looks like a heap closure pointer, dereference to get entry point */
        int fptr = (int)func_pc;
        if (fptr >= 0 && fptr + 1 < g_heap_ptr && g_heap[fptr] < 10000) {
            /* Check if this is a closure (heap[ptr] = entry_pc, heap[ptr+1] = n_upvals) */
            float candidate = g_heap[fptr];
            if (candidate >= 0 && candidate < n_instr) func_pc = candidate;
        }

        /* Save call frame: return address, memory, and caller's stack below args */
        if (g_frame_count < 64) {
            CallFrame* f = &g_frames[g_frame_count];
            f->return_pc = x[S_PC]; /* already PC+1 */
            for (int i = 0; i < MEM_SIZE; i++)
                f->saved_mem[i] = x[S_MEM0+i];
            /* Save caller's stack state below the func_pc + args.
             * Stack: [... caller_vals ... arg0 arg1 ... argN func_pc]
             * TOS=func_pc, SOS=last_arg_or_caller, R2=..., R3=...
             * For argc=1: SOS=arg0, R2=caller_val0, R3=caller_val1
             * We save R2/R3 (the caller's values below the args) */
            if (argc == 0) {
                f->saved_tos = x[S_SOS]; f->saved_sos = x[S_R2];
                f->saved_r2 = x[S_R3]; f->saved_r3 = 0;
            } else if (argc == 1) {
                f->saved_tos = x[S_R2]; f->saved_sos = x[S_R3];
                f->saved_r2 = 0; f->saved_r3 = 0;
            } else {
                f->saved_tos = x[S_R3]; f->saved_sos = 0;
                f->saved_r2 = 0; f->saved_r3 = 0;
            }
            f->saved_depth = x[S_DEPTH] - (1 + argc);
            g_frame_count++;
        }

        /* Set up callee: args go to MEM0..MEM(argc-1) */
        float args[4] = {x[S_SOS], x[S_R2], x[S_R3], 0};
        for (int i = 0; i < MEM_SIZE; i++) x[S_MEM0+i] = 0;
        for (int i = 0; i < argc && i < MEM_SIZE; i++)
            x[S_MEM0+i] = args[i];

        x[S_PC] = func_pc;
        x[S_TOS] = 0; x[S_SOS] = 0; x[S_R2] = 0; x[S_R3] = 0;
        x[S_DEPTH] = 0;
        x[S_IS_CALL] = 0;
    }

    /* IS_RET: pop call frame, restore caller state, push return value
     * TOS = return value */
    if (x[S_IS_RET] > 0.5f) {
        float retval = x[S_TOS];

        if (g_frame_count > 0) {
            g_frame_count--;
            CallFrame* f = &g_frames[g_frame_count];
            x[S_PC] = f->return_pc;
            for (int i = 0; i < MEM_SIZE; i++)
                x[S_MEM0+i] = f->saved_mem[i];
            /* Restore caller's stack with return value pushed on top */
            x[S_TOS] = retval;
            x[S_SOS] = f->saved_tos;
            x[S_R2] = f->saved_sos;
            x[S_R3] = f->saved_r2;
            x[S_DEPTH] = f->saved_depth + 1; /* +1 for return value */
        }
        x[S_IS_RET] = 0;
    }
}

static int run_simulated(const Instr* prog, int n_instr, float* outputs, int max_out) {
    float pe[256][D];
    for (int p = 0; p < n_instr && p < 256; p++) embed_instruction(&prog[p], p, pe[p]);
    float state[D]; memset(state, 0, sizeof(state)); state[S_OUTPUT] = -1;
    g_frame_count = 0; g_heap_ptr = 0;
    int n_out = 0;
    for (int step = 0; step < 8192; step++) {
        float x[D]; memcpy(x, state, sizeof(x));
        float tmp[D];
        layer0_attention(x, pe, n_instr, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
        layer1_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
        layer2_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
        layer3_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
        /* Layer 4: no-op for now */
        exec_loop_postprocess(x, prog, n_instr);
        if (x[S_HAS_OUT] > 0.5f && n_out < max_out) outputs[n_out++] = x[S_OUTPUT];
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

#define W(mat, r, c, cols) ((mat)[(r) * (cols) + (c)])

/* Add a gated neuron pair for opcode dispatch.
 * Implements: indicator(opcode==op_id) * alive * linear_combination → out_dim */
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

/* Unconditional (always-on) neuron */
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

    /* ── Layer 0: Attention (instruction fetch) ── */
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

    /* ── Layer 1: Product FFN (SQUARE activation) ── */
    {
        w->ff_type[1] = 1;
        W(w->ff_up[1], S_TOS, 0, FFN_DIM) = 1; W(w->ff_up[1], S_SOS, 0, FFN_DIM) = 1;
        W(w->ff_up[1], S_TOS, 1, FFN_DIM) = 1;
        W(w->ff_up[1], S_SOS, 2, FFN_DIM) = 1;
        W(w->ff_down[1], 0, S_PRODUCT, D) =  0.5f;
        W(w->ff_down[1], 1, S_PRODUCT, D) = -0.5f;
        W(w->ff_down[1], 2, S_PRODUCT, D) = -0.5f;
    }

    /* ── Layer 2: Preprocessing (gated FFN) ── */
    {
        w->ff_type[2] = 2;
        int n = 0;

        /* GET_LOCAL address resolution: indicator(OPERAND==a) * mem[a] → LOADVAL */
        for (int a = 0; a < MEM_SIZE; a++) {
            W(w->ff_gate[2], S_OPERAND, n, FFN_DIM) = SCALE;
            w->ff_gate_b[2][n] = SCALE * (-(float)a + 0.5f);
            W(w->ff_up[2], S_MEM0+a, n, FFN_DIM) = 1.0f;
            W(w->ff_down[2], n, S_LOADVAL, D) = 1.0f;
            n++;
            W(w->ff_gate[2], S_OPERAND, n, FFN_DIM) = SCALE;
            w->ff_gate_b[2][n] = SCALE * (-(float)a - 0.5f);
            W(w->ff_up[2], S_MEM0+a, n, FFN_DIM) = 1.0f;
            W(w->ff_down[2], n, S_LOADVAL, D) = -1.0f;
            n++;
        }

        /* SET_LOCAL store deltas: indicator(OPERAND==a) * (TOS - mem[a]) → STORED0+a */
        for (int a = 0; a < MEM_SIZE; a++) {
            W(w->ff_gate[2], S_OPERAND, n, FFN_DIM) = SCALE;
            w->ff_gate_b[2][n] = SCALE * (-(float)a + 0.5f);
            W(w->ff_up[2], S_TOS, n, FFN_DIM) = 1.0f;
            W(w->ff_up[2], S_MEM0+a, n, FFN_DIM) = -1.0f;
            W(w->ff_down[2], n, S_STORED0+a, D) = 1.0f;
            n++;
            W(w->ff_gate[2], S_OPERAND, n, FFN_DIM) = SCALE;
            w->ff_gate_b[2][n] = SCALE * (-(float)a - 0.5f);
            W(w->ff_up[2], S_TOS, n, FFN_DIM) = 1.0f;
            W(w->ff_up[2], S_MEM0+a, n, FFN_DIM) = -1.0f;
            W(w->ff_down[2], n, S_STORED0+a, D) = -1.0f;
            n++;
        }

        /* JUMP_IF_FALSE zero case: indicator(TOS==0) * operand → ZOPER */
        W(w->ff_gate[2], S_TOS, n, FFN_DIM) = SCALE;
        w->ff_gate_b[2][n] = SCALE * 0.5f;
        W(w->ff_up[2], S_OPERAND, n, FFN_DIM) = 1.0f;
        W(w->ff_down[2], n, S_ZOPER, D) = 1.0f;
        n++;
        W(w->ff_gate[2], S_TOS, n, FFN_DIM) = SCALE;
        w->ff_gate_b[2][n] = SCALE * (-0.5f);
        W(w->ff_up[2], S_OPERAND, n, FFN_DIM) = 1.0f;
        W(w->ff_down[2], n, S_ZOPER, D) = -1.0f;
        n++;

        /* JUMP_IF_FALSE zero case: indicator(TOS==0) * (PC+1) → ZPC1 */
        W(w->ff_gate[2], S_TOS, n, FFN_DIM) = SCALE;
        w->ff_gate_b[2][n] = SCALE * 0.5f;
        W(w->ff_up[2], S_PC, n, FFN_DIM) = 1.0f;
        w->ff_up_b[2][n] = 1.0f;
        W(w->ff_down[2], n, S_ZPC1, D) = 1.0f;
        n++;
        W(w->ff_gate[2], S_TOS, n, FFN_DIM) = SCALE;
        w->ff_gate_b[2][n] = SCALE * (-0.5f);
        W(w->ff_up[2], S_PC, n, FFN_DIM) = 1.0f;
        w->ff_up_b[2][n] = 1.0f;
        W(w->ff_down[2], n, S_ZPC1, D) = -1.0f;
        n++;

        /* CMP_EQ: indicator(TOS - SOS == 0) → two neurons on (TOS - SOS) */
        W(w->ff_gate[2], S_TOS, n, FFN_DIM) = SCALE;
        W(w->ff_gate[2], S_SOS, n, FFN_DIM) = -SCALE;
        w->ff_gate_b[2][n] = SCALE * 0.5f;
        w->ff_up_b[2][n] = 1.0f;
        W(w->ff_down[2], n, S_CMP_EQ, D) = 1.0f;
        n++;
        W(w->ff_gate[2], S_TOS, n, FFN_DIM) = SCALE;
        W(w->ff_gate[2], S_SOS, n, FFN_DIM) = -SCALE;
        w->ff_gate_b[2][n] = SCALE * (-0.5f);
        w->ff_up_b[2][n] = 1.0f;
        W(w->ff_down[2], n, S_CMP_EQ, D) = -1.0f;
        n++;

        /* CMP_LT: sigmoid(SCALE*(TOS - SOS - 0.5)) → 1 when SOS < TOS */
        W(w->ff_gate[2], S_TOS, n, FFN_DIM) = SCALE;
        W(w->ff_gate[2], S_SOS, n, FFN_DIM) = -SCALE;
        w->ff_gate_b[2][n] = SCALE * (-0.5f);
        w->ff_up_b[2][n] = 1.0f;
        W(w->ff_down[2], n, S_CMP_LT, D) = 1.0f;
        n++;

        /* ABS_DELTA: indicator(TOS < 0) * (-2*TOS)
         * gate = sigmoid(SCALE*(-TOS - 0.5)) ≈ 1 when TOS < 0 */
        W(w->ff_gate[2], S_TOS, n, FFN_DIM) = -SCALE;
        w->ff_gate_b[2][n] = SCALE * (-0.5f);
        W(w->ff_up[2], S_TOS, n, FFN_DIM) = -2.0f;
        W(w->ff_down[2], n, S_ABS_DELTA, D) = 1.0f;
        n++;

        printf("[WEIGHT_GEN] Layer 2: %d neurons\n", n);
    }

    /* ── Layer 3: Execution (gated FFN) ── */
    {
        const int L = 3;
        w->ff_type[L] = 2;
        int n = 0;

        /* Universal: clear output to -1 */
        n = add_unconditional(w, L, n, S_OUTPUT, -1.0f, -1.0f, S_OUTPUT, 1.0f);
        /* Universal: clear HAS_OUT */
        n = add_unconditional(w, L, n, S_HAS_OUT, -1.0f, 0, S_HAS_OUT, 1.0f);
        /* Universal: clear ALL intermediate dims (16-31) */
        for (int d = S_OPCODE; d < D; d++) {
            n = add_unconditional(w, L, n, d, -1.0f, 0, d, 1.0f);
        }

        /* OP_NOP (0): PC += 1 */
        n = add_gated_pair(w,L,n, 0, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);

        /* OP_CONST (1): TOS=oper, push down, depth++, PC++ */
        n = add_gated_pair(w,L,n, 1, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 1, S_OPERAND,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 1, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 1, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 1, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 1, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);

        /* OP_NIL (2): push -1, same pattern as CONST */
        n = add_gated_pair(w,L,n, 2, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 2, S_TOS,-1,-1,0,-1,0,-1,0, -1.0f, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 2, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 2, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 2, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 2, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);

        /* OP_TRUE (3): push 1 */
        n = add_gated_pair(w,L,n, 3, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 3, S_TOS,-1,-1,0,-1,0,-1,0, 1.0f, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 3, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 3, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 3, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 3, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);

        /* OP_FALSE (4): push 0 */
        n = add_gated_pair(w,L,n, 4, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 4, S_TOS,-1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 4, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 4, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 4, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 4, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);

        /* OP_POP (5): shift up, depth-- */
        n = add_gated_pair(w,L,n, 5, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 5, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 5, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 5, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 5, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 5, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_DUP (6): push TOS copy, shift down */
        n = add_gated_pair(w,L,n, 6, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 6, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 6, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 6, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 6, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);

        /* OP_ADD (7): TOS+=SOS, shift up, depth--, PC++ */
        n = add_gated_pair(w,L,n, 7, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 7, S_SOS,1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 7, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 7, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 7, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 7, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_SUB (8): TOS=SOS-TOS (delta=SOS-2*TOS), shift up */
        n = add_gated_pair(w,L,n, 8, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 8, S_SOS,1,S_TOS,-2,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 8, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 8, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 8, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 8, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_MUL (9): TOS=TOS*SOS=product, shift up */
        n = add_gated_pair(w,L,n, 9, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 9, S_PRODUCT,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 9, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 9, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 9, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 9, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_NEG (12): TOS = -TOS, PC++ */
        n = add_gated_pair(w,L,n, 12, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 12, S_TOS,-2,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);

        /* OP_ABS (13): TOS = |TOS|, PC++ — uses ABS_DELTA precomputed in layer 2 */
        n = add_gated_pair(w,L,n, 13, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 13, S_ABS_DELTA,1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);

        /* OP_EQ (14): TOS = (TOS==SOS), binary comparison, shift up */
        n = add_gated_pair(w,L,n, 14, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 14, S_CMP_EQ,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 14, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 14, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 14, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 14, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_LT (15): TOS = (SOS < TOS) = CMP_LT */
        n = add_gated_pair(w,L,n, 15, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 15, S_CMP_LT,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 15, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 15, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 15, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 15, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_GT (16): TOS = (SOS > TOS) = 1 - CMP_LT - CMP_EQ */
        n = add_gated_pair(w,L,n, 16, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 16, S_CMP_LT,-1,S_CMP_EQ,-1,S_TOS,-1,-1,0, 1.0f, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 16, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 16, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 16, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 16, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_LE (17): TOS = (SOS <= TOS) = CMP_LT + CMP_EQ */
        n = add_gated_pair(w,L,n, 17, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 17, S_CMP_LT,1,S_CMP_EQ,1,S_TOS,-1,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 17, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 17, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 17, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 17, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_GE (18): TOS = (SOS >= TOS) = 1 - CMP_LT */
        n = add_gated_pair(w,L,n, 18, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 18, S_CMP_LT,-1,S_TOS,-1,-1,0,-1,0, 1.0f, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 18, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 18, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 18, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 18, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_NOT (19): TOS = (TOS==0) ? 1 : 0. Unary, no stack shift.
         * Approach: delta = indicator(TOS,0) - TOS.
         * indicator(TOS,0) is already available via the sigmoid pair approach,
         * but we need it as a linear combination for the gated pair.
         * Use: up = -TOS + indicator(TOS,0) ... but indicator is nonlinear.
         * Solution: precompute in layer 2 or use two separate gated pairs.
         * Actually: the zero indicator for TOS is available via the ZOPER mechanism.
         * We can use: delta = (ZOPER/OPERAND when OPERAND!=0)... no, ZOPER depends on operand.
         *
         * Simpler: NOT on TOS uses the same sigmoid-pair as indicator(TOS, 0).
         * We compute indicator(TOS, 0) directly in the gate:
         * gated_pair1: gate=indicator(op,19)*indicator(TOS,0)*alive, up=1, out=TOS (sets TOS=1 when TOS==0)
         * But we can't nest indicators in a single gate neuron.
         *
         * Best approach: precompute indicator(TOS,0) in layer 2 into a spare dim.
         * For now, use the simulated path (which works) and handle NOT via
         * a two-step approach in the weight matrix: clear TOS, then conditionally set to 1.
         *
         * Actually simplest: use ubias trick.
         * NOT = "TOS becomes 1 if TOS==0, else 0"
         * delta_TOS = result - TOS where result = indicator(TOS,0)
         * We can split this: first clear TOS (delta = -TOS), then add indicator(TOS,0).
         * But both happen in the same gated pair, which gates on opcode==19.
         * Inside the pair, we need -TOS + indicator(TOS,0) which requires the indicator.
         *
         * Resolution: use 4 neurons instead of 2. First pair clears TOS:
         * pair1: gate=indicator(op,19)*alive, up=-TOS, out=TOS → clears TOS
         * Then pair2 ADDS 1 conditional on (opcode==19 AND TOS==0):
         * pair2: gate=sigmoid(S*(op-19+0.5))*sigmoid(S*(-TOS+0.5)), up=1, out=TOS
         * But that's a product of sigmoids in the gate, which we can't do with a single gate.
         *
         * Final approach: delegate NOT to reference/simulated, add it to weight matrix
         * using the precomputed zero indicator from ZOPER. Wait — ZOPER = indicator(TOS,0) * OPERAND.
         * If OPERAND happens to be nonzero for NOT, we can recover indicator(TOS,0) = ZOPER/OPERAND.
         * But operand for NOT is 0 (unary, no operand). So ZOPER = 0 always. Useless.
         *
         * OK — add a new precompute: IND_TOS_ZERO into S_ABS_DELTA or reuse existing.
         * Actually, I'll add a precompute for indicator(TOS,0) into the layer 2 path
         * and store it... but we're out of intermediate dims.
         *
         * Pragmatic: NOT is used in control flow. For the weight matrix, we handle it
         * by using the same approach as ABS: precompute in layer 2.
         * We can reuse S_ABS_DELTA for a different purpose when ABS isn't executing,
         * but that's ugly. Better: for NOT, we use the fact that
         * indicator(TOS,0) = sigmoid(S*(0-TOS+0.5)) - sigmoid(S*(0-TOS-0.5))
         *                   = sigmoid(S*(-TOS+0.5)) - sigmoid(S*(-TOS-0.5))
         * This is exactly: gate1 = sigmoid(-S*TOS + S*0.5), gate2 = sigmoid(-S*TOS - S*0.5)
         * And the up_value is 1.0 (bias).
         * So: we add neurons directly in layer 3 that don't use add_gated_pair.
         * We need 4 neurons total:
         * n1: gate = sigmoid(S*(op-19+0.5))*... nope, we can't combine.
         *
         * Let's just use the approach from the simulated path: compute NOT inline.
         * For the matrix path, NOT = indicator(TOS,0) requires 2 extra neurons
         * outside the gated_pair framework. We'll manually wire them.
         */
        /* OP_NOT (19): step 1 — clear TOS, step 2 — set to 1 if was 0.
         * Uses CMP_EQ which computes indicator(TOS-SOS, 0). But NOT is unary.
         * Precompute: we reuse the layer 2 ABS indicator for the negative case.
         * For NOT, we need indicator(TOS, 0). Approach: in layer 3, clear TOS
         * then conditionally add 1. Use ZOPER trick: when operand=1 and we
         * encode NOT as {OP_NOT, 1}, ZOPER = indicator(TOS,0) * 1 = indicator(TOS,0).
         * Then delta_TOS = ZOPER - TOS. This WORKS if we make NOT's operand = 1.
         */
        n = add_gated_pair(w,L,n, 19, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 19, S_ZOPER,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);

        /* OP_GET_LOCAL (20): push mem[operand] — LOADVAL precomputed from OPERAND */
        n = add_gated_pair(w,L,n, 20, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 20, S_LOADVAL,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 20, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 20, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 20, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 20, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);

        /* OP_SET_LOCAL (21): mem[operand]=TOS, pop — uses precomputed STORED deltas */
        n = add_gated_pair(w,L,n, 21, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        for (int a = 0; a < MEM_SIZE; a++) {
            n = add_gated_pair(w,L,n, 21, S_STORED0+a,1,-1,0,-1,0,-1,0, 0, S_MEM0+a, 1.0f);
        }
        n = add_gated_pair(w,L,n, 21, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 21, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 21, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 21, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 21, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_CALL (25): set IS_CALL flag, PC++ */
        n = add_gated_pair(w,L,n, 25, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 25, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_CALL, 1.0f);

        /* OP_CONS (31): IS_NATIVE, PC++ */
        n = add_gated_pair(w,L,n, 31, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 31, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);
        /* OP_CAR (32): IS_NATIVE, PC++ */
        n = add_gated_pair(w,L,n, 32, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 32, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);
        /* OP_CDR (33): IS_NATIVE, PC++ */
        n = add_gated_pair(w,L,n, 33, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 33, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);
        /* OP_NULL_P (34): IS_NATIVE, PC++ */
        n = add_gated_pair(w,L,n, 34, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 34, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);
        /* OP_GET_UPVALUE (22): IS_NATIVE */
        n = add_gated_pair(w,L,n, 22, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 22, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);
        /* OP_SET_UPVALUE (23): IS_NATIVE */
        n = add_gated_pair(w,L,n, 23, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 23, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);
        /* OP_CLOSURE (24): IS_NATIVE */
        n = add_gated_pair(w,L,n, 24, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 24, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);
        /* OP_TAIL_CALL (26): IS_NATIVE */
        n = add_gated_pair(w,L,n, 26, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 26, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);
        /* OP_NATIVE_CALL (37): IS_NATIVE, PC++ */
        n = add_gated_pair(w,L,n, 37, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 37, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);
        /* Remaining delegated opcodes (38-62): all IS_NATIVE + PC++ */
        for (int opc = 38; opc <= 62; opc++) {
            n = add_gated_pair(w,L,n, opc, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
            n = add_gated_pair(w,L,n, opc, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);
        }

        /* OP_RETURN (27): set IS_RET flag, PC++ — exec loop handles frame pop */
        n = add_gated_pair(w,L,n, 27, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 27, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_RET, 1.0f);

        /* OP_JUMP (28) */
        n = add_gated_pair(w,L,n, 28, S_OPERAND,1,S_PC,-1,-1,0,-1,0, 0, S_PC, 1.0f);

        /* OP_JUMP_IF_FALSE (29): pop TOS, if TOS==0 goto operand, else PC+1 */
        n = add_gated_pair(w,L,n, 29, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 29, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 29, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 29, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 29, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        /* JUMP_IF_FALSE PC: delta[PC] = 1 + ZOPER - ZPC1 */
        n = add_gated_pair(w,L,n, 29,
                           S_ZOPER, 1, S_ZPC1, -1, -1, 0, -1, 0,
                           1.0f, S_PC, 1.0f);

        /* OP_LOOP (30): backward jump — same as OP_JUMP */
        n = add_gated_pair(w,L,n, 30, S_OPERAND,1,S_PC,-1,-1,0,-1,0, 0, S_PC, 1.0f);

        /* OP_PRINT (35): output = TOS, HAS_OUT=1, pop, PC++ */
        n = add_gated_pair(w,L,n, 35, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 35, S_TOS,1,-1,0,-1,0,-1,0, 1.0f, S_OUTPUT, 1.0f);
        n = add_gated_pair(w,L,n, 35, -1,0,-1,0,-1,0,-1,0, 1.0f, S_HAS_OUT, 1.0f);
        n = add_gated_pair(w,L,n, 35, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 35, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 35, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 35, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 35, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_DIV (10): delegate to exec loop — PC++ and IS_NATIVE, exec loop does the rest */
        n = add_gated_pair(w,L,n, 10, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 10, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);

        /* OP_MOD (11): same pattern as DIV */
        n = add_gated_pair(w,L,n, 11, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 11, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);

        /* OP_HALT (36) */
        n = add_gated_pair(w,L,n, 36, -1,0,-1,0,-1,0,-1,0, 1.0f, S_HALT, 1.0f);

        printf("[WEIGHT_GEN] Layer 3: %d neurons used out of %d\n", n, FFN_DIM);
    }

    /* ── Layer 4: Frame management (gated FFN, currently no-op) ── */
    w->ff_type[4] = 0;

    printf("[WEIGHT_GEN] d_model=%d, layers=%d, FFN=%d\n", D, N_LAYERS, FFN_DIM);
    printf("[WEIGHT_GEN] Weights: %zu params, %.1f KB\n",
           sizeof(InterpreterWeights)/sizeof(float),
           sizeof(InterpreterWeights)/1024.0f);
}

/*******************************************************************************
 * Matrix-based forward pass
 ******************************************************************************/

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
        if (L == 0 && np > 0) {
            float Q[D]; memset(Q, 0, sizeof(Q));
            for (int i=0;i<D;i++) for(int j=0;j<D;j++) Q[i]+=w->wq[L][i*D+j]*x[j];
            for (int i=0;i<D;i++) Q[i]+=w->bq[L][i];

            float scores[256]; float mx=-1e30f;
            float Va[256][D];
            for (int p=0; p<np&&p<256; p++) {
                float K[D]; memset(K,0,sizeof(K));
                memset(Va[p],0,sizeof(Va[p]));
                for(int i=0;i<D;i++) for(int j=0;j<D;j++) {
                    K[i]+=w->wk[L][i*D+j]*pe[p][j];
                    Va[p][i]+=w->wv[L][i*D+j]*pe[p][j];
                }
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
    g_frame_count = 0; g_heap_ptr = 0;
    int n_out=0;
    for(int step=0;step<8192;step++){
        float next[D];
        forward_with_weights(w,state,pe,n_instr,next);
        exec_loop_postprocess(next, prog, n_instr);
        if(next[S_HAS_OUT]>0.5f&&n_out<max_out) outputs[n_out++]=next[S_OUTPUT];
        if(next[S_HALT]>0.5f) break;
        memcpy(state,next,sizeof(state));
    }
    return n_out;
}

/*******************************************************************************
 * Binary Weight Export (for qLLM loading)
 ******************************************************************************/

static void export_weights_binary(const InterpreterWeights* w, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { printf("ERROR: cannot open %s\n", path); return; }
    uint32_t magic = 0x514C4D57; /* "QLMW" */
    uint32_t version = 3;
    uint32_t d = D, nl = N_LAYERS, fd = FFN_DIM, nh = H, hd = HD;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&d, 4, 1, f);
    fwrite(&nl, 4, 1, f);
    fwrite(&fd, 4, 1, f);
    fwrite(&nh, 4, 1, f);
    fwrite(&hd, 4, 1, f);
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

    printf("  %-25s ref=%7.1f sim=%7.1f mat=%7.1f  %s%s%s\n",
           name, rv, sv, mv,
           ok_r?"":"ref:FAIL ", ok_s?"":"sim:FAIL ",
           (g_weights && !ok_m)?"mat:FAIL ":"");

    if (ok_r && ok_s && ok_m) n_pass++; else n_fail++;
}

int main() {
    printf("=== Eshkol VM Weight Compiler v3 ===\n\n");

    g_weights = (InterpreterWeights*)calloc(1, sizeof(InterpreterWeights));
    if (g_weights) generate_weights(g_weights);

    printf("\n  Tests (ref=reference, sim=simulated, mat=matrix-based):\n\n");

    /* ── Stage 0: Original v1 tests (renumbered to canonical opcodes) ── */
    printf("  --- Stage 0: Core arithmetic & control ---\n");

    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_ADD,0},{OP_PRINT,0},{OP_HALT,0}};
      test("3+5", p, 5, 8); }
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_ADD,0},{OP_CONST,2},{OP_MUL,0},{OP_PRINT,0},{OP_HALT,0}};
      test("(3+5)*2", p, 7, 16); }
    { Instr p[]={{OP_CONST,10},{OP_CONST,7},{OP_SUB,0},{OP_PRINT,0},{OP_HALT,0}};
      test("10-7", p, 5, 3); }
    /* mem[0]=42: SET_LOCAL takes operand as address, TOS as value */
    { Instr p[]={{OP_CONST,42},{OP_SET_LOCAL,0},{OP_GET_LOCAL,0},{OP_PRINT,0},{OP_HALT,0}};
      test("mem[0]=42", p, 5, 42); }

    /* sum(1..5) = 15: GET_LOCAL/SET_LOCAL with operand-based addressing */
    { Instr p[]={
        {OP_CONST,0},{OP_SET_LOCAL,0},                       /* mem[0]=0 (sum) */
        {OP_CONST,5},{OP_SET_LOCAL,1},                       /* mem[1]=5 (counter) */
        /* loop (pc=4): */
        {OP_GET_LOCAL,1},                                    /* push counter */
        {OP_JUMP_IF_FALSE,15},                               /* if counter==0 goto end */
        /* body: */
        {OP_GET_LOCAL,0},{OP_GET_LOCAL,1},{OP_ADD,0},        /* sum + counter */
        {OP_SET_LOCAL,0},                                    /* mem[0] = new sum */
        {OP_GET_LOCAL,1},{OP_CONST,1},{OP_SUB,0},            /* counter - 1 */
        {OP_SET_LOCAL,1},                                    /* mem[1] = counter-1 */
        {OP_JUMP,4},                                         /* goto loop */
        /* end (pc=15): */
        {OP_GET_LOCAL,0},{OP_PRINT,0},{OP_HALT,0},
      }; test("sum(1..5)", p, 18, 15); }

    /* 5! = 120 */
    { Instr p[]={
        {OP_CONST,1},{OP_SET_LOCAL,0},                       /* mem[0]=1 (result) */
        {OP_CONST,5},{OP_SET_LOCAL,1},                       /* mem[1]=5 (counter) */
        /* loop (pc=4): */
        {OP_GET_LOCAL,1},                                    /* push counter */
        {OP_JUMP_IF_FALSE,15},                               /* if counter==0 goto end */
        {OP_GET_LOCAL,0},{OP_GET_LOCAL,1},{OP_MUL,0},        /* result * counter */
        {OP_SET_LOCAL,0},                                    /* result = product */
        {OP_GET_LOCAL,1},{OP_CONST,1},{OP_SUB,0},            /* counter - 1 */
        {OP_SET_LOCAL,1},                                    /* counter = counter-1 */
        {OP_JUMP,4},                                         /* goto loop */
        /* end (pc=15): */
        {OP_GET_LOCAL,0},{OP_PRINT,0},{OP_HALT,0},
      }; test("5!", p, 18, 120); }

    /* fib(7) = 13 */
    { Instr p[]={
        {OP_CONST,0},{OP_SET_LOCAL,0},                       /* a = 0 */
        {OP_CONST,1},{OP_SET_LOCAL,1},                       /* b = 1 */
        {OP_CONST,7},{OP_SET_LOCAL,2},                       /* n = 7 */
        /* loop (pc=6): */
        {OP_GET_LOCAL,2},                                    /* push n */
        {OP_JUMP_IF_FALSE,19},                               /* if n==0 goto end */
        /* body: a,b = b, a+b */
        {OP_GET_LOCAL,0},{OP_GET_LOCAL,1},{OP_ADD,0},        /* a + b */
        {OP_GET_LOCAL,1},{OP_SET_LOCAL,0},                   /* a = old b */
        {OP_SET_LOCAL,1},                                    /* b = a+b */
        {OP_GET_LOCAL,2},{OP_CONST,1},{OP_SUB,0},            /* n - 1 */
        {OP_SET_LOCAL,2},                                    /* n = n-1 */
        {OP_JUMP,6},                                         /* goto loop */
        /* end (pc=19): */
        {OP_GET_LOCAL,0},{OP_PRINT,0},{OP_HALT,0},
      }; test("fib(7)", p, 22, 13); }

    { Instr p[]={
        {OP_CONST,2},{OP_CONST,3},{OP_MUL,0},
        {OP_CONST,4},{OP_CONST,5},{OP_MUL,0},
        {OP_ADD,0},{OP_PRINT,0},{OP_HALT,0},
      }; test("(2*3)+(4*5)", p, 9, 26); }
    { Instr p[]={{OP_CONST,7},{OP_CONST,11},{OP_MUL,0},{OP_PRINT,0},{OP_HALT,0}};
      test("7*11", p, 5, 77); }

    /* ── Stage 1: Trivial push opcodes ── */
    printf("\n  --- Stage 1: NIL, TRUE, FALSE ---\n");
    { Instr p[]={{OP_TRUE,0},{OP_PRINT,0},{OP_HALT,0}};
      test("true", p, 3, 1); }
    { Instr p[]={{OP_FALSE,0},{OP_PRINT,0},{OP_HALT,0}};
      test("false", p, 3, 0); }
    { Instr p[]={{OP_NIL,0},{OP_PRINT,0},{OP_HALT,0}};
      test("nil", p, 3, -1); }
    { Instr p[]={{OP_CONST,5},{OP_TRUE,0},{OP_ADD,0},{OP_PRINT,0},{OP_HALT,0}};
      test("5+true(=6)", p, 5, 6); }

    /* ── Stage 2: NEG, ABS ── */
    printf("\n  --- Stage 2: NEG, ABS ---\n");
    { Instr p[]={{OP_CONST,5},{OP_NEG,0},{OP_PRINT,0},{OP_HALT,0}};
      test("neg(5)", p, 4, -5); }
    { Instr p[]={{OP_CONST,-7},{OP_ABS,0},{OP_PRINT,0},{OP_HALT,0}};
      test("abs(-7)", p, 4, 7); }
    { Instr p[]={{OP_CONST,3},{OP_ABS,0},{OP_PRINT,0},{OP_HALT,0}};
      test("abs(3)", p, 4, 3); }
    { Instr p[]={{OP_CONST,5},{OP_NEG,0},{OP_ABS,0},{OP_PRINT,0},{OP_HALT,0}};
      test("abs(neg(5))", p, 5, 5); }

    /* ── Stage 3: Comparisons ── */
    printf("\n  --- Stage 3: EQ, LT, GT, LE, GE, NOT ---\n");
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_EQ,0},{OP_PRINT,0},{OP_HALT,0}};
      test("3==5", p, 5, 0); }
    { Instr p[]={{OP_CONST,5},{OP_CONST,5},{OP_EQ,0},{OP_PRINT,0},{OP_HALT,0}};
      test("5==5", p, 5, 1); }
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_LT,0},{OP_PRINT,0},{OP_HALT,0}};
      test("3<5", p, 5, 1); }
    { Instr p[]={{OP_CONST,5},{OP_CONST,3},{OP_LT,0},{OP_PRINT,0},{OP_HALT,0}};
      test("5<3", p, 5, 0); }
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_GT,0},{OP_PRINT,0},{OP_HALT,0}};
      test("3>5", p, 5, 0); }
    { Instr p[]={{OP_CONST,5},{OP_CONST,3},{OP_GT,0},{OP_PRINT,0},{OP_HALT,0}};
      test("5>3", p, 5, 1); }
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_LE,0},{OP_PRINT,0},{OP_HALT,0}};
      test("3<=5", p, 5, 1); }
    { Instr p[]={{OP_CONST,5},{OP_CONST,5},{OP_LE,0},{OP_PRINT,0},{OP_HALT,0}};
      test("5<=5", p, 5, 1); }
    { Instr p[]={{OP_CONST,5},{OP_CONST,3},{OP_GE,0},{OP_PRINT,0},{OP_HALT,0}};
      test("5>=3", p, 5, 1); }
    { Instr p[]={{OP_CONST,3},{OP_CONST,3},{OP_GE,0},{OP_PRINT,0},{OP_HALT,0}};
      test("3>=3", p, 5, 1); }
    /* Composite: if (3 < 5) print 42 else print 99 */
    { Instr p[]={
        {OP_CONST,3},{OP_CONST,5},{OP_LT,0},
        {OP_JUMP_IF_FALSE,6},{OP_CONST,42},{OP_JUMP,7},
        {OP_CONST,99},{OP_PRINT,0},{OP_HALT,0}};
      test("if(3<5)42", p, 9, 42); }

    /* ── Stage 4: DIV, MOD (delegated to exec loop) ── */
    printf("\n  --- Stage 4: DIV, MOD ---\n");
    { Instr p[]={{OP_CONST,10},{OP_CONST,2},{OP_DIV,0},{OP_PRINT,0},{OP_HALT,0}};
      test("10/2", p, 5, 5); }
    { Instr p[]={{OP_CONST,10},{OP_CONST,3},{OP_MOD,0},{OP_PRINT,0},{OP_HALT,0}};
      test("10%3", p, 5, 1); }
    { Instr p[]={{OP_CONST,21},{OP_CONST,7},{OP_DIV,0},{OP_PRINT,0},{OP_HALT,0}};
      test("21/7", p, 5, 3); }
    { Instr p[]={{OP_CONST,15},{OP_CONST,4},{OP_MOD,0},{OP_PRINT,0},{OP_HALT,0}};
      test("15%4", p, 5, 3); }
    /* DIV in a computation: (10/2) + 3 = 8 */
    { Instr p[]={{OP_CONST,10},{OP_CONST,2},{OP_DIV,0},{OP_CONST,3},{OP_ADD,0},{OP_PRINT,0},{OP_HALT,0}};
      test("10/2+3", p, 7, 8); }

    /* ── Stage 5: CALL, RETURN ── */
    printf("\n  --- Stage 5: CALL, RETURN ---\n");
    /* f(x) = x + 1, call f(5) → 6
     * Layout: CONST 5, CONST func_entry, CALL 1, PRINT, HALT
     *   func_entry(5): GET_LOCAL 0, CONST 1, ADD, RETURN */
    { Instr p[]={
        {OP_CONST,5},{OP_CONST,5},{OP_CALL,1},              /* push arg=5, push func_pc=5, call(argc=1) */
        {OP_PRINT,0},{OP_HALT,0},                           /* print return value, halt */
        /* func entry (pc=5): */
        {OP_GET_LOCAL,0},{OP_CONST,1},{OP_ADD,0},{OP_RETURN,0},
      }; test("f(x)=x+1, f(5)", p, 9, 6); }

    /* f(x) = x * x, call f(7) → 49 */
    { Instr p[]={
        {OP_CONST,7},{OP_CONST,5},{OP_CALL,1},
        {OP_PRINT,0},{OP_HALT,0},
        /* func entry (pc=5): */
        {OP_GET_LOCAL,0},{OP_GET_LOCAL,0},{OP_MUL,0},{OP_RETURN,0},
      }; test("f(x)=x*x, f(7)", p, 9, 49); }

    /* f(a,b) = a + b, call f(3,4) → 7 */
    { Instr p[]={
        {OP_CONST,3},{OP_CONST,4},{OP_CONST,6},{OP_CALL,2},
        {OP_PRINT,0},{OP_HALT,0},
        /* func entry (pc=6): */
        {OP_GET_LOCAL,0},{OP_GET_LOCAL,1},{OP_ADD,0},{OP_RETURN,0},
      }; test("f(a,b)=a+b, f(3,4)", p, 10, 7); }

    /* ── Stage 6: CONS, CAR, CDR, NULL_P ── */
    printf("\n  --- Stage 6: CONS, CAR, CDR, NULL_P ---\n");
    /* (car (cons 3 4)) → 3 */
    { Instr p[]={{OP_CONST,3},{OP_CONST,4},{OP_CONS,0},{OP_CAR,0},{OP_PRINT,0},{OP_HALT,0}};
      test("car(cons 3 4)", p, 6, 3); }
    /* (cdr (cons 3 4)) → 4 */
    { Instr p[]={{OP_CONST,3},{OP_CONST,4},{OP_CONS,0},{OP_CDR,0},{OP_PRINT,0},{OP_HALT,0}};
      test("cdr(cons 3 4)", p, 6, 4); }
    /* (null? nil) → 1 */
    { Instr p[]={{OP_NIL,0},{OP_NULL_P,0},{OP_PRINT,0},{OP_HALT,0}};
      test("null?(nil)", p, 4, 1); }
    /* (null? 5) → 0 */
    { Instr p[]={{OP_CONST,5},{OP_NULL_P,0},{OP_PRINT,0},{OP_HALT,0}};
      test("null?(5)", p, 4, 0); }
    /* (car (cdr (cons 1 (cons 2 (cons 3 nil))))) → 2
     * Build list (1 2 3): CONST 3, NIL, CONS → (3), CONST 2, swap args, CONS → (2 3), etc.
     * Actually: cons expects TOS=cdr SOS=car. So (cons 1 (cons 2 (cons 3 nil))):
     *   NIL, CONST 3, cons → pair(3,nil), CONST 2, swap...
     * Wait — stack order. CONS pops TOS=cdr, SOS=car.
     * To build (cons 3 nil): push 3 (car), push nil (cdr) → CONST 3, NIL, CONS
     * But that gives TOS=nil, SOS=3, so car=3, cdr=nil. ✓
     * (cons 2 (cons 3 nil)): push (cons 3 nil) result, then push 2, then cons
     * But stack: after first cons, TOS=pair_ptr. Need CONST 2 as SOS.
     * CONST 2 pushes 2 onto stack, so stack = [2, pair_ptr].
     * But CONS pops TOS=cdr, SOS=car → car=pair_ptr, cdr=2. Wrong!
     * Need: car=2, cdr=pair_ptr → SOS=2, TOS=pair_ptr.
     * So: first CONS gives pair_ptr on TOS. Then CONST 2 makes [2, pair_ptr].
     * We need TOS=pair_ptr (cdr), SOS=2 (car). But after CONST 2, TOS=2, SOS=pair_ptr.
     * Need a SWAP! But we don't have SWAP... we have DUP and POP but not SWAP.
     *
     * Alternative: build list in reverse. (cons 1 (cons 2 (cons 3 nil))):
     * Push elements in order: 3, nil → cons → (3.nil)
     *                         2, (3.nil) → cons → (2.(3.nil))
     * To get 2 as car, need SOS=2, TOS=(3.nil). After first cons, TOS=(3.nil).
     * CONST 2 → TOS=2, SOS=(3.nil). CONS → car=(3.nil)=SOS, cdr=2=TOS. WRONG.
     * We actually need TOS=(3.nil), SOS=2.
     *
     * OK let's just test simple car/cdr and move on. The swap issue is a
     * known limitation of the 4-register stack without a SWAP opcode.
     */

    /* ── Stage 7: Remaining opcodes + integration tests ── */
    printf("\n  --- Stage 7: Integration tests ---\n");

    /* Recursive factorial: fact(n) = if n==0 then 1 else n * fact(n-1)
     * Uses CALL/RETURN with recursive calls.
     * Layout:
     *   0: CONST 5          ; arg = 5
     *   1: CONST 5          ; func_pc = 5
     *   2: CALL 1           ; call fact(5)
     *   3: PRINT             ; print result
     *   4: HALT
     *   --- fact (pc=5) ---
     *   5: GET_LOCAL 0       ; push n
     *   6: CONST 0
     *   7: EQ                ; n == 0?
     *   8: JUMP_IF_FALSE 11  ; if not, goto recursive case
     *   9: CONST 1           ; return 1
     *  10: RETURN
     *  11: GET_LOCAL 0       ; push n
     *  12: GET_LOCAL 0       ; push n (for n-1)
     *  13: CONST 1
     *  14: SUB               ; n - 1
     *  15: CONST 5           ; func_pc = 5
     *  16: CALL 1            ; call fact(n-1)
     *  17: MUL               ; n * fact(n-1)
     *  18: RETURN
     */
    { Instr p[]={
        {OP_CONST,5},{OP_CONST,5},{OP_CALL,1},{OP_PRINT,0},{OP_HALT,0},
        /* fact: */
        {OP_GET_LOCAL,0},{OP_CONST,0},{OP_EQ,0},{OP_JUMP_IF_FALSE,11},
        {OP_CONST,1},{OP_RETURN,0},
        /* recursive case: */
        {OP_GET_LOCAL,0},
        {OP_GET_LOCAL,0},{OP_CONST,1},{OP_SUB,0},{OP_CONST,5},{OP_CALL,1},
        {OP_MUL,0},{OP_RETURN,0},
      }; test("rec fact(5)", p, 19, 120); }

    /* Recursive fibonacci: fib(n) = if n<=1 then n else fib(n-1)+fib(n-2)
     *   0: CONST 7, 1: CONST 5, 2: CALL 1, 3: PRINT, 4: HALT
     *   5: GET_LOCAL 0, CONST 1, LE, JUMP_IF_FALSE 11
     *   9: GET_LOCAL 0, RETURN                          ; base case: return n
     *  11: GET_LOCAL 0, CONST 1, SUB, CONST 5, CALL 1   ; fib(n-1)
     *  16: GET_LOCAL 0, CONST 2, SUB, CONST 5, CALL 1   ; fib(n-2)
     *  21: ADD, RETURN
     */
    { Instr p[]={
        {OP_CONST,7},{OP_CONST,5},{OP_CALL,1},{OP_PRINT,0},{OP_HALT,0},
        /* fib: */
        {OP_GET_LOCAL,0},{OP_CONST,1},{OP_LE,0},{OP_JUMP_IF_FALSE,11},
        {OP_GET_LOCAL,0},{OP_RETURN,0},
        /* recursive case: */
        {OP_GET_LOCAL,0},{OP_CONST,1},{OP_SUB,0},{OP_CONST,5},{OP_CALL,1},
        {OP_GET_LOCAL,0},{OP_CONST,2},{OP_SUB,0},{OP_CONST,5},{OP_CALL,1},
        {OP_ADD,0},{OP_RETURN,0},
      }; test("rec fib(7)", p, 23, 13); }

    /* set-car!/set-cdr! test: cons pair, mutate, read back */
    { Instr p[]={
        {OP_CONST,10},{OP_CONST,20},{OP_CONS,0},          /* (cons 10 20) → pair_ptr */
        {OP_DUP,0},{OP_CONST,99},{OP_SET_CAR,0},          /* set-car! pair 99 (pops val+pair, but we dup'd) */
        /* Wait — SET_CAR pops TOS=val, SOS=pair. After DUP we have [pair,pair].
         * CONST 99 gives [99,pair,pair]. SET_CAR: val=99, pair=pair → mutate.
         * Pops both, leaving [pair] on stack. Then CAR reads the mutated car. */
        {OP_CAR,0},{OP_PRINT,0},{OP_HALT,0},
      }; test("set-car!", p, 8, 99); }

    /* pair? test */
    { Instr p[]={
        {OP_CONST,1},{OP_CONST,2},{OP_CONS,0},{OP_PAIR_P,0},{OP_PRINT,0},{OP_HALT,0}};
      test("pair?(cons 1 2)", p, 6, 1); }

    /* Composite: (+ (car (cons 10 20)) (cdr (cons 30 40))) = 10 + 40 = 50 */
    { Instr p[]={
        {OP_CONST,10},{OP_CONST,20},{OP_CONS,0},{OP_CAR,0},  /* car(cons 10 20) = 10 */
        {OP_CONST,30},{OP_CONST,40},{OP_CONS,0},{OP_CDR,0},  /* cdr(cons 30 40) = 40 */
        {OP_ADD,0},{OP_PRINT,0},{OP_HALT,0}};
      test("car+cdr", p, 11, 50); }

    /* Closure test: create closure on heap, call it via CALL */
    { Instr p[]={
        /* 0 */ {OP_CLOSURE,5},                               /* push closure ptr (entry at pc=5) */
        /* 1 */ {OP_CALL,0},                                  /* call closure (0 args) */
        /* 2 */ {OP_PRINT,0},
        /* 3 */ {OP_HALT,0},
        /* padding */ {OP_NOP,0},
        /* func entry (pc=5): return 42 */
        /* 5 */ {OP_CONST,42},{OP_RETURN,0},
      }; test("closure()=42", p, 7, 42); }

    /* Vector test: create vec, read element */
    { Instr p[]={
        {OP_CONST,10},{OP_CONST,20},{OP_CONST,30},{OP_VEC_CREATE,3},  /* #(10 20 30) */
        {OP_CONST,1},{OP_VEC_REF,0},                                  /* vec[1] = 20 */
        {OP_PRINT,0},{OP_HALT,0}};
      /* Note: VEC_REF expects TOS=index, SOS=vec_ptr */
      test("vec-ref", p, 8, 20); }

    /* Vector length test */
    { Instr p[]={
        {OP_CONST,10},{OP_CONST,20},{OP_CONST,30},{OP_VEC_CREATE,3},
        {OP_VEC_LEN,0},{OP_PRINT,0},{OP_HALT,0}};
      test("vec-len", p, 7, 3); }

    /* Tail-call optimization test: tail-recursive sum
     * sum(n, acc) = if n==0 then acc else sum(n-1, acc+n)
     *   0: CONST 100, CONST 0, CONST 5, CALL 2  (sum(100, 0))
     *   4: PRINT, HALT
     *   5: GET_LOCAL 0   ; n
     *   6: CONST 0, EQ   ; n == 0?
     *   8: JUMP_IF_FALSE 11
     *   9: GET_LOCAL 1   ; return acc
     *  10: RETURN
     *  11: GET_LOCAL 0, CONST 1, SUB           ; n-1
     *  14: GET_LOCAL 1, GET_LOCAL 0, ADD        ; acc+n
     *  17: CONST 5, TAIL_CALL 2                ; tail-call sum(n-1, acc+n)
     */
    { Instr p[]={
        /* 0 */ {OP_CONST,100},{OP_CONST,0},{OP_CONST,6},{OP_CALL,2},
        /* 4 */ {OP_PRINT,0},{OP_HALT,0},
        /* sum (pc=6): MEM0=acc (first arg=SOS), MEM1=n (second arg=R2) */
        /* 6  */ {OP_GET_LOCAL,1},{OP_CONST,0},{OP_EQ,0},{OP_JUMP_IF_FALSE,12},
        /* 10 */ {OP_GET_LOCAL,0},{OP_RETURN,0},
        /* 12: recursive case */
        /* 12 */ {OP_GET_LOCAL,1},{OP_CONST,1},{OP_SUB,0},          /* n-1 */
        /* 15 */ {OP_GET_LOCAL,0},{OP_GET_LOCAL,1},{OP_ADD,0},       /* acc+n */
        /* 18 */ {OP_CONST,6},{OP_TAIL_CALL,2},
      }; test("tail sum(100)", p, 20, 5050); }

    /* Dynamic multiplication: 100/100 */
    printf("\n  Dynamic multiplication: ");
    int mul_ok = 0;
    for (int a=0; a<10; a++) for (int b=0; b<10; b++) {
        Instr p[]={{OP_CONST,a},{OP_CONST,b},{OP_MUL,0},{OP_PRINT,0},{OP_HALT,0}};
        float r[1],s[1],m[1];
        run_reference(p,5,r,1); run_simulated(p,5,s,1);
        if(g_weights) run_with_weights(g_weights,p,5,m,1);
        if(fabsf(r[0]-(float)(a*b))<0.01f && fabsf(s[0]-(float)(a*b))<0.01f
           && (!g_weights || fabsf(m[0]-(float)(a*b))<0.01f)) mul_ok++;
    }
    printf("%d/100\n", mul_ok);
    if(mul_ok==100) n_pass++; else n_fail++;

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);

    if (g_weights && n_fail == 0) {
        export_weights_binary(g_weights, "/tmp/interpreter_weights_v3.bin");
    }

    if (g_weights) free(g_weights);
    return n_fail > 0 ? 1 : 0;
}
