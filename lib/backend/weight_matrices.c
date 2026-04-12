/**
 * @file weight_matrices.c
 * @brief Universal Eshkol VM interpreter compiled into transformer weights.
 *
 * Full ISA implementation (63 opcodes, d_model=36). Supersedes v1/v2 (archived).
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

/* ESKB binary format reader (single-file include pattern) */
#include "eskb_reader.c"

/* Runtime libraries for extended native call dispatch.
 * Note: vm_tensor.c is included transitively via vm_tensor_ops.c,
 *       vm_string.c is included transitively via vm_io.c. */
#include "vm_numeric.h"
#include "vm_complex.c"
#include "vm_rational.c"
#include "vm_bignum.c"
#include "vm_dual.c"
#include "vm_autodiff.c"
#include "vm_tensor_ops.c"
#include "vm_logic.c"
#include "vm_inference.c"
#include "vm_workspace.c"
#include "vm_io.c"
#include "vm_hashtable.c"
#include "vm_bytevector.c"
#include "vm_multivalue.c"
#include "vm_error.c"
#include "vm_parameter.c"

#define D 128
#define H 16
#define HD 2
#define N_LAYERS 5
#define MEM_SIZE 4
#define FFN_DIM 1024
#define SCALE 100.0f
#define AD_MAX_TAPE 8    /* max tape nodes in state vector */
#define AD_NODE_FIELDS 8 /* fields per tape node */

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
    OP_PACK_REST=60, OP_WIND_PUSH=61, OP_WIND_POP=62, OP_VOID=63,

    /* AD opcodes — native in transformer weights */
    OP_AD_VAR=64, OP_AD_CONST=65,
    OP_AD_ADD=66, OP_AD_SUB=67, OP_AD_MUL=68,
    OP_AD_NEG=69, OP_AD_ABS=70, OP_AD_RELU=71,
    OP_AD_SIGMOID=72, OP_AD_TANH=73,
    OP_AD_EXP=74, OP_AD_LOG=75, OP_AD_SQRT=76,
    OP_AD_BACKWARD=77, OP_AD_GRAD=78,
    /* AD ops delegated to C (transcendentals / division) */
    OP_AD_DIV=79, OP_AD_POW=80, OP_AD_SIN=81, OP_AD_COS=82,

    OP_COUNT=83
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
    S_ABS_DELTA=31,

    /* Type tags for TOS/SOS/R2/R3 (32-35) — persist across steps.
     * Type encoding: 0=number, 1=boolean, 2=pair, 3=closure,
     *                4=string, 5=vector, 6=nil, 7=continuation */
    S_TYPE_TOS=32, S_TYPE_SOS=33, S_TYPE_R2=34, S_TYPE_R3=35,

    /* ── Zone B: AD control state (36-47) — persist across steps ── */
    S_AD_TAPE_LEN=36,    /* number of nodes on tape (0..AD_MAX_TAPE) */
    S_AD_CURSOR=37,      /* backward pass cursor (current node index, decrements) */
    S_AD_MODE=38,        /* 0=normal, 1=forward recording, 2=backward pass */
    S_AD_CUR_OP=39,      /* operation type of node at cursor */
    S_AD_CUR_VALUE=40,   /* forward value of node at cursor */
    S_AD_CUR_GRAD=41,    /* gradient of node at cursor */
    S_AD_CUR_LEFT=42,    /* left parent index */
    S_AD_CUR_RIGHT=43,   /* right parent index */
    S_AD_CUR_SAVED=44,   /* auxiliary saved value */
    S_AD_LEFT_VALUE=45,  /* value of left parent (loaded for backward) */
    S_AD_LEFT_GRAD=46,   /* gradient of left parent */
    S_AD_RIGHT_VALUE=47, /* value of right parent */

    /* ── Zone C: AD tape storage (48-111) — 8 nodes x 8 fields ──
     * Node i at dims (48 + i*8) through (48 + i*8 + 7)
     * Fields: [op, value, gradient, left, right, saved, spare0, spare1] */
    S_AD_TAPE_BASE=48,
    /* Access macro: S_AD_TAPE_BASE + node_idx * AD_NODE_FIELDS + field_offset */

    /* ── Zone D: AD transient / precomputed (112-127) ── */
    S_AD_IS_FORWARD=112,     /* indicator: executing AD forward op this cycle */
    S_AD_IS_BACKWARD=113,    /* indicator: in backward pass */
    S_AD_GRAD_ACCUM=114,     /* gradient accumulator */
    S_AD_PROD_GRAD_LV=115,   /* precomputed: CUR_GRAD * LEFT_VALUE */
    S_AD_PROD_GRAD_RV=116,   /* precomputed: CUR_GRAD * RIGHT_VALUE */
    S_AD_LEFT_GRAD_NEW=117,  /* computed gradient delta for left parent */
    S_AD_RIGHT_GRAD_NEW=118, /* computed gradient delta for right parent */
    S_AD_SPARE0=119,
    S_AD_SPARE1=120, S_AD_SPARE2=121, S_AD_SPARE3=122, S_AD_SPARE4=123,
    S_AD_SPARE5=124, S_AD_SPARE6=125, S_AD_SPARE7=126, S_AD_SPARE8=127
};

/* AD tape node field offsets within each 8-field block */
#define AD_F_OP    0
#define AD_F_VALUE 1
#define AD_F_GRAD  2
#define AD_F_LEFT  3
#define AD_F_RIGHT 4
#define AD_F_SAVED 5

/* Access tape node field: state[S_AD_TAPE_BASE + node * AD_NODE_FIELDS + field] */
#define AD_NODE(s, node, field) ((s)[S_AD_TAPE_BASE + (node) * AD_NODE_FIELDS + (field)])

/* AD operation type encodings (stored in AD_F_OP field) */
#define AD_OP_CONST    0.0f
#define AD_OP_VAR      1.0f
#define AD_OP_ADD      2.0f
#define AD_OP_SUB      3.0f
#define AD_OP_MUL      4.0f
#define AD_OP_NEG      5.0f
#define AD_OP_ABS      6.0f
#define AD_OP_RELU     7.0f
#define AD_OP_SIGMOID  8.0f
#define AD_OP_TANH     9.0f
#define AD_OP_EXP     10.0f
#define AD_OP_LOG     11.0f
#define AD_OP_SQRT    12.0f
#define AD_OP_DIV     13.0f
#define AD_OP_POW     14.0f
#define AD_OP_SIN     15.0f
#define AD_OP_COS     16.0f

/* Type tag values */
#define TYPE_NUMBER  0.0f
#define TYPE_BOOL    1.0f
#define TYPE_PAIR    2.0f
#define TYPE_CLOSURE 3.0f
#define TYPE_STRING  4.0f
#define TYPE_VECTOR  5.0f
#define TYPE_NIL     6.0f
#define TYPE_CONT    7.0f

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

/* Simple heap for CONS/CAR/CDR (pairs stored as consecutive float pairs).
 * WARNING: NOT thread-safe. All globals must be reset between program runs
 * via g_heap_ptr = 0. Do not use from multiple threads concurrently. */
#define HEAP_SIZE 4096
static float g_heap[HEAP_SIZE];
static int g_heap_ptr = 0;

/* Exception handler stack */
#define MAX_EXC_HANDLERS 32
static struct {
    float handler_pc;
    float saved_depth;
    float saved_mem[4]; /* MEM_SIZE */
    float saved_tos, saved_sos, saved_r2, saved_r3;
    float saved_type_tos, saved_type_sos, saved_type_r2, saved_type_r3;
} g_exc_handlers[MAX_EXC_HANDLERS];
static int g_exc_count = 0;
static float g_current_exn = 0.0f;

/* Closure tracking for GET_UPVALUE */
static int g_current_closure_ptr = -1;

/* Dynamic-wind stack */
#define MAX_WINDS 32
static struct {
    float after_thunk_ptr; /* heap index of after thunk closure */
    int frame_depth;
} g_wind_stack[MAX_WINDS];
static int g_wind_depth = 0;

/* Arena-based region stack for runtime libraries (complex, bignum, etc.) */
static VmRegionStack g_vm_regions;
static int g_vm_regions_initialized = 0;

static VmRegionStack* vm_get_regions(void) {
    if (!g_vm_regions_initialized) {
        vm_region_stack_init(&g_vm_regions);
        g_vm_regions_initialized = 1;
    }
    return &g_vm_regions;
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
    next->s[S_HAS_OUT] = 0;
    /* Clear intermediates: Zone A (16-31), Zone B cursor (39-47), Zone D (112-127) */
    for (int i = S_OPCODE; i <= S_ABS_DELTA; i++) next->s[i] = 0;
    for (int i = S_AD_CUR_OP; i <= S_AD_RIGHT_VALUE; i++) next->s[i] = 0;
    for (int i = S_AD_IS_FORWARD; i <= S_AD_SPARE8; i++) next->s[i] = 0;

    int pc = (int)cur->s[S_PC];
    if (pc < 0 || pc >= n_instr || cur->s[S_HALT] > 0.5f) { next->s[S_HALT] = 1; return; }

    float tos = cur->s[S_TOS], sos = cur->s[S_SOS];
    float r2 = cur->s[S_R2], r3 = cur->s[S_R3];
    float operand = (float)prog[pc].operand;
    int addr;

    /* Save current type tags for shifting */
    float tt_tos = cur->s[S_TYPE_TOS], tt_sos = cur->s[S_TYPE_SOS];
    float tt_r2  = cur->s[S_TYPE_R2],  tt_r3  = cur->s[S_TYPE_R3];

    switch (prog[pc].op) {
    case OP_NOP:    next->s[S_PC]=pc+1; break;
    case OP_CONST:  next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=operand; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_R3]=tt_r2; next->s[S_TYPE_R2]=tt_sos; next->s[S_TYPE_SOS]=tt_tos; next->s[S_TYPE_TOS]=TYPE_NUMBER;
        break;
    case OP_NIL:    next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=-1; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_R3]=tt_r2; next->s[S_TYPE_R2]=tt_sos; next->s[S_TYPE_SOS]=tt_tos; next->s[S_TYPE_TOS]=TYPE_NIL;
        break;
    case OP_TRUE:   next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=1; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_R3]=tt_r2; next->s[S_TYPE_R2]=tt_sos; next->s[S_TYPE_SOS]=tt_tos; next->s[S_TYPE_TOS]=TYPE_BOOL;
        break;
    case OP_FALSE:  next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_R3]=tt_r2; next->s[S_TYPE_R2]=tt_sos; next->s[S_TYPE_SOS]=tt_tos; next->s[S_TYPE_TOS]=TYPE_BOOL;
        break;
    case OP_ADD:    next->s[S_TOS]=tos+sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_NUMBER; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_SUB:    next->s[S_TOS]=sos-tos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_NUMBER; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_MUL:    next->s[S_TOS]=tos*sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_NUMBER; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_NEG:    next->s[S_TOS]=-tos; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_NUMBER;
        break;
    case OP_ABS:    next->s[S_TOS]=fabsf(tos); next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_NUMBER;
        break;
    case OP_EQ:     next->s[S_TOS]=(tos==sos)?1.0f:0.0f; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_LT:     next->s[S_TOS]=(sos<tos)?1.0f:0.0f; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_GT:     next->s[S_TOS]=(sos>tos)?1.0f:0.0f; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_LE:     next->s[S_TOS]=(sos<=tos)?1.0f:0.0f; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_GE:     next->s[S_TOS]=(sos>=tos)?1.0f:0.0f; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_DIV:
        if (tos == 0) { next->s[S_HALT] = 1; }
        else { next->s[S_TOS]=sos/tos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_NUMBER; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER; }
        break;
    case OP_MOD:
        if (tos == 0) { next->s[S_HALT] = 1; }
        else { float r=fmodf(sos,tos); if(r!=0&&((r>0)!=(tos>0)))r+=tos; next->s[S_TOS]=r; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_NUMBER; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER; }
        break;
    case OP_NOT:    next->s[S_TOS]=(tos==0)?1.0f:0.0f; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL;
        break;
    case OP_POP:    next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=tt_sos; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_DUP:    next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_R3]=tt_r2; next->s[S_TYPE_R2]=tt_sos; next->s[S_TYPE_SOS]=tt_tos; /* TOS type stays */
        break;
    case OP_GET_LOCAL:
        addr=(int)operand;
        if(addr>=0&&addr<MEM_SIZE) {
            next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos;
            next->s[S_TOS]=cur->s[S_MEM0+addr]; next->s[S_DEPTH]=cur->s[S_DEPTH]+1;
            next->s[S_TYPE_R3]=tt_r2; next->s[S_TYPE_R2]=tt_sos; next->s[S_TYPE_SOS]=tt_tos;
            next->s[S_TYPE_TOS]=TYPE_NUMBER; /* locals are untyped, assume number */
        }
        next->s[S_PC]=pc+1; break;
    case OP_SET_LOCAL:
        addr=(int)operand;
        if(addr>=0&&addr<MEM_SIZE) next->s[S_MEM0+addr]=tos;
        next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1;
        next->s[S_TYPE_TOS]=tt_sos; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        next->s[S_PC]=pc+1; break;
    case OP_CALL:   /* Set IS_CALL for exec loop to handle frame management */
        next->s[S_IS_CALL]=1; next->s[S_PC]=pc+1; break;
    case OP_RETURN: /* Set IS_RET for exec loop to handle frame restore */
        next->s[S_IS_RET]=1; next->s[S_PC]=pc+1; break;
    case OP_JUMP:   next->s[S_PC]=operand; break;
    case OP_JUMP_IF_FALSE:
        next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1;
        next->s[S_TYPE_TOS]=tt_sos; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        next->s[S_PC]=(tos==0)?operand:(float)(pc+1); break;
    case OP_LOOP:   next->s[S_PC]=operand; break;
    case OP_PRINT:  next->s[S_OUTPUT]=tos; next->s[S_HAS_OUT]=1; next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=tt_sos; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_HALT:   next->s[S_HALT]=1; break;

    /* ── AD Forward Ops: record nodes on the embedded tape ── */
    case OP_AD_VAR: { /* (ad-var value) → push tape index */
        int tlen = (int)cur->s[S_AD_TAPE_LEN];
        if (tlen < AD_MAX_TAPE) {
            AD_NODE(next->s, tlen, AD_F_OP) = AD_OP_VAR;
            AD_NODE(next->s, tlen, AD_F_VALUE) = operand;
            AD_NODE(next->s, tlen, AD_F_GRAD) = 0;
            AD_NODE(next->s, tlen, AD_F_LEFT) = -1;
            AD_NODE(next->s, tlen, AD_F_RIGHT) = -1;
            AD_NODE(next->s, tlen, AD_F_SAVED) = 0;
            /* Push tape index onto register stack */
            next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos;
            next->s[S_TOS]=(float)tlen;
            next->s[S_DEPTH]=cur->s[S_DEPTH]+1;
            next->s[S_AD_TAPE_LEN]=(float)(tlen+1);
            next->s[S_AD_MODE]=1;
        }
        next->s[S_PC]=pc+1; break;
    }
    case OP_AD_CONST: { /* (ad-const value) → push tape index */
        int tlen = (int)cur->s[S_AD_TAPE_LEN];
        if (tlen < AD_MAX_TAPE) {
            AD_NODE(next->s, tlen, AD_F_OP) = AD_OP_CONST;
            AD_NODE(next->s, tlen, AD_F_VALUE) = operand;
            AD_NODE(next->s, tlen, AD_F_GRAD) = 0;
            AD_NODE(next->s, tlen, AD_F_LEFT) = -1;
            AD_NODE(next->s, tlen, AD_F_RIGHT) = -1;
            AD_NODE(next->s, tlen, AD_F_SAVED) = 0;
            next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos;
            next->s[S_TOS]=(float)tlen;
            next->s[S_DEPTH]=cur->s[S_DEPTH]+1;
            next->s[S_AD_TAPE_LEN]=(float)(tlen+1);
            next->s[S_AD_MODE]=1;
        }
        next->s[S_PC]=pc+1; break;
    }
    case OP_AD_ADD: case OP_AD_SUB: case OP_AD_MUL: { /* binary: TOS=right_idx, SOS=left_idx */
        int tlen = (int)cur->s[S_AD_TAPE_LEN];
        int li = (int)sos, ri = (int)tos;
        if (tlen < AD_MAX_TAPE && li >= 0 && li < tlen && ri >= 0 && ri < tlen) {
            float lv = AD_NODE(cur->s, li, AD_F_VALUE);
            float rv = AD_NODE(cur->s, ri, AD_F_VALUE);
            float val = 0;
            float op_type = 0;
            OpCode cur_op = prog[pc].op;
            if (cur_op == OP_AD_ADD) { val = lv + rv; op_type = AD_OP_ADD; }
            else if (cur_op == OP_AD_SUB) { val = lv - rv; op_type = AD_OP_SUB; }
            else { val = lv * rv; op_type = AD_OP_MUL; }
            AD_NODE(next->s, tlen, AD_F_OP) = op_type;
            AD_NODE(next->s, tlen, AD_F_VALUE) = val;
            AD_NODE(next->s, tlen, AD_F_GRAD) = 0;
            AD_NODE(next->s, tlen, AD_F_LEFT) = (float)li;
            AD_NODE(next->s, tlen, AD_F_RIGHT) = (float)ri;
            AD_NODE(next->s, tlen, AD_F_SAVED) = 0;
            /* Pop two, push tape index */
            next->s[S_TOS]=(float)tlen; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0;
            next->s[S_DEPTH]=cur->s[S_DEPTH]-1;
            next->s[S_AD_TAPE_LEN]=(float)(tlen+1);
        }
        next->s[S_PC]=pc+1; break;
    }
    case OP_AD_NEG: case OP_AD_ABS: case OP_AD_RELU:
    case OP_AD_SIGMOID: case OP_AD_TANH:
    case OP_AD_EXP: case OP_AD_LOG: case OP_AD_SQRT: { /* unary: TOS=input_idx */
        int tlen = (int)cur->s[S_AD_TAPE_LEN];
        int ii = (int)tos;
        if (tlen < AD_MAX_TAPE && ii >= 0 && ii < tlen) {
            float iv = AD_NODE(cur->s, ii, AD_F_VALUE);
            float val = 0, op_type = 0;
            switch (prog[pc].op) {
                case OP_AD_NEG:     val = -iv;                          op_type = AD_OP_NEG; break;
                case OP_AD_ABS:     val = fabsf(iv);                    op_type = AD_OP_ABS; break;
                case OP_AD_RELU:    val = iv > 0 ? iv : 0;             op_type = AD_OP_RELU; break;
                case OP_AD_SIGMOID: val = 1.0f/(1.0f+expf(-iv));       op_type = AD_OP_SIGMOID; break;
                case OP_AD_TANH:    val = tanhf(iv);                    op_type = AD_OP_TANH; break;
                case OP_AD_EXP:     val = expf(iv);                     op_type = AD_OP_EXP; break;
                case OP_AD_LOG:     val = logf(iv);                     op_type = AD_OP_LOG; break;
                case OP_AD_SQRT:    val = sqrtf(iv);                    op_type = AD_OP_SQRT; break;
                default: break;
            }
            AD_NODE(next->s, tlen, AD_F_OP) = op_type;
            AD_NODE(next->s, tlen, AD_F_VALUE) = val;
            AD_NODE(next->s, tlen, AD_F_GRAD) = 0;
            AD_NODE(next->s, tlen, AD_F_LEFT) = (float)ii;
            AD_NODE(next->s, tlen, AD_F_RIGHT) = -1;
            AD_NODE(next->s, tlen, AD_F_SAVED) = 0;
            /* Replace TOS with tape index */
            next->s[S_TOS]=(float)tlen;
            next->s[S_AD_TAPE_LEN]=(float)(tlen+1);
        }
        next->s[S_PC]=pc+1; break;
    }
    case OP_AD_BACKWARD: { /* Start backward pass from TOS (output node index) */
        int output_idx = (int)tos;
        int tlen = (int)cur->s[S_AD_TAPE_LEN];
        if (output_idx >= 0 && output_idx < tlen) {
            next->s[S_AD_MODE] = 2;
            next->s[S_AD_CURSOR] = (float)output_idx;
            next->s[S_AD_IS_BACKWARD] = 1;
            /* Seed output gradient = 1.0 */
            AD_NODE(next->s, output_idx, AD_F_GRAD) = 1.0f;
            /* Pop the output index from stack */
            next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0;
            next->s[S_DEPTH]=cur->s[S_DEPTH]-1;
        }
        next->s[S_PC]=pc+1; break;
    }
    case OP_AD_GRAD: { /* Push gradient of TOS (node index) onto stack */
        int ni = (int)tos;
        int tlen = (int)cur->s[S_AD_TAPE_LEN];
        float grad = 0;
        if (ni >= 0 && ni < tlen) grad = AD_NODE(cur->s, ni, AD_F_GRAD);
        next->s[S_TOS] = grad;
        next->s[S_PC]=pc+1; break;
    }
    /* AD ops requiring transcendentals — delegate to C */
    case OP_AD_DIV: case OP_AD_POW: case OP_AD_SIN: case OP_AD_COS:
        next->s[S_IS_NATIVE]=1; next->s[S_PC]=pc+1; break;

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

/* ── AD Backward Step: process one tape node per VM cycle ──
 * When AD_IS_BACKWARD is set, this function processes the node at AD_CURSOR,
 * propagates gradients to parent nodes, and decrements the cursor.
 * When cursor goes below 0, the backward pass is complete. */
static void ad_backward_step(float* s) {
    if (s[S_AD_IS_BACKWARD] < 0.5f) return;

    int cursor = (int)s[S_AD_CURSOR];
    if (cursor < 0) {
        /* Backward complete */
        s[S_AD_IS_BACKWARD] = 0;
        s[S_AD_MODE] = 0;
        return;
    }

    float grad = AD_NODE(s, cursor, AD_F_GRAD);
    float op_type = AD_NODE(s, cursor, AD_F_OP);
    int li = (int)AD_NODE(s, cursor, AD_F_LEFT);
    int ri = (int)AD_NODE(s, cursor, AD_F_RIGHT);

    /* Compute and propagate gradient contributions based on operation type */
    if (fabsf(grad) > 1e-15f) {
        if (fabsf(op_type - AD_OP_ADD) < 0.5f) {
            /* d/dL (L+R) = 1, d/dR (L+R) = 1 */
            if (li >= 0) AD_NODE(s, li, AD_F_GRAD) += grad;
            if (ri >= 0) AD_NODE(s, ri, AD_F_GRAD) += grad;
        } else if (fabsf(op_type - AD_OP_SUB) < 0.5f) {
            /* d/dL (L-R) = 1, d/dR (L-R) = -1 */
            if (li >= 0) AD_NODE(s, li, AD_F_GRAD) += grad;
            if (ri >= 0) AD_NODE(s, ri, AD_F_GRAD) -= grad;
        } else if (fabsf(op_type - AD_OP_MUL) < 0.5f) {
            /* d/dL (L*R) = R, d/dR (L*R) = L */
            float lv = (li >= 0) ? AD_NODE(s, li, AD_F_VALUE) : 0;
            float rv = (ri >= 0) ? AD_NODE(s, ri, AD_F_VALUE) : 0;
            if (li >= 0) AD_NODE(s, li, AD_F_GRAD) += grad * rv;
            if (ri >= 0) AD_NODE(s, ri, AD_F_GRAD) += grad * lv;
        } else if (fabsf(op_type - AD_OP_NEG) < 0.5f) {
            if (li >= 0) AD_NODE(s, li, AD_F_GRAD) -= grad;
        } else if (fabsf(op_type - AD_OP_ABS) < 0.5f) {
            if (li >= 0) {
                float lv = AD_NODE(s, li, AD_F_VALUE);
                float sign = (lv > 0) ? 1.0f : (lv < 0) ? -1.0f : 0.0f;
                AD_NODE(s, li, AD_F_GRAD) += grad * sign;
            }
        } else if (fabsf(op_type - AD_OP_RELU) < 0.5f) {
            if (li >= 0 && AD_NODE(s, li, AD_F_VALUE) > 0)
                AD_NODE(s, li, AD_F_GRAD) += grad;
        } else if (fabsf(op_type - AD_OP_SIGMOID) < 0.5f) {
            /* d/dL sigma(L) = sigma(L)*(1-sigma(L)) = value*(1-value) */
            float v = AD_NODE(s, cursor, AD_F_VALUE);
            if (li >= 0) AD_NODE(s, li, AD_F_GRAD) += grad * v * (1.0f - v);
        } else if (fabsf(op_type - AD_OP_TANH) < 0.5f) {
            /* d/dL tanh(L) = 1 - tanh(L)^2 = 1 - value^2 */
            float v = AD_NODE(s, cursor, AD_F_VALUE);
            if (li >= 0) AD_NODE(s, li, AD_F_GRAD) += grad * (1.0f - v * v);
        } else if (fabsf(op_type - AD_OP_EXP) < 0.5f) {
            /* d/dL exp(L) = exp(L) = value */
            float v = AD_NODE(s, cursor, AD_F_VALUE);
            if (li >= 0) AD_NODE(s, li, AD_F_GRAD) += grad * v;
        } else if (fabsf(op_type - AD_OP_LOG) < 0.5f) {
            /* d/dL log(L) = 1/L */
            if (li >= 0) {
                float lv = AD_NODE(s, li, AD_F_VALUE);
                AD_NODE(s, li, AD_F_GRAD) += grad / lv;
            }
        } else if (fabsf(op_type - AD_OP_SQRT) < 0.5f) {
            /* d/dL sqrt(L) = 1/(2*sqrt(L)) = 1/(2*value) */
            float v = AD_NODE(s, cursor, AD_F_VALUE);
            if (li >= 0) AD_NODE(s, li, AD_F_GRAD) += grad / (2.0f * v);
        }
        /* AD_OP_CONST and AD_OP_VAR: leaf nodes, no propagation */
    }

    /* Decrement cursor */
    s[S_AD_CURSOR] = (float)(cursor - 1);
    if (cursor - 1 < 0) {
        s[S_AD_IS_BACKWARD] = 0;
        s[S_AD_MODE] = 0;
    }
}

static int g_last_ref_steps = 0;
static int g_last_sim_steps = 0;
static int g_last_mat_steps = 0;

static int run_reference(const Instr* prog, int n_instr, float* outputs, int max_out) {
    /* Double-buffer instead of 8192-entry trace (saves ~1.15 MB stack) */
    State cur, nxt;
    state_init(&cur);
    g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
    int n_out = 0, step_count = 0;
    while (step_count < 8191 && cur.s[S_HALT] < 0.5f) {
        step_count++;
        int pc = (int)cur.s[S_PC];
        if (pc >= 0 && pc < n_instr) {
            cur.s[S_OPCODE] = (float)prog[pc].op;
            cur.s[S_OPERAND] = (float)prog[pc].operand;
        }
        /* If backward pass is active, process one tape node instead of a normal instruction */
        if (cur.s[S_AD_IS_BACKWARD] > 0.5f) {
            memcpy(&nxt, &cur, sizeof(State));
            ad_backward_step(nxt.s);
        } else {
            execute_step(&cur, prog, n_instr, &nxt);
            exec_loop_postprocess(nxt.s, prog, n_instr);
        }
        if (nxt.s[S_HAS_OUT] > 0.5f && n_out < max_out)
            outputs[n_out++] = nxt.s[S_OUTPUT];
        memcpy(&cur, &nxt, sizeof(State));
    }
    g_last_ref_steps = step_count;
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
    /* Original: TOS * SOS via SQUARE trick */
    float a = x[S_TOS], b = x[S_SOS];
    out[S_PRODUCT] = 0.5f*(a+b)*(a+b) - 0.5f*a*a - 0.5f*b*b;
    /* AD backward: precompute grad * left_value and grad * right_value
     * These are needed for MUL backward: dL = grad*R, dR = grad*L */
    float g = x[S_AD_CUR_GRAD], lv = x[S_AD_LEFT_VALUE], rv = x[S_AD_RIGHT_VALUE];
    out[S_AD_PROD_GRAD_LV] = 0.5f*(g+rv)*(g+rv) - 0.5f*g*g - 0.5f*rv*rv; /* g * rv */
    out[S_AD_PROD_GRAD_RV] = 0.5f*(g+lv)*(g+lv) - 0.5f*g*g - 0.5f*lv*lv; /* g * lv */
}

static void layer2_ffn(const float x[D], float out[D]) {
    memset(out, 0, D*sizeof(float));
    /* GET_LOCAL address resolution: indicator(OPERAND==a) * mem[a] → LOADVAL */
    for (int a = 0; a < MEM_SIZE; a++)
        out[S_LOADVAL] += indicator(x[S_OPERAND], (float)a) * x[S_MEM0+a];
    /* SET_LOCAL store deltas: indicator(OPERAND==a) * (TOS - mem[a]) → STORED0+a */
    for (int a = 0; a < MEM_SIZE; a++)
        out[S_STORED0+a] = indicator(x[S_OPERAND], (float)a) * (x[S_TOS] - x[S_MEM0+a]);
    /* JUMP_IF_FALSE precompute */
    float iz = indicator(x[S_TOS], 0.0f);
    out[S_ZOPER] = iz * x[S_OPERAND];
    out[S_ZPC1]  = iz * (x[S_PC] + 1.0f);
    /* Comparison precomputes */
    out[S_CMP_EQ] = indicator(x[S_TOS] - x[S_SOS], 0.0f);
    out[S_CMP_LT] = sigmoidf(SCALE * (x[S_TOS] - x[S_SOS] - 0.5f));
    /* ABS precompute */
    out[S_ABS_DELTA] = sigmoidf(SCALE * (-x[S_TOS] - 0.5f)) * (-2.0f * x[S_TOS]);

    /* ── AD tape random-access load (backward mode only) ──
     * Load node at AD_CURSOR into AD_CUR_* fields.
     * Only active during backward pass — during forward ops, layer3 sets AD_CUR_* directly. */
    float bw_active = sigmoidf(SCALE * (x[S_AD_IS_BACKWARD] - 0.5f));
    float cursor = x[S_AD_CURSOR];
    for (int i = 0; i < AD_MAX_TAPE; i++) {
        float ci = indicator(cursor, (float)i) * bw_active;
        out[S_AD_CUR_OP]    += ci * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_OP];
        out[S_AD_CUR_VALUE]  += ci * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_VALUE];
        out[S_AD_CUR_GRAD]   += ci * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_GRAD];
        out[S_AD_CUR_LEFT]   += ci * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_LEFT];
        out[S_AD_CUR_RIGHT]  += ci * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_RIGHT];
        out[S_AD_CUR_SAVED]  += ci * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_SAVED];
    }
    /* Load parent values — only during backward */
    float left_idx = x[S_AD_CUR_LEFT] + out[S_AD_CUR_LEFT];
    float right_idx = x[S_AD_CUR_RIGHT] + out[S_AD_CUR_RIGHT];
    for (int i = 0; i < AD_MAX_TAPE; i++) {
        float li = indicator(left_idx, (float)i) * bw_active;
        float ri = indicator(right_idx, (float)i) * bw_active;
        out[S_AD_LEFT_VALUE]  += li * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_VALUE];
        out[S_AD_LEFT_GRAD]   += li * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_GRAD];
        out[S_AD_RIGHT_VALUE] += ri * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_VALUE];
    }
}

static void layer3_ffn(const float x[D], float out[D]) {
    memset(out, 0, D*sizeof(float));
    float op=x[S_OPCODE], oper=x[S_OPERAND], tos=x[S_TOS], sos=x[S_SOS];
    float r2=x[S_R2], r3=x[S_R3], product=x[S_PRODUCT], lv=x[S_LOADVAL];
    float alive = (1.0f - sigmoidf(SCALE*(x[S_HALT]-0.5f)))
               * (1.0f - sigmoidf(SCALE*(x[S_AD_IS_BACKWARD]-0.5f))); /* suppress during backward */

    /* Universal: clear output and HAS_OUT */
    out[S_OUTPUT] = -1.0f - x[S_OUTPUT];
    out[S_HAS_OUT] = -x[S_HAS_OUT];
    /* Universal: clear intermediate dims
     * Zone A transient (16-31) and Zone D transient (112-127) cleared here.
     * Zone B cursor-loaded (39-47) NOT cleared in layer3 — they persist through
     * layer4 which needs them for write-back. Cleared by run_simulated loop. */
    for (int i = S_OPCODE; i <= S_ABS_DELTA; i++) out[i] += -x[i];
    for (int i = S_AD_IS_FORWARD; i <= S_AD_SPARE8; i++) out[i] += -x[i];

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

    /* ── AD Forward Ops (64-78) ──
     * These record nodes on the embedded tape. The actual tape WRITE happens
     * in layer4_ffn — layer3 computes the node fields and sets AD_IS_FORWARD. */
    float tlen = x[S_AD_TAPE_LEN];
    float not_backward = 1.0f - sigmoidf(SCALE * (x[S_AD_IS_BACKWARD] - 0.5f)); /* 1 when NOT in backward */
    float tape_ok = sigmoidf(SCALE * ((float)AD_MAX_TAPE - 0.5f - tlen)) * not_backward;

    /* OP_AD_VAR (64): record variable node, push tape index */
    g=indicator(op,64)*alive*tape_ok;
    out[S_AD_CUR_OP]    += g * AD_OP_VAR;
    out[S_AD_CUR_VALUE]  += g * oper;
    out[S_AD_CUR_LEFT]   += g * (-1);
    out[S_AD_CUR_RIGHT]  += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_R3]+=g*(r2-r3); out[S_R2]+=g*(sos-r2); out[S_SOS]+=g*(tos-sos);
    out[S_TOS]+=g*(tlen-tos); /* push tape index */
    out[S_DEPTH]+=g; out[S_PC]+=g; out[S_AD_MODE]+=g*(1.0f-x[S_AD_MODE]);

    /* OP_AD_CONST (65): record constant node */
    g=indicator(op,65)*alive*tape_ok;
    out[S_AD_CUR_OP]    += g * AD_OP_CONST;
    out[S_AD_CUR_VALUE]  += g * oper;
    out[S_AD_CUR_LEFT]   += g * (-1);
    out[S_AD_CUR_RIGHT]  += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_R3]+=g*(r2-r3); out[S_R2]+=g*(sos-r2); out[S_SOS]+=g*(tos-sos);
    out[S_TOS]+=g*(tlen-tos);
    out[S_DEPTH]+=g; out[S_PC]+=g; out[S_AD_MODE]+=g*(1.0f-x[S_AD_MODE]);

    /* Binary AD ops (66-68): AD_ADD, AD_SUB, AD_MUL
     * TOS=right_idx, SOS=left_idx. Pop both, push tape index.
     * Value computation: read parent values from tape via layer2 loaded fields.
     * BUT: layer2 loaded AD_CUR_* based on AD_CURSOR, not on TOS/SOS.
     * For forward ops, we need parent values at TOS and SOS indices.
     * Since we can't do two random-access loads in one pass, we use the
     * OPERAND trick: layer2 loads based on cursor, but for forward ops the
     * values come from register stack. The reference interpreter reads them
     * directly. For the simulated path, we compute values from the tape. */

    /* Helper: read tape value at index i (simulated via indicator sum) */
    /* For AD_ADD: left=SOS, right=TOS. We need tape[SOS].value and tape[TOS].value */
    float ad_left_val = 0, ad_right_val = 0;
    for (int i = 0; i < AD_MAX_TAPE; i++) {
        float li = indicator(sos, (float)i);
        float ri = indicator(tos, (float)i);
        ad_left_val  += li * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_VALUE];
        ad_right_val += ri * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_VALUE];
    }

    /* OP_AD_ADD (66) */
    g=indicator(op,66)*alive*tape_ok;
    out[S_AD_CUR_OP]    += g * AD_OP_ADD;
    out[S_AD_CUR_VALUE]  += g * (ad_left_val + ad_right_val);
    out[S_AD_CUR_LEFT]   += g * sos;
    out[S_AD_CUR_RIGHT]  += g * tos;
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3);
    out[S_DEPTH]+=g*(-1); out[S_PC]+=g;

    /* OP_AD_SUB (67) */
    g=indicator(op,67)*alive*tape_ok;
    out[S_AD_CUR_OP]    += g * AD_OP_SUB;
    out[S_AD_CUR_VALUE]  += g * (ad_left_val - ad_right_val);
    out[S_AD_CUR_LEFT]   += g * sos;
    out[S_AD_CUR_RIGHT]  += g * tos;
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3);
    out[S_DEPTH]+=g*(-1); out[S_PC]+=g;

    /* OP_AD_MUL (68) */
    g=indicator(op,68)*alive*tape_ok;
    out[S_AD_CUR_OP]    += g * AD_OP_MUL;
    out[S_AD_CUR_VALUE]  += g * (ad_left_val * ad_right_val);
    out[S_AD_CUR_LEFT]   += g * sos;
    out[S_AD_CUR_RIGHT]  += g * tos;
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3);
    out[S_DEPTH]+=g*(-1); out[S_PC]+=g;

    /* Unary AD ops (69-76): AD_NEG, AD_ABS, AD_RELU, AD_SIGMOID, AD_TANH, AD_EXP, AD_LOG, AD_SQRT
     * TOS=input_idx. Replace TOS with tape index.
     * Input value from tape at TOS index: */
    float ad_input_val = 0;
    for (int i = 0; i < AD_MAX_TAPE; i++)
        ad_input_val += indicator(tos, (float)i) * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_VALUE];

    /* OP_AD_NEG (69) */
    g=indicator(op,69)*alive*tape_ok;
    out[S_AD_CUR_OP] += g * AD_OP_NEG;
    out[S_AD_CUR_VALUE] += g * (-ad_input_val);
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g;

    /* OP_AD_ABS (70) */
    g=indicator(op,70)*alive*tape_ok;
    out[S_AD_CUR_OP] += g * AD_OP_ABS;
    out[S_AD_CUR_VALUE] += g * fabsf(ad_input_val);
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g;

    /* OP_AD_RELU (71) */
    g=indicator(op,71)*alive*tape_ok;
    out[S_AD_CUR_OP] += g * AD_OP_RELU;
    out[S_AD_CUR_VALUE] += g * (ad_input_val > 0 ? ad_input_val : 0);
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g;

    /* OP_AD_SIGMOID (72) */
    g=indicator(op,72)*alive*tape_ok;
    out[S_AD_CUR_OP] += g * AD_OP_SIGMOID;
    out[S_AD_CUR_VALUE] += g * (1.0f/(1.0f+expf(-ad_input_val)));
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g;

    /* OP_AD_TANH (73) */
    g=indicator(op,73)*alive*tape_ok;
    out[S_AD_CUR_OP] += g * AD_OP_TANH;
    out[S_AD_CUR_VALUE] += g * tanhf(ad_input_val);
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g;

    /* OP_AD_EXP (74) */
    g=indicator(op,74)*alive*tape_ok;
    out[S_AD_CUR_OP] += g * AD_OP_EXP;
    out[S_AD_CUR_VALUE] += g * expf(ad_input_val);
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g;

    /* OP_AD_LOG (75) */
    g=indicator(op,75)*alive*tape_ok;
    out[S_AD_CUR_OP] += g * AD_OP_LOG;
    out[S_AD_CUR_VALUE] += g * logf(ad_input_val > 0 ? ad_input_val : 1e-15f);
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g;

    /* OP_AD_SQRT (76) */
    g=indicator(op,76)*alive*tape_ok;
    out[S_AD_CUR_OP] += g * AD_OP_SQRT;
    out[S_AD_CUR_VALUE] += g * sqrtf(ad_input_val > 0 ? ad_input_val : 0);
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g;

    /* OP_AD_BACKWARD (77): start backward pass.
     * Seed the output node's gradient = 1.0 directly in the tape.
     * Do NOT set AD_IS_FORWARD (that would trigger a spurious tape write in layer4). */
    g=indicator(op,77)*alive; {
        out[S_AD_MODE] += g * (2.0f - x[S_AD_MODE]);
        out[S_AD_CURSOR] += g * (tos - x[S_AD_CURSOR]);
        out[S_AD_IS_BACKWARD] += g;
        /* Seed gradient: write 1.0 to tape[TOS].gradient via indicator */
        for (int i = 0; i < AD_MAX_TAPE; i++) {
            float ti = indicator(tos, (float)i);
            out[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_GRAD] += g * ti * 1.0f;
        }
        out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3);
        out[S_DEPTH]+=g*(-1); out[S_PC]+=g;
    }

    /* OP_AD_GRAD (78): replace TOS with gradient of tape[TOS] */
    g=indicator(op,78)*alive; {
        /* Read gradient from tape at TOS index */
        float grad_val = 0;
        for (int i = 0; i < AD_MAX_TAPE; i++)
            grad_val += indicator(tos, (float)i) * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_GRAD];
        out[S_TOS] += g * (grad_val - tos);
        out[S_PC] += g;
    }

    /* AD ops delegated to C (79-82) */
    for (int opc = 79; opc <= 82; opc++) {
        g=indicator(op,(float)opc)*alive; out[S_PC]+=g; out[S_IS_NATIVE]+=g;
    }

    /* ── AD Backward gradient computation ──
     * When AD_IS_BACKWARD is set (by run_simulated's backward step),
     * compute gradient contributions based on AD_CUR_OP.
     * Results go into AD_LEFT_GRAD_NEW and AD_RIGHT_GRAD_NEW for layer4. */
    float bw = x[S_AD_IS_BACKWARD]; /* > 0 during backward */
    if (bw > 0.5f) {
        float cur_grad = x[S_AD_CUR_GRAD];
        float cur_op = x[S_AD_CUR_OP];
        float cur_val = x[S_AD_CUR_VALUE];
        float gl = x[S_AD_PROD_GRAD_LV]; /* precomputed: grad * right_value (for MUL dL) */
        float gr = x[S_AD_PROD_GRAD_RV]; /* precomputed: grad * left_value (for MUL dR) */

        /* ADD: dL = grad, dR = grad */
        float ba = indicator(cur_op, AD_OP_ADD);
        out[S_AD_LEFT_GRAD_NEW]  += ba * cur_grad;
        out[S_AD_RIGHT_GRAD_NEW] += ba * cur_grad;
        /* SUB: dL = grad, dR = -grad */
        float bs = indicator(cur_op, AD_OP_SUB);
        out[S_AD_LEFT_GRAD_NEW]  += bs * cur_grad;
        out[S_AD_RIGHT_GRAD_NEW] -= bs * cur_grad;
        /* MUL: dL = grad*R, dR = grad*L (precomputed in layer1) */
        float bm = indicator(cur_op, AD_OP_MUL);
        out[S_AD_LEFT_GRAD_NEW]  += bm * gl; /* grad * right_value */
        out[S_AD_RIGHT_GRAD_NEW] += bm * gr; /* grad * left_value */
        /* NEG: dL = -grad */
        float bn = indicator(cur_op, AD_OP_NEG);
        out[S_AD_LEFT_GRAD_NEW] -= bn * cur_grad;
        /* ABS: dL = grad * sign(left) */
        float babs = indicator(cur_op, AD_OP_ABS);
        float lv_sign = (x[S_AD_LEFT_VALUE] > 0) ? 1.0f : (x[S_AD_LEFT_VALUE] < 0) ? -1.0f : 0.0f;
        out[S_AD_LEFT_GRAD_NEW] += babs * cur_grad * lv_sign;
        /* RELU: dL = grad if left > 0 else 0 */
        float br = indicator(cur_op, AD_OP_RELU);
        out[S_AD_LEFT_GRAD_NEW] += br * cur_grad * (x[S_AD_LEFT_VALUE] > 0 ? 1.0f : 0.0f);
        /* SIGMOID: dL = grad * val * (1 - val) */
        float bsig = indicator(cur_op, AD_OP_SIGMOID);
        out[S_AD_LEFT_GRAD_NEW] += bsig * cur_grad * cur_val * (1.0f - cur_val);
        /* TANH: dL = grad * (1 - val^2) */
        float btanh = indicator(cur_op, AD_OP_TANH);
        out[S_AD_LEFT_GRAD_NEW] += btanh * cur_grad * (1.0f - cur_val * cur_val);
        /* EXP: dL = grad * val */
        float bexp = indicator(cur_op, AD_OP_EXP);
        out[S_AD_LEFT_GRAD_NEW] += bexp * cur_grad * cur_val;
        /* LOG: dL = grad / left_value */
        float blog = indicator(cur_op, AD_OP_LOG);
        float lv_safe = (fabsf(x[S_AD_LEFT_VALUE]) > 1e-15f) ? x[S_AD_LEFT_VALUE] : 1e-15f;
        out[S_AD_LEFT_GRAD_NEW] += blog * cur_grad / lv_safe;
        /* SQRT: dL = grad / (2 * val) */
        float bsqrt = indicator(cur_op, AD_OP_SQRT);
        float val_safe = (fabsf(cur_val) > 1e-15f) ? cur_val : 1e-15f;
        out[S_AD_LEFT_GRAD_NEW] += bsqrt * cur_grad / (2.0f * val_safe);
    }
}

/* Layer 4: AD tape write (forward ops) + gradient write-back (backward).
 *
 * Forward: when AD_IS_FORWARD is set, write AD_CUR_* fields into
 * tape[tape_len] and increment tape_len.
 *
 * Backward: when AD_IS_BACKWARD is set, accumulate gradient deltas
 * from AD_LEFT_GRAD_NEW / AD_RIGHT_GRAD_NEW into parent nodes,
 * then decrement AD_CURSOR. */
/* NOTE: layer4 takes original_state to distinguish "backward was already active"
 * from "backward was just started this cycle by layer3". */
static void layer4_ffn(float x[D], float out[D]) {
    memset(out, 0, D * sizeof(float));

    /* ── Forward: write new tape node ── */
    float fw = x[S_AD_IS_FORWARD];
    if (fw > 0.5f) {
        int tlen = (int)x[S_AD_TAPE_LEN];
        if (tlen >= 0 && tlen < AD_MAX_TAPE) {
            /* Write AD_CUR_* fields into tape[tlen] */
            for (int i = 0; i < AD_MAX_TAPE; i++) {
                float ti = indicator((float)tlen, (float)i);
                out[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_OP]    += ti * x[S_AD_CUR_OP];
                out[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_VALUE]  += ti * x[S_AD_CUR_VALUE];
                out[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_GRAD]   += ti * x[S_AD_CUR_GRAD];
                out[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_LEFT]   += ti * x[S_AD_CUR_LEFT];
                out[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_RIGHT]  += ti * x[S_AD_CUR_RIGHT];
                out[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_SAVED]  += ti * x[S_AD_CUR_SAVED];
            }
            out[S_AD_TAPE_LEN] += 1.0f; /* increment tape length */
        }
    }

    /* ── Backward: propagate gradient to parents ──
     * Note: cursor decrement and mode management are handled by run_simulated,
     * NOT by this layer. This avoids the issue of layer3 setting backward
     * and layer4 immediately consuming it in the same cycle. */
    float bw = x[S_AD_IS_BACKWARD];
    if (bw > 0.5f) {
        float left_idx = x[S_AD_CUR_LEFT];
        float right_idx = x[S_AD_CUR_RIGHT];
        float lg = x[S_AD_LEFT_GRAD_NEW];
        float rg = x[S_AD_RIGHT_GRAD_NEW];

        /* Write gradient deltas to parent tape nodes */
        for (int i = 0; i < AD_MAX_TAPE; i++) {
            float li = indicator(left_idx, (float)i);
            float ri = indicator(right_idx, (float)i);
            out[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_GRAD] += li * lg + ri * rg;
        }
    }
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
                if (tos == 0) {
                    x[S_HALT] = 1;
                } else {
                    x[S_TOS] = sos / tos;
                    x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                    x[S_DEPTH] -= 1;
                    x[S_TYPE_TOS] = TYPE_NUMBER; x[S_TYPE_SOS] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_R3]; x[S_TYPE_R3] = TYPE_NUMBER;
                }
            } else if (opcode == OP_MOD) {
                if (tos == 0) {
                    x[S_HALT] = 1;
                } else {
                    float r = fmodf(sos, tos);
                    if (r != 0 && ((r > 0) != (tos > 0))) r += tos;
                    x[S_TOS] = r;
                    x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                    x[S_DEPTH] -= 1;
                    x[S_TYPE_TOS] = TYPE_NUMBER; x[S_TYPE_SOS] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_R3]; x[S_TYPE_R3] = TYPE_NUMBER;
                }
            } else if (opcode == OP_CONS) {
                /* CONS: allocate pair on heap. TOS=cdr, SOS=car → push heap ptr */
                if (g_heap_ptr + 2 <= HEAP_SIZE) {
                    int ptr = g_heap_ptr;
                    g_heap[g_heap_ptr++] = sos;  /* car */
                    g_heap[g_heap_ptr++] = tos;  /* cdr */
                    x[S_TOS] = (float)ptr;       /* pair reference = heap index */
                    x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                    x[S_DEPTH] -= 1;
                    x[S_TYPE_TOS] = TYPE_PAIR; x[S_TYPE_SOS] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_R3]; x[S_TYPE_R3] = TYPE_NUMBER;
                }
            } else if (opcode == OP_CAR) {
                /* Type check: accept TYPE_PAIR or valid-looking heap pointer
                 * (simulated/matrix paths may not set type tags for CONS) */
                int ptr = (int)tos;
                int is_pair = (x[S_TYPE_TOS] == TYPE_PAIR) ||
                              (ptr >= 0 && ptr + 1 < g_heap_ptr);
                if (!is_pair) {
                    x[S_TOS] = 0; x[S_TYPE_TOS] = TYPE_NUMBER;
                } else {
                    if (ptr >= 0 && ptr + 1 < HEAP_SIZE)
                        x[S_TOS] = g_heap[ptr];      /* car */
                    x[S_TYPE_TOS] = TYPE_NUMBER; /* element type unknown */
                }
            } else if (opcode == OP_CDR) {
                int ptr = (int)tos;
                int is_pair = (x[S_TYPE_TOS] == TYPE_PAIR) ||
                              (ptr >= 0 && ptr + 1 < g_heap_ptr);
                if (!is_pair) {
                    x[S_TOS] = 0; x[S_TYPE_TOS] = TYPE_NUMBER;
                } else {
                    if (ptr >= 0 && ptr + 1 < HEAP_SIZE)
                        x[S_TOS] = g_heap[ptr + 1];  /* cdr */
                    x[S_TYPE_TOS] = TYPE_NUMBER; /* element type unknown */
                }
            } else if (opcode == OP_NULL_P) {
                /* NULL_P: check both type tag and value sentinel for compatibility
                 * with all three execution paths (ref sets type, sim/mat use value) */
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_NIL || tos == -1.0f) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            } else if (opcode == OP_SET_CAR) {
                /* SET_CAR: TOS=val, SOS=pair → mutate car */
                int ptr = (int)sos;
                if (ptr >= 0 && ptr < HEAP_SIZE) g_heap[ptr] = tos;
                x[S_TOS] = r2; x[S_SOS] = r3; x[S_R2] = 0; x[S_R3] = 0;
                x[S_DEPTH] -= 2;
                x[S_TYPE_TOS] = x[S_TYPE_R2]; x[S_TYPE_SOS] = x[S_TYPE_R3]; x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
            } else if (opcode == OP_SET_CDR) {
                /* SET_CDR: TOS=val, SOS=pair → mutate cdr */
                int ptr = (int)sos;
                if (ptr >= 0 && ptr + 1 < HEAP_SIZE) g_heap[ptr + 1] = tos;
                x[S_TOS] = r2; x[S_SOS] = r3; x[S_R2] = 0; x[S_R3] = 0;
                x[S_DEPTH] -= 2;
                x[S_TYPE_TOS] = x[S_TYPE_R2]; x[S_TYPE_SOS] = x[S_TYPE_R3]; x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
            } else if (opcode == OP_PAIR_P) {
                /* PAIR_P: check type tag and heap pointer for compatibility with all paths */
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_PAIR || (tos >= 0 && (int)tos + 1 < g_heap_ptr)) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            } else if (opcode == OP_NUM_P) {
                /* NUM_P: type_tos == TYPE_NUMBER */
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_NUMBER) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            } else if (opcode == OP_POPN) {
                /* POPN: pop N values below TOS, keeping TOS */
                int count = (int)(pc >= 0 && pc < n_instr ? prog[pc].operand : 0);
                if (count < 0) count = 0;
                float regs[4] = {x[S_TOS], x[S_SOS], x[S_R2], x[S_R3]};
                float types[4] = {x[S_TYPE_TOS], x[S_TYPE_SOS], x[S_TYPE_R2], x[S_TYPE_R3]};
                /* Remove 'count' items below TOS. TOS stays, SOS = old reg[count+1], etc. */
                for (int i = 1; i < 4; i++) {
                    int src = i + count;
                    if (src < 4) { regs[i] = regs[src]; types[i] = types[src]; }
                    else { regs[i] = 0; types[i] = TYPE_NUMBER; }
                }
                x[S_TOS] = regs[0]; x[S_SOS] = regs[1]; x[S_R2] = regs[2]; x[S_R3] = regs[3];
                x[S_TYPE_TOS] = types[0]; x[S_TYPE_SOS] = types[1]; x[S_TYPE_R2] = types[2]; x[S_TYPE_R3] = types[3];
                x[S_DEPTH] -= count;
            } else if (opcode == OP_TAIL_CALL) {
                /* TAIL_CALL: like CALL but reuses current frame (no frame push) */
                int argc = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                if (argc < 0) argc = 0;
                if (argc > MEM_SIZE) argc = MEM_SIZE;
                float func_pc = tos;
                float args[4] = {sos, r2, r3, 0};
                for (int i = 0; i < MEM_SIZE; i++) x[S_MEM0+i] = 0;
                for (int i = 0; i < argc && i < MEM_SIZE; i++)
                    x[S_MEM0+i] = args[i];
                x[S_PC] = func_pc;
                x[S_TOS] = 0; x[S_SOS] = 0; x[S_R2] = 0; x[S_R3] = 0;
                x[S_DEPTH] -= (1 + argc);
                x[S_TYPE_TOS] = TYPE_NUMBER; x[S_TYPE_SOS] = TYPE_NUMBER;
                x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
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
                    x[S_TYPE_R3] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_SOS]; x[S_TYPE_SOS] = x[S_TYPE_TOS];
                    x[S_R3] = x[S_R2]; x[S_R2] = x[S_SOS]; x[S_SOS] = x[S_TOS];
                    x[S_TOS] = (float)cptr;
                    x[S_TYPE_TOS] = TYPE_CLOSURE;
                    x[S_DEPTH] += 1;
                }
            } else if (opcode == OP_GET_UPVALUE) {
                int idx = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                float val = 0;
                /* Try closure upvalue array first, fall back to MEM */
                if (g_current_closure_ptr >= 0 && g_current_closure_ptr + 2 + idx < HEAP_SIZE) {
                    val = g_heap[g_current_closure_ptr + 2 + idx];
                } else if (idx >= 0 && idx < MEM_SIZE) {
                    val = x[S_MEM0 + idx];
                }
                /* Push */
                x[S_TYPE_R3] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_SOS]; x[S_TYPE_SOS] = x[S_TYPE_TOS];
                x[S_R3] = x[S_R2]; x[S_R2] = x[S_SOS]; x[S_SOS] = x[S_TOS];
                x[S_TOS] = val;
                x[S_TYPE_TOS] = TYPE_NUMBER;
                x[S_DEPTH] += 1;
            } else if (opcode == OP_SET_UPVALUE) {
                /* SET_UPVALUE: operand = index. Store TOS to upvalue slot, pop. */
                int idx = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                if (idx >= 0 && idx < MEM_SIZE) x[S_MEM0 + idx] = tos;
                x[S_TOS] = sos; x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                x[S_TYPE_TOS] = x[S_TYPE_SOS]; x[S_TYPE_SOS] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_R3]; x[S_TYPE_R3] = TYPE_NUMBER;
                x[S_DEPTH] -= 1;
            } else if (opcode == OP_CLOSE_UPVALUE) {
                int idx = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                /* Copy local MEM[idx] to current closure's upvalue slot */
                if (g_current_closure_ptr >= 0 && idx >= 0 && idx < MEM_SIZE &&
                    g_current_closure_ptr + 2 + idx < HEAP_SIZE) {
                    g_heap[g_current_closure_ptr + 2 + idx] = x[S_MEM0 + idx];
                }
            } else if (opcode == OP_OPEN_CLOSURE) {
                /* Set current closure and populate MEM from upvalue array */
                int cptr = (int)x[S_TOS]; /* closure pointer on TOS */
                if (cptr >= 0 && cptr + 1 < HEAP_SIZE) {
                    g_current_closure_ptr = cptr;
                    int n_upvals = (int)g_heap[cptr + 1];
                    for (int i = 0; i < n_upvals && i < MEM_SIZE && cptr + 2 + i < HEAP_SIZE; i++) {
                        x[S_MEM0 + i] = g_heap[cptr + 2 + i];
                    }
                }
            }
            /* ── Vectors ── */
            else if (opcode == OP_VEC_CREATE) {
                /* VEC_CREATE: operand = count. Pop count values, create vector on heap. */
                int count = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                if (count < 0) count = 0;
                if (count > 4) count = 4; /* max elements from stack registers */
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
                    x[S_TYPE_TOS] = TYPE_VECTOR;
                    x[S_TYPE_SOS] = TYPE_NUMBER; x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
                }
            } else if (opcode == OP_VEC_REF) {
                /* VEC_REF: TOS=index, SOS=vector_ptr → push vector[index] */
                int vptr = (int)sos;
                int idx = (int)tos;
                if (vptr >= 0 && vptr < HEAP_SIZE) {
                    int len = (int)g_heap[vptr];
                    if (idx >= 0 && idx < len && vptr + 1 + idx < HEAP_SIZE)
                        x[S_TOS] = g_heap[vptr + 1 + idx];
                }
                x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                x[S_DEPTH] -= 1;
                x[S_TYPE_TOS] = TYPE_NUMBER; /* element type unknown */
                x[S_TYPE_SOS] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_R3]; x[S_TYPE_R3] = TYPE_NUMBER;
            } else if (opcode == OP_VEC_SET) {
                /* VEC_SET: TOS=value, SOS=index, R2=vector_ptr → mutate */
                int vptr = (int)r2;
                int idx = (int)sos;
                if (vptr >= 0 && vptr < HEAP_SIZE) {
                    int len = (int)g_heap[vptr];
                    if (idx >= 0 && idx < len && vptr + 1 + idx < HEAP_SIZE)
                        g_heap[vptr + 1 + idx] = tos;
                }
                x[S_TOS] = r3; x[S_SOS] = 0; x[S_R2] = 0; x[S_R3] = 0;
                x[S_DEPTH] -= 3;
                x[S_TYPE_TOS] = x[S_TYPE_R3]; x[S_TYPE_SOS] = TYPE_NUMBER; x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
            } else if (opcode == OP_VEC_LEN) {
                /* VEC_LEN: TOS=vector_ptr → push length */
                int vptr = (int)tos;
                if (vptr >= 0 && vptr < HEAP_SIZE)
                    x[S_TOS] = g_heap[vptr];
                x[S_TYPE_TOS] = TYPE_NUMBER;
            }
            /* ── Strings (simplified: stored as vectors of char codes) ── */
            else if (opcode == OP_STR_REF) {
                /* STR_REF: TOS=index, SOS=string_ptr → char at index */
                int sptr = (int)sos;
                int idx = (int)tos;
                if (sptr >= 0 && sptr < HEAP_SIZE) {
                    int len = (int)g_heap[sptr];
                    if (idx >= 0 && idx < len && sptr + 1 + idx < HEAP_SIZE)
                        x[S_TOS] = g_heap[sptr + 1 + idx];
                }
                x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                x[S_DEPTH] -= 1;
                x[S_TYPE_TOS] = TYPE_NUMBER; /* char code is a number */
                x[S_TYPE_SOS] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_R3]; x[S_TYPE_R3] = TYPE_NUMBER;
            } else if (opcode == OP_STR_LEN) {
                /* STR_LEN: TOS=string_ptr → push length */
                int sptr = (int)tos;
                if (sptr >= 0 && sptr < HEAP_SIZE)
                    x[S_TOS] = g_heap[sptr];
                x[S_TYPE_TOS] = TYPE_NUMBER;
            }
            /* ── Type predicates ── */
            else if (opcode == OP_STR_P) {
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_STRING) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            } else if (opcode == OP_BOOL_P) {
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_BOOL) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            } else if (opcode == OP_PROC_P) {
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_CLOSURE) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            } else if (opcode == OP_VEC_P) {
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_VECTOR) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            }
            /* ── Exceptions ── */
            else if (opcode == OP_PUSH_HANDLER) {
                int handler_pc = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                if (g_exc_count < MAX_EXC_HANDLERS) {
                    g_exc_handlers[g_exc_count].handler_pc = (float)handler_pc;
                    g_exc_handlers[g_exc_count].saved_depth = x[S_DEPTH];
                    for (int i = 0; i < MEM_SIZE; i++)
                        g_exc_handlers[g_exc_count].saved_mem[i] = x[S_MEM0 + i];
                    g_exc_handlers[g_exc_count].saved_tos = x[S_TOS];
                    g_exc_handlers[g_exc_count].saved_sos = x[S_SOS];
                    g_exc_handlers[g_exc_count].saved_r2 = x[S_R2];
                    g_exc_handlers[g_exc_count].saved_r3 = x[S_R3];
                    g_exc_handlers[g_exc_count].saved_type_tos = x[S_TYPE_TOS];
                    g_exc_handlers[g_exc_count].saved_type_sos = x[S_TYPE_SOS];
                    g_exc_handlers[g_exc_count].saved_type_r2 = x[S_TYPE_R2];
                    g_exc_handlers[g_exc_count].saved_type_r3 = x[S_TYPE_R3];
                    g_exc_count++;
                }
            } else if (opcode == OP_POP_HANDLER) {
                if (g_exc_count > 0) g_exc_count--;
            } else if (opcode == OP_GET_EXN) {
                /* Push current exception value */
                x[S_TYPE_R3] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_SOS]; x[S_TYPE_SOS] = x[S_TYPE_TOS];
                x[S_R3] = x[S_R2]; x[S_R2] = x[S_SOS]; x[S_SOS] = x[S_TOS];
                x[S_TOS] = g_current_exn;
                x[S_TYPE_TOS] = TYPE_NUMBER;
                x[S_DEPTH] += 1;
            }
            /* ── Continuations ── */
            else if (opcode == OP_CALLCC) {
                /* CALLCC: capture full continuation state onto the heap and call
                 * the function (TOS) with the continuation as its sole argument.
                 *
                 * Heap layout (16 floats per continuation):
                 *   [ptr+ 0]  return_pc         (resume address)
                 *   [ptr+ 1]  saved_depth        (stack depth at capture)
                 *   [ptr+ 2]  saved_mem[0]
                 *   [ptr+ 3]  saved_mem[1]
                 *   [ptr+ 4]  saved_mem[2]
                 *   [ptr+ 5]  saved_mem[3]
                 *   [ptr+ 6]  saved_tos
                 *   [ptr+ 7]  saved_sos
                 *   [ptr+ 8]  saved_r2
                 *   [ptr+ 9]  saved_r3
                 *   [ptr+10]  saved_type_tags[0] (TOS type)
                 *   [ptr+11]  saved_type_tags[1] (SOS type)
                 *   [ptr+12]  saved_type_tags[2] (R2 type)
                 *   [ptr+13]  saved_type_tags[3] (R3 type)
                 *   [ptr+14]  saved_frame_count
                 *   [ptr+15]  saved_wind_depth
                 */
                #define CONT_SIZE 16
                float func_pc_cc = tos;
                float cont_ptr = -1.0f;
                if (g_heap_ptr + CONT_SIZE <= HEAP_SIZE) {
                    int ptr = g_heap_ptr;
                    g_heap[ptr +  0] = x[S_PC];                /* return PC */
                    g_heap[ptr +  1] = x[S_DEPTH] - 1;         /* depth (pop func ref) */
                    g_heap[ptr +  2] = x[S_MEM0];
                    g_heap[ptr +  3] = x[S_MEM1];
                    g_heap[ptr +  4] = x[S_MEM2];
                    g_heap[ptr +  5] = x[S_MEM3];
                    g_heap[ptr +  6] = sos;                    /* caller TOS (below func) */
                    g_heap[ptr +  7] = r2;                     /* caller SOS */
                    g_heap[ptr +  8] = r3;                     /* caller R2 */
                    g_heap[ptr +  9] = 0;                      /* caller R3 */
                    g_heap[ptr + 10] = x[S_TYPE_SOS];          /* type tags */
                    g_heap[ptr + 11] = x[S_TYPE_R2];
                    g_heap[ptr + 12] = x[S_TYPE_R3];
                    g_heap[ptr + 13] = TYPE_NUMBER;
                    g_heap[ptr + 14] = (float)g_frame_count;
                    g_heap[ptr + 15] = (float)g_wind_depth;
                    g_heap_ptr += CONT_SIZE;
                    cont_ptr = (float)ptr;
                }
                /* Set up callee: continuation object is the sole argument (MEM0) */
                x[S_MEM0] = cont_ptr;
                x[S_MEM1] = 0; x[S_MEM2] = 0; x[S_MEM3] = 0;
                x[S_PC] = func_pc_cc;
                x[S_TOS] = 0; x[S_SOS] = 0; x[S_R2] = 0; x[S_R3] = 0;
                x[S_TYPE_TOS] = TYPE_NUMBER; x[S_TYPE_SOS] = TYPE_NUMBER;
                x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
                x[S_DEPTH] = 0;
            } else if (opcode == OP_INVOKE_CC) {
                /* INVOKE_CC: restore the full continuation state from the heap
                 * and resume execution with TOS as the return value.
                 *
                 * SOS = continuation object (heap pointer), TOS = value to return. */
                float retval_cc = tos;
                float rettype_cc = x[S_TYPE_TOS];
                int cptr = (int)sos;
                if (cptr >= 0 && cptr + CONT_SIZE <= HEAP_SIZE) {
                    /* Restore full machine state from heap */
                    x[S_PC]   = g_heap[cptr +  0];
                    x[S_MEM0] = g_heap[cptr +  2];
                    x[S_MEM1] = g_heap[cptr +  3];
                    x[S_MEM2] = g_heap[cptr +  4];
                    x[S_MEM3] = g_heap[cptr +  5];
                    /* Restore stack: return value on top, then saved registers */
                    x[S_TOS]  = retval_cc;
                    x[S_SOS]  = g_heap[cptr +  6];
                    x[S_R2]   = g_heap[cptr +  7];
                    x[S_R3]   = g_heap[cptr +  8];
                    /* Restore type tags */
                    x[S_TYPE_TOS] = rettype_cc;
                    x[S_TYPE_SOS] = g_heap[cptr + 10];
                    x[S_TYPE_R2]  = g_heap[cptr + 11];
                    x[S_TYPE_R3]  = g_heap[cptr + 12];
                    /* Restore depth (+1 for the return value pushed) */
                    x[S_DEPTH] = g_heap[cptr +  1] + 1;
                    /* Restore frame count and wind depth */
                    g_frame_count = (int)g_heap[cptr + 14];
                    g_wind_depth  = (int)g_heap[cptr + 15];
                }
                #undef CONT_SIZE
            }
            /* ── Variadic / dynamic-wind ── */
            else if (opcode == OP_PACK_REST) {
                int n_fixed = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                /* Cons remaining MEM slots [n_fixed..MEM_SIZE-1] into a list */
                float list_ptr = -1.0f; /* nil */
                for (int i = MEM_SIZE - 1; i >= n_fixed; i--) {
                    if (g_heap_ptr + 2 <= HEAP_SIZE) {
                        int ptr = g_heap_ptr;
                        g_heap[g_heap_ptr++] = x[S_MEM0 + i]; /* car */
                        g_heap[g_heap_ptr++] = list_ptr;        /* cdr */
                        list_ptr = (float)ptr;
                    }
                }
                if (n_fixed >= 0 && n_fixed < MEM_SIZE) {
                    x[S_MEM0 + n_fixed] = list_ptr;
                }
            } else if (opcode == OP_WIND_PUSH) {
                /* TOS = after thunk closure */
                if (g_wind_depth < MAX_WINDS) {
                    g_wind_stack[g_wind_depth].after_thunk_ptr = x[S_TOS];
                    g_wind_stack[g_wind_depth].frame_depth = g_frame_count;
                    g_wind_depth++;
                }
                /* Pop the after thunk */
                x[S_TOS] = x[S_SOS]; x[S_SOS] = x[S_R2]; x[S_R2] = x[S_R3]; x[S_R3] = 0;
                x[S_TYPE_TOS] = x[S_TYPE_SOS]; x[S_TYPE_SOS] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_R3]; x[S_TYPE_R3] = TYPE_NUMBER;
                x[S_DEPTH] -= 1;
            } else if (opcode == OP_WIND_POP) {
                if (g_wind_depth > 0) g_wind_depth--;
            }
            /* ── Native call dispatch to runtime libraries ── */
            else if (opcode == OP_NATIVE_CALL) {
                int native_id = (pc >= 0 && pc < n_instr) ? prog[pc].operand : -1;
                VmRegionStack* rs = vm_get_regions();

                /* Complex number operations (300-319) */
                if (native_id >= 300 && native_id < 320) {
                    if (native_id == 300) { /* make-rectangular */
                        VmComplex* z = vm_complex_new(rs, sos, tos);
                        if (z && g_heap_ptr + 2 <= HEAP_SIZE) {
                            int ptr = g_heap_ptr;
                            g_heap[g_heap_ptr++] = (float)z->real;
                            g_heap[g_heap_ptr++] = (float)z->imag;
                            /* Pop two args, push result */
                            x[S_TOS] = (float)ptr;
                            x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                            x[S_DEPTH] -= 1;
                            x[S_TYPE_TOS] = (float)VAL_COMPLEX;
                        }
                    } else if (native_id == 302) { /* real-part */
                        int ptr = (int)tos;
                        if (ptr >= 0 && ptr + 1 < HEAP_SIZE) {
                            x[S_TOS] = g_heap[ptr]; /* real part */
                            x[S_TYPE_TOS] = TYPE_NUMBER;
                        }
                    } else if (native_id == 303) { /* imag-part */
                        int ptr = (int)tos;
                        if (ptr >= 0 && ptr + 1 < HEAP_SIZE) {
                            x[S_TOS] = g_heap[ptr + 1]; /* imag part */
                            x[S_TYPE_TOS] = TYPE_NUMBER;
                        }
                    } else if (native_id == 304) { /* magnitude */
                        int ptr = (int)tos;
                        if (ptr >= 0 && ptr + 1 < HEAP_SIZE) {
                            float re = g_heap[ptr], im = g_heap[ptr + 1];
                            x[S_TOS] = sqrtf(re * re + im * im);
                            x[S_TYPE_TOS] = TYPE_NUMBER;
                        }
                    } else if (native_id == 317) { /* complex? */
                        x[S_TOS] = (x[S_TYPE_TOS] == (float)VAL_COMPLEX) ? 1.0f : 0.0f;
                        x[S_TYPE_TOS] = TYPE_BOOL;
                    }
                    /* Other complex ops use the same pattern: read from g_heap, compute, write back */
                }
                /* All other runtime ID ranges: registered but float VM delegates to C exec_loop */
                /* The weight matrix path handles basic opcodes; complex operations dispatch here */
            } else if (opcode >= OP_AD_VAR && opcode <= OP_AD_SQRT) {
                /* AD forward ops: Layer 3 weight matrices handle stack manipulation
                 * and flag setting. The VALUE computation requires tape random-access
                 * which is handled here for correctness. Layer 2 indicator neurons
                 * could compute this in a fully weight-only path (future work). */
                int tlen = (int)roundf(x[S_AD_TAPE_LEN]) - 1; /* already incremented by Layer 4 */
                if (tlen >= 0 && tlen < AD_MAX_TAPE) {
                    float op_type = x[S_AD_TAPE_BASE + tlen * AD_NODE_FIELDS + AD_F_OP];
                    int li = (int)roundf(x[S_AD_TAPE_BASE + tlen * AD_NODE_FIELDS + AD_F_LEFT]);
                    int ri = (int)roundf(x[S_AD_TAPE_BASE + tlen * AD_NODE_FIELDS + AD_F_RIGHT]);
                    float lv = (li >= 0 && li < tlen) ? x[S_AD_TAPE_BASE + li * AD_NODE_FIELDS + AD_F_VALUE] : 0;
                    float rv = (ri >= 0 && ri < tlen) ? x[S_AD_TAPE_BASE + ri * AD_NODE_FIELDS + AD_F_VALUE] : 0;
                    float val = 0;
                    if (fabsf(op_type - AD_OP_VAR) < 0.5f || fabsf(op_type - AD_OP_CONST) < 0.5f) {
                        val = x[S_AD_TAPE_BASE + tlen * AD_NODE_FIELDS + AD_F_VALUE]; /* already set by operand */
                    } else if (fabsf(op_type - AD_OP_ADD) < 0.5f) { val = lv + rv; }
                    else if (fabsf(op_type - AD_OP_SUB) < 0.5f) { val = lv - rv; }
                    else if (fabsf(op_type - AD_OP_MUL) < 0.5f) { val = lv * rv; }
                    else if (fabsf(op_type - AD_OP_NEG) < 0.5f) { val = -lv; }
                    else if (fabsf(op_type - AD_OP_ABS) < 0.5f) { val = fabsf(lv); }
                    else if (fabsf(op_type - AD_OP_RELU) < 0.5f) { val = lv > 0 ? lv : 0; }
                    else if (fabsf(op_type - AD_OP_SIGMOID) < 0.5f) { val = 1.0f/(1.0f+expf(-lv)); }
                    else if (fabsf(op_type - AD_OP_TANH) < 0.5f) { val = tanhf(lv); }
                    else if (fabsf(op_type - AD_OP_EXP) < 0.5f) { val = expf(lv); }
                    else if (fabsf(op_type - AD_OP_LOG) < 0.5f) { val = logf(lv > 0 ? lv : 1e-15f); }
                    else if (fabsf(op_type - AD_OP_SQRT) < 0.5f) { val = sqrtf(lv > 0 ? lv : 0); }
                    x[S_AD_TAPE_BASE + tlen * AD_NODE_FIELDS + AD_F_VALUE] = val;
                }
            } else if (opcode == OP_AD_BACKWARD) {
                /* Seed gradient for backward pass */
                int output_idx = (int)roundf(x[S_AD_CURSOR]);
                int tlen = (int)roundf(x[S_AD_TAPE_LEN]);
                if (output_idx >= 0 && output_idx < tlen)
                    x[S_AD_TAPE_BASE + output_idx * AD_NODE_FIELDS + AD_F_GRAD] = 1.0f;
            } else if (opcode == OP_AD_GRAD) {
                /* Read gradient of tape[TOS] */
                int ni = (int)roundf(x[S_TOS]);
                int tlen = (int)roundf(x[S_AD_TAPE_LEN]);
                float grad = 0;
                if (ni >= 0 && ni < tlen)
                    grad = x[S_AD_TAPE_BASE + ni * AD_NODE_FIELDS + AD_F_GRAD];
                x[S_TOS] = grad;
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
        if (argc < 0) argc = 0;
        if (argc > MEM_SIZE) argc = MEM_SIZE;
        float func_pc = x[S_TOS];
        /* If func_pc looks like a heap closure pointer, dereference to get entry point */
        int fptr = (int)func_pc;
        if (fptr >= 0 && fptr + 1 < g_heap_ptr && fptr + 1 < HEAP_SIZE && g_heap[fptr] < 10000) {
            /* Check if this is a closure (heap[ptr] = entry_pc, heap[ptr+1] = n_upvals) */
            float candidate = g_heap[fptr];
            if (candidate >= 0 && candidate < n_instr) func_pc = candidate;
            g_current_closure_ptr = fptr; /* track current closure for GET_UPVALUE */
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
        } else {
            fprintf(stderr, "ERROR: call frame overflow (max 64)\n");
            x[S_HALT] = 1;
        }

        /* Set up callee: args go to MEM0..MEM(argc-1) */
        float args[4] = {x[S_SOS], x[S_R2], x[S_R3], 0};
        for (int i = 0; i < MEM_SIZE; i++) x[S_MEM0+i] = 0;
        for (int i = 0; i < argc && i < MEM_SIZE; i++)
            x[S_MEM0+i] = args[i];

        x[S_PC] = func_pc;
        x[S_TOS] = 0; x[S_SOS] = 0; x[S_R2] = 0; x[S_R3] = 0;
        x[S_TYPE_TOS] = TYPE_NUMBER; x[S_TYPE_SOS] = TYPE_NUMBER;
        x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
        x[S_DEPTH] = 0;
        x[S_IS_CALL] = 0;
    }

    /* IS_RET: pop call frame, restore caller state, push return value
     * TOS = return value */
    if (x[S_IS_RET] > 0.5f) {
        float retval = x[S_TOS];
        float rettype = x[S_TYPE_TOS];

        if (g_frame_count > 0) {
            g_frame_count--;
            CallFrame* f = &g_frames[g_frame_count];
            x[S_PC] = f->return_pc;
            for (int i = 0; i < MEM_SIZE; i++)
                x[S_MEM0+i] = f->saved_mem[i];
            /* Restore caller's stack with return value pushed on top */
            x[S_TOS] = retval;
            x[S_TYPE_TOS] = rettype;
            x[S_SOS] = f->saved_tos;
            x[S_R2] = f->saved_sos;
            x[S_R3] = f->saved_r2;
            x[S_TYPE_SOS] = TYPE_NUMBER; x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
            x[S_DEPTH] = f->saved_depth + 1; /* +1 for return value */
        }
        x[S_IS_RET] = 0;
    }
}

static int run_simulated(const Instr* prog, int n_instr, float* outputs, int max_out) {
    float pe[256][D];
    for (int p = 0; p < n_instr && p < 256; p++) embed_instruction(&prog[p], p, pe[p]);
    float state[D]; memset(state, 0, sizeof(state)); state[S_OUTPUT] = -1;
    g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
    int n_out = 0, step_count = 0;
    for (int step = 0; step < 8192; step++) {
        step_count++;
        float x[D]; memcpy(x, state, sizeof(x));
        float tmp[D];

        /* Clear transient dims at start of each cycle (matches execute_step).
         * S_AD_IS_BACKWARD (113) and S_AD_MODE (38) persist — checked by loop. */
        for (int i = S_AD_CUR_OP; i <= S_AD_RIGHT_VALUE; i++) x[i] = 0;
        x[S_AD_IS_FORWARD] = 0; /* 112 */
        /* Skip S_AD_IS_BACKWARD (113) — must persist for backward check */
        for (int i = S_AD_GRAD_ACCUM; i <= S_AD_SPARE8; i++) x[i] = 0;

        if (x[S_AD_IS_BACKWARD] > 0.5f) {
            /* Backward mode: use ad_backward_step directly on the state.
             * This matches the reference interpreter exactly. The layer-based
             * backward (layers 2/3/4) will be used in Phase 3 for the matrix
             * forward pass where direct C manipulation isn't available. */
            ad_backward_step(x);
        } else {
            /* Normal execution: all 5 layers */
            layer0_attention(x, pe, n_instr, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
            layer1_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
            layer2_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
            layer3_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
            layer4_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
            exec_loop_postprocess(x, prog, n_instr);
        }
        if (x[S_HAS_OUT] > 0.5f && n_out < max_out) outputs[n_out++] = x[S_OUTPUT];
        if (x[S_HALT] > 0.5f) break;
        memcpy(state, x, sizeof(state));
    }
    g_last_sim_steps = step_count;
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

/* Gated pair for AD opcodes — same as add_gated_pair but also gates on NOT backward */
static int add_gated_pair_ad(InterpreterWeights* w, int L, int n,
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
        W(w->ff_gate[L], S_AD_IS_BACKWARD, j, FFN_DIM) += -SCALE; /* suppress during backward */
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
        /* Universal: clear intermediate dims (Zone A: 16-31, Zone D: 112-127) */
        for (int d = S_OPCODE; d <= S_ABS_DELTA; d++)
            n = add_unconditional(w, L, n, d, -1.0f, 0, d, 1.0f);
        for (int d = S_AD_IS_FORWARD; d <= S_AD_SPARE8; d++)
            n = add_unconditional(w, L, n, d, -1.0f, 0, d, 1.0f);

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

        /* ── AD Forward Ops (64-78) ── */

        /* OP_AD_VAR (64): push tape_len as tape index, set AD_CUR fields, AD_IS_FORWARD
         * Stack: push down (R3←R2, R2←SOS, SOS←TOS, TOS←tape_len), depth++ */
        n = add_gated_pair_ad(w,L,n, 64, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, S_AD_TAPE_LEN,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, -1,0,-1,0,-1,0,-1,0, (float)AD_OP_VAR, S_AD_CUR_OP, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, S_OPERAND,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_VALUE, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, -1,0,-1,0,-1,0,-1,0, -1.0f, S_AD_CUR_LEFT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, -1,0,-1,0,-1,0,-1,0, -1.0f, S_AD_CUR_RIGHT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_IS_FORWARD, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_MODE, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);

        /* OP_AD_CONST (65): same as VAR but with AD_OP_CONST */
        n = add_gated_pair_ad(w,L,n, 65, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, S_AD_TAPE_LEN,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, -1,0,-1,0,-1,0,-1,0, (float)AD_OP_CONST, S_AD_CUR_OP, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, S_OPERAND,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_VALUE, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, -1,0,-1,0,-1,0,-1,0, -1.0f, S_AD_CUR_LEFT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, -1,0,-1,0,-1,0,-1,0, -1.0f, S_AD_CUR_RIGHT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_IS_FORWARD, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_MODE, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);

        /* OP_AD_ADD (66): binary. Pop 2, push tape_len. Set CUR fields.
         * Note: value computation (left_val + right_val) happens in simulated
         * layer3_ffn via direct tape access. The weight matrix path uses the
         * same simulated code path for value computation — the gated pairs
         * handle stack manipulation and flag setting only.
         * The actual value is set by layer3_ffn's AD forward computation. */
        n = add_gated_pair_ad(w,L,n, 66, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair_ad(w,L,n, 66, S_AD_TAPE_LEN,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 66, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 66, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_ad(w,L,n, 66, -1,0,-1,0,-1,0,-1,0, 0, S_R3, 1.0f); /* R3=0 */
        n = add_gated_pair_ad(w,L,n, 66, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair_ad(w,L,n, 66, -1,0,-1,0,-1,0,-1,0, (float)AD_OP_ADD, S_AD_CUR_OP, 1.0f);
        n = add_gated_pair_ad(w,L,n, 66, S_SOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_LEFT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 66, S_TOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_RIGHT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 66, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_IS_FORWARD, 1.0f);
        n = add_gated_pair_ad(w,L,n, 66, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);

        /* OP_AD_SUB (67): same stack pattern as ADD */
        n = add_gated_pair_ad(w,L,n, 67, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, S_AD_TAPE_LEN,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, -1,0,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, -1,0,-1,0,-1,0,-1,0, (float)AD_OP_SUB, S_AD_CUR_OP, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, S_SOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_LEFT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, S_TOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_RIGHT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_IS_FORWARD, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);

        /* OP_AD_MUL (68): same pattern */
        n = add_gated_pair_ad(w,L,n, 68, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, S_AD_TAPE_LEN,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, -1,0,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, -1,0,-1,0,-1,0,-1,0, (float)AD_OP_MUL, S_AD_CUR_OP, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, S_SOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_LEFT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, S_TOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_RIGHT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_IS_FORWARD, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);

        /* Unary AD ops (69-76): replace TOS with tape_len, set CUR fields */
        for (int uop = 69; uop <= 76; uop++) {
            float ad_op_type;
            switch (uop) {
                case 69: ad_op_type = AD_OP_NEG; break;
                case 70: ad_op_type = AD_OP_ABS; break;
                case 71: ad_op_type = AD_OP_RELU; break;
                case 72: ad_op_type = AD_OP_SIGMOID; break;
                case 73: ad_op_type = AD_OP_TANH; break;
                case 74: ad_op_type = AD_OP_EXP; break;
                case 75: ad_op_type = AD_OP_LOG; break;
                case 76: ad_op_type = AD_OP_SQRT; break;
                default: ad_op_type = 0; break;
            }
            n = add_gated_pair_ad(w,L,n, uop, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
            n = add_gated_pair_ad(w,L,n, uop, S_AD_TAPE_LEN,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
            n = add_gated_pair_ad(w,L,n, uop, -1,0,-1,0,-1,0,-1,0, ad_op_type, S_AD_CUR_OP, 1.0f);
            n = add_gated_pair_ad(w,L,n, uop, S_TOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_LEFT, 1.0f);
            n = add_gated_pair_ad(w,L,n, uop, -1,0,-1,0,-1,0,-1,0, -1.0f, S_AD_CUR_RIGHT, 1.0f);
            n = add_gated_pair_ad(w,L,n, uop, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_IS_FORWARD, 1.0f);
            n = add_gated_pair_ad(w,L,n, uop, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);
        }

        /* OP_AD_BACKWARD (77): set backward mode, seed gradient */
        n = add_gated_pair_ad(w,L,n, 77, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair_ad(w,L,n, 77, -1,0,-1,0,-1,0,-1,0, 2.0f, S_AD_MODE, 1.0f);
        n = add_gated_pair_ad(w,L,n, 77, S_TOS,1,S_AD_CURSOR,-1,-1,0,-1,0, 0, S_AD_CURSOR, 1.0f);
        n = add_gated_pair_ad(w,L,n, 77, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_IS_BACKWARD, 1.0f);
        /* Pop TOS */
        n = add_gated_pair_ad(w,L,n, 77, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 77, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 77, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_ad(w,L,n, 77, -1,0,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair_ad(w,L,n, 77, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair_ad(w,L,n, 77, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);

        /* OP_AD_GRAD (78): replace TOS with gradient of tape[TOS] — IS_NATIVE for now */
        n = add_gated_pair_ad(w,L,n, 78, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair_ad(w,L,n, 78, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);

        /* AD delegated ops (79-82): IS_NATIVE */
        for (int opc = 79; opc <= 82; opc++) {
            n = add_gated_pair_ad(w,L,n, opc, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
            n = add_gated_pair_ad(w,L,n, opc, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);
        }

        printf("[WEIGHT_GEN] Layer 3: %d neurons used out of %d\n", n, FFN_DIM);
    }

    /* ── Layer 4: AD tape write (gated FFN) ──
     * When AD_IS_FORWARD is set, write AD_CUR_* fields into tape[tape_len].
     * Uses indicator(tape_len, slot) gating. Also increments tape_len. */
    {
        const int L = 4;
        w->ff_type[L] = 2;
        int n = 0;

        /* For each tape slot i: gate on indicator(AD_TAPE_LEN==i) * AD_IS_FORWARD
         * Write AD_CUR_OP, AD_CUR_VALUE, AD_CUR_LEFT, AD_CUR_RIGHT, AD_CUR_SAVED to tape[i] */
        for (int slot = 0; slot < AD_MAX_TAPE; slot++) {
            /* Gate: sigmoid(SCALE*(AD_TAPE_LEN - slot + 0.5)) * sigmoid(SCALE*(AD_IS_FORWARD - 0.5))
             * Combined: gate on both conditions via the gated neuron pattern.
             * We use: gate_weight[AD_TAPE_LEN] = SCALE, gate_weight[AD_IS_FORWARD] = SCALE,
             * gate_bias = SCALE*(-slot + 0.5 - 0.5) = SCALE*(-slot)
             * This approximates: indicator(tape_len==slot) AND is_forward. */

            /* op field */
            W(w->ff_gate[L], S_AD_TAPE_LEN, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot + 0.5f);
            W(w->ff_up[L], S_AD_CUR_OP, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_OP, D) = 1.0f;
            n++;
            W(w->ff_gate[L], S_AD_TAPE_LEN, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot - 0.5f);
            W(w->ff_up[L], S_AD_CUR_OP, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_OP, D) = -1.0f;
            n++;

            /* value field */
            W(w->ff_gate[L], S_AD_TAPE_LEN, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot + 0.5f);
            W(w->ff_up[L], S_AD_CUR_VALUE, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_VALUE, D) = 1.0f;
            n++;
            W(w->ff_gate[L], S_AD_TAPE_LEN, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot - 0.5f);
            W(w->ff_up[L], S_AD_CUR_VALUE, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_VALUE, D) = -1.0f;
            n++;

            /* left field */
            W(w->ff_gate[L], S_AD_TAPE_LEN, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot + 0.5f);
            W(w->ff_up[L], S_AD_CUR_LEFT, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_LEFT, D) = 1.0f;
            n++;
            W(w->ff_gate[L], S_AD_TAPE_LEN, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot - 0.5f);
            W(w->ff_up[L], S_AD_CUR_LEFT, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_LEFT, D) = -1.0f;
            n++;

            /* right field */
            W(w->ff_gate[L], S_AD_TAPE_LEN, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot + 0.5f);
            W(w->ff_up[L], S_AD_CUR_RIGHT, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_RIGHT, D) = 1.0f;
            n++;
            W(w->ff_gate[L], S_AD_TAPE_LEN, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot - 0.5f);
            W(w->ff_up[L], S_AD_CUR_RIGHT, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_RIGHT, D) = -1.0f;
            n++;
        }

        /* Increment tape_len when IS_FORWARD is set */
        W(w->ff_gate[L], S_AD_IS_FORWARD, n, FFN_DIM) = SCALE;
        w->ff_gate_b[L][n] = SCALE * (-0.5f);
        w->ff_up_b[L][n] = 1.0f;
        W(w->ff_down[L], n, S_AD_TAPE_LEN, D) = 1.0f;
        n++;

        printf("[WEIGHT_GEN] Layer 4: %d neurons used out of %d\n", n, FFN_DIM);
    }

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
    g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
    int n_out=0, step_count=0;
    for(int step=0;step<8192;step++){
        step_count++;
        /* Clear transient dims at start of cycle */
        for (int i = S_AD_CUR_OP; i <= S_AD_RIGHT_VALUE; i++) state[i] = 0;
        state[S_AD_IS_FORWARD] = 0;
        /* Keep S_AD_IS_BACKWARD */
        for (int i = S_AD_GRAD_ACCUM; i <= S_AD_SPARE8; i++) state[i] = 0;

        float next[D];
        if (state[S_AD_IS_BACKWARD] > 0.5f) {
            /* Backward: use ad_backward_step directly */
            memcpy(next, state, sizeof(next));
            ad_backward_step(next);
        } else {
            forward_with_weights(w,state,pe,n_instr,next);
            exec_loop_postprocess(next, prog, n_instr);
        }
        if(next[S_HAS_OUT]>0.5f&&n_out<max_out) outputs[n_out++]=next[S_OUTPUT];
        if(next[S_HALT]>0.5f) break;
        memcpy(state,next,sizeof(state));
    }
    g_last_mat_steps = step_count;
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
    g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
    int rn = run_reference(prog, n, r, 64);
    g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
    int sn = run_simulated(prog, n, s, 64);
    g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
    int mn = g_weights ? run_with_weights(g_weights, prog, n, m, 64) : 0;

    float rv = rn>0?r[0]:-9999, sv = sn>0?s[0]:-9999, mv = mn>0?m[0]:-9999;
    int ok_r = fabsf(rv-expected)<0.01f;
    int ok_s = fabsf(sv-expected)<0.01f;
    int ok_m = g_weights ? fabsf(mv-expected)<0.01f : 1;

    printf("  %-25s ref=%7.1f sim=%7.1f mat=%7.1f  %s%s%s\n",
           name, rv, sv, mv,
           ok_r?"":"ref:FAIL ", ok_s?"":"sim:FAIL ",
           (g_weights && !ok_m)?"mat:FAIL ":"");

    /* Uncomment for per-test metrics: */
    /* printf("    steps: ref=%d sim=%d mat=%d heap=%d\n", g_last_ref_steps, g_last_sim_steps, g_last_mat_steps, g_heap_ptr); */

    if (ok_r && ok_s && ok_m) n_pass++; else n_fail++;
}

/* Reference + simulated test — 2-way comparison (Phase 2: no matrix weights yet) */
static void test_ref(const char* name, const Instr* prog, int n, float expected) {
    float r[64], s[64];
    g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
    int rn = run_reference(prog, n, r, 64);
    g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
    int sn = run_simulated(prog, n, s, 64);
    float rv = rn>0?r[0]:-9999, sv = sn>0?s[0]:-9999;
    int ok_r = fabsf(rv-expected)<0.01f;
    int ok_s = fabsf(sv-expected)<0.01f;
    printf("  %-40s ref=%7.2f sim=%7.2f  %s%s\n",
           name, rv, sv,
           ok_r?"":"ref:FAIL ", ok_s?"":"sim:FAIL ");
    if (ok_r && ok_s) n_pass++; else n_fail++;
}

int main() {
    printf("=== Eshkol VM Weight Compiler ===\n\n");

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
        g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
        if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
        run_reference(p,5,r,1);
        g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
        if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
        run_simulated(p,5,s,1);
        g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
        if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
        if(g_weights) run_with_weights(g_weights,p,5,m,1);
        if(fabsf(r[0]-(float)(a*b))<0.01f && fabsf(s[0]-(float)(a*b))<0.01f
           && (!g_weights || fabsf(m[0]-(float)(a*b))<0.01f)) mul_ok++;
    }
    printf("%d/100\n", mul_ok);
    if(mul_ok==100) n_pass++; else n_fail++;

    /* ── Stage 8: Edge cases ── */
    printf("\n  --- Stage 8: Edge cases ---\n");

    /* Division by zero -> halt */
    { Instr p[]={{OP_CONST,5},{OP_CONST,0},{OP_DIV,0},{OP_PRINT,0},{OP_HALT,0}};
      /* Should halt (div by zero), no output */
      float out[1];
      g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
      int nout = run_reference(p, 5, out, 1);
      printf("  %-25s %s\n", "div-by-zero", nout == 0 ? "OK" : "FAIL");
      if (nout == 0) n_pass++; else n_fail++;
    }

    /* Modulo by zero -> halt */
    { Instr p[]={{OP_CONST,5},{OP_CONST,0},{OP_MOD,0},{OP_PRINT,0},{OP_HALT,0}};
      float out[1];
      g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
      int nout = run_reference(p, 5, out, 1);
      printf("  %-25s %s\n", "mod-by-zero", nout == 0 ? "OK" : "FAIL");
      if (nout == 0) n_pass++; else n_fail++;
    }

    /* NULL_P on non-nil value */
    { Instr p[]={{OP_CONST,42},{OP_NULL_P,0},{OP_PRINT,0},{OP_HALT,0}};
      test("null?(42)=false", p, 4, 0); }

    /* Nested arithmetic: (3 + 4) * (5 - 2) = 21 */
    { Instr p[]={{OP_CONST,3},{OP_CONST,4},{OP_ADD,0},{OP_CONST,5},{OP_CONST,2},{OP_SUB,0},{OP_MUL,0},{OP_PRINT,0},{OP_HALT,0}};
      test("(3+4)*(5-2)", p, 9, 21); }

    /* ABS of negative */
    { Instr p[]={{OP_CONST,7},{OP_NEG,0},{OP_ABS,0},{OP_PRINT,0},{OP_HALT,0}};
      test("abs(-7)", p, 5, 7); }

    /* ── Stage 9: AD — native autodiff in weights ── */
    printf("\n  --- Stage 9: AD forward + backward ---\n");

    /* (Phase 2 simulated transformer AD verified — ref == sim for all 8 tests) */

    /* f(x) = x^2 at x=3: tape = [var(3), mul(0,0)=9], backward → grad[0] = 6 */
    { Instr p[]={
        {OP_AD_VAR, 3},       /* node 0: x=3 */
        {OP_DUP, 0},          /* duplicate node index 0 */
        {OP_DUP, 0},          /* stack: [0, 0, 0] */
        {OP_AD_MUL, 0},       /* node 1: x*x=9, stack: [0, 1] */
        {OP_AD_BACKWARD, 0},  /* backward from node 1 */
        {OP_CONST, 0},        /* push node index 0 */
        {OP_AD_GRAD, 0},      /* push grad of node 0 */
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx x^2 at 3 = 6", p, 9, 6); }

    /* f(x) = x+x at x=5: grad = 2 (fan-out test) */
    { Instr p[]={
        {OP_AD_VAR, 5},       /* node 0: x=5 */
        {OP_DUP, 0},
        {OP_DUP, 0},
        {OP_AD_ADD, 0},       /* node 1: x+x=10 */
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx (x+x) at 5 = 2", p, 9, 2); }

    /* f(x) = -x at x=7: grad = -1 */
    { Instr p[]={
        {OP_AD_VAR, 7},
        {OP_AD_NEG, 0},       /* node 1: -x=-7 */
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx (-x) at 7 = -1", p, 7, -1); }

    /* f(x,y) = x*y at (3,4): df/dx=4, df/dy=3.
     * We check df/dx. node 0=x=3, node 1=y=4, node 2=x*y */
    { Instr p[]={
        {OP_AD_VAR, 3},       /* node 0: x=3 */
        {OP_AD_VAR, 4},       /* node 1: y=4 */
        {OP_AD_MUL, 0},       /* node 2: x*y=12. Stack after: [2] (popped 0,1, pushed 2) */
        {OP_AD_BACKWARD, 0},  /* backward from node 2 */
        {OP_CONST, 0},        /* push 0 (node index for x) */
        {OP_AD_GRAD, 0},      /* grad of x = y = 4 */
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d(x*y)/dx at (3,4) = 4", p, 8, 4); }

    /* Same but check df/dy = 3 */
    { Instr p[]={
        {OP_AD_VAR, 3},
        {OP_AD_VAR, 4},
        {OP_AD_MUL, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 1},        /* push 1 (node index for y) */
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d(x*y)/dy at (3,4) = 3", p, 8, 3); }

    /* f(x) = exp(0): grad = exp(0) = 1 */
    { Instr p[]={
        {OP_AD_VAR, 0},       /* node 0: x=0 */
        {OP_AD_EXP, 0},       /* node 1: exp(0)=1 */
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx exp(0) = 1", p, 7, 1); }

    /* f(x) = sigmoid(0): grad = 0.25 */
    { Instr p[]={
        {OP_AD_VAR, 0},
        {OP_AD_SIGMOID, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx sigmoid(0) = 0.25", p, 7, 0.25f); }

    /* f(x) = relu(3): grad = 1 */
    { Instr p[]={
        {OP_AD_VAR, 3},
        {OP_AD_RELU, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx relu(3) = 1", p, 7, 1); }

    /* ── MLP gradient demo: f(w,b) = sigmoid(w*2 + b) at w=1, b=-1 ──
     * sigmoid(1*2 + -1) = sigmoid(1) ≈ 0.7311
     * df/dw = x * sigmoid'(w*x+b) = 2 * 0.7311 * (1 - 0.7311) ≈ 0.3932
     * df/db = sigmoid'(w*x+b) = 0.7311 * (1 - 0.7311) ≈ 0.1966
     * Uses 6 tape nodes: var(w=1), const(x=2), var(b=-1), mul(w,x), add(wx,b), sigmoid */
    printf("\n  --- MLP gradient demo ---\n");
    { Instr p[]={
        {OP_AD_VAR, 1},       /* node 0: w=1 */
        {OP_AD_CONST, 2},     /* node 1: x=2 (constant, no gradient) */
        {OP_AD_MUL, 0},       /* node 2: w*x=2 */
        {OP_AD_VAR, -1},      /* node 3: b=-1 */
        /* We need to get node 2 and node 3 on TOS/SOS for AD_ADD.
         * After AD_VAR(-1): stack is [0, 1, 2, 3]
         * AD_MUL popped 0,1 pushed 2. Stack: [2, ...]
         * AD_VAR(-1) pushed 3. Stack: [2, 3, ...]
         * But AD_ADD needs SOS=left, TOS=right. We need [2, 3] on stack.
         * Currently TOS=3, SOS=2. AD_ADD will use SOS=2(left), TOS=3(right). */
        {OP_AD_ADD, 0},       /* node 4: w*x + b = 2 + (-1) = 1 */
        {OP_AD_SIGMOID, 0},   /* node 5: sigmoid(1) ≈ 0.7311 */
        {OP_AD_BACKWARD, 0},  /* backward from node 5 */
        {OP_CONST, 0},        /* push 0 (node index for w) */
        {OP_AD_GRAD, 0},      /* df/dw */
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("MLP: df/dw sigmoid(w*2+b) at w=1,b=-1", p, 11, 0.3932f); }

    /* Same MLP, check df/db */
    { Instr p[]={
        {OP_AD_VAR, 1},
        {OP_AD_CONST, 2},
        {OP_AD_MUL, 0},
        {OP_AD_VAR, -1},
        {OP_AD_ADD, 0},
        {OP_AD_SIGMOID, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 3},        /* push 3 (node index for b) */
        {OP_AD_GRAD, 0},      /* df/db */
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("MLP: df/db sigmoid(w*2+b) at w=1,b=-1", p, 11, 0.1966f); }

    /* ── Edge cases ── */
    printf("\n  --- AD edge cases ---\n");

    /* Backward on constant: gradient should be 0 */
    { Instr p[]={
        {OP_AD_CONST, 42},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD edge: grad of output const = 1 (seed)", p, 6, 1); }

    /* Backward on lone variable: gradient = 1 (identity) */
    { Instr p[]={
        {OP_AD_VAR, 7},
        {OP_DUP, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD edge: grad of var = 1", p, 7, 1); }

    /* Chain: f(x) = x^2 + 2x at x=3 → gradient = 2x + 2 = 8 */
    { Instr p[]={
        {OP_AD_VAR, 3},       /* node 0: x=3 */
        {OP_DUP, 0},          /* dup node index 0 */
        {OP_DUP, 0},          /* stack: [0,0,0] */
        {OP_AD_MUL, 0},       /* node 1: x*x=9, stack: [0,1] */
        {OP_AD_CONST, 2},     /* node 2: const(2), stack: [0,1,2] */
        /* Need to get node 0 and node 2 adjacent. Stack is [0,1,2].
         * We need node0 on SOS for AD_MUL with node2 on TOS.
         * But stack is [0,1,2]. We need to swap. Use a different approach:
         * Just compute 2*x as const(2) * var(x). Get the index 0 from depth. */
        /* Actually: after AD_CONST(2), stack=[0,1,2]. We need [0] * [2] = 2x.
         * But SOS=1 and TOS=2. We'd compute node1 * node2 = x^2 * 2. Wrong.
         * We need a fresh reference to node 0. Use DUP on the 0 that's deep.
         * Alternative: create another var reference by reloading from stack.
         * Simplest: compute via AD_ADD of x to x instead of 2*x.
         * f(x) = x^2 + x + x = x^2 + 2x. grad = 2x + 2 = 8 at x=3. */
        /* Actually let me restructure: */
        {OP_POP, 0},          /* drop node 2, stack: [0,1] */
        {OP_POP, 0},          /* drop node 1, stack: [0] */
        {OP_DUP, 0},          /* stack: [0,0] */
        {OP_AD_ADD, 0},       /* node 3: x + x = 2x = 6, stack: [3] */
        /* Now we need node 1 (x^2) and node 3 (2x) on stack */
        {OP_CONST, 1},        /* push literal 1 (node index for x^2) */
        {OP_AD_ADD, 0},       /* node 4: x^2 + 2x = 9 + 6 = 15, stack: [4] */
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD chain: d/dx (x^2+2x) at 3 = 8", p, 16, 8); }

    printf("\n  [metrics] peak_heap=%d/%d\n", g_heap_ptr, HEAP_SIZE);

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);

    if (g_weights && n_fail == 0) {
        export_weights_binary(g_weights, "/tmp/interpreter_weights.bin");
    }

    /* ── Integration test: load bytecode from eshkol_compiler ── */
    /* Supports both legacy raw format and new ESKB section-based format.
     * Set ESHKOL_BC=path.eskb to load ESKB, or ESHKOL_BC=path.bc for legacy. */
    if (g_weights) {
        const char* bc_path = getenv("ESHKOL_BC");
        if (bc_path) {
            size_t pathlen = strlen(bc_path);
            int is_eskb = (pathlen > 5 && strcmp(bc_path + pathlen - 5, ".eskb") == 0);

            if (is_eskb) {
                /* Load ESKB section-based format via eskb_reader */
                EskbModule mod;
                if (eskb_load_file(bc_path, &mod) == 0) {
                    /* Build Instr array and constants for weight matrix execution */
                    Instr* prog = (Instr*)calloc(mod.code_len, sizeof(Instr));
                    float* constants = (float*)calloc(mod.n_constants > 0 ? mod.n_constants : 1, sizeof(float));
                    if (prog && constants) {
                        for (int i = 0; i < mod.code_len; i++) {
                            prog[i].op = (OpCode)mod.opcodes[i];
                            prog[i].operand = mod.operands[i];
                        }
                        for (int i = 0; i < mod.n_constants; i++) {
                            switch (mod.const_types[i]) {
                            case ESKB_CONST_INT64: constants[i] = (float)mod.const_ints[i]; break;
                            case ESKB_CONST_F64:   constants[i] = (float)mod.const_floats[i]; break;
                            case ESKB_CONST_BOOL:  constants[i] = (float)mod.const_ints[i]; break;
                            default:               constants[i] = 0.0f; break;
                            }
                        }

                        /* Resolve CONST operands: replace constant pool index with actual value.
                         * The eshkol_compiler uses CONST <pool_index>, but our weight matrix
                         * uses CONST <immediate_value>. Inline the constants. */
                        for (int i = 0; i < mod.code_len; i++) {
                            if (prog[i].op == OP_CONST && prog[i].operand >= 0 && prog[i].operand < mod.n_constants) {
                                prog[i].operand = (int)constants[prog[i].operand];
                            }
                        }

                        printf("\n  --- Integration (ESKB): %s (%d instructions, %d constants) ---\n",
                               bc_path, mod.code_len, mod.n_constants);

                        /* Check if program fits within weight matrix interpreter limits.
                         * The weight matrix interpreter has MEM_SIZE=4 local slots and
                         * pe[256] positional embeddings. Programs with closures/prelude
                         * may exceed these limits; run only the reference path for those. */
                        int simple_enough = (mod.code_len <= 256);

                        /* Run through all 3 paths */
                        float r[64], s[64], m[64];
                        g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
                        if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
                        int rn = run_reference(prog, mod.code_len, r, 64);
                        int sn = 0, mn = 0;
                        if (simple_enough) {
                            g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
                            if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
                            sn = run_simulated(prog, mod.code_len, s, 64);
                            g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
                            if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
                            mn = run_with_weights(g_weights, prog, mod.code_len, m, 64);
                        }

                        printf("  Outputs (ref): "); for(int i=0;i<rn;i++) printf("%.4g ", r[i]); printf("\n");
                        if (simple_enough) {
                            printf("  Outputs (sim): "); for(int i=0;i<sn;i++) printf("%.4g ", s[i]); printf("\n");
                            printf("  Outputs (mat): "); for(int i=0;i<mn;i++) printf("%.4g ", m[i]); printf("\n");
                        } else {
                            printf("  (sim/mat skipped: %d instructions > 256 limit)\n", mod.code_len);
                        }

                        int match = 1;
                        if (simple_enough) {
                            int n_max = rn < sn ? rn : sn; n_max = n_max < mn ? n_max : mn;
                            for (int i = 0; i < n_max; i++) {
                                if (fabsf(r[i]-s[i]) > 0.01f || fabsf(r[i]-m[i]) > 0.01f) match = 0;
                            }
                        }
                        printf("  3-way match: %s\n", simple_enough ? (match ? "YES" : "NO") : "REF-ONLY");

                        free(prog);
                        free(constants);
                    } else {
                        printf("ERROR: allocation failed for ESKB bytecode\n");
                        free(prog); free(constants);
                    }
                    eskb_module_free(&mod);
                } else {
                    printf("  ERROR: failed to load ESKB file %s\n", bc_path);
                }
            } else {
                /* Legacy raw bytecode format */
                FILE* bf = fopen(bc_path, "rb");
                if (bf) {
                    uint32_t magic = 0, n_instr = 0, n_const = 0;
                    if (fread(&magic, 4, 1, bf) != 1 || fread(&n_instr, 4, 1, bf) != 1 ||
                        fread(&n_const, 4, 1, bf) != 1) {
                        printf("ERROR: truncated bytecode header\n"); fclose(bf);
                    } else if (magic == 0x45534B42 && n_instr < 8192 && n_const < 8192) {
                        /* Read instructions */
                        Instr* prog = (Instr*)calloc(n_instr, sizeof(Instr));
                        float* constants = (float*)calloc(n_const > 0 ? n_const : 1, sizeof(float));
                        if (!prog || !constants) {
                            printf("ERROR: allocation failed for bytecode\n");
                            free(prog); free(constants); fclose(bf);
                        } else {
                        int read_ok = 1;
                        for (uint32_t i = 0; i < n_instr && read_ok; i++) {
                            uint8_t op; int32_t operand;
                            if (fread(&op, 1, 1, bf) != 1 || fread(&operand, 4, 1, bf) != 1) { read_ok = 0; break; }
                            prog[i].op = (OpCode)op;
                            prog[i].operand = operand;
                        }
                        /* Read constants */
                        for (uint32_t i = 0; i < n_const && read_ok; i++) {
                            uint8_t type; double val;
                            if (fread(&type, 1, 1, bf) != 1 || fread(&val, 8, 1, bf) != 1) { read_ok = 0; break; }
                            constants[i] = (float)val;
                        }
                        fclose(bf);
                        if (!read_ok) { printf("ERROR: truncated bytecode file\n"); free(prog); free(constants); }
                        else {

                        /* Resolve CONST operands */
                        for (uint32_t i = 0; i < n_instr; i++) {
                            if (prog[i].op == OP_CONST && prog[i].operand >= 0 && prog[i].operand < (int)n_const) {
                                prog[i].operand = (int)constants[prog[i].operand];
                            }
                        }

                        printf("\n  --- Integration (legacy): %s (%d instructions, %d constants) ---\n",
                               bc_path, n_instr, n_const);

                        /* Run through all 3 paths */
                        float r[64], s[64], m[64];
                        g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
                        int rn = run_reference(prog, n_instr, r, 64);
                        g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
                        int sn = run_simulated(prog, n_instr, s, 64);
                        g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
                        int mn = run_with_weights(g_weights, prog, n_instr, m, 64);

                        printf("  Outputs (ref): "); for(int i=0;i<rn;i++) printf("%.4g ", r[i]); printf("\n");
                        printf("  Outputs (sim): "); for(int i=0;i<sn;i++) printf("%.4g ", s[i]); printf("\n");
                        printf("  Outputs (mat): "); for(int i=0;i<mn;i++) printf("%.4g ", m[i]); printf("\n");

                        int match = 1;
                        int n_max = rn < sn ? rn : sn; n_max = n_max < mn ? n_max : mn;
                        for (int i = 0; i < n_max; i++) {
                            if (fabsf(r[i]-s[i]) > 0.01f || fabsf(r[i]-m[i]) > 0.01f) match = 0;
                        }
                        printf("  3-way match: %s\n", match ? "YES" : "NO");

                        free(prog);
                        free(constants);
                        } /* end read_ok */
                        } /* end alloc check */
                    } else {
                        printf("  ERROR: invalid bytecode file (magic=0x%08x, n_instr=%u)\n", magic, n_instr);
                        fclose(bf);
                    }
                } else {
                    printf("  ERROR: cannot open bytecode file %s\n", bc_path);
                }
            }
        }
    }

    if (g_weights) free(g_weights);
    return n_fail > 0 ? 1 : 0;
}
