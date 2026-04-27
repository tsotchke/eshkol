/**
 * @file qllm_interpreter.c
 * @brief Load interpreter weights into qLLM and execute programs.
 *
 * Creates a qLLM transformer model programmatically, loads the analytically
 * constructed weight matrices from weight_matrices.c, and executes arbitrary
 * Eshkol programs through qLLM tensor operations (NEON/Metal dispatch).
 *
 * This is the end-to-end integration: Eshkol bytecode → qLLM weights → execution.
 *
 * Three modes:
 *   1. Metal-accelerated: compile against libsemiclassical_qllm with -DUSE_QLLM
 *      cc -O2 -DUSE_QLLM -I$QLLM_ROOT/include -L$QLLM_ROOT/build/lib \
 *         -o qllm_interpreter qllm_interpreter.c \
 *         -lsemiclassical_qllm -lm -framework Metal -framework Foundation
 *
 *   2. Linked but C matmul (default): links qLLM but uses C reference matmul
 *      cc -O2 -I$QLLM_ROOT/include -L$QLLM_ROOT/build/lib \
 *         -o qllm_interpreter qllm_interpreter.c -lsemiclassical_qllm -lm
 *
 *   3. Self-test (no qLLM dependency): cc -DSELF_TEST -O2 -o qllm_interpreter \
 *         qllm_interpreter.c -lm
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#ifdef USE_QLLM
#include <semiclassical_qllm/tensor.h>
#include <semiclassical_qllm/backend.h>
#endif

/* Architecture constants (must match weight_matrices.c) */
#define D 36
#define H 16
#define HD 2
#define N_LAYERS 5
#define FFN_DIM 512
#define MEM_SIZE 4
#define SCALE 100.0f

/* Opcodes -- canonical numbering from eshkol_compiler.c */
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

/* State vector layout (must match weight_matrices.c) */
enum {
    S_PC=0, S_TOS=1, S_SOS=2, S_R2=3, S_R3=4, S_DEPTH=5,
    S_OUTPUT=6, S_HALT=7,
    S_MEM0=8, S_MEM1=9, S_MEM2=10, S_MEM3=11,
    S_SP=12, S_FP=13, S_HAS_OUT=14, S_SPARE=15,
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
    S_TYPE_TOS=32, S_TYPE_SOS=33, S_TYPE_R2=34, S_TYPE_R3=35
};

/* Type tag values */
#define TYPE_NUMBER  0.0f
#define TYPE_BOOL    1.0f
#define TYPE_PAIR    2.0f
#define TYPE_CLOSURE 3.0f
#define TYPE_STRING  4.0f
#define TYPE_VECTOR  5.0f
#define TYPE_NIL     6.0f
#define TYPE_CONT    7.0f

/* Weight file header (QLMW format) */
typedef struct {
    uint32_t magic;     /* 0x514C4D57 = "QLMW" */
    uint32_t version;
    uint32_t d_model;
    uint32_t n_layers;
    uint32_t ffn_dim;
    uint32_t n_heads;
    uint32_t head_dim;
} WeightHeader;

/* Weight arrays (same layout as InterpreterWeights in weight_matrices.c) */
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
} Weights;

static float sigmoidf(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

/*******************************************************************************
 * Weight Loading
 ******************************************************************************/

static Weights* load_weights(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", path); return NULL; }
    WeightHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return NULL; }
    if (hdr.magic != 0x514C4D57) { printf("ERROR: bad magic 0x%08x\n", hdr.magic); fclose(f); return NULL; }
    printf("  Header: d=%d layers=%d ffn=%d heads=%d hd=%d ver=%d\n",
           hdr.d_model, hdr.n_layers, hdr.ffn_dim, hdr.n_heads, hdr.head_dim, hdr.version);
    if (hdr.d_model != D || hdr.n_layers != N_LAYERS || hdr.ffn_dim != FFN_DIM) {
        printf("ERROR: weight dimensions mismatch (expected d=%d l=%d f=%d)\n", D, N_LAYERS, FFN_DIM);
        fclose(f); return NULL;
    }
    Weights* w = (Weights*)calloc(1, sizeof(Weights));
    if (fread(w, sizeof(Weights), 1, f) != 1) { free(w); fclose(f); return NULL; }
    fclose(f);
    return w;
}

/*******************************************************************************
 * Instruction Embedding
 ******************************************************************************/

static void embed_instruction(const Instr* instr, int pos, float out[D]) {
    memset(out, 0, D * sizeof(float));
    out[0] = (float)pos;
    out[1] = -(float)(pos * pos) / 2.0f;
    out[S_OPCODE] = (float)instr->op;
    out[S_OPERAND] = (float)instr->operand;
}

/*******************************************************************************
 * Exec Loop Post-Processing (shared with weight_matrices.c)
 ******************************************************************************/

typedef struct {
    float return_pc;
    float saved_mem[MEM_SIZE];
    float saved_tos, saved_sos, saved_r2, saved_r3, saved_depth;
    float saved_type_tos, saved_type_sos, saved_type_r2, saved_type_r3;
} CallFrame;
static CallFrame g_frames[64];
static int g_frame_count = 0;

#define HEAP_SIZE 4096
static float g_heap[HEAP_SIZE];
static int g_heap_ptr = 0;

/* Exception handler stack */
#define MAX_EXC_HANDLERS 32
static struct {
    float handler_pc, saved_depth;
    float saved_mem[4];
    float saved_tos, saved_sos, saved_r2, saved_r3;
    float saved_type_tos, saved_type_sos, saved_type_r2, saved_type_r3;
} g_exc_handlers[MAX_EXC_HANDLERS];
static int g_exc_count = 0;
static float g_current_exn = 0.0f;
static int g_current_closure_ptr = -1;

/* Dynamic-wind stack */
#define MAX_WINDS 32
static struct {
    float after_thunk_ptr;
    int frame_depth;
} g_wind_stack[MAX_WINDS];
static int g_wind_depth = 0;

static void exec_loop_postprocess(float x[D], const Instr* prog, int n_instr) {
    /* IS_NATIVE: DIV, MOD, etc. */
    if (x[S_IS_NATIVE] > 0.5f) {
        int pc = (int)roundf(x[S_PC]) - 1;
        if (pc >= 0 && pc < n_instr) {
            int opcode = prog[pc].op;
            float tos = x[S_TOS], sos = x[S_SOS];
            float r2 = x[S_R2], r3 = x[S_R3];
            if (opcode == OP_DIV) {
                if (tos == 0) { x[S_HALT] = 1; } /* div-by-zero halt */
                else {
                    x[S_TOS] = sos / tos;
                    x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                    x[S_DEPTH] -= 1;
                    x[S_TYPE_TOS] = TYPE_NUMBER; x[S_TYPE_SOS] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_R3]; x[S_TYPE_R3] = TYPE_NUMBER;
                }
            } else if (opcode == OP_MOD) {
                if (tos == 0) { x[S_HALT] = 1; } /* div-by-zero halt */
                else {
                    /* R7RS floored modulo */
                    float r = fmodf(sos, tos);
                    if (r != 0 && ((r < 0) != (tos < 0))) r += tos;
                    x[S_TOS] = r;
                    x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                    x[S_DEPTH] -= 1;
                    x[S_TYPE_TOS] = TYPE_NUMBER; x[S_TYPE_SOS] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_R3]; x[S_TYPE_R3] = TYPE_NUMBER;
                }
            } else if (opcode == OP_CONS) {
                /* CONS: allocate pair on heap. TOS=cdr, SOS=car -> push heap ptr */
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
                /* CAR: TOS is pair ptr -> replace with car */
                if (x[S_TYPE_TOS] != TYPE_PAIR && tos < 0) { x[S_HALT] = 1; } /* type check */
                else {
                    int ptr = (int)tos;
                    if (ptr >= 0 && ptr + 1 < HEAP_SIZE)
                        x[S_TOS] = g_heap[ptr];      /* car */
                    x[S_TYPE_TOS] = TYPE_NUMBER; /* element type unknown, assume number */
                }
            } else if (opcode == OP_CDR) {
                /* CDR: TOS is pair ptr -> replace with cdr */
                if (x[S_TYPE_TOS] != TYPE_PAIR && tos < 0) { x[S_HALT] = 1; } /* type check */
                else {
                    int ptr = (int)tos;
                    if (ptr >= 0 && ptr + 1 < HEAP_SIZE)
                        x[S_TOS] = g_heap[ptr + 1];  /* cdr */
                    x[S_TYPE_TOS] = TYPE_NUMBER; /* element type unknown, assume number */
                }
            } else if (opcode == OP_NULL_P) {
                /* NULL_P: check both type tag and value sentinel for compatibility */
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_NIL || tos == -1.0f) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            } else if (opcode == OP_SET_CAR) {
                /* SET_CAR: TOS=val, SOS=pair -> mutate car */
                int ptr = (int)sos;
                if (ptr >= 0 && ptr < HEAP_SIZE) g_heap[ptr] = tos;
                x[S_TOS] = r2; x[S_SOS] = r3; x[S_R2] = 0; x[S_R3] = 0;
                x[S_DEPTH] -= 2;
                x[S_TYPE_TOS] = x[S_TYPE_R2]; x[S_TYPE_SOS] = x[S_TYPE_R3]; x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
            } else if (opcode == OP_SET_CDR) {
                /* SET_CDR: TOS=val, SOS=pair -> mutate cdr */
                int ptr = (int)sos;
                if (ptr >= 0 && ptr + 1 < HEAP_SIZE) g_heap[ptr + 1] = tos;
                x[S_TOS] = r2; x[S_SOS] = r3; x[S_R2] = 0; x[S_R3] = 0;
                x[S_DEPTH] -= 2;
                x[S_TYPE_TOS] = x[S_TYPE_R2]; x[S_TYPE_SOS] = x[S_TYPE_R3]; x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
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
            /* -- Closures -- */
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
            /* -- Vectors -- */
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
                /* VEC_REF: TOS=index, SOS=vector_ptr -> push vector[index] */
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
                /* VEC_SET: TOS=value, SOS=index, R2=vector_ptr -> mutate */
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
                /* VEC_LEN: TOS=vector_ptr -> push length */
                int vptr = (int)tos;
                if (vptr >= 0 && vptr < HEAP_SIZE)
                    x[S_TOS] = g_heap[vptr];
                x[S_TYPE_TOS] = TYPE_NUMBER;
            }
            /* -- Strings (simplified: stored as vectors of char codes) -- */
            else if (opcode == OP_STR_REF) {
                /* STR_REF: TOS=index, SOS=string_ptr -> char at index */
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
                /* STR_LEN: TOS=string_ptr -> push length */
                int sptr = (int)tos;
                if (sptr >= 0 && sptr < HEAP_SIZE)
                    x[S_TOS] = g_heap[sptr];
                x[S_TYPE_TOS] = TYPE_NUMBER;
            }
            /* -- Type predicates -- */
            else if (opcode == OP_PAIR_P) {
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_PAIR || (tos >= 0 && (int)tos + 1 < g_heap_ptr)) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            } else if (opcode == OP_NUM_P) {
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_NUMBER) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            } else if (opcode == OP_STR_P) {
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
            /* -- Exceptions -- */
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
            /* -- Continuations -- */
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
            /* -- Variadic / dynamic-wind -- */
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
            /* -- Native call (raise) -- */
            else if (opcode == OP_NATIVE_CALL) {
                /* Check if this is a raise: operand encodes native function ID.
                 * For raise, unwind to exception handler if one exists. */
                int native_id = (pc >= 0 && pc < n_instr) ? prog[pc].operand : -1;
                (void)native_id; /* future: dispatch table for native functions */
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
            /* Save caller's stack state below the func_pc + args. */
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

/*******************************************************************************
 * Forward Pass (C reference matmul — same math as weight_matrices.c)
 * In production, replace with qLLM tensor operations for Metal/NEON.
 ******************************************************************************/

static void matvec(const float* x, const float* W, float* out, int rows, int cols) {
    memset(out, 0, cols * sizeof(float));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out[j] += x[i] * W[i * cols + j];
}

static void forward_pass(const Weights* w, const float state[D],
                          const float pe[][D], int np, float next[D]) {
    float x[D]; memcpy(x, state, sizeof(float)*D);

    for (int L = 0; L < N_LAYERS; L++) {
        /* Attention (layer 0 only) */
        float ao[D]; memset(ao, 0, sizeof(ao));
        if (L == 0 && np > 0) {
            float Q[D]; memset(Q, 0, sizeof(Q));
            for (int i=0;i<D;i++) for(int j=0;j<D;j++) Q[i]+=w->wq[L][i*D+j]*x[j];
            for (int i=0;i<D;i++) Q[i]+=w->bq[L][i];
            float scores[256]; float mx=-1e30f;
            float Va[256][D];
            for (int p=0; p<np&&p<256; p++) {
                float K[D]; memset(K,0,sizeof(K)); memset(Va[p],0,sizeof(Va[p]));
                for(int i=0;i<D;i++) for(int j=0;j<D;j++) {
                    K[i]+=w->wk[L][i*D+j]*pe[p][j]; Va[p][i]+=w->wv[L][i*D+j]*pe[p][j];
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
            matvec(x, w->ff_up[L], h, D, FFN_DIM);
            for(int i=0;i<FFN_DIM;i++) h[i]+=w->ff_up_b[L][i];
            for(int i=0;i<FFN_DIM;i++) h[i]*=h[i];
            matvec(h, w->ff_down[L], fo, FFN_DIM, D);
            for(int i=0;i<D;i++) fo[i]+=w->ff_down_b[L][i];
        } else if (w->ff_type[L]==2) {
            float gate[FFN_DIM], up[FFN_DIM], h[FFN_DIM];
            matvec(x, w->ff_gate[L], gate, D, FFN_DIM);
            for(int i=0;i<FFN_DIM;i++) gate[i]=sigmoidf(gate[i]+w->ff_gate_b[L][i]);
            matvec(x, w->ff_up[L], up, D, FFN_DIM);
            for(int i=0;i<FFN_DIM;i++) up[i]+=w->ff_up_b[L][i];
            for(int i=0;i<FFN_DIM;i++) h[i]=gate[i]*up[i];
            matvec(h, w->ff_down[L], fo, FFN_DIM, D);
            for(int i=0;i<D;i++) fo[i]+=w->ff_down_b[L][i];
        }
        for(int i=0;i<D;i++) x[i]+=fo[i];
    }
    memcpy(next, x, sizeof(float)*D);
}

#ifdef USE_QLLM
/* ── qLLM-accelerated forward pass (Metal/NEON dispatch) ── */
static qllm_tensor_t* make_t2d(const float* data, size_t r, size_t c, qllm_device_t dev) {
    qllm_tensor_options_t opts = qllm_tensor_options_default(dev);
    opts.dtype = QLLM_DTYPE_FLOAT32;
    size_t shape[] = {r, c};
    qllm_tensor_t* t = qllm_tensor_create(2, shape, &opts);
    if (t && data) memcpy(qllm_tensor_get_data(t), data, r*c*sizeof(float));
    return t;
}

static void forward_pass_qllm(const Weights* w, const float state[D],
                                const float pe[][D], int np, float next[D],
                                qllm_device_t dev) {
    float x[D]; memcpy(x, state, sizeof(float)*D);

    for (int L = 0; L < N_LAYERS; L++) {
        /* Attention (layer 0 only — same as C version, attention is small) */
        float ao[D]; memset(ao, 0, sizeof(ao));
        if (L == 0 && np > 0) {
            float Q[D]; memset(Q, 0, sizeof(Q));
            for (int i=0;i<D;i++) for(int j=0;j<D;j++) Q[i]+=w->wq[L][i*D+j]*x[j];
            for (int i=0;i<D;i++) Q[i]+=w->bq[L][i];
            float scores[256]; float mx=-1e30f;
            float Va[256][D];
            for (int p=0;p<np&&p<256;p++) {
                float K[D]; memset(K,0,sizeof(K)); memset(Va[p],0,sizeof(Va[p]));
                for(int i=0;i<D;i++) for(int j=0;j<D;j++) {
                    K[i]+=w->wk[L][i*D+j]*pe[p][j]; Va[p][i]+=w->wv[L][i*D+j]*pe[p][j];
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

        /* FFN via qLLM tensor matmul (dispatches to Metal/NEON) */
        float fo[D]; memset(fo,0,sizeof(fo));
        if (w->ff_type[L]==1) {
            /* SQUARE activation: up → square → down */
            qllm_tensor_t* tx = make_t2d(x, 1, D, dev);
            qllm_tensor_t* tw = make_t2d(w->ff_up[L], D, FFN_DIM, dev);
            qllm_tensor_t* th = qllm_tensor_matmul(tx, tw);
            qllm_tensor_destroy(tx); qllm_tensor_destroy(tw);
            if (th) {
                float* h = (float*)qllm_tensor_get_data(th);
                for(int i=0;i<FFN_DIM;i++) { h[i]+=w->ff_up_b[L][i]; h[i]*=h[i]; }
                qllm_tensor_t* tdn = make_t2d(w->ff_down[L], FFN_DIM, D, dev);
                qllm_tensor_t* tfo = qllm_tensor_matmul(th, tdn);
                qllm_tensor_destroy(th); qllm_tensor_destroy(tdn);
                if (tfo) {
                    float* r = (float*)qllm_tensor_get_data(tfo);
                    for(int i=0;i<D;i++) fo[i]=r[i]+w->ff_down_b[L][i];
                    qllm_tensor_destroy(tfo);
                }
            }
        } else if (w->ff_type[L]==2) {
            /* Gated FFN: gate*up → down */
            qllm_tensor_t* tx = make_t2d(x, 1, D, dev);
            qllm_tensor_t* twg = make_t2d(w->ff_gate[L], D, FFN_DIM, dev);
            qllm_tensor_t* twu = make_t2d(w->ff_up[L], D, FFN_DIM, dev);
            qllm_tensor_t* tg = qllm_tensor_matmul(tx, twg);
            qllm_tensor_t* tu = qllm_tensor_matmul(tx, twu);
            qllm_tensor_destroy(tx); qllm_tensor_destroy(twg); qllm_tensor_destroy(twu);
            if (tg && tu) {
                float* gate = (float*)qllm_tensor_get_data(tg);
                float* up = (float*)qllm_tensor_get_data(tu);
                for(int i=0;i<FFN_DIM;i++) {
                    gate[i]=sigmoidf(gate[i]+w->ff_gate_b[L][i]);
                    up[i]+=w->ff_up_b[L][i];
                }
                qllm_tensor_t* th = qllm_tensor_mul(tg, tu);
                qllm_tensor_destroy(tg); qllm_tensor_destroy(tu);
                if (th) {
                    qllm_tensor_t* tdn = make_t2d(w->ff_down[L], FFN_DIM, D, dev);
                    qllm_tensor_t* tfo = qllm_tensor_matmul(th, tdn);
                    qllm_tensor_destroy(th); qllm_tensor_destroy(tdn);
                    if (tfo) {
                        float* r = (float*)qllm_tensor_get_data(tfo);
                        for(int i=0;i<D;i++) fo[i]=r[i]+w->ff_down_b[L][i];
                        qllm_tensor_destroy(tfo);
                    }
                }
            } else {
                if(tg) qllm_tensor_destroy(tg);
                if(tu) qllm_tensor_destroy(tu);
            }
        }
        for(int i=0;i<D;i++) x[i]+=fo[i];
    }
    memcpy(next, x, sizeof(float)*D);
}
#endif /* USE_QLLM */

static int run_program(const Weights* w, const Instr* prog, int n_instr,
                        float* outputs, int max_out) {
    float pe[256][D];
    for (int p = 0; p < n_instr && p < 256; p++)
        embed_instruction(&prog[p], p, pe[p]);
    float state[D]; memset(state, 0, sizeof(state)); state[S_OUTPUT] = -1;
    g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    int n_out = 0;
    for (int step = 0; step < 100000; step++) {
        float next[D];
        forward_pass(w, state, pe, n_instr, next);
        exec_loop_postprocess(next, prog, n_instr);
        if (next[S_HAS_OUT] > 0.5f && n_out < max_out)
            outputs[n_out++] = next[S_OUTPUT];
        if (next[S_HALT] > 0.5f) break;
        memcpy(state, next, sizeof(state));
    }
    return n_out;
}

#ifdef USE_QLLM
static int run_program_qllm(const Weights* w, const Instr* prog, int n_instr,
                              float* outputs, int max_out, qllm_device_t dev) {
    float pe[256][D];
    for (int p = 0; p < n_instr && p < 256; p++)
        embed_instruction(&prog[p], p, pe[p]);
    float state[D]; memset(state, 0, sizeof(state)); state[S_OUTPUT] = -1;
    g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    int n_out = 0;
    for (int step = 0; step < 100000; step++) {
        float next[D];
        forward_pass_qllm(w, state, pe, n_instr, next, dev);
        exec_loop_postprocess(next, prog, n_instr);
        if (next[S_HAS_OUT] > 0.5f && n_out < max_out)
            outputs[n_out++] = next[S_OUTPUT];
        if (next[S_HALT] > 0.5f) break;
        memcpy(state, next, sizeof(state));
    }
    return n_out;
}
#endif

/*******************************************************************************
 * ESKB Bytecode Loader
 ******************************************************************************/

static Instr* load_eskb(const char* path, int* n_instr_out) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic, n_instr, n_const;
    fread(&magic, 4, 1, f); fread(&n_instr, 4, 1, f); fread(&n_const, 4, 1, f);
    if (magic != 0x45534B42 || n_instr > 8192) { fclose(f); return NULL; }
    Instr* prog = (Instr*)calloc(n_instr, sizeof(Instr));
    for (uint32_t i = 0; i < n_instr; i++) {
        uint8_t op; int32_t operand;
        fread(&op, 1, 1, f); fread(&operand, 4, 1, f);
        prog[i].op = (OpCode)op; prog[i].operand = operand;
    }
    float* constants = NULL;
    if (n_const > 0) {
        constants = (float*)calloc(n_const, sizeof(float));
        for (uint32_t i = 0; i < n_const; i++) {
            uint8_t type; double val;
            fread(&type, 1, 1, f); fread(&val, 8, 1, f);
            constants[i] = (float)val;
        }
        for (uint32_t i = 0; i < n_instr; i++) {
            if (prog[i].op == OP_CONST && prog[i].operand < (int)n_const)
                prog[i].operand = (int)constants[prog[i].operand];
        }
        free(constants);
    }
    fclose(f);
    *n_instr_out = (int)n_instr;
    return prog;
}

/*******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv) {
    const char* weight_path = "/tmp/interpreter_weights.bin";
    const char* bc_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strstr(argv[i], ".bin")) weight_path = argv[i];
        else if (strstr(argv[i], ".bc")) bc_path = argv[i];
    }

    printf("=== qLLM Interpreter ===\n\n");
    printf("  Loading weights: %s\n", weight_path);

    Weights* w = load_weights(weight_path);
    if (!w) return 1;

    int pass = 0, fail = 0;

    #define QTEST(name, prog, n, expected) do { \
        float out[1]; \
        int nout = run_program(w, prog, n, out, 1); \
        float v = nout > 0 ? out[0] : -9999.0f; \
        int ok = (nout > 0 && fabsf(v - (expected)) < 0.1f); \
        printf("  %-25s = %7.1f  %s\n", name, v, ok?"PASS":"FAIL"); \
        if (ok) pass++; else fail++; \
    } while(0)

    printf("\n  Built-in tests:\n\n");

    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_ADD,0},{OP_PRINT,0},{OP_HALT,0}};
      QTEST("3+5", p, 5, 8); }
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_ADD,0},{OP_CONST,2},{OP_MUL,0},{OP_PRINT,0},{OP_HALT,0}};
      QTEST("(3+5)*2", p, 7, 16); }
    { Instr p[]={{OP_CONST,10},{OP_CONST,7},{OP_SUB,0},{OP_PRINT,0},{OP_HALT,0}};
      QTEST("10-7", p, 5, 3); }
    { Instr p[]={{OP_CONST,42},{OP_SET_LOCAL,0},{OP_GET_LOCAL,0},{OP_PRINT,0},{OP_HALT,0}};
      QTEST("mem[0]=42", p, 5, 42); }
    { Instr p[]={{OP_CONST,7},{OP_CONST,11},{OP_MUL,0},{OP_PRINT,0},{OP_HALT,0}};
      QTEST("7*11", p, 5, 77); }
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_LT,0},{OP_PRINT,0},{OP_HALT,0}};
      QTEST("3<5", p, 5, 1); }
    { Instr p[]={{OP_CONST,10},{OP_CONST,2},{OP_DIV,0},{OP_PRINT,0},{OP_HALT,0}};
      QTEST("10/2", p, 5, 5); }
    /* Recursive factorial */
    { Instr p[]={
        {OP_CONST,5},{OP_CONST,5},{OP_CALL,1},{OP_PRINT,0},{OP_HALT,0},
        {OP_GET_LOCAL,0},{OP_CONST,0},{OP_EQ,0},{OP_JUMP_IF_FALSE,11},
        {OP_CONST,1},{OP_RETURN,0},
        {OP_GET_LOCAL,0},{OP_GET_LOCAL,0},{OP_CONST,1},{OP_SUB,0},{OP_CONST,5},{OP_CALL,1},
        {OP_MUL,0},{OP_RETURN,0}};
      QTEST("rec fact(5)", p, 19, 120); }
    /* Cons pair */
    { Instr p[]={{OP_CONST,3},{OP_CONST,4},{OP_CONS,0},{OP_CAR,0},{OP_PRINT,0},{OP_HALT,0}};
      QTEST("car(cons 3 4)", p, 6, 3); }

    #undef QTEST

    /* Load external bytecode if provided */
    if (bc_path) {
        int n_instr;
        Instr* prog = load_eskb(bc_path, &n_instr);
        if (prog) {
            printf("\n  External bytecode: %s (%d instructions)\n", bc_path, n_instr);
            float outputs[64];
            int n_out = run_program(w, prog, n_instr, outputs, 64);
            printf("  Outputs:");
            for (int i = 0; i < n_out; i++) printf(" %.4g", outputs[i]);
            printf("\n");
            free(prog);
        } else {
            printf("  ERROR: cannot load %s\n", bc_path);
        }
    }

    printf("\n=== Results: %d passed, %d failed ===\n", pass, fail);

#ifdef USE_QLLM
    /* ── Metal/NEON Benchmark ── */
    {
        qllm_device_t dev = qllm_metal_is_available() ? QLLM_DEVICE_METAL : QLLM_DEVICE_CPU;
        printf("\n  qLLM accelerated benchmark (device=%s):\n\n",
               dev == QLLM_DEVICE_METAL ? "METAL" : "CPU/NEON");

        /* fib(15) = 610 */
        Instr fib_prog[]={
            {OP_CONST,15},{OP_CONST,5},{OP_CALL,1},{OP_PRINT,0},{OP_HALT,0},
            {OP_GET_LOCAL,0},{OP_CONST,1},{OP_LE,0},{OP_JUMP_IF_FALSE,11},
            {OP_GET_LOCAL,0},{OP_RETURN,0},
            {OP_GET_LOCAL,0},{OP_CONST,1},{OP_SUB,0},{OP_CONST,5},{OP_CALL,1},
            {OP_GET_LOCAL,0},{OP_CONST,2},{OP_SUB,0},{OP_CONST,5},{OP_CALL,1},
            {OP_ADD,0},{OP_RETURN,0},
        };

        /* C reference timing — use CPU device for qLLM to avoid Metal sync issues */
        struct timespec t0, t1;
        float out_c[1], out_q[1];

        clock_gettime(CLOCK_MONOTONIC, &t0);
        int nc = run_program(w, fib_prog, 23, out_c, 1);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (nc == 0) printf("  WARNING: C ref produced 0 outputs\n");
        double c_ms = (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)/1e6;

        /* qLLM NEON timing (CPU device — avoids GPU sync overhead for small tensors) */
        clock_gettime(CLOCK_MONOTONIC, &t0);
        run_program_qllm(w, fib_prog, 23, out_q, 1, QLLM_DEVICE_CPU);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double q_ms = (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)/1e6;

        printf("  fib(15):  C ref = %.0f (%.1f ms)  |  qLLM/NEON = %.0f (%.1f ms)  |  speedup: %.2fx\n",
               out_c[0], c_ms, out_q[0], q_ms, c_ms / q_ms);

        /* Correctness check */
        if (fabsf(out_c[0] - out_q[0]) < 0.1f)
            printf("  Correctness: MATCH\n");
        else
            printf("  Correctness: MISMATCH (c=%.1f q=%.1f)\n", out_c[0], out_q[0]);
    }
#endif

    printf("\n=== Results: %d passed, %d failed ===\n", pass, fail);
    free(w);
    return fail > 0 ? 1 : 0;
}
