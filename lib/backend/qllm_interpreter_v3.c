/**
 * @file qllm_interpreter_v3.c
 * @brief Load v3 interpreter weights into qLLM and execute programs.
 *
 * Creates a qLLM transformer model programmatically, loads the analytically
 * constructed weight matrices from weight_matrices_v3.c, and executes arbitrary
 * Eshkol programs through qLLM tensor operations (NEON/Metal dispatch).
 *
 * This is the end-to-end integration: Eshkol bytecode → qLLM weights → execution.
 *
 * Two modes:
 *   1. Standalone test: compile against libsemiclassical_qllm
 *      cc -O2 -I../../include -L../../build/lib -o qllm_interpreter_v3 \
 *         qllm_interpreter_v3.c -lsemiclassical_qllm -lm
 *
 *   2. Self-test (no qLLM dependency): cc -DSELF_TEST -O2 -o qllm_interpreter_v3 \
 *         qllm_interpreter_v3.c -lm
 *      Uses the same C matmul as weight_matrices_v3.c for verification.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* Architecture constants (must match weight_matrices_v3.c) */
#define D 32
#define H 16
#define HD 2
#define N_LAYERS 5
#define FFN_DIM 512
#define MEM_SIZE 4
#define SCALE 100.0f

/* Opcodes (canonical, matches eshkol_compiler.c) */
typedef enum {
    OP_NOP=0, OP_CONST=1, OP_NIL=2, OP_TRUE=3, OP_FALSE=4, OP_POP=5, OP_DUP=6,
    OP_ADD=7, OP_SUB=8, OP_MUL=9, OP_DIV=10, OP_MOD=11, OP_NEG=12, OP_ABS=13,
    OP_EQ=14, OP_LT=15, OP_GT=16, OP_LE=17, OP_GE=18, OP_NOT=19,
    OP_GET_LOCAL=20, OP_SET_LOCAL=21, OP_GET_UPVALUE=22, OP_SET_UPVALUE=23,
    OP_CLOSURE=24, OP_CALL=25, OP_TAIL_CALL=26, OP_RETURN=27,
    OP_JUMP=28, OP_JUMP_IF_FALSE=29, OP_LOOP=30,
    OP_CONS=31, OP_CAR=32, OP_CDR=33, OP_NULL_P=34,
    OP_PRINT=35, OP_HALT=36, OP_NATIVE_CALL=37,
    OP_COUNT=63
} OpCode;

typedef struct { OpCode op; int operand; } Instr;

/* State vector layout (must match weight_matrices_v3.c) */
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
    S_ABS_DELTA=31
};

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

/* Weight arrays (same layout as InterpreterWeights in weight_matrices_v3.c) */
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
 * Exec Loop Post-Processing (shared with weight_matrices_v3.c)
 ******************************************************************************/

typedef struct {
    float return_pc;
    float saved_mem[MEM_SIZE];
    float saved_tos, saved_sos, saved_r2, saved_r3, saved_depth;
} CallFrame;
static CallFrame g_frames[64];
static int g_frame_count = 0;

#define HEAP_SIZE 4096
static float g_heap[HEAP_SIZE];
static int g_heap_ptr = 0;

static void exec_loop_postprocess(float x[D], const Instr* prog, int n_instr) {
    if (x[S_IS_NATIVE] > 0.5f) {
        int pc = (int)roundf(x[S_PC]) - 1;
        if (pc >= 0 && pc < n_instr) {
            int opcode = prog[pc].op;
            float tos = x[S_TOS], sos = x[S_SOS], r2 = x[S_R2], r3 = x[S_R3];
            if (opcode == OP_DIV) {
                x[S_TOS] = (tos != 0) ? sos / tos : 0;
                x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0; x[S_DEPTH] -= 1;
            } else if (opcode == OP_MOD) {
                x[S_TOS] = (tos != 0) ? fmodf(sos, tos) : 0;
                x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0; x[S_DEPTH] -= 1;
            } else if (opcode == OP_CONS) {
                if (g_heap_ptr + 2 <= HEAP_SIZE) {
                    int ptr = g_heap_ptr;
                    g_heap[g_heap_ptr++] = sos;
                    g_heap[g_heap_ptr++] = tos;
                    x[S_TOS] = (float)ptr;
                    x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0; x[S_DEPTH] -= 1;
                }
            } else if (opcode == OP_CAR) {
                int ptr = (int)tos;
                if (ptr >= 0 && ptr + 1 < HEAP_SIZE) x[S_TOS] = g_heap[ptr];
            } else if (opcode == OP_CDR) {
                int ptr = (int)tos;
                if (ptr >= 0 && ptr + 1 < HEAP_SIZE) x[S_TOS] = g_heap[ptr + 1];
            } else if (opcode == OP_NULL_P) {
                x[S_TOS] = (tos == -1.0f) ? 1.0f : 0.0f;
            }
        }
        x[S_IS_NATIVE] = 0;
    }
    if (x[S_IS_CALL] > 0.5f) {
        int pc = (int)roundf(x[S_PC]) - 1;
        int argc = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
        float func_pc = x[S_TOS];
        int fptr = (int)func_pc;
        if (fptr >= 0 && fptr + 1 < g_heap_ptr) {
            float candidate = g_heap[fptr];
            if (candidate >= 0 && candidate < n_instr) func_pc = candidate;
        }
        if (g_frame_count < 64) {
            CallFrame* f = &g_frames[g_frame_count];
            f->return_pc = x[S_PC];
            for (int i = 0; i < MEM_SIZE; i++) f->saved_mem[i] = x[S_MEM0+i];
            if (argc == 0) { f->saved_tos = x[S_SOS]; f->saved_sos = x[S_R2]; f->saved_r2 = x[S_R3]; f->saved_r3 = 0; }
            else if (argc == 1) { f->saved_tos = x[S_R2]; f->saved_sos = x[S_R3]; f->saved_r2 = 0; f->saved_r3 = 0; }
            else { f->saved_tos = x[S_R3]; f->saved_sos = 0; f->saved_r2 = 0; f->saved_r3 = 0; }
            f->saved_depth = x[S_DEPTH] - (1 + argc);
            g_frame_count++;
        }
        float args[4] = {x[S_SOS], x[S_R2], x[S_R3], 0};
        for (int i = 0; i < MEM_SIZE; i++) x[S_MEM0+i] = 0;
        for (int i = 0; i < argc && i < MEM_SIZE; i++) x[S_MEM0+i] = args[i];
        x[S_PC] = func_pc;
        x[S_TOS] = 0; x[S_SOS] = 0; x[S_R2] = 0; x[S_R3] = 0; x[S_DEPTH] = 0;
        x[S_IS_CALL] = 0;
    }
    if (x[S_IS_RET] > 0.5f) {
        float retval = x[S_TOS];
        if (g_frame_count > 0) {
            g_frame_count--;
            CallFrame* f = &g_frames[g_frame_count];
            x[S_PC] = f->return_pc;
            for (int i = 0; i < MEM_SIZE; i++) x[S_MEM0+i] = f->saved_mem[i];
            x[S_TOS] = retval; x[S_SOS] = f->saved_tos;
            x[S_R2] = f->saved_sos; x[S_R3] = f->saved_r2;
            x[S_DEPTH] = f->saved_depth + 1;
        }
        x[S_IS_RET] = 0;
    }
}

/*******************************************************************************
 * Forward Pass (C reference matmul — same math as weight_matrices_v3.c)
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

static int run_program(const Weights* w, const Instr* prog, int n_instr,
                        float* outputs, int max_out) {
    float pe[256][D];
    for (int p = 0; p < n_instr && p < 256; p++)
        embed_instruction(&prog[p], p, pe[p]);
    float state[D]; memset(state, 0, sizeof(state)); state[S_OUTPUT] = -1;
    g_frame_count = 0; g_heap_ptr = 0;
    int n_out = 0;
    for (int step = 0; step < 8192; step++) {
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
    const char* weight_path = "/tmp/interpreter_weights_v3.bin";
    const char* bc_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strstr(argv[i], ".bin")) weight_path = argv[i];
        else if (strstr(argv[i], ".bc")) bc_path = argv[i];
    }

    printf("=== qLLM Interpreter v3 ===\n\n");
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
    free(w);
    return fail > 0 ? 1 : 0;
}
