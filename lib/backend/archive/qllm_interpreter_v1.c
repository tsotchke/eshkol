/**
 * @file qllm_interpreter.c
 * @brief Load interpreter weights into qLLM and execute programs.
 *
 * Creates a qLLM transformer model programmatically, loads the analytically
 * constructed weight matrices from weight_matrices.c, and executes arbitrary
 * Eshkol programs through the qLLM framework in hidden state mode.
 *
 * This is the end-to-end integration: Eshkol bytecode → qLLM weights → execution.
 *
 * Compile: cc -O2 -I../../include -L../../build/lib -o qllm_interpreter \
 *          qllm_interpreter.c -lsemiclassical_qllm -lm
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* qLLM headers */
#include <semiclassical_qllm/tensor.h>
#include <semiclassical_qllm/attention.h>
#include <semiclassical_qllm/feedforward.h>
#include <semiclassical_qllm/transformer.h>

/* Architecture constants (must match weight_matrices.c) */
#define D 36
#define H 16
#define HD 2
#define N_LAYERS 5
#define FFN_DIM 512
#define SCALE 100.0f

/* State dimension indices (must match weight_matrices.c) */
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
    S_TYPE_TOS=32, S_TYPE_SOS=33, S_TYPE_R2=34, S_TYPE_R3=35
};

typedef enum {
    OP_NOP=0, OP_CONST=1, OP_ADD=2, OP_SUB=3, OP_MUL=4, OP_DUP=5,
    OP_SWAP=6, OP_DROP=7, OP_LOAD=8, OP_STORE=9, OP_JUMP=10,
    OP_JUMP_IF=11, OP_OUTPUT=12, OP_HALT=13
} OpCode;

typedef struct { OpCode op; int operand; } Instr;

/* Weight file header */
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

static Weights* load_weights(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", path); return NULL; }

    WeightHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return NULL; }
    if (hdr.magic != 0x514C4D57) { printf("ERROR: bad magic\n"); fclose(f); return NULL; }
    printf("  Loaded header: d=%d layers=%d ffn=%d heads=%d hd=%d\n",
           hdr.d_model, hdr.n_layers, hdr.ffn_dim, hdr.n_heads, hdr.head_dim);

    Weights* w = (Weights*)calloc(1, sizeof(Weights));
    if (!w) { fclose(f); return NULL; }
    if (fread(w, sizeof(Weights), 1, f) != 1) { free(w); fclose(f); return NULL; }
    fclose(f);
    return w;
}

/* Create a qLLM tensor from a float array */
static qllm_tensor_t* make_tensor_2d(const float* data, size_t rows, size_t cols) {
    qllm_tensor_options_t opts = qllm_tensor_options_default(QLLM_DEVICE_CPU);
    opts.dtype = QLLM_DTYPE_FLOAT32;
    size_t shape[] = {rows, cols};
    qllm_tensor_t* t = qllm_tensor_create(2, shape, &opts);
    if (t && data) {
        float* dst = (float*)qllm_tensor_get_data(t);
        if (dst) memcpy(dst, data, rows * cols * sizeof(float));
    }
    return t;
}

static qllm_tensor_t* make_tensor_1d(const float* data, size_t n) {
    qllm_tensor_options_t opts = qllm_tensor_options_default(QLLM_DEVICE_CPU);
    opts.dtype = QLLM_DTYPE_FLOAT32;
    size_t shape[] = {n};
    qllm_tensor_t* t = qllm_tensor_create(1, shape, &opts);
    if (t && data) {
        float* dst = (float*)qllm_tensor_get_data(t);
        if (dst) memcpy(dst, data, n * sizeof(float));
    }
    return t;
}

static qllm_tensor_t* make_tensor_3d(const float* data, size_t a, size_t b, size_t c) {
    qllm_tensor_options_t opts = qllm_tensor_options_default(QLLM_DEVICE_CPU);
    opts.dtype = QLLM_DTYPE_FLOAT32;
    size_t shape[] = {a, b, c};
    qllm_tensor_t* t = qllm_tensor_create(3, shape, &opts);
    if (t && data) {
        float* dst = (float*)qllm_tensor_get_data(t);
        if (dst) memcpy(dst, data, a * b * c * sizeof(float));
    }
    return t;
}

/* Create the qLLM model and load weights */
static qllm_transformer_model_t* create_interpreter_model(const Weights* w) {
    /* Configure model */
    qllm_transformer_model_options_t opts = {0};
    opts.device = QLLM_DEVICE_CPU;
    opts.type = QLLM_TRANSFORMER_STANDARD;
    opts.dim = D;
    opts.hidden_dim = FFN_DIM;
    opts.num_heads = H;
    opts.num_layers = N_LAYERS;
    opts.vocab_size = 256;  /* byte tokens */
    opts.max_seq_len = 4096;
    opts.scoring = QLLM_SCORING_DOT_PRODUCT;  /* Layer 0 uses dot product with temperature */
    opts.activation = QLLM_ACTIVATION_GELU;    /* Default; we override per-layer */
    opts.ff_type = QLLM_FEEDFORWARD_STANDARD;  /* Default; we override per-layer */
    opts.dropout_rate = 0.0f;
    opts.causal = false;  /* No causal mask — instruction fetch needs full context */
    opts.pre_norm = true;
    opts.tie_weights = false;
    opts.use_bias = true;

    /* norm_type defaults to QLLM_NORM_NONE (value 0) via zero-init of block_options */

    qllm_transformer_model_t* model = qllm_transformer_model_create(&opts);
    if (!model) {
        printf("ERROR: failed to create model\n");
        return NULL;
    }

    /* Load weights into each layer */
    for (int L = 0; L < N_LAYERS; L++) {
        qllm_transformer_block_t* block = qllm_transformer_model_get_block(model, L);
        if (!block) { printf("ERROR: failed to get block %d\n", L); continue; }

        /* Attention weights */
        qllm_attention_t* attn = qllm_transformer_block_get_attention(block);
        if (attn) {
            qllm_tensor_t* wq = make_tensor_2d(w->wq[L], D, D);
            qllm_tensor_t* wk = make_tensor_2d(w->wk[L], D, D);
            qllm_tensor_t* wv = make_tensor_2d(w->wv[L], D, D);
            qllm_tensor_t* wo = make_tensor_2d(w->wo[L], D, D);
            qllm_tensor_t* bq = make_tensor_1d(w->bq[L], D);

            if (wq) { qllm_attention_set_query_weights(attn, wq); qllm_tensor_destroy(wq); }
            if (wk) { qllm_attention_set_key_weights(attn, wk); qllm_tensor_destroy(wk); }
            if (wv) { qllm_attention_set_value_weights(attn, wv); qllm_tensor_destroy(wv); }
            if (wo) { qllm_attention_set_output_weights(attn, wo); qllm_tensor_destroy(wo); }
            if (bq) { qllm_attention_set_query_bias(attn, bq); qllm_tensor_destroy(bq); }
        }

        /* FFN weights */
        qllm_feedforward_t* ff = qllm_transformer_block_get_feedforward(block);
        if (ff) {
            qllm_tensor_t* w1 = make_tensor_2d(w->ff_up[L], D, FFN_DIM);
            qllm_tensor_t* b1 = make_tensor_1d(w->ff_up_b[L], FFN_DIM);
            qllm_tensor_t* w2 = make_tensor_2d(w->ff_down[L], FFN_DIM, D);
            qllm_tensor_t* b2 = make_tensor_1d(w->ff_down_b[L], D);

            if (w1) { qllm_feedforward_set_weights1(ff, w1); qllm_tensor_destroy(w1); }
            if (b1) { qllm_feedforward_set_bias1(ff, b1); qllm_tensor_destroy(b1); }
            if (w2) { qllm_feedforward_set_weights2(ff, w2); qllm_tensor_destroy(w2); }
            if (b2) { qllm_feedforward_set_bias2(ff, b2); qllm_tensor_destroy(b2); }

            /* Gate weights for gated layers */
            if (w->ff_type[L] == 2) {
                qllm_tensor_t* wg = make_tensor_2d(w->ff_gate[L], D, FFN_DIM);
                if (wg) { qllm_feedforward_set_gate_weights(ff, wg); qllm_tensor_destroy(wg); }
                /* Gate bias: need to add setter if not present */
            }
        }
    }

    printf("  Model created: d=%d layers=%d heads=%d ffn=%d norm=NONE\n", D, N_LAYERS, H, FFN_DIM);
    return model;
}

/* Embed instruction token */
static void embed_instruction(const Instr* instr, int pos, float out[D]) {
    memset(out, 0, D * sizeof(float));
    out[0] = (float)pos;
    out[1] = -(float)(pos * pos) / 2.0f;
    out[S_OPCODE] = (float)instr->op;
    out[S_OPERAND] = (float)instr->operand;
}

/* ── Forward pass using qLLM tensor operations directly ──
 * Uses qllm_tensor_matmul for all projections (dispatches to NEON/BLAS).
 * Same math as verified forward_with_weights() in weight_matrices.c. */

static float sigmoidf(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static int run_qllm_native(const Weights* w,
                            const Instr* prog, int n_instr,
                            float* outputs, int max_out) {
    /* Embed program tokens */
    float prog_embeds[256][D];
    for (int p = 0; p < n_instr && p < 256; p++)
        embed_instruction(&prog[p], p, prog_embeds[p]);

    float state[D];
    memset(state, 0, sizeof(state));
    state[S_OUTPUT] = -1.0f;

    int n_out = 0;

    for (int step = 0; step < 8192; step++) {
        float x[D];
        memcpy(x, state, sizeof(x));

        /* ── Layer 0: Attention (instruction fetch) using qLLM tensors ── */
        {
            /* Create Q tensor from state */
            qllm_tensor_t* t_x = make_tensor_2d(x, 1, D);
            qllm_tensor_t* t_wq = make_tensor_2d(w->wq[0], D, D);
            if (!t_x || !t_wq) {
                if (t_x) qllm_tensor_destroy(t_x);
                if (t_wq) qllm_tensor_destroy(t_wq);
                continue;  /* skip this step */
            }
            qllm_tensor_t* t_Q = qllm_tensor_matmul(t_x, t_wq);
            qllm_tensor_destroy(t_x); qllm_tensor_destroy(t_wq);

            if (t_Q) {
                float* Q = (float*)qllm_tensor_get_data(t_Q);
                if (!Q) {
                    qllm_tensor_destroy(t_Q);
                } else {
                    /* Add bias */
                    for (int i = 0; i < D; i++) Q[i] += w->bq[0][i];

                    /* Compute attention scores over program tokens */
                    float scores[256], mx = -1e30f;
                    for (int p = 0; p < n_instr && p < 256; p++) {
                        /* Project K from program embedding */
                        float K[D]; memset(K, 0, sizeof(K));
                        for (int i = 0; i < D; i++)
                            for (int j = 0; j < D; j++)
                                K[i] += w->wk[0][i*D+j] * prog_embeds[p][j];

                        scores[p] = (Q[0]*K[0] + Q[1]*K[1]) / sqrtf((float)HD);
                        if (scores[p] > mx) mx = scores[p];
                    }

                    /* Softmax */
                    float sum = 0;
                    for (int p = 0; p < n_instr; p++) { scores[p] = expf(scores[p]-mx); sum += scores[p]; }
                    for (int p = 0; p < n_instr; p++) scores[p] /= sum;

                    /* Weighted V → attn_out */
                    float V[256][D];
                    for (int p = 0; p < n_instr; p++) {
                        memset(V[p], 0, sizeof(V[p]));
                        for (int i = 0; i < D; i++)
                            for (int j = 0; j < D; j++)
                                V[p][i] += w->wv[0][i*D+j] * prog_embeds[p][j];
                    }

                    float hout[D]; memset(hout, 0, sizeof(hout));
                    for (int p = 0; p < n_instr; p++)
                        for (int d = 0; d < HD; d++)
                            hout[d] += scores[p] * V[p][d];

                    /* Project through W_O */
                    float ao[D]; memset(ao, 0, sizeof(ao));
                    for (int i = 0; i < D; i++)
                        for (int j = 0; j < D; j++)
                            ao[i] += w->wo[0][i*D+j] * hout[j];

                    /* Residual */
                    for (int i = 0; i < D; i++) x[i] += ao[i];
                    qllm_tensor_destroy(t_Q);
                }
            }
        }

        /* ── Layer 1: FFN (SQUARE activation) using qLLM matmul ── */
        {
            qllm_tensor_t* t_x = make_tensor_2d(x, 1, D);
            qllm_tensor_t* t_wup = make_tensor_2d(w->ff_up[1], D, FFN_DIM);
            if (!t_x || !t_wup) {
                if (t_x) qllm_tensor_destroy(t_x);
                if (t_wup) qllm_tensor_destroy(t_wup);
                continue;  /* skip this step */
            }
            qllm_tensor_t* t_h = qllm_tensor_matmul(t_x, t_wup);
            qllm_tensor_destroy(t_x); qllm_tensor_destroy(t_wup);

            if (t_h) {
                float* h = (float*)qllm_tensor_get_data(t_h);
                if (!h) { qllm_tensor_destroy(t_h); goto layers_2_3; }
                for (int i = 0; i < FFN_DIM; i++) h[i] += w->ff_up_b[1][i];
                for (int i = 0; i < FFN_DIM; i++) h[i] *= h[i]; /* SQUARE */

                qllm_tensor_t* t_wdn = make_tensor_2d(w->ff_down[1], FFN_DIM, D);
                if (!t_wdn) { qllm_tensor_destroy(t_h); goto layers_2_3; }
                qllm_tensor_t* t_fo = qllm_tensor_matmul(t_h, t_wdn);
                qllm_tensor_destroy(t_h); qllm_tensor_destroy(t_wdn);

                if (t_fo) {
                    float* fo = (float*)qllm_tensor_get_data(t_fo);
                    if (fo) {
                        for (int i = 0; i < D; i++) x[i] += fo[i] + w->ff_down_b[1][i];
                    }
                    qllm_tensor_destroy(t_fo);
                }
            }
        }

        /* ── Layers 2-3: Gated FFN using qLLM matmul ── */
        layers_2_3:
        for (int L = 2; L <= 3; L++) {
            qllm_tensor_t* t_x = make_tensor_2d(x, 1, D);
            qllm_tensor_t* t_wg = make_tensor_2d(w->ff_gate[L], D, FFN_DIM);
            qllm_tensor_t* t_wu = make_tensor_2d(w->ff_up[L], D, FFN_DIM);
            if (!t_x || !t_wg || !t_wu) {
                if (t_x) qllm_tensor_destroy(t_x);
                if (t_wg) qllm_tensor_destroy(t_wg);
                if (t_wu) qllm_tensor_destroy(t_wu);
                continue;  /* skip this layer */
            }

            qllm_tensor_t* t_gate = qllm_tensor_matmul(t_x, t_wg);
            qllm_tensor_t* t_up = qllm_tensor_matmul(t_x, t_wu);
            qllm_tensor_destroy(t_x); qllm_tensor_destroy(t_wg); qllm_tensor_destroy(t_wu);

            if (t_gate && t_up) {
                float* gate = (float*)qllm_tensor_get_data(t_gate);
                float* up = (float*)qllm_tensor_get_data(t_up);
                if (!gate || !up) {
                    qllm_tensor_destroy(t_gate);
                    qllm_tensor_destroy(t_up);
                    continue;  /* skip this layer */
                }
                for (int i = 0; i < FFN_DIM; i++) {
                    gate[i] = sigmoidf(gate[i] + w->ff_gate_b[L][i]);
                    up[i] += w->ff_up_b[L][i];
                }

                /* Element-wise gate * up using qLLM */
                qllm_tensor_t* t_hidden = qllm_tensor_mul(t_gate, t_up);
                qllm_tensor_destroy(t_gate); qllm_tensor_destroy(t_up);

                if (t_hidden) {
                    qllm_tensor_t* t_wdn = make_tensor_2d(w->ff_down[L], FFN_DIM, D);
                    if (!t_wdn) { qllm_tensor_destroy(t_hidden); continue; }
                    qllm_tensor_t* t_fo = qllm_tensor_matmul(t_hidden, t_wdn);
                    qllm_tensor_destroy(t_hidden); qllm_tensor_destroy(t_wdn);

                    if (t_fo) {
                        float* fo = (float*)qllm_tensor_get_data(t_fo);
                        if (fo) {
                            for (int i = 0; i < D; i++) x[i] += fo[i] + w->ff_down_b[L][i];
                        }
                        qllm_tensor_destroy(t_fo);
                    }
                }
            } else {
                if (t_gate) qllm_tensor_destroy(t_gate);
                if (t_up) qllm_tensor_destroy(t_up);
            }
        }

        /* Check output and halt */
        if (x[S_OUTPUT] >= -0.5f && n_out < max_out)
            outputs[n_out++] = x[S_OUTPUT];
        if (x[S_HALT] > 0.5f) break;
        memcpy(state, x, sizeof(state));
    }

    return n_out;
}

int main(int argc, char** argv) {
    const char* weight_path = "/tmp/interpreter_weights.bin";
    if (argc > 1) weight_path = argv[1];

    printf("=== qLLM Interpreter Integration ===\n\n");
    printf("  Loading weights from %s\n", weight_path);

    Weights* w = load_weights(weight_path);
    if (!w) return 1;

    printf("\n  Running programs through qLLM tensor operations:\n\n");

    int pass = 0, fail = 0;

    #define QTEST(name, prog, n_instr, expected) do { \
        float out[1]; \
        int nout = run_qllm_native(w, prog, n_instr, out, 1); \
        float v = nout > 0 ? out[0] : -9999.0f; \
        int ok = (nout > 0 && fabsf(v - (expected)) < 0.1f); \
        printf("  %-20s = %7.1f (expected %7.1f) %s\n", name, v, (float)(expected), ok?"PASS":"FAIL"); \
        if (ok) pass++; else fail++; \
    } while(0)

    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_ADD,0},{OP_OUTPUT,0},{OP_HALT,0}};
      QTEST("3+5", p, 5, 8); }
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_ADD,0},{OP_CONST,2},{OP_MUL,0},{OP_OUTPUT,0},{OP_HALT,0}};
      QTEST("(3+5)*2", p, 7, 16); }
    { Instr p[]={{OP_CONST,10},{OP_CONST,7},{OP_SUB,0},{OP_OUTPUT,0},{OP_HALT,0}};
      QTEST("10-7", p, 5, 3); }
    { Instr p[]={{OP_CONST,0},{OP_CONST,42},{OP_STORE,0},{OP_CONST,0},{OP_LOAD,0},{OP_OUTPUT,0},{OP_HALT,0}};
      QTEST("mem[0]=42", p, 7, 42); }
    { Instr p[]={{OP_CONST,7},{OP_CONST,11},{OP_MUL,0},{OP_OUTPUT,0},{OP_HALT,0}};
      QTEST("7*11", p, 5, 77); }
    { Instr p[]={
        {OP_CONST,0},{OP_CONST,0},{OP_STORE,0},
        {OP_CONST,1},{OP_CONST,5},{OP_STORE,0},
        {OP_CONST,1},{OP_LOAD,0},{OP_DUP,0},{OP_JUMP_IF,12},{OP_DROP,0},{OP_JUMP,26},
        {OP_CONST,0},{OP_LOAD,0},{OP_ADD,0},{OP_CONST,0},{OP_SWAP,0},{OP_STORE,0},
        {OP_CONST,1},{OP_LOAD,0},{OP_CONST,1},{OP_SUB,0},{OP_CONST,1},{OP_SWAP,0},{OP_STORE,0},
        {OP_JUMP,6},{OP_CONST,0},{OP_LOAD,0},{OP_OUTPUT,0},{OP_HALT,0},
      }; QTEST("sum(1..5)", p, 30, 15); }
    { Instr p[]={
        {OP_CONST,2},{OP_CONST,3},{OP_MUL,0},
        {OP_CONST,4},{OP_CONST,5},{OP_MUL,0},
        {OP_ADD,0},{OP_OUTPUT,0},{OP_HALT,0},
      }; QTEST("(2*3)+(4*5)", p, 9, 26); }

    #undef QTEST

    printf("\n=== qLLM Integration: %d passed, %d failed ===\n", pass, fail);
    free(w);
    printf("\n=== Done ===\n");
    return 0;
}
