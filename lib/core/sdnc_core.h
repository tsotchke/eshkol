/*
 * SDNC Core — shared bytecode-VM-as-transformer weight math.
 *
 * Self-contained header exposing the trainable core of
 * lib/backend/weight_matrices.c (the SDNC paper artifact) so that the
 * Eshkol runtime (lib/core/sdnc_api.c) can drive the self-improvement loop
 * — forward -> loss -> backward_through_weights -> apply_weight_gradient_step —
 * on real program weights theta from .esk.
 *
 * The struct layout (InterpreterWeights / WeightGrads) and the forward /
 * backward / gradient-step math here are byte-for-byte mirrors of the
 * definitions in lib/backend/weight_matrices.c. They depend ONLY on the weight
 * structs, the model constants below, and libm — they touch NONE of the
 * standalone artifact's VM globals (g_frames, g_heap, ...). Keeping them in a
 * standalone header (rather than un-static'ing symbols that live in a separate
 * link unit with its own main()) means the `weight_matrices` executable and its
 * 127-test suite build entirely unchanged.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_SDNC_CORE_H
#define ESHKOL_SDNC_CORE_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Model constants (mirror weight_matrices.c) ---- */
#ifndef SDNC_D
#define SDNC_D 256
#endif
#ifndef SDNC_HD
#define SDNC_HD 2
#endif
#ifndef SDNC_N_LAYERS
#define SDNC_N_LAYERS 6
#endif
#ifndef SDNC_FFN_DIM
#define SDNC_FFN_DIM 2304
#endif

/* Local short aliases used inside this header only. */
#define D       SDNC_D
#define HD      SDNC_HD
#define N_LAYERS SDNC_N_LAYERS
#define FFN_DIM SDNC_FFN_DIM

/* ---- Trainable weights (mirror of InterpreterWeights' trainable subset
 *      plus the per-layer ff_type schedule) ---- */
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
} SdncWeights;

/* ---- Gradients: mirror the trainable subset of SdncWeights ---- */
typedef struct {
    float dwq[N_LAYERS][D * D];
    float dwk[N_LAYERS][D * D];
    float dwv[N_LAYERS][D * D];
    float dwo[N_LAYERS][D * D];
    float dbq[N_LAYERS][D];
    float dff_up[N_LAYERS][D * FFN_DIM];
    float dff_up_b[N_LAYERS][FFN_DIM];
    float dff_down[N_LAYERS][FFN_DIM * D];
    float dff_down_b[N_LAYERS][D];
    float dff_gate[N_LAYERS][D * FFN_DIM];
    float dff_gate_b[N_LAYERS][FFN_DIM];
} SdncGrads;

/* Per-layer forward cache so the backward pass reuses exact intermediates. */
typedef struct {
    float x_in[N_LAYERS][D];
    float x_post_attn[N_LAYERS][D];
    int   attn_active;
    float Q[D];
    float K[256][D];
    float Va[256][D];
    float scores[256];
    float hout[D];
    int   np;
    float ff_h[N_LAYERS][FFN_DIM];
    float ff_gate[N_LAYERS][FFN_DIM];
    float ff_up[N_LAYERS][FFN_DIM];
} SdncFwdCache;

/* ===================================================================== */

static inline float sdnc_sigmoidf(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static inline void sdnc_matvec_t(const float* x, const float* W, float* out,
                                 int rows, int cols) {
    memset(out, 0, cols * sizeof(float));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out[j] += x[i] * W[i * cols + j];
}

/* Apply a single FFN layer via actual weight matrices (W @ x + b).
 * type 0 (noop), 1 (SQUARE), 2 (gated sigmoid). Mirrors apply_ffn_layer. */
static inline void sdnc_apply_ffn_layer(const SdncWeights* w, int L, float x[D]) {
    float fo[D]; memset(fo, 0, sizeof(fo));
    if (w->ff_type[L] == 1) {
        float h[FFN_DIM];
        sdnc_matvec_t(x, w->ff_up[L], h, D, FFN_DIM);
        for (int i = 0; i < FFN_DIM; i++) h[i] += w->ff_up_b[L][i];
        for (int i = 0; i < FFN_DIM; i++) h[i] *= h[i]; /* SQUARE */
        sdnc_matvec_t(h, w->ff_down[L], fo, FFN_DIM, D);
        for (int i = 0; i < D; i++) fo[i] += w->ff_down_b[L][i];
    } else if (w->ff_type[L] == 2) {
        float gate[FFN_DIM], up[FFN_DIM], h[FFN_DIM];
        sdnc_matvec_t(x, w->ff_gate[L], gate, D, FFN_DIM);
        for (int i = 0; i < FFN_DIM; i++) gate[i] = sdnc_sigmoidf(gate[i] + w->ff_gate_b[L][i]);
        sdnc_matvec_t(x, w->ff_up[L], up, D, FFN_DIM);
        for (int i = 0; i < FFN_DIM; i++) up[i] += w->ff_up_b[L][i];
        for (int i = 0; i < FFN_DIM; i++) h[i] = gate[i] * up[i];
        sdnc_matvec_t(h, w->ff_down[L], fo, FFN_DIM, D);
        for (int i = 0; i < D; i++) fo[i] += w->ff_down_b[L][i];
    }
    for (int i = 0; i < D; i++) x[i] += fo[i];
}

/* Forward pass through layers 0..N_LAYERS-2 (layer 5 is backward-only).
 * Mirrors forward_with_weights. */
static inline void sdnc_forward(const SdncWeights* w,
                                const float state[D],
                                const float pe[][D], int np,
                                float next[D]) {
    float x[D]; memcpy(x, state, sizeof(float)*D);
    for (int L = 0; L < N_LAYERS - 1; L++) {
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
        sdnc_apply_ffn_layer(w, L, x);
    }
    memcpy(next, x, sizeof(float)*D);
}

static inline void sdnc_zero_grads(SdncGrads* g) { memset(g, 0, sizeof(SdncGrads)); }

/* Backprop through one matvec_t. dx may be NULL. Mirrors matvec_backward. */
static inline void sdnc_matvec_backward(const float* x, const float* W,
                                        const float* dout,
                                        float* dW, float* dx, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float dxi = 0.0f;
        const float xi = x[i];
        const float* Wi = W + (size_t)i * cols;
        float* dWi = dW + (size_t)i * cols;
        for (int j = 0; j < cols; j++) {
            float dj = dout[j];
            dWi[j] += xi * dj;
            dxi += dj * Wi[j];
        }
        if (dx) dx[i] += dxi;
    }
}

/* Re-run forward caching intermediates, then backprop. Accumulates into g and
 * writes grad w.r.t. input state to dL_dstate (may be NULL).
 * Mirrors backward_through_weights. */
static inline void sdnc_backward(const SdncWeights* w,
                                 const float state[D],
                                 const float pe[][D], int np,
                                 const float dL_dnext[D],
                                 SdncGrads* g,
                                 float dL_dstate[D],
                                 SdncFwdCache* cache) {
    SdncFwdCache* cp = cache;
    memset(cp, 0, sizeof(*cp));
    SdncFwdCache* c_ = cp;
#define c (*c_)
    c.np = np;

    float x[D]; memcpy(x, state, sizeof(float) * D);
    for (int L = 0; L < N_LAYERS - 1; L++) {
        memcpy(c.x_in[L], x, sizeof(float) * D);
        float ao[D]; memset(ao, 0, sizeof(ao));
        if (L == 0 && np > 0) {
            c.attn_active = 1;
            float Q[D]; memset(Q, 0, sizeof(Q));
            for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) Q[i] += w->wq[L][i*D+j]*x[j];
            for (int i = 0; i < D; i++) Q[i] += w->bq[L][i];
            memcpy(c.Q, Q, sizeof(Q));
            float scores[256]; float mx = -1e30f;
            for (int p = 0; p < np && p < 256; p++) {
                float K[D]; memset(K, 0, sizeof(K));
                memset(c.Va[p], 0, sizeof(float)*D);
                for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) {
                    K[i] += w->wk[L][i*D+j]*pe[p][j];
                    c.Va[p][i] += w->wv[L][i*D+j]*pe[p][j];
                }
                memcpy(c.K[p], K, sizeof(float)*D);
                scores[p] = (Q[0]*K[0] + Q[1]*K[1]) / sqrtf((float)HD);
                if (scores[p] > mx) mx = scores[p];
            }
            float sum = 0;
            for (int p = 0; p < np; p++) { scores[p] = expf(scores[p]-mx); sum += scores[p]; }
            for (int p = 0; p < np; p++) scores[p] /= sum;
            memcpy(c.scores, scores, sizeof(float)*np);
            float hout[D]; memset(hout, 0, sizeof(hout));
            for (int p = 0; p < np; p++) for (int d = 0; d < HD; d++) hout[d] += scores[p]*c.Va[p][d];
            memcpy(c.hout, hout, sizeof(hout));
            for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) ao[i] += w->wo[L][i*D+j]*hout[j];
        }
        for (int i = 0; i < D; i++) x[i] += ao[i];
        memcpy(c.x_post_attn[L], x, sizeof(float)*D);

        if (w->ff_type[L] == 1) {
            float h[FFN_DIM];
            sdnc_matvec_t(x, w->ff_up[L], h, D, FFN_DIM);
            for (int i = 0; i < FFN_DIM; i++) h[i] += w->ff_up_b[L][i];
            memcpy(c.ff_h[L], h, sizeof(float)*FFN_DIM);
            for (int i = 0; i < FFN_DIM; i++) h[i] *= h[i];
            float fo[D]; sdnc_matvec_t(h, w->ff_down[L], fo, FFN_DIM, D);
            for (int i = 0; i < D; i++) fo[i] += w->ff_down_b[L][i];
            for (int i = 0; i < D; i++) x[i] += fo[i];
        } else if (w->ff_type[L] == 2) {
            float gate[FFN_DIM], up[FFN_DIM], h[FFN_DIM];
            sdnc_matvec_t(x, w->ff_gate[L], gate, D, FFN_DIM);
            for (int i = 0; i < FFN_DIM; i++) gate[i] = sdnc_sigmoidf(gate[i] + w->ff_gate_b[L][i]);
            sdnc_matvec_t(x, w->ff_up[L], up, D, FFN_DIM);
            for (int i = 0; i < FFN_DIM; i++) up[i] += w->ff_up_b[L][i];
            for (int i = 0; i < FFN_DIM; i++) h[i] = gate[i]*up[i];
            memcpy(c.ff_gate[L], gate, sizeof(float)*FFN_DIM);
            memcpy(c.ff_up[L], up, sizeof(float)*FFN_DIM);
            memcpy(c.ff_h[L], h, sizeof(float)*FFN_DIM);
            float fo[D]; sdnc_matvec_t(h, w->ff_down[L], fo, FFN_DIM, D);
            for (int i = 0; i < D; i++) fo[i] += w->ff_down_b[L][i];
            for (int i = 0; i < D; i++) x[i] += fo[i];
        }
    }

    float dx[D]; memcpy(dx, dL_dnext, sizeof(float)*D);
    for (int L = N_LAYERS - 2; L >= 0; L--) {
        if (w->ff_type[L] == 1) {
            float* dfo = dx;
            for (int i = 0; i < D; i++) g->dff_down_b[L][i] += dfo[i];
            float dh2[FFN_DIM]; memset(dh2, 0, sizeof(dh2));
            float h[FFN_DIM];
            for (int i = 0; i < FFN_DIM; i++) h[i] = c.ff_h[L][i]*c.ff_h[L][i];
            sdnc_matvec_backward(h, w->ff_down[L], dfo, g->dff_down[L], dh2, FFN_DIM, D);
            float dh[FFN_DIM];
            for (int i = 0; i < FFN_DIM; i++) dh[i] = dh2[i] * 2.0f * c.ff_h[L][i];
            for (int i = 0; i < FFN_DIM; i++) g->dff_up_b[L][i] += dh[i];
            float dxin[D]; memset(dxin, 0, sizeof(dxin));
            sdnc_matvec_backward(c.x_post_attn[L], w->ff_up[L], dh, g->dff_up[L], dxin, D, FFN_DIM);
            for (int i = 0; i < D; i++) dx[i] += dxin[i];
        } else if (w->ff_type[L] == 2) {
            float* dfo = dx;
            for (int i = 0; i < D; i++) g->dff_down_b[L][i] += dfo[i];
            float dh[FFN_DIM]; memset(dh, 0, sizeof(dh));
            sdnc_matvec_backward(c.ff_h[L], w->ff_down[L], dfo, g->dff_down[L], dh, FFN_DIM, D);
            float dgate[FFN_DIM], dup[FFN_DIM];
            for (int i = 0; i < FFN_DIM; i++) { dgate[i] = dh[i]*c.ff_up[L][i]; dup[i] = dh[i]*c.ff_gate[L][i]; }
            for (int i = 0; i < FFN_DIM; i++) g->dff_up_b[L][i] += dup[i];
            float dxin[D]; memset(dxin, 0, sizeof(dxin));
            sdnc_matvec_backward(c.x_post_attn[L], w->ff_up[L], dup, g->dff_up[L], dxin, D, FFN_DIM);
            float dpre[FFN_DIM];
            for (int i = 0; i < FFN_DIM; i++) { float gv = c.ff_gate[L][i]; dpre[i] = dgate[i]*gv*(1.0f-gv); }
            for (int i = 0; i < FFN_DIM; i++) g->dff_gate_b[L][i] += dpre[i];
            sdnc_matvec_backward(c.x_post_attn[L], w->ff_gate[L], dpre, g->dff_gate[L], dxin, D, FFN_DIM);
            for (int i = 0; i < D; i++) dx[i] += dxin[i];
        }

        if (L == 0 && c.attn_active) {
            float* dxpa = dx;
            float dhout[D]; memset(dhout, 0, sizeof(dhout));
            for (int i = 0; i < D; i++) {
                float doi = dxpa[i];
                for (int j = 0; j < D; j++) {
                    g->dwo[0][i*D+j] += c.hout[j]*doi;
                    dhout[j]        += w->wo[0][i*D+j]*doi;
                }
            }
            int np_ = c.np;
            float ds[256]; memset(ds, 0, sizeof(float)*np_);
            for (int p = 0; p < np_; p++) {
                float acc = 0.0f;
                for (int d = 0; d < HD; d++) acc += dhout[d]*c.Va[p][d];
                ds[p] = acc;
                for (int d = 0; d < HD; d++) {
                    float dVad = c.scores[p]*dhout[d];
                    for (int j = 0; j < D; j++) g->dwv[0][d*D+j] += pe[p][j]*dVad;
                }
            }
            float dot = 0.0f;
            for (int p = 0; p < np_; p++) dot += c.scores[p]*ds[p];
            float dscore[256];
            for (int p = 0; p < np_; p++) dscore[p] = c.scores[p]*(ds[p] - dot);
            float inv = 1.0f/sqrtf((float)HD);
            float dQ[D]; memset(dQ, 0, sizeof(dQ));
            for (int p = 0; p < np_; p++) {
                float dsp = dscore[p]*inv;
                dQ[0] += dsp*c.K[p][0];
                dQ[1] += dsp*c.K[p][1];
                float dK0 = dsp*c.Q[0];
                float dK1 = dsp*c.Q[1];
                for (int j = 0; j < D; j++) {
                    g->dwk[0][0*D+j] += pe[p][j]*dK0;
                    g->dwk[0][1*D+j] += pe[p][j]*dK1;
                }
            }
            for (int i = 0; i < D; i++) g->dbq[0][i] += dQ[i];
            float dxin[D]; memset(dxin, 0, sizeof(dxin));
            for (int i = 0; i < D; i++) {
                float dqi = dQ[i];
                if (dqi == 0.0f) continue;
                for (int j = 0; j < D; j++) {
                    g->dwq[0][i*D+j] += c.x_in[0][j]*dqi;
                    dxin[j]          += w->wq[0][i*D+j]*dqi;
                }
            }
            for (int i = 0; i < D; i++) dx[i] += dxin[i];
        }
    }
    if (dL_dstate) memcpy(dL_dstate, dx, sizeof(float)*D);
#undef c
}

/* SGD step. Mirrors apply_weight_gradient_step. */
static inline void sdnc_apply_grad_step(SdncWeights* w, const SdncGrads* g, float lr) {
    for (int L = 0; L < N_LAYERS; L++) {
        for (int i = 0; i < D*D; i++) {
            w->wq[L][i] -= lr*g->dwq[L][i];
            w->wk[L][i] -= lr*g->dwk[L][i];
            w->wv[L][i] -= lr*g->dwv[L][i];
            w->wo[L][i] -= lr*g->dwo[L][i];
        }
        for (int i = 0; i < D; i++) w->bq[L][i] -= lr*g->dbq[L][i];
        for (int i = 0; i < D*FFN_DIM; i++) {
            w->ff_up[L][i]   -= lr*g->dff_up[L][i];
            w->ff_gate[L][i] -= lr*g->dff_gate[L][i];
        }
        for (int i = 0; i < FFN_DIM; i++) {
            w->ff_up_b[L][i]   -= lr*g->dff_up_b[L][i];
            w->ff_gate_b[L][i] -= lr*g->dff_gate_b[L][i];
        }
        for (int i = 0; i < FFN_DIM*D; i++) w->ff_down[L][i] -= lr*g->dff_down[L][i];
        for (int i = 0; i < D; i++) w->ff_down_b[L][i] -= lr*g->dff_down_b[L][i];
    }
}

/* Deterministic LCG (mirror si_randf) for reproducible weight init. */
typedef struct { unsigned long state; float scale; } SdncRng;
static inline float sdnc_randf(SdncRng* r) {
    r->state = r->state*6364136223846793005UL + 1442695040888963407UL;
    unsigned int v = (unsigned int)(r->state >> 33);
    return ((float)v / (float)0x7fffffffu - 1.0f) * r->scale;
}

/* Initialise a fresh weight set exactly like self_improve_demo: small random
 * weights with 1/sqrt(fan-in) scaling and the L0=type1(+attn), L1=type2 FFN
 * schedule that keeps the stacked SQUARE non-linearity numerically bounded and
 * makes the self-improvement loss strictly decrease. */
static inline void sdnc_init_weights(SdncWeights* w) {
    memset(w, 0, sizeof(*w));
    SdncRng r; r.state = 0xC0FFEEUL; r.scale = 0.1f;
    const float s_d   = 0.35f / sqrtf((float)D);
    const float s_ffn = 0.35f / sqrtf((float)FFN_DIM);
    const float s_qk  = 1.6f / sqrtf((float)D);
    for (int L = 0; L < N_LAYERS; L++) {
        r.scale = s_qk;
        for (int i = 0; i < D*D; i++) { w->wq[L][i]=sdnc_randf(&r); w->wk[L][i]=sdnc_randf(&r); }
        r.scale = s_d;
        for (int i = 0; i < D*D; i++) { w->wv[L][i]=sdnc_randf(&r); w->wo[L][i]=sdnc_randf(&r); }
        for (int i = 0; i < D*FFN_DIM; i++){ w->ff_up[L][i]=sdnc_randf(&r); w->ff_gate[L][i]=sdnc_randf(&r); }
        r.scale = s_ffn;
        for (int i = 0; i < FFN_DIM*D; i++) w->ff_down[L][i]=sdnc_randf(&r);
        r.scale = 0.6f;
        for (int i = 0; i < D; i++) w->bq[L][i]=sdnc_randf(&r);
        r.scale = 0.02f;
        for (int i = 0; i < FFN_DIM; i++){ w->ff_up_b[L][i]=sdnc_randf(&r); w->ff_gate_b[L][i]=sdnc_randf(&r); }
        for (int i = 0; i < D; i++) w->ff_down_b[L][i]=sdnc_randf(&r);
    }
    w->ff_type[0]=1; w->ff_type[1]=2; w->ff_type[2]=0;
    w->ff_type[3]=0; w->ff_type[4]=0; w->ff_type[5]=0;
}

#undef D
#undef HD
#undef N_LAYERS
#undef FFN_DIM

#endif /* ESHKOL_SDNC_CORE_H */
