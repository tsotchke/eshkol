/* diff_memory_prototype.c — Differentiable external memory addressing head.
 *
 * STANDALONE de-risking experiment for the SDNC (lib/backend/weight_matrices.c).
 *
 * The SDNC currently does *bounded* differentiable memory: an in-state arena
 * (Zone E, cells 128-207, 16 cells) addressed by saturated sigmoid "indicator"
 * gates:
 *
 *     indicator(x,k) = sigmoid(SCALE*(x-k+0.5)) - sigmoid(SCALE*(x-k-0.5))
 *
 * a soft one-hot bump at address k; with SCALE=300 this is effectively a hard
 * step, so addressing is bit-exact. This prototype GENERALIZES that to an
 * NTM/DNC-style external bank of arbitrary size (N rows x W cols) with a
 * TEMPERATURE knob (beta, the analogue of SCALE) so the SAME head is:
 *
 *   - bit-exact at high beta  (verified execution: the bit-identity property),
 *   - smooth/differentiable at low beta (learning).
 *
 * It implements location addressing, content addressing (cosine + softmax),
 * NTM erase/add write, dynamic least-used allocation, and a hand-derived
 * backprop through read + content addressing checked against double-precision
 * central finite differences.
 *
 * This file is self-contained (its own main()); it does NOT touch
 * weight_matrices.c or anything else. Build:
 *   add_executable(diff_memory_prototype lib/backend/diff_memory_prototype.c)
 *   target_link_libraries(diff_memory_prototype PRIVATE m)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- dimensions ----------------------------------------------------------- */
#define N 64   /* number of memory rows (>> the SDNC's 16-cell arena bound)    */
#define W 8    /* width of each row / read/write/key vector                    */

typedef struct {
    double mem[N][W];   /* the external bank                                    */
    double usage[N];    /* per-row usage / allocation vector (bump allocator)  */
} Memory;

/* ===========================================================================
 * Primitives
 * =========================================================================== */

/* sigmoid (double) mirroring weight_matrices.c's sigmoidf saturation logic. */
static double sigmoidd(double x) {
    if (x > 40.0) return 1.0;
    if (x < -40.0) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

/* Generalized indicator bump at integer address k, evaluated at x.
 * Identical in form to weight_matrices.c's indicator(); beta plays SCALE. */
static double indicator(double x, double k, double beta) {
    return sigmoidd(beta * (x - k + 0.5)) - sigmoidd(beta * (x - k - 0.5));
}

/* Numerically stable softmax of `logits[n]` into `out[n]`. */
static void softmax(const double *logits, int n, double *out) {
    double m = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > m) m = logits[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) { out[i] = exp(logits[i] - m); s += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= s;
}

/* ===========================================================================
 * Location addressing
 *
 * loc_weights(addr,beta): a weighting over the N rows that, as beta -> inf,
 * becomes a hard one-hot at round(addr). We use the indicator bump (the SDNC
 * form) which gives exactly-1 at the matching integer row at high beta.
 * =========================================================================== */
static void loc_weights(double addr, double beta, double *w) {
    for (int i = 0; i < N; i++) w[i] = indicator(addr, (double)i, beta);
    /* At low beta the bumps don't sum to 1; normalize so it stays a proper
     * weighting. At high beta the matching bump is ~1 and the rest ~0, so
     * normalization is a no-op and exactness is preserved. */
    double s = 0.0;
    for (int i = 0; i < N; i++) s += w[i];
    if (s > 1e-300) for (int i = 0; i < N; i++) w[i] /= s;
}

/* ===========================================================================
 * Content addressing: softmax(beta * cosine_sim(key, row_i))
 * =========================================================================== */
static double cosine_sim(const double *a, const double *b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    double denom = sqrt(na) * sqrt(nb) + 1e-12;
    return dot / denom;
}

static void content_weights(const Memory *M, const double *key, double beta, double *w) {
    double logits[N];
    for (int i = 0; i < N; i++) logits[i] = beta * cosine_sim(key, M->mem[i], W);
    softmax(logits, N, w);
}

/* ===========================================================================
 * Read / Write
 * =========================================================================== */
static void read_mem(const Memory *M, const double *w, double *out) {
    for (int j = 0; j < W; j++) {
        double acc = 0.0;
        for (int i = 0; i < N; i++) acc += w[i] * M->mem[i][j];
        out[j] = acc;
    }
}

/* NTM erase/add: mem[i] = mem[i]*(1 - w[i]*erase) + w[i]*add */
static void write_mem(Memory *M, const double *w, const double *erase, const double *add) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < W; j++) {
            M->mem[i][j] = M->mem[i][j] * (1.0 - w[i]*erase[j]) + w[i]*add[j];
        }
        M->usage[i] += w[i];   /* track usage for allocation */
    }
}

/* ===========================================================================
 * Dynamic allocation: weighting that picks the least-used row.
 * logits = -beta * usage  -> softmax. As beta -> inf this is a deterministic
 * one-hot at the minimum-usage row, the differentiable analogue of the SDNC's
 * S_ARENA_NEXT bump allocator ("next free cell").
 * =========================================================================== */
static void alloc_weights(const Memory *M, double beta, double *w) {
    double logits[N];
    for (int i = 0; i < N; i++) logits[i] = -beta * M->usage[i];
    softmax(logits, N, w);
}

/* ===========================================================================
 * Backprop through L = 0.5 * sum_j (read_j - target_j)^2
 * where read = sum_i w_i M_i, w = softmax(beta * s), s_i = cosine(key, M_i).
 *
 * Returns L. Fills grad_key[W] = dL/dkey and grad_row[W] = dL/dM_row for the
 * single memory row index `row`.
 * =========================================================================== */
static double loss_and_grads(const Memory *M, const double *key, const double *target,
                             double beta, int row,
                             double *grad_key, double *grad_row) {
    /* ---- forward ---- */
    double s[N], w[N], rd[W];
    for (int i = 0; i < N; i++) s[i] = cosine_sim(key, M->mem[i], W);
    {
        double logits[N];
        for (int i = 0; i < N; i++) logits[i] = beta * s[i];
        softmax(logits, N, w);
    }
    read_mem(M, w, rd);

    double L = 0.0;
    double dread[W];               /* dL/dread_j = (read_j - target_j)      */
    for (int j = 0; j < W; j++) {
        double e = rd[j] - target[j];
        L += 0.5 * e * e;
        dread[j] = e;
    }

    /* dL/dw_i = sum_j dread_j * M[i][j] */
    double dw[N];
    for (int i = 0; i < N; i++) {
        double acc = 0.0;
        for (int j = 0; j < W; j++) acc += dread[j] * M->mem[i][j];
        dw[i] = acc;
    }

    /* Through softmax: dL/dlogit_k = w_k * (dw_k - sum_i w_i dw_i)
     * and logit_i = beta * s_i  => dL/ds_k = beta * dL/dlogit_k. */
    double wdotdw = 0.0;
    for (int i = 0; i < N; i++) wdotdw += w[i] * dw[i];
    double ds[N];
    for (int k = 0; k < N; k++) ds[k] = beta * w[k] * (dw[k] - wdotdw);

    /* ---- dL/dkey via cosine sims ----
     * s_i = (key . M_i) / (|key| |M_i| + eps).
     * Let na=|key|^2, nbi=|M_i|^2, denom_i = sqrt(na)*sqrt(nbi)+eps,
     * dot_i = key . M_i.
     * ds_i/dkey_t = M_i[t]/denom_i - dot_i * (sqrt(nbi)*key[t]/sqrt(na)) / denom_i^2
     */
    double na = 0.0;
    for (int t = 0; t < W; t++) na += key[t]*key[t];
    double sqrt_na = sqrt(na);

    for (int t = 0; t < W; t++) grad_key[t] = 0.0;
    for (int i = 0; i < N; i++) {
        double dot = 0.0, nbi = 0.0;
        for (int t = 0; t < W; t++) { dot += key[t]*M->mem[i][t]; nbi += M->mem[i][t]*M->mem[i][t]; }
        double sqrt_nbi = sqrt(nbi);
        double denom = sqrt_na * sqrt_nbi + 1e-12;
        double inv_denom = 1.0 / denom;
        /* d denom / d key[t] = sqrt_nbi * key[t] / sqrt_na  (when sqrt_na>0) */
        double dsqrt_na_term = (sqrt_na > 1e-300) ? (sqrt_nbi / sqrt_na) : 0.0;
        for (int t = 0; t < W; t++) {
            double ddot = M->mem[i][t];
            double ddenom = dsqrt_na_term * key[t];
            double dsi_dkeyt = ddot * inv_denom - dot * ddenom * inv_denom * inv_denom;
            grad_key[t] += ds[i] * dsi_dkeyt;
        }
    }

    /* ---- dL/dM_row ----
     * M_row enters BOTH the read (read = sum_i w_i M_i) and the similarity s_row.
     * (1) direct read path: dL/dM_row[t] += dread_t * w_row
     * (2) through s_row: ds_row/dM_row[t] = key[t]/denom_row
     *        - dot_row * (sqrt_na * M_row[t]/sqrt_nb_row) / denom_row^2
     */
    {
        double dot = 0.0, nbr = 0.0;
        for (int t = 0; t < W; t++) { dot += key[t]*M->mem[row][t]; nbr += M->mem[row][t]*M->mem[row][t]; }
        double sqrt_nbr = sqrt(nbr);
        double denom = sqrt_na * sqrt_nbr + 1e-12;
        double inv_denom = 1.0 / denom;
        double dsqrt_nb_term = (sqrt_nbr > 1e-300) ? (sqrt_na / sqrt_nbr) : 0.0;
        for (int t = 0; t < W; t++) {
            double ddot = key[t];
            double ddenom = dsqrt_nb_term * M->mem[row][t];
            double dsrow_dMt = ddot * inv_denom - dot * ddenom * inv_denom * inv_denom;
            grad_row[t] = dread[t] * w[row]          /* read path  */
                        + ds[row] * dsrow_dMt;        /* sim path   */
        }
    }

    return L;
}

/* Loss-only helper for finite differences (forward pass only). */
static double loss_only(const Memory *M, const double *key, const double *target, double beta) {
    double w[N], rd[W];
    content_weights(M, key, beta, w);
    read_mem(M, w, rd);
    double L = 0.0;
    for (int j = 0; j < W; j++) { double e = rd[j]-target[j]; L += 0.5*e*e; }
    return L;
}

/* ===========================================================================
 * main: three acceptance checks
 * =========================================================================== */
static double frand(void) { return (double)rand() / (double)RAND_MAX * 2.0 - 1.0; }

int main(void) {
    srand(12345);
    int all_pass = 1;

    /* ----------------------------------------------------------------------
     * CHECK 1 — Exactness at high temperature (bit-identity property).
     * Write distinct W-vectors to integer addresses 0..M-1 with M=40 (> the
     * old 16-cell bound), then read back by integer address. Each read must
     * equal the written vector to < 1e-5.
     * ---------------------------------------------------------------------- */
    {
        const double beta = 1e4;
        const int M_addrs = 40;     /* exceeds the SDNC's 16-cell arena bound */
        Memory M; memset(&M, 0, sizeof(M));

        double written[40][W];
        double zero_erase[W];
        for (int j = 0; j < W; j++) zero_erase[j] = 1.0; /* full erase, so write = add */

        for (int a = 0; a < M_addrs; a++) {
            double w[N];
            loc_weights((double)a, beta, w);
            double add[W];
            for (int j = 0; j < W; j++) { add[j] = (double)(a*W + j) * 0.5 - 7.0; written[a][j] = add[j]; }
            write_mem(&M, w, zero_erase, add);
        }

        double max_err = 0.0;
        for (int a = 0; a < M_addrs; a++) {
            double w[N], rd[W];
            loc_weights((double)a, beta, w);
            read_mem(&M, w, rd);
            for (int j = 0; j < W; j++) {
                double e = fabs(rd[j] - written[a][j]);
                if (e > max_err) max_err = e;
            }
        }
        int pass = (max_err < 1e-5);
        all_pass &= pass;
        printf("CHECK 1 (exactness @ beta=%.0e, M=%d > 16): max_err = %.3e  -> %s\n",
               beta, M_addrs, max_err, pass ? "PASS" : "FAIL");
    }

    /* ----------------------------------------------------------------------
     * CHECK 2 — Differentiability (gradient check) at beta=2.0.
     * Analytic dL/dkey and dL/dM_row vs central finite differences (+/-1e-4).
     * max relative error < 1e-3.
     * ---------------------------------------------------------------------- */
    {
        const double beta = 2.0;
        Memory M; memset(&M, 0, sizeof(M));
        for (int i = 0; i < N; i++) for (int j = 0; j < W; j++) M.mem[i][j] = frand();

        double key[W], target[W];
        for (int j = 0; j < W; j++) { key[j] = frand(); target[j] = frand(); }
        const int row = 7;

        double grad_key[W], grad_row[W];
        loss_and_grads(&M, key, target, beta, row, grad_key, grad_row);

        const double h = 1e-4;
        double max_rel = 0.0;

        /* finite-diff dL/dkey */
        for (int t = 0; t < W; t++) {
            double saved = key[t];
            key[t] = saved + h; double Lp = loss_only(&M, key, target, beta);
            key[t] = saved - h; double Lm = loss_only(&M, key, target, beta);
            key[t] = saved;
            double fd = (Lp - Lm) / (2.0*h);
            double denom = fmax(1e-8, fmax(fabs(fd), fabs(grad_key[t])));
            double rel = fabs(fd - grad_key[t]) / denom;
            if (rel > max_rel) max_rel = rel;
        }
        /* finite-diff dL/dM_row */
        for (int t = 0; t < W; t++) {
            double saved = M.mem[row][t];
            M.mem[row][t] = saved + h; double Lp = loss_only(&M, key, target, beta);
            M.mem[row][t] = saved - h; double Lm = loss_only(&M, key, target, beta);
            M.mem[row][t] = saved;
            double fd = (Lp - Lm) / (2.0*h);
            double denom = fmax(1e-8, fmax(fabs(fd), fabs(grad_row[t])));
            double rel = fabs(fd - grad_row[t]) / denom;
            if (rel > max_rel) max_rel = rel;
        }
        int pass = (max_rel < 1e-3);
        all_pass &= pass;
        printf("CHECK 2 (grad-check @ beta=%.1f): max rel-err = %.3e  -> %s\n",
               beta, max_rel, pass ? "PASS" : "FAIL");
    }

    /* ----------------------------------------------------------------------
     * CHECK 3 — Learning: SGD on key to retrieve a fixed target row.
     * Loss must strictly decrease (first vs last).
     * ---------------------------------------------------------------------- */
    {
        const double beta = 2.0;
        Memory M; memset(&M, 0, sizeof(M));
        for (int i = 0; i < N; i++) for (int j = 0; j < W; j++) M.mem[i][j] = frand();

        const int target_row = 23;
        double target[W];
        for (int j = 0; j < W; j++) target[j] = M.mem[target_row][j]; /* retrieve this row */

        double key[W];
        for (int j = 0; j < W; j++) key[j] = frand();

        double lr = 0.5;
        double first_loss = loss_only(&M, key, target, beta);
        double last_loss = first_loss;
        for (int step = 0; step < 100; step++) {
            double gk[W], gr[W];
            last_loss = loss_and_grads(&M, key, target, beta, target_row, gk, gr);
            for (int j = 0; j < W; j++) key[j] -= lr * gk[j];   /* train key only */
        }
        last_loss = loss_only(&M, key, target, beta);

        int pass = (last_loss < first_loss);
        all_pass &= pass;
        printf("CHECK 3 (learning @ beta=%.1f, 100 SGD steps): loss %.6e -> %.6e  -> %s\n",
               beta, first_loss, last_loss, pass ? "PASS" : "FAIL");
    }

    printf("\nOVERALL: %s\n", all_pass ? "ALL PASS" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
