/* Exercise the exported VJP against the full trainer and a separate
 * double-precision forward pass with one parameter or input varied at a
 * time. Every gradient the kernel writes is checked: dW_q, dW_k, dW_v,
 * dW_o, db_q and dx. */
#define main weight_matrices_program_main
#include "../../lib/backend/weight_matrices.c"
#undef main
#include <eshkol/backend/qllm_backward.h>

enum { positions = 3, features = 4 };
enum { VARY_Q, VARY_K, VARY_V, VARY_O, VARY_BQ, VARY_X, VARY_COUNT };
static InterpreterWeights weights;
static WeightGrads exact, actual;
static float embedding[positions][D], state[D], upstream[D], query[D];
static float keys[positions][D], values[positions][D], probabilities[positions], head[D];

/* Attention output contracted with the upstream gradient, in double
 * precision, with parameter or input `index` of kind `vary` shifted by
 * `delta`. With cache set it also records the forward intermediates the
 * kernel consumes. */
static double loss(int vary, int index, double delta, int cache) {
    double q[HD], logits[positions], projected[positions][HD], denominator = 0;
    for (int d = 0; d < HD; ++d) {
        q[d] = (vary == VARY_BQ && index == d) ? delta : 0;
        for (int j = 0; j < features; ++j) {
            int at = d * D + j;
            double wq = weights.wq[0][at] + ((vary == VARY_Q && at == index) ? delta : 0);
            double x = state[j] + ((vary == VARY_X && j == index) ? delta : 0);
            q[d] += wq * x;
        }
    }
    for (int p = 0; p < positions; ++p) {
        double logit = 0;
        for (int d = 0; d < HD; ++d) {
            double k = 0, v = 0;
            for (int j = 0; j < features; ++j) {
                int at = d * D + j;
                double wk = weights.wk[0][at] + ((vary == VARY_K && at == index) ? delta : 0);
                double wv = weights.wv[0][at] + ((vary == VARY_V && at == index) ? delta : 0);
                k += wk * embedding[p][j];
                v += wv * embedding[p][j];
            }
            if (cache) { keys[p][d] = k; values[p][d] = v; }
            projected[p][d] = v;
            logit += k * q[d];
        }
        logits[p] = exp(logit / sqrt((double)HD));
        denominator += logits[p];
    }
    double result = 0;
    for (int p = 0; p < positions; ++p) {
        double probability = logits[p] / denominator;
        if (cache) probabilities[p] = probability;
        for (int d = 0; d < HD; ++d) {
            double contribution = probability * projected[p][d];
            if (cache) head[d] += contribution;
            for (int j = 0; j < features; ++j) {
                int at = j * D + d;
                double wo = weights.wo[0][at] + ((vary == VARY_O && at == index) ? delta : 0);
                result += upstream[j] * wo * contribution;
            }
        }
    }
    return result;
}

/* True when `index` is a coordinate the forward pass reads. */
static int active(int vary, int index) {
    switch (vary) {
    case VARY_O:  return index / D < features && index % D < HD;
    case VARY_BQ: return index < HD;
    case VARY_X:  return index < features;
    default:      return index / D < HD && index % D < features;
    }
}

static int close_enough(double a, double b) {
    return fabs(a - b) <= 1e-6 * fmax(1.0, fabs(b));
}

int main(void) {
    for (int j = 0; j < features; ++j) {
        state[j] = 0.2f * (j + 1);
        upstream[j] = 0.3f - 0.17f * j;
        for (int p = 0; p < positions; ++p)
            embedding[p][j] = sin((p + 1.0) * (j + 1));
        for (int d = 0; d < HD; ++d) {
            weights.wq[0][d * D + j] = 0.3f * (d + 1) - 0.11f * j;
            weights.wk[0][d * D + j] = 0.2f * j - 0.07f * d;
            weights.wv[0][d * D + j] = 0.19f * (d + 1) - 0.22f * j;
            weights.wo[0][j * D + d] = 0.6f - 0.18f * j + 0.1f * d;
            query[d] += weights.wq[0][d * D + j] * state[j];
        }
    }
    loss(VARY_Q, -1, 0, 1);
    float dx[D] = {0}, trainer_dstate[D] = {0}, trainer_dx[D];
    qllm_backward_attention(positions, embedding, state, query, keys, values, probabilities,
        head, weights.wq[0], weights.wo[0], upstream, dx, actual.dwq[0], actual.dwk[0],
        actual.dwv[0], actual.dwo[0], actual.dbq[0]);
    backward_through_weights(&weights, state, embedding, positions, upstream, &exact,
                             trainer_dstate);
    /* The trainer's state gradient also carries the residual connection,
     * which is the upstream gradient itself; the kernel reports the
     * attention branch alone. */
    for (int j = 0; j < D; ++j) trainer_dx[j] = trainer_dstate[j] - upstream[j];

    static const char* names[VARY_COUNT] = {"dW_q", "dW_k", "dW_v", "dW_o", "db_q", "dx"};
    for (int vary = 0; vary < VARY_COUNT; ++vary) {
        const float* grad; const float* reference; int count = D * D;
        switch (vary) {
        case VARY_Q:  grad = actual.dwq[0]; reference = exact.dwq[0]; break;
        case VARY_K:  grad = actual.dwk[0]; reference = exact.dwk[0]; break;
        case VARY_V:  grad = actual.dwv[0]; reference = exact.dwv[0]; break;
        case VARY_O:  grad = actual.dwo[0]; reference = exact.dwo[0]; break;
        case VARY_BQ: grad = actual.dbq[0]; reference = exact.dbq[0]; count = D; break;
        default:      grad = dx; reference = trainer_dx; count = D; break;
        }
        double largest = 0, max_trainer_error = 0, max_fd_error = 0;
        for (int i = 0; i < count; ++i) {
            if (!close_enough(grad[i], reference[i])) {
                fprintf(stderr, "%s[%d] = %.9g, trainer = %.9g\n",
                        names[vary], i, (double)grad[i], (double)reference[i]);
                return 1;
            }
            max_trainer_error = fmax(max_trainer_error, fabs(grad[i] - reference[i]));
            if (!active(vary, i)) {
                if (grad[i] != 0.0f) {
                    fprintf(stderr, "%s[%d] = %.9g on an unused coordinate\n",
                            names[vary], i, (double)grad[i]);
                    return 1;
                }
                continue;
            }
            double fd = (loss(vary, i, 1e-5, 0) - loss(vary, i, -1e-5, 0)) / 2e-5;
            if (!close_enough(grad[i], fd)) {
                fprintf(stderr, "%s[%d] = %.9g, finite difference = %.9g\n",
                        names[vary], i, (double)grad[i], fd);
                return 1;
            }
            max_fd_error = fmax(max_fd_error, fabs(fd - grad[i]));
            largest = fmax(largest, fabs(fd));
        }
        printf("%s max magnitude %.9g, max trainer error %.9g, max finite difference error %.9g\n",
               names[vary], largest, max_trainer_error, max_fd_error);
        if (largest < 1e-5) return 1;
    }
    return 0;
}
