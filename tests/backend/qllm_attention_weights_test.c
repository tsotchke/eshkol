/* Exercise the exported VJP against the full trainer and a separate
 * double-precision forward pass with one weight varied at a time. */
#define main weight_matrices_program_main
#include "../../lib/backend/weight_matrices.c"
#undef main
#include <eshkol/backend/qllm_backward.h>

enum { positions = 3, features = 4 };
static InterpreterWeights weights;
static WeightGrads exact, actual;
static float embedding[positions][D], state[D], upstream[D], query[D];
static float keys[positions][D], values[positions][D], probabilities[positions], head[D];

static double loss(int varying_value, int index, double delta, int cache) {
    double logits[positions], projected[positions][HD], denominator = 0;
    for (int p = 0; p < positions; ++p) {
        double logit = 0;
        for (int d = 0; d < HD; ++d) {
            double k = 0, v = 0;
            for (int j = 0; j < features; ++j) {
                int at = d * D + j;
                double wk = weights.wk[0][at], wv = weights.wv[0][at];
                if (at == index) {
                    if (varying_value) wv += delta;
                    else wk += delta;
                }
                k += wk * embedding[p][j];
                v += wv * embedding[p][j];
            }
            if (cache) { keys[p][d] = k; values[p][d] = v; }
            projected[p][d] = v;
            logit += k * query[d];
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
            for (int j = 0; j < features; ++j)
                result += upstream[j] * weights.wo[0][j * D + d] * contribution;
        }
    }
    return result;
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
    loss(0, -1, 0, 1);
    float dx[D] = {0};
    qllm_backward_attention(positions, embedding, state, query, keys, values, probabilities,
        head, weights.wq[0], weights.wo[0], upstream, dx, actual.dwq[0], actual.dwk[0],
        actual.dwv[0], actual.dwo[0], actual.dbq[0]);
    backward_through_weights(&weights, state, embedding, positions, upstream, &exact, NULL);
    for (int value = 0; value <= 1; ++value) {
        const float* grad = value ? actual.dwv[0] : actual.dwk[0];
        const float* reference = value ? exact.dwv[0] : exact.dwk[0];
        const char* name = value ? "dW_v" : "dW_k";
        double largest = 0, max_trainer_error = 0, max_fd_error = 0;
        for (int i = 0; i < D * D; ++i) {
            double trainer_error = fabs(grad[i] - reference[i]);
            if (!(trainer_error <= 1e-6)) {
                fprintf(stderr, "%s[%d,%d] = %.9g, trainer = %.9g\n",
                        name, i / D, i % D, (double)grad[i], (double)reference[i]);
                return 1;
            }
            max_trainer_error = fmax(max_trainer_error, trainer_error);
            if (i / D >= HD || i % D >= features) continue;
            double fd = (loss(value, i, 1e-5, 0) - loss(value, i, -1e-5, 0)) / 2e-5;
            double fd_error = fabs(fd - grad[i]);
            if (!(fd_error < 1e-6)) {
                fprintf(stderr, "%s[%d,%d] = %.9g, finite difference = %.9g\n",
                        name, i / D, i % D, (double)grad[i], fd);
                return 1;
            }
            max_fd_error = fmax(max_fd_error, fd_error);
            largest = fmax(largest, fabs(fd));
        }
        printf("%s max magnitude %.9g, max trainer error %.9g, max finite difference error %.9g\n",
               name, largest, max_trainer_error, max_fd_error);
        if (largest < 1e-5) return 1;
    }
    return 0;
}
