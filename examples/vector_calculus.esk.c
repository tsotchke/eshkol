#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "core/vector.h"
#include "core/memory.h"

// Eshkol value type
typedef union {
    long integer;
    double floating;
    bool boolean;
    char character;
    char* string;
    void* pointer;
} eshkol_value_t;

// Forward declarations
int f(int);
int F(int);
int g(int);
int h(int);
int vector-func(int);
int neural-net(int, int, int, int, int);
int mse-loss(int, int);
int main();

v1 = vector_f_create_from_array(arena, (float[]){1, 2, 3}, 3);

v2 = vector_f_create_from_array(arena, (float[]){4, 5, 6}, 3);

v-sum = vector_f_add(arena, v1, v2);

v-diff = vector_f_sub(arena, v1, v2);

v-prod = vector_f_mul_scalar(arena, v1, v2);

dot-prod = vector_f_dot(v1, v2);

cross-prod = vector_f_cross(arena, v1, v2);

v1-norm = vector_f_magnitude(v1);

int f(int v) {
    return ({ eshkol_value_t x = vector-ref(v, 0);

; eshkol_value_t y = vector-ref(v, 1);

; eshkol_value_t z = vector-ref(v, 2);

; +((x * x), (y * y), (z * z)); });
}

int F(int v) {
    return ({ eshkol_value_t x = vector-ref(v, 0);

; eshkol_value_t y = vector-ref(v, 1);

; eshkol_value_t z = vector-ref(v, 2);

; vector_f_create_from_array(arena, (float[]){(x * x), (y * y), (z * z)}, 3); });
}

grad-f = compute_gradient(arena, f, v1);

div-F = compute_divergence(arena, F, v1);

curl-F = compute_curl(arena, F, v1);

laplacian-f = compute_laplacian(arena, f, v1);

int g(int x) {
    return *(x, x, x);
}

dg/dx = derivative(g, 2);

int h(int v) {
    return ({ eshkol_value_t x = vector-ref(v, 0);

; eshkol_value_t y = vector-ref(v, 1);

; (*(x, x, y) + (y * y)); });
}

grad-h = compute_gradient(arena, h, vector_f_create_from_array(arena, (float[]){1, 2}, 2));

int vector-func(int v) {
    return ({ eshkol_value_t x = vector-ref(v, 0);

; eshkol_value_t y = vector-ref(v, 1);

; vector_f_create_from_array(arena, (float[]){(x * y), *(x, x, y)}, 2); });
}

jacobian-matrix = jacobian(vector-func, vector_f_create_from_array(arena, (float[]){1, 2}, 2));

int neural-net(int x, int w1, int b1, int w2, int b2) {
    return let*(h1(tanh(((w1 * x) + b1)))(y(((w2 * h1) + b2))), y);
}

int mse-loss(int y-pred, int y-true) {
    return ({ eshkol_value_t diff = (y-pred - y-true);

; (diff * diff); });
}

input = 1;

target = 2;

w1 = 0.5;

b1 = 0.1;

w2 = 0.3;

b2 = 0.2;

prediction = neural-net(input, w1, b1, w2, b2);

loss = mse-loss(prediction, target);

gradients = gradients(loss, 