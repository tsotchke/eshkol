#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "core/vector.h"
#include "core/memory.h"
#include "core/autodiff.h"

// Global arena for memory allocations
Arena* arena = NULL;

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
int vector_func(int);
int neural_net(int, int, int, int, int);
int mse_loss(int, int);
int main();

int v1 = vector_f_create_from_array(arena, (float[]){1, 2, 3}, 3);

int v2 = vector_f_create_from_array(arena, (float[]){4, 5, 6}, 3);

int v_sum = vector_f_add(arena, v1, v2);

int v_diff = vector_f_sub(arena, v1, v2);

int v_prod = vector_f_mul_scalar(arena, v1, v2);

int dot_prod = vector_f_dot(v1, v2);

int cross_prod = vector_f_cross(arena, v1, v2);

int v1_norm = vector_f_magnitude(v1);

int f(int v) {
    return ({ float int x = (v->data[0]);

; float int y = (v->data[1]);

; float int z = (v->data[2]);

; +((x * x), (y * y), (z * z)); });
}

int F(int v) {
    return ({ float int x = (v->data[0]);

; float int y = (v->data[1]);

; float int z = (v->data[2]);

; vector_f_create_from_array(arena, (float[]){(x * x), (y * y), (z * z)}, 3); });
}

int grad_f = compute_gradient(arena, f, v1);

int div_F = compute_divergence(arena, F, v1);

int curl_F = compute_curl(arena, F, v1);

int laplacian_f = compute_laplacian(arena, f, v1);

int g(int x) {
    return *(x, x, x);
}

int dg/dx = compute_nth_derivative(arena, g, 2, 1);

int h(int v) {
    return ({ float int x = (v->data[0]);

; float int y = (v->data[1]);

; (*(x, x, y) + (y * y)); });
}

int grad_h = compute_gradient(arena, h, vector_f_create_from_array(arena, (float[]){1, 2}, 2));

int vector_func(int v) {
    return ({ float int x = (v->data[0]);

; float int y = (v->data[1]);

; vector_f_create_from_array(arena, (float[]){(x * y), *(x, x, y)}, 2); });
}

int jacobian_matrix = jacobian(vector_func, vector_f_create_from_array(arena, (float[]){1, 2}, 2));

int neural_net(int x, int w1, int b1, int w2, int b2) {
    return let*(h1(tanh(((w1 * x) + b1)))(y(((w2 * h1) + b2))), y);
}

int mse_loss(int y-pred, int y-true) {
    return ({ float int diff = (y_pred - y_true);

; (diff * diff); });
}

int input = 1;

int target = 2;

int w1 = 0.5;

int b1 = 0.1;

int w2 = 0.3;

int b2 = 0.2;

int prediction = neural_net(input, w1, b1, w2, b2);

int loss = mse_loss(prediction, target);

int gradients = gradients(loss, 