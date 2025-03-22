#include "core/autodiff.h"
#include "core/memory.h"
#include "core/vector.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Test function: f(x) = x^2
static float test_function_square(VectorF* x) {
    assert(x != NULL);
    assert(x->dim == 1);
    return x->data[0] * x->data[0];
}

// Test function: f(x, y) = x^2 + y^2
static float test_function_sum_of_squares(VectorF* x) {
    assert(x != NULL);
    assert(x->dim == 2);
    return x->data[0] * x->data[0] + x->data[1] * x->data[1];
}

// Test function: f(x, y) = sin(x) * cos(y)
static float test_function_sin_cos(VectorF* x) {
    assert(x != NULL);
    assert(x->dim == 2);
    return sinf(x->data[0]) * cosf(x->data[1]);
}

// Test vector function: f(x, y) = [x^2, y^2]
static VectorF* test_vector_function(Arena* arena, VectorF* x) {
    assert(arena != NULL);
    assert(x != NULL);
    assert(x->dim == 2);
    
    VectorF* result = vector_f_create(arena, 2);
    if (!result) return NULL;
    
    result->data[0] = x->data[0] * x->data[0];
    result->data[1] = x->data[1] * x->data[1];
    
    return result;
}

// Test dual number operations
static void test_dual_number_operations(void) {
    printf("Testing dual number operations...\n");
    
    // Test creation
    DualNumber a = dual_number_create(2.0f, 1.0f);
    DualNumber b = dual_number_create(3.0f, 0.5f);
    
    // Test addition
    DualNumber c = dual_number_add(a, b);
    assert(fabsf(c.value - 5.0f) < 1e-6f);
    assert(fabsf(c.derivative - 1.5f) < 1e-6f);
    
    // Test subtraction
    DualNumber d = dual_number_sub(a, b);
    assert(fabsf(d.value - (-1.0f)) < 1e-6f);
    assert(fabsf(d.derivative - 0.5f) < 1e-6f);
    
    // Test multiplication
    DualNumber e = dual_number_mul(a, b);
    assert(fabsf(e.value - 6.0f) < 1e-6f);
    assert(fabsf(e.derivative - 4.0f) < 1e-6f);  // a.derivative * b.value + a.value * b.derivative = 1.0 * 3.0 + 2.0 * 0.5 = 4.0
    
    // Test division
    DualNumber f = dual_number_div(a, b);
    assert(fabsf(f.value - (2.0f / 3.0f)) < 1e-6f);
    assert(fabsf(f.derivative - (1.0f * 3.0f - 2.0f * 0.5f) / (3.0f * 3.0f)) < 1e-6f);
    
    // Test sin
    DualNumber g = dual_number_sin(a);
    assert(fabsf(g.value - sinf(2.0f)) < 1e-6f);
    assert(fabsf(g.derivative - cosf(2.0f) * 1.0f) < 1e-6f);
    
    // Test cos
    DualNumber h = dual_number_cos(a);
    assert(fabsf(h.value - cosf(2.0f)) < 1e-6f);
    assert(fabsf(h.derivative - (-sinf(2.0f) * 1.0f)) < 1e-6f);
    
    // Test exp
    DualNumber i = dual_number_exp(a);
    assert(fabsf(i.value - expf(2.0f)) < 1e-6f);
    assert(fabsf(i.derivative - expf(2.0f) * 1.0f) < 1e-6f);
    
    // Test log
    DualNumber j = dual_number_log(a);
    assert(fabsf(j.value - logf(2.0f)) < 1e-6f);
    assert(fabsf(j.derivative - 1.0f / 2.0f) < 1e-6f);
    
    // Test pow
    DualNumber k = dual_number_pow(a, 3.0f);
    assert(fabsf(k.value - powf(2.0f, 3.0f)) < 1e-6f);
    assert(fabsf(k.derivative - 3.0f * powf(2.0f, 2.0f) * 1.0f) < 1e-6f);
    
    printf("Dual number operations tests passed!\n");
}

// Test dual vector operations
static void test_dual_vector_operations(void) {
    printf("Testing dual vector operations...\n");
    
    // Create arena
    Arena* arena = arena_create(1024 * 1024);
    assert(arena != NULL);
    
    // Create vectors
    VectorF* a_value = vector_f_create(arena, 3);
    assert(a_value != NULL);
    a_value->data[0] = 1.0f;
    a_value->data[1] = 2.0f;
    a_value->data[2] = 3.0f;
    
    VectorF* a_derivative = vector_f_create(arena, 3);
    assert(a_derivative != NULL);
    a_derivative->data[0] = 0.5f;
    a_derivative->data[1] = 1.0f;
    a_derivative->data[2] = 1.5f;
    
    VectorF* b_value = vector_f_create(arena, 3);
    assert(b_value != NULL);
    b_value->data[0] = 4.0f;
    b_value->data[1] = 5.0f;
    b_value->data[2] = 6.0f;
    
    VectorF* b_derivative = vector_f_create(arena, 3);
    assert(b_derivative != NULL);
    b_derivative->data[0] = 2.0f;
    b_derivative->data[1] = 2.5f;
    b_derivative->data[2] = 3.0f;
    
    // Create dual vectors
    DualVector* a = dual_vector_create(arena, a_value, a_derivative);
    assert(a != NULL);
    
    DualVector* b = dual_vector_create(arena, b_value, b_derivative);
    assert(b != NULL);
    
    // Test addition
    DualVector* c = dual_vector_add(arena, a, b);
    assert(c != NULL);
    assert(fabsf(c->value->data[0] - 5.0f) < 1e-6f);
    assert(fabsf(c->value->data[1] - 7.0f) < 1e-6f);
    assert(fabsf(c->value->data[2] - 9.0f) < 1e-6f);
    assert(fabsf(c->derivative->data[0] - 2.5f) < 1e-6f);
    assert(fabsf(c->derivative->data[1] - 3.5f) < 1e-6f);
    assert(fabsf(c->derivative->data[2] - 4.5f) < 1e-6f);
    
    // Test subtraction
    DualVector* d = dual_vector_sub(arena, a, b);
    assert(d != NULL);
    assert(fabsf(d->value->data[0] - (-3.0f)) < 1e-6f);
    assert(fabsf(d->value->data[1] - (-3.0f)) < 1e-6f);
    assert(fabsf(d->value->data[2] - (-3.0f)) < 1e-6f);
    assert(fabsf(d->derivative->data[0] - (-1.5f)) < 1e-6f);
    assert(fabsf(d->derivative->data[1] - (-1.5f)) < 1e-6f);
    assert(fabsf(d->derivative->data[2] - (-1.5f)) < 1e-6f);
    
    // Test scalar multiplication
    DualVector* e = dual_vector_mul_scalar(arena, a, 2.0f);
    assert(e != NULL);
    assert(fabsf(e->value->data[0] - 2.0f) < 1e-6f);
    assert(fabsf(e->value->data[1] - 4.0f) < 1e-6f);
    assert(fabsf(e->value->data[2] - 6.0f) < 1e-6f);
    assert(fabsf(e->derivative->data[0] - 1.0f) < 1e-6f);
    assert(fabsf(e->derivative->data[1] - 2.0f) < 1e-6f);
    assert(fabsf(e->derivative->data[2] - 3.0f) < 1e-6f);
    
    // Test dot product
    DualNumber f = dual_vector_dot(a, b);
    assert(fabsf(f.value - (1.0f * 4.0f + 2.0f * 5.0f + 3.0f * 6.0f)) < 1e-6f);
    float expected_derivative = 
        0.5f * 4.0f + 1.0f * 5.0f + 1.5f * 6.0f + 
        1.0f * 2.0f + 2.0f * 2.5f + 3.0f * 3.0f;
    assert(fabsf(f.derivative - expected_derivative) < 1e-6f);
    
    // Test cross product
    DualVector* g = dual_vector_cross(arena, a, b);
    assert(g != NULL);
    assert(fabsf(g->value->data[0] - (2.0f * 6.0f - 3.0f * 5.0f)) < 1e-6f);
    assert(fabsf(g->value->data[1] - (3.0f * 4.0f - 1.0f * 6.0f)) < 1e-6f);
    assert(fabsf(g->value->data[2] - (1.0f * 5.0f - 2.0f * 4.0f)) < 1e-6f);
    
    // Test magnitude
    DualNumber h = dual_vector_magnitude(a);
    float magnitude = sqrtf(1.0f * 1.0f + 2.0f * 2.0f + 3.0f * 3.0f);
    assert(fabsf(h.value - magnitude) < 1e-6f);
    float expected_mag_derivative = 
        (1.0f * 0.5f + 2.0f * 1.0f + 3.0f * 1.5f) / magnitude;
    assert(fabsf(h.derivative - expected_mag_derivative) < 1e-6f);
    
    // Clean up
    arena_destroy(arena);
    
    printf("Dual vector operations tests passed!\n");
}

// Test gradient computation
static void test_gradient_computation(void) {
    printf("Testing gradient computation...\n");
    
    // Create arena
    Arena* arena = arena_create(1024 * 1024);
    assert(arena != NULL);
    
    // Test gradient of f(x) = x^2 at x = 3
    VectorF* x1 = vector_f_create(arena, 1);
    assert(x1 != NULL);
    x1->data[0] = 3.0f;
    
    VectorF* grad1 = compute_gradient_autodiff(arena, test_function_square, x1);
    assert(grad1 != NULL);
    assert(fabsf(grad1->data[0] - 6.0f) < 1e-6f);
    
    // Test gradient of f(x, y) = x^2 + y^2 at (2, 3)
    VectorF* x2 = vector_f_create(arena, 2);
    assert(x2 != NULL);
    x2->data[0] = 2.0f;
    x2->data[1] = 3.0f;
    
    VectorF* grad2 = compute_gradient_autodiff(arena, test_function_sum_of_squares, x2);
    assert(grad2 != NULL);
    assert(fabsf(grad2->data[0] - 4.0f) < 1e-6f);
    assert(fabsf(grad2->data[1] - 6.0f) < 1e-6f);
    
    // Test gradient of f(x, y) = sin(x) * cos(y) at (1, 2)
    VectorF* x3 = vector_f_create(arena, 2);
    assert(x3 != NULL);
    x3->data[0] = 1.0f;
    x3->data[1] = 2.0f;
    
    VectorF* grad3 = compute_gradient_autodiff(arena, test_function_sin_cos, x3);
    assert(grad3 != NULL);
    assert(fabsf(grad3->data[0] - cosf(1.0f) * cosf(2.0f)) < 1e-6f);
    assert(fabsf(grad3->data[1] - sinf(1.0f) * (-sinf(2.0f))) < 1e-6f);
    
    // Clean up
    arena_destroy(arena);
    
    printf("Gradient computation tests passed!\n");
}

// Test Jacobian computation
static void test_jacobian_computation(void) {
    printf("Testing Jacobian computation...\n");
    
    // Create arena
    Arena* arena = arena_create(1024 * 1024);
    assert(arena != NULL);
    
    // Test Jacobian of f(x, y) = [x^2, y^2] at (2, 3)
    VectorF* x = vector_f_create(arena, 2);
    assert(x != NULL);
    x->data[0] = 2.0f;
    x->data[1] = 3.0f;
    
    VectorF** jacobian = compute_jacobian(arena, test_vector_function, x);
    assert(jacobian != NULL);
    assert(jacobian[0] != NULL);
    assert(jacobian[1] != NULL);
    
    // J = [df1/dx, df1/dy; df2/dx, df2/dy] = [2x, 0; 0, 2y] = [4, 0; 0, 6]
    assert(fabsf(jacobian[0]->data[0] - 4.0f) < 1e-6f);
    assert(fabsf(jacobian[0]->data[1] - 0.0f) < 1e-6f);
    assert(fabsf(jacobian[1]->data[0] - 0.0f) < 1e-6f);
    assert(fabsf(jacobian[1]->data[1] - 6.0f) < 1e-6f);
    
    // Clean up
    arena_destroy(arena);
    
    printf("Jacobian computation tests passed!\n");
}

// Test Hessian computation
static void test_hessian_computation(void) {
    printf("Testing Hessian computation...\n");
    
    // Create arena
    Arena* arena = arena_create(1024 * 1024);
    assert(arena != NULL);
    
    // Test Hessian of f(x, y) = x^2 + y^2 at (x, y) = (2, 3)
    VectorF* x = vector_f_create(arena, 2);
    assert(x != NULL);
    x->data[0] = 2.0f;
    x->data[1] = 3.0f;
    
    VectorF** hessian = compute_hessian(arena, test_function_sum_of_squares, x);
    assert(hessian != NULL);
    assert(hessian[0] != NULL);
    assert(hessian[1] != NULL);
    
    // H = [d^2f/dx^2, d^2f/dxdy; d^2f/dydx, d^2f/dy^2] = [2, 0; 0, 2]
    assert(fabsf(hessian[0]->data[0] - 2.0f) < 1e-6f);
    assert(fabsf(hessian[0]->data[1] - 0.0f) < 1e-6f);
    assert(fabsf(hessian[1]->data[0] - 0.0f) < 1e-6f);
    assert(fabsf(hessian[1]->data[1] - 2.0f) < 1e-6f);
    
    // Clean up
    arena_destroy(arena);
    
    printf("Hessian computation tests passed!\n");
}

// Test reverse-mode automatic differentiation
static void test_reverse_mode_autodiff(void) {
    printf("Testing reverse-mode automatic differentiation...\n");
    
    // Create arena
    Arena* arena = arena_create(1024 * 1024);
    assert(arena != NULL);
    
    // Test gradient of f(x) = x^2 at x = 3
    {
        VectorF* x = vector_f_create(arena, 1);
        assert(x != NULL);
        x->data[0] = 3.0f;
        
        VectorF* grad = compute_gradient_reverse_mode(arena, test_function_square, x);
        assert(grad != NULL);
        
        printf("f(x) = x^2 at x = %f\n", x->data[0]);
        printf("f'(x) = %f (reverse-mode)\n", grad->data[0]);
        printf("Expected: f'(x) = 2x = %f\n\n", 2.0f * x->data[0]);
        
        assert(fabsf(grad->data[0] - 2.0f * x->data[0]) < 1e-6f);
    }
    
    // Test gradient of f(x, y) = x^2 + y^2 at (x, y) = (2, 3)
    {
        VectorF* x = vector_f_create(arena, 2);
        assert(x != NULL);
        x->data[0] = 2.0f;
        x->data[1] = 3.0f;
        
        VectorF* grad = compute_gradient_reverse_mode(arena, test_function_sum_of_squares, x);
        assert(grad != NULL);
        
        printf("f(x, y) = x^2 + y^2 at (x, y) = (%f, %f)\n", x->data[0], x->data[1]);
        printf("∇f(x, y) = [%f, %f] (reverse-mode)\n", grad->data[0], grad->data[1]);
        printf("Expected: ∇f(x, y) = [2x, 2y] = [%f, %f]\n\n", 
               2.0f * x->data[0], 2.0f * x->data[1]);
        
        assert(fabsf(grad->data[0] - 2.0f * x->data[0]) < 1e-6f);
        assert(fabsf(grad->data[1] - 2.0f * x->data[1]) < 1e-6f);
    }
    
    // Test gradient of f(x, y) = sin(x) * cos(y) at (x, y) = (1, 2)
    {
        VectorF* x = vector_f_create(arena, 2);
        assert(x != NULL);
        x->data[0] = 1.0f;
        x->data[1] = 2.0f;
        
        VectorF* grad = compute_gradient_reverse_mode(arena, test_function_sin_cos, x);
        assert(grad != NULL);
        
        printf("f(x, y) = sin(x) * cos(y) at (x, y) = (%f, %f)\n", x->data[0], x->data[1]);
        printf("∇f(x, y) = [%f, %f] (reverse-mode)\n", grad->data[0], grad->data[1]);
        printf("Expected: ∇f(x, y) = [cos(x) * cos(y), -sin(x) * sin(y)] = [%f, %f]\n\n", 
               cosf(x->data[0]) * cosf(x->data[1]), -sinf(x->data[0]) * sinf(x->data[1]));
        
        assert(fabsf(grad->data[0] - cosf(x->data[0]) * cosf(x->data[1])) < 1e-6f);
        assert(fabsf(grad->data[1] - (-sinf(x->data[0]) * sinf(x->data[1]))) < 1e-6f);
    }
    
    // Clean up
    arena_destroy(arena);
    
    printf("Reverse-mode automatic differentiation tests passed!\n");
}

int main(void) {
    printf("Running autodiff tests...\n");
    
    test_dual_number_operations();
    test_dual_vector_operations();
    test_gradient_computation();
    test_jacobian_computation();
    test_hessian_computation();
    test_reverse_mode_autodiff();
    
    printf("All autodiff tests passed!\n");
    return 0;
}
