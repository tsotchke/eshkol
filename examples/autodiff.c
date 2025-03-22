#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "core/memory.h"
#include "core/vector.h"
#include "core/autodiff.h"

// Example function: f(x) = x^2
static float square(VectorF* x) {
    return x->data[0] * x->data[0];
}

// Example function: f(x, y) = x^2 + y^2
static float sum_of_squares(VectorF* x) {
    return x->data[0] * x->data[0] + x->data[1] * x->data[1];
}

// Example function: f(x, y) = sin(x) * cos(y)
static float sin_cos(VectorF* x) {
    return sinf(x->data[0]) * cosf(x->data[1]);
}

// Example vector function: f(x, y) = [x^2, y^2]
static VectorF* vector_function(Arena* arena, VectorF* x) {
    VectorF* result = vector_f_create(arena, 2);
    if (!result) return NULL;
    
    result->data[0] = x->data[0] * x->data[0];
    result->data[1] = x->data[1] * x->data[1];
    
    return result;
}

int main(int argc __attribute__((unused)), char** argv __attribute__((unused))) {
    // Create memory arena
    Arena* arena = arena_create(1024 * 1024);
    if (!arena) {
        fprintf(stderr, "Failed to create memory arena\n");
        return 1;
    }
    
    printf("Automatic Differentiation Example\n");
    printf("=================================\n\n");
    
    // Example 1: Compute the derivative of f(x) = x^2 at x = 3
    {
        printf("Example 1: Derivative of f(x) = x^2 at x = 3\n");
        
        // Create input vector
        VectorF* x = vector_f_create(arena, 1);
        if (!x) {
            fprintf(stderr, "Failed to create vector\n");
            arena_destroy(arena);
            return 1;
        }
        x->data[0] = 3.0f;
        
        // Compute gradient
        VectorF* grad = compute_gradient_autodiff(arena, square, x);
        if (!grad) {
            fprintf(stderr, "Failed to compute gradient\n");
            arena_destroy(arena);
            return 1;
        }
        
        printf("f(x) = x^2 at x = %f\n", x->data[0]);
        printf("f'(x) = %f\n", grad->data[0]);
        printf("Expected: f'(x) = 2x = %f\n\n", 2.0f * x->data[0]);
    }
    
    // Example 2: Compute the gradient of f(x, y) = x^2 + y^2 at (x, y) = (2, 3)
    {
        printf("Example 2: Gradient of f(x, y) = x^2 + y^2 at (x, y) = (2, 3)\n");
        
        // Create input vector
        VectorF* x = vector_f_create(arena, 2);
        if (!x) {
            fprintf(stderr, "Failed to create vector\n");
            arena_destroy(arena);
            return 1;
        }
        x->data[0] = 2.0f;
        x->data[1] = 3.0f;
        
        // Compute gradient
        VectorF* grad = compute_gradient_autodiff(arena, sum_of_squares, x);
        if (!grad) {
            fprintf(stderr, "Failed to compute gradient\n");
            arena_destroy(arena);
            return 1;
        }
        
        printf("f(x, y) = x^2 + y^2 at (x, y) = (%f, %f)\n", x->data[0], x->data[1]);
        printf("∇f(x, y) = [%f, %f]\n", grad->data[0], grad->data[1]);
        printf("Expected: ∇f(x, y) = [2x, 2y] = [%f, %f]\n\n", 
               2.0f * x->data[0], 2.0f * x->data[1]);
    }
    
    // Example 3: Compute the gradient of f(x, y) = sin(x) * cos(y) at (x, y) = (1, 2)
    {
        printf("Example 3: Gradient of f(x, y) = sin(x) * cos(y) at (x, y) = (1, 2)\n");
        
        // Create input vector
        VectorF* x = vector_f_create(arena, 2);
        if (!x) {
            fprintf(stderr, "Failed to create vector\n");
            arena_destroy(arena);
            return 1;
        }
        x->data[0] = 1.0f;
        x->data[1] = 2.0f;
        
        // Compute gradient
        VectorF* grad = compute_gradient_autodiff(arena, sin_cos, x);
        if (!grad) {
            fprintf(stderr, "Failed to compute gradient\n");
            arena_destroy(arena);
            return 1;
        }
        
        printf("f(x, y) = sin(x) * cos(y) at (x, y) = (%f, %f)\n", x->data[0], x->data[1]);
        printf("∇f(x, y) = [%f, %f]\n", grad->data[0], grad->data[1]);
        printf("Expected: ∇f(x, y) = [cos(x) * cos(y), -sin(x) * sin(y)] = [%f, %f]\n\n", 
               cosf(x->data[0]) * cosf(x->data[1]), -sinf(x->data[0]) * sinf(x->data[1]));
    }
    
    // Example 4: Compute the Jacobian of f(x, y) = [x^2, y^2] at (x, y) = (2, 3)
    {
        printf("Example 4: Jacobian of f(x, y) = [x^2, y^2] at (x, y) = (2, 3)\n");
        
        // Create input vector
        VectorF* x = vector_f_create(arena, 2);
        if (!x) {
            fprintf(stderr, "Failed to create vector\n");
            arena_destroy(arena);
            return 1;
        }
        x->data[0] = 2.0f;
        x->data[1] = 3.0f;
        
        // Compute Jacobian
        VectorF** jacobian = compute_jacobian(arena, vector_function, x);
        if (!jacobian) {
            fprintf(stderr, "Failed to compute Jacobian\n");
            arena_destroy(arena);
            return 1;
        }
        
        printf("f(x, y) = [x^2, y^2] at (x, y) = (%f, %f)\n", x->data[0], x->data[1]);
        printf("J = [%f, %f; %f, %f]\n", 
               jacobian[0]->data[0], jacobian[0]->data[1],
               jacobian[1]->data[0], jacobian[1]->data[1]);
        printf("Expected: J = [2x, 0; 0, 2y] = [%f, %f; %f, %f]\n\n", 
               2.0f * x->data[0], 0.0f, 0.0f, 2.0f * x->data[1]);
    }
    
    // Example 5: Compute the Hessian of f(x, y) = x^2 + y^2 at (x, y) = (2, 3)
    {
        printf("Example 5: Hessian of f(x, y) = x^2 + y^2 at (x, y) = (2, 3)\n");
        
        // Create input vector
        VectorF* x = vector_f_create(arena, 2);
        if (!x) {
            fprintf(stderr, "Failed to create vector\n");
            arena_destroy(arena);
            return 1;
        }
        x->data[0] = 2.0f;
        x->data[1] = 3.0f;
        
        // Compute Hessian
        VectorF** hessian = compute_hessian(arena, sum_of_squares, x);
        if (!hessian) {
            fprintf(stderr, "Failed to compute Hessian\n");
            arena_destroy(arena);
            return 1;
        }
        
        printf("f(x, y) = x^2 + y^2 at (x, y) = (%f, %f)\n", x->data[0], x->data[1]);
        printf("H = [%f, %f; %f, %f]\n", 
               hessian[0]->data[0], hessian[0]->data[1],
               hessian[1]->data[0], hessian[1]->data[1]);
        printf("Expected: H = [2, 0; 0, 2]\n");
    }
    
    // Clean up
    arena_destroy(arena);
    
    return 0;
}
