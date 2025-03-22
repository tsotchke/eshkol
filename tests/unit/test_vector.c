/**
 * @file test_vector.c
 * @brief Unit tests for vector and matrix operations
 */

#include "core/vector.h"
#include "core/simd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

/**
 * @brief Test vector creation and initialization
 */
static void test_vector_create(void) {
    printf("Testing vector creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create vectors
    VectorF* vf = vector_f_create(arena, 3);
    assert(vf != NULL);
    assert(vf->dim == 3);
    assert(vf->data[0] == 0.0f && vf->data[1] == 0.0f && vf->data[2] == 0.0f);
    
    VectorD* vd = vector_d_create(arena, 4);
    assert(vd != NULL);
    assert(vd->dim == 4);
    assert(vd->data[0] == 0.0 && vd->data[1] == 0.0 && vd->data[2] == 0.0 && vd->data[3] == 0.0);
    
    VectorI* vi = vector_i_create(arena, 2);
    assert(vi != NULL);
    assert(vi->dim == 2);
    assert(vi->data[0] == 0 && vi->data[1] == 0);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: vector_create\n");
}

/**
 * @brief Test vector setting
 */
static void test_vector_set(void) {
    printf("Testing vector setting...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create vectors
    VectorF* vf = vector_f_create(arena, 3);
    assert(vf != NULL);
    
    VectorD* vd = vector_d_create(arena, 4);
    assert(vd != NULL);
    
    VectorI* vi = vector_i_create(arena, 2);
    assert(vi != NULL);
    
    // Set vectors
    float f_data[3] = {1.0f, 2.0f, 3.0f};
    assert(vector_f_set(vf, f_data, 3));
    assert(vf->data[0] == 1.0f && vf->data[1] == 2.0f && vf->data[2] == 3.0f);
    
    double d_data[4] = {1.0, 2.0, 3.0, 4.0};
    assert(vector_d_set(vd, d_data, 4));
    assert(vd->data[0] == 1.0 && vd->data[1] == 2.0 && vd->data[2] == 3.0 && vd->data[3] == 4.0);
    
    int i_data[2] = {1, 2};
    assert(vector_i_set(vi, i_data, 2));
    assert(vi->data[0] == 1 && vi->data[1] == 2);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: vector_set\n");
}

/**
 * @brief Test vector addition
 */
static void test_vector_add(void) {
    printf("Testing vector addition...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create vectors
    VectorF* a = vector_f_create(arena, 3);
    assert(a != NULL);
    
    VectorF* b = vector_f_create(arena, 3);
    assert(b != NULL);
    
    // Set vectors
    float a_data[3] = {1.0f, 2.0f, 3.0f};
    assert(vector_f_set(a, a_data, 3));
    
    float b_data[3] = {4.0f, 5.0f, 6.0f};
    assert(vector_f_set(b, b_data, 3));
    
    // Add vectors
    VectorF* c = vector_f_add(arena, a, b);
    assert(c != NULL);
    assert(c->dim == 3);
    assert(c->data[0] == 5.0f && c->data[1] == 7.0f && c->data[2] == 9.0f);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: vector_add\n");
}

/**
 * @brief Test vector subtraction
 */
static void test_vector_sub(void) {
    printf("Testing vector subtraction...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create vectors
    VectorF* a = vector_f_create(arena, 3);
    assert(a != NULL);
    
    VectorF* b = vector_f_create(arena, 3);
    assert(b != NULL);
    
    // Set vectors
    float a_data[3] = {4.0f, 5.0f, 6.0f};
    assert(vector_f_set(a, a_data, 3));
    
    float b_data[3] = {1.0f, 2.0f, 3.0f};
    assert(vector_f_set(b, b_data, 3));
    
    // Subtract vectors
    VectorF* c = vector_f_sub(arena, a, b);
    assert(c != NULL);
    assert(c->dim == 3);
    assert(c->data[0] == 3.0f && c->data[1] == 3.0f && c->data[2] == 3.0f);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: vector_sub\n");
}

/**
 * @brief Test vector scalar multiplication
 */
static void test_vector_mul_scalar(void) {
    printf("Testing vector scalar multiplication...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create vector
    VectorF* a = vector_f_create(arena, 3);
    assert(a != NULL);
    
    // Set vector
    float a_data[3] = {1.0f, 2.0f, 3.0f};
    assert(vector_f_set(a, a_data, 3));
    
    // Multiply vector by scalar
    VectorF* b = vector_f_mul_scalar(arena, a, 2.0f);
    assert(b != NULL);
    assert(b->dim == 3);
    assert(b->data[0] == 2.0f && b->data[1] == 4.0f && b->data[2] == 6.0f);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: vector_mul_scalar\n");
}

/**
 * @brief Test vector dot product
 */
static void test_vector_dot(void) {
    printf("Testing vector dot product...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create vectors
    VectorF* a = vector_f_create(arena, 3);
    assert(a != NULL);
    
    VectorF* b = vector_f_create(arena, 3);
    assert(b != NULL);
    
    // Set vectors
    float a_data[3] = {1.0f, 2.0f, 3.0f};
    assert(vector_f_set(a, a_data, 3));
    
    float b_data[3] = {4.0f, 5.0f, 6.0f};
    assert(vector_f_set(b, b_data, 3));
    
    // Compute dot product
    float dot = vector_f_dot(a, b);
    assert(dot == 32.0f); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: vector_dot\n");
}

/**
 * @brief Test vector cross product
 */
static void test_vector_cross(void) {
    printf("Testing vector cross product...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create vectors
    VectorF* a = vector_f_create(arena, 3);
    assert(a != NULL);
    
    VectorF* b = vector_f_create(arena, 3);
    assert(b != NULL);
    
    // Set vectors
    float a_data[3] = {1.0f, 0.0f, 0.0f};
    assert(vector_f_set(a, a_data, 3));
    
    float b_data[3] = {0.0f, 1.0f, 0.0f};
    assert(vector_f_set(b, b_data, 3));
    
    // Compute cross product
    VectorF* c = vector_f_cross(arena, a, b);
    assert(c != NULL);
    assert(c->dim == 3);
    assert(c->data[0] == 0.0f && c->data[1] == 0.0f && c->data[2] == 1.0f);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: vector_cross\n");
}

/**
 * @brief Test vector magnitude
 */
static void test_vector_magnitude(void) {
    printf("Testing vector magnitude...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create vector
    VectorF* a = vector_f_create(arena, 3);
    assert(a != NULL);
    
    // Set vector
    float a_data[3] = {3.0f, 4.0f, 0.0f};
    assert(vector_f_set(a, a_data, 3));
    
    // Compute magnitude
    float mag = vector_f_magnitude(a);
    assert(fabsf(mag - 5.0f) < 1e-6f);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: vector_magnitude\n");
}

/**
 * @brief Test vector normalization
 */
static void test_vector_normalize(void) {
    printf("Testing vector normalization...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create vector
    VectorF* a = vector_f_create(arena, 3);
    assert(a != NULL);
    
    // Set vector
    float a_data[3] = {3.0f, 4.0f, 0.0f};
    assert(vector_f_set(a, a_data, 3));
    
    // Normalize vector
    VectorF* b = vector_f_normalize(arena, a);
    assert(b != NULL);
    assert(b->dim == 3);
    assert(fabsf(b->data[0] - 0.6f) < 1e-6f && fabsf(b->data[1] - 0.8f) < 1e-6f && fabsf(b->data[2]) < 1e-6f);
    
    // Verify magnitude is 1
    float mag = vector_f_magnitude(b);
    assert(fabsf(mag - 1.0f) < 1e-6f);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: vector_normalize\n");
}

/**
 * @brief Test matrix creation and initialization
 */
static void test_matrix_create(void) {
    printf("Testing matrix creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create matrices
    MatrixF* mf = matrix_f_create(arena, 2, 3);
    assert(mf != NULL);
    assert(mf->rows == 2 && mf->cols == 3);
    assert(mf->data[0][0] == 0.0f && mf->data[0][1] == 0.0f && mf->data[0][2] == 0.0f);
    assert(mf->data[1][0] == 0.0f && mf->data[1][1] == 0.0f && mf->data[1][2] == 0.0f);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: matrix_create\n");
}

/**
 * @brief Test matrix setting
 */
static void test_matrix_set(void) {
    printf("Testing matrix setting...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create matrix
    MatrixF* mf = matrix_f_create(arena, 2, 3);
    assert(mf != NULL);
    
    // Set matrix
    float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    assert(matrix_f_set(mf, data, 2, 3));
    assert(mf->data[0][0] == 1.0f && mf->data[0][1] == 2.0f && mf->data[0][2] == 3.0f);
    assert(mf->data[1][0] == 4.0f && mf->data[1][1] == 5.0f && mf->data[1][2] == 6.0f);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: matrix_set\n");
}

/**
 * @brief Test matrix addition
 */
static void test_matrix_add(void) {
    printf("Testing matrix addition...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create matrices
    MatrixF* a = matrix_f_create(arena, 2, 2);
    assert(a != NULL);
    
    MatrixF* b = matrix_f_create(arena, 2, 2);
    assert(b != NULL);
    
    // Set matrices
    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    assert(matrix_f_set(a, a_data, 2, 2));
    
    float b_data[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    assert(matrix_f_set(b, b_data, 2, 2));
    
    // Add matrices
    MatrixF* c = matrix_f_add(arena, a, b);
    assert(c != NULL);
    assert(c->rows == 2 && c->cols == 2);
    assert(c->data[0][0] == 6.0f && c->data[0][1] == 8.0f);
    assert(c->data[1][0] == 10.0f && c->data[1][1] == 12.0f);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: matrix_add\n");
}

/**
 * @brief Test matrix subtraction
 */
static void test_matrix_sub(void) {
    printf("Testing matrix subtraction...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create matrices
    MatrixF* a = matrix_f_create(arena, 2, 2);
    assert(a != NULL);
    
    MatrixF* b = matrix_f_create(arena, 2, 2);
    assert(b != NULL);
    
    // Set matrices
    float a_data[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    assert(matrix_f_set(a, a_data, 2, 2));
    
    float b_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    assert(matrix_f_set(b, b_data, 2, 2));
    
    // Subtract matrices
    MatrixF* c = matrix_f_sub(arena, a, b);
    assert(c != NULL);
    assert(c->rows == 2 && c->cols == 2);
    assert(c->data[0][0] == 4.0f && c->data[0][1] == 4.0f);
    assert(c->data[1][0] == 4.0f && c->data[1][1] == 4.0f);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: matrix_sub\n");
}

/**
 * @brief Test matrix multiplication
 */
static void test_matrix_mul(void) {
    printf("Testing matrix multiplication...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create matrices
    MatrixF* a = matrix_f_create(arena, 2, 3);
    assert(a != NULL);
    
    MatrixF* b = matrix_f_create(arena, 3, 2);
    assert(b != NULL);
    
    // Set matrices
    float a_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    assert(matrix_f_set(a, a_data, 2, 3));
    
    float b_data[6] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    assert(matrix_f_set(b, b_data, 3, 2));
    
    // Multiply matrices
    MatrixF* c = matrix_f_mul(arena, a, b);
    assert(c != NULL);
    assert(c->rows == 2 && c->cols == 2);
    assert(c->data[0][0] == 58.0f && c->data[0][1] == 64.0f);
    assert(c->data[1][0] == 139.0f && c->data[1][1] == 154.0f);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: matrix_mul\n");
}

/**
 * @brief Test matrix scalar multiplication
 */
static void test_matrix_mul_scalar(void) {
    printf("Testing matrix scalar multiplication...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create matrix
    MatrixF* a = matrix_f_create(arena, 2, 2);
    assert(a != NULL);
    
    // Set matrix
    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    assert(matrix_f_set(a, a_data, 2, 2));
    
    // Multiply matrix by scalar
    MatrixF* b = matrix_f_mul_scalar(arena, a, 2.0f);
    assert(b != NULL);
    assert(b->rows == 2 && b->cols == 2);
    assert(b->data[0][0] == 2.0f && b->data[0][1] == 4.0f);
    assert(b->data[1][0] == 6.0f && b->data[1][1] == 8.0f);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: matrix_mul_scalar\n");
}

/**
 * @brief Test matrix transpose
 */
static void test_matrix_transpose(void) {
    printf("Testing matrix transpose...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create matrix
    MatrixF* a = matrix_f_create(arena, 2, 3);
    assert(a != NULL);
    
    // Set matrix
    float a_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    assert(matrix_f_set(a, a_data, 2, 3));
    
    // Transpose matrix
    MatrixF* b = matrix_f_transpose(arena, a);
    assert(b != NULL);
    assert(b->rows == 3 && b->cols == 2);
    assert(b->data[0][0] == 1.0f && b->data[0][1] == 4.0f);
    assert(b->data[1][0] == 2.0f && b->data[1][1] == 5.0f);
    assert(b->data[2][0] == 3.0f && b->data[2][1] == 6.0f);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: matrix_transpose\n");
}

/**
 * @brief Test matrix determinant
 */
static void test_matrix_determinant(void) {
    printf("Testing matrix determinant...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create matrices
    MatrixF* a = matrix_f_create(arena, 2, 2);
    assert(a != NULL);
    
    MatrixF* b = matrix_f_create(arena, 3, 3);
    assert(b != NULL);
    
    // Set matrices
    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    assert(matrix_f_set(a, a_data, 2, 2));
    
    float b_data[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    assert(matrix_f_set(b, b_data, 3, 3));
    
    // Compute determinants
    float det_a = matrix_f_determinant(a);
    assert(det_a == -2.0f); // 1*4 - 2*3 = 4 - 6 = -2
    
    float det_b = matrix_f_determinant(b);
    assert(fabsf(det_b) < 1e-6f); // Singular matrix
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: matrix_determinant\n");
}

/**
 * @brief Test matrix inverse
 */
static void test_matrix_inverse(void) {
    printf("Testing matrix inverse...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create matrix
    MatrixF* a = matrix_f_create(arena, 2, 2);
    assert(a != NULL);
    
    // Set matrix
    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    assert(matrix_f_set(a, a_data, 2, 2));
    
    // Compute inverse
    MatrixF* b = matrix_f_inverse(arena, a);
    assert(b != NULL);
    assert(b->rows == 2 && b->cols == 2);
    assert(fabsf(b->data[0][0] - -2.0f) < 1e-6f && fabsf(b->data[0][1] - 1.0f) < 1e-6f);
    assert(fabsf(b->data[1][0] - 1.5f) < 1e-6f && fabsf(b->data[1][1] - -0.5f) < 1e-6f);
    
    // Verify A * A^-1 = I
    MatrixF* c = matrix_f_mul(arena, a, b);
    assert(c != NULL);
    assert(c->rows == 2 && c->cols == 2);
    assert(fabsf(c->data[0][0] - 1.0f) < 1e-6f && fabsf(c->data[0][1]) < 1e-6f);
    assert(fabsf(c->data[1][0]) < 1e-6f && fabsf(c->data[1][1] - 1.0f) < 1e-6f);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: matrix_inverse\n");
}

/**
 * @brief Test matrix-vector multiplication
 */
static void test_matrix_mul_vector(void) {
    printf("Testing matrix-vector multiplication...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create matrix and vector
    MatrixF* a = matrix_f_create(arena, 2, 3);
    assert(a != NULL);
    
    VectorF* v = vector_f_create(arena, 3);
    assert(v != NULL);
    
    // Set matrix and vector
    float a_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    assert(matrix_f_set(a, a_data, 2, 3));
    
    float v_data[3] = {7.0f, 8.0f, 9.0f};
    assert(vector_f_set(v, v_data, 3));
    
    // Multiply matrix by vector
    VectorF* r = matrix_f_mul_vector(arena, a, v);
    assert(r != NULL);
    assert(r->dim == 2);
    assert(r->data[0] == 50.0f); // 1*7 + 2*8 + 3*9 = 7 + 16 + 27 = 50
    assert(r->data[1] == 122.0f); // 4*7 + 5*8 + 6*9 = 28 + 40 + 54 = 122
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: matrix_mul_vector\n");
}

/**
 * @brief Test identity matrix
 */
static void test_matrix_identity(void) {
    printf("Testing identity matrix...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create identity matrix
    MatrixF* a = matrix_f_identity(arena, 3);
    assert(a != NULL);
    assert(a->rows == 3 && a->cols == 3);
    assert(a->data[0][0] == 1.0f && a->data[0][1] == 0.0f && a->data[0][2] == 0.0f);
    assert(a->data[1][0] == 0.0f && a->data[1][1] == 1.0f && a->data[1][2] == 0.0f);
    assert(a->data[2][0] == 0.0f && a->data[2][1] == 0.0f && a->data[2][2] == 1.0f);
    
    // Create a vector
    VectorF* v = vector_f_create(arena, 3);
    assert(v != NULL);
    
    // Set vector
    float v_data[3] = {1.0f, 2.0f, 3.0f};
    assert(vector_f_set(v, v_data, 3));
    
    // Multiply identity matrix by vector
    VectorF* r = matrix_f_mul_vector(arena, a, v);
    assert(r != NULL);
    assert(r->dim == 3);
    assert(r->data[0] == 1.0f && r->data[1] == 2.0f && r->data[2] == 3.0f);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: matrix_identity\n");
}

/**
 * @brief Test SIMD detection
 */
static void test_simd_detection(void) {
    printf("Testing SIMD detection...\n");
    
    // Initialize SIMD detection
    simd_init();
    
    // Get SIMD information
    const SimdInfo* info = simd_get_info();
    assert(info != NULL);
    
    // Print SIMD information
    simd_print_info();
    
    printf("PASS: simd_detection\n");
}

/**
 * @brief Main function
 */
int main(void) {
    printf("Running vector and matrix tests...\n");
    
    // Test SIMD detection
    test_simd_detection();
    
    // Test vector operations
    test_vector_create();
    test_vector_set();
    test_vector_add();
    test_vector_sub();
    test_vector_mul_scalar();
    test_vector_dot();
    test_vector_cross();
    test_vector_magnitude();
    test_vector_normalize();
    
    // Test matrix operations
    test_matrix_create();
    test_matrix_set();
    test_matrix_add();
    test_matrix_sub();
    test_matrix_mul();
    test_matrix_mul_scalar();
    test_matrix_transpose();
    test_matrix_determinant();
    test_matrix_inverse();
    test_matrix_mul_vector();
    test_matrix_identity();
    
    printf("All vector and matrix tests passed!\n");
    return 0;
}
