/**
 * @file vector.h
 * @brief Vector and matrix data structures for Eshkol
 * 
 * This file defines vector and matrix data structures with proper memory alignment
 * for efficient SIMD operations.
 */

#ifndef ESHKOL_VECTOR_H
#define ESHKOL_VECTOR_H

#include "core/memory.h"
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Alignment for SIMD operations
 * 
 * This is set to 32 bytes to support AVX/AVX2 operations.
 */
#define VECTOR_ALIGNMENT 32

/**
 * @brief Maximum vector dimension
 * 
 * This is set to 4 for common vector operations (x, y, z, w).
 */
#define VECTOR_MAX_DIM 4

/**
 * @brief Vector of floats
 * 
 * This structure represents a vector of floats with proper memory alignment.
 */
typedef struct {
    float data[VECTOR_MAX_DIM] __attribute__((aligned(VECTOR_ALIGNMENT)));
    size_t dim;
} VectorF;

/**
 * @brief Vector of doubles
 * 
 * This structure represents a vector of doubles with proper memory alignment.
 */
typedef struct {
    double data[VECTOR_MAX_DIM] __attribute__((aligned(VECTOR_ALIGNMENT)));
    size_t dim;
} VectorD;

/**
 * @brief Vector of integers
 * 
 * This structure represents a vector of integers with proper memory alignment.
 */
typedef struct {
    int data[VECTOR_MAX_DIM] __attribute__((aligned(VECTOR_ALIGNMENT)));
    size_t dim;
} VectorI;

/**
 * @brief Matrix of floats
 * 
 * This structure represents a matrix of floats with proper memory alignment.
 */
typedef struct {
    float data[VECTOR_MAX_DIM][VECTOR_MAX_DIM] __attribute__((aligned(VECTOR_ALIGNMENT)));
    size_t rows;
    size_t cols;
} MatrixF;

/**
 * @brief Matrix of doubles
 * 
 * This structure represents a matrix of doubles with proper memory alignment.
 */
typedef struct {
    double data[VECTOR_MAX_DIM][VECTOR_MAX_DIM] __attribute__((aligned(VECTOR_ALIGNMENT)));
    size_t rows;
    size_t cols;
} MatrixD;

/**
 * @brief Matrix of integers
 * 
 * This structure represents a matrix of integers with proper memory alignment.
 */
typedef struct {
    int data[VECTOR_MAX_DIM][VECTOR_MAX_DIM] __attribute__((aligned(VECTOR_ALIGNMENT)));
    size_t rows;
    size_t cols;
} MatrixI;

/**
 * @brief Create a new float vector
 * 
 * @param arena Arena to allocate from
 * @param dim Dimension of the vector
 * @return A new float vector, or NULL on failure
 */
VectorF* vector_f_create(Arena* arena, size_t dim);

/**
 * @brief Create a new double vector
 * 
 * @param arena Arena to allocate from
 * @param dim Dimension of the vector
 * @return A new double vector, or NULL on failure
 */
VectorD* vector_d_create(Arena* arena, size_t dim);

/**
 * @brief Create a new integer vector
 * 
 * @param arena Arena to allocate from
 * @param dim Dimension of the vector
 * @return A new integer vector, or NULL on failure
 */
VectorI* vector_i_create(Arena* arena, size_t dim);

/**
 * @brief Create a new float matrix
 * 
 * @param arena Arena to allocate from
 * @param rows Number of rows
 * @param cols Number of columns
 * @return A new float matrix, or NULL on failure
 */
MatrixF* matrix_f_create(Arena* arena, size_t rows, size_t cols);

/**
 * @brief Create a new double matrix
 * 
 * @param arena Arena to allocate from
 * @param rows Number of rows
 * @param cols Number of columns
 * @return A new double matrix, or NULL on failure
 */
MatrixD* matrix_d_create(Arena* arena, size_t rows, size_t cols);

/**
 * @brief Create a new integer matrix
 * 
 * @param arena Arena to allocate from
 * @param rows Number of rows
 * @param cols Number of columns
 * @return A new integer matrix, or NULL on failure
 */
MatrixI* matrix_i_create(Arena* arena, size_t rows, size_t cols);

/**
 * @brief Set a float vector from an array
 * 
 * @param vec The vector
 * @param data The array of floats
 * @param dim Dimension of the array
 * @return true if successful, false otherwise
 */
bool vector_f_set(VectorF* vec, const float* data, size_t dim);

/**
 * @brief Set a double vector from an array
 * 
 * @param vec The vector
 * @param data The array of doubles
 * @param dim Dimension of the array
 * @return true if successful, false otherwise
 */
bool vector_d_set(VectorD* vec, const double* data, size_t dim);

/**
 * @brief Set an integer vector from an array
 * 
 * @param vec The vector
 * @param data The array of integers
 * @param dim Dimension of the array
 * @return true if successful, false otherwise
 */
bool vector_i_set(VectorI* vec, const int* data, size_t dim);

/**
 * @brief Set a float matrix from an array
 * 
 * @param mat The matrix
 * @param data The array of floats (row-major order)
 * @param rows Number of rows
 * @param cols Number of columns
 * @return true if successful, false otherwise
 */
bool matrix_f_set(MatrixF* mat, const float* data, size_t rows, size_t cols);

/**
 * @brief Set a double matrix from an array
 * 
 * @param mat The matrix
 * @param data The array of doubles (row-major order)
 * @param rows Number of rows
 * @param cols Number of columns
 * @return true if successful, false otherwise
 */
bool matrix_d_set(MatrixD* mat, const double* data, size_t rows, size_t cols);

/**
 * @brief Set an integer matrix from an array
 * 
 * @param mat The matrix
 * @param data The array of integers (row-major order)
 * @param rows Number of rows
 * @param cols Number of columns
 * @return true if successful, false otherwise
 */
bool matrix_i_set(MatrixI* mat, const int* data, size_t rows, size_t cols);

/**
 * @brief Add two float vectors
 * 
 * @param arena Arena to allocate from
 * @param a First vector
 * @param b Second vector
 * @return Result vector, or NULL on failure
 */
VectorF* vector_f_add(Arena* arena, const VectorF* a, const VectorF* b);

/**
 * @brief Add two double vectors
 * 
 * @param arena Arena to allocate from
 * @param a First vector
 * @param b Second vector
 * @return Result vector, or NULL on failure
 */
VectorD* vector_d_add(Arena* arena, const VectorD* a, const VectorD* b);

/**
 * @brief Add two integer vectors
 * 
 * @param arena Arena to allocate from
 * @param a First vector
 * @param b Second vector
 * @return Result vector, or NULL on failure
 */
VectorI* vector_i_add(Arena* arena, const VectorI* a, const VectorI* b);

/**
 * @brief Subtract two float vectors
 * 
 * @param arena Arena to allocate from
 * @param a First vector
 * @param b Second vector
 * @return Result vector, or NULL on failure
 */
VectorF* vector_f_sub(Arena* arena, const VectorF* a, const VectorF* b);

/**
 * @brief Subtract two double vectors
 * 
 * @param arena Arena to allocate from
 * @param a First vector
 * @param b Second vector
 * @return Result vector, or NULL on failure
 */
VectorD* vector_d_sub(Arena* arena, const VectorD* a, const VectorD* b);

/**
 * @brief Subtract two integer vectors
 * 
 * @param arena Arena to allocate from
 * @param a First vector
 * @param b Second vector
 * @return Result vector, or NULL on failure
 */
VectorI* vector_i_sub(Arena* arena, const VectorI* a, const VectorI* b);

/**
 * @brief Multiply a float vector by a scalar
 * 
 * @param arena Arena to allocate from
 * @param vec The vector
 * @param scalar The scalar
 * @return Result vector, or NULL on failure
 */
VectorF* vector_f_mul_scalar(Arena* arena, const VectorF* vec, float scalar);

/**
 * @brief Multiply a double vector by a scalar
 * 
 * @param arena Arena to allocate from
 * @param vec The vector
 * @param scalar The scalar
 * @return Result vector, or NULL on failure
 */
VectorD* vector_d_mul_scalar(Arena* arena, const VectorD* vec, double scalar);

/**
 * @brief Multiply an integer vector by a scalar
 * 
 * @param arena Arena to allocate from
 * @param vec The vector
 * @param scalar The scalar
 * @return Result vector, or NULL on failure
 */
VectorI* vector_i_mul_scalar(Arena* arena, const VectorI* vec, int scalar);

/**
 * @brief Compute the dot product of two float vectors
 * 
 * @param a First vector
 * @param b Second vector
 * @return Dot product
 */
float vector_f_dot(const VectorF* a, const VectorF* b);

/**
 * @brief Compute the dot product of two double vectors
 * 
 * @param a First vector
 * @param b Second vector
 * @return Dot product
 */
double vector_d_dot(const VectorD* a, const VectorD* b);

/**
 * @brief Compute the dot product of two integer vectors
 * 
 * @param a First vector
 * @param b Second vector
 * @return Dot product
 */
int vector_i_dot(const VectorI* a, const VectorI* b);

/**
 * @brief Compute the cross product of two float vectors
 * 
 * @param arena Arena to allocate from
 * @param a First vector
 * @param b Second vector
 * @return Result vector, or NULL on failure
 */
VectorF* vector_f_cross(Arena* arena, const VectorF* a, const VectorF* b);

/**
 * @brief Compute the cross product of two double vectors
 * 
 * @param arena Arena to allocate from
 * @param a First vector
 * @param b Second vector
 * @return Result vector, or NULL on failure
 */
VectorD* vector_d_cross(Arena* arena, const VectorD* a, const VectorD* b);

/**
 * @brief Compute the cross product of two integer vectors
 * 
 * @param arena Arena to allocate from
 * @param a First vector
 * @param b Second vector
 * @return Result vector, or NULL on failure
 */
VectorI* vector_i_cross(Arena* arena, const VectorI* a, const VectorI* b);

/**
 * @brief Compute the magnitude of a float vector
 * 
 * @param vec The vector
 * @return Magnitude
 */
float vector_f_magnitude(const VectorF* vec);

/**
 * @brief Compute the magnitude of a double vector
 * 
 * @param vec The vector
 * @return Magnitude
 */
double vector_d_magnitude(const VectorD* vec);

/**
 * @brief Compute the magnitude of an integer vector
 * 
 * @param vec The vector
 * @return Magnitude
 */
float vector_i_magnitude(const VectorI* vec);

/**
 * @brief Normalize a float vector
 * 
 * @param arena Arena to allocate from
 * @param vec The vector
 * @return Normalized vector, or NULL on failure
 */
VectorF* vector_f_normalize(Arena* arena, const VectorF* vec);

/**
 * @brief Normalize a double vector
 * 
 * @param arena Arena to allocate from
 * @param vec The vector
 * @return Normalized vector, or NULL on failure
 */
VectorD* vector_d_normalize(Arena* arena, const VectorD* vec);

/**
 * @brief Add two float matrices
 * 
 * @param arena Arena to allocate from
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on failure
 */
MatrixF* matrix_f_add(Arena* arena, const MatrixF* a, const MatrixF* b);

/**
 * @brief Add two double matrices
 * 
 * @param arena Arena to allocate from
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on failure
 */
MatrixD* matrix_d_add(Arena* arena, const MatrixD* a, const MatrixD* b);

/**
 * @brief Add two integer matrices
 * 
 * @param arena Arena to allocate from
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on failure
 */
MatrixI* matrix_i_add(Arena* arena, const MatrixI* a, const MatrixI* b);

/**
 * @brief Subtract two float matrices
 * 
 * @param arena Arena to allocate from
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on failure
 */
MatrixF* matrix_f_sub(Arena* arena, const MatrixF* a, const MatrixF* b);

/**
 * @brief Subtract two double matrices
 * 
 * @param arena Arena to allocate from
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on failure
 */
MatrixD* matrix_d_sub(Arena* arena, const MatrixD* a, const MatrixD* b);

/**
 * @brief Subtract two integer matrices
 * 
 * @param arena Arena to allocate from
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on failure
 */
MatrixI* matrix_i_sub(Arena* arena, const MatrixI* a, const MatrixI* b);

/**
 * @brief Multiply two float matrices
 * 
 * @param arena Arena to allocate from
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on failure
 */
MatrixF* matrix_f_mul(Arena* arena, const MatrixF* a, const MatrixF* b);

/**
 * @brief Multiply two double matrices
 * 
 * @param arena Arena to allocate from
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on failure
 */
MatrixD* matrix_d_mul(Arena* arena, const MatrixD* a, const MatrixD* b);

/**
 * @brief Multiply two integer matrices
 * 
 * @param arena Arena to allocate from
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on failure
 */
MatrixI* matrix_i_mul(Arena* arena, const MatrixI* a, const MatrixI* b);

/**
 * @brief Multiply a float matrix by a scalar
 * 
 * @param arena Arena to allocate from
 * @param mat The matrix
 * @param scalar The scalar
 * @return Result matrix, or NULL on failure
 */
MatrixF* matrix_f_mul_scalar(Arena* arena, const MatrixF* mat, float scalar);

/**
 * @brief Multiply a double matrix by a scalar
 * 
 * @param arena Arena to allocate from
 * @param mat The matrix
 * @param scalar The scalar
 * @return Result matrix, or NULL on failure
 */
MatrixD* matrix_d_mul_scalar(Arena* arena, const MatrixD* mat, double scalar);

/**
 * @brief Multiply an integer matrix by a scalar
 * 
 * @param arena Arena to allocate from
 * @param mat The matrix
 * @param scalar The scalar
 * @return Result matrix, or NULL on failure
 */
MatrixI* matrix_i_mul_scalar(Arena* arena, const MatrixI* mat, int scalar);

/**
 * @brief Transpose a float matrix
 * 
 * @param arena Arena to allocate from
 * @param mat The matrix
 * @return Transposed matrix, or NULL on failure
 */
MatrixF* matrix_f_transpose(Arena* arena, const MatrixF* mat);

/**
 * @brief Transpose a double matrix
 * 
 * @param arena Arena to allocate from
 * @param mat The matrix
 * @return Transposed matrix, or NULL on failure
 */
MatrixD* matrix_d_transpose(Arena* arena, const MatrixD* mat);

/**
 * @brief Transpose an integer matrix
 * 
 * @param arena Arena to allocate from
 * @param mat The matrix
 * @return Transposed matrix, or NULL on failure
 */
MatrixI* matrix_i_transpose(Arena* arena, const MatrixI* mat);

/**
 * @brief Compute the determinant of a float matrix
 * 
 * @param mat The matrix
 * @return Determinant
 */
float matrix_f_determinant(const MatrixF* mat);

/**
 * @brief Compute the determinant of a double matrix
 * 
 * @param mat The matrix
 * @return Determinant
 */
double matrix_d_determinant(const MatrixD* mat);

/**
 * @brief Compute the determinant of an integer matrix
 * 
 * @param mat The matrix
 * @return Determinant
 */
int matrix_i_determinant(const MatrixI* mat);

/**
 * @brief Compute the inverse of a float matrix
 * 
 * @param arena Arena to allocate from
 * @param mat The matrix
 * @return Inverse matrix, or NULL on failure
 */
MatrixF* matrix_f_inverse(Arena* arena, const MatrixF* mat);

/**
 * @brief Compute the inverse of a double matrix
 * 
 * @param arena Arena to allocate from
 * @param mat The matrix
 * @return Inverse matrix, or NULL on failure
 */
MatrixD* matrix_d_inverse(Arena* arena, const MatrixD* mat);

/**
 * @brief Multiply a float matrix by a float vector
 * 
 * @param arena Arena to allocate from
 * @param mat The matrix
 * @param vec The vector
 * @return Result vector, or NULL on failure
 */
VectorF* matrix_f_mul_vector(Arena* arena, const MatrixF* mat, const VectorF* vec);

/**
 * @brief Multiply a double matrix by a double vector
 * 
 * @param arena Arena to allocate from
 * @param mat The matrix
 * @param vec The vector
 * @return Result vector, or NULL on failure
 */
VectorD* matrix_d_mul_vector(Arena* arena, const MatrixD* mat, const VectorD* vec);

/**
 * @brief Multiply an integer matrix by an integer vector
 * 
 * @param arena Arena to allocate from
 * @param mat The matrix
 * @param vec The vector
 * @return Result vector, or NULL on failure
 */
VectorI* matrix_i_mul_vector(Arena* arena, const MatrixI* mat, const VectorI* vec);

/**
 * @brief Create a float identity matrix
 * 
 * @param arena Arena to allocate from
 * @param dim Dimension of the matrix
 * @return Identity matrix, or NULL on failure
 */
MatrixF* matrix_f_identity(Arena* arena, size_t dim);

/**
 * @brief Create a double identity matrix
 * 
 * @param arena Arena to allocate from
 * @param dim Dimension of the matrix
 * @return Identity matrix, or NULL on failure
 */
MatrixD* matrix_d_identity(Arena* arena, size_t dim);

/**
 * @brief Create an integer identity matrix
 * 
 * @param arena Arena to allocate from
 * @param dim Dimension of the matrix
 * @return Identity matrix, or NULL on failure
 */
MatrixI* matrix_i_identity(Arena* arena, size_t dim);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_VECTOR_H */
