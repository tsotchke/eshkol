/**
 * @file vector.c
 * @brief Implementation of vector and matrix operations
 */

#include "core/vector.h"
#include "core/simd.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

/**
 * @brief Create a new float vector
 * 
 * @param arena Arena to allocate from
 * @param dim Dimension of the vector
 * @return A new float vector, or NULL on failure
 */
VectorF* vector_f_create(Arena* arena, size_t dim) {
    assert(arena != NULL);
    assert(dim > 0 && dim <= VECTOR_MAX_DIM);
    
    VectorF* vec = arena_alloc_aligned(arena, sizeof(VectorF), VECTOR_ALIGNMENT);
    if (!vec) {
        return NULL;
    }
    
    vec->dim = dim;
    memset(vec->data, 0, sizeof(vec->data));
    
    return vec;
}

/**
 * @brief Create a new float vector from an array
 * 
 * @param arena Arena to allocate from
 * @param data Array of floats
 * @param dim Dimension of the vector
 * @return A new float vector, or NULL on failure
 */
VectorF* vector_f_create_from_array(Arena* arena, const float* data, size_t dim) {
    assert(arena != NULL);
    assert(data != NULL);
    assert(dim > 0 && dim <= VECTOR_MAX_DIM);
    
    VectorF* vec = vector_f_create(arena, dim);
    if (!vec) {
        return NULL;
    }
    
    memcpy(vec->data, data, dim * sizeof(float));
    
    return vec;
}

/**
 * @brief Create a new double vector
 * 
 * @param arena Arena to allocate from
 * @param dim Dimension of the vector
 * @return A new double vector, or NULL on failure
 */
VectorD* vector_d_create(Arena* arena, size_t dim) {
    assert(arena != NULL);
    assert(dim > 0 && dim <= VECTOR_MAX_DIM);
    
    VectorD* vec = arena_alloc_aligned(arena, sizeof(VectorD), VECTOR_ALIGNMENT);
    if (!vec) {
        return NULL;
    }
    
    vec->dim = dim;
    memset(vec->data, 0, sizeof(vec->data));
    
    return vec;
}

/**
 * @brief Create a new integer vector
 * 
 * @param arena Arena to allocate from
 * @param dim Dimension of the vector
 * @return A new integer vector, or NULL on failure
 */
VectorI* vector_i_create(Arena* arena, size_t dim) {
    assert(arena != NULL);
    assert(dim > 0 && dim <= VECTOR_MAX_DIM);
    
    VectorI* vec = arena_alloc_aligned(arena, sizeof(VectorI), VECTOR_ALIGNMENT);
    if (!vec) {
        return NULL;
    }
    
    vec->dim = dim;
    memset(vec->data, 0, sizeof(vec->data));
    
    return vec;
}

/**
 * @brief Set a float vector from an array
 * 
 * @param vec The vector
 * @param data The array of floats
 * @param dim Dimension of the array
 * @return true if successful, false otherwise
 */
bool vector_f_set(VectorF* vec, const float* data, size_t dim) {
    assert(vec != NULL);
    assert(data != NULL);
    assert(dim > 0 && dim <= VECTOR_MAX_DIM);
    
    if (dim > vec->dim) {
        return false;
    }
    
    memcpy(vec->data, data, dim * sizeof(float));
    vec->dim = dim;
    
    return true;
}

/**
 * @brief Set a double vector from an array
 * 
 * @param vec The vector
 * @param data The array of doubles
 * @param dim Dimension of the array
 * @return true if successful, false otherwise
 */
bool vector_d_set(VectorD* vec, const double* data, size_t dim) {
    assert(vec != NULL);
    assert(data != NULL);
    assert(dim > 0 && dim <= VECTOR_MAX_DIM);
    
    if (dim > vec->dim) {
        return false;
    }
    
    memcpy(vec->data, data, dim * sizeof(double));
    vec->dim = dim;
    
    return true;
}

/**
 * @brief Set an integer vector from an array
 * 
 * @param vec The vector
 * @param data The array of integers
 * @param dim Dimension of the array
 * @return true if successful, false otherwise
 */
bool vector_i_set(VectorI* vec, const int* data, size_t dim) {
    assert(vec != NULL);
    assert(data != NULL);
    assert(dim > 0 && dim <= VECTOR_MAX_DIM);
    
    if (dim > vec->dim) {
        return false;
    }
    
    memcpy(vec->data, data, dim * sizeof(int));
    vec->dim = dim;
    
    return true;
}

/**
 * @brief Generic implementation of float vector addition
 * 
 * @param result Result vector
 * @param a First vector
 * @param b Second vector
 */
static void vector_f_add_generic(VectorF* result, const VectorF* a, const VectorF* b) {
    assert(result != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(a->dim == b->dim);
    assert(result->dim >= a->dim);
    
    for (size_t i = 0; i < a->dim; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
}

#if defined(__SSE__) || defined(__SSE2__)
/**
 * @brief SSE implementation of float vector addition
 * 
 * @param result Result vector
 * @param a First vector
 * @param b Second vector
 */
static void vector_f_add_sse(VectorF* result, const VectorF* a, const VectorF* b) {
    assert(result != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(a->dim == b->dim);
    assert(result->dim >= a->dim);
    
    // Use SSE instructions for 4D vectors
    if (a->dim == 4) {
        __m128 va = _mm_load_ps(a->data);
        __m128 vb = _mm_load_ps(b->data);
        __m128 vr = _mm_add_ps(va, vb);
        _mm_store_ps(result->data, vr);
    } else {
        // Fall back to generic implementation for other dimensions
        vector_f_add_generic(result, a, b);
    }
}
#endif

#if defined(__AVX__) || defined(__AVX2__)
/**
 * @brief AVX implementation of float vector addition
 * 
 * @param result Result vector
 * @param a First vector
 * @param b Second vector
 */
static void vector_f_add_avx(VectorF* result, const VectorF* a, const VectorF* b) {
    assert(result != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(a->dim == b->dim);
    assert(result->dim >= a->dim);
    
    // Use AVX instructions for 4D vectors
    if (a->dim == 4) {
        __m128 va = _mm_load_ps(a->data);
        __m128 vb = _mm_load_ps(b->data);
        __m128 vr = _mm_add_ps(va, vb);
        _mm_store_ps(result->data, vr);
    } else {
        // Fall back to generic implementation for other dimensions
        vector_f_add_generic(result, a, b);
    }
}
#endif

/**
 * @brief Add two float vectors
 * 
 * @param arena Arena to allocate from
 * @param a First vector
 * @param b Second vector
 * @return Result vector, or NULL on failure
 */
VectorF* vector_f_add(Arena* arena, const VectorF* a, const VectorF* b) {
    assert(arena != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(a->dim == b->dim);
    
    VectorF* result = vector_f_create(arena, a->dim);
    if (!result) {
        return NULL;
    }
    
    // Get the best implementation
    typedef void (*AddFunc)(VectorF*, const VectorF*, const VectorF*);
    AddFunc add_func = (AddFunc)simd_get_best_impl(
        (void*)vector_f_add_generic,
#if defined(__SSE__) || defined(__SSE2__)
        (void*)vector_f_add_sse,
#else
        NULL,
#endif
#if defined(__AVX__) || defined(__AVX2__)
        (void*)vector_f_add_avx,
#else
        NULL,
#endif
        NULL,
        NULL,
        NULL
    );
    
    // Add the vectors
    add_func(result, a, b);
    
    return result;
}

/**
 * @brief Generic implementation of float vector subtraction
 * 
 * @param result Result vector
 * @param a First vector
 * @param b Second vector
 */
static void vector_f_sub_generic(VectorF* result, const VectorF* a, const VectorF* b) {
    assert(result != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(a->dim == b->dim);
    assert(result->dim >= a->dim);
    
    for (size_t i = 0; i < a->dim; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
}

#if defined(__SSE__) || defined(__SSE2__)
/**
 * @brief SSE implementation of float vector subtraction
 * 
 * @param result Result vector
 * @param a First vector
 * @param b Second vector
 */
static void vector_f_sub_sse(VectorF* result, const VectorF* a, const VectorF* b) {
    assert(result != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(a->dim == b->dim);
    assert(result->dim >= a->dim);
    
    // Use SSE instructions for 4D vectors
    if (a->dim == 4) {
        __m128 va = _mm_load_ps(a->data);
        __m128 vb = _mm_load_ps(b->data);
        __m128 vr = _mm_sub_ps(va, vb);
        _mm_store_ps(result->data, vr);
    } else {
        // Fall back to generic implementation for other dimensions
        vector_f_sub_generic(result, a, b);
    }
}
#endif

#if defined(__AVX__) || defined(__AVX2__)
/**
 * @brief AVX implementation of float vector subtraction
 * 
 * @param result Result vector
 * @param a First vector
 * @param b Second vector
 */
static void vector_f_sub_avx(VectorF* result, const VectorF* a, const VectorF* b) {
    assert(result != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(a->dim == b->dim);
    assert(result->dim >= a->dim);
    
    // Use AVX instructions for 4D vectors
    if (a->dim == 4) {
        __m128 va = _mm_load_ps(a->data);
        __m128 vb = _mm_load_ps(b->data);
        __m128 vr = _mm_sub_ps(va, vb);
        _mm_store_ps(result->data, vr);
    } else {
        // Fall back to generic implementation for other dimensions
        vector_f_sub_generic(result, a, b);
    }
}
#endif

/**
 * @brief Subtract two float vectors
 * 
 * @param arena Arena to allocate from
 * @param a First vector
 * @param b Second vector
 * @return Result vector, or NULL on failure
 */
VectorF* vector_f_sub(Arena* arena, const VectorF* a, const VectorF* b) {
    assert(arena != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(a->dim == b->dim);
    
    VectorF* result = vector_f_create(arena, a->dim);
    if (!result) {
        return NULL;
    }
    
    // Get the best implementation
    typedef void (*SubFunc)(VectorF*, const VectorF*, const VectorF*);
    SubFunc sub_func = (SubFunc)simd_get_best_impl(
        (void*)vector_f_sub_generic,
#if defined(__SSE__) || defined(__SSE2__)
        (void*)vector_f_sub_sse,
#else
        NULL,
#endif
#if defined(__AVX__) || defined(__AVX2__)
        (void*)vector_f_sub_avx,
#else
        NULL,
#endif
        NULL,
        NULL,
        NULL
    );
    
    // Subtract the vectors
    sub_func(result, a, b);
    
    return result;
}

/**
 * @brief Generic implementation of float vector scalar multiplication
 * 
 * @param result Result vector
 * @param vec The vector
 * @param scalar The scalar
 */
static void vector_f_mul_scalar_generic(VectorF* result, const VectorF* vec, float scalar) {
    assert(result != NULL);
    assert(vec != NULL);
    assert(result->dim >= vec->dim);
    
    for (size_t i = 0; i < vec->dim; i++) {
        result->data[i] = vec->data[i] * scalar;
    }
}

#if defined(__SSE__) || defined(__SSE2__)
/**
 * @brief SSE implementation of float vector scalar multiplication
 * 
 * @param result Result vector
 * @param vec The vector
 * @param scalar The scalar
 */
static void vector_f_mul_scalar_sse(VectorF* result, const VectorF* vec, float scalar) {
    assert(result != NULL);
    assert(vec != NULL);
    assert(result->dim >= vec->dim);
    
    // Use SSE instructions for 4D vectors
    if (vec->dim == 4) {
        __m128 vv = _mm_load_ps(vec->data);
        __m128 vs = _mm_set1_ps(scalar);
        __m128 vr = _mm_mul_ps(vv, vs);
        _mm_store_ps(result->data, vr);
    } else {
        // Fall back to generic implementation for other dimensions
        vector_f_mul_scalar_generic(result, vec, scalar);
    }
}
#endif

#if defined(__AVX__) || defined(__AVX2__)
/**
 * @brief AVX implementation of float vector scalar multiplication
 * 
 * @param result Result vector
 * @param vec The vector
 * @param scalar The scalar
 */
static void vector_f_mul_scalar_avx(VectorF* result, const VectorF* vec, float scalar) {
    assert(result != NULL);
    assert(vec != NULL);
    assert(result->dim >= vec->dim);
    
    // Use AVX instructions for 4D vectors
    if (vec->dim == 4) {
        __m128 vv = _mm_load_ps(vec->data);
        __m128 vs = _mm_set1_ps(scalar);
        __m128 vr = _mm_mul_ps(vv, vs);
        _mm_store_ps(result->data, vr);
    } else {
        // Fall back to generic implementation for other dimensions
        vector_f_mul_scalar_generic(result, vec, scalar);
    }
}
#endif

/**
 * @brief Multiply a float vector by a scalar
 * 
 * @param arena Arena to allocate from
 * @param vec The vector
 * @param scalar The scalar
 * @return Result vector, or NULL on failure
 */
VectorF* vector_f_mul_scalar(Arena* arena, const VectorF* vec, float scalar) {
    assert(arena != NULL);
    assert(vec != NULL);
    
    VectorF* result = vector_f_create(arena, vec->dim);
    if (!result) {
        return NULL;
    }
    
    // Get the best implementation
    typedef void (*MulScalarFunc)(VectorF*, const VectorF*, float);
    MulScalarFunc mul_scalar_func = (MulScalarFunc)simd_get_best_impl(
        (void*)vector_f_mul_scalar_generic,
#if defined(__SSE__) || defined(__SSE2__)
        (void*)vector_f_mul_scalar_sse,
#else
        NULL,
#endif
#if defined(__AVX__) || defined(__AVX2__)
        (void*)vector_f_mul_scalar_avx,
#else
        NULL,
#endif
        NULL,
        NULL,
        NULL
    );
    
    // Multiply the vector by the scalar
    mul_scalar_func(result, vec, scalar);
    
    return result;
}

/**
 * @brief Generic implementation of float vector dot product
 * 
 * @param a First vector
 * @param b Second vector
 * @return Dot product
 */
static float vector_f_dot_generic(const VectorF* a, const VectorF* b) {
    assert(a != NULL);
    assert(b != NULL);
    assert(a->dim == b->dim);
    
    float result = 0.0f;
    for (size_t i = 0; i < a->dim; i++) {
        result += a->data[i] * b->data[i];
    }
    
    return result;
}

#if defined(__SSE__) || defined(__SSE2__)
/**
 * @brief SSE implementation of float vector dot product
 * 
 * @param a First vector
 * @param b Second vector
 * @return Dot product
 */
static float vector_f_dot_sse(const VectorF* a, const VectorF* b) {
    assert(a != NULL);
    assert(b != NULL);
    assert(a->dim == b->dim);
    
    // Use SSE instructions for 4D vectors
    if (a->dim == 4) {
        __m128 va = _mm_load_ps(a->data);
        __m128 vb = _mm_load_ps(b->data);
        __m128 vr = _mm_mul_ps(va, vb);
        
        // Horizontal sum
        __m128 shuf = _mm_shuffle_ps(vr, vr, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(vr, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        
        float result;
        _mm_store_ss(&result, sums);
        return result;
    } else {
        // Fall back to generic implementation for other dimensions
        return vector_f_dot_generic(a, b);
    }
}
#endif

#if defined(__AVX__) || defined(__AVX2__)
/**
 * @brief AVX implementation of float vector dot product
 * 
 * @param a First vector
 * @param b Second vector
 * @return Dot product
 */
static float vector_f_dot_avx(const VectorF* a, const VectorF* b) {
    assert(a != NULL);
    assert(b != NULL);
    assert(a->dim == b->dim);
    
    // Use AVX instructions for 4D vectors
    if (a->dim == 4) {
        __m128 va = _mm_load_ps(a->data);
        __m128 vb = _mm_load_ps(b->data);
        __m128 vr = _mm_mul_ps(va, vb);
        
        // Horizontal sum
        __m128 shuf = _mm_shuffle_ps(vr, vr, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(vr, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        
        float result;
        _mm_store_ss(&result, sums);
        return result;
    } else {
        // Fall back to generic implementation for other dimensions
        return vector_f_dot_generic(a, b);
    }
}
#endif

/**
 * @brief Compute the dot product of two float vectors
 * 
 * @param a First vector
 * @param b Second vector
 * @return Dot product
 */
float vector_f_dot(const VectorF* a, const VectorF* b) {
    assert(a != NULL);
    assert(b != NULL);
    assert(a->dim == b->dim);
    
    // Get the best implementation
    typedef float (*DotFunc)(const VectorF*, const VectorF*);
    DotFunc dot_func = (DotFunc)simd_get_best_impl(
        (void*)vector_f_dot_generic,
#if defined(__SSE__) || defined(__SSE2__)
        (void*)vector_f_dot_sse,
#else
        NULL,
#endif
#if defined(__AVX__) || defined(__AVX2__)
        (void*)vector_f_dot_avx,
#else
        NULL,
#endif
        NULL,
        NULL,
        NULL
    );
    
    // Compute the dot product
    return dot_func(a, b);
}

/**
 * @brief Compute the cross product of two float vectors
 * 
 * @param arena Arena to allocate from
 * @param a First vector
 * @param b Second vector
 * @return Result vector, or NULL on failure
 */
VectorF* vector_f_cross(Arena* arena, const VectorF* a, const VectorF* b) {
    assert(arena != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(a->dim == 3 && b->dim == 3);
    
    VectorF* result = vector_f_create(arena, 3);
    if (!result) {
        return NULL;
    }
    
    // Compute the cross product
    result->data[0] = a->data[1] * b->data[2] - a->data[2] * b->data[1];
    result->data[1] = a->data[2] * b->data[0] - a->data[0] * b->data[2];
    result->data[2] = a->data[0] * b->data[1] - a->data[1] * b->data[0];
    
    return result;
}

/**
 * @brief Compute the magnitude of a float vector
 * 
 * @param vec The vector
 * @return Magnitude
 */
float vector_f_magnitude(const VectorF* vec) {
    assert(vec != NULL);
    
    return sqrtf(vector_f_dot(vec, vec));
}

/**
 * @brief Normalize a float vector
 * 
 * @param arena Arena to allocate from
 * @param vec The vector
 * @return Normalized vector, or NULL on failure
 */
VectorF* vector_f_normalize(Arena* arena, const VectorF* vec) {
    assert(arena != NULL);
    assert(vec != NULL);
    
    float mag = vector_f_magnitude(vec);
    if (mag < 1e-6f) {
        // Avoid division by zero
        return vector_f_create(arena, vec->dim);
    }
    
    return vector_f_mul_scalar(arena, vec, 1.0f / mag);
}

/**
 * @brief Create a new float matrix
 * 
 * @param arena Arena to allocate from
 * @param rows Number of rows
 * @param cols Number of columns
 * @return A new float matrix, or NULL on failure
 */
MatrixF* matrix_f_create(Arena* arena, size_t rows, size_t cols) {
    assert(arena != NULL);
    assert(rows > 0 && rows <= VECTOR_MAX_DIM);
    assert(cols > 0 && cols <= VECTOR_MAX_DIM);
    
    MatrixF* mat = arena_alloc_aligned(arena, sizeof(MatrixF), VECTOR_ALIGNMENT);
    if (!mat) {
        return NULL;
    }
    
    mat->rows = rows;
    mat->cols = cols;
    memset(mat->data, 0, sizeof(mat->data));
    
    return mat;
}

/**
 * @brief Create a new double matrix
 * 
 * @param arena Arena to allocate from
 * @param rows Number of rows
 * @param cols Number of columns
 * @return A new double matrix, or NULL on failure
 */
MatrixD* matrix_d_create(Arena* arena, size_t rows, size_t cols) {
    assert(arena != NULL);
    assert(rows > 0 && rows <= VECTOR_MAX_DIM);
    assert(cols > 0 && cols <= VECTOR_MAX_DIM);
    
    MatrixD* mat = arena_alloc_aligned(arena, sizeof(MatrixD), VECTOR_ALIGNMENT);
    if (!mat) {
        return NULL;
    }
    
    mat->rows = rows;
    mat->cols = cols;
    memset(mat->data, 0, sizeof(mat->data));
    
    return mat;
}

/**
 * @brief Create a new integer matrix
 * 
 * @param arena Arena to allocate from
 * @param rows Number of rows
 * @param cols Number of columns
 * @return A new integer matrix, or NULL on failure
 */
MatrixI* matrix_i_create(Arena* arena, size_t rows, size_t cols) {
    assert(arena != NULL);
    assert(rows > 0 && rows <= VECTOR_MAX_DIM);
    assert(cols > 0 && cols <= VECTOR_MAX_DIM);
    
    MatrixI* mat = arena_alloc_aligned(arena, sizeof(MatrixI), VECTOR_ALIGNMENT);
    if (!mat) {
        return NULL;
    }
    
    mat->rows = rows;
    mat->cols = cols;
    memset(mat->data, 0, sizeof(mat->data));
    
    return mat;
}

/**
 * @brief Set a float matrix from an array
 * 
 * @param mat The matrix
 * @param data The array of floats (row-major order)
 * @param rows Number of rows
 * @param cols Number of columns
 * @return true if successful, false otherwise
 */
bool matrix_f_set(MatrixF* mat, const float* data, size_t rows, size_t cols) {
    assert(mat != NULL);
    assert(data != NULL);
    assert(rows > 0 && rows <= VECTOR_MAX_DIM);
    assert(cols > 0 && cols <= VECTOR_MAX_DIM);
    
    if (rows > mat->rows || cols > mat->cols) {
        return false;
    }
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            mat->data[i][j] = data[i * cols + j];
        }
    }
    
    mat->rows = rows;
    mat->cols = cols;
    
    return true;
}

/**
 * @brief Add two float matrices
 * 
 * @param arena Arena to allocate from
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on failure
 */
MatrixF* matrix_f_add(Arena* arena, const MatrixF* a, const MatrixF* b) {
    assert(arena != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(a->rows == b->rows && a->cols == b->cols);
    
    MatrixF* result = matrix_f_create(arena, a->rows, a->cols);
    if (!result) {
        return NULL;
    }
    
    // Add the matrices
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }
    
    return result;
}

/**
 * @brief Subtract two float matrices
 * 
 * @param arena Arena to allocate from
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on failure
 */
MatrixF* matrix_f_sub(Arena* arena, const MatrixF* a, const MatrixF* b) {
    assert(arena != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(a->rows == b->rows && a->cols == b->cols);
    
    MatrixF* result = matrix_f_create(arena, a->rows, a->cols);
    if (!result) {
        return NULL;
    }
    
    // Subtract the matrices
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] - b->data[i][j];
        }
    }
    
    return result;
}

/**
 * @brief Multiply two float matrices
 * 
 * @param arena Arena to allocate from
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on failure
 */
MatrixF* matrix_f_mul(Arena* arena, const MatrixF* a, const MatrixF* b) {
    assert(arena != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(a->cols == b->rows);
    
    MatrixF* result = matrix_f_create(arena, a->rows, b->cols);
    if (!result) {
        return NULL;
    }
    
    // Multiply the matrices
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a->cols; k++) {
                sum += a->data[i][k] * b->data[k][j];
            }
            result->data[i][j] = sum;
        }
    }
    
    return result;
}

/**
 * @brief Multiply a float matrix by a scalar
 * 
 * @param arena Arena to allocate from
 * @param mat The matrix
 * @param scalar The scalar
 * @return Result matrix, or NULL on failure
 */
MatrixF* matrix_f_mul_scalar(Arena* arena, const MatrixF* mat, float scalar) {
    assert(arena != NULL);
    assert(mat != NULL);
    
    MatrixF* result = matrix_f_create(arena, mat->rows, mat->cols);
    if (!result) {
        return NULL;
    }
    
    // Multiply the matrix by the scalar
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            result->data[i][j] = mat->data[i][j] * scalar;
        }
    }
    
    return result;
}

/**
 * @brief Transpose a float matrix
 * 
 * @param arena Arena to allocate from
 * @param mat The matrix
 * @return Transposed matrix, or NULL on failure
 */
MatrixF* matrix_f_transpose(Arena* arena, const MatrixF* mat) {
    assert(arena != NULL);
    assert(mat != NULL);
    
    MatrixF* result = matrix_f_create(arena, mat->cols, mat->rows);
    if (!result) {
        return NULL;
    }
    
    // Transpose the matrix
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            result->data[j][i] = mat->data[i][j];
        }
    }
    
    return result;
}

/**
 * @brief Compute the determinant of a float matrix
 * 
 * @param mat The matrix
 * @return Determinant
 */
float matrix_f_determinant(const MatrixF* mat) {
    assert(mat != NULL);
    assert(mat->rows == mat->cols);
    
    // 1x1 matrix
    if (mat->rows == 1) {
        return mat->data[0][0];
    }
    
    // 2x2 matrix
    if (mat->rows == 2) {
        return mat->data[0][0] * mat->data[1][1] - mat->data[0][1] * mat->data[1][0];
    }
    
    // 3x3 matrix
    if (mat->rows == 3) {
        return mat->data[0][0] * (mat->data[1][1] * mat->data[2][2] - mat->data[1][2] * mat->data[2][1])
             - mat->data[0][1] * (mat->data[1][0] * mat->data[2][2] - mat->data[1][2] * mat->data[2][0])
             + mat->data[0][2] * (mat->data[1][0] * mat->data[2][1] - mat->data[1][1] * mat->data[2][0]);
    }
    
    // 4x4 matrix
    if (mat->rows == 4) {
        float det = 0.0f;
        
        // Compute the determinant using cofactor expansion along the first row
        for (size_t j = 0; j < 4; j++) {
            // Create a 3x3 submatrix by removing the first row and column j
            float submat[3][3];
            for (size_t r = 1; r < 4; r++) {
                size_t sr = r - 1;
                for (size_t c = 0; c < 4; c++) {
                    if (c == j) continue;
                    size_t sc = c < j ? c : c - 1;
                    submat[sr][sc] = mat->data[r][c];
                }
            }
            
            // Compute the determinant of the 3x3 submatrix
            float subdet = submat[0][0] * (submat[1][1] * submat[2][2] - submat[1][2] * submat[2][1])
                         - submat[0][1] * (submat[1][0] * submat[2][2] - submat[1][2] * submat[2][0])
                         + submat[0][2] * (submat[1][0] * submat[2][1] - submat[1][1] * submat[2][0]);
            
            // Add or subtract the term to the determinant
            det += (j % 2 == 0 ? 1.0f : -1.0f) * mat->data[0][j] * subdet;
        }
        
        return det;
    }
    
    // Unsupported matrix size
    assert(0 && "Unsupported matrix size");
    return 0.0f;
}

/**
 * @brief Compute the inverse of a float matrix
 * 
 * @param arena Arena to allocate from
 * @param mat The matrix
 * @return Inverse matrix, or NULL on failure
 */
MatrixF* matrix_f_inverse(Arena* arena, const MatrixF* mat) {
    assert(arena != NULL);
    assert(mat != NULL);
    assert(mat->rows == mat->cols);
    
    // Get the determinant
    float det = matrix_f_determinant(mat);
    if (fabsf(det) < 1e-6f) {
        // Singular matrix
        return NULL;
    }
    
    MatrixF* result = matrix_f_create(arena, mat->rows, mat->cols);
    if (!result) {
        return NULL;
    }
    
    // 1x1 matrix
    if (mat->rows == 1) {
        result->data[0][0] = 1.0f / mat->data[0][0];
        return result;
    }
    
    // 2x2 matrix
    if (mat->rows == 2) {
        float inv_det = 1.0f / det;
        result->data[0][0] = mat->data[1][1] * inv_det;
        result->data[0][1] = -mat->data[0][1] * inv_det;
        result->data[1][0] = -mat->data[1][0] * inv_det;
        result->data[1][1] = mat->data[0][0] * inv_det;
        return result;
    }
    
    // 3x3 matrix
    if (mat->rows == 3) {
        float inv_det = 1.0f / det;
        
        // Compute the adjugate matrix
        result->data[0][0] = (mat->data[1][1] * mat->data[2][2] - mat->data[1][2] * mat->data[2][1]) * inv_det;
        result->data[0][1] = (mat->data[0][2] * mat->data[2][1] - mat->data[0][1] * mat->data[2][2]) * inv_det;
        result->data[0][2] = (mat->data[0][1] * mat->data[1][2] - mat->data[0][2] * mat->data[1][1]) * inv_det;
        
        result->data[1][0] = (mat->data[1][2] * mat->data[2][0] - mat->data[1][0] * mat->data[2][2]) * inv_det;
        result->data[1][1] = (mat->data[0][0] * mat->data[2][2] - mat->data[0][2] * mat->data[2][0]) * inv_det;
        result->data[1][2] = (mat->data[0][2] * mat->data[1][0] - mat->data[0][0] * mat->data[1][2]) * inv_det;
        
        result->data[2][0] = (mat->data[1][0] * mat->data[2][1] - mat->data[1][1] * mat->data[2][0]) * inv_det;
        result->data[2][1] = (mat->data[0][1] * mat->data[2][0] - mat->data[0][0] * mat->data[2][1]) * inv_det;
        result->data[2][2] = (mat->data[0][0] * mat->data[1][1] - mat->data[0][1] * mat->data[1][0]) * inv_det;
        
        return result;
    }
    
    // 4x4 matrix
    if (mat->rows == 4) {
        float inv_det = 1.0f / det;
        
        // Compute the adjugate matrix
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                // Create a 3x3 submatrix by removing row i and column j
                float submat[3][3];
                for (size_t r = 0; r < 4; r++) {
                    if (r == i) continue;
                    size_t sr = r < i ? r : r - 1;
                    for (size_t c = 0; c < 4; c++) {
                        if (c == j) continue;
                        size_t sc = c < j ? c : c - 1;
                        submat[sr][sc] = mat->data[r][c];
                    }
                }
                
                // Compute the determinant of the 3x3 submatrix
                float subdet = submat[0][0] * (submat[1][1] * submat[2][2] - submat[1][2] * submat[2][1])
                             - submat[0][1] * (submat[1][0] * submat[2][2] - submat[1][2] * submat[2][0])
                             + submat[0][2] * (submat[1][0] * submat[2][1] - submat[1][1] * submat[2][0]);
                
                // Set the element in the adjugate matrix
                result->data[j][i] = ((i + j) % 2 == 0 ? 1.0f : -1.0f) * subdet * inv_det;
            }
        }
        
        return result;
    }
    
    // Unsupported matrix size
    assert(0 && "Unsupported matrix size");
    return NULL;
}

/**
 * @brief Multiply a float matrix by a float vector
 * 
 * @param arena Arena to allocate from
 * @param mat The matrix
 * @param vec The vector
 * @return Result vector, or NULL on failure
 */
VectorF* matrix_f_mul_vector(Arena* arena, const MatrixF* mat, const VectorF* vec) {
    assert(arena != NULL);
    assert(mat != NULL);
    assert(vec != NULL);
    assert(mat->cols == vec->dim);
    
    VectorF* result = vector_f_create(arena, mat->rows);
    if (!result) {
        return NULL;
    }
    
    // Multiply the matrix by the vector
    for (size_t i = 0; i < mat->rows; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < mat->cols; j++) {
            sum += mat->data[i][j] * vec->data[j];
        }
        result->data[i] = sum;
    }
    
    return result;
}

/**
 * @brief Create a float identity matrix
 * 
 * @param arena Arena to allocate from
 * @param dim Dimension of the matrix
 * @return Identity matrix, or NULL on failure
 */
MatrixF* matrix_f_identity(Arena* arena, size_t dim) {
    assert(arena != NULL);
    assert(dim > 0 && dim <= VECTOR_MAX_DIM);
    
    MatrixF* mat = matrix_f_create(arena, dim, dim);
    if (!mat) {
        return NULL;
    }
    
    // Set the diagonal elements to 1
    for (size_t i = 0; i < dim; i++) {
        mat->data[i][i] = 1.0f;
    }
    
    return mat;
}

/**
 * @brief Create a double identity matrix
 * 
 * @param arena Arena to allocate from
 * @param dim Dimension of the matrix
 * @return Identity matrix, or NULL on failure
 */
MatrixD* matrix_d_identity(Arena* arena, size_t dim) {
    assert(arena != NULL);
    assert(dim > 0 && dim <= VECTOR_MAX_DIM);
    
    MatrixD* mat = matrix_d_create(arena, dim, dim);
    if (!mat) {
        return NULL;
    }
    
    // Set the diagonal elements to 1
    for (size_t i = 0; i < dim; i++) {
        mat->data[i][i] = 1.0;
    }
    
    return mat;
}

/**
 * @brief Create an integer identity matrix
 * 
 * @param arena Arena to allocate from
 * @param dim Dimension of the matrix
 * @return Identity matrix, or NULL on failure
 */
MatrixI* matrix_i_identity(Arena* arena, size_t dim) {
    assert(arena != NULL);
    assert(dim > 0 && dim <= VECTOR_MAX_DIM);
    
    MatrixI* mat = matrix_i_create(arena, dim, dim);
    if (!mat) {
        return NULL;
    }
    
    // Set the diagonal elements to 1
    for (size_t i = 0; i < dim; i++) {
        mat->data[i][i] = 1;
    }
    
    return mat;
}

/**
 * @brief Compute the gradient of a scalar field at a point
 * 
 * The gradient of a scalar field is a vector field that points in the direction
 * of the greatest rate of increase of the scalar field, with magnitude equal to
 * the rate of increase in that direction.
 * 
 * @param arena Arena to allocate from
 * @param f The scalar field function
 * @param v The point at which to compute the gradient
 * @return The gradient vector, or NULL on failure
 */
VectorF* compute_gradient(Arena* arena, ScalarFieldFunc f, const VectorF* v) {
    assert(arena != NULL);
    assert(f != NULL);
    assert(v != NULL);
    
    // Create a result vector
    VectorF* result = vector_f_create(arena, v->dim);
    if (!result) {
        return NULL;
    }
    
    // Compute the gradient using finite differences
    const float h = 1e-4f; // Step size for finite differences
    const float f_v = f(v); // Value of f at v
    
    // For each dimension, compute the partial derivative
    for (size_t i = 0; i < v->dim; i++) {
        // Create a perturbed vector
        VectorF perturbed = *v;
        perturbed.data[i] += h;
        
        // Compute the partial derivative
        float f_perturbed = f(&perturbed);
        result->data[i] = (f_perturbed - f_v) / h;
    }
    
    return result;
}

/**
 * @brief Compute the divergence of a vector field at a point
 * 
 * The divergence of a vector field is a scalar field that measures the rate at
 * which the vector field "flows" away from a point.
 * 
 * @param arena Arena to allocate from
 * @param F The vector field function
 * @param v The point at which to compute the divergence
 * @return The divergence value
 */
float compute_divergence(Arena* arena, VectorFieldFunc F, const VectorF* v) {
    assert(arena != NULL);
    assert(F != NULL);
    assert(v != NULL);
    
    // Compute the vector field at the point
    VectorF* F_v = F(arena, v);
    if (!F_v) {
        return 0.0f;
    }
    
    // Compute the divergence using finite differences
    const float h = 1e-4f; // Step size for finite differences
    float divergence = 0.0f;
    
    // For each dimension, compute the partial derivative of the corresponding component
    for (size_t i = 0; i < v->dim; i++) {
        // Create a perturbed vector
        VectorF perturbed = *v;
        perturbed.data[i] += h;
        
        // Compute the vector field at the perturbed point
        VectorF* F_perturbed = F(arena, &perturbed);
        if (!F_perturbed) {
            continue;
        }
        
        // Compute the partial derivative
        divergence += (F_perturbed->data[i] - F_v->data[i]) / h;
    }
    
    return divergence;
}

/**
 * @brief Compute the curl of a vector field at a point
 * 
 * The curl of a vector field is a vector field that measures the rotation of the
 * vector field around a point.
 * 
 * @param arena Arena to allocate from
 * @param F The vector field function
 * @param v The point at which to compute the curl
 * @return The curl vector, or NULL on failure
 */
VectorF* compute_curl(Arena* arena, VectorFieldFunc F, const VectorF* v) {
    assert(arena != NULL);
    assert(F != NULL);
    assert(v != NULL);
    assert(v->dim == 3); // Curl is only defined for 3D vector fields
    
    // Create a result vector
    VectorF* result = vector_f_create(arena, 3);
    if (!result) {
        return NULL;
    }
    
    // Compute the curl using finite differences
    const float h = 1e-4f; // Step size for finite differences
    
    // Compute the vector field at the point
    VectorF* F_v = F(arena, v);
    if (!F_v) {
        return result;
    }
    
    // For each component of the curl, compute the cross derivatives
    // curl[0] = dF_z/dy - dF_y/dz
    // curl[1] = dF_x/dz - dF_z/dx
    // curl[2] = dF_y/dx - dF_x/dy
    
    // Compute dF_y/dz
    VectorF perturbed_z = *v;
    perturbed_z.data[2] += h;
    VectorF* F_perturbed_z = F(arena, &perturbed_z);
    float dF_y_dz = F_perturbed_z ? (F_perturbed_z->data[1] - F_v->data[1]) / h : 0.0f;
    
    // Compute dF_z/dy
    VectorF perturbed_y = *v;
    perturbed_y.data[1] += h;
    VectorF* F_perturbed_y = F(arena, &perturbed_y);
    float dF_z_dy = F_perturbed_y ? (F_perturbed_y->data[2] - F_v->data[2]) / h : 0.0f;
    
    // Compute dF_x/dz
    float dF_x_dz = F_perturbed_z ? (F_perturbed_z->data[0] - F_v->data[0]) / h : 0.0f;
    
    // Compute dF_z/dx
    VectorF perturbed_x = *v;
    perturbed_x.data[0] += h;
    VectorF* F_perturbed_x = F(arena, &perturbed_x);
    float dF_z_dx = F_perturbed_x ? (F_perturbed_x->data[2] - F_v->data[2]) / h : 0.0f;
    
    // Compute dF_y/dx
    float dF_y_dx = F_perturbed_x ? (F_perturbed_x->data[1] - F_v->data[1]) / h : 0.0f;
    
    // Compute dF_x/dy
    float dF_x_dy = F_perturbed_y ? (F_perturbed_y->data[0] - F_v->data[0]) / h : 0.0f;
    
    // Set the curl components
    result->data[0] = dF_z_dy - dF_y_dz;
    result->data[1] = dF_x_dz - dF_z_dx;
    result->data[2] = dF_y_dx - dF_x_dy;
    
    return result;
}

/**
 * @brief Compute the Laplacian of a scalar field at a point
 * 
 * The Laplacian of a scalar field is a scalar field that measures the divergence
 * of the gradient of the scalar field.
 * 
 * @param arena Arena to allocate from
 * @param f The scalar field function
 * @param v The point at which to compute the Laplacian
 * @return The Laplacian value
 */
float compute_laplacian(Arena* arena, ScalarFieldFunc f, const VectorF* v) {
    assert(arena != NULL);
    assert(f != NULL);
    assert(v != NULL);
    
    // Compute the Laplacian using finite differences
    const float h = 1e-4f; // Step size for finite differences
    const float f_v = f(v); // Value of f at v
    float laplacian = 0.0f;
    
    // For each dimension, compute the second partial derivative
    for (size_t i = 0; i < v->dim; i++) {
        // Create perturbed vectors
        VectorF perturbed_plus = *v;
        perturbed_plus.data[i] += h;
        
        VectorF perturbed_minus = *v;
        perturbed_minus.data[i] -= h;
        
        // Compute the second partial derivative using central differences
        float f_plus = f(&perturbed_plus);
        float f_minus = f(&perturbed_minus);
        laplacian += (f_plus - 2.0f * f_v + f_minus) / (h * h);
    }
    
    return laplacian;
}
