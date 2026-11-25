/*
 * Native Vector Symbolic Architectures / Hyperdimensional Computing (VSA/HDC)
 *
 * Implements:
 *   - FHRR (Fourier Holographic Reduced Representations): unit-modulus complex
 *     phasors e^{i*theta_k} with O(D) Hadamard binding, conjugate unbinding,
 *     and circular-mean bundling
 *   - HRR (Holographic Reduced Representations): real-valued vectors with
 *     circular convolution binding and circular correlation unbinding
 *   - Fractional Power Encoding (FPE): phi(x) = prod_j b_j^{x_j} reproducing
 *     exact shift-invariant continuous similarity kernels (RBF / sinc)
 *   - Unbounded associative cleanup memory with nearest-atom retrieval & margin
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_VSA_HDC_H
#define ESHKOL_CORE_VSA_HDC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 1. FHRR (Phase-Encoded Unit-Modulus Complex Phasors in [-pi, pi)^D)
 * ============================================================================ */

/* Generate a deterministic random FHRR phase vector in [-pi, pi)^dim. */
bool eshkol_fhrr_random(double* out_phases, uint32_t dim, uint64_t seed);

/* Generate a Gaussian-spectral FPE axis vector ~ N(0, sigma^2) for RBF kernels. */
bool eshkol_fhrr_gaussian_axis(double* out_phases, uint32_t dim, double sigma, uint64_t seed);

/* O(D) Hadamard binding: out[k] = wrap(a[k] + b[k]). */
bool eshkol_fhrr_bind(const double* a, const double* b, double* out, uint32_t dim);

/* O(D) conjugate unbinding: out[k] = wrap(a[k] - b[k]). */
bool eshkol_fhrr_unbind(const double* a, const double* b, double* out, uint32_t dim);

/* O(D) circular-mean bundling over `num_vecs` phase vectors (optional `weights`). */
bool eshkol_fhrr_bundle(const double* const* vecs,
                        const double* weights,
                        uint32_t num_vecs,
                        double* out,
                        uint32_t dim);

/* Real inner-product similarity (1/D) * sum_k cos(a[k] - b[k]) in [-1, 1]. */
double eshkol_fhrr_similarity(const double* a, const double* b, uint32_t dim);

/*
 * Fractional Power Encoding (FPE):
 *   phi(x)[k] = wrap(sum_{j=0..num_axes-1} coords[j] * axes[j][k]).
 */
bool eshkol_fhrr_fpe_encode(const double* const* axes,
                            const double* coords,
                            uint32_t num_axes,
                            double* out_phases,
                            uint32_t dim);

/* ============================================================================
 * 2. HRR (Real-Valued Circular Convolution / Correlation)
 * ============================================================================ */

/* Generate a deterministic random HRR real vector ~ N(0, 1/dim). */
bool eshkol_hrr_random(double* out_vec, uint32_t dim, uint64_t seed);

/* Circular convolution binding: out[k] = sum_j a[j] * b[(k - j) mod dim]. */
bool eshkol_hrr_bind(const double* a, const double* b, double* out, uint32_t dim);

/* Circular correlation unbinding: out[k] = sum_j a[(k + j) mod dim] * b[j]. */
bool eshkol_hrr_unbind(const double* a, const double* b, double* out, uint32_t dim);

/* Weighted real bundling with optional unit-norm normalization. */
bool eshkol_hrr_bundle(const double* const* vecs,
                       const double* weights,
                       uint32_t num_vecs,
                       bool normalize,
                       double* out,
                       uint32_t dim);

/* Cosine similarity in [-1, 1]. */
double eshkol_hrr_similarity(const double* a, const double* b, uint32_t dim);

/* ============================================================================
 * 3. Associative Cleanup Memory (Unbounded Codebook)
 * ============================================================================ */

typedef enum eshkol_vsa_space {
    ESHKOL_VSA_FHRR = 0,
    ESHKOL_VSA_HRR  = 1
} eshkol_vsa_space_t;

typedef struct eshkol_vsa_match {
    uint32_t atom_id;
    const char* name;
    double similarity;
    double runner_up_similarity;
    double margin;
    bool found;
} eshkol_vsa_match_t;

typedef struct eshkol_vsa_cleanup eshkol_vsa_cleanup_t;

eshkol_vsa_cleanup_t* eshkol_vsa_cleanup_create(eshkol_vsa_space_t space, uint32_t dim);
void eshkol_vsa_cleanup_destroy(eshkol_vsa_cleanup_t* mem);

/* Add a named hypervector atom to the cleanup codebook. Returns atom_id or UINT32_MAX. */
uint32_t eshkol_vsa_cleanup_add(eshkol_vsa_cleanup_t* mem,
                                const char* name,
                                const double* vec,
                                uint32_t dim);

uint32_t eshkol_vsa_cleanup_size(const eshkol_vsa_cleanup_t* mem);

/* Retrieve nearest codebook atom to `query` with similarity and margin. */
bool eshkol_vsa_cleanup_query(const eshkol_vsa_cleanup_t* mem,
                              const double* query,
                              uint32_t dim,
                              eshkol_vsa_match_t* out_match);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_VSA_HDC_H */
