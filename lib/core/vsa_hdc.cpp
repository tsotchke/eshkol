/*
 * Native Vector Symbolic Architectures / Hyperdimensional Computing (VSA/HDC)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/vsa_hdc.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr double kTwoPi = 2.0 * kPi;

inline uint64_t splitmix64_next(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

inline double uniform01(uint64_t& state) {
    /* 53-bit uniform in (0, 1) */
    uint64_t r = splitmix64_next(state) >> 11;
    return (static_cast<double>(r) + 0.5) * (1.0 / 9007199254740992.0);
}

inline double wrap_angle(double x) {
    double r = std::fmod(x + kPi, kTwoPi);
    if (r < 0.0) r += kTwoPi;
    return r - kPi;
}

} /* anonymous namespace */

extern "C" bool eshkol_fhrr_random(double* out_phases, uint32_t dim, uint64_t seed) {
    if (!out_phases || dim == 0) return false;
    uint64_t st = seed ^ 0x243f6a8885a308d3ULL;
    for (uint32_t i = 0; i < dim; i++) {
        out_phases[i] = (uniform01(st) * kTwoPi) - kPi;
    }
    return true;
}

extern "C" bool eshkol_fhrr_gaussian_axis(double* out_phases, uint32_t dim,
                                          double sigma, uint64_t seed) {
    if (!out_phases || dim == 0 || !std::isfinite(sigma) || sigma <= 0.0) return false;
    uint64_t st = seed ^ 0x13198a2e03707344ULL;
    for (uint32_t i = 0; i < dim; i += 2) {
        double u1 = std::max(1e-15, uniform01(st));
        double u2 = uniform01(st);
        double mag = sigma * std::sqrt(-2.0 * std::log(u1));
        out_phases[i] = mag * std::cos(kTwoPi * u2);
        if (i + 1u < dim) {
            out_phases[i + 1u] = mag * std::sin(kTwoPi * u2);
        }
    }
    return true;
}

extern "C" bool eshkol_fhrr_bind(const double* a, const double* b, double* out, uint32_t dim) {
    if (!a || !b || !out || dim == 0) return false;
    for (uint32_t i = 0; i < dim; i++) {
        out[i] = wrap_angle(a[i] + b[i]);
    }
    return true;
}

extern "C" bool eshkol_fhrr_unbind(const double* a, const double* b, double* out, uint32_t dim) {
    if (!a || !b || !out || dim == 0) return false;
    for (uint32_t i = 0; i < dim; i++) {
        out[i] = wrap_angle(a[i] - b[i]);
    }
    return true;
}

extern "C" bool eshkol_fhrr_bundle(const double* const* vecs,
                                   const double* weights,
                                   uint32_t num_vecs,
                                   double* out,
                                   uint32_t dim) {
    if (!vecs || !out || num_vecs == 0 || dim == 0) return false;
    for (uint32_t v = 0; v < num_vecs; v++) {
        if (!vecs[v]) return false;
    }
    for (uint32_t k = 0; k < dim; k++) {
        double re = 0.0;
        double im = 0.0;
        for (uint32_t v = 0; v < num_vecs; v++) {
            double w = weights ? weights[v] : 1.0;
            re += w * std::cos(vecs[v][k]);
            im += w * std::sin(vecs[v][k]);
        }
        out[k] = std::atan2(im, re);
    }
    return true;
}

extern "C" double eshkol_fhrr_similarity(const double* a, const double* b, uint32_t dim) {
    if (!a || !b || dim == 0) return 0.0;
    double acc = 0.0;
    for (uint32_t k = 0; k < dim; k++) {
        acc += std::cos(a[k] - b[k]);
    }
    return acc / static_cast<double>(dim);
}

extern "C" bool eshkol_fhrr_fpe_encode(const double* const* axes,
                                       const double* coords,
                                       uint32_t num_axes,
                                       double* out_phases,
                                       uint32_t dim) {
    if (!axes || !coords || !out_phases || num_axes == 0 || dim == 0) return false;
    for (uint32_t j = 0; j < num_axes; j++) {
        if (!axes[j] || !std::isfinite(coords[j])) return false;
    }
    for (uint32_t k = 0; k < dim; k++) {
        double acc = 0.0;
        for (uint32_t j = 0; j < num_axes; j++) {
            acc += coords[j] * axes[j][k];
        }
        out_phases[k] = wrap_angle(acc);
    }
    return true;
}

extern "C" bool eshkol_hrr_random(double* out_vec, uint32_t dim, uint64_t seed) {
    if (!out_vec || dim == 0) return false;
    double sigma = 1.0 / std::sqrt(static_cast<double>(dim));
    uint64_t st = seed ^ 0xa4093822299f31d0ULL;
    for (uint32_t i = 0; i < dim; i += 2) {
        double u1 = std::max(1e-15, uniform01(st));
        double u2 = uniform01(st);
        double mag = sigma * std::sqrt(-2.0 * std::log(u1));
        out_vec[i] = mag * std::cos(kTwoPi * u2);
        if (i + 1u < dim) {
            out_vec[i + 1u] = mag * std::sin(kTwoPi * u2);
        }
    }
    return true;
}

extern "C" bool eshkol_hrr_bind(const double* a, const double* b, double* out, uint32_t dim) {
    if (!a || !b || !out || dim == 0) return false;
    std::vector<double> tmp(dim, 0.0);
    for (uint32_t k = 0; k < dim; k++) {
        double acc = 0.0;
        for (uint32_t j = 0; j < dim; j++) {
            uint32_t idx = (k >= j) ? (k - j) : (k + dim - j);
            acc += a[j] * b[idx];
        }
        tmp[k] = acc;
    }
    std::memcpy(out, tmp.data(), static_cast<size_t>(dim) * sizeof(double));
    return true;
}

extern "C" bool eshkol_hrr_unbind(const double* a, const double* b, double* out, uint32_t dim) {
    if (!a || !b || !out || dim == 0) return false;
    std::vector<double> tmp(dim, 0.0);
    for (uint32_t k = 0; k < dim; k++) {
        double acc = 0.0;
        for (uint32_t j = 0; j < dim; j++) {
            uint32_t idx = (k + j) % dim;
            acc += a[idx] * b[j];
        }
        tmp[k] = acc;
    }
    std::memcpy(out, tmp.data(), static_cast<size_t>(dim) * sizeof(double));
    return true;
}

extern "C" bool eshkol_hrr_bundle(const double* const* vecs,
                                  const double* weights,
                                  uint32_t num_vecs,
                                  bool normalize,
                                  double* out,
                                  uint32_t dim) {
    if (!vecs || !out || num_vecs == 0 || dim == 0) return false;
    for (uint32_t v = 0; v < num_vecs; v++) {
        if (!vecs[v]) return false;
    }
    std::vector<double> tmp(dim, 0.0);
    for (uint32_t v = 0; v < num_vecs; v++) {
        double w = weights ? weights[v] : 1.0;
        for (uint32_t k = 0; k < dim; k++) {
            tmp[k] += w * vecs[v][k];
        }
    }
    if (normalize) {
        double n2 = 0.0;
        for (double x : tmp) n2 += x * x;
        double norm = std::sqrt(n2);
        if (norm > 1e-15) {
            for (double& x : tmp) x /= norm;
        }
    }
    std::memcpy(out, tmp.data(), static_cast<size_t>(dim) * sizeof(double));
    return true;
}

extern "C" double eshkol_hrr_similarity(const double* a, const double* b, uint32_t dim) {
    if (!a || !b || dim == 0) return 0.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (uint32_t k = 0; k < dim; k++) {
        dot += a[k] * b[k];
        na += a[k] * a[k];
        nb += b[k] * b[k];
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    return (denom > 1e-15) ? (dot / denom) : 0.0;
}

struct CleanupEntry {
    std::string name;
    std::vector<double> vec;
};

struct eshkol_vsa_cleanup {
    eshkol_vsa_space_t space = ESHKOL_VSA_FHRR;
    uint32_t dim = 0;
    std::vector<CleanupEntry> entries;

    eshkol_vsa_cleanup(eshkol_vsa_space_t sp, uint32_t d) : space(sp), dim(d) {}
};

extern "C" eshkol_vsa_cleanup_t* eshkol_vsa_cleanup_create(eshkol_vsa_space_t space, uint32_t dim) {
    if (dim == 0) return nullptr;
    return new (std::nothrow) eshkol_vsa_cleanup(space, dim);
}

extern "C" void eshkol_vsa_cleanup_destroy(eshkol_vsa_cleanup_t* mem) {
    delete mem;
}

extern "C" uint32_t eshkol_vsa_cleanup_add(eshkol_vsa_cleanup_t* mem,
                                           const char* name,
                                           const double* vec,
                                           uint32_t dim) {
    if (!mem || !vec || dim != mem->dim) return UINT32_MAX;
    CleanupEntry e;
    e.name = name ? name : "";
    e.vec.assign(vec, vec + dim);
    uint32_t id = static_cast<uint32_t>(mem->entries.size());
    mem->entries.push_back(std::move(e));
    return id;
}

extern "C" uint32_t eshkol_vsa_cleanup_size(const eshkol_vsa_cleanup_t* mem) {
    return mem ? static_cast<uint32_t>(mem->entries.size()) : 0u;
}

extern "C" bool eshkol_vsa_cleanup_query(const eshkol_vsa_cleanup_t* mem,
                                         const double* query,
                                         uint32_t dim,
                                         eshkol_vsa_match_t* out_match) {
    if (!out_match) return false;
    std::memset(out_match, 0, sizeof(*out_match));
    if (!mem || !query || dim != mem->dim || mem->entries.empty()) return false;

    double best_sim = -std::numeric_limits<double>::infinity();
    double second_sim = -std::numeric_limits<double>::infinity();
    uint32_t best_id = 0;

    for (uint32_t i = 0; i < mem->entries.size(); i++) {
        double sim = (mem->space == ESHKOL_VSA_FHRR)
            ? eshkol_fhrr_similarity(query, mem->entries[i].vec.data(), dim)
            : eshkol_hrr_similarity(query, mem->entries[i].vec.data(), dim);
        if (sim > best_sim) {
            second_sim = best_sim;
            best_sim = sim;
            best_id = i;
        } else if (sim > second_sim) {
            second_sim = sim;
        }
    }

    out_match->atom_id = best_id;
    out_match->name = mem->entries[best_id].name.c_str();
    out_match->similarity = best_sim;
    out_match->runner_up_similarity = std::isfinite(second_sim) ? second_sim : best_sim;
    out_match->margin = std::isfinite(second_sim) ? (best_sim - second_sim) : 0.0;
    out_match->found = true;
    return true;
}
