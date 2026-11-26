/*
 * Native NARS NAL-1..9 AIKR Truth-Value Calculus
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/nars_truth.h>

#include <cmath>

namespace {

inline bool valid_horizon(double k) {
    return std::isfinite(k) && k > 0.0;
}

inline double clamp_unit(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

/* Convert non-negative finite evidence w to AIKR confidence c in [0, 1). */
inline double w_to_c(double w, double k) {
    if (w <= 0.0) return 0.0;
    double c = w / (w + k);
    if (!std::isfinite(c) || c >= 1.0) {
        c = std::nextafter(1.0, 0.0);
    }
    if (c < 0.0) c = 0.0;
    return c;
}

inline double c_to_w(double c, double k) {
    return (k * c) / (1.0 - c);
}

} /* anonymous namespace */

extern "C" bool eshkol_nars_truth_is_valid(eshkol_nars_truth_t tv) {
    if (!std::isfinite(tv.frequency) || !std::isfinite(tv.confidence)) return false;
    if (tv.frequency < 0.0 || tv.frequency > 1.0) return false;
    if (tv.confidence < 0.0 || tv.confidence >= 1.0) return false;
    return true;
}

extern "C" bool eshkol_nars_make_truth(double frequency, double confidence,
                                       eshkol_nars_truth_t* out) {
    if (!out) return false;
    eshkol_nars_truth_t cand{frequency, confidence};
    if (!eshkol_nars_truth_is_valid(cand)) return false;
    *out = cand;
    return true;
}

extern "C" bool eshkol_nars_to_evidence(eshkol_nars_truth_t tv, double k,
                                        eshkol_nars_evidence_t* out) {
    if (!out || !eshkol_nars_truth_is_valid(tv) || !valid_horizon(k)) return false;
    double w = c_to_w(tv.confidence, k);
    if (!std::isfinite(w) || w < 0.0) return false;
    out->total = w;
    out->positive = tv.frequency * w;
    return true;
}

extern "C" bool eshkol_nars_from_evidence(double w_pos, double w_total, double k,
                                          eshkol_nars_truth_t* out) {
    if (!out || !valid_horizon(k)) return false;
    if (!std::isfinite(w_pos) || !std::isfinite(w_total)) return false;
    if (w_pos < 0.0 || w_total < 0.0 || w_pos > w_total) return false;
    double f = (w_total > 0.0) ? clamp_unit(w_pos / w_total) : 0.5;
    double c = w_to_c(w_total, k);
    out->frequency = f;
    out->confidence = c;
    return eshkol_nars_truth_is_valid(*out);
}

extern "C" bool eshkol_nars_expectation(eshkol_nars_truth_t tv, double* out_expectation) {
    if (!out_expectation || !eshkol_nars_truth_is_valid(tv)) return false;
    *out_expectation = tv.confidence * (tv.frequency - 0.5) + 0.5;
    return std::isfinite(*out_expectation);
}

extern "C" bool eshkol_nars_negation(eshkol_nars_truth_t tv, eshkol_nars_truth_t* out) {
    if (!out || !eshkol_nars_truth_is_valid(tv)) return false;
    out->frequency = clamp_unit(1.0 - tv.frequency);
    out->confidence = tv.confidence;
    return eshkol_nars_truth_is_valid(*out);
}

extern "C" bool eshkol_nars_conversion(eshkol_nars_truth_t tv, double k,
                                       eshkol_nars_truth_t* out) {
    if (!out || !eshkol_nars_truth_is_valid(tv) || !valid_horizon(k)) return false;
    double w = tv.frequency * tv.confidence;
    out->frequency = 1.0;
    out->confidence = w_to_c(w, k);
    return eshkol_nars_truth_is_valid(*out);
}

extern "C" bool eshkol_nars_contraposition(eshkol_nars_truth_t tv, double k,
                                           eshkol_nars_truth_t* out) {
    if (!out || !eshkol_nars_truth_is_valid(tv) || !valid_horizon(k)) return false;
    double w = (1.0 - tv.frequency) * tv.confidence;
    out->frequency = 0.0;
    out->confidence = w_to_c(w, k);
    return eshkol_nars_truth_is_valid(*out);
}

extern "C" bool eshkol_nars_deduction(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                                      eshkol_nars_truth_t* out) {
    if (!out || !eshkol_nars_truth_is_valid(v1) || !eshkol_nars_truth_is_valid(v2)) {
        return false;
    }
    double f = clamp_unit(v1.frequency * v2.frequency);
    double c = f * v1.confidence * v2.confidence;
    if (c >= 1.0) c = std::nextafter(1.0, 0.0);
    out->frequency = f;
    out->confidence = c;
    return eshkol_nars_truth_is_valid(*out);
}

extern "C" bool eshkol_nars_abduction(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                                      double k, eshkol_nars_truth_t* out) {
    if (!out || !eshkol_nars_truth_is_valid(v1) || !eshkol_nars_truth_is_valid(v2) ||
        !valid_horizon(k)) {
        return false;
    }
    /* Abduction uses <M --> P> for weight and <S --> P> for result frequency. */
    double w = v1.frequency * v1.confidence * v2.confidence;
    out->frequency = v2.frequency;
    out->confidence = w_to_c(w, k);
    return eshkol_nars_truth_is_valid(*out);
}

extern "C" bool eshkol_nars_induction(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                                      double k, eshkol_nars_truth_t* out) {
    if (!out || !eshkol_nars_truth_is_valid(v1) || !eshkol_nars_truth_is_valid(v2) ||
        !valid_horizon(k)) {
        return false;
    }
    /* Induction uses <M --> S> for weight and <M --> P> for result frequency. */
    double w = v2.frequency * v1.confidence * v2.confidence;
    out->frequency = v1.frequency;
    out->confidence = w_to_c(w, k);
    return eshkol_nars_truth_is_valid(*out);
}

extern "C" bool eshkol_nars_exemplification(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                                            double k, eshkol_nars_truth_t* out) {
    if (!out || !eshkol_nars_truth_is_valid(v1) || !eshkol_nars_truth_is_valid(v2) ||
        !valid_horizon(k)) {
        return false;
    }
    double w = v1.frequency * v2.frequency * v1.confidence * v2.confidence;
    out->frequency = 1.0;
    out->confidence = w_to_c(w, k);
    return eshkol_nars_truth_is_valid(*out);
}

extern "C" bool eshkol_nars_revision(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                                     double k, eshkol_nars_truth_t* out) {
    if (!out || !eshkol_nars_truth_is_valid(v1) || !eshkol_nars_truth_is_valid(v2) ||
        !valid_horizon(k)) {
        return false;
    }
    double w1 = c_to_w(v1.confidence, k);
    double w2 = c_to_w(v2.confidence, k);
    double w1_pos = v1.frequency * w1;
    double w2_pos = v2.frequency * w2;
    double w = w1 + w2;
    double w_pos = w1_pos + w2_pos;

    if (!std::isfinite(w) || !std::isfinite(w_pos)) {
        /* Extreme confidence near 1.0: use normalized weighting */
        double inv1 = 1.0 - v1.confidence;
        double inv2 = 1.0 - v2.confidence;
        double denom = v1.confidence * inv2 + v2.confidence * inv1;
        out->frequency = (denom > 0.0)
            ? clamp_unit((v1.frequency * v1.confidence * inv2 +
                          v2.frequency * v2.confidence * inv1) / denom)
            : 0.5;
        out->confidence = std::nextafter(1.0, 0.0);
        return eshkol_nars_truth_is_valid(*out);
    }

    out->frequency = (w > 0.0) ? clamp_unit(w_pos / w) : 0.5;
    out->confidence = w_to_c(w, k);
    return eshkol_nars_truth_is_valid(*out);
}

extern "C" bool eshkol_nars_analogy(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                                    eshkol_nars_truth_t* out) {
    if (!out || !eshkol_nars_truth_is_valid(v1) || !eshkol_nars_truth_is_valid(v2)) {
        return false;
    }
    double f = clamp_unit(v1.frequency * v2.frequency);
    double c = v2.frequency * v1.confidence * v2.confidence;
    if (c >= 1.0) c = std::nextafter(1.0, 0.0);
    out->frequency = f;
    out->confidence = c;
    return eshkol_nars_truth_is_valid(*out);
}

extern "C" bool eshkol_nars_comparison(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                                       double k, eshkol_nars_truth_t* out) {
    if (!out || !eshkol_nars_truth_is_valid(v1) || !eshkol_nars_truth_is_valid(v2) ||
        !valid_horizon(k)) {
        return false;
    }
    double f0 = clamp_unit(v1.frequency + v2.frequency - v1.frequency * v2.frequency);
    double f = (f0 > 0.0) ? clamp_unit((v1.frequency * v2.frequency) / f0) : 0.0;
    double w = f0 * v1.confidence * v2.confidence;
    out->frequency = f;
    out->confidence = w_to_c(w, k);
    return eshkol_nars_truth_is_valid(*out);
}

extern "C" bool eshkol_nars_intersection(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                                         eshkol_nars_truth_t* out) {
    if (!out || !eshkol_nars_truth_is_valid(v1) || !eshkol_nars_truth_is_valid(v2)) {
        return false;
    }
    double f = clamp_unit(v1.frequency * v2.frequency);
    double c = v1.confidence * v2.confidence;
    if (c >= 1.0) c = std::nextafter(1.0, 0.0);
    out->frequency = f;
    out->confidence = c;
    return eshkol_nars_truth_is_valid(*out);
}

extern "C" bool eshkol_nars_union(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                                  eshkol_nars_truth_t* out) {
    if (!out || !eshkol_nars_truth_is_valid(v1) || !eshkol_nars_truth_is_valid(v2)) {
        return false;
    }
    double f = clamp_unit(v1.frequency + v2.frequency - v1.frequency * v2.frequency);
    double c = v1.confidence * v2.confidence;
    if (c >= 1.0) c = std::nextafter(1.0, 0.0);
    out->frequency = f;
    out->confidence = c;
    return eshkol_nars_truth_is_valid(*out);
}
