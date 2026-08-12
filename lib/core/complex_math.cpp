/**
 * @file complex_math.cpp
 * @brief Native-runtime symbols for the complex transcendentals.
 *
 * Thin `extern "C"` wrappers over the shared inline core in
 * `<eshkol/core/complex_math.h>`. The LLVM back end emits calls to these
 * symbols when a math builtin is applied to an ESHKOL_VALUE_COMPLEX operand;
 * the bytecode VM includes the same header and calls the inline functions
 * directly, so the two engines compute identical bits.
 *
 * Calling convention: operands and results are passed by pointer to a
 * 16-byte `eshkol_complex_number_t`, so no struct-return ABI variation can
 * come between the compiler-emitted call and this definition. `out` may alias
 * `z` — every wrapper computes into a local before storing.
 *
 * Before these existed, a complex operand reaching `sin`/`sqrt`/`exp`/... was
 * routed into the scalar path, which read the tagged value's payload — a HEAP
 * POINTER — as an IEEE double: `(sqrt (make-rectangular -1.0 0.0))` printed
 * 73278.56614317723 and `(exp (make-rectangular 0.0 3.14159))` printed
 * +inf.0. See tests/numeric/complex_transcendentals_test.esk.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <eshkol/core/complex_math.h>
#include <eshkol/eshkol.h>

namespace {

/** @brief Reinterpret a runtime complex struct as the shared pair type. */
inline eshkol_cpx load_cpx(const eshkol_complex_number_t* z) {
    return eshkol_cpx_make(z->real, z->imag);
}

/** @brief Store a shared pair back into a runtime complex struct. */
inline void store_cpx(eshkol_complex_number_t* out, eshkol_cpx v) {
    out->real = v.re;
    out->imag = v.im;
}

}  // namespace

/**
 * Define one `extern "C"` wrapper per unary complex function. Each is named
 * `eshkol_complex_<name>` — the name the LLVM back end looks up.
 */
#define ESHKOL_DEFINE_COMPLEX_UNARY(name)                                     \
    extern "C" void eshkol_complex_##name(const eshkol_complex_number_t* z,   \
                                          eshkol_complex_number_t* out) {     \
        store_cpx(out, eshkol_cpx_##name(load_cpx(z)));                       \
    }

ESHKOL_DEFINE_COMPLEX_UNARY(sqrt)
ESHKOL_DEFINE_COMPLEX_UNARY(exp)
ESHKOL_DEFINE_COMPLEX_UNARY(log)
ESHKOL_DEFINE_COMPLEX_UNARY(log2)
ESHKOL_DEFINE_COMPLEX_UNARY(log10)
ESHKOL_DEFINE_COMPLEX_UNARY(exp2)
ESHKOL_DEFINE_COMPLEX_UNARY(sin)
ESHKOL_DEFINE_COMPLEX_UNARY(cos)
ESHKOL_DEFINE_COMPLEX_UNARY(tan)
ESHKOL_DEFINE_COMPLEX_UNARY(asin)
ESHKOL_DEFINE_COMPLEX_UNARY(acos)
ESHKOL_DEFINE_COMPLEX_UNARY(atan)
ESHKOL_DEFINE_COMPLEX_UNARY(sinh)
ESHKOL_DEFINE_COMPLEX_UNARY(cosh)
ESHKOL_DEFINE_COMPLEX_UNARY(tanh)
ESHKOL_DEFINE_COMPLEX_UNARY(asinh)
ESHKOL_DEFINE_COMPLEX_UNARY(acosh)
ESHKOL_DEFINE_COMPLEX_UNARY(atanh)

#undef ESHKOL_DEFINE_COMPLEX_UNARY

/** @brief Principal complex power a^b = exp(b log a) (R7RS `expt`). */
extern "C" void eshkol_complex_pow(const eshkol_complex_number_t* a,
                                   const eshkol_complex_number_t* b,
                                   eshkol_complex_number_t* out) {
    store_cpx(out, eshkol_cpx_pow(load_cpx(a), load_cpx(b)));
}
