/**
 * @file arity_contract.h
 * @brief The one wording every engine uses to refuse a wrong-arity call.
 *
 * WHY THIS EXISTS. A wrong-arity call to a builtin must be refused the same
 * way on every engine — that is the P8 axis-3 contract
 * (scripts/p8/p8_arity_sweep.py, diagnostic_class()), and the escape it closes
 * is a call that one engine rejects while the other quietly executes it.
 *
 * Refusing was not enough. Both engines already refused `(ceiling)`, but the
 * bytecode VM said "Arity mismatch: ceiling expects 1 argument but got 0"
 * while native lowering said "ceil requires exactly 1 argument" — a private
 * per-handler sentence, one of ~200 spread across the backend, naming the
 * LLVM intrinsic rather than the procedure the programmer wrote. A reader
 * (and the parity ratchet) could not tell that the two engines had reached
 * the same verdict, so five builtins reported as native-vs-VM divergences
 * when their behaviour was in fact identical.
 *
 * So the CLASS MARKER lives here, in one place, and every arity refusal on
 * every engine carries it:
 *
 *     Arity mismatch: <detail>
 *
 * eshkol_format_arity_mismatch() renders the canonical detail for the common
 * case (a fixed minimum the caller undershot). A backend guard with something
 * more specific to say (`1 to 3 arguments`, `at least 2`) keeps its own
 * sentence and still gets the marker, because eshkol_arity_error_current()
 * prepends it centrally — see lib/core/logger.cpp.
 *
 * Header-only on purpose: lib/backend/eshkol_vm.c is compiled standalone for
 * the hosted-VM test binary and for the WASM lane, so the shared wording must
 * cost no link-time dependency.
 */
#ifndef ESHKOL_CORE_ARITY_CONTRACT_H
#define ESHKOL_CORE_ARITY_CONTRACT_H

#include <stdio.h>

/** The canonical class marker. Anything an engine prints to refuse a call for
 *  argument-count reasons starts with this, and nothing else does. */
#define ESHKOL_ARITY_MISMATCH_PREFIX "Arity mismatch: "

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Render the canonical wrong-arity diagnostic into @p buf.
 * @param buf      Destination buffer.
 * @param buf_size Its size in bytes.
 * @param name     The PUBLIC procedure name the programmer wrote — never the
 *                 internal lowering or intrinsic spelling.
 * @param expected The minimum number of arguments the callee requires.
 * @param got      The number the call supplied.
 * @return The snprintf() return value.
 */
static inline int eshkol_format_arity_mismatch(char* buf, size_t buf_size,
                                               const char* name,
                                               int expected, long long got) {
    return snprintf(buf, buf_size,
                    ESHKOL_ARITY_MISMATCH_PREFIX
                    "%s expects %d argument%s but got %lld",
                    name ? name : "<procedure>", expected,
                    expected == 1 ? "" : "s", got);
}

/**
 * @brief The minimum number of arguments a BUILTIN requires — the ONE arity
 *        fact both engines consult.
 *
 * Implemented over BUILTINS[] in lib/backend/eshkol_vm.c, which is also what
 * scripts/gen_language_surface.py turns into
 * tests/coverage/language_surface.json. The native LLVM backend and the
 * bytecode VM compiler both call this rather than keeping their own copy, so
 * neither can drift into refusing a call the other accepts.
 *
 * @param name Public procedure name.
 * @return The minimum argument count, or -1 when the table makes no claim:
 *         an unregistered name, or a row the table declares variadic in the
 *         native lowering (`gcd`, `lcm`), where there is no minimum to
 *         enforce and a refusal would reject a legal call.
 *
 * @note Native Windows builds have no bytecode VM (lib/backend/eshkol_vm_stub.c)
 *       and therefore no parity obligation; the stub answers -1 and the
 *       backend's own per-lowering guards remain the only refusal there.
 */
int eshkol_builtin_min_arity(const char* name);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* ESHKOL_CORE_ARITY_CONTRACT_H */
