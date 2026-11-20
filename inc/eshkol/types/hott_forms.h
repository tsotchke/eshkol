/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_TYPES_HOTT_FORMS_H
#define ESHKOL_TYPES_HOTT_FORMS_H

/**
 * @file hott_forms.h
 * @brief The HoTT forms of Eshkol source, handed to the cubical kernel (ADR-0043).
 *
 *   (hott decl ...)    declarations checked by the kernel at compile time;
 *                      no runtime code;
 *   (hott-eval term)   the kernel normalizes term; the form is replaced by
 *                      the normal form as a quoted datum, so the program
 *                      sees the computed value (both native engines and the
 *                      bytecode VM).
 *
 * A rejected form is a compile-time error in every mode: a proof the kernel
 * does not accept is not a proof.
 */

#include "eshkol/eshkol.h"

#ifdef __cplusplus
#include <cstdint>
#include <string>

namespace eshkol::hott {
namespace cubical { class Kernel; }

/** True when @p ast is an ESHKOL_HOTT_OP node spelled `(hott-eval ...)`. */
bool hottFormIsEval(const eshkol_ast_t* ast);

struct HottFormOutcome {
    bool ok = true;
    bool rewritten = false;   ///< hott-eval: the node now holds the quoted normal form
    std::string message;
    uint32_t line = 0, column = 0;
};

/** Elaborate one HoTT form node in @p kernel (rewriting hott-eval in place). */
HottFormOutcome elaborateHottForm(cubical::Kernel& kernel, eshkol_ast_t* ast);

}  // namespace eshkol::hott

extern "C" {
#endif

/**
 * Bytecode-VM entry (the VM reads source with its own reader; see
 * inc/eshkol/backend/vm.h). @p form_text is one `(hott ...)` or
 * `(hott-eval ...)` form; NULL starts a new compilation (fresh kernel).
 * Returns a malloc'd string: for hott-eval the normal form, for hott an
 * empty string; on rejection the diagnostic, with *ok set to 0.
 */
char* eshkol_hott_form_source(const char* form_text, int* ok);

#ifdef __cplusplus
}
#endif

#endif  // ESHKOL_TYPES_HOTT_FORMS_H
