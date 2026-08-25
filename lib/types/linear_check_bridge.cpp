/**
 * @file linear_check_bridge.cpp
 * @brief The linear (no-cloning) judgment, reachable from the bytecode VM.
 *
 * ENGINE PARITY, not a second checker. The linear-type rule lives in exactly
 * one place — TypeChecker::analyzeLinearUses() and the enforcement points that
 * call it — and it stays there. What was missing was a way to REACH it from the
 * VM: `eshkol_emit_eskb()` and the VM's source-execution entry both go through
 * `vm_compiler.c`, which has its own S-expression reader producing `Node*`, a
 * different AST from `eshkol_ast_t` entirely, and never constructs a
 * TypeChecker. The consequence #471 measured was an engine-parity defect on a
 * type-system GUARANTEE: `eshkol-vm-standalone-test` ran a qubit clone to
 * completion, exit 0, with no diagnostic of any kind, for a program the native
 * engine refuses to compile.
 *
 * Reimplementing the judgment over `Node*` would have produced two rules that
 * drift, which is the failure mode this project treats as unacceptable
 * elsewhere. Instead the VM path runs the REAL front end — the same parser, the
 * same macro expander, the same TypeChecker — over the same source text, purely
 * to draw the linear verdict, and then hands the source to its own compiler as
 * before. One source of truth, and a divergence between the engines becomes
 * impossible by construction rather than by discipline.
 *
 * This bridge deliberately reports ONLY the linear verdict. Gradual typing's
 * other findings belong to the engine that is generating code and are none of
 * the VM's business; a type warning here would be noise the VM cannot act on.
 */

#include "eshkol/eshkol.h"
#include "eshkol/core/config.h"
#include "eshkol/types/hott_types.h"
#include "eshkol/types/type_checker.h"
#include "eshkol/frontend/macro_expander.h"

#include <sstream>
#include <string>
#include <vector>

extern "C" {

/**
 * @brief Run the linear (no-cloning) check over @p source.
 *
 * @param source      Eshkol source text.
 * @param source_name Diagnostic name for the source (may be NULL).
 * @return The number of linearity violations found (0 when clean), or -1 when
 *         the source could not be analysed at all — a parse failure, which the
 *         caller must treat as "unknown", never as "clean".
 */
int eshkol_linear_check_source(const char* source, const char* source_name) {
    if (!source) return -1;

    eshkol_reset_parse_errors();
    eshkol_reset_parse_line_counter();
    if (source_name) eshkol_set_parse_source_context(source_name);

    std::vector<eshkol_ast_t> asts;
    {
        std::istringstream in(source);
        while (true) {
            eshkol_ast_t ast = eshkol_parse_next_ast_from_stream(in);
            if (ast.type == ESHKOL_INVALID) break;
            asts.push_back(ast);
            if (eshkol_parse_had_error()) break;
        }
    }
    if (eshkol_parse_had_error()) {
        // A front end that could not read the program cannot vouch for it. The
        // caller decides what to do with that; saying "no violations" here
        // would be the silent acceptance this bridge exists to remove.
        return -1;
    }
    if (asts.empty()) return 0;

    // Macro expansion, then the same top-level SEQUENCE_OP flattening the
    // native path performs — `define-record-type` and friends lower to one
    // SEQUENCE_OP wrapping their sub-defines, and a linear parameter inside one
    // of those must be seen as the define it is.
    eshkol::MacroExpander expander;
    std::vector<eshkol_ast_t> expanded = expander.expandAll(asts);
    std::vector<eshkol_ast_t> flat;
    flat.reserve(expanded.size());
    for (auto& a : expanded) {
        if (a.type == ESHKOL_OP && a.operation.op == ESHKOL_SEQUENCE_OP) {
            for (uint64_t i = 0; i < a.operation.sequence_op.num_expressions; ++i) {
                flat.push_back(a.operation.sequence_op.expressions[i]);
            }
        } else {
            flat.push_back(a);
        }
    }

    eshkol::hott::TypeEnvironment env;
    // strict_types stays FALSE: this bridge draws the linear verdict only, and
    // strictness governs whether ordinary type errors are fatal — a decision
    // for the engine emitting code, not for a linearity probe. unsafe_mode is
    // honoured because `--unsafe` documents that linear types may be duplicated
    // (no-cloning bypassed), and that hatch must mean the same thing on both
    // engines.
    const eshkol_config_t* cfg = eshkol_config_get();
    const bool unsafe = cfg ? cfg->unsafe_mode != 0 : false;
    eshkol::hott::TypeChecker checker(env, /*strict_types=*/false, unsafe);

    // Register type aliases first, exactly as the native path does, so a linear
    // type reached through a `define-type` is resolved and not silently missed.
    for (auto& a : flat) {
        if (a.type == ESHKOL_OP && a.operation.op == ESHKOL_DEFINE_TYPE_OP) {
            checker.synthesize(&a);
        }
    }
    for (auto& a : flat) {
        if (a.type == ESHKOL_OP && a.operation.op == ESHKOL_DEFINE_TYPE_OP) continue;
        checker.synthesize(&a);
    }

    return static_cast<int>(checker.linearityViolations());
}

}  // extern "C"
