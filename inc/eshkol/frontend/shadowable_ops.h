/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_FRONTEND_SHADOWABLE_OPS_H
#define ESHKOL_FRONTEND_SHADOWABLE_OPS_H

/**
 * @file shadowable_ops.h
 * @brief Builtins the parser lowers to a dedicated operation node that a
 *        program may still rebind.
 *
 * The parser maps a fixed set of consciousness-engine, workspace and DNC
 * names to their own OP tags so codegen can call the runtime directly. R7RS
 * lets a program bind any of those names (Noesis defines `walk`, `unify`,
 * `make-kb` and others), and a use inside that binding's scope calls the
 * program's procedure.
 *
 * Two facilities consult this one table:
 *   - the macro expander, which owns lexical scope: a node whose name is
 *     bound by an enclosing local binder becomes an ordinary call of that
 *     binder (after renaming, the binder's spelling is no longer the
 *     builtin's, so nothing downstream could see the shadow);
 *   - codegen, for top-level and REPL-batch definitions, which are not
 *     lexical and are not renamed.
 */

#include <eshkol/eshkol.h>

#ifdef __cplusplus
#include <unordered_map>

namespace eshkol {

inline const std::unordered_map<eshkol_op_t, const char*>& userShadowableBuiltinOps() {
    static const std::unordered_map<eshkol_op_t, const char*> table = {
        {ESHKOL_UNIFY_OP,               "unify"},
        {ESHKOL_MAKE_SUBST_OP,          "make-substitution"},
        {ESHKOL_WALK_OP,                "walk"},
        {ESHKOL_MAKE_FACT_OP,           "make-fact"},
        {ESHKOL_MAKE_KB_OP,             "make-kb"},
        {ESHKOL_KB_ASSERT_OP,           "kb-assert!"},
        {ESHKOL_KB_QUERY_OP,            "kb-query"},
        {ESHKOL_KB_QUERY_PREFIX_OP,     "kb-query-prefix"},
        {ESHKOL_MAKE_FACTOR_GRAPH_OP,   "make-factor-graph"},
        {ESHKOL_FG_ADD_FACTOR_OP,       "fg-add-factor!"},
        {ESHKOL_FG_INFER_OP,            "fg-infer!"},
        {ESHKOL_FG_OBSERVE_OP,          "fg-observe!"},
        {ESHKOL_FG_UPDATE_CPT_OP,       "fg-update-cpt!"},
        {ESHKOL_FREE_ENERGY_OP,         "free-energy"},
        {ESHKOL_EXPECTED_FREE_ENERGY_OP,"expected-free-energy"},
        {ESHKOL_MAKE_WORKSPACE_OP,      "make-workspace"},
        {ESHKOL_WS_REGISTER_OP,         "ws-register!"},
        {ESHKOL_WS_STEP_OP,             "ws-step!"},
        {ESHKOL_DNC_MAKE_OP,            "make-dnc-memory"},
        {ESHKOL_DNC_CONTENT_ADDR_OP,    "dnc-content-address"},
        {ESHKOL_DNC_LOC_ADDR_OP,        "dnc-loc-address"},
        {ESHKOL_DNC_READ_OP,            "dnc-read"},
        {ESHKOL_DNC_WRITE_OP,           "dnc-write!"},
        {ESHKOL_DNC_ALLOC_WEIGHTS_OP,   "dnc-alloc-weights"},
        {ESHKOL_DNC_READ_GRAD_OP,       "dnc-read-grad"},
        {ESHKOL_DNC_PRED_OP,            "dnc-memory?"},
        {ESHKOL_SDNC_PROGRAM_OP,        "sdnc-program"},
        {ESHKOL_SDNC_RUN_OP,            "sdnc-run"},
        {ESHKOL_SDNC_WEIGHT_GRAD_OP,    "sdnc-weight-grad"},
        {ESHKOL_SDNC_PARAMS_OP,         "sdnc-params"},
        {ESHKOL_SDNC_SET_PARAMS_OP,     "sdnc-set-params!"},
        {ESHKOL_SDNC_IMPROVE_OP,        "sdnc-improve!"},
        {ESHKOL_SDNC_PRED_OP,           "sdnc?"},
    };
    return table;
}

} // namespace eshkol
#endif /* __cplusplus */

#endif /* ESHKOL_FRONTEND_SHADOWABLE_OPS_H */
