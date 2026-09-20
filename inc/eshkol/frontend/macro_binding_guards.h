#ifndef ESHKOL_FRONTEND_MACRO_BINDING_GUARDS_H
#define ESHKOL_FRONTEND_MACRO_BINDING_GUARDS_H

#include <string.h>

/*
 * These parser-lowered names cannot be rebound by syntax-rules: their call
 * heads are converted to dedicated AST/VM operations before ordinary macro
 * expansion. Keep this shared table in the parser-facing interface so the
 * native and bytecode frontends reject the same unsupported request.
 * True special-form shadowing remains a separate language-design item.
 */
#define ESHKOL_UNSUPPORTED_MACRO_SHADOW_TABLE(X) \
    X(if) X(lambda) X(let) X(let*) X(letrec) X(letrec*) X(<named-let>) \
    X(and) X(or) X(cond) X(case) X(match) X(do) X(when) X(unless) \
    X(quote) X(quasiquote) X(unquote) X(unquote-splicing) \
    X(define) X(define-values) X(define-type) X(define-syntax) X(let-syntax) X(letrec-syntax) \
    X(call/cc) X(call-with-current-continuation) X(dynamic-wind) X(set!) \
    X(import) X(require) X(load) X(provide) \
    X(with-region) X(owned) X(move) X(borrow) X(shared) X(weak-ref) \
    X(extern) X(extern-var) X(tensor) X(matrix) X(diff) X(differentiate) \
    X(derivative) X(D) X(taylor) X(derivative-n) X(gradient) X(jacobian) \
    X(hessian) X(divergence) X(curl) X(laplacian) X(directional-derivative) \
    X(guard) X(raise) X(values) X(call-with-values) X(let-values) X(let*-values) \
    X(unify) X(make-substitution) X(walk) X(make-fact) X(make-kb) \
    X(kb-assert!) X(kb-query) X(kb-query-prefix) \
    X(make-dnc-memory) X(dnc-content-address) X(dnc-loc-address) X(dnc-read) \
    X(dnc-write!) X(dnc-alloc-weights) X(dnc-read-grad) X(dnc-memory?) \
    X(sdnc-program) X(sdnc-run) X(sdnc-weight-grad) X(sdnc-params) \
    X(sdnc-set-params!) X(sdnc-improve!) X(sdnc?) X(logic-var?) \
    X(substitution?) X(kb?) X(make-factor-graph) X(fg-add-factor!) \
    X(fg-infer!) X(fg-observe!) X(free-energy) X(expected-free-energy) \
    X(make-workspace) X(ws-register!) X(ws-step!) X(fg-update-cpt!) \
    X(fact?) X(factor-graph?) X(workspace?) X(case-lambda) \
    X(define-record-type) X(parameterize) X(make-parameter) X(cond-expand) \
    X(include) X(include-ci) X(syntax-error)

static inline int eshkol_is_unsupported_macro_shadow(const char *name) {
    if (!name) return 0;
#define ESHKOL_MACRO_SHADOW_MATCH(token) \
    do { if (strcmp(name, #token) == 0) return 1; } while (0);
    ESHKOL_UNSUPPORTED_MACRO_SHADOW_TABLE(ESHKOL_MACRO_SHADOW_MATCH)
#undef ESHKOL_MACRO_SHADOW_MATCH
    return 0;
}

#endif
