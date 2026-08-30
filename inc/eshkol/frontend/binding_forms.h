#ifndef ESHKOL_FRONTEND_BINDING_FORMS_H
#define ESHKOL_FRONTEND_BINDING_FORMS_H

/*
 * The parser owns this enumeration.  Backend analyses consume it rather
 * than maintaining a second list of binding forms.  A row is present for
 * every syntax form whose expansion or AST layout introduces a lexical or
 * dynamic binding relevant to assignment conversion.
 *
 * flags are parser metadata.  The observation flag marks forms whose
 * lowering can retain a location beyond the source expression that created
 * it.  The table is intentionally an X-macro so C and C++ consumers can
 * generate identical ids and policies.
 */
#define ESHKOL_PARSER_BINDING_FORM_OBSERVES 1

#define ESHKOL_PARSER_BINDING_FORM_TABLE(X) \
    X(LET,             "let",             0) \
    X(LET_STAR,        "let*",            0) \
    X(LETREC,          "letrec",          0) \
    X(LETREC_STAR,     "letrec*",         0) \
    X(LET_VALUES,      "let-values",      0) \
    X(LET_STAR_VALUES, "let*-values",     0) \
    X(NAMED_LET,       "<named-let>",     ESHKOL_PARSER_BINDING_FORM_OBSERVES) \
    X(DO,              "do",              0) \
    X(INTERNAL_DEFINE, "define",          ESHKOL_PARSER_BINDING_FORM_OBSERVES) \
    X(LAMBDA_PARAMS,   "lambda",          ESHKOL_PARSER_BINDING_FORM_OBSERVES) \
    X(GUARD,           "guard",           ESHKOL_PARSER_BINDING_FORM_OBSERVES) \
    X(DYNAMIC_WIND,    "dynamic-wind",    ESHKOL_PARSER_BINDING_FORM_OBSERVES) \
    X(PARAMETERIZE,    "parameterize",    ESHKOL_PARSER_BINDING_FORM_OBSERVES) \
    X(CASE_LAMBDA,     "case-lambda",     ESHKOL_PARSER_BINDING_FORM_OBSERVES) \
    X(DEFINE_VALUES,   "define-values",   ESHKOL_PARSER_BINDING_FORM_OBSERVES)

enum {
#define ESHKOL_PARSER_BINDING_FORM_ENUM(id, spelling, flags) \
    ESHKOL_PARSER_BINDING_FORM_##id,
    ESHKOL_PARSER_BINDING_FORM_TABLE(ESHKOL_PARSER_BINDING_FORM_ENUM)
#undef ESHKOL_PARSER_BINDING_FORM_ENUM
    ESHKOL_PARSER_BINDING_FORM_COUNT
};

#endif /* ESHKOL_FRONTEND_BINDING_FORMS_H */
