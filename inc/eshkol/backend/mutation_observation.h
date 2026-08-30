#ifndef ESHKOL_MUTATION_OBSERVATION_H
#define ESHKOL_MUTATION_OBSERVATION_H

/*
 * Native and bytecode ASTs have different layouts, but assignment conversion
 * needs one shared conservative policy: can a mutable location be observed
 * after the current frame is copied? Keep every binding/context form here.
 */

#include <string.h>
#include <eshkol/frontend/binding_forms.h>

#define ESHKOL_MUTATION_FORM_OBSERVES ESHKOL_PARSER_BINDING_FORM_OBSERVES
#define ESHKOL_MUTATION_OBSERVATION_FORM_TABLE(X) \
    ESHKOL_PARSER_BINDING_FORM_TABLE(X)

enum {
#define ESHKOL_MUTATION_FORM_ENUM(id, spelling, flags) \
    ESHKOL_MUTATION_FORM_##id = ESHKOL_PARSER_BINDING_FORM_##id,
    ESHKOL_MUTATION_OBSERVATION_FORM_TABLE(ESHKOL_MUTATION_FORM_ENUM)
#undef ESHKOL_MUTATION_FORM_ENUM
    ESHKOL_MUTATION_FORM_COUNT = ESHKOL_PARSER_BINDING_FORM_COUNT
};

#if defined(__cplusplus)
static_assert((int)ESHKOL_MUTATION_FORM_COUNT ==
              (int)ESHKOL_PARSER_BINDING_FORM_COUNT,
              "assignment-conversion form table is incomplete");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert((int)ESHKOL_MUTATION_FORM_COUNT ==
               (int)ESHKOL_PARSER_BINDING_FORM_COUNT,
               "assignment-conversion form table is incomplete");
#endif

static inline int eshkol_mutation_form_observes(int form) {
    switch (form) {
#define ESHKOL_MUTATION_FORM_POLICY(id, spelling, flags) \
        case ESHKOL_MUTATION_FORM_##id: \
            return ((flags) & ESHKOL_MUTATION_FORM_OBSERVES) != 0;
        ESHKOL_MUTATION_OBSERVATION_FORM_TABLE(ESHKOL_MUTATION_FORM_POLICY)
#undef ESHKOL_MUTATION_FORM_POLICY
        default: return 0;
    }
}

static inline int eshkol_mutation_head_observes(const char* head) {
    if (!head) return 0;
#define ESHKOL_MUTATION_HEAD_POLICY(id, spelling, flags) \
    if (((flags) & ESHKOL_MUTATION_FORM_OBSERVES) && strcmp(head, spelling) == 0) return 1;
    ESHKOL_MUTATION_OBSERVATION_FORM_TABLE(ESHKOL_MUTATION_HEAD_POLICY)
#undef ESHKOL_MUTATION_HEAD_POLICY
    return 0;
}

/* Shared decision used by both backend-specific AST adapters. */
static inline int eshkol_mutation_may_be_observed_after_mutation(
    int has_set, int has_observing_context, int has_callcc) {
    return has_set && (has_observing_context || has_callcc);
}

#endif /* ESHKOL_MUTATION_OBSERVATION_H */
