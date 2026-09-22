/**
 * @file vm_macro.c
 * @brief `syntax-rules` for the Eshkol bytecode compiler.
 *
 * The transformer registry and the adapter between the VM reader's nodes and
 * the one `syntax-rules` engine both engines share
 * (inc/eshkol/frontend/syntax_rules_core.h, ADR-0026). The compiler expands
 * each form it reaches, in that form's own scope; identifiers a template
 * introduces arrive colored (inc/eshkol/frontend/syntax_color.h) and
 * vm_compiler.c resolves them by the shared renaming rule.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#ifndef VM_MACRO_C_INCLUDED
#define VM_MACRO_C_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * AST Node (matches stackvm_codegen.c Node)
 ******************************************************************************/

#ifndef VM_MACRO_NODE_DEFINED
#define VM_MACRO_NODE_DEFINED

typedef enum {
    N_NUMBER  = 0,
    N_SYMBOL  = 1,
    N_LIST    = 2,
    N_STRING  = 3,
    N_BOOL    = 4
} MacroNodeType;

typedef struct MacroNode {
    MacroNodeType    type;
    double           numval;
    char             symbol[128];
    char*            string_data;
    size_t           string_len;
    struct MacroNode** children;
    int              n_children;
    int              _cap;       /* allocation capacity for children */
    int              is_char;
    int              is_inexact;
    int              is_int;
    int              is_verbatim;
    int              is_vector;
    long long        ival;
    int              is_bignum;
    int              macro_scope_limit;
    int              macro_value_owner;
    int              macro_value_slot;
    int              macro_value_context;
} MacroNode;

#endif /* VM_MACRO_NODE_DEFINED */

/*******************************************************************************
 * Node Construction / Deep Copy / Free
 ******************************************************************************/

static MacroNode* macro_node_new(MacroNodeType type) {
    MacroNode* n = (MacroNode*)calloc(1, sizeof(MacroNode));
    if (!n) { fprintf(stderr, "ERROR: macro_node_new: alloc failed\n"); return NULL; }
    n->type = type;
    return n;
}

/** @brief Append @p child to @p parent's children array, growing it
 *         (doubling, minimum 4) via realloc. */
static void macro_node_add_child(MacroNode* parent, MacroNode* child) {
    if (!parent || !child) return;
    if (parent->n_children >= parent->_cap) {
        int new_cap = parent->_cap < 4 ? 4 : parent->_cap * 2;
        MacroNode** nc = (MacroNode**)realloc(parent->children, new_cap * sizeof(MacroNode*));
        if (!nc) { fprintf(stderr, "ERROR: macro_node_add_child: alloc failed\n"); return; }
        parent->children = nc;
        parent->_cap = new_cap;
    }
    parent->children[parent->n_children++] = child;
}

/** @brief Recursively deep-copy a MacroNode tree (used when instantiating
 *         a template so each expansion gets independent nodes).
 *
 *  SW-13: this used to copy only `type`, `numval` and `symbol`, dropping every
 *  other scalar the hub's node carries.  On the VM hub those are the literal's
 *  EXACTNESS TAGS (`is_int`/`ival`, `is_inexact`, `is_char`), so a literal that
 *  travelled through a macro — either from the template or substituted in from
 *  the call site — came out of the expander untagged: `(dbl 2.0)` lost
 *  `is_inexact` and folded to the exact 2, a large exact literal lost
 *  `is_int`/`ival` and degraded to a double, and `#\a` lost `is_char` and
 *  became an integer.  Whole-struct assignment copies every scalar field the
 *  hub defines, and stays correct if fields are added later; only the
 *  children array is re-derived, because it must be independently owned. */
static MacroNode* macro_node_deep_copy(const MacroNode* src) {
    if (!src) return NULL;
    MacroNode* dst = (MacroNode*)calloc(1, sizeof(MacroNode));
    if (!dst) { fprintf(stderr, "ERROR: macro_node_deep_copy: alloc failed\n"); return NULL; }
    *dst = *src;                 /* every scalar field, hub-agnostic */
    dst->string_data = NULL;
    if (src->string_data) {
        dst->string_data = (char*)malloc(src->string_len + 1);
        if (!dst->string_data) { free(dst); return NULL; }
        memcpy(dst->string_data, src->string_data, src->string_len);
        dst->string_data[src->string_len] = 0;
    }
    dst->children   = NULL;      /* deep-copied below; never aliased */
    dst->n_children = 0;
    dst->_cap       = 0;
    for (int i = 0; i < src->n_children; i++) {
        macro_node_add_child(dst, macro_node_deep_copy(src->children[i]));
    }
    return dst;
}

/** @brief Recursively free a MacroNode tree and its children array. */
static void macro_node_free(MacroNode* n) {
    if (!n) return;
    for (int i = 0; i < n->n_children; i++) {
        macro_node_free(n->children[i]);
    }
    free(n->string_data);
    free(n->children);
    free(n);
}

/* Create a symbol node */
static MacroNode* macro_make_symbol(const char* name) {
    MacroNode* n = macro_node_new(N_SYMBOL);
    if (!n) return NULL;
    strncpy(n->symbol, name, 127);
    n->symbol[127] = '\0';
    return n;
}

/* Create a number node */
static MacroNode* macro_make_number(double val) {
    MacroNode* n = macro_node_new(N_NUMBER);
    if (!n) return NULL;
    n->numval = val;
    return n;
}

/* Create a list node */
static MacroNode* macro_make_list(void) {
    return macro_node_new(N_LIST);
}

/* Create a boolean node */
static MacroNode* macro_make_bool(int val) {
    MacroNode* n = macro_node_new(N_BOOL);
    if (!n) return NULL;
    n->numval = val ? 1.0 : 0.0;
    strncpy(n->symbol, val ? "#t" : "#f", 127);
    return n;
}

/*******************************************************************************
 * Macro Definition and Registry
 *
 * A transformer keeps its rules as reader nodes (deep copies owned by the
 * registry) and, alongside, the same rules converted once to the neutral
 * syntax of the shared engine (eshkol/frontend/syntax_rules_core.h).
 ******************************************************************************/

#include "eshkol/frontend/syntax_rules_core.h"

typedef struct {
    MacroNode* pattern;         /* list pattern (e.g., (_ x y)) */
    MacroNode* template_node;   /* template (e.g., (+ x y)) */
} MacroRule;

typedef struct {
    char         name[128];
    char         ellipsis[128];      /* R7RS 4.3.2: `...` unless a custom one is named */
    MacroRule*   rules;
    int          n_rules;
    char**       literals;
    int          n_literals;
    eshkol_syn** syn_patterns;       /* engine view of rules[i].pattern */
    eshkol_syn** syn_templates;      /* engine view of rules[i].template_node */
    int          definition_limit;
    unsigned     serial;             /* binding order; see vm_macro_shadowed() */
} VmMacro;

static VmMacro* g_macros = NULL;
static int      g_n_macros = 0;
static int      g_macro_cap = 0;
static int      g_macro_global_limit = 0;
static int      g_macro_scope_depth = 0;
static int      g_macro_value_context_serial = 0;
/* One counter orders every binding the compiler creates -- lexical locals
 * (add_local) and macro keywords -- so the innermost binding of a name is
 * the one created last among those in scope. */
static unsigned g_vm_binding_serial = 0;
/* Colors of expansions (eshkol/frontend/syntax_color.h). */
static unsigned g_macro_color_counter = 0;

static void vm_macro_stamp_definition_scope(MacroNode* node, int limit) {
    if (!node) return;
    node->macro_scope_limit = limit;
    for (int i = 0; i < node->n_children; ++i)
        vm_macro_stamp_definition_scope(node->children[i], limit);
}

/* ── Reader nodes <-> engine syntax ─────────────────────────────────────── */

static int vm_macro_is_dot(const MacroNode* n) {
    return n && n->type == N_SYMBOL && !n->is_verbatim && strcmp(n->symbol, ".") == 0;
}

/** Equality key of a literal datum: two atoms match a literal pattern iff
 *  their keys are equal (type, exactness and value). */
static char* vm_macro_atom_key(const MacroNode* n) {
    char head[96];
    const char* tail = "";
    size_t tail_len = 0;
    switch (n->type) {
        case N_STRING:
            snprintf(head, sizeof(head), "s%zu:", n->string_len);
            tail = n->string_data ? n->string_data : "";
            tail_len = n->string_len;
            break;
        case N_BOOL:
            snprintf(head, sizeof(head), "b:%d", n->numval != 0);
            break;
        case N_NUMBER:
            if (n->is_bignum && n->string_data) {
                snprintf(head, sizeof(head), "n:big:");
                tail = n->string_data;
                tail_len = n->string_len;
            } else if (n->is_char) {
                snprintf(head, sizeof(head), "c:%lld", (long long)n->numval);
            } else if (n->is_int) {
                snprintf(head, sizeof(head), "i:%lld", (long long)n->ival);
            } else {
                snprintf(head, sizeof(head), "n:%d:%.17g", n->is_inexact, n->numval);
            }
            break;
        default:
            snprintf(head, sizeof(head), "?:%d", (int)n->type);
            break;
    }
    size_t head_len = strlen(head);
    char* key = (char*)malloc(head_len + tail_len + 1);
    if (!key) return NULL;
    memcpy(key, head, head_len);
    /* Embedded NULs would end the key early; they never matter for equality
     * of the literal patterns programs write, but keep the length exact. */
    for (size_t i = 0; i < tail_len; ++i) key[head_len + i] = tail[i] ? tail[i] : '\x01';
    key[head_len + tail_len] = '\0';
    return key;
}

static eshkol_syn* vm_macro_to_syn(const MacroNode* n) {
    if (!n) return NULL;
    if (n->type == N_SYMBOL) return eshkol_syn_new(ESHKOL_SYN_SYMBOL, n->symbol, n);
    if (n->type != N_LIST) {
        char* key = vm_macro_atom_key(n);
        eshkol_syn* atom = key ? eshkol_syn_new(ESHKOL_SYN_ATOM, key, n) : NULL;
        free(key);
        return atom;
    }
    if (n->is_vector) {
        /* The reader spells #(a b) as the call (vector a b); its datum is the
         * vector of the elements. */
        eshkol_syn* vec = eshkol_syn_new(ESHKOL_SYN_VECTOR, "#(", n);
        for (int i = 1; vec && i < n->n_children; ++i)
            if (!eshkol_syn_push(vec, vm_macro_to_syn(n->children[i]))) {
                eshkol_syn_free(vec);
                return NULL;
            }
        return vec;
    }
    eshkol_syn* list = eshkol_syn_new(ESHKOL_SYN_LIST, "(", n);
    if (!list) return NULL;
    const int count = n->n_children;
    const int dotted = count >= 3 && vm_macro_is_dot(n->children[count - 2]);
    for (int i = 0; i < count; ++i) {
        if (dotted && i == count - 2) continue;
        if (!eshkol_syn_push(list, vm_macro_to_syn(n->children[i]))) {
            eshkol_syn_free(list);
            return NULL;
        }
    }
    list->dotted = dotted;
    return list;
}

static MacroNode* vm_macro_from_syn(const eshkol_syn* s) {
    const MacroNode* origin = (const MacroNode*)s->origin;
    switch (s->kind) {
        case ESHKOL_SYN_SYMBOL: {
            if (strlen(s->text) >= sizeof(((MacroNode*)0)->symbol)) {
                fprintf(stderr, "ERROR: macro expansion produced an identifier longer than %zu bytes\n",
                        sizeof(((MacroNode*)0)->symbol) - 1);
                return NULL;
            }
            /* Copy the source identifier first so its definition-site
             * annotations (scope limit, captured value) travel with it. */
            MacroNode* sym = (origin && origin->type == N_SYMBOL)
                ? macro_node_deep_copy(origin) : macro_make_symbol("");
            if (!sym) return NULL;
            snprintf(sym->symbol, sizeof(sym->symbol), "%s", s->text);
            return sym;
        }
        case ESHKOL_SYN_ATOM:
            return macro_node_deep_copy(origin);
        case ESHKOL_SYN_PREFIX: {
            /* The VM reader desugars prefixes; one only arises from matching. */
            const char* name = eshkol_syn_prefix_name(s->text);
            MacroNode* list = macro_make_list();
            if (!list) return NULL;
            macro_node_add_child(list, macro_make_symbol(name ? name : "quote"));
            if (s->n_items > 0) macro_node_add_child(list, vm_macro_from_syn(s->items[0]));
            return list;
        }
        case ESHKOL_SYN_LIST:
        case ESHKOL_SYN_VECTOR: {
            MacroNode* list = macro_make_list();
            if (!list) return NULL;
            if (s->kind == ESHKOL_SYN_VECTOR) {
                list->is_vector = 1;
                macro_node_add_child(list, macro_make_symbol("vector"));
            }
            for (int i = 0; i < s->n_items; ++i) {
                if (s->dotted && i == s->n_items - 1)
                    macro_node_add_child(list, macro_make_symbol("."));
                MacroNode* child = vm_macro_from_syn(s->items[i]);
                if (!child) { macro_node_free(list); return NULL; }
                macro_node_add_child(list, child);
            }
            return list;
        }
    }
    return NULL;
}

/* ── Registry ───────────────────────────────────────────────────────────── */

static void vm_macro_release(VmMacro* m) {
    for (int j = 0; j < m->n_rules; j++) {
        eshkol_syn_free(m->syn_patterns[j]);
        eshkol_syn_free(m->syn_templates[j]);
        macro_node_free(m->rules[j].pattern);
        macro_node_free(m->rules[j].template_node);
    }
    for (int j = 0; j < m->n_literals; j++) free(m->literals[j]);
    free(m->rules);
    free(m->literals);
    free(m->syn_patterns);
    free(m->syn_templates);
    memset(m, 0, sizeof(*m));
}

/** @brief Register a transformer (name, ellipsis, literals, rules). The
 *         registry copies everything it keeps. */
static int vm_macro_register(const char* name, const char* ellipsis,
                             const char* const* literals, int n_literals,
                             const MacroRule* rules, int n_rules) {
    if (!name || strlen(name) >= sizeof(((VmMacro*)0)->name)) return 0;
    if (g_n_macros == g_macro_cap) {
        int cap = g_macro_cap ? g_macro_cap * 2 : 64;
        VmMacro* grown = (VmMacro*)realloc(g_macros, (size_t)cap * sizeof(VmMacro));
        if (!grown) return 0;
        g_macros = grown;
        g_macro_cap = cap;
    }
    VmMacro* m = &g_macros[g_n_macros];
    memset(m, 0, sizeof(*m));
    snprintf(m->name, sizeof(m->name), "%s", name);
    snprintf(m->ellipsis, sizeof(m->ellipsis), "%s", ellipsis ? ellipsis : "...");
    m->rules = (MacroRule*)calloc(n_rules > 0 ? (size_t)n_rules : 1u, sizeof(MacroRule));
    m->syn_patterns = (eshkol_syn**)calloc(n_rules > 0 ? (size_t)n_rules : 1u, sizeof(eshkol_syn*));
    m->syn_templates = (eshkol_syn**)calloc(n_rules > 0 ? (size_t)n_rules : 1u, sizeof(eshkol_syn*));
    m->literals = (char**)calloc(n_literals > 0 ? (size_t)n_literals : 1u, sizeof(char*));
    if (!m->rules || !m->syn_patterns || !m->syn_templates || !m->literals) {
        vm_macro_release(m);
        return 0;
    }
    for (int i = 0; i < n_literals; i++) m->literals[i] = eshkol_syn_strdup(literals[i]);
    m->n_literals = n_literals;
    g_n_macros++;
    m->definition_limit = g_macro_scope_depth ? g_n_macros + 1 : -1;
    if (!g_macro_scope_depth) g_macro_global_limit = g_n_macros;
    for (int i = 0; i < n_rules; i++) {
        m->rules[i].pattern = macro_node_deep_copy(rules[i].pattern);
        m->rules[i].template_node = macro_node_deep_copy(rules[i].template_node);
        m->n_rules = i + 1;
    }
    /* The definition scope and engine view are fixed by vm_macro_seal()
     * once the compiler has settled the scope (a letrec-syntax group) and
     * annotated the templates' free values (vm_macro_capture_definition). */
    m->serial = ++g_vm_binding_serial;
    return 1;
}

/** @brief Build the engine view of @p m's rules. Called once its template
 *         nodes carry their final annotations; the view's origins point
 *         into the registry's own nodes. */
static void vm_macro_seal(VmMacro* m) {
    for (int i = 0; i < m->n_rules; i++) {
        /* A template's own macro keywords resolve where it was defined. */
        vm_macro_stamp_definition_scope(m->rules[i].template_node, m->definition_limit);
        eshkol_syn_free(m->syn_patterns[i]);
        eshkol_syn_free(m->syn_templates[i]);
        m->syn_patterns[i] = vm_macro_to_syn(m->rules[i].pattern);
        m->syn_templates[i] = vm_macro_to_syn(m->rules[i].template_node);
    }
}

/** @brief Innermost registered transformer spelled exactly @p name within
 *         the first @p limit registrations. */
static VmMacro* vm_macro_find(const char* name, int limit) {
    if (limit > g_n_macros) limit = g_n_macros;
    for (int i = limit - 1; i >= 0; --i)
        if (strcmp(g_macros[i].name, name) == 0) return &g_macros[i];
    return NULL;
}

/* The transformer a keyword identifier denotes. An identifier a template
 * introduced resolves where its template was defined (its scope limit), one
 * color at a time (ADR-0026); substituted caller identifiers keep the
 * caller's scope. A live global boundary preserves forward references
 * without admitting a later local shadow. */
static VmMacro* vm_macro_lookup_node(const MacroNode* identifier) {
    if (!identifier || identifier->type != N_SYMBOL) return NULL;
    int limit = identifier->macro_scope_limit;
    limit = limit < 0 ? g_macro_global_limit : (limit ? limit - 1 : g_n_macros);
    char name[sizeof(((MacroNode*)0)->symbol)];
    snprintf(name, sizeof(name), "%s", identifier->symbol);
    for (;;) {
        VmMacro* found = vm_macro_find(name, limit);
        if (found) return found;
        size_t prefix = 0;
        if (!eshkol_syntax_last_color(name, &prefix)) return NULL;
        name[prefix] = '\0';
    }
}

/** @brief Free every registered macro from index @p saved onward. */
static void vm_macro_restore(int saved) {
    for (int i = saved; i < g_n_macros; i++) vm_macro_release(&g_macros[i]);
    g_n_macros = saved;
}

/*******************************************************************************
 * Expansion
 ******************************************************************************/

/** @brief Expand one use of the macro @p macro. On a use no rule matches, or
 *         a malformed template, reports and returns NULL.
 * @return The expansion (caller frees), or NULL. */
static MacroNode* vm_macro_expand_use(VmMacro* macro, const MacroNode* node) {
    eshkol_syn* use = vm_macro_to_syn(node);
    if (!use) return NULL;
    eshkol_syn* result = NULL;
    char error[256] = {0};
    eshkol_syn_outcome outcome = eshkol_syntax_rules_apply(
        macro->ellipsis, (const char* const*)macro->literals, macro->n_literals,
        macro->syn_patterns, macro->syn_templates, macro->n_rules,
        use, ++g_macro_color_counter, NULL, NULL, &result, error, sizeof(error));
    eshkol_syn_free(use);
    if (outcome == ESHKOL_SYN_NO_MATCH) {
        fprintf(stderr, "ERROR: syntax error: no matching pattern for macro '%s'\n", macro->name);
        return NULL;
    }
    if (outcome == ESHKOL_SYN_ERROR) {
        fprintf(stderr, "ERROR: macro '%s': %s\n", macro->name, error);
        return NULL;
    }
    MacroNode* expanded = vm_macro_from_syn(result);
    eshkol_syn_free(result);
    return expanded;
}

/** @brief Expand the use @p node of @p macro, and every macro use its
 *         expansion immediately is, to the first form that is not a macro
 *         use. Operands are left to the compiler, which expands each form
 *         it reaches in its own scope.
 * @return The expansion (caller frees), or NULL after reporting. */
static MacroNode* vm_macro_expand(VmMacro* macro, const MacroNode* node) {
    MacroNode* expanded = vm_macro_expand_use(macro, node);
    /* Continuation-passing transformers rewrite a use into another use of
     * themselves once per element; only a chain that never reaches an
     * ordinary form is an error. */
    for (int steps = 0; expanded && steps < 100000; ++steps) {
        if (expanded->type != N_LIST || expanded->n_children == 0 ||
            expanded->children[0]->type != N_SYMBOL)
            return expanded;
        VmMacro* next = vm_macro_lookup_node(expanded->children[0]);
        if (!next) return expanded;
        /* The compiler decides lexical shadowing of a keyword; only a head
         * the template itself colored (and did not bind) is expanded here. */
        if (!eshkol_syntax_is_colored(expanded->children[0]->symbol)) return expanded;
        MacroNode* again = vm_macro_expand_use(next, expanded);
        macro_node_free(expanded);
        expanded = again;
    }
    if (expanded) {
        fprintf(stderr, "ERROR: macro expansion of '%s' did not terminate\n", macro->name);
        macro_node_free(expanded);
    }
    return NULL;
}

/*******************************************************************************
 * Parse define-syntax from AST
 *
 * Expected form:
 *   (define-syntax name
 *     (syntax-rules [ellipsis] (literal ...) (pattern template) ...))
 ******************************************************************************/

static int vm_macro_define_binding(const MacroNode* identifier,
                                   const MacroNode* syntax_rules) {
    if (!identifier || identifier->type != N_SYMBOL || !syntax_rules) return 0;
    if (syntax_rules->type != N_LIST || syntax_rules->n_children < 2) return 0;
    if (syntax_rules->children[0]->type != N_SYMBOL ||
        !eshkol_syntax_base_is(syntax_rules->children[0]->symbol, "syntax-rules")) return 0;

    int next = 1;
    const char* ellipsis = "...";
    if (syntax_rules->children[next]->type == N_SYMBOL) {
        ellipsis = syntax_rules->children[next]->symbol;   /* R7RS 4.3.2 */
        ++next;
    }
    if (next >= syntax_rules->n_children) return 0;
    const MacroNode* lit_list = syntax_rules->children[next++];
    if (lit_list->type != N_LIST) return 0;
    const char** lits = (const char**)calloc(lit_list->n_children > 0 ? (size_t)lit_list->n_children : 1u,
                                             sizeof(char*));
    MacroRule* rules = (MacroRule*)calloc(syntax_rules->n_children > 0 ? (size_t)syntax_rules->n_children : 1u,
                                          sizeof(MacroRule));
    if (!lits || !rules) { free((void*)lits); free(rules); return 0; }
    int n_lits = 0, n_rules = 0, ok = 1;
    for (int i = 0; i < lit_list->n_children; i++) {
        if (lit_list->children[i]->type != N_SYMBOL) { ok = 0; break; }
        lits[n_lits++] = lit_list->children[i]->symbol;
    }
    for (int i = next; ok && i < syntax_rules->n_children; i++) {
        const MacroNode* rule = syntax_rules->children[i];
        if (rule->type != N_LIST || rule->n_children != 2 ||
            rule->children[0]->type != N_LIST || rule->children[0]->n_children == 0) {
            ok = 0;
            break;
        }
        rules[n_rules].pattern = (MacroNode*)rule->children[0];
        rules[n_rules].template_node = (MacroNode*)rule->children[1];
        n_rules++;
    }
    if (ok) ok = vm_macro_register(identifier->symbol, ellipsis, lits, n_lits, rules, n_rules);
    free((void*)lits);
    free(rules);
    return ok;
}

static int vm_macro_define_syntax(const MacroNode* form) {
    if (!form || form->type != N_LIST || form->n_children != 3) return 0;
    if (form->children[0]->type != N_SYMBOL ||
        !eshkol_syntax_base_is(form->children[0]->symbol, "define-syntax")) return 0;
    return vm_macro_define_binding(form->children[1], form->children[2]);
}

/*******************************************************************************
 * Utility: print AST (for debugging)
 ******************************************************************************/

/** @brief Recursively print a MacroNode tree as S-expression text, for
 *         debugging. */
static void macro_node_print(const MacroNode* n, int depth) {
    if (!n) { printf("NULL"); return; }
    switch (n->type) {
        case N_NUMBER:
            printf("%.15g", n->numval);
            break;
        case N_SYMBOL:
            printf("%s", n->symbol);
            break;
        case N_STRING:
            printf("\"");
            if (n->string_data) fwrite(n->string_data, 1, n->string_len, stdout);
            printf("\"");
            break;
        case N_BOOL:
            printf("%s", n->numval ? "#t" : "#f");
            break;
        case N_LIST:
            printf("(");
            for (int i = 0; i < n->n_children; i++) {
                if (i > 0) printf(" ");
                macro_node_print(n->children[i], depth + 1);
            }
            printf(")");
            break;
    }
}

#endif /* VM_MACRO_C_INCLUDED */

