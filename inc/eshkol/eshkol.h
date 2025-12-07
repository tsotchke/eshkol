/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#ifndef ESHKOL_ESHKOL_H
#define ESHKOL_ESHKOL_H

#include <stdint.h>

#ifdef __cplusplus

#include <fstream>

extern "C" {
#endif

typedef enum {
    ESHKOL_INVALID,
    ESHKOL_UNTYPED,
    ESHKOL_UINT8,
    ESHKOL_UINT16,
    ESHKOL_UINT32,
    ESHKOL_UINT64,
    ESHKOL_INT8,
    ESHKOL_INT16,
    ESHKOL_INT32,
    ESHKOL_INT64,
    ESHKOL_DOUBLE,
    ESHKOL_STRING,
    ESHKOL_FUNC,
    ESHKOL_VAR,
    ESHKOL_OP,
    ESHKOL_CONS,
    ESHKOL_NULL,
    ESHKOL_TENSOR,
    ESHKOL_CHAR,
    ESHKOL_BOOL
} eshkol_type_t;

// Mixed type list support - Value type tags for tagged cons cells
typedef enum {
    ESHKOL_VALUE_NULL        = 0,  // Empty/null value
    ESHKOL_VALUE_INT64       = 1,  // 64-bit signed integer
    ESHKOL_VALUE_DOUBLE      = 2,  // Double-precision floating point
    ESHKOL_VALUE_CONS_PTR    = 3,  // Pointer to another cons cell
    ESHKOL_VALUE_DUAL_NUMBER = 4,  // Dual number for forward-mode AD
    ESHKOL_VALUE_AD_NODE_PTR = 5,  // Pointer to AD computation graph node
    ESHKOL_VALUE_TENSOR_PTR  = 6,  // Pointer to tensor structure
    ESHKOL_VALUE_LAMBDA_SEXPR = 7, // Lambda S-expression metadata (homoiconicity)
    ESHKOL_VALUE_STRING_PTR  = 8,  // Pointer to string (char* with length prefix)
    ESHKOL_VALUE_CHAR        = 9,  // Character (stored as Unicode codepoint in data field)
    ESHKOL_VALUE_VECTOR_PTR  = 10, // Pointer to Scheme vector (heterogeneous array of tagged values)
    ESHKOL_VALUE_SYMBOL      = 11, // Symbol (interned string for identifiers)
    ESHKOL_VALUE_CLOSURE_PTR = 12, // Pointer to closure (func_ptr + captured environment)
    ESHKOL_VALUE_BOOL        = 13, // Boolean (#t or #f, stored as 1 or 0)
    // Reserved for future expansion
    ESHKOL_VALUE_MAX         = 15  // 4-bit type field limit
} eshkol_value_type_t;

// Type flags for Scheme exactness tracking
#define ESHKOL_VALUE_EXACT_FLAG   0x10
#define ESHKOL_VALUE_INEXACT_FLAG 0x20

// Combined type constants for common cases
#define ESHKOL_VALUE_EXACT_INT64     (ESHKOL_VALUE_INT64 | ESHKOL_VALUE_EXACT_FLAG)
#define ESHKOL_VALUE_INEXACT_DOUBLE  (ESHKOL_VALUE_DOUBLE | ESHKOL_VALUE_INEXACT_FLAG)

// Tagged data union for cons cell values
typedef union eshkol_tagged_data {
    int64_t int_val;     // Integer value
    double double_val;   // Double-precision floating point value
    uint64_t ptr_val;    // Pointer value (for cons cell pointers)
    uint64_t raw_val;    // Raw 64-bit value for manipulation
} eshkol_tagged_data_t;

// Runtime tagged value representation for ALL Eshkol values
// This struct is used throughout the system to preserve type information
typedef struct eshkol_tagged_value {
    uint8_t type;        // Value type (eshkol_value_type_t)
    uint8_t flags;       // Exactness and other flags
    uint16_t reserved;   // Reserved for future use
    union {
        int64_t int_val;
        double double_val;
        uint64_t ptr_val;
        uint64_t raw_val;   // For raw manipulation and zero-initialization
    } data;
} eshkol_tagged_value_t;

// Compile-time size validation for tagged values
_Static_assert(sizeof(eshkol_tagged_value_t) <= 16,
               "Tagged value must fit in 16 bytes for efficiency");

// Dual number for forward-mode automatic differentiation
// Stores value and derivative simultaneously for efficient chain rule computation
typedef struct eshkol_dual_number {
    double value;       // f(x) - the function value
    double derivative;  // f'(x) - the derivative value
} eshkol_dual_number_t;

// Compile-time size validation for dual numbers
_Static_assert(sizeof(eshkol_dual_number_t) == 16,
               "Dual number must be 16 bytes for cache efficiency");

// Helper functions for tagged value manipulation
static inline eshkol_tagged_value_t eshkol_make_int64(int64_t val, bool exact) {
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_INT64;
    result.flags = exact ? ESHKOL_VALUE_EXACT_FLAG : 0;
    result.reserved = 0;
    result.data.int_val = val;
    return result;
}

static inline eshkol_tagged_value_t eshkol_make_double(double val) {
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_DOUBLE;
    result.flags = ESHKOL_VALUE_INEXACT_FLAG;
    result.reserved = 0;
    result.data.double_val = val;
    return result;
}

static inline eshkol_tagged_value_t eshkol_make_ptr(uint64_t ptr, uint8_t type) {
    eshkol_tagged_value_t result;
    result.type = type;
    result.flags = 0;
    result.reserved = 0;
    result.data.ptr_val = ptr;
    return result;
}

static inline int64_t eshkol_unpack_int64(const eshkol_tagged_value_t* val) {
    return val->data.int_val;
}

static inline double eshkol_unpack_double(const eshkol_tagged_value_t* val) {
    return val->data.double_val;
}

static inline uint64_t eshkol_unpack_ptr(const eshkol_tagged_value_t* val) {
    return val->data.ptr_val;
}

// Type checking helper macros
#define ESHKOL_IS_INT64_TYPE(type)       (((type) & 0x0F) == ESHKOL_VALUE_INT64 || ((type) & 0x0F) == ESHKOL_VALUE_CHAR || ((type) & 0x0F) == ESHKOL_VALUE_BOOL)
#define ESHKOL_IS_DOUBLE_TYPE(type)      (((type) & 0x0F) == ESHKOL_VALUE_DOUBLE)
#define ESHKOL_IS_CONS_PTR_TYPE(type)    (((type) & 0x0F) == ESHKOL_VALUE_CONS_PTR)
#define ESHKOL_IS_NULL_TYPE(type)        (((type) & 0x0F) == ESHKOL_VALUE_NULL)
#define ESHKOL_IS_DUAL_NUMBER_TYPE(type) (((type) & 0x0F) == ESHKOL_VALUE_DUAL_NUMBER)
#define ESHKOL_IS_AD_NODE_PTR_TYPE(type) (((type) & 0x0F) == ESHKOL_VALUE_AD_NODE_PTR)
#define ESHKOL_IS_TENSOR_PTR_TYPE(type)  (((type) & 0x0F) == ESHKOL_VALUE_TENSOR_PTR)
#define ESHKOL_IS_LAMBDA_SEXPR_TYPE(type) (((type) & 0x0F) == ESHKOL_VALUE_LAMBDA_SEXPR)
#define ESHKOL_IS_STRING_PTR_TYPE(type)  (((type) & 0x0F) == ESHKOL_VALUE_STRING_PTR)
#define ESHKOL_IS_CHAR_TYPE(type)        (((type) & 0x0F) == ESHKOL_VALUE_CHAR)
#define ESHKOL_IS_VECTOR_PTR_TYPE(type)  (((type) & 0x0F) == ESHKOL_VALUE_VECTOR_PTR)
#define ESHKOL_IS_SYMBOL_TYPE(type)      (((type) & 0x0F) == ESHKOL_VALUE_SYMBOL)
#define ESHKOL_IS_CLOSURE_PTR_TYPE(type) (((type) & 0x0F) == ESHKOL_VALUE_CLOSURE_PTR)
#define ESHKOL_IS_BOOL_TYPE(type)        (((type) & 0x0F) == ESHKOL_VALUE_BOOL)
// General pointer type check: any type that stores a pointer value (not int64 or double)
#define ESHKOL_IS_ANY_PTR_TYPE(type)     (ESHKOL_IS_CONS_PTR_TYPE(type) || \
                                          ESHKOL_IS_STRING_PTR_TYPE(type) || \
                                          ESHKOL_IS_VECTOR_PTR_TYPE(type) || \
                                          ESHKOL_IS_TENSOR_PTR_TYPE(type) || \
                                          ESHKOL_IS_AD_NODE_PTR_TYPE(type) || \
                                          ESHKOL_IS_LAMBDA_SEXPR_TYPE(type) || \
                                          ESHKOL_IS_CLOSURE_PTR_TYPE(type))

// Exactness checking macros
#define ESHKOL_IS_EXACT(type)         (((type) & ESHKOL_VALUE_EXACT_FLAG) != 0)
#define ESHKOL_IS_INEXACT(type)       (((type) & ESHKOL_VALUE_INEXACT_FLAG) != 0)

// Type manipulation macros
#define ESHKOL_MAKE_EXACT(type)       ((type) | ESHKOL_VALUE_EXACT_FLAG)
#define ESHKOL_MAKE_INEXACT(type)     ((type) | ESHKOL_VALUE_INEXACT_FLAG)
#define ESHKOL_GET_BASE_TYPE(type)    ((type) & 0x0F)

// Dual number helper functions for forward-mode automatic differentiation
static inline eshkol_dual_number_t eshkol_make_dual(double value, double derivative) {
    eshkol_dual_number_t result;
    result.value = value;
    result.derivative = derivative;
    return result;
}

static inline double eshkol_dual_value(const eshkol_dual_number_t* d) {
    return d->value;
}

static inline double eshkol_dual_derivative(const eshkol_dual_number_t* d) {
    return d->derivative;
}

// ===== COMPUTATIONAL GRAPH NODE TYPES =====
// AD node types for reverse-mode automatic differentiation

typedef enum {
    AD_NODE_CONSTANT,
    AD_NODE_VARIABLE,
    AD_NODE_ADD,
    AD_NODE_SUB,
    AD_NODE_MUL,
    AD_NODE_DIV,
    AD_NODE_SIN,
    AD_NODE_COS,
    AD_NODE_EXP,
    AD_NODE_LOG,
    AD_NODE_POW,
    AD_NODE_NEG
} ad_node_type_t;

// Computational graph node for reverse-mode AD
// Stores the computational graph for backpropagation
typedef struct ad_node {
    ad_node_type_t type;     // Type of operation
    double value;            // Computed value during forward pass
    double gradient;         // Accumulated gradient during backward pass
    struct ad_node* input1;  // First parent node (null for constants/variables)
    struct ad_node* input2;  // Second parent node (null for unary ops)
    size_t id;              // Unique node ID for topological sorting
} ad_node_t;

// Computational graph tape for recording operations
// Maintains all nodes in evaluation order for backpropagation
typedef struct ad_tape {
    ad_node_t** nodes;       // Array of nodes in evaluation order
    size_t num_nodes;        // Current number of nodes
    size_t capacity;         // Allocated capacity
    ad_node_t** variables;   // Input variable nodes
    size_t num_variables;    // Number of input variables
} ad_tape_t;

// ===== CLOSURE ENVIRONMENT STRUCTURES =====
// Support for lexical closures - capturing parent scope variables

// Closure environment structure (arena-allocated)
// Holds captured variables from parent scope for nested functions
//
// VARIADIC ENCODING: The num_captures field encodes both capture count and variadic info:
//   - Bits 0-15:  num_captures (up to 65535 captures)
//   - Bits 16-31: fixed_param_count (up to 65535 fixed params)
//   - Bit 63:     is_variadic flag (1 = variadic, 0 = not variadic)
//
// Use the macros below to extract/encode these values:
#define CLOSURE_ENV_GET_NUM_CAPTURES(packed) ((packed) & 0xFFFF)
#define CLOSURE_ENV_GET_FIXED_PARAMS(packed) (((packed) >> 16) & 0xFFFF)
#define CLOSURE_ENV_IS_VARIADIC(packed) (((packed) >> 63) & 1)
#define CLOSURE_ENV_PACK(num_caps, fixed_params, is_var) \
    (((size_t)(num_caps) & 0xFFFF) | \
     (((size_t)(fixed_params) & 0xFFFF) << 16) | \
     ((size_t)(is_var) << 63))

typedef struct eshkol_closure_env {
    size_t num_captures;                  // Packed: num_captures | (fixed_params << 16) | (is_variadic << 63)
    eshkol_tagged_value_t captures[];     // Flexible array of captured values
} eshkol_closure_env_t;

// Compile-time size validation
_Static_assert(sizeof(eshkol_closure_env_t) == sizeof(size_t),
               "Closure environment header must be minimal");

// Full closure structure combining function pointer and environment
// This is what gets allocated when a closure-returning function is called
typedef struct eshkol_closure {
    uint64_t func_ptr;                    // Pointer to the lambda function
    eshkol_closure_env_t* env;            // Pointer to captured environment (may be NULL for no captures)
    uint64_t sexpr_ptr;                   // Pointer to S-expression representation for homoiconicity
} eshkol_closure_t;

// ===== END CLOSURE ENVIRONMENT STRUCTURES =====

// ===== LAMBDA REGISTRY FOR HOMOICONICITY =====
// Runtime table mapping function pointers to their S-expression representations
// This enables full homoiconicity: (display (list double)) shows the lambda source

typedef struct eshkol_lambda_entry {
    uint64_t func_ptr;      // Function pointer as uint64
    uint64_t sexpr_ptr;     // Pointer to S-expression cons cell (0 if none)
    const char* name;       // Function name for debugging (may be NULL)
} eshkol_lambda_entry_t;

typedef struct eshkol_lambda_registry {
    eshkol_lambda_entry_t* entries;
    size_t count;
    size_t capacity;
} eshkol_lambda_registry_t;

// Global lambda registry (defined in arena_memory.cpp)
extern eshkol_lambda_registry_t* g_lambda_registry;

// Lambda registry API
void eshkol_lambda_registry_init(void);
void eshkol_lambda_registry_destroy(void);
void eshkol_lambda_registry_add(uint64_t func_ptr, uint64_t sexpr_ptr, const char* name);
uint64_t eshkol_lambda_registry_lookup(uint64_t func_ptr);

// ===== END LAMBDA REGISTRY =====

// ===== UNIFIED DISPLAY SYSTEM =====
// Single source of truth for displaying all Eshkol values

// Forward declaration for tagged cons cell
struct arena_tagged_cons_cell;

// Display options for customizing output
typedef struct eshkol_display_opts {
    int max_depth;          // Maximum recursion depth (default: 100)
    int current_depth;      // Current depth (internal use)
    uint8_t quote_strings;  // Quote strings with "" (true for 'write', false for 'display')
    uint8_t show_types;     // Debug: show type tags
    void* output;           // Output stream (FILE*, default: stdout)
} eshkol_display_opts_t;

// Default display options
static inline eshkol_display_opts_t eshkol_display_default_opts(void) {
    eshkol_display_opts_t opts;
    opts.max_depth = 100;
    opts.current_depth = 0;
    opts.quote_strings = 0;
    opts.show_types = 0;
    opts.output = 0;  // NULL means stdout
    return opts;
}

// Main display functions (implemented in arena_memory.cpp)
void eshkol_display_value(const eshkol_tagged_value_t* value);
void eshkol_display_value_opts(const eshkol_tagged_value_t* value, eshkol_display_opts_t* opts);
void eshkol_write_value(const eshkol_tagged_value_t* value);  // Scheme 'write' semantics

// Display a list (cons cell chain)
void eshkol_display_list(uint64_t cons_ptr, eshkol_display_opts_t* opts);

// Display a lambda by looking up its S-expression in the registry
void eshkol_display_lambda(uint64_t func_ptr, eshkol_display_opts_t* opts);

// Display a closure by extracting its embedded S-expression
void eshkol_display_closure(uint64_t closure_ptr, eshkol_display_opts_t* opts);

// ===== END UNIFIED DISPLAY SYSTEM =====

// ===== END COMPUTATIONAL GRAPH TYPES =====

typedef enum {
    ESHKOL_INVALID_OP,
    ESHKOL_COMPOSE_OP,
    ESHKOL_IF_OP,
    ESHKOL_ADD_OP,
    ESHKOL_SUB_OP,
    ESHKOL_MUL_OP,
    ESHKOL_DIV_OP,
    ESHKOL_CALL_OP,
    ESHKOL_DEFINE_OP,
    ESHKOL_SEQUENCE_OP,
    ESHKOL_EXTERN_OP,
    ESHKOL_EXTERN_VAR_OP,
    ESHKOL_LAMBDA_OP,
    ESHKOL_LET_OP,
    ESHKOL_LET_STAR_OP,  // let* - sequential bindings
    ESHKOL_LETREC_OP,    // letrec - recursive bindings (all bindings visible to all values)
    ESHKOL_AND_OP,       // short-circuit and
    ESHKOL_OR_OP,        // short-circuit or
    ESHKOL_COND_OP,      // multi-branch conditional
    ESHKOL_CASE_OP,      // case expression (switch on value)
    ESHKOL_DO_OP,        // do loop (iteration construct)
    ESHKOL_WHEN_OP,      // when - one-armed if (execute when true)
    ESHKOL_UNLESS_OP,    // unless - negated when (execute when false)
    ESHKOL_QUOTE_OP,     // quote - literal data
    ESHKOL_SET_OP,       // set! - variable mutation
    ESHKOL_IMPORT_OP,    // import - load another Eshkol file (legacy string path)
    ESHKOL_REQUIRE_OP,   // require - import module by symbolic name (new module system)
    ESHKOL_PROVIDE_OP,   // provide - export symbols from module
    // Memory management operators (OALR - Ownership-Aware Lexical Regions)
    ESHKOL_WITH_REGION_OP,  // with-region - lexical region for batch allocation/free
    ESHKOL_OWNED_OP,        // owned - linear type for resources
    ESHKOL_MOVE_OP,         // move - transfer ownership
    ESHKOL_BORROW_OP,       // borrow - temporary read-only access
    ESHKOL_SHARED_OP,       // shared - reference-counted allocation
    ESHKOL_WEAK_REF_OP,     // weak-ref - weak reference (doesn't prevent cleanup)
    ESHKOL_TENSOR_OP,
    ESHKOL_DIFF_OP,
    // Automatic differentiation operators
    ESHKOL_DERIVATIVE_OP,
    ESHKOL_GRADIENT_OP,
    ESHKOL_JACOBIAN_OP,
    ESHKOL_HESSIAN_OP,
    ESHKOL_DIVERGENCE_OP,
    ESHKOL_CURL_OP,
    ESHKOL_LAPLACIAN_OP,
    ESHKOL_DIRECTIONAL_DERIV_OP
} eshkol_op_t;

struct eshkol_ast;
struct eshkol_operation;

typedef struct eshkol_operation {
    eshkol_op_t op;
    union {
        struct {
            struct eshkol_ast *base;
            struct eshkol_ast *ptr;
        } assign_op;
        struct {
            struct eshkol_ast *func_a;
            struct eshkol_ast *func_b;
        } compose_op;
        struct {
            struct eshkol_operation *if_true;
            struct eshkol_operation *if_false;
        } if_op;
        struct {
            struct eshkol_ast *func;
            struct eshkol_ast *variables;
            uint64_t num_vars;
        } call_op;
        struct {
            char *name;
            struct eshkol_ast *value;
            uint8_t is_function;
            struct eshkol_ast *parameters;
            uint64_t num_params;
            uint8_t is_variadic;      // True if function accepts variable arguments
            char *rest_param;         // Name of rest parameter (for variadic functions)
            uint8_t is_external;      // True if function is external (body from linked .o)
        } define_op;
        struct {
            struct eshkol_ast *expressions;
            uint64_t num_expressions;
        } sequence_op;
        struct {
            char *name;
            char *real_name;
            char *return_type;
            struct eshkol_ast *parameters;
            uint64_t num_params;
        } extern_op;
        struct {
            char *name;
            char *type;
        } extern_var_op;
	struct {
	           struct eshkol_ast *parameters;
	           uint64_t num_params;
	           struct eshkol_ast *body;
	           struct eshkol_ast *captured_vars;
	           uint64_t num_captured;
	           uint8_t is_variadic;       // True if lambda accepts variable arguments
	           char *rest_param;          // Name of rest parameter (for variadic lambdas)
	       } lambda_op;
	       struct {
	           struct eshkol_ast *bindings;      // Array of (variable value) pairs
	           uint64_t num_bindings;
	           struct eshkol_ast *body;
	           char *name;                       // Named let: loop name (NULL for regular let)
	       } let_op;
	       struct {
	           char *name;                       // Variable name to mutate
	           struct eshkol_ast *value;         // New value
	       } set_op;
	       struct {
	           char *path;                       // Path to file to import
	       } import_op;
	       struct {
	           char **module_names;              // Array of symbolic module names (e.g., "data.json")
	           uint64_t num_modules;             // Number of modules to require
	       } require_op;
	       struct {
	           char **export_names;              // Array of exported symbol names
	           uint64_t num_exports;             // Number of symbols to export
	       } provide_op;
	       // ===== MEMORY MANAGEMENT OPERATIONS (OALR) =====
	       struct {
	           char *name;                       // Optional region name (NULL for anonymous)
	           uint64_t size_hint;               // Optional size hint in bytes (0 for default)
	           struct eshkol_ast *body;          // Body expressions to execute in region
	           uint64_t num_body_exprs;          // Number of body expressions
	       } with_region_op;
	       struct {
	           struct eshkol_ast *value;         // Value to mark as owned
	       } owned_op;
	       struct {
	           struct eshkol_ast *value;         // Value to transfer ownership of
	       } move_op;
	       struct {
	           struct eshkol_ast *value;         // Value to borrow
	           struct eshkol_ast *body;          // Body expressions during borrow
	           uint64_t num_body_exprs;          // Number of body expressions
	       } borrow_op;
	       struct {
	           struct eshkol_ast *value;         // Value to make shared (ref-counted)
	       } shared_op;
	       struct {
	           struct eshkol_ast *value;         // Shared value to create weak ref from
	       } weak_ref_op;
	       // ===== END MEMORY MANAGEMENT OPERATIONS =====
	       struct {
            struct eshkol_ast *elements;
            uint64_t *dimensions;
            uint64_t num_dimensions;
            uint64_t total_elements;
        } tensor_op;
        struct {
            struct eshkol_ast *expression;  // Expression to differentiate
            char *variable;                 // Variable to differentiate with respect to
        } diff_op;
        struct {
            struct eshkol_ast *function;    // Function to differentiate (lambda or function reference)
            struct eshkol_ast *point;       // Point to evaluate derivative at
            uint8_t mode;                   // 0=forward, 1=reverse, 2=auto (for future use)
        } derivative_op;
        struct {
            struct eshkol_ast *function;    // Scalar field function: ℝⁿ → ℝ
            struct eshkol_ast *point;       // Point to evaluate gradient at
        } gradient_op;
        struct {
            struct eshkol_ast *function;    // Vector field function: ℝⁿ → ℝᵐ
            struct eshkol_ast *point;       // Point to evaluate jacobian at
        } jacobian_op;
        struct {
            struct eshkol_ast *function;    // Scalar field function: ℝⁿ → ℝ
            struct eshkol_ast *point;       // Point to evaluate hessian at
        } hessian_op;
        struct {
            struct eshkol_ast *function;    // Vector field function: ℝⁿ → ℝⁿ
            struct eshkol_ast *point;       // Point to evaluate divergence at
        } divergence_op;
        struct {
            struct eshkol_ast *function;    // Vector field function: ℝ³ → ℝ³
            struct eshkol_ast *point;       // Point to evaluate curl at
        } curl_op;
        struct {
            struct eshkol_ast *function;    // Scalar field function: ℝⁿ → ℝ
            struct eshkol_ast *point;       // Point to evaluate laplacian at
        } laplacian_op;
        struct {
            struct eshkol_ast *function;    // Scalar field function: ℝⁿ → ℝ
            struct eshkol_ast *point;       // Point to evaluate directional derivative at
            struct eshkol_ast *direction;   // Direction vector
        } directional_deriv_op;
    };
} eshkol_operations_t;

typedef struct eshkol_ast {
    eshkol_type_t type;
    union {
        void *untyped_data;
        uint8_t uint8_val;
        uint16_t uint16_val;
        uint32_t uint32_val;
        uint64_t uint64_val;
        int8_t int8_val;
        int16_t int16_val;
        int32_t int32_val;
        int64_t int64_val;
        double double_val;
        struct {
            char *ptr;
            uint64_t size;
        } str_val;
        struct {
            char *id;
            uint8_t is_lambda;
            eshkol_operations_t *func_commands;
            struct eshkol_ast *variables;
            uint64_t num_variables;
            uint64_t size;
            uint8_t is_variadic;      // True if function accepts variable arguments
            char *rest_param;         // Name of rest parameter (for variadic functions)
        } eshkol_func;
        struct {
            char *id;
            struct eshkol_ast *data;
        } variable;
        struct {
            struct eshkol_ast *car;
            struct eshkol_ast *cdr;
        } cons_cell;
        struct {
            struct eshkol_ast *elements;
            uint64_t *dimensions;
            uint64_t num_dimensions;
            uint64_t total_elements;
        } tensor_val;
        eshkol_operations_t operation;
    };
} eshkol_ast_t;

void eshkol_ast_clean(eshkol_ast_t *ast);
void eshkol_ast_pretty_print(const eshkol_ast_t *ast, int indent);

// Symbolic differentiation AST helpers
eshkol_ast_t* eshkol_alloc_symbolic_ast(void);
eshkol_ast_t* eshkol_make_var_ast(const char* name);
eshkol_ast_t* eshkol_make_int_ast(int64_t value);
eshkol_ast_t* eshkol_make_double_ast(double value);
eshkol_ast_t* eshkol_make_binary_op_ast(const char* op, eshkol_ast_t* left, eshkol_ast_t* right);
eshkol_ast_t* eshkol_make_unary_call_ast(const char* func, eshkol_ast_t* arg);
eshkol_ast_t* eshkol_copy_ast(const eshkol_ast_t* ast);

// REPL display helper
eshkol_ast_t* eshkol_wrap_with_display(eshkol_ast_t* expr);

#ifdef __cplusplus
};

// Parse next AST from file stream
eshkol_ast_t eshkol_parse_next_ast(std::ifstream &in_file);

// Parse next AST from any input stream (including string streams for stdlib)
eshkol_ast_t eshkol_parse_next_ast_from_stream(std::istream &in_stream);

#endif

#endif
