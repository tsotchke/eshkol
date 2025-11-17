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
    ESHKOL_TENSOR
} eshkol_type_t;

// Mixed type list support - Value type tags for tagged cons cells
typedef enum {
    ESHKOL_VALUE_NULL     = 0,  // Empty/null value
    ESHKOL_VALUE_INT64    = 1,  // 64-bit signed integer
    ESHKOL_VALUE_DOUBLE   = 2,  // Double-precision floating point
    ESHKOL_VALUE_CONS_PTR = 3,  // Pointer to another cons cell
    // Reserved for future expansion
    ESHKOL_VALUE_MAX      = 15  // 4-bit type field limit
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
#define ESHKOL_IS_INT64_TYPE(type)    (((type) & 0x0F) == ESHKOL_VALUE_INT64)
#define ESHKOL_IS_DOUBLE_TYPE(type)   (((type) & 0x0F) == ESHKOL_VALUE_DOUBLE)
#define ESHKOL_IS_CONS_PTR_TYPE(type) (((type) & 0x0F) == ESHKOL_VALUE_CONS_PTR)
#define ESHKOL_IS_NULL_TYPE(type)     (((type) & 0x0F) == ESHKOL_VALUE_NULL)

// Exactness checking macros
#define ESHKOL_IS_EXACT(type)         (((type) & ESHKOL_VALUE_EXACT_FLAG) != 0)
#define ESHKOL_IS_INEXACT(type)       (((type) & ESHKOL_VALUE_INEXACT_FLAG) != 0)

// Type manipulation macros
#define ESHKOL_MAKE_EXACT(type)       ((type) | ESHKOL_VALUE_EXACT_FLAG)
#define ESHKOL_MAKE_INEXACT(type)     ((type) | ESHKOL_VALUE_INEXACT_FLAG)
#define ESHKOL_GET_BASE_TYPE(type)    ((type) & 0x0F)

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
    ESHKOL_TENSOR_OP,
    ESHKOL_DIFF_OP
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
        } lambda_op;
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

#ifdef __cplusplus
};

eshkol_ast_t eshkol_parse_next_ast(std::ifstream &in_file);

#endif

#endif
