/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TaggedValueCodegen - LLVM code generation for tagged value operations
 *
 * This module handles packing and unpacking values to/from the eshkol_tagged_value_t
 * struct type. It provides the foundation for Eshkol's polymorphic type system.
 *
 * The tagged_value_t struct layout (16 bytes):
 *   - offset 0: uint8_t type      (value type tag)
 *   - offset 1: uint8_t flags     (exactness, special flags)
 *   - offset 2: uint16_t reserved (alignment padding)
 *   - offset 4: uint32_t padding  (ensure 8-byte alignment for data)
 *   - offset 8: int64_t/double/ptr data (actual value storage)
 */
#ifndef ESHKOL_BACKEND_TAGGED_VALUE_CODEGEN_H
#define ESHKOL_BACKEND_TAGGED_VALUE_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Constants.h>

#include "eshkol/eshkol.h"

namespace eshkol {

/**
 * TaggedValueCodegen handles packing/unpacking of tagged values.
 *
 * Tagged values are the runtime representation of all Eshkol values.
 * This class provides methods to:
 * - Pack primitive values (int64, double, bool, char, null) into tagged form
 * - Pack pointer values (cons, vector, tensor, string, closure) into tagged form
 * - Unpack tagged values back to primitive LLVM types
 * - Query type tags and flags from tagged values
 * - Safely extract i64 from various value types
 */
class TaggedValueCodegen {
public:
    /**
     * Construct TaggedValueCodegen with a CodegenContext reference.
     */
    explicit TaggedValueCodegen(CodegenContext& ctx);

    // === Pack Functions ===
    // These create tagged_value structs from primitive LLVM values

    /**
     * Pack an i64 integer into a tagged value.
     * @param int64_val The LLVM i64 value to pack
     * @param is_exact Whether the number is exact (Scheme exactness)
     * @return A tagged_value struct containing the integer
     */
    llvm::Value* packInt64(llvm::Value* int64_val, bool is_exact = true);

    /**
     * Pack an i64 with explicit type tag.
     * Used for values like ports where the i64 represents a pointer
     * but we need a specific type tag (not INT64).
     * @param int64_val The LLVM i64 value to pack
     * @param type The value type (e.g., ESHKOL_VALUE_CONS_PTR | 0x10 for input port)
     * @param flags Optional flags (default 0)
     * @return A tagged_value struct
     */
    llvm::Value* packInt64WithType(llvm::Value* int64_val,
                                    eshkol_value_type_t type,
                                    uint8_t flags = 0);

    /**
     * Pack a boolean into a tagged value.
     * @param bool_val The LLVM i1 value to pack
     * @return A tagged_value struct containing the boolean
     */
    llvm::Value* packBool(llvm::Value* bool_val);

    /**
     * Pack an i64 with explicit type and flags (for car/cdr wrappers).
     * @param int64_val The LLVM i64 value to pack
     * @param type_val The LLVM i8 type tag
     * @param flags_val The LLVM i8 flags value
     * @return A tagged_value struct
     */
    llvm::Value* packInt64WithTypeAndFlags(llvm::Value* int64_val,
                                            llvm::Value* type_val,
                                            llvm::Value* flags_val);

    /**
     * Pack a double into a tagged value.
     * @param double_val The LLVM double value to pack
     * @return A tagged_value struct containing the double
     */
    llvm::Value* packDouble(llvm::Value* double_val);

    /**
     * Pack a pointer into a tagged value with static type.
     * @param ptr_val The LLVM pointer or i64 value
     * @param type The value type (e.g., ESHKOL_VALUE_CONS_PTR)
     * @param flags Optional flags (default 0)
     * @return A tagged_value struct containing the pointer
     */
    llvm::Value* packPtr(llvm::Value* ptr_val,
                         eshkol_value_type_t type,
                         uint8_t flags = 0);

    /**
     * Pack a pointer with dynamic (runtime) type and flags.
     * @param ptr_val The LLVM pointer or i64 value
     * @param type_val The LLVM i8 type tag (runtime value)
     * @param flags_val The LLVM i8 flags (runtime value)
     * @return A tagged_value struct
     */
    llvm::Value* packPtrWithFlags(llvm::Value* ptr_val,
                                   llvm::Value* type_val,
                                   llvm::Value* flags_val);

    /**
     * Pack a null value into a tagged value.
     * @return A tagged_value struct with NULL type
     */
    llvm::Value* packNull();

    /**
     * Pack a character (Unicode codepoint) into a tagged value.
     * @param char_val The LLVM i64 or smaller integer value
     * @return A tagged_value struct containing the character
     */
    llvm::Value* packChar(llvm::Value* char_val);

    // === Unpack Functions ===
    // These extract primitive values from tagged_value structs

    /**
     * Get the type tag from a tagged value.
     * @param tagged_val The tagged_value struct
     * @return The i8 type tag
     */
    llvm::Value* getType(llvm::Value* tagged_val);

    /**
     * Get the flags from a tagged value.
     * @param tagged_val The tagged_value struct
     * @return The i8 flags value
     */
    llvm::Value* getFlags(llvm::Value* tagged_val);

    /**
     * Unpack an i64 from a tagged value.
     * @param tagged_val The tagged_value struct
     * @return The i64 data field
     */
    llvm::Value* unpackInt64(llvm::Value* tagged_val);

    /**
     * Unpack a double from a tagged value.
     * @param tagged_val The tagged_value struct
     * @return The double value (bitcast from i64)
     */
    llvm::Value* unpackDouble(llvm::Value* tagged_val);

    /**
     * Unpack a pointer from a tagged value.
     * @param tagged_val The tagged_value struct
     * @return The pointer (converted from i64)
     */
    llvm::Value* unpackPtr(llvm::Value* tagged_val);

    // === Utility Functions ===

    /**
     * Safely extract i64 from various value types.
     * Handles: i64, tagged_value, other integers, pointers, floats.
     * Used to prevent ICmp type mismatch assertions.
     * @param val Any LLVM value
     * @return An i64 value
     */
    llvm::Value* safeExtractInt64(llvm::Value* val);

    /**
     * Check if a value is a tagged_value struct type.
     * @param val The LLVM value to check
     * @return true if val has tagged_value_type
     */
    bool isTaggedValue(llvm::Value* val) const;

private:
    CodegenContext& ctx_;

    /**
     * Helper to create alloca at function entry for SSA dominance.
     * Saves current insert point, creates alloca at entry, restores insert point.
     * @param name Name for the alloca instruction
     * @return Pointer to the allocated tagged_value
     */
    llvm::Value* createEntryAlloca(const char* name);
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_TAGGED_VALUE_CODEGEN_H
