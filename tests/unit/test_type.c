#include "core/type.h"
#include "core/type_creation.h"
#include "core/type_comparison.h"
#include "core/type_conversion.h"
#include "core/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Test type creation
static void test_type_creation(Arena* arena) {
    printf("Testing type creation...\n");
    
    // Create primitive types
    Type* void_type = type_void_create(arena);
    Type* bool_type = type_boolean_create(arena);
    Type* int32_type = type_integer_create(arena, INT_SIZE_32);
    Type* float64_type = type_float_create(arena, FLOAT_SIZE_64);
    Type* char_type = type_char_create(arena);
    Type* string_type = type_string_create(arena);
    Type* symbol_type = type_symbol_create(arena);
    
    // Check primitive types
    assert(void_type != NULL);
    assert(void_type->kind == TYPE_VOID);
    
    assert(bool_type != NULL);
    assert(bool_type->kind == TYPE_BOOLEAN);
    
    assert(int32_type != NULL);
    assert(int32_type->kind == TYPE_INTEGER);
    assert(int32_type->int_size == INT_SIZE_32);
    
    assert(float64_type != NULL);
    assert(float64_type->kind == TYPE_FLOAT);
    assert(float64_type->float_size == FLOAT_SIZE_64);
    
    assert(char_type != NULL);
    assert(char_type->kind == TYPE_CHAR);
    
    assert(string_type != NULL);
    assert(string_type->kind == TYPE_STRING);
    
    assert(symbol_type != NULL);
    assert(symbol_type->kind == TYPE_SYMBOL);
    
    // Create compound types
    Type* vector_type = type_vector_create(arena, int32_type, 10);
    
    Type** params = arena_alloc(arena, 2 * sizeof(Type*));
    params[0] = int32_type;
    params[1] = float64_type;
    Type* function_type = type_function_create(arena, 2, params, bool_type, false);
    
    StructField* fields = arena_alloc(arena, 2 * sizeof(StructField));
    fields[0].name = "x";
    fields[0].type = int32_type;
    fields[1].name = "y";
    fields[1].type = float64_type;
    Type* struct_type = type_struct_create(arena, 2, fields);
    
    Type** variants = arena_alloc(arena, 2 * sizeof(Type*));
    variants[0] = int32_type;
    variants[1] = float64_type;
    Type* union_type = type_union_create(arena, 2, variants);
    
    // Check compound types
    assert(vector_type != NULL);
    assert(vector_type->kind == TYPE_VECTOR);
    assert(vector_type->vector.element_type == int32_type);
    assert(vector_type->vector.size == 10);
    
    assert(function_type != NULL);
    assert(function_type->kind == TYPE_FUNCTION);
    assert(function_type->function.param_count == 2);
    assert(function_type->function.params[0] == int32_type);
    assert(function_type->function.params[1] == float64_type);
    assert(function_type->function.return_type == bool_type);
    assert(function_type->function.variadic == false);
    
    assert(struct_type != NULL);
    assert(struct_type->kind == TYPE_STRUCT);
    assert(struct_type->structure.field_count == 2);
    assert(strcmp(struct_type->structure.fields[0].name, "x") == 0);
    assert(struct_type->structure.fields[0].type == int32_type);
    assert(strcmp(struct_type->structure.fields[1].name, "y") == 0);
    assert(struct_type->structure.fields[1].type == float64_type);
    
    assert(union_type != NULL);
    assert(union_type->kind == TYPE_UNION);
    assert(union_type->union_type.variant_count == 2);
    assert(union_type->union_type.variants[0] == int32_type);
    assert(union_type->union_type.variants[1] == float64_type);
    
    printf("Type creation tests passed!\n");
}

// Test type equality
static void test_type_equality(Arena* arena) {
    printf("Testing type equality...\n");
    
    // Create primitive types
    Type* void_type1 = type_void_create(arena);
    Type* void_type2 = type_void_create(arena);
    Type* bool_type = type_boolean_create(arena);
    Type* int32_type1 = type_integer_create(arena, INT_SIZE_32);
    Type* int32_type2 = type_integer_create(arena, INT_SIZE_32);
    Type* int64_type = type_integer_create(arena, INT_SIZE_64);
    
    // Check primitive type equality
    assert(type_equals(void_type1, void_type2));
    assert(!type_equals(void_type1, bool_type));
    assert(type_equals(int32_type1, int32_type2));
    assert(!type_equals(int32_type1, int64_type));
    
    // Create compound types
    Type* vector_type1 = type_vector_create(arena, int32_type1, 10);
    Type* vector_type2 = type_vector_create(arena, int32_type2, 10);
    Type* vector_type3 = type_vector_create(arena, int64_type, 10);
    Type* vector_type4 = type_vector_create(arena, int32_type1, 20);
    
    // Check compound type equality
    assert(type_equals(vector_type1, vector_type2));
    assert(!type_equals(vector_type1, vector_type3));
    assert(!type_equals(vector_type1, vector_type4));
    
    printf("Type equality tests passed!\n");
}

// Test type subtyping
static void test_type_subtyping(Arena* arena) {
    printf("Testing type subtyping...\n");
    
    // Create primitive types
    Type* any_type = type_any_create(arena);
    Type* unknown_type = type_unknown_create(arena);
    Type* int8_type = type_integer_create(arena, INT_SIZE_8);
    Type* int16_type = type_integer_create(arena, INT_SIZE_16);
    Type* int32_type = type_integer_create(arena, INT_SIZE_32);
    Type* float32_type = type_float_create(arena, FLOAT_SIZE_32);
    Type* float64_type = type_float_create(arena, FLOAT_SIZE_64);
    
    // Check primitive type subtyping
    assert(type_is_subtype(int8_type, any_type));
    assert(type_is_subtype(unknown_type, int8_type));
    assert(type_is_subtype(int8_type, int16_type));
    assert(type_is_subtype(int16_type, int32_type));
    assert(!type_is_subtype(int32_type, int16_type));
    assert(type_is_subtype(int32_type, float32_type));
    assert(type_is_subtype(float32_type, float64_type));
    assert(!type_is_subtype(float64_type, float32_type));
    
    // Create function types
    Type** params1 = arena_alloc(arena, 2 * sizeof(Type*));
    params1[0] = int32_type;
    params1[1] = float64_type;
    Type* function_type1 = type_function_create(arena, 2, params1, int16_type, false);
    
    Type** params2 = arena_alloc(arena, 2 * sizeof(Type*));
    params2[0] = int16_type;
    params2[1] = float32_type;
    Type* function_type2 = type_function_create(arena, 2, params2, int32_type, false);
    
    // Check function type subtyping (contravariant parameters, covariant return)
    assert(type_is_subtype(function_type1, function_type2));
    assert(!type_is_subtype(function_type2, function_type1));
    
    printf("Type subtyping tests passed!\n");
}

// Test type common supertype
static void test_type_common_supertype(Arena* arena) {
    printf("Testing type common supertype...\n");
    
    // Create primitive types
    Type* int8_type = type_integer_create(arena, INT_SIZE_8);
    Type* int16_type = type_integer_create(arena, INT_SIZE_16);
    Type* int32_type = type_integer_create(arena, INT_SIZE_32);
    Type* float32_type = type_float_create(arena, FLOAT_SIZE_32);
    Type* float64_type = type_float_create(arena, FLOAT_SIZE_64);
    
    // Check primitive type common supertype
    Type* common1 = type_common_supertype(arena, int8_type, int16_type);
    assert(common1 != NULL);
    assert(common1->kind == TYPE_INTEGER);
    assert(common1->int_size == INT_SIZE_16);
    
    Type* common2 = type_common_supertype(arena, int16_type, int32_type);
    assert(common2 != NULL);
    assert(common2->kind == TYPE_INTEGER);
    assert(common2->int_size == INT_SIZE_32);
    
    Type* common3 = type_common_supertype(arena, int32_type, float32_type);
    assert(common3 != NULL);
    assert(common3->kind == TYPE_FLOAT);
    assert(common3->float_size == FLOAT_SIZE_32);
    
    Type* common4 = type_common_supertype(arena, float32_type, float64_type);
    assert(common4 != NULL);
    assert(common4->kind == TYPE_FLOAT);
    assert(common4->float_size == FLOAT_SIZE_64);
    
    printf("Type common supertype tests passed!\n");
}

// Test type to string conversion
static void test_type_to_string(Arena* arena) {
    printf("Testing type to string conversion...\n");
    
    // Create primitive types
    Type* void_type = type_void_create(arena);
    Type* bool_type = type_boolean_create(arena);
    Type* int32_type = type_integer_create(arena, INT_SIZE_32);
    Type* float64_type = type_float_create(arena, FLOAT_SIZE_64);
    
    // Check primitive type to string
    char* void_str = type_to_string(arena, void_type);
    assert(void_str != NULL);
    assert(strcmp(void_str, "void") == 0);
    
    char* bool_str = type_to_string(arena, bool_type);
    assert(bool_str != NULL);
    assert(strcmp(bool_str, "boolean") == 0);
    
    char* int32_str = type_to_string(arena, int32_type);
    assert(int32_str != NULL);
    assert(strcmp(int32_str, "int32") == 0);
    
    char* float64_str = type_to_string(arena, float64_type);
    assert(float64_str != NULL);
    assert(strcmp(float64_str, "float64") == 0);
    
    // Create compound types
    Type* vector_type = type_vector_create(arena, int32_type, 10);
    
    Type** params = arena_alloc(arena, 2 * sizeof(Type*));
    params[0] = int32_type;
    params[1] = float64_type;
    Type* function_type = type_function_create(arena, 2, params, bool_type, false);
    
    // Check compound type to string
    char* vector_str = type_to_string(arena, vector_type);
    assert(vector_str != NULL);
    assert(strcmp(vector_str, "(Vector int32 10)") == 0);
    
    char* function_str = type_to_string(arena, function_type);
    assert(function_str != NULL);
    assert(strcmp(function_str, "(-> (int32 float64) boolean)") == 0);
    
    printf("Type to string conversion tests passed!\n");
}

// Test type from string parsing
static void test_type_from_string(Arena* arena) {
    printf("Testing type from string parsing...\n");
    
    // Parse primitive types
    Type* void_type = type_from_string(arena, "void");
    assert(void_type != NULL);
    assert(void_type->kind == TYPE_VOID);
    
    Type* bool_type = type_from_string(arena, "boolean");
    assert(bool_type != NULL);
    assert(bool_type->kind == TYPE_BOOLEAN);
    
    Type* int32_type = type_from_string(arena, "int32");
    assert(int32_type != NULL);
    assert(int32_type->kind == TYPE_INTEGER);
    assert(int32_type->int_size == INT_SIZE_32);
    
    Type* float64_type = type_from_string(arena, "float64");
    assert(float64_type != NULL);
    assert(float64_type->kind == TYPE_FLOAT);
    assert(float64_type->float_size == FLOAT_SIZE_64);
    
    // Parse compound types
    Type* vector_type = type_from_string(arena, "(Vector int32 10)");
    assert(vector_type != NULL);
    assert(vector_type->kind == TYPE_VECTOR);
    assert(vector_type->vector.element_type->kind == TYPE_INTEGER);
    assert(vector_type->vector.element_type->int_size == INT_SIZE_32);
    assert(vector_type->vector.size == 10);
    
    printf("Type from string parsing tests passed!\n");
}

int main(void) {
    printf("Running type system tests...\n");
    
    // Create memory arena
    Arena* arena = arena_create(1024 * 1024);
    if (!arena) {
        fprintf(stderr, "Failed to create memory arena\n");
        return 1;
    }
    
    // Run tests
    test_type_creation(arena);
    test_type_equality(arena);
    test_type_subtyping(arena);
    test_type_common_supertype(arena);
    test_type_to_string(arena);
    test_type_from_string(arena);
    
    // Clean up
    arena_destroy(arena);
    
    printf("All type system tests passed!\n");
    return 0;
}
