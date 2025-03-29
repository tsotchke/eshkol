# Unit tests

# Memory tests
eshkol_add_test(arena
    SOURCES
        test_arena.c
    DEPS
        eshkol_core
)

# Object pool tests
eshkol_add_test(object_pool
    SOURCES
        test_object_pool.c
    DEPS
        eshkol_core
)

# Memory tracking tests
eshkol_add_test(memory_tracking
    SOURCES
        test_memory_tracking.c
    DEPS
        eshkol_core
)

# Dynamic array tests
eshkol_add_test(dynamic_array
    SOURCES
        test_dynamic_array.c
    DEPS
        eshkol_core
)

# File I/O tests
eshkol_add_test(file_io
    SOURCES
        test_file_io.c
    DEPS
        eshkol_core
)

# Vector and SIMD tests
eshkol_add_test(vector
    SOURCES
        test_vector.c
    DEPS
        eshkol_core
)

# Automatic differentiation tests
eshkol_add_test(autodiff
    SOURCES
        test_autodiff.c
    DEPS
        eshkol_core
)

# Type system tests
eshkol_add_test(type
    SOURCES
        test_type.c
    DEPS
        eshkol_core
)

# String table tests
eshkol_add_test(string_table
    SOURCES
        test_string_table.c
    DEPS
        eshkol_core
)

# List tests
eshkol_add_test(list
    SOURCES
        test_list.c
    DEPS
        eshkol_core
)

# Diagnostics tests
eshkol_add_test(diagnostics
    SOURCES
        test_diagnostics.c
    DEPS
        eshkol_core
)

# Lexer tests
eshkol_add_test(lexer
    SOURCES
        test_lexer.c
    DEPS
        eshkol_core
        eshkol_frontend
)

# Parser tests
eshkol_add_test(parser
    SOURCES
        test_parser.c
    DEPS
        eshkol_core
        eshkol_frontend
)

# AST tests
eshkol_add_test(ast
    SOURCES
        test_ast.c
    DEPS
        eshkol_core
        eshkol_frontend
)

# Code generation tests
if(ESHKOL_BUILD_BACKEND)
    eshkol_add_test(code_generator
        SOURCES
            test_codegen.c
        DEPS
            eshkol_core
            eshkol_frontend
            eshkol_backend
    )
endif()
