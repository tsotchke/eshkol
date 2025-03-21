# Integration tests

# Simple compilation test
eshkol_add_test(simple_compile
    SOURCES
        test_simple_compile.c
    DEPS
        eshkol_core
        eshkol_frontend
        eshkol_backend
)

# Function definition test
eshkol_add_test(function_definition
    SOURCES
        test_function_definition.c
    DEPS
        eshkol_core
        eshkol_frontend
        eshkol_backend
)

# Control flow test
eshkol_add_test(control_flow
    SOURCES
        test_control_flow.c
    DEPS
        eshkol_core
        eshkol_frontend
        eshkol_backend
)

# Factorial test
eshkol_add_test(factorial
    SOURCES
        test_factorial.c
    DEPS
        eshkol_core
        eshkol_frontend
        eshkol_backend
)

# End-to-end compilation test
eshkol_add_test(end_to_end
    SOURCES
        test_end_to_end.c
    DEPS
        eshkol_core
        eshkol_frontend
        eshkol_backend
)
