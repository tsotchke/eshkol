# Examples

# Define example macro
macro(eshkol_add_example NAME)
    # Parse arguments
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SOURCES DEPS)
    cmake_parse_arguments(EXAMPLE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Create example executable
    add_executable(example_${NAME} ${EXAMPLE_SOURCES})
    
    # Link dependencies
    target_link_libraries(example_${NAME} ${EXAMPLE_DEPS})
    
    # Install example
    install(TARGETS example_${NAME}
        RUNTIME DESTINATION bin/examples
    )
endmacro()

# Hello world example
eshkol_add_example(hello
    SOURCES
        hello.esk.c
    DEPS
        eshkol_backend
)

# Factorial example
eshkol_add_example(factorial
    SOURCES
        factorial.esk.c
    DEPS
        eshkol_backend
)

# Only include examples that have corresponding .esk.c files
# Function composition and arithmetic don't have .esk.c files yet

# # Function composition example
# eshkol_add_example(function_composition
#     SOURCES
#         function_composition.c
#     DEPS
#         eshkol_core
#         eshkol_frontend
#         eshkol_backend
# )

# # Arithmetic example
# eshkol_add_example(arithmetic
#     SOURCES
#         arithmetic.c
#     DEPS
#         eshkol_core
#         eshkol_frontend
#         eshkol_backend
# )

# Automatic differentiation example
eshkol_add_example(autodiff
    SOURCES
        autodiff.c
    DEPS
        eshkol_core
)
