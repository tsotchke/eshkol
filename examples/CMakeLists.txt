# Examples

# Define example macro for Eshkol source files
macro(eshkol_add_example NAME)
    # Parse arguments
    set(options "")
    set(oneValueArgs ESK_SOURCE)
    set(multiValueArgs DEPS)
    cmake_parse_arguments(EXAMPLE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Define the output C file
    set(OUTPUT_C_FILE "${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_ESK_SOURCE}.c")
    
    # Add custom command to generate C file from Eshkol source
    add_custom_command(
        OUTPUT ${OUTPUT_C_FILE}
        COMMAND $<TARGET_FILE:eshkol> ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_ESK_SOURCE}
        DEPENDS eshkol ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_ESK_SOURCE}
        COMMENT "Compiling ${EXAMPLE_ESK_SOURCE} to C"
        VERBATIM
    )
    
    # Create example executable
    add_executable(example_${NAME} ${OUTPUT_C_FILE})
    
    # Link dependencies
    target_link_libraries(example_${NAME} ${EXAMPLE_DEPS})
    
    # Install example
    install(TARGETS example_${NAME}
        RUNTIME DESTINATION bin/examples
    )
endmacro()

# Hello world example
# eshkol_add_example(hello
#     ESK_SOURCE hello.esk
#     DEPS eshkol_backend
# )

# Factorial example
# eshkol_add_example(factorial
#     ESK_SOURCE factorial.esk
#     DEPS eshkol_backend
# )

# Autodiff example
# eshkol_add_example(autodiff_example
#     ESK_SOURCE autodiff_example.esk
#     DEPS eshkol_backend
# )

# Vector calculus example
# eshkol_add_example(vector_calculus
#     ESK_SOURCE vector_calculus.esk
#     DEPS eshkol_backend
# )

# These examples are commented out until they're ready to be included

# # Function composition example
# eshkol_add_example(function_composition
#     ESK_SOURCE function_composition.esk
#     DEPS eshkol_backend
# )

# # Arithmetic example
# eshkol_add_example(arithmetic
#     ESK_SOURCE arithmetic.esk
#     DEPS eshkol_backend
# )

# Define example macro for C source files
macro(eshkol_add_c_example NAME)
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

# # Automatic differentiation example
# eshkol_add_c_example(autodiff
#     SOURCES autodiff.c
#     DEPS eshkol_core
# )
