# Define sources
set(FRONTEND_SOURCES
    lexer/lexer.c
    ast/core/ast_core.c
    ast/create/ast_create.c
    ast/print/ast_print.c
    ast/analysis/ast_analysis.c
    ast/analysis/ast_parent.c
    ast/ast_visualize.c
    parser/parser_core.c
    parser/parser_helpers.c
    parser/parser_error.c
    parser/parser_expressions.c
    parser/parser_define.c
    parser/parser_lambda.c
    parser/parser_conditionals.c
    parser/parser_binding.c
    parser/parser_sequence.c
    parser/parser_quote.c
    parser/type_parser.c
    ast/parameter.c
    binding/binding.c
    binding/core/binding_core.c
    binding/core/binding_scope.c
    binding/core/binding_management.c
    binding/lambda/binding_lambda.c
    binding/environment/binding_environment.c
    binding/analysis/binding_analysis.c
    type_inference/context.c
    type_inference/inference.c
    type_inference/conversion.c
    type_inference/type_inference.c
    type_inference/analysis.c
)

# Define headers
set(FRONTEND_HEADERS
    ${CMAKE_SOURCE_DIR}/include/frontend/lexer/lexer.h
    ${CMAKE_SOURCE_DIR}/include/frontend/ast/ast.h
    ${CMAKE_SOURCE_DIR}/include/frontend/ast/core/ast_core.h
    ${CMAKE_SOURCE_DIR}/include/frontend/ast/create/ast_create.h
    ${CMAKE_SOURCE_DIR}/include/frontend/ast/print/ast_print.h
    ${CMAKE_SOURCE_DIR}/include/frontend/ast/analysis/ast_analysis.h
    ${CMAKE_SOURCE_DIR}/include/frontend/ast/analysis/ast_parent.h
    ${CMAKE_SOURCE_DIR}/include/frontend/parser/parser.h
    ${CMAKE_SOURCE_DIR}/include/frontend/parser/parser_core.h
    ${CMAKE_SOURCE_DIR}/include/frontend/parser/parser_helpers.h
    ${CMAKE_SOURCE_DIR}/include/frontend/parser/parser_error.h
    ${CMAKE_SOURCE_DIR}/include/frontend/parser/parser_expressions.h
    ${CMAKE_SOURCE_DIR}/include/frontend/parser/parser_special_forms.h
    ${CMAKE_SOURCE_DIR}/include/frontend/parser/parser_define.h
    ${CMAKE_SOURCE_DIR}/include/frontend/parser/parser_lambda.h
    ${CMAKE_SOURCE_DIR}/include/frontend/parser/parser_conditionals.h
    ${CMAKE_SOURCE_DIR}/include/frontend/parser/parser_binding.h
    ${CMAKE_SOURCE_DIR}/include/frontend/parser/parser_sequence.h
    ${CMAKE_SOURCE_DIR}/include/frontend/parser/parser_quote.h
    ${CMAKE_SOURCE_DIR}/include/frontend/parser/type_parser.h
    ${CMAKE_SOURCE_DIR}/include/frontend/ast/parameter.h
    ${CMAKE_SOURCE_DIR}/include/frontend/binding/binding.h
    ${CMAKE_SOURCE_DIR}/include/frontend/type_inference.h
    ${CMAKE_SOURCE_DIR}/include/frontend/type_inference/context.h
    ${CMAKE_SOURCE_DIR}/include/frontend/type_inference/inference.h
    ${CMAKE_SOURCE_DIR}/include/frontend/type_inference/conversion.h
    ${CMAKE_SOURCE_DIR}/include/frontend/type_inference/type_inference.h
    ${CMAKE_SOURCE_DIR}/include/frontend/type_inference/analysis.h
)

# Create library
add_library(eshkol_frontend ${FRONTEND_SOURCES})

# Set include directories
target_include_directories(eshkol_frontend
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set compile definitions
target_compile_definitions(eshkol_frontend
    PRIVATE
        ESHKOL_BUILDING_FRONTEND
)

# Link dependencies
target_link_libraries(eshkol_frontend
    PRIVATE
        eshkol_core
)

# Install headers
install(FILES ${FRONTEND_HEADERS}
    DESTINATION include/eshkol/frontend
)

# Install library
install(TARGETS eshkol_frontend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
