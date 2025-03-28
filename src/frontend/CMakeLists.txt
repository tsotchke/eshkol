# Define sources
set(FRONTEND_SOURCES
    lexer/lexer.c
    ast/ast.c
    ast/ast_visualize.c
    parser/parser.c
    parser/type_parser.c
    ast/parameter.c
    binding/binding.c
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
    ${CMAKE_SOURCE_DIR}/include/frontend/parser/parser.h
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
