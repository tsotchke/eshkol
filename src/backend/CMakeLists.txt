# Define sources
set(BACKEND_SOURCES
    codegen/expressions.c
    codegen/literals.c
    codegen/identifiers.c
    codegen/calls.c
    codegen/conditionals.c
    codegen/blocks.c
    codegen/definitions.c
    codegen/type_conversion.c
    codegen/context.c
    codegen/compiler.c
    codegen/program.c
    codegen/statements.c
    codegen/debug.c
)

# Define headers
set(BACKEND_HEADERS
    ${CMAKE_SOURCE_DIR}/include/backend/codegen.h
    ${CMAKE_SOURCE_DIR}/include/backend/codegen/context.h
    ${CMAKE_SOURCE_DIR}/include/backend/codegen/expressions.h
    ${CMAKE_SOURCE_DIR}/include/backend/codegen/literals.h
    ${CMAKE_SOURCE_DIR}/include/backend/codegen/identifiers.h
    ${CMAKE_SOURCE_DIR}/include/backend/codegen/calls.h
    ${CMAKE_SOURCE_DIR}/include/backend/codegen/conditionals.h
    ${CMAKE_SOURCE_DIR}/include/backend/codegen/blocks.h
    ${CMAKE_SOURCE_DIR}/include/backend/codegen/definitions.h
    ${CMAKE_SOURCE_DIR}/include/backend/codegen/type_conversion.h
    ${CMAKE_SOURCE_DIR}/include/backend/codegen/statements.h
    ${CMAKE_SOURCE_DIR}/include/backend/codegen/compiler.h
    ${CMAKE_SOURCE_DIR}/include/backend/codegen/program.h
    ${CMAKE_SOURCE_DIR}/include/backend/codegen/debug.h
)

# Create library
add_library(eshkol_backend ${BACKEND_SOURCES})

# Set include directories
target_include_directories(eshkol_backend
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set compile definitions
target_compile_definitions(eshkol_backend
    PRIVATE
        ESHKOL_BUILDING_BACKEND
)

# Link dependencies
target_link_libraries(eshkol_backend
    PRIVATE
        eshkol_core
        eshkol_frontend
)

# Install headers
install(FILES ${BACKEND_HEADERS}
    DESTINATION include/eshkol/backend
)

# Install library
install(TARGETS eshkol_backend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
