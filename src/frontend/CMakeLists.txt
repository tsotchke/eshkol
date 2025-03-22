# Define sources
set(FRONTEND_SOURCES
    lexer/lexer.c
    ast/ast.c
    parser/parser.c
)

# Define headers
set(FRONTEND_HEADERS
    ${CMAKE_SOURCE_DIR}/include/frontend/lexer/lexer.h
    ${CMAKE_SOURCE_DIR}/include/frontend/ast/ast.h
    ${CMAKE_SOURCE_DIR}/include/frontend/parser/parser.h
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
