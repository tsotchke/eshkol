# Define sources
set(FRONTEND_SOURCES
    lexer/lexer.c
    parser/parser.c
    ast/ast.c
)

# Define headers
set(FRONTEND_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/lexer/lexer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/parser/parser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ast/ast.h
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
