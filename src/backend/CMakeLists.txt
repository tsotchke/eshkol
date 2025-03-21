# Define sources
set(BACKEND_SOURCES
    codegen/codegen.c
)

# Define headers
set(BACKEND_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/codegen.h
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
