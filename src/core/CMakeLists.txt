# Define sources
set(CORE_SOURCES
    memory/arena.c
    memory/object_pool.c
    memory/memory_tracking.c
    utils/string_table.c
    utils/dynamic_array.c
    diagnostics/diagnostics.c
)

# Define headers
set(CORE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/arena.h
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/object_pool.h
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/memory_tracking.h
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/string_table.h
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/dynamic_array.h
    ${CMAKE_CURRENT_SOURCE_DIR}/diagnostics/diagnostics.h
)

# Create library
add_library(eshkol_core ${CORE_SOURCES})

# Set include directories
target_include_directories(eshkol_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set compile definitions
target_compile_definitions(eshkol_core
    PRIVATE
        ESHKOL_BUILDING_CORE
)

# Install headers
install(FILES ${CORE_HEADERS}
    DESTINATION include/eshkol/core
)

# Install library
install(TARGETS eshkol_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
