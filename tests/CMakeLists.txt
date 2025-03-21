# Add subdirectories
add_subdirectory(unit)
add_subdirectory(integration)

# Add test includes
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
)

# Define test macros
macro(eshkol_add_test NAME)
    # Parse arguments
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SOURCES DEPS)
    cmake_parse_arguments(TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Create test executable
    add_executable(test_${NAME} ${TEST_SOURCES})
    
    # Link dependencies
    target_link_libraries(test_${NAME} ${TEST_DEPS})
    
    # Add test
    add_test(NAME ${NAME} COMMAND test_${NAME})
endmacro()
