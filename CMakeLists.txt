cmake_minimum_required(VERSION 3.14)
project(Eshkol VERSION 0.1.0 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
endif()

# Options
option(ESHKOL_BUILD_TESTS "Build tests" ON)
option(ESHKOL_BUILD_EXAMPLES "Build examples" OFF)
option(ESHKOL_BUILD_BACKEND "Build backend" ON)
option(ESHKOL_BUILD_INTEGRATION_TESTS "Build integration tests" OFF)
option(ESHKOL_ENABLE_ASAN "Enable Address Sanitizer" OFF)
option(ESHKOL_ENABLE_UBSAN "Enable Undefined Behavior Sanitizer" OFF)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add subdirectories
add_subdirectory(src)

if(ESHKOL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ESHKOL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
