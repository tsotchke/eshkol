#
# Copyright (C) tsotchke
#
# SPDX-License-Identifier: MIT
#
cmake_minimum_required(VERSION 3.14)

set(ESHKOL_VERSION "1.0.0")

project("Eshkol" VERSION ${ESHKOL_VERSION} LANGUAGES C CXX)

include(CTest)
include(GNUInstallDirs)
include(FindPkgConfig)

# Set C standard
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)

# Options
option(ESHKOL_BUILD_TESTS "Build tests" ON)
option(ESHKOL_BUILD_EXAMPLES "Build examples" OFF)
option(ESHKOL_BUILD_INTEGRATION_TESTS "Build integration tests" OFF)
option(ESHKOL_ENABLE_ASAN "Enable Address Sanitizer" OFF)
option(ESHKOL_ENABLE_UBSAN "Enable Undefined Behavior Sanitizer" OFF)

# XLA Backend Support (optional, default OFF)
# When enabled, provides accelerated tensor operations via XLA/StableHLO
# Requires MLIR and StableHLO libraries
option(ESHKOL_XLA_ENABLED "Enable XLA backend for accelerated tensor operations" OFF)

# LLVM backend is always enabled
set(ESHKOL_BUILD_BACKEND ON)

# LLVM Configuration - Always required
find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config REQUIRED)

# Get LLVM configuration
execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --cxxflags OUTPUT_VARIABLE LLVM_CXXFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --ldflags OUTPUT_VARIABLE LLVM_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --libs all OUTPUT_VARIABLE LLVM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --system-libs OUTPUT_VARIABLE LLVM_SYSTEM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)

# Enhanced LLVM include directory detection (backwards-compatible)
# Extract include directories from CXXFLAGS for better cross-platform support
string(REGEX MATCHALL "-I[^ ]+" LLVM_INCLUDE_FLAGS "${LLVM_CXXFLAGS}")
set(LLVM_INCLUDE_DIRS "")
foreach(include_flag ${LLVM_INCLUDE_FLAGS})
    string(REGEX REPLACE "^-I" "" include_dir "${include_flag}")
    list(APPEND LLVM_INCLUDE_DIRS "${include_dir}")
endforeach()

# Fallback: Try --includedir if no includes found in CXXFLAGS
if(NOT LLVM_INCLUDE_DIRS)
    execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --includedir 
                   OUTPUT_VARIABLE LLVM_INCLUDE_DIR 
                   OUTPUT_STRIP_TRAILING_WHITESPACE
                   ERROR_QUIET)
    if(LLVM_INCLUDE_DIR AND EXISTS "${LLVM_INCLUDE_DIR}")
        list(APPEND LLVM_INCLUDE_DIRS "${LLVM_INCLUDE_DIR}")
    endif()
endif()

# Additional fallback for common LLVM installation patterns
if(NOT LLVM_INCLUDE_DIRS)
    # Try to derive include path from llvm-config executable location
    get_filename_component(LLVM_CONFIG_DIR "${LLVM_CONFIG_EXECUTABLE}" DIRECTORY)
    get_filename_component(LLVM_ROOT "${LLVM_CONFIG_DIR}" DIRECTORY)
    set(POTENTIAL_INCLUDE_DIRS
        "${LLVM_ROOT}/include"
        "${LLVM_ROOT}/include/llvm"
        "/usr/include/llvm"
        "/usr/local/include/llvm"
    )
    foreach(potential_dir ${POTENTIAL_INCLUDE_DIRS})
        if(EXISTS "${potential_dir}/llvm/IR/LLVMContext.h")
            list(APPEND LLVM_INCLUDE_DIRS "${potential_dir}")
            break()
        endif()
    endforeach()
endif()

# Convert to CMake format
separate_arguments(LLVM_CXXFLAGS_LIST UNIX_COMMAND "${LLVM_CXXFLAGS}")
separate_arguments(LLVM_LDFLAGS_LIST UNIX_COMMAND "${LLVM_LDFLAGS}")
separate_arguments(LLVM_LIBS_LIST UNIX_COMMAND "${LLVM_LIBS}")
separate_arguments(LLVM_SYSTEM_LIBS_LIST UNIX_COMMAND "${LLVM_SYSTEM_LIBS}")

# Debug output for troubleshooting (can be removed in production)
message(STATUS "LLVM Config: ${LLVM_CONFIG_EXECUTABLE}")
message(STATUS "LLVM Include Dirs: ${LLVM_INCLUDE_DIRS}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "LLVM CXXFLAGS: ${LLVM_CXXFLAGS}")
endif()

# ===== XLA Backend Configuration (Optional) =====
if(ESHKOL_XLA_ENABLED)
    message(STATUS "XLA backend: ENABLED")

    # XLA source files list (for verification)
    set(XLA_SOURCES
        lib/backend/xla/xla_codegen.cpp
        lib/backend/xla/stablehlo_emitter.cpp
        lib/backend/xla/xla_compiler.cpp
        lib/backend/xla/xla_runtime.cpp
        lib/backend/xla/xla_types.cpp
        lib/backend/xla/xla_memory.cpp
    )

    # Verify all XLA source files exist
    set(XLA_FILES_EXIST TRUE)
    foreach(xla_src ${XLA_SOURCES})
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${xla_src}")
            set(XLA_FILES_EXIST FALSE)
            message(STATUS "XLA stub file missing: ${xla_src}")
        endif()
    endforeach()

    if(XLA_FILES_EXIST)
        # Note: XLA sources are included via GLOB_RECURSE when enabled
        message(STATUS "XLA sources found and will be included in build")
    else()
        message(WARNING "XLA enabled but source files not found. Create stub files first.")
    endif()

    # Define XLA enabled for code
    add_compile_definitions(ESHKOL_XLA_ENABLED)
else()
    message(STATUS "XLA backend: DISABLED (enable with -DESHKOL_XLA_ENABLED=ON)")
endif()

# ===== BLAS Acceleration (Platform-specific) =====
# Provides optimized matrix operations via platform BLAS libraries
option(ESHKOL_BLAS_ENABLED "Enable BLAS acceleration for matrix operations" ON)

if(ESHKOL_BLAS_ENABLED)
    set(BLAS_FOUND FALSE)
    set(BLAS_LIBRARIES "")

    if(APPLE)
        # Use Apple Accelerate framework (built into macOS)
        # Provides highly optimized BLAS for Apple Silicon (NEON, AMX)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if(ACCELERATE_FRAMEWORK)
            set(BLAS_FOUND TRUE)
            set(BLAS_LIBRARIES ${ACCELERATE_FRAMEWORK})
            add_compile_definitions(ESHKOL_BLAS_ACCELERATE)
            message(STATUS "BLAS: Apple Accelerate framework")
        endif()
    else()
        # Try OpenBLAS on Linux/other platforms
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            add_compile_definitions(ESHKOL_BLAS_OPENBLAS)
            message(STATUS "BLAS: ${BLAS_LIBRARIES}")
        else()
            # Try pkg-config for OpenBLAS
            pkg_check_modules(OPENBLAS QUIET openblas)
            if(OPENBLAS_FOUND)
                set(BLAS_FOUND TRUE)
                set(BLAS_LIBRARIES ${OPENBLAS_LIBRARIES})
                add_compile_definitions(ESHKOL_BLAS_OPENBLAS)
                message(STATUS "BLAS: OpenBLAS via pkg-config")
            endif()
        endif()
    endif()

    if(BLAS_FOUND)
        add_compile_definitions(ESHKOL_BLAS_ENABLED)
        message(STATUS "BLAS acceleration: ENABLED")
    else()
        message(STATUS "BLAS acceleration: DISABLED (no BLAS library found)")
        set(ESHKOL_BLAS_ENABLED OFF)
    endif()
else()
    message(STATUS "BLAS acceleration: DISABLED (enable with -DESHKOL_BLAS_ENABLED=ON)")
endif()

# Make uninstall.
add_custom_target("uninstall" COMMENT "Uninstall installed files")
add_custom_command(
    TARGET "uninstall"
    POST_BUILD
    COMMENT "Uninstall files with install_manifest.txt"
    COMMAND xargs rm -vf < install_manifest.txt || echo Nothing in
            install_manifest.txt to be uninstalled!
)

# Debug repos
add_custom_target("debug" COMMENT "Creates all debug files for the supported operating systems.")
add_custom_command(
  TARGET "debug"
  POST_BUILD
  COMMENT "Creates the debian files for building the debug files."
  COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && ${CMAKE_CURRENT_SOURCE_DIR}/scripts/make-docker.sh ${CMAKE_CURRENT_BINARY_DIR} ${ESHKOL_VERSION} debug
)

# Get all the source code recursively.
file(GLOB_RECURSE EXE_SRC RELATIVE ${CMAKE_SOURCE_DIR} "exe/*.cpp")
file(GLOB_RECURSE LIB_SRC RELATIVE ${CMAKE_SOURCE_DIR} "lib/*.c*")

# Exclude REPL sources from main library (REPL will be separate)
list(FILTER LIB_SRC EXCLUDE REGEX "lib/repl/.*")

# Exclude XLA sources from main library when XLA is disabled
if(NOT ESHKOL_XLA_ENABLED)
    list(FILTER LIB_SRC EXCLUDE REGEX "lib/backend/xla/.*")
endif()

# Get REPL sources separately
file(GLOB REPL_SRC RELATIVE ${CMAKE_SOURCE_DIR} "lib/repl/*.cpp")

add_compile_definitions(
  ESHKOL_VER="${ESHKOL_VERSION}"
)

# Create static library for executables.
add_library(eshkol-static STATIC ${LIB_SRC})
target_include_directories(
  eshkol-static
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/>
)

# Always add LLVM support to static library
target_compile_options(eshkol-static PRIVATE ${LLVM_CXXFLAGS})
target_compile_definitions(eshkol-static PRIVATE ESHKOL_LLVM_BACKEND_ENABLED)

# Add LLVM include directories for enhanced cross-platform support
if(LLVM_INCLUDE_DIRS)
    target_include_directories(eshkol-static PRIVATE ${LLVM_INCLUDE_DIRS})
endif()

# Extract library directories from LDFLAGS for proper linking
string(REGEX MATCHALL "-L[^ ]+" LLVM_LIBRARY_DIR_FLAGS "${LLVM_LDFLAGS}")
set(LLVM_LIBRARY_DIRS "")
foreach(lib_dir_flag ${LLVM_LIBRARY_DIR_FLAGS})
    string(REGEX REPLACE "^-L" "" lib_dir "${lib_dir_flag}")
    list(APPEND LLVM_LIBRARY_DIRS "${lib_dir}")
endforeach()

# Add library directories if found
if(LLVM_LIBRARY_DIRS)
    target_link_directories(eshkol-static PRIVATE ${LLVM_LIBRARY_DIRS})
endif()

# Link BLAS libraries if available
if(ESHKOL_BLAS_ENABLED AND BLAS_LIBRARIES)
    target_link_libraries(eshkol-static PUBLIC ${BLAS_LIBRARIES})
endif()

# For each executable inside the executable files create the executable with LLVM backend.
set(EXES)
foreach (exe ${EXE_SRC})
  get_filename_component(exename ${exe} NAME_WE)

  # Skip eshkol-repl - it's handled separately below
  if(exename STREQUAL "eshkol-repl")
    continue()
  endif()

  list(APPEND EXES ${exename})

  # Compile main executable source as object file
  add_executable(${exename} ${exe})
  target_compile_options(${exename} PRIVATE ${LLVM_CXXFLAGS})
  target_compile_definitions(${exename} PRIVATE ESHKOL_LLVM_BACKEND_ENABLED)
  target_include_directories(${exename} PRIVATE inc/)
  
  # Add LLVM include directories for enhanced cross-platform support
  if(LLVM_INCLUDE_DIRS)
      target_include_directories(${exename} PRIVATE ${LLVM_INCLUDE_DIRS})
  endif()
  
  # Add library directories for executables
  if(LLVM_LIBRARY_DIRS)
      target_link_directories(${exename} PRIVATE ${LLVM_LIBRARY_DIRS})
  endif()
  
  target_link_libraries(${exename} PUBLIC eshkol-static ${LLVM_LIBS} ${LLVM_SYSTEM_LIBS})
endforeach ()

# ===== Pre-compiled Standard Library =====
# Compile stdlib.esk to stdlib.o for faster compilation of user programs
set(STDLIB_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/lib/stdlib.esk")
set(STDLIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/stdlib.o")

add_custom_command(
  OUTPUT ${STDLIB_OUTPUT}
  COMMAND $<TARGET_FILE:eshkol-run> --shared-lib -o "${CMAKE_CURRENT_BINARY_DIR}/stdlib" "${STDLIB_SOURCE}"
  DEPENDS eshkol-run ${STDLIB_SOURCE}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Compiling standard library to ${STDLIB_OUTPUT}"
  VERBATIM
)

add_custom_target(stdlib ALL DEPENDS ${STDLIB_OUTPUT})

# Install stdlib.o to lib directory
install(FILES ${STDLIB_OUTPUT} DESTINATION ${CMAKE_INSTALL_LIBDIR}/eshkol)

message(STATUS "Standard library will be pre-compiled to: ${STDLIB_OUTPUT}")

# ===== REPL Support =====
# Optional REPL with JIT execution

option(BUILD_REPL "Build the interactive REPL" ON)

if(BUILD_REPL AND REPL_SRC)
  # REPL library
  add_library(eshkol-repl-lib STATIC ${REPL_SRC})

  target_include_directories(eshkol-repl-lib
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/repl
  )

  # REPL needs LLVM ORC JIT components
  target_compile_options(eshkol-repl-lib PRIVATE ${LLVM_CXXFLAGS})
  target_compile_definitions(eshkol-repl-lib PRIVATE ESHKOL_LLVM_BACKEND_ENABLED)

  if(LLVM_INCLUDE_DIRS)
    target_include_directories(eshkol-repl-lib PRIVATE ${LLVM_INCLUDE_DIRS})
  endif()

  if(LLVM_LIBRARY_DIRS)
    target_link_directories(eshkol-repl-lib PRIVATE ${LLVM_LIBRARY_DIRS})
  endif()

  # Link against main compiler library and LLVM
  target_link_libraries(eshkol-repl-lib PUBLIC eshkol-static ${LLVM_LIBS} ${LLVM_SYSTEM_LIBS})

  # REPL executable (only if we have the source)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/exe/eshkol-repl.cpp")
    add_executable(eshkol-repl exe/eshkol-repl.cpp)

    target_compile_options(eshkol-repl PRIVATE ${LLVM_CXXFLAGS})
    target_compile_definitions(eshkol-repl PRIVATE ESHKOL_LLVM_BACKEND_ENABLED)
    target_include_directories(eshkol-repl PRIVATE inc/)

    if(LLVM_INCLUDE_DIRS)
      target_include_directories(eshkol-repl PRIVATE ${LLVM_INCLUDE_DIRS})
    endif()

    if(LLVM_LIBRARY_DIRS)
      target_link_directories(eshkol-repl PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()

    # Find readline library (optional but recommended)
    find_library(READLINE_LIBRARY readline)
    if(READLINE_LIBRARY)
      message(STATUS "Found readline: ${READLINE_LIBRARY}")
      target_link_libraries(eshkol-repl PRIVATE ${READLINE_LIBRARY})
      target_compile_definitions(eshkol-repl PRIVATE HAVE_READLINE)
    else()
      message(WARNING "readline not found - REPL will use basic input (no history)")
    endif()

    target_link_libraries(eshkol-repl PRIVATE eshkol-repl-lib)

    # Export all symbols for JIT symbol resolution
    # This makes arena_*, printf, etc. available to JIT-compiled code
    if(APPLE)
      target_link_options(eshkol-repl PRIVATE "-Wl,-export_dynamic")
    else()
      target_link_options(eshkol-repl PRIVATE "-Wl,--export-dynamic")
    endif()

    # Install REPL executable
    install(TARGETS eshkol-repl DESTINATION bin)

    message(STATUS "REPL build enabled - target: eshkol-repl")
  endif()
endif()

# where to find our CMake modules
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Packing)
