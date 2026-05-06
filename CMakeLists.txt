#
# Copyright (C) tsotchke
#
# SPDX-License-Identifier: MIT
#
cmake_minimum_required(VERSION 3.14)
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()

set(ESHKOL_VERSION "1.2.0")

project("Eshkol" VERSION ${ESHKOL_VERSION} LANGUAGES C CXX)

set(ESHKOL_USING_MSVC FALSE)
if(MSVC)
    set(ESHKOL_USING_MSVC TRUE)
endif()

set(ESHKOL_CLANG_BUILTINS_LIB "")
set(ESHKOL_EXTRA_LINK_LIBS "")
set(ESHKOL_HOST_CXX_COMPILER_EXPLICIT FALSE)
if(DEFINED ESHKOL_HOST_CXX_COMPILER AND NOT "${ESHKOL_HOST_CXX_COMPILER}" STREQUAL "")
    set(ESHKOL_HOST_CXX_COMPILER_EXPLICIT TRUE)
    file(TO_CMAKE_PATH "${ESHKOL_HOST_CXX_COMPILER}" ESHKOL_HOST_CXX_COMPILER)
else()
    set(ESHKOL_HOST_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
endif()
set(ESHKOL_HOST_CXX_COMPILER "${ESHKOL_HOST_CXX_COMPILER}" CACHE FILEPATH
    "C++ compiler driver used by eshkol-run when linking generated programs" FORCE)

if(ESHKOL_USING_MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Visual Studio's ClangCL toolset drives lld-link directly, so compiler-rt
    # builtins do not get auto-added the way they would through the Clang driver.
    #
    # Generated executable links inside eshkol-run/LLVM codegen use GCC-style
    # flags (-L/-l/-o), so prefer the sibling clang++ driver there instead of
    # clang-cl's CL-style frontend.
    get_filename_component(_eshkol_clang_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    if(NOT ESHKOL_HOST_CXX_COMPILER_EXPLICIT)
        set(_eshkol_host_clangxx_candidate "${_eshkol_clang_bin_dir}/clang++.exe")
        if(EXISTS "${_eshkol_host_clangxx_candidate}")
            file(TO_CMAKE_PATH "${_eshkol_host_clangxx_candidate}" ESHKOL_HOST_CXX_COMPILER)
            set(ESHKOL_HOST_CXX_COMPILER "${ESHKOL_HOST_CXX_COMPILER}" CACHE FILEPATH
                "C++ compiler driver used by eshkol-run when linking generated programs" FORCE)
        endif()
    endif()

    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" --print-resource-dir
        OUTPUT_VARIABLE _eshkol_clang_resource_dir_raw
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _eshkol_clang_resource_dir_result
        ERROR_QUIET
    )
    set(_eshkol_clang_builtins_arch "x86_64")
    if(CMAKE_VS_PLATFORM_NAME STREQUAL "ARM64" OR
       CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|AARCH64|aarch64)$")
        set(_eshkol_clang_builtins_arch "aarch64")
    endif()

    if(_eshkol_clang_resource_dir_result EQUAL 0 AND NOT "${_eshkol_clang_resource_dir_raw}" STREQUAL "")
        file(TO_CMAKE_PATH "${_eshkol_clang_resource_dir_raw}" _eshkol_clang_resource_dir)
        set(_eshkol_clang_builtins_candidate
            "${_eshkol_clang_resource_dir}/lib/windows/clang_rt.builtins-${_eshkol_clang_builtins_arch}.lib")
        if(EXISTS "${_eshkol_clang_builtins_candidate}")
            set(ESHKOL_CLANG_BUILTINS_LIB "${_eshkol_clang_builtins_candidate}")
        endif()
    endif()

    if(NOT ESHKOL_CLANG_BUILTINS_LIB)
        get_filename_component(_eshkol_clang_root "${_eshkol_clang_bin_dir}" DIRECTORY)
        file(GLOB _eshkol_clang_builtins LIST_DIRECTORIES FALSE
            "${_eshkol_clang_root}/lib/clang/*/lib/windows/clang_rt.builtins-${_eshkol_clang_builtins_arch}.lib"
        )
        list(LENGTH _eshkol_clang_builtins _eshkol_clang_builtins_count)
        if(_eshkol_clang_builtins_count GREATER 0)
            list(GET _eshkol_clang_builtins 0 ESHKOL_CLANG_BUILTINS_LIB)
        endif()
    endif()

    if(ESHKOL_CLANG_BUILTINS_LIB)
        list(APPEND ESHKOL_EXTRA_LINK_LIBS "${ESHKOL_CLANG_BUILTINS_LIB}")
    else()
        message(WARNING
            "ClangCL compiler-rt builtins were not found; native Windows links may fail "
            "on __divti3/__chkstk_ms-style symbols.")
    endif()
endif()

if(WIN32)
    list(APPEND ESHKOL_EXTRA_LINK_LIBS Advapi32.lib)
endif()

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
option(ESHKOL_ENABLE_ASAN "Enable Address Sanitizer"            OFF)
option(ESHKOL_ENABLE_UBSAN "Enable Undefined Behavior Sanitizer" OFF)
option(ESHKOL_ENABLE_TSAN "Enable Thread Sanitizer"              OFF)
option(ESHKOL_ENABLE_MSAN "Enable Memory Sanitizer"              OFF)
option(ESHKOL_ENABLE_LSAN "Enable Leak Sanitizer (standalone)"   OFF)
option(ESHKOL_ENABLE_FUZZ "Build libFuzzer harnesses (clang only)" OFF)

if(ESHKOL_BUILD_TESTS)
    enable_testing()
endif()

# XLA Backend Support (optional, default OFF)
# When enabled, provides accelerated tensor operations via XLA/StableHLO
# Requires MLIR and StableHLO libraries
option(ESHKOL_XLA_ENABLED "Enable XLA backend for accelerated tensor operations" OFF)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/LLVMToolchain.cmake)
set(ESHKOL_LLVM_USE_CMAKE_PACKAGE FALSE)
set(ESHKOL_LLVM_ROOT "")

# StableHLO Root (for XLA builds with bundled LLVM/MLIR/StableHLO)
# Set this to the path where StableHLO was built (e.g., /tmp/stablehlo)
# When set, uses the bundled LLVM/MLIR instead of the lite/native system LLVM toolchain
set(STABLEHLO_ROOT "" CACHE PATH "Path to StableHLO installation with bundled LLVM/MLIR")

# ===== Sanitizers =====
# Mutual-exclusivity check: ASan / TSan / MSan each rewrite the runtime
# in incompatible ways. A build that enables more than one fails at link
# time with a cryptic message; fail loudly up front instead. UBSan is
# compatible with ASan + TSan but not MSan.
set(ESHKOL_PRIMARY_SANITIZER_COUNT 0)
foreach(san ASAN TSAN MSAN)
    if(ESHKOL_ENABLE_${san})
        math(EXPR ESHKOL_PRIMARY_SANITIZER_COUNT "${ESHKOL_PRIMARY_SANITIZER_COUNT} + 1")
    endif()
endforeach()
if(ESHKOL_PRIMARY_SANITIZER_COUNT GREATER 1)
    message(FATAL_ERROR
        "At most one of ESHKOL_ENABLE_ASAN / _TSAN / _MSAN may be set. "
        "They use incompatible runtimes. UBSan may combine with ASan or TSan.")
endif()

if(ESHKOL_ENABLE_ASAN)
    message(STATUS "Address Sanitizer: ENABLED")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address)
    # ASan needs the runtime loaded before LLVM static init; disable the
    # deprecated -bind_at_load trick elsewhere in this file for ASan
    # builds by signaling via a compile definition.
    add_compile_definitions(ESHKOL_SANITIZER_BUILD=1)
endif()

if(ESHKOL_ENABLE_UBSAN)
    message(STATUS "Undefined Behavior Sanitizer: ENABLED")
    # -fno-sanitize-recover makes the first UB a hard abort so CI fails
    # fast instead of logging the violation and continuing with a
    # corrupt state. Disable alignment checks on the LLVM source that
    # deliberately misaligns (see lib/backend) via per-function
    # attributes if needed.
    add_compile_options(-fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=undefined)
    add_compile_definitions(ESHKOL_SANITIZER_BUILD=1)
endif()

if(ESHKOL_ENABLE_TSAN)
    message(STATUS "Thread Sanitizer: ENABLED")
    if(APPLE)
        message(STATUS "  Note: TSan on Apple silicon requires macOS 12+ clang")
    endif()
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=thread)
    add_compile_definitions(ESHKOL_SANITIZER_BUILD=1)
endif()

if(ESHKOL_ENABLE_MSAN)
    message(STATUS "Memory Sanitizer: ENABLED (requires instrumented libstdc++/libc++)")
    add_compile_options(-fsanitize=memory -fsanitize-memory-track-origins=2
                        -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=memory)
    add_compile_definitions(ESHKOL_SANITIZER_BUILD=1)
endif()

if(ESHKOL_ENABLE_LSAN)
    # LSan standalone — only valid when NOT combined with ASan (ASan
    # enables LSan automatically). Useful on Linux for leak-only CI.
    if(ESHKOL_ENABLE_ASAN)
        message(STATUS "Leak Sanitizer: subsumed by ASan")
    else()
        message(STATUS "Leak Sanitizer: ENABLED (standalone)")
        add_compile_options(-fsanitize=leak -fno-omit-frame-pointer -g)
        add_link_options(-fsanitize=leak)
        add_compile_definitions(ESHKOL_SANITIZER_BUILD=1)
    endif()
endif()

# LLVM backend is always enabled
set(ESHKOL_BUILD_BACKEND ON)

# ===== LLVM Configuration =====
# Two modes:
# 1. STABLEHLO_ROOT set: Use StableHLO's bundled LLVM (XLA build)
# 2. STABLEHLO_ROOT not set: Use system llvm-config (Lite/native build)

if(STABLEHLO_ROOT AND EXISTS "${STABLEHLO_ROOT}/llvm-build")
    # ===== XLA BUILD: Use StableHLO's bundled LLVM =====
    message(STATUS "=== XLA BUILD MODE: Using StableHLO's bundled LLVM ===")
    set(USING_STABLEHLO_LLVM TRUE)

    set(STABLEHLO_LLVM_BUILD "${STABLEHLO_ROOT}/llvm-build")
    set(STABLEHLO_LLVM_LIB "${STABLEHLO_LLVM_BUILD}/lib")
    set(ESHKOL_LLVM_ROOT "${STABLEHLO_LLVM_BUILD}")

    # StableHLO build may not include llvm-config, so use direct paths
    # Include directories for bundled LLVM/MLIR
    set(LLVM_INCLUDE_DIRS
        "${STABLEHLO_LLVM_BUILD}/include"
        "${STABLEHLO_ROOT}/llvm-project/llvm/include"
        "${STABLEHLO_ROOT}/llvm-project/mlir/include"
        "${STABLEHLO_LLVM_BUILD}/tools/mlir/include"
    )

    # LLVM compile flags for bundled LLVM (manual since no llvm-config)
    # Don't include -std= since CMAKE_CXX_STANDARD handles it
    # IMPORTANT: StableHLO's LLVM is built with RTTI disabled, so we must also disable RTTI
    # Use separate_arguments to convert to list
    set(LLVM_CXXFLAGS_RAW "-D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -fno-rtti")
    separate_arguments(LLVM_CXXFLAGS UNIX_COMMAND "${LLVM_CXXFLAGS_RAW}")

    # LLVM link flags
    set(LLVM_LDFLAGS "-L${STABLEHLO_LLVM_LIB}")

    # Find LLVM libraries in the build directory - link ALL static libraries directly
    # Use full paths to static libraries for proper symbol resolution on both ARM and x86
    file(GLOB LLVM_STATIC_LIBS "${STABLEHLO_LLVM_LIB}/libLLVM*.a")
    list(LENGTH LLVM_STATIC_LIBS LLVM_LIB_COUNT)
    message(STATUS "Found ${LLVM_LIB_COUNT} LLVM libraries")

    # Handle circular dependencies in LLVM static libraries
    # On macOS: force-load LLVMSupport to ensure typeinfo symbols are exported
    # On Linux: use linker groups
    if(APPLE)
        # Find LLVMSupport specifically for force loading
        find_library(LLVM_SUPPORT_LIB LLVMSupport HINTS ${STABLEHLO_LLVM_LIB} NO_DEFAULT_PATH)
        if(LLVM_SUPPORT_LIB)
            set(LLVM_LIBS "-Wl,-force_load,${LLVM_SUPPORT_LIB}" ${LLVM_STATIC_LIBS})
        else()
            set(LLVM_LIBS ${LLVM_STATIC_LIBS})
        endif()
    else()
        # On Linux, wrap with linker group to handle circular deps
        set(LLVM_LIBS "-Wl,--start-group" ${LLVM_STATIC_LIBS} "-Wl,--end-group")
    endif()

    # System libraries
    # Note: StableHLO's LLVM is built with zstd support
    # Find zstd - search standard paths (no hardcoded Homebrew paths)
    find_library(ZSTD_LIB zstd)
    if(ZSTD_LIB)
        message(STATUS "Found zstd: ${ZSTD_LIB}")
        set(ZSTD_LINK "${ZSTD_LIB}")
    else()
        message(STATUS "zstd not found, trying -lzstd")
        set(ZSTD_LINK "-lzstd")
    endif()

    if(APPLE)
        set(LLVM_SYSTEM_LIBS "-lm -lz -lcurses -lxml2")
    else()
        set(LLVM_SYSTEM_LIBS "-lrt -ldl -lpthread -lm -lz")
    endif()
    # Append zstd separately (full path works better)
    set(LLVM_SYSTEM_LIBS "${LLVM_SYSTEM_LIBS} ${ZSTD_LINK}")

    set(LLVM_CONFIG_EXECUTABLE "STABLEHLO_BUNDLED")

    message(STATUS "StableHLO LLVM Build: ${STABLEHLO_LLVM_BUILD}")
    message(STATUS "StableHLO LLVM Include: ${STABLEHLO_LLVM_BUILD}/include")
else()
    # ===== LITE BUILD: Use system LLVM selected by ESHKOL_REQUIRED_LLVM_MAJOR =====
    message(STATUS "=== LITE BUILD MODE: Using system LLVM ===")
    set(USING_STABLEHLO_LLVM FALSE)

    eshkol_find_lite_llvm()

    # Generated build config for toolchain/runtime helpers (Windows port)
    set(ESHKOL_HOST_RUNTIME_LINK_ARGS "")
    set(ESHKOL_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${ESHKOL_GENERATED_INCLUDE_DIR}/eshkol")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_config.h.in"
      "${ESHKOL_GENERATED_INCLUDE_DIR}/eshkol/build_config.h"
      @ONLY
    )

    if(WIN32)
        set(ESHKOL_LLVM_USE_CMAKE_PACKAGE TRUE)
        find_package(LibXml2 QUIET)
        set(ESHKOL_WINDOWS_LLVM_HAS_LIBXML2 FALSE)
        if(TARGET LibXml2::LibXml2)
            set(ESHKOL_WINDOWS_LLVM_HAS_LIBXML2 TRUE)
        else()
            # The official Windows LLVM SDK can reference LibXml2::LibXml2 from
            # LLVMWindowsManifest even when libxml2 is not installed locally.
            # Provide a placeholder target so the package loads, then omit that
            # optional LLVM library from our link set below.
            add_library(LibXml2::LibXml2 INTERFACE IMPORTED)
        endif()
        find_package(LLVM CONFIG REQUIRED)

        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING
            "Match the official LLVM Windows SDK runtime linkage" FORCE)

        eshkol_validate_llvm_major("${LLVM_PACKAGE_VERSION}" "${LLVM_DIR}")

        if(TARGET LLVMDebugInfoPDB)
            set(_eshkol_diaguids_lib_amd64 "")
            set(_eshkol_diaguids_lib_arm64 "")
            foreach(_eshkol_dia_arch IN ITEMS amd64 arm64)
                file(GLOB _eshkol_dia_sdk_libs LIST_DIRECTORIES FALSE
                    "C:/Program Files/Microsoft Visual Studio/*/*/DIA SDK/lib/${_eshkol_dia_arch}/diaguids.lib"
                    "C:/Program Files (x86)/Microsoft Visual Studio/*/*/DIA SDK/lib/${_eshkol_dia_arch}/diaguids.lib"
                )
                list(LENGTH _eshkol_dia_sdk_libs _eshkol_dia_sdk_count)
                if(_eshkol_dia_sdk_count GREATER 0)
                    list(GET _eshkol_dia_sdk_libs 0 _eshkol_diaguids_lib)
                    if(_eshkol_dia_arch STREQUAL "amd64")
                        set(_eshkol_diaguids_lib_amd64 "${_eshkol_diaguids_lib}")
                    else()
                        set(_eshkol_diaguids_lib_arm64 "${_eshkol_diaguids_lib}")
                    endif()
                endif()
            endforeach()

            get_target_property(_eshkol_pdb_link_libs LLVMDebugInfoPDB INTERFACE_LINK_LIBRARIES)
            if(_eshkol_pdb_link_libs)
                set(_eshkol_pdb_link_libs_patched "")
                foreach(_llvm_dep IN LISTS _eshkol_pdb_link_libs)
                    if(_llvm_dep MATCHES "DIA SDK/lib/amd64/diaguids\\.lib$" AND _eshkol_diaguids_lib_amd64)
                        list(APPEND _eshkol_pdb_link_libs_patched "${_eshkol_diaguids_lib_amd64}")
                    elseif(_llvm_dep MATCHES "DIA SDK/lib/arm64/diaguids\\.lib$" AND _eshkol_diaguids_lib_arm64)
                        list(APPEND _eshkol_pdb_link_libs_patched "${_eshkol_diaguids_lib_arm64}")
                    else()
                        list(APPEND _eshkol_pdb_link_libs_patched "${_llvm_dep}")
                    endif()
                endforeach()
                set_target_properties(LLVMDebugInfoPDB PROPERTIES
                    INTERFACE_LINK_LIBRARIES "${_eshkol_pdb_link_libs_patched}")
            endif()
        endif()

        set(ESHKOL_LLVM_VERSION "${LLVM_PACKAGE_VERSION}")
        set(ESHKOL_LLVM_ROOT "${LLVM_INSTALL_PREFIX}")
        set(LLVM_CXXFLAGS "${LLVM_DEFINITIONS}")
        set(LLVM_LDFLAGS "")
        set(LLVM_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS})
        set(LLVM_LIBRARY_DIRS ${LLVM_LIBRARY_DIRS})
        # The installed Windows package exposes imported targets for every
        # library via LLVM_AVAILABLE_LIBS, but the "all" pseudo-component does
        # not expand reliably outside LLVM's own build tree.
        set(LLVM_LIBS_LIST ${LLVM_AVAILABLE_LIBS})
        if(NOT ESHKOL_WINDOWS_LLVM_HAS_LIBXML2)
            list(REMOVE_ITEM LLVM_LIBS_LIST LLVMWindowsManifest)
        endif()
        set(LLVM_SYSTEM_LIBS_LIST "")

        # The official Windows SDK ships release libraries only. Map the other
        # multi-config variants to Release so native VS generators stay usable.
        set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG Release)
        set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL Release)
        set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
    else()
        # Get LLVM configuration
        execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --cxxflags OUTPUT_VARIABLE LLVM_CXXFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --ldflags OUTPUT_VARIABLE LLVM_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --libs all OUTPUT_VARIABLE LLVM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --system-libs OUTPUT_VARIABLE LLVM_SYSTEM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --includedir OUTPUT_VARIABLE LLVM_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

        # Enhanced LLVM include directory detection (backwards-compatible)
        # Extract include directories from CXXFLAGS for better cross-platform support
        string(REGEX MATCHALL "-I[^ ]+" LLVM_INCLUDE_FLAGS "${LLVM_CXXFLAGS}")
        set(LLVM_INCLUDE_DIRS "")

        foreach(include_flag ${LLVM_INCLUDE_FLAGS})
            string(REGEX REPLACE "^-I" "" include_dir "${include_flag}")
            list(APPEND LLVM_INCLUDE_DIRS "${include_dir}")
        endforeach()

        # Prefer llvm-config's canonical include directory when available.
        if(LLVM_INCLUDE_DIR AND EXISTS "${LLVM_INCLUDE_DIR}")
            list(APPEND LLVM_INCLUDE_DIRS "${LLVM_INCLUDE_DIR}")
        endif()

        get_filename_component(LLVM_CONFIG_DIR "${LLVM_CONFIG_EXECUTABLE}" DIRECTORY)
        get_filename_component(ESHKOL_LLVM_ROOT "${LLVM_CONFIG_DIR}" DIRECTORY)

        # Additional fallback for common LLVM installation patterns
        if(NOT LLVM_INCLUDE_DIRS)
            set(POTENTIAL_INCLUDE_DIRS
                "${ESHKOL_LLVM_ROOT}/include"
                "${ESHKOL_LLVM_ROOT}/include/llvm"
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
    endif()
endif()

# Convert llvm-config output to CMake lists.
if(ESHKOL_LLVM_USE_CMAKE_PACKAGE)
    separate_arguments(LLVM_CXXFLAGS_LIST WINDOWS_COMMAND "${LLVM_CXXFLAGS}")
    set(LLVM_LDFLAGS_LIST "")
elseif(WIN32)
    separate_arguments(LLVM_CXXFLAGS_LIST WINDOWS_COMMAND "${LLVM_CXXFLAGS}")
    separate_arguments(LLVM_LDFLAGS_LIST WINDOWS_COMMAND "${LLVM_LDFLAGS}")
    separate_arguments(LLVM_LIBS_LIST WINDOWS_COMMAND "${LLVM_LIBS}")
    separate_arguments(LLVM_SYSTEM_LIBS_LIST WINDOWS_COMMAND "${LLVM_SYSTEM_LIBS}")

    set(_normalized_llvm_libs "")
    foreach(llvm_item IN LISTS LLVM_LIBS_LIST)
        file(TO_CMAKE_PATH "${llvm_item}" llvm_item_normalized)
        list(APPEND _normalized_llvm_libs "${llvm_item_normalized}")
    endforeach()
    set(LLVM_LIBS_LIST "${_normalized_llvm_libs}")

    set(_normalized_llvm_system_libs "")
    foreach(llvm_item IN LISTS LLVM_SYSTEM_LIBS_LIST)
        file(TO_CMAKE_PATH "${llvm_item}" llvm_item_normalized)
        list(APPEND _normalized_llvm_system_libs "${llvm_item_normalized}")
    endforeach()
    set(LLVM_SYSTEM_LIBS_LIST "${_normalized_llvm_system_libs}")
else()
    separate_arguments(LLVM_CXXFLAGS_LIST UNIX_COMMAND "${LLVM_CXXFLAGS}")
    separate_arguments(LLVM_LDFLAGS_LIST UNIX_COMMAND "${LLVM_LDFLAGS}")
    separate_arguments(LLVM_LIBS_LIST UNIX_COMMAND "${LLVM_LIBS}")
    separate_arguments(LLVM_SYSTEM_LIBS_LIST UNIX_COMMAND "${LLVM_SYSTEM_LIBS}")
endif()

# Keep LLVM's defines and unwind flags, but let the project control language mode
# and exception support.
if(ESHKOL_USING_MSVC)
    list(FILTER LLVM_CXXFLAGS_LIST EXCLUDE REGEX "^-I")
    list(FILTER LLVM_CXXFLAGS_LIST EXCLUDE REGEX "^[-/]std[:=]")
    list(FILTER LLVM_CXXFLAGS_LIST EXCLUDE REGEX "^/EH")
    list(FILTER LLVM_CXXFLAGS_LIST EXCLUDE REGEX "^-fno-exceptions$")
else()
    list(FILTER LLVM_CXXFLAGS_LIST EXCLUDE REGEX "^-std=")
    list(FILTER LLVM_CXXFLAGS_LIST EXCLUDE REGEX "^-fno-exceptions$")
endif()

# Debug output for troubleshooting
if(ESHKOL_LLVM_USE_CMAKE_PACKAGE)
    message(STATUS "LLVM Package: ${LLVM_DIR}")
else()
    message(STATUS "LLVM Config: ${LLVM_CONFIG_EXECUTABLE}")
endif()
if(ESHKOL_LLVM_VERSION)
    message(STATUS "LLVM Version: ${ESHKOL_LLVM_VERSION}")
endif()
message(STATUS "LLVM Include Dirs: ${LLVM_INCLUDE_DIRS}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "LLVM CXXFLAGS: ${LLVM_CXXFLAGS}")
endif()

# ===== XLA Backend Configuration (Optional) =====
if(ESHKOL_XLA_ENABLED)
    message(STATUS "XLA backend: ENABLED")

    # Define XLA enabled for code
    add_compile_definitions(ESHKOL_XLA_ENABLED)

    # ===== MLIR/StableHLO Detection =====
    # Two modes:
    # 1. STABLEHLO_ROOT set: Use bundled LLVM/MLIR/StableHLO (XLA build)
    # 2. STABLEHLO_ROOT not set: Try the configured system LLVM (lite build fallback)

    if(STABLEHLO_ROOT AND EXISTS "${STABLEHLO_ROOT}")
        message(STATUS "Using StableHLO installation at: ${STABLEHLO_ROOT}")

        # StableHLO build structure (two modes):
        # Mode 1: Bundled LLVM — ${STABLEHLO_ROOT}/llvm-build/ exists
        # Mode 2: System LLVM — StableHLO built against system LLVM (no llvm-build/)
        if(EXISTS "${STABLEHLO_ROOT}/llvm-build")
            set(STABLEHLO_LLVM_BUILD "${STABLEHLO_ROOT}/llvm-build")
            message(STATUS "StableHLO: using bundled LLVM at ${STABLEHLO_LLVM_BUILD}")
        else()
            # StableHLO built against system LLVM — reuse the resolved lite toolchain root.
            set(STABLEHLO_LLVM_BUILD "${ESHKOL_LLVM_ROOT}")
            message(STATUS "StableHLO: using system LLVM at ${STABLEHLO_LLVM_BUILD}")
        endif()
        set(STABLEHLO_BUILD "${STABLEHLO_ROOT}/build")

        # Set CMake paths for find_package
        set(MLIR_DIR "${STABLEHLO_LLVM_BUILD}/lib/cmake/mlir" CACHE PATH "MLIR CMake dir" FORCE)
        set(LLVM_DIR "${STABLEHLO_LLVM_BUILD}/lib/cmake/llvm" CACHE PATH "LLVM CMake dir" FORCE)

        # Include directories for StableHLO and MLIR
        # Note: STABLEHLO_ROOT is the base so "stablehlo/dialect/..." resolves correctly
        set(STABLEHLO_INCLUDE_DIRS
            "${STABLEHLO_ROOT}"
            "${STABLEHLO_ROOT}/llvm-project/mlir/include"
            "${STABLEHLO_LLVM_BUILD}/include"
            "${STABLEHLO_LLVM_BUILD}/tools/mlir/include"
            "${STABLEHLO_BUILD}"
        )

        # Add include directories
        foreach(inc_dir ${STABLEHLO_INCLUDE_DIRS})
            if(EXISTS "${inc_dir}")
                include_directories("${inc_dir}")
                message(STATUS "  Include: ${inc_dir}")
            endif()
        endforeach()

        # Find MLIR/LLVM libraries from StableHLO build
        set(MLIR_LIB_DIR "${STABLEHLO_LLVM_BUILD}/lib")
        set(STABLEHLO_LIB_DIR "${STABLEHLO_BUILD}/lib")

        # Link ALL MLIR libraries - the dependency chain is too complex to manage manually
        file(GLOB MLIR_STATIC_LIBS "${MLIR_LIB_DIR}/libMLIR*.a")
        set(MLIR_LIBRARIES ${MLIR_STATIC_LIBS})
        message(STATUS "Found ${CMAKE_MATCH_COUNT} MLIR libraries")

        # StableHLO libraries (dialect + lowering passes)
        set(STABLEHLO_LIBS
            StablehloOps
            StablehloBase
            StablehloBroadcastUtils
            StablehloBroadcastLowering
            StablehloAssemblyFormat
            StablehloTypeInference
            StablehloLinalgTransforms
            StablehloPasses
            StablehloTypeConversion
            ChloOps
        )

        set(STABLEHLO_LIBRARIES "")
        foreach(lib ${STABLEHLO_LIBS})
            find_library(${lib}_LIB ${lib} HINTS ${STABLEHLO_LIB_DIR} NO_DEFAULT_PATH)
            if(${lib}_LIB)
                list(APPEND STABLEHLO_LIBRARIES ${${lib}_LIB})
            endif()
        endforeach()

        # Enable full XLA if we have both MLIR and StableHLO
        if(MLIR_LIBRARIES AND STABLEHLO_LIBRARIES)
            set(MLIR_AVAILABLE ON)
            set(STABLEHLO_AVAILABLE ON)
            add_compile_definitions(ESHKOL_MLIR_AVAILABLE)
            add_compile_definitions(ESHKOL_STABLEHLO_AVAILABLE)
            add_compile_definitions(ESHKOL_XLA_FULL_MLIR)
            message(STATUS "MLIR: AVAILABLE (bundled)")
            message(STATUS "StableHLO: AVAILABLE (bundled)")
            message(STATUS "XLA: FULL MLIR SUPPORT ENABLED")
        elseif(MLIR_LIBRARIES)
            set(MLIR_AVAILABLE ON)
            set(STABLEHLO_AVAILABLE OFF)
            add_compile_definitions(ESHKOL_MLIR_AVAILABLE)
            message(STATUS "MLIR: AVAILABLE (bundled)")
            message(STATUS "StableHLO: libraries not found in ${STABLEHLO_LIB_DIR}")
        else()
            set(MLIR_AVAILABLE OFF)
            set(STABLEHLO_AVAILABLE OFF)
            message(WARNING "STABLEHLO_ROOT set but MLIR libraries not found in ${MLIR_LIB_DIR}")
        endif()

        # Link directories for StableHLO build
        link_directories("${MLIR_LIB_DIR}")
        link_directories("${STABLEHLO_LIB_DIR}")

    else()
        # Fallback: Try system LLVM (Lite build mode)
        set(LLVM_ROOT_DIR "${ESHKOL_LLVM_ROOT}")

        # Check for MLIR headers in system LLVM
        if(EXISTS "${LLVM_ROOT_DIR}/include/mlir/IR/MLIRContext.h")
            set(MLIR_AVAILABLE ON)
            set(MLIR_INCLUDE_DIR "${LLVM_ROOT_DIR}/include")
            add_compile_definitions(ESHKOL_MLIR_AVAILABLE)
            message(STATUS "MLIR: AVAILABLE at ${MLIR_INCLUDE_DIR}")

            # Find MLIR libraries in LLVM lib directory
            # Full set includes core + StableHLO→LLVM lowering pipeline passes
            set(MLIR_LIB_DIR "${LLVM_ROOT_DIR}/lib")
            set(MLIR_LIBS
                # Core MLIR infrastructure
                MLIRIR
                MLIRSupport
                MLIRParser
                MLIRTransforms
                # Dialects used by StableHLO pipeline
                MLIRFuncDialect
                MLIRArithDialect
                MLIRLinalgDialect
                MLIRSCFDialect
                MLIRMemRefDialect
                MLIRLLVMDialect
                MLIRControlFlowDialect
                # Lowering passes: StableHLO → Linalg → SCF → CF → LLVM
                MLIRLinalgTransforms
                MLIRSCFToControlFlow
                MLIRFuncToLLVM
                MLIRArithToLLVM
                MLIRMemRefToLLVM
                MLIRReconcileUnrealizedCasts
                # LLVM IR translation (MLIR LLVM dialect → actual LLVM IR)
                MLIRLLVMToLLVMIRTranslation
                MLIRTargetLLVMIRExport
                MLIRBuiltinToLLVMIRTranslation
            )
            set(MLIR_LIBRARIES "")
            foreach(lib ${MLIR_LIBS})
                find_library(${lib}_LIB ${lib} HINTS ${MLIR_LIB_DIR} NO_DEFAULT_PATH)
                if(${lib}_LIB)
                    list(APPEND MLIR_LIBRARIES ${${lib}_LIB})
                endif()
            endforeach()

            if(MLIR_LIBRARIES)
                message(STATUS "MLIR libraries: FOUND")
            else()
                message(STATUS "MLIR libraries: NOT FOUND (headers only)")
                set(MLIR_AVAILABLE OFF)
                remove_definitions(-DESHKOL_MLIR_AVAILABLE)
            endif()

            # Check for StableHLO (rare in system LLVM)
            if(EXISTS "${LLVM_ROOT_DIR}/include/stablehlo/dialect/StablehloOps.h")
                set(STABLEHLO_AVAILABLE ON)
                add_compile_definitions(ESHKOL_STABLEHLO_AVAILABLE)
                message(STATUS "StableHLO: AVAILABLE")
            else()
                set(STABLEHLO_AVAILABLE OFF)
                message(STATUS "StableHLO: NOT FOUND (use -DSTABLEHLO_ROOT=path for full XLA)")
            endif()
        else()
            set(MLIR_AVAILABLE OFF)
            set(STABLEHLO_AVAILABLE OFF)
            message(STATUS "MLIR: NOT FOUND in system LLVM (XLA will use stubs)")
        endif()
    endif()
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
        message(WARNING "BLAS acceleration: DISABLED (no BLAS library found). "
            "On Ubuntu/Debian: sudo apt install libopenblas-dev. "
            "On Fedora/RHEL: sudo dnf install openblas-devel. "
            "On Arch: sudo pacman -S openblas.")
        set(ESHKOL_BLAS_ENABLED OFF)
    endif()
else()
    message(STATUS "BLAS acceleration: DISABLED (enable with -DESHKOL_BLAS_ENABLED=ON)")
endif()

# ===== GPU Acceleration (Platform-specific) =====
# Provides GPU-accelerated tensor operations via Metal (macOS) or CUDA (Linux/Windows)
option(ESHKOL_GPU_ENABLED "Enable GPU acceleration for tensor operations" ON)

if(ESHKOL_GPU_ENABLED)
    set(GPU_FOUND FALSE)
    set(GPU_LIBRARIES "")
    set(GPU_SOURCES "")

    if(APPLE)
        # Use Apple Metal framework (macOS/iOS)
        find_library(METAL_FRAMEWORK Metal)
        find_library(MPS_FRAMEWORK MetalPerformanceShaders)
        find_library(FOUNDATION_FRAMEWORK Foundation)

        if(METAL_FRAMEWORK AND MPS_FRAMEWORK)
            set(GPU_FOUND TRUE)
            set(GPU_LIBRARIES ${METAL_FRAMEWORK} ${MPS_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
            add_compile_definitions(ESHKOL_GPU_METAL_ENABLED)
            message(STATUS "GPU: Apple Metal (unified memory)")

            # Generate embedded Metal shader source from canonical header
            # metal_softfloat.h is the single source of truth for sf64 emulation
            set(METAL_SF64_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/lib/backend/gpu/metal_softfloat.h")
            set(METAL_SF64_EMBEDDED "${CMAKE_BINARY_DIR}/metal_sf64_embedded.inc")
            add_custom_command(
                OUTPUT ${METAL_SF64_EMBEDDED}
                COMMAND ${CMAKE_COMMAND}
                    -DINPUT_FILE=${METAL_SF64_HEADER}
                    -DOUTPUT_FILE=${METAL_SF64_EMBEDDED}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_metal_shader.cmake
                DEPENDS ${METAL_SF64_HEADER}
                        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_metal_shader.cmake
                COMMENT "Embedding Metal SoftFloat shader from metal_softfloat.h"
            )
            add_custom_target(metal_sf64_shader DEPENDS ${METAL_SF64_EMBEDDED})

            # Add GPU source file (Objective-C++)
            set(GPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/lib/backend/gpu/gpu_memory.mm")

            # Enable Objective-C++ with ARC for the GPU source
            set_source_files_properties(${GPU_SOURCES} PROPERTIES
                COMPILE_FLAGS "-x objective-c++ -fobjc-arc"
            )
        endif()
    else()
        # Try CUDA on Linux/Windows
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(GPU_FOUND TRUE)
            set(GPU_LIBRARIES CUDA::cudart CUDA::cublas)
            if(TARGET CUDA::cublasLt)
                list(APPEND GPU_LIBRARIES CUDA::cublasLt)
            elseif(CUDA_cublasLt_LIBRARY)
                list(APPEND GPU_LIBRARIES "${CUDA_cublasLt_LIBRARY}")
            endif()

            if(WIN32)
                set(ESHKOL_CUDA_DEVICE_RUNTIME "")
                find_library(
                    ESHKOL_CUDA_DEVICE_RUNTIME
                    NAMES cudadevrt
                    HINTS
                        "${CUDAToolkit_ROOT}/lib/x64"
                        "${CUDAToolkit_BIN_DIR}/../lib/x64"
                    NO_DEFAULT_PATH
                )
            endif()

            add_compile_definitions(ESHKOL_GPU_CUDA_ENABLED)
            enable_language(CUDA)
            message(STATUS "GPU: NVIDIA CUDA")

            # CUDA backend — .cpp host code + .cu kernels
            set(GPU_SOURCES
                "${CMAKE_CURRENT_SOURCE_DIR}/lib/backend/gpu/gpu_memory_cuda.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/lib/backend/gpu/gpu_cuda_kernels.cu")
        else()
            message(STATUS "GPU: CUDA not found (install NVIDIA CUDA Toolkit for GPU support)")
        endif()
    endif()

    if(GPU_FOUND)
        add_compile_definitions(ESHKOL_GPU_ENABLED)
        message(STATUS "GPU acceleration: ENABLED")
    else()
        message(STATUS "GPU acceleration: DISABLED (no GPU framework found)")
        # Still compile the stub so GPU API symbols resolve at link time
        set(GPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/lib/backend/gpu/gpu_memory_stub.cpp")
        set(ESHKOL_GPU_ENABLED OFF)
    endif()
else()
    message(STATUS "GPU acceleration: DISABLED (enable with -DESHKOL_GPU_ENABLED=ON)")
    # Keep the GPU API linked even when the feature is explicitly disabled.
    # XLA and tensor runtime code call into this surface and rely on the
    # CPU-backed stub implementation when no real GPU backend is enabled.
    set(GPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/lib/backend/gpu/gpu_memory_stub.cpp")
endif()

macro(eshkol_append_host_runtime_link_arg item)
    if(NOT "${item}" STREQUAL "")
        set(_eshkol_runtime_link_arg "")

        if(TARGET "${item}")
            if(WIN32)
                foreach(_prop
                    IMPORTED_IMPLIB_RELEASE
                    IMPORTED_IMPLIB_RELWITHDEBINFO
                    IMPORTED_IMPLIB_MINSIZEREL
                    IMPORTED_IMPLIB_DEBUG
                    IMPORTED_IMPLIB
                    IMPORTED_LOCATION_RELEASE
                    IMPORTED_LOCATION_RELWITHDEBINFO
                    IMPORTED_LOCATION_MINSIZEREL
                    IMPORTED_LOCATION_DEBUG
                    IMPORTED_LOCATION
                )
                    if("${_eshkol_runtime_link_arg}" STREQUAL "")
                        get_target_property(_candidate "${item}" ${_prop})
                        if(_candidate AND NOT _candidate MATCHES "-NOTFOUND$")
                            set(_eshkol_runtime_link_arg "${_candidate}")
                        endif()
                    endif()
                endforeach()
            else()
                foreach(_prop
                    IMPORTED_LOCATION_RELEASE
                    IMPORTED_LOCATION_RELWITHDEBINFO
                    IMPORTED_LOCATION_MINSIZEREL
                    IMPORTED_LOCATION_DEBUG
                    IMPORTED_LOCATION
                )
                    if("${_eshkol_runtime_link_arg}" STREQUAL "")
                        get_target_property(_candidate "${item}" ${_prop})
                        if(_candidate AND NOT _candidate MATCHES "-NOTFOUND$")
                            set(_eshkol_runtime_link_arg "${_candidate}")
                        endif()
                    endif()
                endforeach()
            endif()
        else()
            set(_eshkol_runtime_link_arg "${item}")
            if(NOT _eshkol_runtime_link_arg MATCHES "^-" AND
               NOT _eshkol_runtime_link_arg MATCHES "[/\\\\]" AND
               NOT _eshkol_runtime_link_arg MATCHES "\\.(a|so|dylib|lib)$")
                set(_eshkol_runtime_link_arg "-l${_eshkol_runtime_link_arg}")
            endif()
        endif()

        if(NOT "${_eshkol_runtime_link_arg}" STREQUAL "")
            file(TO_CMAKE_PATH "${_eshkol_runtime_link_arg}" _eshkol_runtime_link_arg_norm)
            list(APPEND ESHKOL_HOST_RUNTIME_LINK_ARGS "${_eshkol_runtime_link_arg_norm}")
        endif()
    endif()
endmacro()

if(NOT APPLE)
    foreach(_extra_runtime_lib IN LISTS ESHKOL_EXTRA_LINK_LIBS)
        eshkol_append_host_runtime_link_arg("${_extra_runtime_lib}")
    endforeach()

    if(BLAS_FOUND AND BLAS_LIBRARIES)
        foreach(_blas_lib IN LISTS BLAS_LIBRARIES)
            eshkol_append_host_runtime_link_arg("${_blas_lib}")
        endforeach()
    endif()

    if(GPU_FOUND AND GPU_LIBRARIES)
        foreach(_gpu_lib IN LISTS GPU_LIBRARIES)
            eshkol_append_host_runtime_link_arg("${_gpu_lib}")
        endforeach()
    endif()

    if(WIN32 AND ESHKOL_CUDA_DEVICE_RUNTIME)
        eshkol_append_host_runtime_link_arg("${ESHKOL_CUDA_DEVICE_RUNTIME}")
    endif()

    # ── Sanitizer link-flag propagation ──────────────────────────────
    # When libeshkol-static is built with -fsanitize=address /
    # -fsanitize=undefined / -fsanitize=thread / -fsanitize=memory,
    # its objects emit unresolved references to the sanitizer runtime
    # (__asan_*, __ubsan_*, __tsan_*, __msan_*). User programs that
    # link against libeshkol-static must therefore also pull in the
    # sanitizer runtimes — otherwise `ld` fails with `undefined
    # reference to __asan_report_storeN` etc.
    #
    # eshkol-run drives the user link step, and reads
    # ESHKOL_HOST_RUNTIME_LINK_ARGS from build_config.h to compose
    # that command line. Append the sanitizer flags here so the user
    # link automatically picks them up; nothing is added on a
    # non-sanitized build.
    if(ESHKOL_ENABLE_ASAN)
        eshkol_append_host_runtime_link_arg("-fsanitize=address")
    endif()
    if(ESHKOL_ENABLE_UBSAN)
        eshkol_append_host_runtime_link_arg("-fsanitize=undefined")
    endif()
    if(ESHKOL_ENABLE_TSAN)
        eshkol_append_host_runtime_link_arg("-fsanitize=thread")
    endif()
    if(ESHKOL_ENABLE_MSAN)
        eshkol_append_host_runtime_link_arg("-fsanitize=memory")
    endif()
endif()

if(DEFINED ESHKOL_GENERATED_INCLUDE_DIR)
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_config.h.in"
      "${ESHKOL_GENERATED_INCLUDE_DIR}/eshkol/build_config.h"
      @ONLY
    )
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

# Exclude GPU backend sources — they are added explicitly via GPU_SOURCES
# (Metal .mm on macOS, stub .cpp on Linux/no-GPU)
list(FILTER LIB_SRC EXCLUDE REGEX "lib/backend/gpu/.*")

# Exclude agent FFI sources — they are in eshkol-agent-ffi static library
# with their own dependency management (sqlite3, pcre2, etc.)
list(FILTER LIB_SRC EXCLUDE REGEX "lib/agent/.*")

# Exclude ALL .c files from lib/backend/ — they are either standalone programs
# (eshkol_compiler.c, weight_matrices.c, eshkol_asm.c, etc.) or unity-build
# includes (vm_*.c, eskb_*.c are #included by eshkol_vm.c).
# Only eshkol_vm.c is added back below (compiled in library mode).
list(FILTER LIB_SRC EXCLUDE REGEX "lib/backend/[^/]*\\.c$")
# `archive/` holds historical SDNC predecessors (weight_matrices_v1.c,
# weight_matrices_v2.c, eshkol_compiler_standalone.c, qllm_interpreter_v1.c).
# These are reference-only — kept for paper-section history but never linked.
list(FILTER LIB_SRC EXCLUDE REGEX "lib/backend/archive/.*")

# Exclude standalone .cpp programs with main()
list(FILTER LIB_SRC EXCLUDE REGEX "lib/backend/weight_compiler\\.cpp$")
# platform_runtime.cpp stays in eshkol-static so the REPL, compiler, and JIT
# share the same path/process helpers on every platform.

# Native Windows REPL builds do not use the bytecode VM / ESKB compiler path.
# Building the VM currently drags in a large POSIX-only unity build, so use a
# stub there and keep the full VM everywhere else.
if(WIN32)
    list(APPEND LIB_SRC lib/backend/eshkol_vm_stub.c)
    list(APPEND LIB_SRC lib/backend/vm_autodiff.c)
    message(STATUS "Bytecode VM / ESKB emission: DISABLED for native Windows builds")
else()
    # Add eshkol_vm.c (unity build: includes vm_*.c, eskb_*.c internally)
    list(APPEND LIB_SRC lib/backend/eshkol_vm.c)
endif()

# Image I/O (stb_image — single-header, MIT/public domain)
list(APPEND LIB_SRC lib/core/image_io.c)

# C FFI (stable embedding API)
list(APPEND LIB_SRC lib/ffi/eshkol_ffi.cpp)

# Exclude XLA sources from main library when XLA is disabled
if(NOT ESHKOL_XLA_ENABLED)
    list(FILTER LIB_SRC EXCLUDE REGEX "lib/backend/xla/.*")
endif()

# Get REPL sources separately
file(GLOB REPL_SRC RELATIVE ${CMAKE_SOURCE_DIR} "lib/repl/*.cpp")

add_compile_definitions(
  ESHKOL_VER="${ESHKOL_VERSION}-scale"
)

if(ESHKOL_USING_MSVC)
    set(ESHKOL_WARNING_FLAGS /W4 /wd4100 /wd4018)
    set(ESHKOL_HARDENING_FLAGS /GS)
    set(ESHKOL_EXE_COMPILE_FLAGS "")
else()
    set(ESHKOL_WARNING_FLAGS
      -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
      -Wformat -Wformat-security
    )
    set(ESHKOL_HARDENING_FLAGS -fstack-protector-strong)
    set(ESHKOL_EXE_COMPILE_FLAGS -fPIE)
endif()

# Create static library for executables.
add_library(eshkol-static STATIC ${LIB_SRC})
target_include_directories(
  eshkol-static
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/>
    $<BUILD_INTERFACE:${ESHKOL_GENERATED_INCLUDE_DIR}>
)

# Always add LLVM support to static library
# Enable strict warnings and security hardening
target_compile_options(eshkol-static PRIVATE
  $<$<COMPILE_LANGUAGE:C,CXX>:${LLVM_CXXFLAGS_LIST}>
  $<$<COMPILE_LANGUAGE:C,CXX>:${ESHKOL_WARNING_FLAGS}>
  $<$<COMPILE_LANGUAGE:C,CXX>:${ESHKOL_HARDENING_FLAGS}>
)
# Buffer overflow detection at compile time (requires optimization)
if(NOT ESHKOL_USING_MSVC)
    target_compile_definitions(eshkol-static PRIVATE $<$<NOT:$<CONFIG:Debug>>:_FORTIFY_SOURCE=2>)
endif()
target_compile_definitions(eshkol-static PRIVATE ESHKOL_LLVM_BACKEND_ENABLED)
target_compile_definitions(eshkol-static PRIVATE ESHKOL_VM_LIBRARY_MODE)

# Add LLVM include directories for enhanced cross-platform support
if(LLVM_INCLUDE_DIRS)
    target_include_directories(eshkol-static PRIVATE ${LLVM_INCLUDE_DIRS})
endif()

# Extract library directories from LDFLAGS for proper linking
if(NOT LLVM_LIBRARY_DIRS)
    set(LLVM_LIBRARY_DIRS "")
    if(ESHKOL_USING_MSVC)
        string(REGEX MATCHALL "[-/]LIBPATH:[^ ]+" LLVM_LIBRARY_DIR_FLAGS "${LLVM_LDFLAGS}")
        foreach(lib_dir_flag ${LLVM_LIBRARY_DIR_FLAGS})
            string(REGEX REPLACE "^[-/]LIBPATH:" "" lib_dir "${lib_dir_flag}")
            file(TO_CMAKE_PATH "${lib_dir}" lib_dir)
            list(APPEND LLVM_LIBRARY_DIRS "${lib_dir}")
        endforeach()
    else()
        string(REGEX MATCHALL "-L[^ ]+" LLVM_LIBRARY_DIR_FLAGS "${LLVM_LDFLAGS}")
        foreach(lib_dir_flag ${LLVM_LIBRARY_DIR_FLAGS})
            string(REGEX REPLACE "^-L" "" lib_dir "${lib_dir_flag}")
            list(APPEND LLVM_LIBRARY_DIRS "${lib_dir}")
        endforeach()
    endif()
endif()

# Add library directories if found
if(LLVM_LIBRARY_DIRS)
    target_link_directories(eshkol-static PRIVATE ${LLVM_LIBRARY_DIRS})
endif()

# Link BLAS libraries if available
if(ESHKOL_BLAS_ENABLED AND BLAS_LIBRARIES)
    target_link_libraries(eshkol-static PUBLIC ${BLAS_LIBRARIES})
endif()

# Link MLIR and StableHLO libraries if available (for full XLA backend)
if(ESHKOL_XLA_ENABLED AND MLIR_AVAILABLE AND MLIR_LIBRARIES)
    target_link_libraries(eshkol-static PUBLIC ${MLIR_LIBRARIES})
    if(STABLEHLO_AVAILABLE AND STABLEHLO_LIBRARIES)
        target_link_libraries(eshkol-static PUBLIC ${STABLEHLO_LIBRARIES})
        message(STATUS "Linking StableHLO libraries: ${STABLEHLO_LIBRARIES}")
    endif()
endif()

# Add GPU sources (Metal on macOS, CUDA on Linux, or stub for no-GPU builds)
if(GPU_SOURCES)
    target_sources(eshkol-static PRIVATE ${GPU_SOURCES})
endif()
if(GPU_FOUND AND GPU_LIBRARIES)
    target_link_libraries(eshkol-static PUBLIC ${GPU_LIBRARIES})
    message(STATUS "Linking GPU libraries: ${GPU_LIBRARIES}")
endif()
# Metal shader embedding: gpu_memory.mm includes the generated .inc file
if(TARGET metal_sf64_shader)
    add_dependencies(eshkol-static metal_sf64_shader)
    target_include_directories(eshkol-static PRIVATE ${CMAKE_BINARY_DIR})
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
  target_compile_options(${exename} PRIVATE ${LLVM_CXXFLAGS_LIST} ${ESHKOL_EXE_COMPILE_FLAGS})
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
  
  target_link_libraries(${exename} PUBLIC eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
endforeach ()

# ===== Pre-compiled Standard Library =====
# Compile stdlib.esk to stdlib.o for faster compilation of user programs
set(STDLIB_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/lib/stdlib.esk")
set(STDLIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/stdlib.o")
set(STDLIB_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# stdlib.esk is a thin top-level that (require)s every module in
# lib/core/, lib/math/, lib/signal/, etc.  CMake's DEPENDS list must
# track those transitive sources too — otherwise edits to
# lib/core/json_schema.esk (or any other required module) don't
# trigger a stdlib rebuild and the user gets stale stdlib.o.  Glob
# the union; CONFIGURE_DEPENDS asks CMake to re-glob on every build,
# so newly-added .esk modules are picked up without needing a manual
# `cmake ..` re-run.
file(GLOB_RECURSE STDLIB_TRANSITIVE_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/lib/core/*.esk"
     "${CMAKE_CURRENT_SOURCE_DIR}/lib/math/*.esk"
     "${CMAKE_CURRENT_SOURCE_DIR}/lib/signal/*.esk"
     "${CMAKE_CURRENT_SOURCE_DIR}/lib/random/*.esk"
     "${CMAKE_CURRENT_SOURCE_DIR}/lib/web/*.esk"
     "${CMAKE_CURRENT_SOURCE_DIR}/lib/tensor/*.esk"
     "${CMAKE_CURRENT_SOURCE_DIR}/lib/quantum/*.esk"
     "${CMAKE_CURRENT_SOURCE_DIR}/lib/ml/*.esk"
     "${CMAKE_CURRENT_SOURCE_DIR}/lib/math.esk")

add_custom_command(
  OUTPUT ${STDLIB_OUTPUT}
  COMMAND ${CMAKE_COMMAND} -E env "ESHKOL_PATH=${CMAKE_CURRENT_SOURCE_DIR}/lib"
          $<TARGET_FILE:eshkol-run> --shared-lib -o "${CMAKE_CURRENT_BINARY_DIR}/stdlib" "${STDLIB_SOURCE}"
  DEPENDS eshkol-run ${STDLIB_SOURCE} ${STDLIB_TRANSITIVE_SOURCES}
  WORKING_DIRECTORY ${STDLIB_WORKING_DIRECTORY}
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

if(REPL_SRC)
  # REPL library
  add_library(eshkol-repl-lib STATIC ${REPL_SRC})

  target_include_directories(eshkol-repl-lib
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/repl
  )

  # REPL needs LLVM ORC JIT components
  target_compile_options(eshkol-repl-lib PRIVATE ${LLVM_CXXFLAGS_LIST})
  target_compile_definitions(eshkol-repl-lib PRIVATE ESHKOL_LLVM_BACKEND_ENABLED)

  if(LLVM_INCLUDE_DIRS)
    target_include_directories(eshkol-repl-lib PRIVATE ${LLVM_INCLUDE_DIRS})
  endif()

  if(LLVM_LIBRARY_DIRS)
    target_link_directories(eshkol-repl-lib PRIVATE ${LLVM_LIBRARY_DIRS})
  endif()

  # Link against main compiler library and LLVM
  target_link_libraries(eshkol-repl-lib PUBLIC eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})

  # REPL executable (only if we have the source)
  if(BUILD_REPL AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/exe/eshkol-repl.cpp")
    add_executable(eshkol-repl exe/eshkol-repl.cpp)

    target_compile_options(eshkol-repl PRIVATE ${LLVM_CXXFLAGS_LIST})
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

    target_link_libraries(eshkol-repl PRIVATE eshkol-repl-lib ${ESHKOL_EXTRA_LINK_LIBS})

    if(WIN32)
      set(_eshkol_llvm_c_dll "${ESHKOL_LLVM_ROOT}/bin/LLVM-C.dll")
      if(EXISTS "${_eshkol_llvm_c_dll}")
        add_custom_command(TARGET eshkol-repl POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
                  "${_eshkol_llvm_c_dll}"
                  "$<TARGET_FILE_DIR:eshkol-repl>/LLVM-C.dll"
          COMMENT "Copying LLVM-C.dll for native Windows REPL"
        )
      endif()

      set(_eshkol_repl_target_libs
        LLVMTarget
        LLVMTargetParser
        LLVMAsmPrinter
        LLVMCodeGen
      )
      if(CMAKE_VS_PLATFORM_NAME STREQUAL "ARM64" OR
         CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|AARCH64|aarch64)$")
        list(APPEND _eshkol_repl_target_libs
          LLVMAArch64CodeGen
          LLVMAArch64AsmParser
          LLVMAArch64Desc
          LLVMAArch64Info
          LLVMAArch64Disassembler
        )
      else()
        list(APPEND _eshkol_repl_target_libs
          LLVMX86CodeGen
          LLVMX86AsmParser
          LLVMX86Desc
          LLVMX86Info
          LLVMX86Disassembler
        )
      endif()
      foreach(_llvm_repl_lib IN LISTS _eshkol_repl_target_libs)
        set(_llvm_repl_lib_path "${ESHKOL_LLVM_ROOT}/lib/${_llvm_repl_lib}.lib")
        if(EXISTS "${_llvm_repl_lib_path}")
          target_link_options(eshkol-repl PRIVATE "/WHOLEARCHIVE:${_llvm_repl_lib_path}")
        endif()
      endforeach()
    endif()

    # Force-load ALL symbols from eshkol-static into the REPL binary.
    # Without this, the macOS linker dead-strips archive members that aren't
    # directly referenced (e.g. XLA runtime) but ARE needed by JIT-compiled
    # code at runtime. Combined with -export_dynamic below, this makes
    # DynamicLibrarySearchGenerator auto-resolve ALL runtime symbols.
    if(APPLE)
      target_link_options(eshkol-repl PRIVATE
        "-Wl,-force_load" "$<TARGET_FILE:eshkol-static>")
    elseif(WIN32)
      # Native Windows links the needed runtime objects directly via
      # eshkol-repl-lib's references and explicit JIT symbol registration.
    else()
      target_link_options(eshkol-repl PRIVATE
        "-Wl,--whole-archive" "$<TARGET_FILE:eshkol-static>" "-Wl,--no-whole-archive")
    endif()

    # Export all symbols for JIT symbol resolution
    if(APPLE)
      target_link_options(eshkol-repl PRIVATE "-Wl,-export_dynamic")
    elseif(WIN32)
      # No export-dynamic equivalent is required for the native Windows REPL path.
    else()
      target_link_options(eshkol-repl PRIVATE "-Wl,--export-dynamic")
    endif()

    # Install REPL executable
    install(TARGETS eshkol-repl DESTINATION bin)

    message(STATUS "REPL build enabled - target: eshkol-repl")
  endif()

  # ===== Link eshkol-run with REPL library for -e/--eval JIT support =====
  # This must come after eshkol-repl-lib is defined
  target_link_libraries(eshkol-run PRIVATE eshkol-repl-lib)

  # Agent FFI linkage is applied AFTER the eshkol-agent-ffi target is
  # defined (see the block at the end of the agent-FFI section further
  # down). At this point in the file the target doesn't exist yet, so
  # if(TARGET …) would evaluate false and silently drop the linkage.
  # Force-load eshkol-static for JIT symbol resolution (same as REPL).
  # Also force-load eshkol-repl-lib so its strong eval_bridge_impl.cpp
  # overrides (eshkol_eval_jit_get_*) beat the weak defaults in
  # eshkol-static. Without force_load, ld64 skips the TU because no
  # *other* symbol references it.
  if(APPLE)
    # ld64's -force_load only applies to the single archive that
    # immediately follows it. Combining the flag with two paths on the
    # same line still force-loads only the first archive. Use the
    # comma-prefixed -Wl,-force_load,PATH form so each archive gets its
    # own flag. This matters for eshkol-repl-lib: its eval_bridge_impl
    # TU has a static constructor that registers bridge pointers with
    # eshkol-static, but with only `eshkol_eval_bridge_linked` as an
    # external symbol the archive gets dead-stripped without
    # force_load.
    target_link_options(eshkol-run PRIVATE
      "-Wl,-force_load,$<TARGET_FILE:eshkol-static>"
      "-Wl,-force_load,$<TARGET_FILE:eshkol-repl-lib>")
    # NOTE: agent-ffi force_load is applied separately AFTER the
    # eshkol-agent-ffi target is defined later in this file (line
    # 1467+ area). See the matching block at the bottom of the agent
    # FFI section. This split is because `if(TARGET eshkol-agent-ffi)`
    # here evaluates at configure time and the target doesn't exist
    # yet.
    target_link_options(eshkol-run PRIVATE "-Wl,-export_dynamic")
    # 512MB stack for deep recursion (main thread stack set at link time on macOS)
    target_link_options(eshkol-run PRIVATE "-Wl,-stack_size,0x20000000")
    target_link_options(eshkol-run PRIVATE "-Wl,-bind_at_load")
  elseif(WIN32)
    # Native Windows uses the REPL/JIT symbol registration path and does not
    # accept GNU ld options here.
  else()
    target_link_options(eshkol-run PRIVATE
      "-Wl,--whole-archive"
      "$<TARGET_FILE:eshkol-static>"
      "$<TARGET_FILE:eshkol-repl-lib>"
      "-Wl,--no-whole-archive")
    target_link_options(eshkol-run PRIVATE "-Wl,--export-dynamic")
    # 512MB stack for deep recursion
    target_link_options(eshkol-run PRIVATE "-Wl,-z,stack-size=536870912")
    # Linker hardening: full RELRO + immediate binding
    target_link_options(eshkol-run PRIVATE "-Wl,-z,relro" "-Wl,-z,now")
  endif()
  message(STATUS "eshkol-run linked with REPL JIT for -e/--eval support")

  # Install compiler executable
  install(TARGETS eshkol-run DESTINATION bin)
endif()

# ===== Package Manager =====
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/pkg/eshkol_pkg.cpp")
    message(STATUS "Building package manager: eshkol-pkg")

    add_executable(eshkol-pkg tools/pkg/eshkol_pkg.cpp)
    target_compile_features(eshkol-pkg PRIVATE cxx_std_17)
    target_include_directories(eshkol-pkg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)

    install(TARGETS eshkol-pkg DESTINATION bin)

    if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/pkg/path_validation_test.cpp")
        add_executable(eshkol-pkg-path-validation-test tests/pkg/path_validation_test.cpp)
        target_compile_features(eshkol-pkg-path-validation-test PRIVATE cxx_std_17)
        target_include_directories(eshkol-pkg-path-validation-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
        add_test(
            NAME eshkol-pkg-path-validation
            COMMAND eshkol-pkg-path-validation-test $<TARGET_FILE:eshkol-pkg>
        )
    endif()

    if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/pkg/command_injection_test.cpp")
        add_executable(eshkol-pkg-command-injection-test tests/pkg/command_injection_test.cpp)
        target_compile_features(eshkol-pkg-command-injection-test PRIVATE cxx_std_17)
        target_include_directories(eshkol-pkg-command-injection-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
        add_test(
            NAME eshkol-pkg-command-injection
            COMMAND eshkol-pkg-command-injection-test $<TARGET_FILE:eshkol-pkg>
        )
    endif()
endif()

# ===== LSP Server =====
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/lsp/eshkol_lsp.cpp")
    message(STATUS "Building LSP server: eshkol-lsp")

    add_executable(eshkol-lsp tools/lsp/eshkol_lsp.cpp)

    target_compile_options(eshkol-lsp PRIVATE ${LLVM_CXXFLAGS_LIST})
    target_compile_definitions(eshkol-lsp PRIVATE ESHKOL_LLVM_BACKEND_ENABLED)
    target_include_directories(eshkol-lsp PRIVATE inc/)

    if(LLVM_INCLUDE_DIRS)
        target_include_directories(eshkol-lsp PRIVATE ${LLVM_INCLUDE_DIRS})
    endif()

    if(LLVM_LIBRARY_DIRS)
        target_link_directories(eshkol-lsp PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()

    target_link_libraries(eshkol-lsp PRIVATE eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})

    install(TARGETS eshkol-lsp DESTINATION bin)
endif()

# ===== HTTP Request Utility Tests =====
if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/web/content_length_parse_test.cpp")
    add_executable(content_length_parse_test tests/web/content_length_parse_test.cpp)
    target_compile_features(content_length_parse_test PRIVATE cxx_std_17)
    target_include_directories(content_length_parse_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    add_test(NAME content_length_parse_test COMMAND content_length_parse_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/model_io_test.cpp")
    add_executable(model_io_test tests/core/model_io_test.cpp)
    target_compile_features(model_io_test PRIVATE cxx_std_20)
    target_include_directories(model_io_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    if(LLVM_INCLUDE_DIRS)
        target_include_directories(model_io_test PRIVATE ${LLVM_INCLUDE_DIRS})
    endif()
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(model_io_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(model_io_test PRIVATE eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    add_test(NAME model_io_test COMMAND model_io_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/repl/assoc_eval_test.cpp")
    add_executable(assoc_eval_test tests/repl/assoc_eval_test.cpp)
    target_compile_features(assoc_eval_test PRIVATE cxx_std_17)
    add_dependencies(assoc_eval_test stdlib)
    add_test(
        NAME assoc_eval_test
        COMMAND assoc_eval_test
                $<TARGET_FILE:eshkol-run>
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${STDLIB_OUTPUT}
    )
endif()

# ===== SDNC Paper Artifact: weight_matrices ===================================
# The paper "The Self-Differentiating Neural Computer: Computable Transformers
# via Analytical Weight Construction" (tsotchke 2026) reports a 2.8M-parameter
# transformer (d_model=128, n_layers=6, FFN=1024) whose weights ARE the Eshkol
# VM's microcode. The 74-test verification suite runs the same programs three
# ways (reference C interpreter / simulated transformer / matrix forward pass)
# and asserts bit-identical agreement.
#
# `lib/backend/weight_matrices.c` is a standalone program with its own main();
# it is NOT compiled into libeshkol-static (see the `lib/backend/[^/]*\\.c$`
# exclude regex above). Build it as a dedicated executable so
# `scripts/paper/export_weights.sh` and `scripts/paper/run_paper_suite.sh`
# can invoke it without out-of-band compile commands.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/backend/weight_matrices.c")
    add_executable(weight_matrices lib/backend/weight_matrices.c)
    target_include_directories(weight_matrices PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/backend
    )
    target_link_libraries(weight_matrices PRIVATE eshkol-static m)
    if(UNIX AND NOT APPLE)
        target_link_libraries(weight_matrices PRIVATE pthread)
    endif()
    set_target_properties(weight_matrices PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tools
    )
    if(ESHKOL_BUILD_TESTS)
        add_test(
            NAME sdnc_paper_74_tests
            COMMAND $<TARGET_FILE:weight_matrices>
        )
        # The test program prints "Results: 74 passed, 0 failed" on success.
        set_tests_properties(sdnc_paper_74_tests PROPERTIES
            PASS_REGULAR_EXPRESSION "74 passed, 0 failed"
        )
    endif()
endif()

# ===== XLA Integration Tests =====
if(ESHKOL_XLA_ENABLED AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/xla/xla_codegen_test.cpp")
    message(STATUS "Building XLA integration tests")

    add_executable(xla_codegen_test tests/xla/xla_codegen_test.cpp)

    target_include_directories(xla_codegen_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
    )

    target_compile_options(xla_codegen_test PRIVATE ${LLVM_CXXFLAGS_LIST})

    if(LLVM_INCLUDE_DIRS)
        target_include_directories(xla_codegen_test PRIVATE ${LLVM_INCLUDE_DIRS})
    endif()

    if(LLVM_LIBRARY_DIRS)
        target_link_directories(xla_codegen_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()

    # Link against the static library (includes XLA runtime)
    target_link_libraries(xla_codegen_test PRIVATE
        eshkol-static
        ${ESHKOL_EXTRA_LINK_LIBS}
        ${LLVM_LIBS_LIST}
        ${LLVM_SYSTEM_LIBS_LIST}
    )

    # Link MLIR/StableHLO if available
    if(MLIR_AVAILABLE AND MLIR_LIBRARIES)
        target_link_libraries(xla_codegen_test PRIVATE ${MLIR_LIBRARIES})
    endif()
    if(STABLEHLO_LIBRARIES)
        target_link_libraries(xla_codegen_test PRIVATE ${STABLEHLO_LIBRARIES})
    endif()

    # Platform-specific link options
    if(APPLE)
        target_link_options(xla_codegen_test PRIVATE "-Wl,-export_dynamic")
        target_link_libraries(xla_codegen_test PRIVATE "-framework Accelerate")
    else()
        target_link_options(xla_codegen_test PRIVATE "-Wl,--export-dynamic")
    endif()
endif()

# ===== Agent FFI Modules =====
# C bridge libraries for the Eshkol agent (terminal, regex, sqlite, crypto).
# The Unix implementations use POSIX-only APIs (termios, pthread, fork/exec,
# /dev/tty); the Windows ports (ConPTY, CreateProcess+pipes, Windows
# Credential Manager) are slated for v1.4-platform. Default to OFF on
# Windows so the rest of the toolchain (LLVM backend, REPL, codegen,
# stdlib) builds and tests cleanly there in the meantime.
if(WIN32)
    option(ESHKOL_BUILD_AGENT_FFI "Build agent FFI modules (POSIX-only)" OFF)
else()
    option(ESHKOL_BUILD_AGENT_FFI "Build agent FFI modules" ON)
endif()

if(ESHKOL_BUILD_AGENT_FFI)
    set(AGENT_FFI_DIR "${CMAKE_SOURCE_DIR}/lib/agent/c")

    if(EXISTS "${AGENT_FFI_DIR}")
        set(AGENT_FFI_SOURCES
            ${AGENT_FFI_DIR}/agent_terminal.c
            ${AGENT_FFI_DIR}/agent_crypto.c
            ${AGENT_FFI_DIR}/agent_subprocess.c
        )

        # Optional: PCRE2 for regex. Homebrew ships the .pc file as
        # libpcre2-8.pc, some distros (Debian) as pcre2-8.pc — try
        # the homebrew-style name first, fall back to the bare one
        # so Linux distros continue to build.
        pkg_check_modules(PCRE2_AGENT libpcre2-8)
        if(NOT PCRE2_AGENT_FOUND)
            pkg_check_modules(PCRE2_AGENT pcre2-8)
        endif()
        if(PCRE2_AGENT_FOUND)
            list(APPEND AGENT_FFI_SOURCES ${AGENT_FFI_DIR}/agent_regex.c)
        endif()

        # Optional: SQLite3
        pkg_check_modules(SQLITE3_AGENT sqlite3)
        if(SQLITE3_AGENT_FOUND)
            list(APPEND AGENT_FFI_SOURCES ${AGENT_FFI_DIR}/agent_sqlite.c)
        endif()

        add_library(eshkol-agent-ffi STATIC ${AGENT_FFI_SOURCES})

        target_include_directories(eshkol-agent-ffi PRIVATE
            ${CMAKE_SOURCE_DIR}/include
        )

        # NOTE: The following dependencies are declared PUBLIC rather
        # than PRIVATE so they propagate transitively to any target
        # that links eshkol-agent-ffi (most importantly eshkol-run).
        # When eshkol-run uses -force_load / --whole-archive to keep
        # agent FFI symbols alive for JIT dlsym, the linker also
        # needs sqlite3 / pcre2 / Security framework / OpenSSL to
        # resolve the symbols those agent objects reference
        # internally — otherwise we get "Undefined symbols for
        # architecture arm64: _sqlite3_exec, _SecRandomCopyBytes,
        # etc." at eshkol-run's link step.

        # Terminal needs ncurses
        find_library(NCURSES_AGENT ncurses)
        if(NCURSES_AGENT)
            target_link_libraries(eshkol-agent-ffi PUBLIC ${NCURSES_AGENT})
        endif()

        if(PCRE2_AGENT_FOUND)
            target_include_directories(eshkol-agent-ffi PRIVATE ${PCRE2_AGENT_INCLUDE_DIRS})
            # Use LINK_LIBRARIES (full absolute paths resolved by
            # pkg_check_modules) instead of LIBRARIES (short names
            # that also need LIBRARY_DIRS via target_link_directories,
            # easy to forget). Falls back to the short-name+dirs form
            # if the installed CMake is too old to populate
            # *_LINK_LIBRARIES.
            if(PCRE2_AGENT_LINK_LIBRARIES)
                target_link_libraries(eshkol-agent-ffi PUBLIC ${PCRE2_AGENT_LINK_LIBRARIES})
            else()
                target_link_directories(eshkol-agent-ffi PUBLIC ${PCRE2_AGENT_LIBRARY_DIRS})
                target_link_libraries(eshkol-agent-ffi PUBLIC ${PCRE2_AGENT_LIBRARIES})
            endif()
            target_compile_definitions(eshkol-agent-ffi PRIVATE PCRE2_CODE_UNIT_WIDTH=8)
        endif()

        if(SQLITE3_AGENT_FOUND)
            if(SQLITE3_AGENT_LINK_LIBRARIES)
                target_link_libraries(eshkol-agent-ffi PUBLIC ${SQLITE3_AGENT_LINK_LIBRARIES})
            else()
                target_link_directories(eshkol-agent-ffi PUBLIC ${SQLITE3_AGENT_LIBRARY_DIRS})
                target_link_libraries(eshkol-agent-ffi PUBLIC ${SQLITE3_AGENT_LIBRARIES})
            endif()
        endif()

        # Crypto: CommonCrypto on macOS, OpenSSL on Linux
        if(APPLE)
            target_link_libraries(eshkol-agent-ffi PUBLIC
                "-framework Security"
                "-framework CoreFoundation"
            )
        else()
            find_package(OpenSSL)
            if(OpenSSL_FOUND)
                target_link_libraries(eshkol-agent-ffi PUBLIC OpenSSL::SSL OpenSSL::Crypto)
            endif()
        endif()

        message(STATUS "Agent FFI modules: ENABLED")
        message(STATUS "  Terminal:  YES")
        message(STATUS "  Crypto:    YES")
        message(STATUS "  Regex:     ${PCRE2_AGENT_FOUND}")
        message(STATUS "  SQLite:    ${SQLITE3_AGENT_FOUND}")

        # Force-load eshkol-agent-ffi into eshkol-run so the (extern …
        # :real eshkol_*) declarations in lib/agent/*.esk can be
        # resolved by the JIT at runtime. The agent FFI symbols are
        # never referenced from eshkol-run's own C++ code — they exist
        # purely to satisfy Scheme-side FFI lookups via
        # dlsym(RTLD_DEFAULT). Without force_load, the whole archive
        # gets dead-stripped and every (load "lib/agent/regex.esk")
        # under -e JIT dies with "Symbols not found".
        #
        # Must happen here, not in the eshkol-run target section above,
        # because the eshkol-agent-ffi target doesn't exist yet at that
        # point (CMake evaluates if(TARGET…) at configure time).
        #
        # Audit reference: 2026-04-18 v2 BUG 3 ("all lib/agent/*.esk
        # FFI modules fail under JIT").
        if(TARGET eshkol-run)
            # Link eshkol-agent-ffi + its transitive PUBLIC deps
            # (sqlite3, pcre2, Security framework, OpenSSL, ncurses)
            # into eshkol-run. MUST happen here, not in the eshkol-run
            # config block above, because agent-ffi is defined here.
            target_link_libraries(eshkol-run PRIVATE eshkol-agent-ffi)
            if(APPLE)
                target_link_options(eshkol-run PRIVATE
                    "-Wl,-force_load,$<TARGET_FILE:eshkol-agent-ffi>")
            elseif(NOT WIN32)
                target_link_options(eshkol-run PRIVATE
                    "-Wl,--whole-archive"
                    "$<TARGET_FILE:eshkol-agent-ffi>"
                    "-Wl,--no-whole-archive")
            endif()
        endif()

        if(TARGET eshkol-repl)
            # Keep the long-lived REPL's JIT resolver on the same footing
            # as eshkol-run -r: agent FFI externs loaded from .esk files
            # should be available in persistent control sessions too.
            target_link_libraries(eshkol-repl PRIVATE eshkol-agent-ffi)
            if(APPLE)
                target_link_options(eshkol-repl PRIVATE
                    "-Wl,-force_load,$<TARGET_FILE:eshkol-agent-ffi>")
            elseif(NOT WIN32)
                target_link_options(eshkol-repl PRIVATE
                    "-Wl,--whole-archive"
                    "$<TARGET_FILE:eshkol-agent-ffi>"
                    "-Wl,--no-whole-archive")
            endif()
        endif()
    else()
        message(STATUS "Agent FFI modules: lib/agent/c/ not found, skipping")
    endif()
endif()

# ===== Python Bindings (pybind11) =====
option(ESHKOL_PYTHON_BINDINGS "Build Python bindings via pybind11" OFF)

if(ESHKOL_PYTHON_BINDINGS)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        pybind11_add_module(eshkol_py bindings/python/eshkol_module.cpp)
        target_include_directories(eshkol_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
        if(LLVM_LIBRARY_DIRS)
            target_link_directories(eshkol_py PRIVATE ${LLVM_LIBRARY_DIRS})
        endif()
        # Force-load eshkol-static so JIT can resolve all runtime symbols
        if(APPLE)
            target_link_libraries(eshkol_py PRIVATE eshkol-repl-lib
                "-Wl,-force_load" "$<TARGET_FILE:eshkol-static>"
                ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
        else()
            target_link_libraries(eshkol_py PRIVATE eshkol-repl-lib
                "-Wl,--whole-archive" "$<TARGET_FILE:eshkol-static>" "-Wl,--no-whole-archive"
                ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
        endif()
        if(TARGET eshkol-agent-ffi)
            target_link_libraries(eshkol_py PRIVATE eshkol-agent-ffi)
        endif()
        # Platform libraries
        if(APPLE)
            target_link_libraries(eshkol_py PRIVATE
                "-framework Accelerate" "-framework Metal"
                "-framework MetalPerformanceShaders" "-framework Foundation"
                "-lobjc" "-lncurses")
        endif()
        set_target_properties(eshkol_py PROPERTIES OUTPUT_NAME "eshkol")
        message(STATUS "Python bindings: ENABLED (pybind11 ${pybind11_VERSION})")
    else()
        message(STATUS "Python bindings: pybind11 not found, skipping")
    endif()
endif()

if(ESHKOL_ENABLE_FUZZ AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/fuzz/CMakeLists.txt")
    add_subdirectory(tests/fuzz)
endif()

# where to find our CMake modules
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Packing)
