#
# Copyright (C) tsotchke
#
# SPDX-License-Identifier: MIT
#
cmake_minimum_required(VERSION 3.14)
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()

set(ESHKOL_VERSION "1.3.1")

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

if(APPLE AND EXISTS "/usr/bin/c++" AND
   (NOT ESHKOL_HOST_CXX_COMPILER_EXPLICIT OR
    ESHKOL_HOST_CXX_COMPILER MATCHES "/Library/Developer/CommandLineTools/usr/bin/c\\+\\+$"))
    # Generated executable links run outside CMake's full driver context.
    # The CLT-internal c++ path can fail to locate libc++; the xcrun shim
    # preserves Apple's SDK lookup behavior.
    set(ESHKOL_HOST_CXX_COMPILER "/usr/bin/c++" CACHE FILEPATH
        "C++ compiler driver used by eshkol-run when linking generated programs" FORCE)
endif()

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
    list(APPEND ESHKOL_EXTRA_LINK_LIBS Advapi32.lib Ntdll.lib)
endif()

include(CTest)
include(GNUInstallDirs)
include(FindPkgConfig)

# Native image I/O backends: macOS uses ImageIO/CoreGraphics, Windows uses
# GDI+, and Linux uses distro codec libraries.
if(APPLE)
    find_library(ESHKOL_IMAGEIO_FRAMEWORK ImageIO REQUIRED)
    find_library(ESHKOL_COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)
    find_library(ESHKOL_COREFOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
    add_compile_definitions(ESHKOL_IMAGE_IO_APPLE=1)
    list(APPEND ESHKOL_EXTRA_LINK_LIBS
        "-framework ImageIO"
        "-framework CoreGraphics"
        "-framework CoreFoundation")
    message(STATUS "Image I/O: Apple ImageIO/CoreGraphics")
elseif(WIN32)
    add_compile_definitions(ESHKOL_IMAGE_IO_GDIPLUS=1)
    list(APPEND ESHKOL_EXTRA_LINK_LIBS Gdiplus.lib)
    message(STATUS "Image I/O: Windows GDI+")
else()
    find_package(PNG QUIET)
    if(PNG_FOUND)
        find_package(JPEG QUIET)
        pkg_check_modules(WEBP QUIET libwebp)

        add_compile_definitions(ESHKOL_IMAGE_IO_LIBPNG=1)
        include_directories(SYSTEM ${PNG_INCLUDE_DIRS})
        list(APPEND ESHKOL_EXTRA_LINK_LIBS PNG::PNG)

        set(_eshkol_image_io_backends "libpng")
        if(JPEG_FOUND)
            add_compile_definitions(ESHKOL_IMAGE_IO_LIBJPEG=1)
            include_directories(SYSTEM ${JPEG_INCLUDE_DIRS})
            list(APPEND ESHKOL_EXTRA_LINK_LIBS JPEG::JPEG)
            list(APPEND _eshkol_image_io_backends "libjpeg")
        endif()
        if(WEBP_FOUND)
            add_compile_definitions(ESHKOL_IMAGE_IO_LIBWEBP=1)
            include_directories(SYSTEM ${WEBP_INCLUDE_DIRS})
            list(APPEND ESHKOL_EXTRA_LINK_LIBS ${WEBP_LDFLAGS})
            list(APPEND _eshkol_image_io_backends "libwebp")
        endif()
        string(JOIN "/" _eshkol_image_io_backend_label ${_eshkol_image_io_backends})
        message(STATUS "Image I/O: ${_eshkol_image_io_backend_label}")
    else()
        message(STATUS
            "Image I/O: native codec backends disabled "
            "(install libpng-dev/libjpeg-dev/libwebp-dev to enable Linux image read/write)")
    endif()
endif()

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

# ===== Profile-Guided Optimization (v1.3-evolve) =====
# Two-phase build:
#   1. `cmake -DESHKOL_PGO=generate -DCMAKE_BUILD_TYPE=Release`
#      builds the compiler/runtime instrumented to emit profile data
#      via `-fprofile-instr-generate=<dir>/eshkol-%p.profraw`.
#   2. Run a representative workload (typically the v1.2 edge-case
#      tests + the smoke harness + a couple of canonical examples).
#   3. `llvm-profdata merge` the raw files into a single profile.
#   4. `cmake -DESHKOL_PGO=use -DESHKOL_PGO_PROFILE=<merged.profdata>
#         -DCMAKE_BUILD_TYPE=Release` rebuilds using the profile.
#
# The default (OFF / unset) is the existing non-PGO build path.
set(ESHKOL_PGO "OFF" CACHE STRING
    "Profile-Guided Optimization mode: OFF (default), generate (instrument), or use (consume profile)")
set_property(CACHE ESHKOL_PGO PROPERTY STRINGS OFF generate use)
set(ESHKOL_PGO_PROFILE "" CACHE FILEPATH
    "Path to the merged .profdata file when ESHKOL_PGO=use")
set(ESHKOL_PGO_RAW_DIR "${CMAKE_BINARY_DIR}/pgo-raw" CACHE PATH
    "Directory where instrumented binaries write .profraw files")

if(NOT ESHKOL_PGO STREQUAL "OFF")
    if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang"))
        message(WARNING "PGO support is currently clang-only; ESHKOL_PGO ignored.")
    elseif(ESHKOL_PGO STREQUAL "generate")
        message(STATUS "PGO: instrumentation phase (writes .profraw to ${ESHKOL_PGO_RAW_DIR})")
        file(MAKE_DIRECTORY "${ESHKOL_PGO_RAW_DIR}")
        add_compile_options(-fprofile-instr-generate=${ESHKOL_PGO_RAW_DIR}/eshkol-%p.profraw)
        add_link_options(-fprofile-instr-generate=${ESHKOL_PGO_RAW_DIR}/eshkol-%p.profraw)
        add_compile_definitions(ESHKOL_PGO_INSTRUMENTED=1)
    elseif(ESHKOL_PGO STREQUAL "use")
        if(NOT ESHKOL_PGO_PROFILE)
            message(FATAL_ERROR "ESHKOL_PGO=use requires -DESHKOL_PGO_PROFILE=<merged.profdata>")
        endif()
        if(NOT EXISTS "${ESHKOL_PGO_PROFILE}")
            message(FATAL_ERROR "ESHKOL_PGO_PROFILE not found: ${ESHKOL_PGO_PROFILE}")
        endif()
        message(STATUS "PGO: consumption phase (reading ${ESHKOL_PGO_PROFILE})")
        # -Wno-profile-instr-out-of-date keeps the build alive when the
        # profile is captured against a slightly different source tree.
        add_compile_options(-fprofile-instr-use=${ESHKOL_PGO_PROFILE}
                            -Wno-profile-instr-out-of-date
                            -Wno-profile-instr-unprofiled)
        add_link_options(-fprofile-instr-use=${ESHKOL_PGO_PROFILE})
        add_compile_definitions(ESHKOL_PGO_OPTIMIZED=1)
    else()
        message(FATAL_ERROR "ESHKOL_PGO must be OFF, generate, or use (got '${ESHKOL_PGO}')")
    endif()
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

# Generated build config for toolchain/runtime helpers.  Keep this available
# for every build mode; hosted runtime code includes it before the lite/XLA
# target graph is assembled.
set(ESHKOL_HOST_RUNTIME_LINK_ARGS "")
set(ESHKOL_HOST_LLVM_LINK_ARGS "")
set(ESHKOL_HOST_AGENT_FFI_LINK_ARGS "")
set(ESHKOL_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${ESHKOL_GENERATED_INCLUDE_DIR}/eshkol")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_config.h.in"
  "${ESHKOL_GENERATED_INCLUDE_DIR}/eshkol/build_config.h"
  @ONLY
)

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

    if(WIN32 AND NOT MINGW)
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
        set(LLVM_SYSTEM_LIBS_LIST ${LLVM_SYSTEM_LIBS})

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

function(eshkol_resolve_host_link_arg item out_var)
    set(_eshkol_host_link_arg "")

    if(NOT "${item}" STREQUAL "")
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
                    if("${_eshkol_host_link_arg}" STREQUAL "")
                        get_target_property(_candidate "${item}" ${_prop})
                        if(_candidate AND NOT _candidate MATCHES "-NOTFOUND$")
                            set(_eshkol_host_link_arg "${_candidate}")
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
                    if("${_eshkol_host_link_arg}" STREQUAL "")
                        get_target_property(_candidate "${item}" ${_prop})
                        if(_candidate AND NOT _candidate MATCHES "-NOTFOUND$")
                            set(_eshkol_host_link_arg "${_candidate}")
                        endif()
                    endif()
                endforeach()
            endif()
        else()
            set(_eshkol_host_link_arg "${item}")
            if(WIN32 AND
               NOT _eshkol_host_link_arg MATCHES "^[-/]" AND
               NOT _eshkol_host_link_arg MATCHES "[/\\\\]" AND
               NOT _eshkol_host_link_arg MATCHES "\\.(a|so|dylib|lib)$")
                set(_eshkol_found_lib "")
                foreach(_llvm_lib_dir IN LISTS LLVM_LIBRARY_DIRS)
                    foreach(_suffix IN ITEMS ".lib" ".a")
                        if("${_eshkol_found_lib}" STREQUAL "" AND
                           EXISTS "${_llvm_lib_dir}/${_eshkol_host_link_arg}${_suffix}")
                            set(_eshkol_found_lib "${_llvm_lib_dir}/${_eshkol_host_link_arg}${_suffix}")
                        endif()
                    endforeach()
                endforeach()
                if(NOT "${_eshkol_found_lib}" STREQUAL "")
                    set(_eshkol_host_link_arg "${_eshkol_found_lib}")
                endif()
            endif()
        endif()
    endif()

    if(NOT "${_eshkol_host_link_arg}" STREQUAL "")
        file(TO_CMAKE_PATH "${_eshkol_host_link_arg}" _eshkol_host_link_arg)
    endif()
    set(${out_var} "${_eshkol_host_link_arg}" PARENT_SCOPE)
endfunction()

set(ESHKOL_HOST_LLVM_LINK_ARGS "")
foreach(_llvm_lib_dir IN LISTS LLVM_LIBRARY_DIRS)
    list(APPEND ESHKOL_HOST_LLVM_LINK_ARGS "-L${_llvm_lib_dir}")
    # Also bake an RPATH so AOT-compiled programs (which link libLLVM through
    # these args, incl. the -r persistent run-cache binary) can find libLLVM.so
    # at runtime. On ELF (Linux/*nix — including Nix-store library paths) a -L
    # alone leaves NO runtime search path, so the cached run binary dies with
    # "libLLVM.so.NN: cannot open shared object file". macOS resolves via the
    # dylib's absolute install_name and Windows via PATH, so this is ELF-only.
    if(NOT APPLE AND NOT WIN32 AND NOT ESHKOL_USING_MSVC)
        list(APPEND ESHKOL_HOST_LLVM_LINK_ARGS "-Wl,-rpath,${_llvm_lib_dir}")
    endif()
endforeach()
# In CMake-package mode, LLVM_LIBS_LIST holds CMake target / component names
# (e.g. "LLVMSupport", "LLVMX86CodeGen") rather than file paths or -l flags.
# These are baked into ESHKOL_HOST_LLVM_LINK_ARGS, which the `-r` persistent
# run-cache passes verbatim to the compiler driver when linking the cached AOT
# binary. The MinGW/GNU driver then treats each bare name as a filename and
# fails: "clang++: error: no such file or directory: 'LLVMSupport'". Resolve
# each component name to its concrete archive path so the run-cache link works.
# (MSVC/ClangCL keeps the legacy behaviour; its run-cache path is unaffected.)
function(eshkol_resolve_llvm_lib_arg out_var lib_arg)
    set(${out_var} "${lib_arg}" PARENT_SCOPE)
    if(NOT ESHKOL_LLVM_USE_CMAKE_PACKAGE OR ESHKOL_USING_MSVC)
        return()
    endif()
    # Already a path or a flag? leave it.
    if(lib_arg MATCHES "[/\\]" OR lib_arg MATCHES "^-")
        return()
    endif()
    foreach(_dir IN LISTS LLVM_LIBRARY_DIRS)
        foreach(_cand "${_dir}/lib${lib_arg}.a" "${_dir}/${lib_arg}.lib" "${_dir}/lib${lib_arg}.dll.a")
            if(EXISTS "${_cand}")
                set(${out_var} "${_cand}" PARENT_SCOPE)
                return()
            endif()
        endforeach()
    endforeach()
endfunction()

foreach(_llvm_link_arg IN LISTS LLVM_LDFLAGS_LIST LLVM_LIBS_LIST LLVM_SYSTEM_LIBS_LIST)
    if(NOT "${_llvm_link_arg}" STREQUAL "")
        eshkol_resolve_host_link_arg("${_llvm_link_arg}" _eshkol_host_llvm_link_arg)
        if(NOT "${_eshkol_host_llvm_link_arg}" STREQUAL "")
            eshkol_resolve_llvm_lib_arg(_eshkol_host_llvm_link_arg_resolved "${_eshkol_host_llvm_link_arg}")
            list(APPEND ESHKOL_HOST_LLVM_LINK_ARGS "${_eshkol_host_llvm_link_arg_resolved}")
        endif()
    endif()
endforeach()

# Keep LLVM's defines and unwind flags, but let the project control language mode
# and exception support.
if(ESHKOL_USING_MSVC)
    list(FILTER LLVM_CXXFLAGS_LIST EXCLUDE REGEX "^-I")
    list(FILTER LLVM_CXXFLAGS_LIST EXCLUDE REGEX "^[-/]std[:=]")
    list(FILTER LLVM_CXXFLAGS_LIST EXCLUDE REGEX "^/EH")
    list(FILTER LLVM_CXXFLAGS_LIST EXCLUDE REGEX "^-fno-exceptions$")
else()
    list(FILTER LLVM_CXXFLAGS_LIST EXCLUDE REGEX "^-I")
    list(FILTER LLVM_CXXFLAGS_LIST EXCLUDE REGEX "^-std=")
    list(FILTER LLVM_CXXFLAGS_LIST EXCLUDE REGEX "^-fno-exceptions$")
endif()

set(ESHKOL_LLVM_INCLUDE_OPTIONS "")
if(LLVM_INCLUDE_DIRS AND NOT ESHKOL_USING_MSVC)
    foreach(llvm_include_dir IN LISTS LLVM_INCLUDE_DIRS)
        list(APPEND ESHKOL_LLVM_INCLUDE_OPTIONS "SHELL:-idirafter ${llvm_include_dir}")
    endforeach()
endif()

function(eshkol_target_llvm_includes target_name)
    if(NOT LLVM_INCLUDE_DIRS)
        return()
    endif()

    if(ESHKOL_USING_MSVC)
        target_include_directories(${target_name} PRIVATE ${LLVM_INCLUDE_DIRS})
    else()
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:C,CXX>:${ESHKOL_LLVM_INCLUDE_OPTIONS}>)
    endif()
endfunction()

# ===== Windows (MinGW/MSYS2) runtime symbol export for the in-process JIT =====
# Generates a .def listing the Eshkol runtime namespace from the runtime
# archives and links the given executable against it, so
# DynamicLibrarySearchGenerator::GetForCurrentProcess can resolve host symbols
# (the PE equivalent of --export-dynamic). See cmake/gen_runtime_def.cmake for
# the rationale (PE 65535-export cap rules out --export-all-symbols).
function(eshkol_windows_export_runtime target_name)
    if(NOT WIN32 OR ESHKOL_USING_MSVC)
        return()
    endif()

    # Locate an nm for symbol discovery (prefer llvm-nm shipped with the SDK).
    if(NOT ESHKOL_RUNTIME_DEF_NM)
        find_program(ESHKOL_RUNTIME_DEF_NM
            NAMES llvm-nm llvm-nm.exe nm nm.exe
            HINTS "${ESHKOL_LLVM_ROOT}/bin")
    endif()
    if(NOT ESHKOL_RUNTIME_DEF_NM)
        message(WARNING
            "eshkol_windows_export_runtime: no nm found; ${target_name} runtime "
            "symbols will not be exported and `-r` JIT lookups will fail.")
        return()
    endif()

    set(_def_file "${CMAKE_BINARY_DIR}/${target_name}_runtime_exports.def")
    add_custom_command(
        OUTPUT "${_def_file}"
        COMMAND "${CMAKE_COMMAND}"
                "-DARCHIVES=$<TARGET_FILE:eshkol-static>;$<TARGET_FILE:eshkol-runtime>;$<TARGET_FILE:eshkol-repl-lib>"
                "-DNM=${ESHKOL_RUNTIME_DEF_NM}"
                "-DOUTPUT=${_def_file}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gen_runtime_def.cmake"
        DEPENDS eshkol-static eshkol-runtime eshkol-repl-lib
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gen_runtime_def.cmake"
        COMMENT "Generating Eshkol runtime export .def for ${target_name}"
        VERBATIM)

    add_custom_target(${target_name}_runtime_def DEPENDS "${_def_file}")
    add_dependencies(${target_name} ${target_name}_runtime_def)

    # Passing a .def as a linker input makes the GNU driver export exactly its
    # EXPORTS list (and nothing else).
    target_link_options(${target_name} PRIVATE "${_def_file}")
endfunction()

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
            if((NOT DEFINED CMAKE_CUDA_HOST_COMPILER OR "${CMAKE_CUDA_HOST_COMPILER}" STREQUAL "") AND
               CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
               CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
                find_program(ESHKOL_CUDA_HOST_COMPILER_CANDIDATE
                    NAMES g++-14 gcc-14 g++-13 gcc-13 g++-12 gcc-12
                )
                if(ESHKOL_CUDA_HOST_COMPILER_CANDIDATE)
                    set(CMAKE_CUDA_HOST_COMPILER "${ESHKOL_CUDA_HOST_COMPILER_CANDIDATE}"
                        CACHE FILEPATH "Host compiler used by nvcc" FORCE)
                    message(STATUS
                        "CUDA: system GNU ${CMAKE_CXX_COMPILER_VERSION} detected; "
                        "using host compiler ${CMAKE_CUDA_HOST_COMPILER}")
                else()
                    message(WARNING
                        "CUDA: system GNU ${CMAKE_CXX_COMPILER_VERSION} may be unsupported by nvcc; "
                        "install gcc-14/g++-14 or set CMAKE_CUDA_HOST_COMPILER explicitly.")
                endif()
            endif()

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
            if(UNIX AND NOT APPLE)
                if(NOT CMAKE_CUDA_FLAGS MATCHES "(^| )-U_GNU_SOURCE($| )")
                    string(APPEND CMAKE_CUDA_FLAGS " -U_GNU_SOURCE")
                    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" CACHE STRING
                        "Flags used by the CUDA compiler" FORCE)
                endif()
                execute_process(
                    COMMAND getconf GNU_LIBC_VERSION
                    OUTPUT_VARIABLE _eshkol_glibc_version_raw
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                )
                if(_eshkol_glibc_version_raw MATCHES "glibc[ ]+([0-9]+\\.[0-9]+)")
                    set(_eshkol_glibc_version "${CMAKE_MATCH_1}")
                    if(_eshkol_glibc_version VERSION_GREATER_EQUAL 2.43)
                        set(ESHKOL_CUDA_GLIBC_COMPAT_HEADER
                            "${CMAKE_BINARY_DIR}/eshkol_cuda_glibc_compat.h")
                        file(WRITE "${ESHKOL_CUDA_GLIBC_COMPAT_HEADER}" [=[
#pragma once
/*
 * CUDA/glibc compatibility pre-include.
 *
 * CUDA 13.x point releases and glibc 2.43 have disagreed about a few
 * IEC-60559 math declarations in crt/math_functions.h. Keep this header
 * intentionally narrow: it is force-included only for CUDA compilation and
 * gives us a stable CMake-controlled shim point without patching CUDA's
 * installation headers.
 */
#if defined(__CUDACC__) && defined(__GLIBC__) && defined(__GLIBC_MINOR__)
#  if (__GLIBC__ > 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 43)
#    ifndef __STDC_WANT_IEC_60559_FUNCS_EXT__
#      define __STDC_WANT_IEC_60559_FUNCS_EXT__ 1
#    endif
#  endif
#endif
]=])
                        if(NOT CMAKE_CUDA_FLAGS MATCHES "eshkol_cuda_glibc_compat\\.h")
                            string(APPEND CMAKE_CUDA_FLAGS " -include ${ESHKOL_CUDA_GLIBC_COMPAT_HEADER}")
                            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" CACHE STRING
                                "Flags used by the CUDA compiler" FORCE)
                        endif()
                        message(STATUS
                            "CUDA: glibc ${_eshkol_glibc_version} compatibility shim enabled")
                    endif()
                endif()
            endif()
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
            if(WIN32 AND _eshkol_runtime_link_arg MATCHES "^[A-Za-z0-9_.-]+\\.lib$")
                string(REGEX REPLACE "\\.lib$" "" _eshkol_runtime_link_arg "${_eshkol_runtime_link_arg}")
                set(_eshkol_runtime_link_arg "-l${_eshkol_runtime_link_arg}")
            elseif(NOT _eshkol_runtime_link_arg MATCHES "^-" AND
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

if(NOT APPLE AND NOT WIN32)
    # lib/core/crypto_primitives.c (SHA256/HMAC/secure random) is always
    # compiled into eshkol-runtime (see ESHKOL_RUNTIME_SUPPORT_SRC) — probe
    # for OpenSSL here, outside the optional ESHKOL_BUILD_AGENT_FFI block
    # below, so libcrypto is linked into every AOT/-r build unconditionally.
    # Apple uses CommonCrypto+Security (system-provided, no probe needed);
    # Windows uses BCrypt (system-provided, no probe needed). See Noesis bug
    # report #2 (2026-07-04): these symbols must not depend on the separate,
    # more-frequently-rebuilt eshkol-agent-ffi archive being complete/whole
    # at AOT-link time.
    find_package(OpenSSL)
endif()

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

    # Crypto system lib for lib/core/crypto_primitives.c — see comment above.
    if(WIN32)
        eshkol_append_host_runtime_link_arg("bcrypt")
    elseif(OpenSSL_FOUND)
        eshkol_append_host_runtime_link_arg("OpenSSL::Crypto")
    else()
        message(WARNING
            "OpenSSL (libcrypto) not found: eshkol_sha256/eshkol_hmac_sha256/"
            "eshkol_random_bytes/eshkol_random_hex (lib/core/crypto_primitives.c, "
            "always linked into eshkol-run AOT/-r binaries) will fail to LINK on "
            "this host. Install libssl-dev (Debian/Ubuntu), openssl-devel "
            "(Fedora/RHEL), or the platform equivalent.")
    endif()
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

# file(GLOB_RECURSE) is evaluated at configure time, so make newly added
# optional backend bridges explicit for incremental build trees too.
if(NOT "lib/backend/tensorcore_codegen.cpp" IN_LIST LIB_SRC)
    list(APPEND LIB_SRC lib/backend/tensorcore_codegen.cpp)
endif()

set(ESHKOL_VM_CORE_COMPONENT_SRC
  lib/backend/vm_core.c
  lib/backend/vm_run.c
  lib/backend/vm_complex.c
  lib/backend/vm_rational.c
  lib/backend/vm_bignum.c
  lib/backend/vm_dual.c
  lib/backend/vm_hyperdual.c
  lib/backend/vm_autodiff.c
  lib/backend/vm_tensor.c
  lib/backend/vm_tensor_ops.c
  lib/backend/vm_logic.c
  lib/backend/vm_inference.c
  lib/backend/vm_workspace.c
  lib/backend/vm_string.c
  lib/backend/vm_hashtable.c
  lib/backend/vm_bytevector.c
  lib/backend/vm_multivalue.c
  lib/backend/vm_error.c
  lib/backend/vm_parameter.c
  lib/backend/vm_geometric.c
  lib/backend/vm_symbolic_ad.c
)

set(ESHKOL_VM_HOSTED_COMPONENT_SRC
  lib/backend/eskb_writer.c
  lib/backend/eskb_reader.c
  lib/backend/vm_io.c
  lib/backend/vm_model_io.c
  lib/backend/vm_parallel.c
  lib/backend/vm_native.c
)

set(ESHKOL_VM_TOOLCHAIN_COMPONENT_SRC
  lib/backend/vm_parser.c
  lib/backend/vm_macro.c
  lib/backend/vm_compiler.c
  lib/backend/vm_peephole.c
)

set(ESHKOL_VM_TEST_COMPONENT_SRC
  lib/backend/vm_tests.c
)

set(ESHKOL_VM_FULL_UNITY_SRC
  lib/backend/eshkol_vm.c
)

set(ESHKOL_VM_STUB_SRC
  lib/backend/eshkol_vm_stub.c
  lib/backend/vm_autodiff.c
)

set(ESHKOL_VM_HEAP_SIZE "65536" CACHE STRING
    "Bytecode VM heap object table capacity")
set(ESHKOL_VM_STACK_SIZE "4096" CACHE STRING
    "Bytecode VM value stack slot count")
set(ESHKOL_VM_MAX_FRAMES "256" CACHE STRING
    "Bytecode VM call frame capacity")
set(ESHKOL_VM_MAX_CONSTS "1024" CACHE STRING
    "Bytecode VM constant pool capacity")
set(ESHKOL_VM_MAX_CODE "100000" CACHE STRING
    "Bytecode VM decoded instruction capacity")

foreach(_eshkol_vm_limit_name IN ITEMS
    ESHKOL_VM_HEAP_SIZE
    ESHKOL_VM_STACK_SIZE
    ESHKOL_VM_MAX_FRAMES
    ESHKOL_VM_MAX_CONSTS
    ESHKOL_VM_MAX_CODE
)
    if(NOT "${${_eshkol_vm_limit_name}}" MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR
            "${_eshkol_vm_limit_name} must be a positive integer, got "
            "'${${_eshkol_vm_limit_name}}'")
    endif()
endforeach()

function(eshkol_apply_vm_limit_definitions target_name)
    target_compile_definitions(${target_name} PRIVATE
        ESHKOL_VM_HEAP_SIZE=${ESHKOL_VM_HEAP_SIZE}
        ESHKOL_VM_STACK_SIZE=${ESHKOL_VM_STACK_SIZE}
        ESHKOL_VM_MAX_FRAMES=${ESHKOL_VM_MAX_FRAMES}
        ESHKOL_VM_MAX_CONSTS=${ESHKOL_VM_MAX_CONSTS}
        ESHKOL_VM_MAX_CODE=${ESHKOL_VM_MAX_CODE}
    )
endfunction()

set(ESHKOL_VM_UNITY_SRC)

# Native Windows REPL builds do not use the bytecode VM / ESKB compiler path.
# Building the VM currently drags in a large POSIX-only unity build, so use a
# stub there and keep the full VM everywhere else.
if(WIN32)
    list(APPEND LIB_SRC ${ESHKOL_VM_STUB_SRC})
    message(STATUS "Bytecode VM / ESKB emission: DISABLED for native Windows builds")
else()
    set(ESHKOL_VM_UNITY_SRC ${ESHKOL_VM_FULL_UNITY_SRC})
endif()

list(LENGTH ESHKOL_VM_CORE_COMPONENT_SRC ESHKOL_VM_CORE_COMPONENT_SRC_COUNT)
list(LENGTH ESHKOL_VM_HOSTED_COMPONENT_SRC ESHKOL_VM_HOSTED_COMPONENT_SRC_COUNT)
list(LENGTH ESHKOL_VM_TOOLCHAIN_COMPONENT_SRC ESHKOL_VM_TOOLCHAIN_COMPONENT_SRC_COUNT)
list(LENGTH ESHKOL_VM_TEST_COMPONENT_SRC ESHKOL_VM_TEST_COMPONENT_SRC_COUNT)
message(STATUS
  "VM source families: core=${ESHKOL_VM_CORE_COMPONENT_SRC_COUNT}, "
  "hosted=${ESHKOL_VM_HOSTED_COMPONENT_SRC_COUNT}, "
  "toolchain=${ESHKOL_VM_TOOLCHAIN_COMPONENT_SRC_COUNT}, "
  "test=${ESHKOL_VM_TEST_COMPONENT_SRC_COUNT}"
)
message(STATUS
  "VM limits: heap=${ESHKOL_VM_HEAP_SIZE}, "
  "stack=${ESHKOL_VM_STACK_SIZE}, "
  "frames=${ESHKOL_VM_MAX_FRAMES}, "
  "consts=${ESHKOL_VM_MAX_CONSTS}, "
  "code=${ESHKOL_VM_MAX_CODE}"
)

# Image I/O (native platform/system codec backends)
list(APPEND LIB_SRC lib/core/image_io.c)
set_source_files_properties(lib/core/image_io.c PROPERTIES LANGUAGE CXX)

# C FFI (stable embedding API)
list(APPEND LIB_SRC lib/ffi/eshkol_ffi.cpp)

# Exclude XLA sources from main library when XLA is disabled
if(NOT ESHKOL_XLA_ENABLED)
    list(FILTER LIB_SRC EXCLUDE REGEX "lib/backend/xla/.*")
endif()

# Get REPL sources separately
file(GLOB REPL_SRC RELATIVE ${CMAKE_SOURCE_DIR} "lib/repl/*.cpp")

add_compile_definitions(
  ESHKOL_VER="${ESHKOL_VERSION}-evolve"
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

# Runtime stratification starts here. `eshkol-runtime` is the archive linked
# into generated programs; `eshkol-static` remains the compiler/tool aggregate.
set(ESHKOL_RUNTIME_CORE_SRC
  lib/core/ad_tape_builtins.c
  lib/core/bignum.cpp
  lib/core/rational.cpp
  lib/core/runtime.cpp
  lib/core/runtime_arena_core.cpp
  lib/core/runtime_autodiff.cpp
  lib/core/runtime_bytevector.cpp
  lib/core/runtime_closure_alloc.cpp
  lib/core/runtime_closure_reflection.cpp
  lib/core/runtime_continuations.cpp
  lib/core/runtime_deep_equal.cpp
  lib/core/runtime_hash_table.cpp
  lib/core/runtime_list_helpers.cpp
  lib/core/runtime_object_alloc.cpp
  lib/core/runtime_regions.cpp
  lib/core/runtime_shared_memory.cpp
  lib/core/runtime_string.cpp
  lib/core/runtime_tagged_cons.cpp
  lib/core/runtime_taylor.c
  lib/core/runtime_tensor_alloc.cpp
  lib/core/runtime_tensor_dtype.cpp
  lib/core/runtime_tensor_quant.cpp
  lib/core/runtime_tensor_index.cpp
  lib/core/runtime_tensor_fill.cpp
  lib/core/runtime_tensor_math.cpp
)

set(ESHKOL_RUNTIME_HOSTED_SRC
  lib/backend/thread_pool.cpp
  lib/core/config.cpp
  lib/core/logger.cpp
  lib/core/platform_runtime.cpp
  lib/core/resource_limits.cpp
  lib/core/runtime_arena_diagnostics_hosted.cpp
  lib/core/runtime_arena_sync_hosted.cpp
  lib/core/runtime_display_hosted.cpp
  lib/core/runtime_errors_hosted.cpp
  lib/core/runtime_exceptions_hosted.cpp
  lib/core/runtime_exports_hosted.cpp
  lib/core/runtime_lifecycle_hosted.cpp
  lib/core/runtime_operations_hosted.cpp
  lib/core/runtime_parameters_hosted.cpp
  lib/core/runtime_reader_hosted.cpp
  lib/core/runtime_shutdown_hooks_hosted.cpp
  lib/core/runtime_signals_hosted.cpp
  lib/core/runtime_stack_hosted.cpp
  lib/core/runtime_string_ports_hosted.cpp
  lib/core/system_builtins.c
)

list(REMOVE_ITEM LIB_SRC
  ${ESHKOL_RUNTIME_CORE_SRC}
  ${ESHKOL_RUNTIME_HOSTED_SRC}
)

set(ESHKOL_NON_RUNTIME_SRC ${LIB_SRC})

set(ESHKOL_BACKEND_RUNTIME_SRC
  lib/backend/blas_backend.cpp
  lib/backend/cpu_features.cpp
  lib/backend/parallel_codegen.cpp
  lib/backend/tensor_backward.cpp
  # Shared conv2d forward kernel (eshkol_rt_conv2d) the generated code calls;
  # must live in the slim runtime archive so AOT user binaries resolve it.
  lib/backend/tensor_conv_kernel.cpp
)

set(ESHKOL_RUNTIME_SUPPORT_SRC
  lib/bridge/tensor_backward.cpp
  lib/core/crypto_primitives.c
  lib/core/dnc_api.c
  lib/core/image_io.c
  lib/core/inference.cpp
  lib/core/kb_persistence.cpp
  lib/core/sdnc_api.c
  lib/core/logic.cpp
  lib/core/logic_builtins.cpp
  lib/core/merkle.c
  lib/core/model_io.cpp
  lib/core/onnx_export.c
  lib/core/printer.cpp
  lib/core/prng.cpp
  lib/core/runtime_arena_cpp.cpp
  lib/core/runtime_frame_trace.cpp
  lib/core/symbol_intern.cpp
  lib/core/threads.c
  lib/core/workspace.cpp
  lib/quantum/quantum_rng.c
  lib/quantum/quantum_rng_wrapper.c
)

list(REMOVE_ITEM ESHKOL_NON_RUNTIME_SRC
  ${ESHKOL_BACKEND_RUNTIME_SRC}
  ${ESHKOL_RUNTIME_SUPPORT_SRC}
)

# xla_runtime.cpp defines the runtime dispatch (eshkol_xla_elementwise/transpose
# /…) that generated code calls. It must live in the slim eshkol-runtime archive
# so AOT user binaries resolve those symbols — the runtime stratification
# otherwise leaves all of lib/backend/xla/ in eshkol-static only, breaking the
# XLA-build AOT link of the .esk tests. It has no MLIR codegen dependencies.
# Move it out of NON_RUNTIME so it isn't double-compiled into eshkol-static.
set(ESHKOL_XLA_RUNTIME_SRC "")
if(ESHKOL_XLA_ENABLED)
  set(ESHKOL_XLA_RUNTIME_SRC lib/backend/xla/xla_runtime.cpp)
  list(REMOVE_ITEM ESHKOL_NON_RUNTIME_SRC ${ESHKOL_XLA_RUNTIME_SRC})
endif()

list(LENGTH ESHKOL_RUNTIME_CORE_SRC ESHKOL_RUNTIME_CORE_SRC_COUNT)
list(LENGTH ESHKOL_RUNTIME_HOSTED_SRC ESHKOL_RUNTIME_HOSTED_SRC_COUNT)
message(STATUS
  "Runtime source sets: core=${ESHKOL_RUNTIME_CORE_SRC_COUNT}, "
  "hosted=${ESHKOL_RUNTIME_HOSTED_SRC_COUNT}"
)

set(ESHKOL_BUILD_INTERFACE_INCLUDE_DIRS
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/>"
)
if(DEFINED ESHKOL_GENERATED_INCLUDE_DIR)
    list(APPEND ESHKOL_BUILD_INTERFACE_INCLUDE_DIRS
      "$<BUILD_INTERFACE:${ESHKOL_GENERATED_INCLUDE_DIR}>"
    )
endif()

function(eshkol_apply_common_compile_settings target_name)
    target_include_directories(${target_name} PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/inc/"
    )
    if(DEFINED ESHKOL_GENERATED_INCLUDE_DIR)
        target_include_directories(${target_name} PRIVATE "${ESHKOL_GENERATED_INCLUDE_DIR}")
    endif()
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:C,CXX>:${LLVM_CXXFLAGS_LIST}>
      $<$<COMPILE_LANGUAGE:C,CXX>:${ESHKOL_WARNING_FLAGS}>
      $<$<COMPILE_LANGUAGE:C,CXX>:${ESHKOL_HARDENING_FLAGS}>
    )
    if(NOT ESHKOL_USING_MSVC)
        target_compile_definitions(${target_name} PRIVATE $<$<NOT:$<CONFIG:Debug>>:_FORTIFY_SOURCE=2>)
    endif()
    target_compile_definitions(${target_name} PRIVATE ESHKOL_LLVM_BACKEND_ENABLED)
    target_compile_definitions(${target_name} PRIVATE ESHKOL_VM_LIBRARY_MODE)
    eshkol_target_llvm_includes(${target_name})
endfunction()

add_library(eshkol-runtime-core-obj OBJECT ${ESHKOL_RUNTIME_CORE_SRC})
add_library(eshkol-runtime-hosted-obj OBJECT ${ESHKOL_RUNTIME_HOSTED_SRC})
add_library(eshkol-backend-runtime-obj OBJECT ${ESHKOL_BACKEND_RUNTIME_SRC})
add_library(eshkol-runtime-support-obj OBJECT ${ESHKOL_RUNTIME_SUPPORT_SRC})
set(ESHKOL_VM_UNITY_OBJECTS)
if(ESHKOL_VM_UNITY_SRC)
    add_library(eshkol-vm-unity-obj OBJECT ${ESHKOL_VM_UNITY_SRC})
    set(ESHKOL_VM_UNITY_OBJECTS $<TARGET_OBJECTS:eshkol-vm-unity-obj>)
endif()
set(ESHKOL_GPU_RUNTIME_OBJECTS)
if(GPU_SOURCES)
    add_library(eshkol-gpu-runtime-obj OBJECT ${GPU_SOURCES})
    set(ESHKOL_GPU_RUNTIME_OBJECTS $<TARGET_OBJECTS:eshkol-gpu-runtime-obj>)
endif()
set(ESHKOL_XLA_RUNTIME_OBJECTS)
if(ESHKOL_XLA_RUNTIME_SRC)
    add_library(eshkol-xla-runtime-obj OBJECT ${ESHKOL_XLA_RUNTIME_SRC})
    set(ESHKOL_XLA_RUNTIME_OBJECTS $<TARGET_OBJECTS:eshkol-xla-runtime-obj>)
endif()

foreach(_eshkol_runtime_obj_target IN ITEMS
    eshkol-runtime-core-obj
    eshkol-runtime-hosted-obj
    eshkol-backend-runtime-obj
    eshkol-runtime-support-obj
)
    eshkol_apply_common_compile_settings(${_eshkol_runtime_obj_target})
endforeach()
if(TARGET eshkol-vm-unity-obj)
    eshkol_apply_common_compile_settings(eshkol-vm-unity-obj)
    eshkol_apply_vm_limit_definitions(eshkol-vm-unity-obj)
endif()
if(TARGET eshkol-gpu-runtime-obj)
    eshkol_apply_common_compile_settings(eshkol-gpu-runtime-obj)
    if(CUDAToolkit_FOUND)
        target_include_directories(eshkol-gpu-runtime-obj SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    endif()
endif()
if(TARGET eshkol-xla-runtime-obj)
    eshkol_apply_common_compile_settings(eshkol-xla-runtime-obj)
    eshkol_target_llvm_includes(eshkol-xla-runtime-obj)
endif()

set(ESHKOL_RUNTIME_OBJECTS
  $<TARGET_OBJECTS:eshkol-runtime-core-obj>
  $<TARGET_OBJECTS:eshkol-runtime-hosted-obj>
  $<TARGET_OBJECTS:eshkol-backend-runtime-obj>
  $<TARGET_OBJECTS:eshkol-runtime-support-obj>
  ${ESHKOL_GPU_RUNTIME_OBJECTS}
  ${ESHKOL_XLA_RUNTIME_OBJECTS}
  ${ESHKOL_VM_UNITY_OBJECTS}
)

add_library(eshkol-runtime STATIC
  ${ESHKOL_RUNTIME_OBJECTS}
)
target_include_directories(
  eshkol-runtime
  PUBLIC
    ${ESHKOL_BUILD_INTERFACE_INCLUDE_DIRS}
)

# Create static library for compiler/tools. The generated-program runtime is
# split into libeshkol-runtime.a so user binaries no longer force-load LLVM
# compiler objects.
add_library(eshkol-static STATIC
  ${ESHKOL_NON_RUNTIME_SRC}
  ${ESHKOL_RUNTIME_OBJECTS}
)
target_include_directories(
  eshkol-static
  PUBLIC
    ${ESHKOL_BUILD_INTERFACE_INCLUDE_DIRS}
)
eshkol_apply_common_compile_settings(eshkol-static)

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

# Propagate the runtime's transitive system link deps (Apple ImageIO/CoreGraphics/
# CoreFoundation for image_io.c, Windows Gdiplus/Advapi32/Ntdll, libpng/jpeg/webp,
# clang builtins) to the static/runtime libraries' PUBLIC interface. Previously
# these lived only on the eshkol-run/eshkol-repl executables, so a downstream
# consumer linking libeshkol-static.a directly (e.g. libnoesis-core) hit
# "Undefined symbols: _CGImageDestinationCreateWithURL ..." and had to re-declare
# the frameworks itself. The library should own its transitive link deps.
if(ESHKOL_EXTRA_LINK_LIBS)
    target_link_libraries(eshkol-runtime PUBLIC ${ESHKOL_EXTRA_LINK_LIBS})
    target_link_libraries(eshkol-static PUBLIC ${ESHKOL_EXTRA_LINK_LIBS})
endif()

# Link BLAS libraries if available
if(ESHKOL_BLAS_ENABLED AND BLAS_LIBRARIES)
    target_link_libraries(eshkol-runtime PUBLIC ${BLAS_LIBRARIES})
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

if(GPU_FOUND AND GPU_LIBRARIES)
    target_link_libraries(eshkol-runtime PUBLIC ${GPU_LIBRARIES})
    target_link_libraries(eshkol-static PUBLIC ${GPU_LIBRARIES})
    message(STATUS "Linking GPU libraries: ${GPU_LIBRARIES}")
endif()
# Metal shader embedding: gpu_memory.mm includes the generated .inc file
if(TARGET metal_sf64_shader AND TARGET eshkol-gpu-runtime-obj)
    add_dependencies(eshkol-gpu-runtime-obj metal_sf64_shader)
    target_include_directories(eshkol-gpu-runtime-obj PRIVATE ${CMAKE_BINARY_DIR})
endif()

# Crypto system libs for lib/core/crypto_primitives.c (SHA256/HMAC/secure
# random — see ESHKOL_RUNTIME_SUPPORT_SRC above). Always compiled into
# eshkol-runtime/eshkol-static, so these link unconditionally here,
# independent of ESHKOL_BUILD_AGENT_FFI (which historically gated the same
# symbols via a separate, optional archive — Noesis bug report #2,
# 2026-07-04). This covers the `eshkol-run`/`eshkol-repl` tool binaries and
# any other CMake target linking eshkol-runtime/eshkol-static directly; the
# AOT-compiled-user-program link path (raw clang++ invocation in
# exe/eshkol-run.cpp) is covered separately — see append_host_runtime_link_args()
# there for non-Apple, and the hardcoded Apple framework list for Apple.
if(APPLE)
    target_link_libraries(eshkol-runtime PUBLIC "-framework Security")
    target_link_libraries(eshkol-static PUBLIC "-framework Security")
elseif(WIN32)
    target_link_libraries(eshkol-runtime PUBLIC bcrypt)
    target_link_libraries(eshkol-static PUBLIC bcrypt)
elseif(OpenSSL_FOUND)
    target_link_libraries(eshkol-runtime PUBLIC OpenSSL::Crypto)
    target_link_libraries(eshkol-static PUBLIC OpenSSL::Crypto)
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
  eshkol_target_llvm_includes(${exename})
  
  # Add library directories for executables
  if(LLVM_LIBRARY_DIRS)
      target_link_directories(${exename} PRIVATE ${LLVM_LIBRARY_DIRS})
  endif()
  
  target_link_libraries(${exename} PUBLIC eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
  if(exename STREQUAL "eshkol-run")
      add_dependencies(${exename} eshkol-runtime)
  endif()
endforeach ()

# ===== Pre-compiled Standard Library =====
# Compile stdlib.esk to stdlib.o/stdlib.bc for faster loading of user programs
set(STDLIB_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/lib/stdlib.esk")
set(STDLIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/stdlib.o")
set(STDLIB_BITCODE "${CMAKE_CURRENT_BINARY_DIR}/stdlib.bc")
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
  BYPRODUCTS ${STDLIB_BITCODE}
  COMMAND ${CMAKE_COMMAND} -E env "ESHKOL_PATH=${CMAKE_CURRENT_SOURCE_DIR}/lib"
          $<TARGET_FILE:eshkol-run> --shared-lib -o "${CMAKE_CURRENT_BINARY_DIR}/stdlib" "${STDLIB_SOURCE}"
  DEPENDS eshkol-run ${STDLIB_SOURCE} ${STDLIB_TRANSITIVE_SOURCES}
  WORKING_DIRECTORY ${STDLIB_WORKING_DIRECTORY}
  COMMENT "Compiling standard library to ${STDLIB_OUTPUT}"
  VERBATIM
)

add_custom_target(stdlib ALL DEPENDS ${STDLIB_OUTPUT})

# Install stdlib artifacts to lib directory.  The REPL prefers stdlib.bc so
# stdlib and later REPL batches share the same ORC IR compilation pipeline;
# stdlib.o remains available for legacy/load-time fallback paths.
install(FILES ${STDLIB_OUTPUT} ${STDLIB_BITCODE} DESTINATION ${CMAKE_INSTALL_LIBDIR}/eshkol)

message(STATUS "Standard library will be pre-compiled to: ${STDLIB_OUTPUT} and ${STDLIB_BITCODE}")

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

  eshkol_target_llvm_includes(eshkol-repl-lib)

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

    eshkol_target_llvm_includes(eshkol-repl)

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
    elseif(WIN32 AND ESHKOL_USING_MSVC)
      # ClangCL/MSVC links the needed runtime objects directly via
      # eshkol-repl-lib's references and explicit JIT symbol registration.
    elseif(WIN32)
      # MSYS2/MinGW GNU driver: whole-archive the runtime so every object
      # (incl. JIT-only-referenced runtime helpers) is pulled into the exe,
      # matching the macOS -force_load / Linux --whole-archive behaviour.
      target_link_options(eshkol-repl PRIVATE
        "-Wl,--whole-archive" "$<TARGET_FILE:eshkol-static>" "-Wl,--no-whole-archive")
    else()
      target_link_options(eshkol-repl PRIVATE
        "-Wl,--whole-archive" "$<TARGET_FILE:eshkol-static>" "-Wl,--no-whole-archive")
    endif()

    # Export the runtime symbols so the in-process JIT can resolve host symbols
    # via DynamicLibrarySearchGenerator::GetForCurrentProcess.
    if(APPLE)
      target_link_options(eshkol-repl PRIVATE "-Wl,-export_dynamic")
    elseif(WIN32 AND ESHKOL_USING_MSVC)
      # No export-dynamic equivalent is required for the ClangCL/MSVC REPL path.
    elseif(WIN32)
      # PE/COFF: export only the bounded Eshkol runtime namespace via a
      # generated .def (--export-all-symbols overflows the 65535 PE export
      # limit because the exe statically links all of LLVM).
      eshkol_windows_export_runtime(eshkol-repl)
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
  elseif(WIN32 AND ESHKOL_USING_MSVC)
    # ClangCL/MSVC uses the REPL/JIT symbol registration path and does not
    # accept GNU ld options here.
  elseif(WIN32)
    # MSYS2/MinGW GNU driver: mirror the macOS/Linux path so the in-process
    # JIT (used by `-r` and -e/--eval) can resolve every host runtime symbol.
    # whole-archive pulls in the full runtime closure (incl. eval_bridge
    # overrides + JIT-only helpers); the generated .def exports exactly the
    # Eshkol runtime namespace so GetForCurrentProcess resolves them. PE does
    # not export symbols by default — see cmake/gen_runtime_def.cmake.
    target_link_options(eshkol-run PRIVATE
      "-Wl,--whole-archive"
      "$<TARGET_FILE:eshkol-static>"
      "$<TARGET_FILE:eshkol-repl-lib>"
      "-Wl,--no-whole-archive")
    eshkol_windows_export_runtime(eshkol-run)
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

    if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/pkg/subprocess_timeout_test.cpp")
        add_executable(eshkol-pkg-subprocess-timeout-test tests/pkg/subprocess_timeout_test.cpp)
        target_compile_features(eshkol-pkg-subprocess-timeout-test PRIVATE cxx_std_17)
        target_include_directories(eshkol-pkg-subprocess-timeout-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
        add_test(
            NAME eshkol-pkg-subprocess-timeout
            COMMAND eshkol-pkg-subprocess-timeout-test
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

    eshkol_target_llvm_includes(eshkol-lsp)

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
    eshkol_target_llvm_includes(model_io_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(model_io_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(model_io_test PRIVATE eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    add_test(NAME model_io_test COMMAND model_io_test)
endif()

if(ESHKOL_BUILD_TESTS AND TARGET eshkol-run)
    add_test(
        NAME tensor_integer_vector_literal_runtime_smoke
        COMMAND $<TARGET_FILE:eshkol-run> -e "(display (tensor 1 2 3))"
    )
    set_tests_properties(tensor_integer_vector_literal_runtime_smoke
        PROPERTIES PASS_REGULAR_EXPRESSION "#\\(1 2 3\\)")
    add_test(
        NAME tensor_integer_shaped_literal_runtime_smoke
        COMMAND $<TARGET_FILE:eshkol-run> -e "(display (tensor 2 2 1 2 3 4))"
    )
    set_tests_properties(tensor_integer_shaped_literal_runtime_smoke
        PROPERTIES PASS_REGULAR_EXPRESSION "#\\(\\(1 2\\) \\(3 4\\)\\)")
    add_test(
        NAME autodiff_tensor_reductions_runtime_smoke
        COMMAND $<TARGET_FILE:eshkol-run>
                -r "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/autodiff_tensor_test.esk"
                -L"${CMAKE_CURRENT_BINARY_DIR}"
    )
    set_tests_properties(autodiff_tensor_reductions_runtime_smoke
        PROPERTIES
            PASS_REGULAR_EXPRESSION "PASS: scaled-dot-attention K/V gradients match closed forms"
            FAIL_REGULAR_EXPRESSION "FAIL:")
    add_test(
        NAME ad_structural_int_guard_runtime_smoke
        COMMAND $<TARGET_FILE:eshkol-run>
                -r "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/ad_structural_int_guard_test.esk"
                -L"${CMAKE_CURRENT_BINARY_DIR}"
    )
    set_tests_properties(ad_structural_int_guard_runtime_smoke
        PROPERTIES
            TIMEOUT 300
            PASS_REGULAR_EXPRESSION "PASS: AD dual tensor reduction axis raises catchable type error"
            FAIL_REGULAR_EXPRESSION "FAIL:|Attempted to get int64 from non-int-storage cell|LLVM module verification failed|fatal signal")
    add_test(
        NAME guw_multivariate_runtime_smoke
        COMMAND $<TARGET_FILE:eshkol-run>
                -r "${CMAKE_CURRENT_SOURCE_DIR}/tests/ad/guw_multivariate_test.esk"
                -L"${CMAKE_CURRENT_BINARY_DIR}"
    )
    set_tests_properties(guw_multivariate_runtime_smoke
        PROPERTIES
            TIMEOUT 300
            PASS_REGULAR_EXPRESSION "PASS: GUW multivariate"
            FAIL_REGULAR_EXPRESSION "FAIL:|LLVM module verification failed|fatal signal")
    add_test(
        NAME sparse_tensors_runtime_smoke
        COMMAND $<TARGET_FILE:eshkol-run>
                -r "${CMAKE_CURRENT_SOURCE_DIR}/tests/ad/sparse_tensors_test.esk"
                -L"${CMAKE_CURRENT_BINARY_DIR}"
    )
    set_tests_properties(sparse_tensors_runtime_smoke
        PROPERTIES
            TIMEOUT 300
            PASS_REGULAR_EXPRESSION "PASS: sparse tensors"
            FAIL_REGULAR_EXPRESSION "FAIL:|LLVM module verification failed|fatal signal")
    add_test(
        NAME reverse_over_taylor_runtime_smoke
        COMMAND $<TARGET_FILE:eshkol-run>
                -r "${CMAKE_CURRENT_SOURCE_DIR}/tests/ad/reverse_over_taylor_test.esk"
                -L"${CMAKE_CURRENT_BINARY_DIR}"
    )
    set_tests_properties(reverse_over_taylor_runtime_smoke
        PROPERTIES
            TIMEOUT 300
            PASS_REGULAR_EXPRESSION "PASS: reverse-over-Taylor"
            FAIL_REGULAR_EXPRESSION "FAIL:|LLVM module verification failed|fatal signal")
    add_test(
        NAME checkpointed_reverse_runtime_smoke
        COMMAND $<TARGET_FILE:eshkol-run>
                -r "${CMAKE_CURRENT_SOURCE_DIR}/tests/ad/checkpointed_reverse_test.esk"
                -L"${CMAKE_CURRENT_BINARY_DIR}"
    )
    set_tests_properties(checkpointed_reverse_runtime_smoke
        PROPERTIES
            TIMEOUT 300
            PASS_REGULAR_EXPRESSION "PASS: checkpointed_reverse_test"
            FAIL_REGULAR_EXPRESSION "FAIL:|LLVM module verification failed|fatal signal")
    add_test(
        NAME diff_control_flow_runtime_smoke
        COMMAND $<TARGET_FILE:eshkol-run>
                -r "${CMAKE_CURRENT_SOURCE_DIR}/tests/ad/diff_control_flow_test.esk"
                -L"${CMAKE_CURRENT_BINARY_DIR}"
    )
    set_tests_properties(diff_control_flow_runtime_smoke
        PROPERTIES
            TIMEOUT 300
            PASS_REGULAR_EXPRESSION "PASS: diff-control-flow"
            FAIL_REGULAR_EXPRESSION "FAIL:|LLVM module verification failed|fatal signal")
    add_test(
        NAME tensor_tower_runtime_smoke
        COMMAND $<TARGET_FILE:eshkol-run>
                -r "${CMAKE_CURRENT_SOURCE_DIR}/tests/ad/tensor_tower_test.esk"
                -L"${CMAKE_CURRENT_BINARY_DIR}"
    )
    set_tests_properties(tensor_tower_runtime_smoke
        PROPERTIES
            TIMEOUT 300
            PASS_REGULAR_EXPRESSION "PASS: tensor_tower_test"
            FAIL_REGULAR_EXPRESSION "FAIL:|LLVM module verification failed|fatal signal")
    add_test(
        NAME taylor_numerics_runtime_smoke
        COMMAND $<TARGET_FILE:eshkol-run>
                -r "${CMAKE_CURRENT_SOURCE_DIR}/tests/ad/taylor_numerics_test.esk"
                -L"${CMAKE_CURRENT_BINARY_DIR}"
    )
    set_tests_properties(taylor_numerics_runtime_smoke
        PROPERTIES
            TIMEOUT 300
            PASS_REGULAR_EXPRESSION "taylor-numerics tests: [0-9]+ passed, 0 failed"
            FAIL_REGULAR_EXPRESSION "FAIL |LLVM module verification failed|fatal signal")
    add_test(
        NAME taylor_models_runtime_smoke
        COMMAND $<TARGET_FILE:eshkol-run>
                -r "${CMAKE_CURRENT_SOURCE_DIR}/tests/ad/taylor_models_test.esk"
                -L"${CMAKE_CURRENT_BINARY_DIR}"
    )
    set_tests_properties(taylor_models_runtime_smoke
        PROPERTIES
            TIMEOUT 300
            PASS_REGULAR_EXPRESSION "PASS: taylor_models_test"
            FAIL_REGULAR_EXPRESSION "FAIL:|LLVM module verification failed|fatal signal")
    add_test(
        NAME empty_collection_runtime_smoke
        COMMAND $<TARGET_FILE:eshkol-run>
                -r "${CMAKE_CURRENT_SOURCE_DIR}/tests/collections/empty_collection_test.esk"
                -L"${CMAKE_CURRENT_BINARY_DIR}"
    )
    set_tests_properties(empty_collection_runtime_smoke
        PROPERTIES
            TIMEOUT 300
            PASS_REGULAR_EXPRESSION "TEST: Empty collection edge cases"
            FAIL_REGULAR_EXPRESSION "FAIL:|LLVM module verification failed|fatal signal")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/v1_3_edge_cases/v1_3_smoke_features_test.esk")
        add_test(
            NAME v1_3_smoke_features_runtime_smoke
            COMMAND $<TARGET_FILE:eshkol-run>
                    -r "${CMAKE_CURRENT_SOURCE_DIR}/tests/v1_3_edge_cases/v1_3_smoke_features_test.esk"
                    -L"${CMAKE_CURRENT_BINARY_DIR}"
        )
        set_tests_properties(v1_3_smoke_features_runtime_smoke
            PROPERTIES
                TIMEOUT 300
                PASS_REGULAR_EXPRESSION "PASS: v1_3_smoke_features_test"
                FAIL_REGULAR_EXPRESSION "FAIL:|LLVM module verification failed|fatal signal")
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/v1_3_edge_cases/ad_input2_test.esk")
        add_test(
            NAME v1_3_ad_input2_runtime_smoke
            COMMAND $<TARGET_FILE:eshkol-run>
                    -r "${CMAKE_CURRENT_SOURCE_DIR}/tests/v1_3_edge_cases/ad_input2_test.esk"
                    -L"${CMAKE_CURRENT_BINARY_DIR}"
        )
        set_tests_properties(v1_3_ad_input2_runtime_smoke
            PROPERTIES
                TIMEOUT 300
                PASS_REGULAR_EXPRESSION "PASS: ad_input2_test"
                FAIL_REGULAR_EXPRESSION "FAIL:|LLVM module verification failed|fatal signal")
    endif()
    if(NOT WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/v1_3_edge_cases/jit_cache_test.sh")
        add_test(
            NAME v1_3_jit_cache_runtime_smoke
            COMMAND bash
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/v1_3_edge_cases/jit_cache_test.sh"
                    $<TARGET_FILE:eshkol-run>
        )
        set_tests_properties(v1_3_jit_cache_runtime_smoke
            PROPERTIES
                TIMEOUT 120
                PASS_REGULAR_EXPRESSION "PASS: jit_cache_test"
                FAIL_REGULAR_EXPRESSION "FAIL:")
    endif()
    if(NOT WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/static_link_getenv_capability_test.sh")
        add_test(
            NAME static_link_getenv_capability_test
            COMMAND bash
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/static_link_getenv_capability_test.sh"
                    $<TARGET_FILE:eshkol-run>
                    "${CMAKE_CURRENT_SOURCE_DIR}"
                    "${CMAKE_CURRENT_BINARY_DIR}"
        )
        set_tests_properties(static_link_getenv_capability_test
            PROPERTIES
                TIMEOUT 300
                PASS_REGULAR_EXPRESSION "PASS: static_link_getenv_capability_test"
                FAIL_REGULAR_EXPRESSION "FAIL:")
    endif()
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/runtime_shutdown_hooks_test.cpp")
    add_executable(runtime_shutdown_hooks_test tests/core/runtime_shutdown_hooks_test.cpp)
    target_compile_features(runtime_shutdown_hooks_test PRIVATE cxx_std_17)
    target_include_directories(runtime_shutdown_hooks_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(runtime_shutdown_hooks_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(runtime_shutdown_hooks_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(runtime_shutdown_hooks_test PRIVATE eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    add_test(NAME runtime_shutdown_hooks_test COMMAND runtime_shutdown_hooks_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/resource_limits_test.cpp")
    add_executable(resource_limits_test tests/core/resource_limits_test.cpp)
    target_compile_features(resource_limits_test PRIVATE cxx_std_17)
    target_include_directories(resource_limits_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(resource_limits_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(resource_limits_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(resource_limits_test PRIVATE eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    add_test(NAME resource_limits_test COMMAND resource_limits_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/runtime_signal_handlers_test.cpp")
    add_executable(runtime_signal_handlers_test tests/core/runtime_signal_handlers_test.cpp)
    target_compile_features(runtime_signal_handlers_test PRIVATE cxx_std_17)
    target_include_directories(runtime_signal_handlers_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(runtime_signal_handlers_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(runtime_signal_handlers_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(runtime_signal_handlers_test PRIVATE eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    add_test(NAME runtime_signal_handlers_test COMMAND runtime_signal_handlers_test)
endif()

# ESH-0216: SIGTERM-during-shutdown teardown-ordering race — a shutdown hook
# frees shared arena state while the global thread pool is still busy. A
# single run is inherently racy; tests/core/run_shutdown_teardown_race_cycles.sh
# runs this binary 50x (the mandated repro-survival gate) and is invoked
# directly rather than through ctest, matching the other scripts/run_*.sh /
# tests/**/*.sh regression drivers in this repo.
if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/runtime_shutdown_teardown_race_test.cpp")
    add_executable(runtime_shutdown_teardown_race_test tests/core/runtime_shutdown_teardown_race_test.cpp)
    target_compile_features(runtime_shutdown_teardown_race_test PRIVATE cxx_std_17)
    target_include_directories(runtime_shutdown_teardown_race_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(runtime_shutdown_teardown_race_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(runtime_shutdown_teardown_race_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(runtime_shutdown_teardown_race_test PRIVATE eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    add_test(NAME runtime_shutdown_teardown_race_test COMMAND runtime_shutdown_teardown_race_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/runtime_shared_memory_test.cpp")
    add_executable(runtime_shared_memory_test tests/core/runtime_shared_memory_test.cpp)
    target_compile_features(runtime_shared_memory_test PRIVATE cxx_std_17)
    target_include_directories(runtime_shared_memory_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(runtime_shared_memory_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(runtime_shared_memory_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(runtime_shared_memory_test PRIVATE eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    add_test(NAME runtime_shared_memory_test COMMAND runtime_shared_memory_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/runtime_closure_alloc_test.cpp")
    add_executable(runtime_closure_alloc_test tests/core/runtime_closure_alloc_test.cpp)
    target_compile_features(runtime_closure_alloc_test PRIVATE cxx_std_17)
    target_include_directories(runtime_closure_alloc_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(runtime_closure_alloc_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(runtime_closure_alloc_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(runtime_closure_alloc_test PRIVATE eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    add_test(NAME runtime_closure_alloc_test COMMAND runtime_closure_alloc_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/runtime_regions_test.cpp")
    add_executable(runtime_regions_test tests/core/runtime_regions_test.cpp)
    target_compile_features(runtime_regions_test PRIVATE cxx_std_17)
    target_include_directories(runtime_regions_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(runtime_regions_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(runtime_regions_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(runtime_regions_test PRIVATE eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    add_test(NAME runtime_regions_test COMMAND runtime_regions_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/runtime_deep_equal_test.cpp")
    add_executable(runtime_deep_equal_test tests/core/runtime_deep_equal_test.cpp)
    target_compile_features(runtime_deep_equal_test PRIVATE cxx_std_17)
    target_include_directories(runtime_deep_equal_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(runtime_deep_equal_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(runtime_deep_equal_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(runtime_deep_equal_test PRIVATE eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    add_test(NAME runtime_deep_equal_test COMMAND runtime_deep_equal_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/runtime_arena_core_test.cpp")
    add_executable(runtime_arena_core_test tests/core/runtime_arena_core_test.cpp)
    target_compile_features(runtime_arena_core_test PRIVATE cxx_std_17)
    target_include_directories(runtime_arena_core_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(runtime_arena_core_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(runtime_arena_core_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(runtime_arena_core_test PRIVATE eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    add_test(NAME runtime_arena_core_test COMMAND runtime_arena_core_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/runtime_arena_cpp_test.cpp")
    add_executable(runtime_arena_cpp_test tests/core/runtime_arena_cpp_test.cpp)
    target_compile_features(runtime_arena_cpp_test PRIVATE cxx_std_17)
    target_include_directories(runtime_arena_cpp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(runtime_arena_cpp_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(runtime_arena_cpp_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(runtime_arena_cpp_test PRIVATE eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    add_test(NAME runtime_arena_cpp_test COMMAND runtime_arena_cpp_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/runtime_object_alloc_test.cpp")
    add_executable(runtime_object_alloc_test tests/core/runtime_object_alloc_test.cpp)
    target_compile_features(runtime_object_alloc_test PRIVATE cxx_std_17)
    target_include_directories(runtime_object_alloc_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(runtime_object_alloc_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(runtime_object_alloc_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(runtime_object_alloc_test PRIVATE eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    add_test(NAME runtime_object_alloc_test COMMAND runtime_object_alloc_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/runtime_tagged_cons_test.cpp")
    add_executable(runtime_tagged_cons_test tests/core/runtime_tagged_cons_test.cpp)
    target_compile_features(runtime_tagged_cons_test PRIVATE cxx_std_17)
    target_include_directories(runtime_tagged_cons_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(runtime_tagged_cons_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(runtime_tagged_cons_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(runtime_tagged_cons_test PRIVATE eshkol-static ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    add_test(NAME runtime_tagged_cons_test COMMAND runtime_tagged_cons_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/execution_profile_test.cpp")
    add_executable(execution_profile_test tests/toolchain/execution_profile_test.cpp)
    target_compile_features(execution_profile_test PRIVATE cxx_std_17)
    target_include_directories(execution_profile_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/inc")
    eshkol_target_llvm_includes(execution_profile_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(execution_profile_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(execution_profile_test PRIVATE
        eshkol-static
        ${ESHKOL_EXTRA_LINK_LIBS}
        ${LLVM_LIBS_LIST}
        ${LLVM_SYSTEM_LIBS_LIST}
    )
    add_test(NAME execution_profile_test COMMAND execution_profile_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/eshkol_run_profile_cli_test.cpp")
    add_executable(eshkol_run_profile_cli_test tests/toolchain/eshkol_run_profile_cli_test.cpp)
    target_compile_features(eshkol_run_profile_cli_test PRIVATE cxx_std_17)
    add_dependencies(eshkol_run_profile_cli_test eshkol-run)
    add_test(
        NAME eshkol_run_profile_cli_test
        COMMAND eshkol_run_profile_cli_test $<TARGET_FILE:eshkol-run>
    )
endif()

if(NOT WIN32 AND ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/freestanding_object_smoke_test.cpp")
    get_filename_component(ESHKOL_LLVM_BIN_DIR "${LLVM_CONFIG_EXECUTABLE}" DIRECTORY)
    find_program(ESHKOL_LLVM_DIS_EXECUTABLE
        NAMES llvm-dis
        HINTS "${ESHKOL_LLVM_BIN_DIR}"
        NO_DEFAULT_PATH)
    if(NOT ESHKOL_LLVM_DIS_EXECUTABLE)
        find_program(ESHKOL_LLVM_DIS_EXECUTABLE NAMES llvm-dis)
    endif()

    add_executable(freestanding_object_smoke_test tests/toolchain/freestanding_object_smoke_test.cpp)
    target_compile_features(freestanding_object_smoke_test PRIVATE cxx_std_17)
    add_dependencies(freestanding_object_smoke_test eshkol-run)
    if(ESHKOL_LLVM_DIS_EXECUTABLE)
        add_test(
            NAME freestanding_object_smoke_test
            COMMAND freestanding_object_smoke_test $<TARGET_FILE:eshkol-run> "${ESHKOL_LLVM_DIS_EXECUTABLE}"
        )
    else()
        add_test(
            NAME freestanding_object_smoke_test
            COMMAND freestanding_object_smoke_test $<TARGET_FILE:eshkol-run>
        )
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/freestanding_lowlevel_object_test.cpp")
        add_executable(freestanding_lowlevel_object_test tests/toolchain/freestanding_lowlevel_object_test.cpp)
        target_compile_features(freestanding_lowlevel_object_test PRIVATE cxx_std_17)
        add_dependencies(freestanding_lowlevel_object_test eshkol-run)
        if(ESHKOL_LLVM_DIS_EXECUTABLE)
            add_test(
                NAME freestanding_lowlevel_object_test
                COMMAND freestanding_lowlevel_object_test $<TARGET_FILE:eshkol-run> "${ESHKOL_LLVM_DIS_EXECUTABLE}"
            )
        else()
            add_test(
                NAME freestanding_lowlevel_object_test
                COMMAND freestanding_lowlevel_object_test $<TARGET_FILE:eshkol-run>
            )
        endif()
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/freestanding_elf_link_smoke_test.cpp")
        add_executable(freestanding_elf_link_smoke_test tests/toolchain/freestanding_elf_link_smoke_test.cpp)
        target_compile_features(freestanding_elf_link_smoke_test PRIVATE cxx_std_17)
        add_dependencies(freestanding_elf_link_smoke_test eshkol-run)
        add_test(
            NAME freestanding_elf_link_smoke_test
            COMMAND freestanding_elf_link_smoke_test $<TARGET_FILE:eshkol-run>
        )
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/freestanding_reject_hosted_builtin_test.cpp")
        add_executable(freestanding_reject_hosted_builtin_test tests/toolchain/freestanding_reject_hosted_builtin_test.cpp)
        target_compile_features(freestanding_reject_hosted_builtin_test PRIVATE cxx_std_17)
        add_dependencies(freestanding_reject_hosted_builtin_test eshkol-run)
        add_test(
            NAME freestanding_reject_hosted_builtin_test
            COMMAND freestanding_reject_hosted_builtin_test $<TARGET_FILE:eshkol-run>
        )
    endif()
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/machine_integer_types_test.cpp")
    add_executable(machine_integer_types_test tests/toolchain/machine_integer_types_test.cpp)
    target_compile_features(machine_integer_types_test PRIVATE cxx_std_17)
    target_include_directories(machine_integer_types_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(machine_integer_types_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(machine_integer_types_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(machine_integer_types_test PRIVATE
        eshkol-static
        ${ESHKOL_EXTRA_LINK_LIBS}
        ${LLVM_LIBS_LIST}
        ${LLVM_SYSTEM_LIBS_LIST}
    )
    add_test(NAME machine_integer_types_test COMMAND machine_integer_types_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/pointer_type_surface_test.cpp")
    add_executable(pointer_type_surface_test tests/toolchain/pointer_type_surface_test.cpp)
    target_compile_features(pointer_type_surface_test PRIVATE cxx_std_17)
    target_compile_definitions(pointer_type_surface_test PRIVATE ESHKOL_LLVM_BACKEND_ENABLED)
    target_include_directories(pointer_type_surface_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(pointer_type_surface_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(pointer_type_surface_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(pointer_type_surface_test PRIVATE
        eshkol-static
        ${ESHKOL_EXTRA_LINK_LIBS}
        ${LLVM_LIBS_LIST}
        ${LLVM_SYSTEM_LIBS_LIST}
    )
    add_test(NAME pointer_type_surface_test COMMAND pointer_type_surface_test)
    if(TARGET eshkol-run)
        add_test(
            NAME pointer_conversion_direct_runtime_smoke
            COMMAND $<TARGET_FILE:eshkol-run> -e "(display (ptr->usize (usize->ptr 4096)))"
        )
        set_tests_properties(pointer_conversion_direct_runtime_smoke
            PROPERTIES PASS_REGULAR_EXPRESSION "4096")
        add_test(
            NAME pointer_conversion_define_runtime_smoke
            COMMAND $<TARGET_FILE:eshkol-run> -e "(begin (define p (usize->ptr 4096)) (display (ptr->usize p)))"
        )
        set_tests_properties(pointer_conversion_define_runtime_smoke
            PROPERTIES PASS_REGULAR_EXPRESSION "4096")
        add_test(
            NAME pointer_conversion_capture_runtime_smoke
            COMMAND $<TARGET_FILE:eshkol-run> -e "(display ((lambda (base) (let ((reader (lambda () (ptr->usize base)))) (reader))) (usize->ptr 4096)))"
        )
        set_tests_properties(pointer_conversion_capture_runtime_smoke
            PROPERTIES PASS_REGULAR_EXPRESSION "4096")
    endif()
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/fence_ops_test.cpp")
    add_executable(fence_ops_test tests/toolchain/fence_ops_test.cpp)
    target_compile_features(fence_ops_test PRIVATE cxx_std_17)
    target_compile_definitions(fence_ops_test PRIVATE ESHKOL_LLVM_BACKEND_ENABLED)
    target_include_directories(fence_ops_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(fence_ops_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(fence_ops_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(fence_ops_test PRIVATE
        eshkol-static
        ${ESHKOL_EXTRA_LINK_LIBS}
        ${LLVM_LIBS_LIST}
        ${LLVM_SYSTEM_LIBS_LIST}
    )
    add_test(NAME fence_ops_test COMMAND fence_ops_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/volatile_ops_test.cpp")
    add_executable(volatile_ops_test tests/toolchain/volatile_ops_test.cpp)
    target_compile_features(volatile_ops_test PRIVATE cxx_std_17)
    target_compile_definitions(volatile_ops_test PRIVATE ESHKOL_LLVM_BACKEND_ENABLED)
    target_include_directories(volatile_ops_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(volatile_ops_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(volatile_ops_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(volatile_ops_test PRIVATE
        eshkol-static
        ${ESHKOL_EXTRA_LINK_LIBS}
        ${LLVM_LIBS_LIST}
        ${LLVM_SYSTEM_LIBS_LIST}
    )
    add_test(NAME volatile_ops_test COMMAND volatile_ops_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/atomic_ops_test.cpp")
    add_executable(atomic_ops_test tests/toolchain/atomic_ops_test.cpp)
    target_compile_features(atomic_ops_test PRIVATE cxx_std_17)
    target_compile_definitions(atomic_ops_test PRIVATE ESHKOL_LLVM_BACKEND_ENABLED)
    target_include_directories(atomic_ops_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(atomic_ops_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(atomic_ops_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(atomic_ops_test PRIVATE
        eshkol-static
        ${ESHKOL_EXTRA_LINK_LIBS}
        ${LLVM_LIBS_LIST}
        ${LLVM_SYSTEM_LIBS_LIST}
    )
    add_test(NAME atomic_ops_test COMMAND atomic_ops_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/target_intrinsic_test.cpp")
    add_executable(target_intrinsic_test tests/toolchain/target_intrinsic_test.cpp)
    target_compile_features(target_intrinsic_test PRIVATE cxx_std_17)
    target_compile_definitions(target_intrinsic_test PRIVATE ESHKOL_LLVM_BACKEND_ENABLED)
    target_include_directories(target_intrinsic_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(target_intrinsic_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(target_intrinsic_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(target_intrinsic_test PRIVATE
        eshkol-static
        ${ESHKOL_EXTRA_LINK_LIBS}
        ${LLVM_LIBS_LIST}
        ${LLVM_SYSTEM_LIBS_LIST}
    )
    add_test(NAME target_intrinsic_test COMMAND target_intrinsic_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/decl_attribute_test.cpp")
    add_executable(decl_attribute_test tests/toolchain/decl_attribute_test.cpp)
    target_compile_features(decl_attribute_test PRIVATE cxx_std_17)
    target_compile_definitions(decl_attribute_test PRIVATE ESHKOL_LLVM_BACKEND_ENABLED)
    target_include_directories(decl_attribute_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(decl_attribute_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(decl_attribute_test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(decl_attribute_test PRIVATE
        eshkol-static
        ${ESHKOL_EXTRA_LINK_LIBS}
        ${LLVM_LIBS_LIST}
        ${LLVM_SYSTEM_LIBS_LIST}
    )
    add_test(NAME decl_attribute_test COMMAND decl_attribute_test)
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/runtime_core_boundary_test.cpp")
    add_executable(runtime_core_boundary_test tests/toolchain/runtime_core_boundary_test.cpp)
    target_compile_features(runtime_core_boundary_test PRIVATE cxx_std_17)
    add_test(
        NAME runtime_core_boundary_test
        COMMAND runtime_core_boundary_test "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/vm_source_boundary_test.cpp")
    add_executable(vm_source_boundary_test tests/toolchain/vm_source_boundary_test.cpp)
    target_compile_features(vm_source_boundary_test PRIVATE cxx_std_17)
    add_test(
        NAME vm_source_boundary_test
        COMMAND vm_source_boundary_test "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/vm_limits_surface_test.cpp")
    add_executable(vm_limits_surface_test tests/toolchain/vm_limits_surface_test.cpp)
    target_compile_features(vm_limits_surface_test PRIVATE cxx_std_17)
    add_test(
        NAME vm_limits_surface_test
        COMMAND vm_limits_surface_test "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/forward_ref_scan_boundary_test.cpp")
    add_executable(forward_ref_scan_boundary_test
        tests/toolchain/forward_ref_scan_boundary_test.cpp)
    target_compile_features(forward_ref_scan_boundary_test PRIVATE cxx_std_17)
    target_include_directories(forward_ref_scan_boundary_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    eshkol_target_llvm_includes(forward_ref_scan_boundary_test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(forward_ref_scan_boundary_test PRIVATE
            ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(forward_ref_scan_boundary_test PRIVATE
        eshkol-static
        ${ESHKOL_EXTRA_LINK_LIBS}
        ${LLVM_LIBS_LIST}
        ${LLVM_SYSTEM_LIBS_LIST}
    )
    add_test(
        NAME forward_ref_scan_boundary_test
        COMMAND forward_ref_scan_boundary_test "${CMAKE_CURRENT_BINARY_DIR}"
    )
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

if(ESHKOL_BUILD_TESTS AND NOT WIN32)
    add_executable(eshkol-vm-standalone-test lib/backend/eshkol_vm.c)
    target_include_directories(eshkol-vm-standalone-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${CMAKE_CURRENT_BINARY_DIR}/generated)
    eshkol_apply_vm_limit_definitions(eshkol-vm-standalone-test)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(eshkol-vm-standalone-test PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(eshkol-vm-standalone-test PRIVATE
        eshkol-static
        ${ESHKOL_EXTRA_LINK_LIBS}
        ${LLVM_LIBS_LIST}
        ${LLVM_SYSTEM_LIBS_LIST}
    )
    add_test(NAME eshkol_vm_standalone_smoke
        COMMAND eshkol-vm-standalone-test)
    set_tests_properties(eshkol_vm_standalone_smoke PROPERTIES
        ENVIRONMENT "ESHKOL_VM_NO_DISASM=1")
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vm/test_vm_c_api.cpp")
    add_executable(test_vm_c_api tests/vm/test_vm_c_api.cpp)
    target_compile_features(test_vm_c_api PRIVATE cxx_std_17)
    target_include_directories(test_vm_c_api PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/backend)
    eshkol_apply_vm_limit_definitions(test_vm_c_api)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(test_vm_c_api PRIVATE ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(test_vm_c_api PRIVATE
        eshkol-static
        ${ESHKOL_EXTRA_LINK_LIBS}
        ${LLVM_LIBS_LIST}
        ${LLVM_SYSTEM_LIBS_LIST}
    )
    add_test(NAME test_vm_c_api COMMAND test_vm_c_api)
endif()

# Direct C++ test of ReplJITContext::executeTagged value-capture path.
# Regression for the 2026-05-08 qLLM-bridge bug where
# eshkol_eval_string("42") returned {type=INT64, int_val=0} because the
# JIT-emitted main always emitted `ret i32 0`. Test links the full
# eshkol-static + eshkol-repl-lib stack so eshkol_eval_string and the
# JIT bridge resolve at link time.
if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/repl/test_repl_jit_executetagged.cpp")
    add_executable(test_repl_jit_executetagged
        tests/repl/test_repl_jit_executetagged.cpp)
    target_compile_features(test_repl_jit_executetagged PRIVATE cxx_std_17)
    target_include_directories(test_repl_jit_executetagged PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    if(LLVM_LIBRARY_DIRS)
        target_link_directories(test_repl_jit_executetagged PRIVATE
            ${LLVM_LIBRARY_DIRS})
    endif()
    target_link_libraries(test_repl_jit_executetagged PRIVATE
        eshkol-repl-lib eshkol-static ${ESHKOL_EXTRA_LINK_LIBS}
        ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    if(APPLE)
        # Force-load eshkol-static + eshkol-repl-lib so the strong
        # eval-bridge implementation in eval_bridge_impl.cpp wins over
        # the weak default in eshkol-static (matches eshkol-run linkage
        # at line ~1330 above).
        # -undefined dynamic_lookup lets unresolved weak agent-FFI
        # qllm_process_* refs (declared __attribute__((weak_import))
        # in repl_jit.cpp) fall through to a runtime dlsym, since the
        # test does not link the eshkol-agent-ffi archive.
        target_link_options(test_repl_jit_executetagged PRIVATE
            "-Wl,-force_load,$<TARGET_FILE:eshkol-static>"
            "-Wl,-force_load,$<TARGET_FILE:eshkol-repl-lib>"
            "-Wl,-export_dynamic"
            "-Wl,-undefined,dynamic_lookup")
    elseif(UNIX)
        target_link_options(test_repl_jit_executetagged PRIVATE
            "-Wl,--whole-archive"
            "$<TARGET_FILE:eshkol-static>"
            "$<TARGET_FILE:eshkol-repl-lib>"
            "-Wl,--no-whole-archive"
            "-Wl,--export-dynamic"
            "-Wl,--unresolved-symbols=ignore-in-object-files")
    endif()
    add_test(NAME test_repl_jit_executetagged
             COMMAND test_repl_jit_executetagged)
endif()

if(ESHKOL_BUILD_TESTS AND TARGET eshkol-run AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/repl/repl_stdlib_shadow_test.esk")
    add_test(
        NAME repl_stdlib_shadow_test
        COMMAND $<TARGET_FILE:eshkol-run>
                -r "${CMAKE_CURRENT_SOURCE_DIR}/tests/repl/repl_stdlib_shadow_test.esk"
                -L"${CMAKE_CURRENT_BINARY_DIR}"
    )
    set_tests_properties(repl_stdlib_shadow_test PROPERTIES
        TIMEOUT 300
        PASS_REGULAR_EXPRESSION "PASS: repl stdlib/math shadow")
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/windows_suite_surface_test.cpp")
    add_executable(windows_suite_surface_test tests/toolchain/windows_suite_surface_test.cpp)
    target_compile_features(windows_suite_surface_test PRIVATE cxx_std_17)
    add_test(
        NAME windows_suite_surface_test
        COMMAND windows_suite_surface_test "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/remote_windows_verify_surface_test.cpp")
    add_executable(remote_windows_verify_surface_test
        tests/toolchain/remote_windows_verify_surface_test.cpp)
    target_compile_features(remote_windows_verify_surface_test PRIVATE cxx_std_17)
    add_test(
        NAME remote_windows_verify_surface_test
        COMMAND remote_windows_verify_surface_test "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/aggregate_shell_suite_surface_test.cpp")
    add_executable(aggregate_shell_suite_surface_test
        tests/toolchain/aggregate_shell_suite_surface_test.cpp)
    target_compile_features(aggregate_shell_suite_surface_test PRIVATE cxx_std_17)
    add_test(
        NAME aggregate_shell_suite_surface_test
        COMMAND aggregate_shell_suite_surface_test "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/debian_package_surface_test.cpp")
    add_executable(debian_package_surface_test
        tests/toolchain/debian_package_surface_test.cpp)
    target_compile_features(debian_package_surface_test PRIVATE cxx_std_17)
    add_test(
        NAME debian_package_surface_test
        COMMAND debian_package_surface_test "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/docker_build_surface_test.cpp")
    add_executable(docker_build_surface_test
        tests/toolchain/docker_build_surface_test.cpp)
    target_compile_features(docker_build_surface_test PRIVATE cxx_std_17)
    add_test(
        NAME docker_build_surface_test
        COMMAND docker_build_surface_test "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/eshkol_server_security_surface_test.cpp")
    add_executable(eshkol_server_security_surface_test
        tests/toolchain/eshkol_server_security_surface_test.cpp)
    target_compile_features(eshkol_server_security_surface_test PRIVATE cxx_std_17)
    add_test(
        NAME eshkol_server_security_surface_test
        COMMAND eshkol_server_security_surface_test "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/macos_build_surface_test.cpp")
    add_executable(macos_build_surface_test
        tests/toolchain/macos_build_surface_test.cpp)
    target_compile_features(macos_build_surface_test PRIVATE cxx_std_17)
    add_test(
        NAME macos_build_surface_test
        COMMAND macos_build_surface_test "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

if(ESHKOL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/toolchain/release_workflow_surface_test.cpp")
    add_executable(release_workflow_surface_test
        tests/toolchain/release_workflow_surface_test.cpp)
    target_compile_features(release_workflow_surface_test PRIVATE cxx_std_17)
    add_test(
        NAME release_workflow_surface_test
        COMMAND release_workflow_surface_test "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

# ===== SDNC Paper Artifact: weight_matrices ===================================
# The paper "The Self-Differentiating Neural Computer: Computable Transformers
# via Analytical Weight Construction" (tsotchke 2026) reports a 2.8M-parameter
# transformer (d_model=128, n_layers=6, FFN=1024) whose weights ARE the Eshkol
# VM's microcode. The verification suite runs the same programs three
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
            NAME sdnc_paper_weight_tests
            COMMAND $<TARGET_FILE:weight_matrices>
        )
        # The suite count grows as VM-as-transformer coverage expands.
        set_tests_properties(sdnc_paper_weight_tests PROPERTIES
            PASS_REGULAR_EXPRESSION "Results: [0-9]+ passed, 0 failed"
        )
    endif()
endif()

# ===== SDNC de-risking experiment: differentiable external memory head ========
# Standalone prototype (its own main()) for a differentiable NTM/DNC-style
# external memory addressing head — the keystone primitive that will later let
# the SDNC lift its bounded 16-cell in-state arena to unbounded external memory
# while staying differentiable. Not compiled into libeshkol-static.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/backend/diff_memory_prototype.c")
    add_executable(diff_memory_prototype lib/backend/diff_memory_prototype.c)
    target_link_libraries(diff_memory_prototype PRIVATE m)
    set_target_properties(diff_memory_prototype PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tools
    )
endif()

# ===== XLA Integration Tests =====
if(ESHKOL_XLA_ENABLED AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/xla/xla_codegen_test.cpp")
    message(STATUS "Building XLA integration tests")

    add_executable(xla_codegen_test tests/xla/xla_codegen_test.cpp)

    target_include_directories(xla_codegen_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
    )

    target_compile_options(xla_codegen_test PRIVATE ${LLVM_CXXFLAGS_LIST})

    eshkol_target_llvm_includes(xla_codegen_test)

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
        # NOTE: agent_crypto.c (HMAC-SHA256/SHA256/secure-random) moved to
        # lib/core/crypto_primitives.c — it is now always compiled into the
        # core eshkol-runtime archive rather than the optional agent-FFI
        # bundle. See the ESHKOL_RUNTIME_SUPPORT_SRC crypto-libs block below
        # for why (Noesis bug report #2, 2026-07-04: an AOT link that only
        # pulled these symbols in via the separately-rebuilt
        # libeshkol-agent-ffi.a could race a concurrent rebuild of that
        # archive and fail with spurious "undefined symbols" errors).
        set(AGENT_FFI_SOURCES
            ${AGENT_FFI_DIR}/agent_terminal.c
            ${AGENT_FFI_DIR}/agent_subprocess.c
        )

        # HTTP server / WebSocket / Unix domain socket primitives.
        # POSIX-only (uses sockets, poll, accept) — already gated by the
        # surrounding ESHKOL_BUILD_AGENT_FFI=OFF on Windows.
        if(EXISTS "${AGENT_FFI_DIR}/agent_http_server.c")
            list(APPEND AGENT_FFI_SOURCES ${AGENT_FFI_DIR}/agent_http_server.c)
        endif()

        # Additional agent FFI primitives — all POSIX-only and gated above.
        # Each module is independent and gracefully degrades when its
        # optional system library is absent (HAS_* compile defines).
        #
        # NOTE: agent_concurrency.c is intentionally OMITTED — it duplicates
        # eshkol_mutex_* / eshkol_condvar_* / etc. already defined by
        # core.threads (lib/core/threads.c, exposed via core.threads.esk).
        # If we ever need handle-based mutex/channel/timer primitives we
        # should rename them with an explicit suffix (e.g. _h) before
        # re-enabling.  See agent's own eshkol-side wrappers — they use
        # the core.threads primitives directly.
        foreach(_agent_extra_src
                agent_compression.c     # zlib / gzip — compiles to stub w/o HAS_ZLIB
                agent_eagle_training.c  # native EAGLE linear-head train step
                agent_kb_io.c           # KB JSON persistence
                agent_platform.c        # filesystem, time, mmap, glob, base64url, uuid
                agent_poll.c            # IO multiplexing (poll/select)
                agent_signal.c          # SIGINT/SIGTERM/SIGHUP + atexit
                agent_tensor_io.c       # tensor save/load
                agent_treesitter.c      # tree-sitter — stubs w/o HAS_TREE_SITTER
                agent_watch.c           # kqueue/inotify file watch
                agent_yoga.c            # Facebook Yoga flexbox — stubs w/o HAS_YOGA
        )
            if(EXISTS "${AGENT_FFI_DIR}/${_agent_extra_src}")
                list(APPEND AGENT_FFI_SOURCES "${AGENT_FFI_DIR}/${_agent_extra_src}")
            endif()
        endforeach()

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

        # Optional: libcurl for native HTTP client (Anthropic API,
        # WebFetch, OAuth, generic HTTPS). Without this the agent's
        # http.esk falls back to qLLM symbols (returns "explicit
        # unavailable error") — see plans/note-for-eshkol-agent.md.
        pkg_check_modules(LIBCURL_AGENT libcurl)
        if(LIBCURL_AGENT_FOUND AND EXISTS "${AGENT_FFI_DIR}/agent_http_client.c")
            list(APPEND AGENT_FFI_SOURCES ${AGENT_FFI_DIR}/agent_http_client.c)
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

        if(LIBCURL_AGENT_FOUND)
            target_include_directories(eshkol-agent-ffi PRIVATE ${LIBCURL_AGENT_INCLUDE_DIRS})
            target_compile_definitions(eshkol-agent-ffi PRIVATE ESHKOL_HAVE_LIBCURL=1)
            if(LIBCURL_AGENT_LINK_LIBRARIES)
                target_link_libraries(eshkol-agent-ffi PUBLIC ${LIBCURL_AGENT_LINK_LIBRARIES})
            else()
                target_link_directories(eshkol-agent-ffi PUBLIC ${LIBCURL_AGENT_LIBRARY_DIRS})
                target_link_libraries(eshkol-agent-ffi PUBLIC ${LIBCURL_AGENT_LIBRARIES})
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
        # (Crypto — SHA256/HMAC/secure random — is unconditional now, see
        # lib/core/crypto_primitives.c / eshkol-runtime, not gated by this flag.)
        message(STATUS "  Regex:     ${PCRE2_AGENT_FOUND}")
        message(STATUS "  SQLite:    ${SQLITE3_AGENT_FOUND}")
        message(STATUS "  HTTP/curl: ${LIBCURL_AGENT_FOUND}")

        # ─────────────────────────────────────────────────────────────
        # Build the link-arg string AOT-compiled user binaries need so
        # symbols from agent_http_client.c / agent_subprocess.c / etc.
        # resolve. Writes to ESHKOL_HOST_AGENT_FFI_LINK_ARGS, which the
        # final configure_file at the end of this CMakeLists splices
        # into build_config.h. eshkol-run.cpp reads it from there at
        # the AOT link step (only when the user's source contains
        # `(require agent.…)`, otherwise we don't pull in libcurl/etc.
        # for programs that don't need them).
        #
        # Order matters for static-archive resolution on POSIX
        # toolchains: agent-ffi.a first (force-loaded so qllm_*
        # symbols stay live), then deps (libcurl, sqlite3, pcre2,
        # ncurses, OpenSSL/Security framework). For macOS we also need
        # -framework Security and -framework CoreFoundation; on Linux,
        # OpenSSL via OpenSSL::SSL/Crypto if found, otherwise libsecret.
        # Use the well-known archive output path (CMake creates it as
        # libeshkol-agent-ffi.a in CMAKE_BINARY_DIR for STATIC libs).
        # Avoids generator expressions which configure_file can't
        # evaluate. The path matters at AOT link time, not configure
        # time, so a string here is fine.
        set(_agent_ffi_archive
            "${CMAKE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}eshkol-agent-ffi${CMAKE_STATIC_LIBRARY_SUFFIX}")
        if(APPLE)
            list(APPEND ESHKOL_HOST_AGENT_FFI_LINK_ARGS
                "-Wl,-force_load,${_agent_ffi_archive}")
        else()
            list(APPEND ESHKOL_HOST_AGENT_FFI_LINK_ARGS
                "-Wl,--whole-archive" "${_agent_ffi_archive}" "-Wl,--no-whole-archive")
        endif()

        # Transitive deps. Each is gated by the same flag that
        # determined whether the C source got compiled into agent-ffi.
        if(LIBCURL_AGENT_FOUND)
            if(LIBCURL_AGENT_LINK_LIBRARIES)
                foreach(_lib IN LISTS LIBCURL_AGENT_LINK_LIBRARIES)
                    list(APPEND ESHKOL_HOST_AGENT_FFI_LINK_ARGS "${_lib}")
                endforeach()
            else()
                foreach(_dir IN LISTS LIBCURL_AGENT_LIBRARY_DIRS)
                    list(APPEND ESHKOL_HOST_AGENT_FFI_LINK_ARGS "-L${_dir}")
                endforeach()
                foreach(_lib IN LISTS LIBCURL_AGENT_LIBRARIES)
                    list(APPEND ESHKOL_HOST_AGENT_FFI_LINK_ARGS "-l${_lib}")
                endforeach()
            endif()
        endif()
        if(SQLITE3_AGENT_FOUND)
            if(SQLITE3_AGENT_LINK_LIBRARIES)
                foreach(_lib IN LISTS SQLITE3_AGENT_LINK_LIBRARIES)
                    list(APPEND ESHKOL_HOST_AGENT_FFI_LINK_ARGS "${_lib}")
                endforeach()
            else()
                foreach(_lib IN LISTS SQLITE3_AGENT_LIBRARIES)
                    list(APPEND ESHKOL_HOST_AGENT_FFI_LINK_ARGS "-l${_lib}")
                endforeach()
            endif()
        endif()
        if(PCRE2_AGENT_FOUND)
            if(PCRE2_AGENT_LINK_LIBRARIES)
                foreach(_lib IN LISTS PCRE2_AGENT_LINK_LIBRARIES)
                    list(APPEND ESHKOL_HOST_AGENT_FFI_LINK_ARGS "${_lib}")
                endforeach()
            else()
                foreach(_lib IN LISTS PCRE2_AGENT_LIBRARIES)
                    list(APPEND ESHKOL_HOST_AGENT_FFI_LINK_ARGS "-l${_lib}")
                endforeach()
            endif()
        endif()
        if(NCURSES_AGENT)
            list(APPEND ESHKOL_HOST_AGENT_FFI_LINK_ARGS "${NCURSES_AGENT}")
        endif()
        if(APPLE)
            list(APPEND ESHKOL_HOST_AGENT_FFI_LINK_ARGS
                "-framework" "Security"
                "-framework" "CoreFoundation")
        else()
            if(OpenSSL_FOUND)
                list(APPEND ESHKOL_HOST_AGENT_FFI_LINK_ARGS "-lssl" "-lcrypto")
            endif()
        endif()

        # The CMake variable carries semicolons as list separators; for
        # build_config.h.in's @VAR@ substitution we want the args
        # whitespace-separated. Convert here. Items that themselves
        # contain spaces (cmake's $<TARGET_FILE:…> generator expressions
        # don't, framework arg pairs are split into two tokens above)
        # are safe; user paths with spaces would not be — but the
        # agent-FFI deps are all system libraries from pkg-config, not
        # user-provided.
        string(REPLACE ";" " " ESHKOL_HOST_AGENT_FFI_LINK_ARGS
                       "${ESHKOL_HOST_AGENT_FFI_LINK_ARGS}")

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
            if(APPLE)
                target_link_libraries(eshkol-run PRIVATE eshkol-agent-ffi)
                target_link_options(eshkol-run PRIVATE
                    "-Wl,-force_load,$<TARGET_FILE:eshkol-agent-ffi>")
            elseif(NOT WIN32)
                # Keep the whole-archive group in the ordered library list.
                # CMake de-duplicates LINK_OPTIONS and can otherwise split this
                # group away from other --whole-archive uses in sanitizer builds.
                target_link_libraries(eshkol-run PRIVATE
                    "-Wl,--whole-archive" eshkol-agent-ffi "-Wl,--no-whole-archive")
            else()
                target_link_libraries(eshkol-run PRIVATE eshkol-agent-ffi)
            endif()
        endif()

        if(TARGET eshkol-repl)
            # Keep the long-lived REPL's JIT resolver on the same footing
            # as eshkol-run -r: agent FFI externs loaded from .esk files
            # should be available in persistent control sessions too.
            if(APPLE)
                target_link_libraries(eshkol-repl PRIVATE eshkol-agent-ffi)
                target_link_options(eshkol-repl PRIVATE
                    "-Wl,-force_load,$<TARGET_FILE:eshkol-agent-ffi>")
            elseif(NOT WIN32)
                target_link_libraries(eshkol-repl PRIVATE
                    "-Wl,--whole-archive" eshkol-agent-ffi "-Wl,--no-whole-archive")
            else()
                target_link_libraries(eshkol-repl PRIVATE eshkol-agent-ffi)
            endif()
        endif()
    else()
        message(STATUS "Agent FFI modules: lib/agent/c/ not found, skipping")
    endif()
endif()

# Re-emit build_config.h after the agent-FFI block so
# ESHKOL_HOST_AGENT_FFI_LINK_ARGS gets baked in. The earlier
# configure_file (line ~917) ran before the agent block was processed,
# so without this re-emit the macro would always be empty in the
# header. Generator expressions like $<TARGET_FILE:eshkol-agent-ffi>
# are evaluated at build time, so the resulting string carries the
# unresolved expression; eshkol-run.cpp resolves it at link step by
# string-replacing against the build directory.
if(DEFINED ESHKOL_GENERATED_INCLUDE_DIR AND TARGET eshkol-agent-ffi)
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_config.h.in"
      "${ESHKOL_GENERATED_INCLUDE_DIR}/eshkol/build_config.h"
      @ONLY
    )
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

# ESH-0215: build-integration fixture exercising cmake/EshkolCompile.cmake's
# --emit-depfile-backed dependency tracking (a (load …)ed dependency must
# retrigger a recompile+relink on a single `ninja` invocation).
if(ESHKOL_BUILD_TESTS AND TARGET eshkol-run AND TARGET eshkol-runtime AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/build_integration/CMakeLists.txt")
    add_subdirectory(tests/build_integration)
endif()

# where to find our CMake modules
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Packing)
