#
# Copyright (C) tsotchke
#
# SPDX-License-Identifier: MIT
#
cmake_minimum_required(VERSION 3.14)

set(ESHKOL_VERSION "1.1.11")

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
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/LLVMToolchain.cmake)

# StableHLO Root (for XLA builds with bundled LLVM/MLIR/StableHLO)
# Set this to the path where StableHLO was built (e.g., /tmp/stablehlo)
# When set, uses the bundled LLVM/MLIR instead of the lite/native LLVM 21 toolchain
set(STABLEHLO_ROOT "" CACHE PATH "Path to StableHLO installation with bundled LLVM/MLIR")

# ===== Sanitizers =====
if(ESHKOL_ENABLE_ASAN)
    message(STATUS "Address Sanitizer: ENABLED")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

if(ESHKOL_ENABLE_UBSAN)
    message(STATUS "Undefined Behavior Sanitizer: ENABLED")
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=undefined)
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
    # ===== LITE BUILD: Use system LLVM 21 =====
    message(STATUS "=== LITE BUILD MODE: Using system LLVM ===")
    set(USING_STABLEHLO_LLVM FALSE)

    eshkol_find_lite_llvm()

    # Generated build config for toolchain/runtime helpers (Windows port)
    set(ESHKOL_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${ESHKOL_GENERATED_INCLUDE_DIR}/eshkol")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_config.h.in"
      "${ESHKOL_GENERATED_INCLUDE_DIR}/eshkol/build_config.h"
      @ONLY
    )

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
endif()

# Convert to CMake format (common for both modes)
separate_arguments(LLVM_CXXFLAGS_LIST UNIX_COMMAND "${LLVM_CXXFLAGS}")
separate_arguments(LLVM_LDFLAGS_LIST UNIX_COMMAND "${LLVM_LDFLAGS}")
separate_arguments(LLVM_LIBS_LIST UNIX_COMMAND "${LLVM_LIBS}")
separate_arguments(LLVM_SYSTEM_LIBS_LIST UNIX_COMMAND "${LLVM_SYSTEM_LIBS}")

# Keep LLVM's defines and unwind flags, but let the project control language mode
# and exception support.
list(FILTER LLVM_CXXFLAGS_LIST EXCLUDE REGEX "^-std=")
list(FILTER LLVM_CXXFLAGS_LIST EXCLUDE REGEX "^-fno-exceptions$")

# Debug output for troubleshooting
message(STATUS "LLVM Config: ${LLVM_CONFIG_EXECUTABLE}")
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
    # 2. STABLEHLO_ROOT not set: Try system LLVM 21 (Lite build fallback)

    if(STABLEHLO_ROOT AND EXISTS "${STABLEHLO_ROOT}")
        message(STATUS "Using StableHLO installation at: ${STABLEHLO_ROOT}")

        # StableHLO build structure (two modes):
        # Mode 1: Bundled LLVM — ${STABLEHLO_ROOT}/llvm-build/ exists
        # Mode 2: System LLVM — StableHLO built against system LLVM (no llvm-build/)
        if(EXISTS "${STABLEHLO_ROOT}/llvm-build")
            set(STABLEHLO_LLVM_BUILD "${STABLEHLO_ROOT}/llvm-build")
            message(STATUS "StableHLO: using bundled LLVM at ${STABLEHLO_LLVM_BUILD}")
        else()
            # StableHLO built against system LLVM — derive root from llvm-config
            get_filename_component(_sys_llvm_root "${LLVM_CONFIG_EXECUTABLE}" DIRECTORY)
            get_filename_component(_sys_llvm_root "${_sys_llvm_root}" DIRECTORY)
            set(STABLEHLO_LLVM_BUILD "${_sys_llvm_root}")
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
        get_filename_component(LLVM_ROOT_DIR "${LLVM_CONFIG_EXECUTABLE}" DIRECTORY)
        get_filename_component(LLVM_ROOT_DIR "${LLVM_ROOT_DIR}" DIRECTORY)

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
list(FILTER LIB_SRC EXCLUDE REGEX "lib/backend/archive/.*")

# Exclude standalone .cpp programs with main()
list(FILTER LIB_SRC EXCLUDE REGEX "lib/backend/weight_compiler\\.cpp$")
# platform_runtime.cpp excluded from static lib (linked separately for REPL/executables)
list(FILTER LIB_SRC EXCLUDE REGEX "lib/core/platform_runtime\\.cpp$")

# Add eshkol_vm.c (unity build: includes vm_*.c, eskb_*.c internally)
list(APPEND LIB_SRC lib/backend/eshkol_vm.c)

# Exclude XLA sources from main library when XLA is disabled
if(NOT ESHKOL_XLA_ENABLED)
    list(FILTER LIB_SRC EXCLUDE REGEX "lib/backend/xla/.*")
endif()

# Get REPL sources separately
file(GLOB REPL_SRC RELATIVE ${CMAKE_SOURCE_DIR} "lib/repl/*.cpp")

add_compile_definitions(
  ESHKOL_VER="${ESHKOL_VERSION}-accelerate"
)

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
  ${LLVM_CXXFLAGS_LIST}
  -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
  -Wformat -Wformat-security
  -fstack-protector-strong
)
# Buffer overflow detection at compile time (requires optimization)
target_compile_definitions(eshkol-static PRIVATE $<$<NOT:$<CONFIG:Debug>>:_FORTIFY_SOURCE=2>)
target_compile_definitions(eshkol-static PRIVATE ESHKOL_LLVM_BACKEND_ENABLED)
target_compile_definitions(eshkol-static PRIVATE ESHKOL_VM_LIBRARY_MODE)

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
  target_compile_options(${exename} PRIVATE ${LLVM_CXXFLAGS_LIST} -fPIE)
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
  
  target_link_libraries(${exename} PUBLIC eshkol-static ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
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
  target_compile_options(eshkol-repl-lib PRIVATE ${LLVM_CXXFLAGS_LIST})
  target_compile_definitions(eshkol-repl-lib PRIVATE ESHKOL_LLVM_BACKEND_ENABLED)

  if(LLVM_INCLUDE_DIRS)
    target_include_directories(eshkol-repl-lib PRIVATE ${LLVM_INCLUDE_DIRS})
  endif()

  if(LLVM_LIBRARY_DIRS)
    target_link_directories(eshkol-repl-lib PRIVATE ${LLVM_LIBRARY_DIRS})
  endif()

  # Link against main compiler library and LLVM
  target_link_libraries(eshkol-repl-lib PUBLIC eshkol-static ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})

  # REPL executable (only if we have the source)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/exe/eshkol-repl.cpp")
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

    target_link_libraries(eshkol-repl PRIVATE eshkol-repl-lib)

    # Force-load ALL symbols from eshkol-static into the REPL binary.
    # Without this, the macOS linker dead-strips archive members that aren't
    # directly referenced (e.g. XLA runtime) but ARE needed by JIT-compiled
    # code at runtime. Combined with -export_dynamic below, this makes
    # DynamicLibrarySearchGenerator auto-resolve ALL runtime symbols.
    if(APPLE)
      target_link_options(eshkol-repl PRIVATE
        "-Wl,-force_load" "$<TARGET_FILE:eshkol-static>")
    else()
      target_link_options(eshkol-repl PRIVATE
        "-Wl,--whole-archive" "$<TARGET_FILE:eshkol-static>" "-Wl,--no-whole-archive")
    endif()

    # Export all symbols for JIT symbol resolution
    if(APPLE)
      target_link_options(eshkol-repl PRIVATE "-Wl,-export_dynamic")
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
  # Force-load eshkol-static for JIT symbol resolution (same as REPL)
  if(APPLE)
    target_link_options(eshkol-run PRIVATE
      "-Wl,-force_load" "$<TARGET_FILE:eshkol-static>")
    target_link_options(eshkol-run PRIVATE "-Wl,-export_dynamic")
    # 512MB stack for deep recursion (main thread stack set at link time on macOS)
    target_link_options(eshkol-run PRIVATE "-Wl,-stack_size,0x20000000")
    target_link_options(eshkol-run PRIVATE "-Wl,-bind_at_load")
  else()
    target_link_options(eshkol-run PRIVATE
      "-Wl,--whole-archive" "$<TARGET_FILE:eshkol-static>" "-Wl,--no-whole-archive")
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

    install(TARGETS eshkol-pkg DESTINATION bin)
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

    target_link_libraries(eshkol-lsp PRIVATE eshkol-static ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})

    install(TARGETS eshkol-lsp DESTINATION bin)
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
# C bridge libraries for the Eshkol agent (terminal, regex, sqlite, crypto)

option(ESHKOL_BUILD_AGENT_FFI "Build agent FFI modules" ON)

if(ESHKOL_BUILD_AGENT_FFI)
    set(AGENT_FFI_DIR "${CMAKE_SOURCE_DIR}/lib/agent/c")

    if(EXISTS "${AGENT_FFI_DIR}")
        set(AGENT_FFI_SOURCES
            ${AGENT_FFI_DIR}/agent_terminal.c
            ${AGENT_FFI_DIR}/agent_crypto.c
        )

        # Optional: PCRE2 for regex
        pkg_check_modules(PCRE2_AGENT pcre2-8)
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

        # Terminal needs ncurses
        find_library(NCURSES_AGENT ncurses)
        if(NCURSES_AGENT)
            target_link_libraries(eshkol-agent-ffi PRIVATE ${NCURSES_AGENT})
        endif()

        if(PCRE2_AGENT_FOUND)
            target_include_directories(eshkol-agent-ffi PRIVATE ${PCRE2_AGENT_INCLUDE_DIRS})
            target_link_libraries(eshkol-agent-ffi PRIVATE ${PCRE2_AGENT_LIBRARIES})
            target_compile_definitions(eshkol-agent-ffi PRIVATE PCRE2_CODE_UNIT_WIDTH=8)
        endif()

        if(SQLITE3_AGENT_FOUND)
            target_link_libraries(eshkol-agent-ffi PRIVATE ${SQLITE3_AGENT_LIBRARIES})
        endif()

        # Crypto: CommonCrypto on macOS, OpenSSL on Linux
        if(APPLE)
            target_link_libraries(eshkol-agent-ffi PRIVATE
                "-framework Security"
                "-framework CoreFoundation"
            )
        else()
            find_package(OpenSSL)
            if(OpenSSL_FOUND)
                target_link_libraries(eshkol-agent-ffi PRIVATE OpenSSL::SSL OpenSSL::Crypto)
            endif()
        endif()

        message(STATUS "Agent FFI modules: ENABLED")
        message(STATUS "  Terminal:  YES")
        message(STATUS "  Crypto:    YES")
        message(STATUS "  Regex:     ${PCRE2_AGENT_FOUND}")
        message(STATUS "  SQLite:    ${SQLITE3_AGENT_FOUND}")
    else()
        message(STATUS "Agent FFI modules: lib/agent/c/ not found, skipping")
    endif()
endif()

# where to find our CMake modules
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Packing)
