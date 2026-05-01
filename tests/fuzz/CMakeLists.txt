# Fuzzing harnesses. Enable with -DESHKOL_ENABLE_FUZZ=ON at the
# top-level CMake configure step. Built with `-fsanitize=fuzzer`
# which requires clang; gcc builds skip the target.
#
# Each harness is a standalone libFuzzer driver that links against
# eshkol-static and runs a specific ingest point in a loop.
#
#   fuzz_parser     — feeds arbitrary bytes into the Scheme parser
#   (more harnesses land as the ingest surface expands)

if(NOT ESHKOL_ENABLE_FUZZ)
    return()
endif()

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(WARNING "libFuzzer harnesses require clang; skipping fuzz targets")
    return()
endif()

# macOS AppleClang does NOT ship libclang_rt.fuzzer_*. Homebrew clang does;
# Linux system clang does. Rather than fail configure, detect and emit a
# concrete command-line hint.
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(WARNING
        "Xcode AppleClang does not ship libclang_rt.fuzzer. "
        "On macOS: `brew install llvm` then configure with "
        "`-DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang "
        "-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ "
        "-DESHKOL_ENABLE_FUZZ=ON`. Skipping fuzz targets for now.")
    return()
endif()

set(FUZZ_FLAGS -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer -g)

add_executable(fuzz_parser fuzz_parser.cpp)
target_compile_options(fuzz_parser PRIVATE ${FUZZ_FLAGS})
target_link_options(fuzz_parser PRIVATE ${FUZZ_FLAGS})
target_link_libraries(fuzz_parser PRIVATE eshkol-static)
target_include_directories(fuzz_parser PRIVATE ${PROJECT_SOURCE_DIR}/inc)

message(STATUS "Fuzz: enabled — fuzz_parser target built with fuzzer+asan+ubsan")
