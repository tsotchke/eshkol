# Fuzzing harnesses. Enable with -DESHKOL_ENABLE_FUZZ=ON at the
# top-level CMake configure step.
#
#   reader_fuzz     — seeded adversarial generator + fork/watchdog
#                      isolation harness for the hosted S-expression
#                      reader (eshkol_read_sexpr). Plain executable,
#                      any compiler, no libFuzzer runtime required.
#   eskm_model_fuzz — deterministic ESKM generator/mutator driving the
#                      public model/tensor loaders through a bounded child
#                      probe. Plain executable + Python driver, any compiler.
#   fuzz_parser     — libFuzzer driver: feeds arbitrary bytes into the
#                      frontend Scheme source parser. Requires clang
#                      (`-fsanitize=fuzzer`); gcc/AppleClang builds skip it.
#   (more harnesses land as the ingest surface expands)

if(NOT ESHKOL_ENABLE_FUZZ)
    return()
endif()

# reader_fuzz — deterministic, seeded adversarial-input harness for the
# hosted S-expression reader (lib/core/runtime_reader_hosted.cpp, entry
# point eshkol_read_sexpr). Unlike the libFuzzer-based targets below,
# this is a plain executable with its own main() and seeded PRNG (no
# coverage-guided mutation, no libFuzzer runtime needed), so it builds
# with any compiler (including AppleClang) and links only against the
# slim eshkol-runtime archive — no LLVM/MLIR compiler objects required,
# which keeps `cmake --build build-fuzz --target reader_fuzz` fast.
#
#   -DESHKOL_FUZZ_ASAN=ON  builds it (and the runtime objects it
#   exercises) with ASan+UBSan for a "separate cheap build" hardening
#   pass when a crash class needs root-causing.
if(TARGET eshkol-runtime)
    add_executable(reader_fuzz reader_fuzz_driver.cpp)
    target_include_directories(reader_fuzz PRIVATE
        ${PROJECT_SOURCE_DIR}/inc
        ${PROJECT_SOURCE_DIR}/lib/core
    )
    target_link_libraries(reader_fuzz PRIVATE eshkol-runtime)
    if(ESHKOL_FUZZ_ASAN)
        set(ESHKOL_SEEDED_FUZZ_ASAN_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -g)
        target_compile_options(reader_fuzz PRIVATE ${ESHKOL_SEEDED_FUZZ_ASAN_FLAGS})
        target_link_options(reader_fuzz PRIVATE ${ESHKOL_SEEDED_FUZZ_ASAN_FLAGS})
        # ASan needs the reader/runtime objects instrumented too, or heap
        # overflows inside them go undetected.
        if(TARGET eshkol-runtime-hosted-obj)
            target_compile_options(eshkol-runtime-hosted-obj PRIVATE ${ESHKOL_SEEDED_FUZZ_ASAN_FLAGS})
        endif()
        if(TARGET eshkol-runtime-core-obj)
            target_compile_options(eshkol-runtime-core-obj PRIVATE ${ESHKOL_SEEDED_FUZZ_ASAN_FLAGS})
        endif()
        if(TARGET eshkol-runtime-support-obj)
            target_compile_options(eshkol-runtime-support-obj PRIVATE ${ESHKOL_SEEDED_FUZZ_ASAN_FLAGS})
        endif()
        message(STATUS "Fuzz: reader_fuzz built with ASan+UBSan (ESHKOL_FUZZ_ASAN=ON)")
    endif()
    message(STATUS "Fuzz: enabled — reader_fuzz target (seeded adversarial reader harness)")

    add_executable(eskm_model_fuzz_probe eskm_model_fuzz_probe.cpp)
    target_compile_features(eskm_model_fuzz_probe PRIVATE cxx_std_17)
    target_include_directories(eskm_model_fuzz_probe PRIVATE
        ${PROJECT_SOURCE_DIR}/inc
        ${PROJECT_SOURCE_DIR}/lib/core
    )
    target_link_libraries(eskm_model_fuzz_probe PRIVATE eshkol-runtime)
    if(ESHKOL_FUZZ_ASAN)
        target_compile_options(eskm_model_fuzz_probe PRIVATE ${ESHKOL_SEEDED_FUZZ_ASAN_FLAGS})
        target_link_options(eskm_model_fuzz_probe PRIVATE ${ESHKOL_SEEDED_FUZZ_ASAN_FLAGS})
    endif()
    message(STATUS "Fuzz: enabled — eskm_model_fuzz_probe target (seeded ESKM campaign)")

    if(ESHKOL_BUILD_TESTS AND ESHKOL_PYTHON3_EXECUTABLE)
        set(ESHKOL_ESKM_FUZZ_MEMORY_MB 256)
        if(ESHKOL_FUZZ_ASAN OR ESHKOL_ENABLE_ASAN OR ESHKOL_ENABLE_TSAN OR
           ESHKOL_ENABLE_MSAN OR ESHKOL_ENABLE_LSAN)
            # Memory-checking sanitizers reserve large virtual address ranges,
            # so RLIMIT_AS cannot be combined with them. Ordinary builds carry
            # the memory cap; sanitizer builds retain per-process timeouts.
            set(ESHKOL_ESKM_FUZZ_MEMORY_MB 0)
        endif()
        add_test(
            NAME eskm_model_fuzz_smoke
            COMMAND "${ESHKOL_PYTHON3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/scripts/run_eskm_model_fuzz.py"
                    --smoke --self-test
                    --probe $<TARGET_FILE:eskm_model_fuzz_probe>
                    --memory-mb "${ESHKOL_ESKM_FUZZ_MEMORY_MB}"
        )
        set_tests_properties(eskm_model_fuzz_smoke PROPERTIES
            LABELS "fuzz;model-io;security"
            TIMEOUT 180
            PASS_REGULAR_EXPRESSION "RESULT: OK"
        )
    endif()
else()
    message(WARNING "Fuzz: eshkol-runtime target not available; skipping seeded fuzz probes")
endif()

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(WARNING "libFuzzer harnesses require clang; skipping remaining fuzz targets")
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
