# The repaired Windows x64 paths are covered by default. Other Windows
# architectures may opt in, but were not part of the x64 validation.
# A failure remains a failure; do not use WILL_FAIL or suppress a case.
if(ESHKOL_BUILD_TESTS AND WIN32 AND TARGET eshkol-run)
    set(_eshkol_windows_jit_default OFF)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$" OR
       CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64" OR
       MSVC_CXX_ARCHITECTURE_ID STREQUAL "x64" OR
       CMAKE_VS_PLATFORM_NAME STREQUAL "x64")
        set(_eshkol_windows_jit_default ON)
    endif()
    option(ESHKOL_TEST_WINDOWS_JIT_REGRESSIONS
        "Run the native Windows JIT regression probes with AOT controls"
        ${_eshkol_windows_jit_default})
    if(ESHKOL_TEST_WINDOWS_JIT_REGRESSIONS)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        set(_eshkol_native_jit_probe_root "${CMAKE_CURRENT_LIST_DIR}/..")
        add_test(NAME windows_jit_regressions
            COMMAND ${Python3_EXECUTABLE}
                ${_eshkol_native_jit_probe_root}/scripts/run_native_jit_regressions.py
                --runner $<TARGET_FILE:eshkol-run>
                --work-dir ${CMAKE_CURRENT_BINARY_DIR}/windows-jit-regressions
                --timeout 30)
        set_tests_properties(windows_jit_regressions PROPERTIES
            LABELS "windows;jit;regression" TIMEOUT 750)
        add_test(NAME windows_jit_regressions_harness
            COMMAND ${Python3_EXECUTABLE}
                ${_eshkol_native_jit_probe_root}/tests/toolchain/test_native_jit_regressions.py)
        set_tests_properties(windows_jit_regressions_harness PROPERTIES
            LABELS "windows;jit;regression" TIMEOUT 60)
    endif()
endif()
