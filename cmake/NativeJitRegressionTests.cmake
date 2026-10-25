# Opt-in until the recorded Windows in-process JIT regressions are repaired.
# A failing baseline remains a failure; do not use WILL_FAIL or suppress a case.
if(ESHKOL_BUILD_TESTS AND WIN32 AND TARGET eshkol-run)
    option(ESHKOL_TEST_WINDOWS_JIT_REGRESSIONS
        "Run the native Windows JIT regression probes with AOT controls" OFF)
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
