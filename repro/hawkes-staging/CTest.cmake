# Scoped consumer gate; enable explicitly for this review unit.
option(ESHKOL_HAWKES_STAGING_TESTS "Register Specialize certified Hawkes operators into checked native kernels" OFF)
if(ESHKOL_BUILD_TESTS AND ESHKOL_HAWKES_STAGING_TESTS AND TARGET eshkol-run)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME hawkes_staging
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/repro/hawkes-staging/run_staging_gate.py"
            --eshkol-bin $<TARGET_FILE:eshkol-run>)
    set_tests_properties(hawkes_staging PROPERTIES TIMEOUT 900 LABELS "bounded;hawkes-staging")
endif()
