# Scoped consumer gate; enable explicitly for this review unit.
option(ESHKOL_SDNC_HOLES_TESTS "Register Learn a designated Hawkes program hole in the constructed SDNC" OFF)
if(ESHKOL_BUILD_TESTS AND ESHKOL_SDNC_HOLES_TESTS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME sdnc_holes
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/repro/sdnc-holes/run_sdnc_gate.py"
            --cc "${CMAKE_C_COMPILER}")
    set_tests_properties(sdnc_holes PROPERTIES TIMEOUT 180 LABELS "bounded;sdnc-holes")
endif()
