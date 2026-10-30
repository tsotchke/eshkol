# Scoped consumer gate; enable explicitly for this review unit.
option(ESHKOL_HAWKES_LIVE_TESTS "Register Maintain signed Hawkes forecasts and sensitivities through DBSP" OFF)
if(ESHKOL_BUILD_TESTS AND ESHKOL_HAWKES_LIVE_TESTS AND TARGET eshkol-run)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    option(ESHKOL_CONSUMER_COMPILE_LINALG "Compile linalg for an older precompiled stdlib" OFF)
    set(_consumer_linalg)
    if(ESHKOL_CONSUMER_COMPILE_LINALG)
        list(APPEND _consumer_linalg --compile-linalg-source)
    endif()
    add_test(NAME hawkes_live
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/repro/hawkes-live/run_live_gate.py"
            --eshkol-bin $<TARGET_FILE:eshkol-run> ${_consumer_linalg})
    set_tests_properties(hawkes_live PROPERTIES TIMEOUT 900 LABELS "bounded;hawkes-live")
endif()
