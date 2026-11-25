# Selected MIT contribution: bounded standalone qualification.
option(ESHKOL_NONLINEAR_OBSERVABILITY_TESTS "Register nonlinear-observability qualification" OFF)
if(ESHKOL_BUILD_TESTS AND ESHKOL_NONLINEAR_OBSERVABILITY_TESTS AND TARGET eshkol-run)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(ESHKOL_OBSERVABILITY_CERTCHECK "${CMAKE_CURRENT_SOURCE_DIR}/tools/certcheck/eshkol_certcheck.py" CACHE FILEPATH "Pinned independent certificate checker from PR 900")
    if(NOT EXISTS "${ESHKOL_OBSERVABILITY_CERTCHECK}")
        message(FATAL_ERROR "Nonlinear observability requires the independent checker from PR 900")
    endif()
    add_test(NAME nonlinear-observability_native
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/repro/nonlinear-observability/gate.py"
            --eshkol-bin $<TARGET_FILE:eshkol-run> --certcheck "${ESHKOL_OBSERVABILITY_CERTCHECK}")
    set_tests_properties(nonlinear-observability_native PROPERTIES TIMEOUT 1500 LABELS "bounded;math;proof")
    add_test(NAME nonlinear-observability_readiness
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/repro/nonlinear-observability/self_test.py")
    set_tests_properties(nonlinear-observability_readiness PROPERTIES TIMEOUT 45 LABELS "bounded;math;proof")
endif()
