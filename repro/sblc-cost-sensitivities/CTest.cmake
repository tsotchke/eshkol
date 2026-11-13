# Scoped consumer gate; enable explicitly for this review unit.
option(ESHKOL_SBLC_COST_SENSITIVITIES_TESTS "Register Differentiate source-bound Bernoulli SBLC work laws" OFF)
if(ESHKOL_BUILD_TESTS AND ESHKOL_SBLC_COST_SENSITIVITIES_TESTS AND TARGET eshkol-run)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME sblc_cost_sensitivities
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/repro/sblc-cost-sensitivities/run_cost_gate.py"
            --eshkol-bin $<TARGET_FILE:eshkol-run>)
    set_tests_properties(sblc_cost_sensitivities PROPERTIES TIMEOUT 900 LABELS "bounded;sblc-cost-sensitivities")
endif()
