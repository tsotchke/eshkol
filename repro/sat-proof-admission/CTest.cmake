option(ESHKOL_SAT_PROOF_ADMISSION_TESTS "Register bounded SAT proof-admission qualification" OFF)
if(ESHKOL_BUILD_TESTS AND ESHKOL_SAT_PROOF_ADMISSION_TESTS AND TARGET eshkol-run)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME sat_proof_admission
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_sat_proof_gate.py"
            --eshkol-bin $<TARGET_FILE:eshkol-run>)
    set_tests_properties(sat_proof_admission PROPERTIES TIMEOUT 1200 LABELS "bounded;sat;proof")
    add_test(NAME sat_proof_admission_readiness
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/sat_proof_gate_self_test.py")
    set_tests_properties(sat_proof_admission_readiness PROPERTIES TIMEOUT 45 LABELS "bounded;sat;proof")
endif()
