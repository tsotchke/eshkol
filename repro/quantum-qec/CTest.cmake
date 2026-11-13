# Selected MIT contribution: bounded standalone qualification.
option(ESHKOL_QUANTUM_QEC_TESTS "Register quantum-qec qualification" OFF)
if(ESHKOL_BUILD_TESTS AND ESHKOL_QUANTUM_QEC_TESTS AND TARGET eshkol-run)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    find_program(ESHKOL_QEC_LEAN_EXECUTABLE lean)
    if(NOT ESHKOL_QEC_LEAN_EXECUTABLE)
        message(FATAL_ERROR "QEC qualification requires an explicit Lean executable")
    endif()
    add_test(NAME quantum-qec_native
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/repro/quantum-qec/gate.py"
            --eshkol-bin $<TARGET_FILE:eshkol-run> --lean "${ESHKOL_QEC_LEAN_EXECUTABLE}")
    set_tests_properties(quantum-qec_native PROPERTIES TIMEOUT 1500 LABELS "bounded;math;proof")
    add_test(NAME quantum-qec_readiness
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/repro/quantum-qec/self_test.py")
    set_tests_properties(quantum-qec_readiness PROPERTIES TIMEOUT 45 LABELS "bounded;math;proof")
endif()
