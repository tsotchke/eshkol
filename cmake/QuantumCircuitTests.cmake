# Portable source-module gate: does not require the opt-in Moonlab integration.
option(ESHKOL_QUANTUM_CIRCUIT_QISKIT_ORACLE
       "Enable independent OpenQASM parser/Qiskit verification (Python extras required)" OFF)
if(ESHKOL_BUILD_TESTS AND TARGET eshkol-run)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME quantum_circuit_export_driver_selftest
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/quantum/circuit_export_driver.py --self-test)
    set_tests_properties(quantum_circuit_export_driver_selftest PROPERTIES
        LABELS "quantum;regression;harness" TIMEOUT 30
        FAIL_REGULAR_EXPRESSION "FAIL:")
    add_test(NAME quantum_circuit_export_aot
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/quantum/circuit_export_driver.py
            --eshkol-run $<TARGET_FILE:eshkol-run>
            --repo-root ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(quantum_circuit_export_aot PROPERTIES
        LABELS "quantum;aot;regression" TIMEOUT 180
        FAIL_REGULAR_EXPRESSION "FAIL:")
    if(ESHKOL_QUANTUM_CIRCUIT_QISKIT_ORACLE)
        # The driver fails closed if any oracle dependency is missing.
        add_test(NAME quantum_circuit_export_qiskit_oracle
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/quantum/circuit_export_driver.py
                --eshkol-run $<TARGET_FILE:eshkol-run>
                --repo-root ${CMAKE_CURRENT_SOURCE_DIR} --qiskit-oracle)
        set_tests_properties(quantum_circuit_export_qiskit_oracle PROPERTIES
            LABELS "quantum;aot;oracle" TIMEOUT 300
            FAIL_REGULAR_EXPRESSION "FAIL:")
    endif()
endif()
