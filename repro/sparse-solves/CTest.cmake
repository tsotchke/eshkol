# Selected MIT contribution: bounded standalone qualification.
option(ESHKOL_SPARSE_SOLVES_TESTS "Register sparse-solves qualification" OFF)
if(ESHKOL_BUILD_TESTS AND ESHKOL_SPARSE_SOLVES_TESTS AND TARGET eshkol-run)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME sparse-solves_native
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/repro/sparse-solves/gate.py"
            --eshkol-bin $<TARGET_FILE:eshkol-run>)
    set_tests_properties(sparse-solves_native PROPERTIES TIMEOUT 1500 LABELS "bounded;math;proof")
    add_test(NAME sparse-solves_readiness
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/repro/sparse-solves/self_test.py")
    set_tests_properties(sparse-solves_readiness PROPERTIES TIMEOUT 45 LABELS "bounded;math;proof")
endif()
