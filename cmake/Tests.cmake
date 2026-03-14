function(eshkol_add_suite_test suite_name test_glob)
    add_test(
        NAME suite-${suite_name}
        COMMAND
            ${CMAKE_COMMAND}
            -DPROJECT_ROOT=${CMAKE_SOURCE_DIR}
            -DBUILD_DIR=${CMAKE_BINARY_DIR}
            -DEXECUTABLE_SUFFIX=${CMAKE_EXECUTABLE_SUFFIX}
            -DSUITE_NAME=${suite_name}
            -DTEST_GLOB=${test_glob}
            -P ${CMAKE_SOURCE_DIR}/cmake/RunSuite.cmake
    )
    set_tests_properties(suite-${suite_name} PROPERTIES LABELS "suite")
endfunction()

function(eshkol_add_negative_suite_test suite_name test_glob negative_mode)
    add_test(
        NAME suite-${suite_name}
        COMMAND
            ${CMAKE_COMMAND}
            -DPROJECT_ROOT=${CMAKE_SOURCE_DIR}
            -DBUILD_DIR=${CMAKE_BINARY_DIR}
            -DEXECUTABLE_SUFFIX=${CMAKE_EXECUTABLE_SUFFIX}
            -DSUITE_NAME=${suite_name}
            -DTEST_GLOB=${test_glob}
            -DNEGATIVE_MODE=${negative_mode}
            -P ${CMAKE_SOURCE_DIR}/cmake/RunSuite.cmake
    )
    set_tests_properties(suite-${suite_name} PROPERTIES LABELS "suite")
endfunction()

if(ESHKOL_BUILD_TESTS)
    eshkol_add_suite_test(features "tests/features/*.esk")
    eshkol_add_suite_test(stdlib "tests/stdlib/*.esk")
    eshkol_add_suite_test(lists "tests/lists/*.esk")
    eshkol_add_negative_suite_test(memory "tests/memory/*.esk" "compile")
    eshkol_add_negative_suite_test(modules "tests/modules/*.esk" "compile_or_runtime")
    eshkol_add_suite_test(types "tests/types/*.esk")
    eshkol_add_suite_test(autodiff "tests/autodiff/*.esk")
    eshkol_add_suite_test(ml "tests/ml/*.esk")
    eshkol_add_suite_test(neural "tests/neural/*.esk")
    eshkol_add_suite_test(json "tests/json/*.esk")
    eshkol_add_suite_test(system "tests/system/*.esk")

    add_executable(eshkol-hott-types-test tests/types/hott_types_test.cpp)
    target_link_libraries(eshkol-hott-types-test PRIVATE eshkol-static ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    eshkol_apply_llvm_target_settings(eshkol-hott-types-test)
    add_test(NAME cpp-hott-types COMMAND eshkol-hott-types-test)
    set_tests_properties(cpp-hott-types PROPERTIES LABELS "suite;cpp")

    add_executable(eshkol-type-checker-test tests/types/type_checker_test.cpp)
    target_link_libraries(eshkol-type-checker-test PRIVATE eshkol-static ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
    eshkol_apply_llvm_target_settings(eshkol-type-checker-test)
    add_test(NAME cpp-type-checker COMMAND eshkol-type-checker-test)
    set_tests_properties(cpp-type-checker PROPERTIES LABELS "suite;cpp")
endif()
