add_executable(runtime_string_byte_keys_test string_keys_test.cpp)
target_compile_features(runtime_string_byte_keys_test PRIVATE cxx_std_17)
target_include_directories(runtime_string_byte_keys_test PRIVATE ${PROJECT_SOURCE_DIR}/inc)
eshkol_target_llvm_includes(runtime_string_byte_keys_test)
if(LLVM_LIBRARY_DIRS)
    target_link_directories(runtime_string_byte_keys_test PRIVATE ${LLVM_LIBRARY_DIRS})
endif()
target_link_libraries(runtime_string_byte_keys_test PRIVATE eshkol-static
    ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
add_test(NAME runtime_string_byte_keys_test COMMAND runtime_string_byte_keys_test)
set_tests_properties(runtime_string_byte_keys_test PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION "string_keys: 616 checks, 0 failures"
    FAIL_REGULAR_EXPRESSION "FAIL:|fatal signal")

if(TARGET eshkol-run)
    add_test(NAME string_byte_keys_jit
        COMMAND $<TARGET_FILE:eshkol-run> -O3 -r ${CMAKE_CURRENT_SOURCE_DIR}/string_keys_test.esk
            -L${PROJECT_BINARY_DIR})
    add_test(NAME string_byte_keys_aot
        COMMAND ${CMAKE_COMMAND}
            -DESHKOL_RUN=$<TARGET_FILE:eshkol-run>
            -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/string_keys_test.esk
            -DOUT=${CMAKE_CURRENT_BINARY_DIR}/string_byte_keys_aot
            -DLIBDIR=${PROJECT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_aot.cmake)
    # Compilation/JIT has no elapsed cutoff; the AOT driver's runtime is bounded.
    set_tests_properties(string_byte_keys_jit string_byte_keys_aot PROPERTIES
        TIMEOUT 0
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ENVIRONMENT "ESHKOL_PATH=${PROJECT_SOURCE_DIR}/lib;ESHKOL_JIT_CACHE=0"
        PASS_REGULAR_EXPRESSION "string_keys: 23 checks, 0 failures"
        FAIL_REGULAR_EXPRESSION "FAIL:|fatal signal|LLVM module verification failed")
endif()
