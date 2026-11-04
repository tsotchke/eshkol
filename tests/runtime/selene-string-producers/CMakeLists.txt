add_executable(runtime_string_producers_test string_producers_native_test.cpp)
target_compile_features(runtime_string_producers_test PRIVATE cxx_std_17)
target_include_directories(runtime_string_producers_test PRIVATE ${PROJECT_SOURCE_DIR}/inc)
eshkol_target_llvm_includes(runtime_string_producers_test)
if(LLVM_LIBRARY_DIRS)
    target_link_directories(runtime_string_producers_test PRIVATE ${LLVM_LIBRARY_DIRS})
endif()
target_link_libraries(runtime_string_producers_test PRIVATE eshkol-static eshkol-agent-ffi
    ${ESHKOL_EXTRA_LINK_LIBS} ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
add_test(NAME runtime_string_producers_test COMMAND runtime_string_producers_test)
set_tests_properties(runtime_string_producers_test PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION "string_producers_native: [0-9]+ checks, 0 failures"
    FAIL_REGULAR_EXPRESSION "FAIL:|fatal signal")

if(TARGET eshkol-run)
    add_test(NAME string_producers_jit
        COMMAND $<TARGET_FILE:eshkol-run> -O3 -r ${CMAKE_CURRENT_SOURCE_DIR}/string_producers_test.esk
            -L${PROJECT_BINARY_DIR})
    add_test(NAME string_producers_aot
        COMMAND ${CMAKE_COMMAND}
            -DESHKOL_RUN=$<TARGET_FILE:eshkol-run>
            -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/string_producers_test.esk
            -DOUT=${CMAKE_CURRENT_BINARY_DIR}/string_producers_aot
            -DLIBDIR=${PROJECT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/../selene-byte-exact/run_aot.cmake)
    set_tests_properties(string_producers_jit string_producers_aot PROPERTIES
        TIMEOUT 0
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ENVIRONMENT "ESHKOL_PATH=${PROJECT_SOURCE_DIR}/lib;ESHKOL_JIT_CACHE=0"
        PASS_REGULAR_EXPRESSION "string_producers: 9 checks, 0 failures"
        FAIL_REGULAR_EXPRESSION "FAIL:|fatal signal|LLVM module verification failed")
endif()
