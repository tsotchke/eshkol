if(NOT DEFINED PROJECT_ROOT OR NOT DEFINED BUILD_DIR OR NOT DEFINED SUITE_NAME OR NOT DEFINED TEST_GLOB OR NOT DEFINED EXECUTABLE_SUFFIX)
    message(FATAL_ERROR "RunSuite.cmake requires PROJECT_ROOT, BUILD_DIR, EXECUTABLE_SUFFIX, SUITE_NAME, and TEST_GLOB")
endif()

if(NOT DEFINED NEGATIVE_MODE)
    set(NEGATIVE_MODE "none")
endif()

set(ESHKOL_RUN "${BUILD_DIR}/eshkol-run${EXECUTABLE_SUFFIX}")
if(NOT EXISTS "${ESHKOL_RUN}")
    set(ESHKOL_RUN "${BUILD_DIR}/eshkol-run")
endif()

if(NOT EXISTS "${ESHKOL_RUN}")
    message(FATAL_ERROR "eshkol-run not found at ${BUILD_DIR}")
endif()

file(GLOB TEST_FILES LIST_DIRECTORIES false "${PROJECT_ROOT}/${TEST_GLOB}")
if(NOT TEST_FILES)
    message(FATAL_ERROR "No tests matched ${TEST_GLOB}")
endif()

set(SUITE_OUTPUT_ROOT "${BUILD_DIR}/suite-outputs/${SUITE_NAME}")
file(MAKE_DIRECTORY "${SUITE_OUTPUT_ROOT}")

function(eshkol_cleanup_outputs output_base)
    set(candidates
        "${output_base}${EXECUTABLE_SUFFIX}"
        "${output_base}.exe"
        "${output_base}"
        "${output_base}${EXECUTABLE_SUFFIX}.tmp.o"
        "${output_base}.exe.tmp.o"
        "${output_base}.tmp.o"
    )
    foreach(path IN LISTS candidates)
        file(REMOVE "${path}")
    endforeach()
endfunction()

function(eshkol_find_output out_var output_base)
    set(candidates
        "${output_base}${EXECUTABLE_SUFFIX}"
        "${output_base}.exe"
        "${output_base}"
    )
    foreach(candidate IN LISTS candidates)
        if(EXISTS "${candidate}")
            set(${out_var} "${candidate}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${out_var} "" PARENT_SCOPE)
endfunction()

set(failures "")
set(pass_count 0)
set(fail_count 0)

message(STATUS "Running Eshkol suite ${SUITE_NAME}")

foreach(test_file IN LISTS TEST_FILES)
    file(RELATIVE_PATH TEST_RELATIVE_PATH "${PROJECT_ROOT}" "${test_file}")
    string(REGEX REPLACE "[^A-Za-z0-9_.-]" "_" TEST_WORK_ID "${TEST_RELATIVE_PATH}")
    set(TEST_WORK_DIR "${SUITE_OUTPUT_ROOT}/${TEST_WORK_ID}")
    set(TEST_OUTPUT_BASE "${TEST_WORK_DIR}/a.out")

    file(REMOVE_RECURSE "${TEST_WORK_DIR}")
    file(MAKE_DIRECTORY "${TEST_WORK_DIR}")
    eshkol_cleanup_outputs("${TEST_OUTPUT_BASE}")

    file(READ "${test_file}" TEST_CONTENTS)
    string(FIND "${TEST_CONTENTS}" ";;; Expected: Error" EXPECTED_ERROR_MARKER)
    set(IS_NEGATIVE FALSE)
    if(NOT EXPECTED_ERROR_MARKER EQUAL -1)
        set(IS_NEGATIVE TRUE)
    endif()

    execute_process(
        COMMAND "${ESHKOL_RUN}" "${test_file}" "-o" "a.out" "-L${BUILD_DIR}"
        WORKING_DIRECTORY "${TEST_WORK_DIR}"
        RESULT_VARIABLE COMPILE_STATUS
        OUTPUT_VARIABLE COMPILE_STDOUT
        ERROR_VARIABLE COMPILE_STDERR
    )
    set(COMPILE_OUTPUT "${COMPILE_STDOUT}\n${COMPILE_STDERR}")

    if(NEGATIVE_MODE STREQUAL "compile" AND IS_NEGATIVE)
        if(COMPILE_STATUS EQUAL 0)
            math(EXPR fail_count "${fail_count} + 1")
            string(APPEND failures "\n${test_file}: expected compile failure but compilation succeeded")
        else()
            math(EXPR pass_count "${pass_count} + 1")
        endif()
        continue()
    endif()

    if(NEGATIVE_MODE STREQUAL "compile_or_runtime" AND IS_NEGATIVE)
        string(FIND "${COMPILE_OUTPUT}" "error:" COMPILE_ERROR_MARKER)
        if(NOT COMPILE_STATUS EQUAL 0 OR NOT COMPILE_ERROR_MARKER EQUAL -1)
            eshkol_find_output(TEST_OUTPUT "${TEST_OUTPUT_BASE}")
            if(TEST_OUTPUT STREQUAL "")
                file(REMOVE_RECURSE "${TEST_WORK_DIR}")
                math(EXPR pass_count "${pass_count} + 1")
                continue()
            endif()

            execute_process(
                COMMAND "${TEST_OUTPUT}"
                WORKING_DIRECTORY "${TEST_WORK_DIR}"
                RESULT_VARIABLE RUN_STATUS
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(RUN_STATUS EQUAL 0)
                math(EXPR fail_count "${fail_count} + 1")
                string(APPEND failures "\n${test_file}: expected failure but runtime exited successfully")
            else()
                file(REMOVE_RECURSE "${TEST_WORK_DIR}")
                math(EXPR pass_count "${pass_count} + 1")
            endif()
            continue()
        endif()

        math(EXPR fail_count "${fail_count} + 1")
        string(APPEND failures "\n${test_file}: expected compile/runtime failure but no error was observed")
        continue()
    endif()

    if(NOT COMPILE_STATUS EQUAL 0)
        math(EXPR fail_count "${fail_count} + 1")
        string(APPEND failures "\n${test_file}: compile failed\n${COMPILE_OUTPUT}")
        continue()
    endif()

    eshkol_find_output(TEST_OUTPUT "${TEST_OUTPUT_BASE}")
    if(TEST_OUTPUT STREQUAL "")
        math(EXPR fail_count "${fail_count} + 1")
        string(APPEND failures "\n${test_file}: compilation succeeded but no output executable was produced")
        continue()
    endif()

    execute_process(
        COMMAND "${TEST_OUTPUT}"
        WORKING_DIRECTORY "${TEST_WORK_DIR}"
        RESULT_VARIABLE RUN_STATUS
        OUTPUT_VARIABLE RUN_STDOUT
        ERROR_VARIABLE RUN_STDERR
    )
    set(RUN_OUTPUT "${RUN_STDOUT}\n${RUN_STDERR}")

    string(FIND "${RUN_OUTPUT}" "error:" RUNTIME_ERROR_MARKER)
    if(NOT RUN_STATUS EQUAL 0 OR NOT RUNTIME_ERROR_MARKER EQUAL -1)
        math(EXPR fail_count "${fail_count} + 1")
        string(APPEND failures "\n${test_file}: runtime failed\n${RUN_OUTPUT}")
        continue()
    endif()

    file(REMOVE_RECURSE "${TEST_WORK_DIR}")
    math(EXPR pass_count "${pass_count} + 1")
endforeach()

if(fail_count GREATER 0)
    message(FATAL_ERROR "Suite ${SUITE_NAME} failed: ${pass_count} passed, ${fail_count} failed${failures}")
endif()

file(REMOVE_RECURSE "${SUITE_OUTPUT_ROOT}")
message(STATUS "Suite ${SUITE_NAME} passed (${pass_count} tests)")
