# Include from a consumer build after selecting an independently trusted cohort.
set(_ESHKOL_CONSUMER_ADMISSION_SCRIPT
    "${CMAKE_CURRENT_LIST_DIR}/../scripts/check_consumer_toolchain.py")

function(eshkol_require_consumer_toolchain)
    cmake_parse_arguments(ADMIT "" "MANIFEST;MANIFEST_SHA256;SOURCE_ROOT;ARTIFACT_ROOT;COMPILER;STDLIB;RUNTIME" "" ${ARGN})
    if(ADMIT_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown consumer admission arguments: ${ADMIT_UNPARSED_ARGUMENTS}")
    endif()
    foreach(_field MANIFEST MANIFEST_SHA256 SOURCE_ROOT ARTIFACT_ROOT)
        if(NOT ADMIT_${_field})
            message(FATAL_ERROR "Consumer admission requires explicit ${_field}; no default cohort")
        endif()
    endforeach()
    set(_selected "")
    if(ADMIT_COMPILER OR ADMIT_STDLIB OR ADMIT_RUNTIME)
        foreach(_field COMPILER STDLIB RUNTIME)
            if(NOT ADMIT_${_field})
                message(FATAL_ERROR "Consumer binding requires explicit ${_field}")
            endif()
        endforeach()
        list(APPEND _selected --compiler-path "${ADMIT_COMPILER}"
            --stdlib-path "${ADMIT_STDLIB}" --runtime-path "${ADMIT_RUNTIME}")
    endif()
    find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${_ESHKOL_CONSUMER_ADMISSION_SCRIPT}"
            --manifest "${ADMIT_MANIFEST}"
            --expected-manifest-sha256 "${ADMIT_MANIFEST_SHA256}"
            --source-root "${ADMIT_SOURCE_ROOT}"
            --artifact-root "${ADMIT_ARTIFACT_ROOT}"
            ${_selected}
        RESULT_VARIABLE _status OUTPUT_VARIABLE _receipt ERROR_VARIABLE _error
        TIMEOUT 120)
    if(NOT _status STREQUAL "0")
        message(FATAL_ERROR "Consumer toolchain admission refused: ${_receipt}${_error}")
    endif()
    set(ESHKOL_CONSUMER_ADMISSION_JSON "${_receipt}" PARENT_SCOPE)
endfunction()
