# Compile and run one library suite without platform-shell quoting.
foreach(required ESHKOL_RUN ESHKOL_SOURCE ESHKOL_BINARY ESHKOL_LINK_DIR)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "Missing ${required}")
    endif()
endforeach()
execute_process(COMMAND "${ESHKOL_RUN}" -O 1 -o "${ESHKOL_BINARY}" "${ESHKOL_SOURCE}" "-L${ESHKOL_LINK_DIR}"
    RESULT_VARIABLE compile_status OUTPUT_VARIABLE compile_output ERROR_VARIABLE compile_error TIMEOUT 720)
message("${compile_output}${compile_error}")
if(NOT compile_status STREQUAL "0" OR
   "${compile_output}${compile_error}" MATCHES "FAIL:|RESULT: FAIL|LLVM module verification failed|fatal signal")
    message(FATAL_ERROR "Library AOT compilation failed: ${compile_status}")
endif()
execute_process(COMMAND "${ESHKOL_BINARY}" RESULT_VARIABLE run_status
    OUTPUT_VARIABLE run_output ERROR_VARIABLE run_error TIMEOUT 120)
message("${run_output}${run_error}")
if(NOT run_status STREQUAL "0")
    message(FATAL_ERROR "Library AOT execution failed: ${run_status}")
endif()
if(NOT run_output MATCHES "(^|\n)RESULT: PASS(\r?\n|$)" OR
   "${run_output}${run_error}" MATCHES "FAIL:|RESULT: FAIL|fatal signal")
    message(FATAL_ERROR "Library AOT suite lacks a clean explicit verdict")
endif()
