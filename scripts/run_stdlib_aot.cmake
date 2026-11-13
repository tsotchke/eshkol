# Compile and run one library suite without platform-shell quoting.
#
# Required: ESHKOL_RUN, ESHKOL_SOURCE, ESHKOL_BINARY, ESHKOL_LINK_DIR.
# Optional:
#   ESHKOL_VM               run the suite as bytecode on this VM instead of
#                           compiling it natively
#   ESHKOL_VERDICT          the verdict line the suite must print, exactly one
#                           of "PASS" (the default, `RESULT: PASS`) or
#                           "ALL PASS" (`RESULT: ALL PASS`, for suites that
#                           report a catalogue); any other value is refused, so
#                           a caller can select a verdict but never widen one
#   ESHKOL_COMPILE_TIMEOUT  seconds for the compile step (default 720)
#   ESHKOL_RUN_TIMEOUT      seconds for the run step (default 120)
#
# A suite passes only when it compiles cleanly, exits 0, prints its selected
# verdict on a line of its own and prints no failure marker. A native compile
# also requires the build tree's own stdlib.o and stdlib.bc in ESHKOL_LINK_DIR,
# so a suite never links an installed standard library by accident.
foreach(required ESHKOL_RUN ESHKOL_SOURCE ESHKOL_BINARY ESHKOL_LINK_DIR)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "Missing ${required}")
    endif()
endforeach()
if(NOT DEFINED ESHKOL_VERDICT)
    set(ESHKOL_VERDICT "PASS")
endif()
if(NOT ESHKOL_VERDICT STREQUAL "PASS" AND NOT ESHKOL_VERDICT STREQUAL "ALL PASS")
    message(FATAL_ERROR "Unsupported library verdict: ${ESHKOL_VERDICT}")
endif()
if(NOT DEFINED ESHKOL_COMPILE_TIMEOUT)
    set(ESHKOL_COMPILE_TIMEOUT 720)
endif()
if(NOT DEFINED ESHKOL_RUN_TIMEOUT)
    set(ESHKOL_RUN_TIMEOUT 120)
endif()
if(DEFINED ESHKOL_VM)
    execute_process(COMMAND "${ESHKOL_RUN}" --profile hosted-vm --emit-eskb "${ESHKOL_BINARY}" "${ESHKOL_SOURCE}" "-L${ESHKOL_LINK_DIR}"
        RESULT_VARIABLE compile_status OUTPUT_VARIABLE compile_output ERROR_VARIABLE compile_error TIMEOUT ${ESHKOL_COMPILE_TIMEOUT})
else()
    include("${CMAKE_CURRENT_LIST_DIR}/require_local_stdlib.cmake")
    execute_process(COMMAND "${ESHKOL_RUN}" -O 1 -o "${ESHKOL_BINARY}" "${ESHKOL_SOURCE}" "-L${ESHKOL_LINK_DIR}"
        RESULT_VARIABLE compile_status OUTPUT_VARIABLE compile_output ERROR_VARIABLE compile_error TIMEOUT ${ESHKOL_COMPILE_TIMEOUT})
endif()
# A compile stopped at its deadline cannot release the scratch object it
# reserved beside its output, and the compiler never overwrites a file it did
# not create. This runner owns the output location, so it clears the empty
# reservation of the compile it stopped; the next run starts from a clean slot.
if(compile_status MATCHES "timeout" AND EXISTS "${ESHKOL_BINARY}.tmp.o")
    file(SIZE "${ESHKOL_BINARY}.tmp.o" _reserved_size)
    if(_reserved_size EQUAL 0)
        file(REMOVE "${ESHKOL_BINARY}.tmp.o")
    endif()
endif()
message("${compile_output}${compile_error}")
if(NOT compile_status STREQUAL "0" OR
   "${compile_output}${compile_error}" MATCHES "FAIL:|RESULT: FAIL|LLVM module verification failed|fatal signal")
    message(FATAL_ERROR "Library AOT compilation failed: ${compile_status}")
endif()
if(DEFINED ESHKOL_VM)
    execute_process(COMMAND "${ESHKOL_VM}" "${ESHKOL_BINARY}" RESULT_VARIABLE run_status
        OUTPUT_VARIABLE run_output ERROR_VARIABLE run_error TIMEOUT ${ESHKOL_RUN_TIMEOUT})
else()
    execute_process(COMMAND "${ESHKOL_BINARY}" RESULT_VARIABLE run_status
        OUTPUT_VARIABLE run_output ERROR_VARIABLE run_error TIMEOUT ${ESHKOL_RUN_TIMEOUT})
endif()
message("${run_output}${run_error}")
if(NOT run_status STREQUAL "0")
    message(FATAL_ERROR "Library AOT execution failed: ${run_status}")
endif()
if(NOT run_output MATCHES "(^|\n)RESULT: ${ESHKOL_VERDICT}(\r?\n|$)" OR
   "${run_output}${run_error}" MATCHES "FAIL:|RESULT: FAIL|fatal signal")
    message(FATAL_ERROR "Library AOT suite lacks a clean explicit verdict")
endif()
