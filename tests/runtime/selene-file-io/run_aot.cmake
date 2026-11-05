foreach(required ESHKOL_RUN SOURCE OUT LIBDIR FIXTURE_LIBDIR)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "Missing ${required}")
    endif()
endforeach()
# Optimized compilation has no elapsed-time kill limit. The runtime child is
# bounded independently and restores its own fault-injection process state.
execute_process(COMMAND "${ESHKOL_RUN}" -O3 -o "${OUT}" "${SOURCE}"
    "-L${LIBDIR}" "-L${FIXTURE_LIBDIR}" -lselene_file_io_faults
    RESULT_VARIABLE build_result)
if(NOT "${build_result}" STREQUAL "0")
    message(FATAL_ERROR "AOT compilation failed: ${build_result}")
endif()
execute_process(COMMAND "${OUT}" TIMEOUT 60
    RESULT_VARIABLE runtime_result OUTPUT_VARIABLE output ERROR_VARIABLE errors)
if(NOT "${runtime_result}" STREQUAL "0")
    message(FATAL_ERROR "AOT runtime failed: ${runtime_result}\n${output}\n${errors}")
endif()
set(runtime_log "${output}\n${errors}")
if(runtime_log MATCHES "FAIL:|fatal signal|LLVM module verification failed")
    message(FATAL_ERROR "AOT runtime emitted a failure marker:\n${runtime_log}")
endif()
string(REPLACE "\r\n" "\n" normalized_output "${output}")
string(REGEX MATCHALL "(^|\n)file_io:[^\n]*" summaries "${normalized_output}")
list(LENGTH summaries summary_count)
if(NOT summary_count EQUAL 1)
    message(FATAL_ERROR "Expected exactly one file_io summary, found ${summary_count}:\n${runtime_log}")
endif()
list(GET summaries 0 summary)
string(REGEX REPLACE "^\n" "" summary "${summary}")
if(NOT summary STREQUAL "file_io: 22 checks, 0 failures")
    message(FATAL_ERROR "Unexpected file_io summary: ${summary}\n${runtime_log}")
endif()
# CTest relies on this driver's exit status, never PASS_REGULAR_EXPRESSION
# (which can override a nonzero process exit). Emit success only after every
# runtime check above, so a post-summary crash cannot appear successful.
message("PASS: selene_file_io_aot (file_io: 22 checks, 0 failures)")
