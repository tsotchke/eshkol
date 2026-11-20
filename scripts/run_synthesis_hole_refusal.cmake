# A program with a ?? hole that has no well-typed completion must not compile:
# the compiler exits non-zero, names the hole, and writes no binary.
# -DESHKOL_RUN=<eshkol-run> -DESHKOL_SOURCE=<.esk> -DESHKOL_BINARY=<path> -DESHKOL_LINK_DIR=<dir>
file(REMOVE "${ESHKOL_BINARY}")
execute_process(COMMAND "${ESHKOL_RUN}" "${ESHKOL_SOURCE}" -o "${ESHKOL_BINARY}" "-L${ESHKOL_LINK_DIR}"
    RESULT_VARIABLE status OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 600)
set(all "${out}${err}")
if(status STREQUAL "0")
    message(FATAL_ERROR "compile succeeded for a program with an unfillable hole:\n${all}")
endif()
if(EXISTS "${ESHKOL_BINARY}")
    message(FATAL_ERROR "a binary was written for a program with an unfillable hole")
endif()
if(NOT all MATCHES "synthesis hole \\?\\? in 'f': no well-typed completion of type")
    message(FATAL_ERROR "the hole was not named in the diagnostic:\n${all}")
endif()
message("PASS: synthesis_hole_refusal")
