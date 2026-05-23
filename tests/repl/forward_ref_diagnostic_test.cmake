if(NOT DEFINED ESHKOL_RUN OR ESHKOL_RUN STREQUAL "")
    message(FATAL_ERROR "ESHKOL_RUN is required")
endif()
if(NOT DEFINED ESHKOL_BUILD_DIR OR ESHKOL_BUILD_DIR STREQUAL "")
    message(FATAL_ERROR "ESHKOL_BUILD_DIR is required")
endif()
if(NOT DEFINED ESHKOL_FORWARD_REF_WORKDIR OR ESHKOL_FORWARD_REF_WORKDIR STREQUAL "")
    message(FATAL_ERROR "ESHKOL_FORWARD_REF_WORKDIR is required")
endif()

file(REMOVE_RECURSE "${ESHKOL_FORWARD_REF_WORKDIR}")
file(MAKE_DIRECTORY "${ESHKOL_FORWARD_REF_WORKDIR}/build-codex-verify/generated")
file(WRITE
    "${ESHKOL_FORWARD_REF_WORKDIR}/build-codex-verify/generated/provider.esk"
    "(provide missing-forward-ref)\n(define (missing-forward-ref x) x)\n")

execute_process(
    COMMAND "${ESHKOL_RUN}" -e "(missing-forward-ref 1)" "-L${ESHKOL_BUILD_DIR}"
    WORKING_DIRECTORY "${ESHKOL_FORWARD_REF_WORKDIR}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
    TIMEOUT 120
)

if(result EQUAL 0)
    message(FATAL_ERROR "forward-reference expression unexpectedly succeeded\n${stdout}\n${stderr}")
endif()

if(NOT stderr MATCHES "forward-referenced")
    message(FATAL_ERROR "forward-reference diagnostic missing expected text\n${stdout}\n${stderr}")
endif()

if(stderr MATCHES "build-codex-verify" OR stderr MATCHES "Likely missing")
    message(FATAL_ERROR "forward-reference scanner used generated build-tree hint\n${stdout}\n${stderr}")
endif()
