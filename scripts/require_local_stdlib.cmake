# Require local artifacts to prevent a silent system-library fallback.
# This checks presence and location only; it does not authenticate artifact versions.
if(NOT DEFINED ESHKOL_LINK_DIR OR "${ESHKOL_LINK_DIR}" STREQUAL "")
    message(FATAL_ERROR "Missing ESHKOL_LINK_DIR")
endif()

foreach(artifact stdlib.o stdlib.bc)
    if(NOT EXISTS "${ESHKOL_LINK_DIR}/${artifact}")
        message(FATAL_ERROR
            "Local standard-library artifact is missing: ${ESHKOL_LINK_DIR}/${artifact}; refusing system fallback")
    endif()
endforeach()
