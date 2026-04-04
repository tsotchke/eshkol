set(ESHKOL_EXPECTED_LLVM_MAJOR 21)

function(eshkol_find_lite_llvm)
    if(NOT LLVM_CONFIG_EXECUTABLE)
        set(_llvm_hint_dirs "")

        if(APPLE)
            list(APPEND _llvm_hint_dirs
                /opt/homebrew/opt/llvm@21/bin
                /usr/local/opt/llvm@21/bin
            )
        elseif(WIN32)
            if(DEFINED ENV{MSYSTEM_PREFIX} AND NOT "$ENV{MSYSTEM_PREFIX}" STREQUAL "")
                list(APPEND _llvm_hint_dirs "$ENV{MSYSTEM_PREFIX}/bin")
            endif()
            list(APPEND _llvm_hint_dirs C:/msys64/mingw64/bin)
        else()
            list(APPEND _llvm_hint_dirs
                /usr/lib/llvm-21/bin
                /usr/local/lib/llvm-21/bin
                /usr/local/bin
                /usr/bin
            )
        endif()

        find_program(LLVM_CONFIG_EXECUTABLE
            NAMES llvm-config-21 llvm-config
            HINTS ${_llvm_hint_dirs}
            NO_DEFAULT_PATH
        )

        if(NOT LLVM_CONFIG_EXECUTABLE)
            find_program(LLVM_CONFIG_EXECUTABLE
                NAMES llvm-config-21 llvm-config
            )
        endif()
    endif()

    if(NOT LLVM_CONFIG_EXECUTABLE)
        message(FATAL_ERROR
            "LLVM 21 llvm-config not found. Install LLVM 21 and set LLVM_CONFIG_EXECUTABLE.")
    endif()

    execute_process(
        COMMAND "${LLVM_CONFIG_EXECUTABLE}" --version
        OUTPUT_VARIABLE ESHKOL_LLVM_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    string(REGEX MATCH "^([0-9]+)" _llvm_major_match "${ESHKOL_LLVM_VERSION}")

    if(NOT CMAKE_MATCH_1 STREQUAL "${ESHKOL_EXPECTED_LLVM_MAJOR}")
        message(FATAL_ERROR
            "Expected LLVM ${ESHKOL_EXPECTED_LLVM_MAJOR}, got ${ESHKOL_LLVM_VERSION} from ${LLVM_CONFIG_EXECUTABLE}")
    endif()

    set(LLVM_CONFIG_EXECUTABLE "${LLVM_CONFIG_EXECUTABLE}" PARENT_SCOPE)
    set(ESHKOL_LLVM_VERSION "${ESHKOL_LLVM_VERSION}" PARENT_SCOPE)
endfunction()
