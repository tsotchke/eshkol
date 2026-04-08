if(NOT DEFINED ESHKOL_REQUIRED_LLVM_MAJOR OR ESHKOL_REQUIRED_LLVM_MAJOR STREQUAL "")
    set(ESHKOL_REQUIRED_LLVM_MAJOR 21 CACHE STRING
        "Required LLVM major version for lite/native builds")
endif()

function(eshkol_validate_llvm_major llvm_version llvm_source)
    string(REGEX MATCH "^([0-9]+)" _llvm_major_match "${llvm_version}")

    if(NOT CMAKE_MATCH_1 STREQUAL "${ESHKOL_REQUIRED_LLVM_MAJOR}")
        message(FATAL_ERROR
            "Expected LLVM ${ESHKOL_REQUIRED_LLVM_MAJOR}, got ${llvm_version} from ${llvm_source}")
    endif()
endfunction()

function(eshkol_find_lite_llvm)
    if(WIN32)
        if(NOT LLVM_DIR)
            set(_llvm_package_hint_dirs "")

            if(DEFINED ENV{LLVM_DIR} AND NOT "$ENV{LLVM_DIR}" STREQUAL "")
                list(APPEND _llvm_package_hint_dirs "$ENV{LLVM_DIR}")
            endif()
            if(DEFINED ENV{LLVM_HOME} AND NOT "$ENV{LLVM_HOME}" STREQUAL "")
                list(APPEND _llvm_package_hint_dirs "$ENV{LLVM_HOME}")
            endif()

            file(GLOB _llvm_sdk_hint_dirs LIST_DIRECTORIES TRUE
                "C:/src/llvm-${ESHKOL_REQUIRED_LLVM_MAJOR}*"
                "C:/src/clang+llvm-${ESHKOL_REQUIRED_LLVM_MAJOR}*"
            )
            list(APPEND _llvm_package_hint_dirs
                ${_llvm_sdk_hint_dirs}
                "C:/Program Files/LLVM"
                "C:/LLVM"
            )

            foreach(_llvm_hint IN LISTS _llvm_package_hint_dirs)
                file(TO_CMAKE_PATH "${_llvm_hint}" _llvm_hint_normalized)
                if(EXISTS "${_llvm_hint_normalized}/LLVMConfig.cmake")
                    set(LLVM_DIR "${_llvm_hint_normalized}")
                    break()
                endif()
                if(EXISTS "${_llvm_hint_normalized}/lib/cmake/llvm/LLVMConfig.cmake")
                    set(LLVM_DIR "${_llvm_hint_normalized}/lib/cmake/llvm")
                    break()
                endif()
            endforeach()
        endif()

        if(NOT LLVM_DIR)
            message(FATAL_ERROR
                "LLVM ${ESHKOL_REQUIRED_LLVM_MAJOR} CMake package not found. Install the official LLVM ${ESHKOL_REQUIRED_LLVM_MAJOR} Windows SDK and set LLVM_DIR or LLVM_HOME.")
        endif()

        set(LLVM_DIR "${LLVM_DIR}" CACHE PATH "Path to the LLVM CMake package" FORCE)
        set(LLVM_DIR "${LLVM_DIR}" PARENT_SCOPE)
        return()
    endif()

    if(NOT LLVM_CONFIG_EXECUTABLE)
        set(_llvm_hint_dirs "")

        if(APPLE)
            list(APPEND _llvm_hint_dirs
                "/opt/homebrew/opt/llvm@${ESHKOL_REQUIRED_LLVM_MAJOR}/bin"
                "/usr/local/opt/llvm@${ESHKOL_REQUIRED_LLVM_MAJOR}/bin"
            )
        elseif(WIN32)
            if(DEFINED ENV{LLVM_HOME} AND NOT "$ENV{LLVM_HOME}" STREQUAL "")
                list(APPEND _llvm_hint_dirs "$ENV{LLVM_HOME}/bin")
            endif()
            if(DEFINED ENV{LLVM_DIR} AND NOT "$ENV{LLVM_DIR}" STREQUAL "")
                list(APPEND _llvm_hint_dirs "$ENV{LLVM_DIR}/../bin")
            endif()

            file(GLOB _llvm_sdk_hint_dirs LIST_DIRECTORIES TRUE
                "C:/src/llvm-${ESHKOL_REQUIRED_LLVM_MAJOR}*/bin"
                "C:/src/clang+llvm-${ESHKOL_REQUIRED_LLVM_MAJOR}*/bin"
            )
            list(APPEND _llvm_hint_dirs
                ${_llvm_sdk_hint_dirs}
                C:/Program Files/LLVM/bin
                C:/LLVM/bin
            )
        else()
            list(APPEND _llvm_hint_dirs
                "/usr/lib/llvm-${ESHKOL_REQUIRED_LLVM_MAJOR}/bin"
                "/usr/local/lib/llvm-${ESHKOL_REQUIRED_LLVM_MAJOR}/bin"
                /usr/local/bin
                /usr/bin
            )
        endif()

        find_program(LLVM_CONFIG_EXECUTABLE
            NAMES "llvm-config-${ESHKOL_REQUIRED_LLVM_MAJOR}" llvm-config
            HINTS ${_llvm_hint_dirs}
            NO_DEFAULT_PATH
        )

        if(NOT LLVM_CONFIG_EXECUTABLE)
            find_program(LLVM_CONFIG_EXECUTABLE
                NAMES "llvm-config-${ESHKOL_REQUIRED_LLVM_MAJOR}" llvm-config
            )
        endif()
    endif()

    if(NOT LLVM_CONFIG_EXECUTABLE)
        message(FATAL_ERROR
            "LLVM ${ESHKOL_REQUIRED_LLVM_MAJOR} llvm-config not found. Install LLVM ${ESHKOL_REQUIRED_LLVM_MAJOR} and set LLVM_CONFIG_EXECUTABLE.")
    endif()

    execute_process(
        COMMAND "${LLVM_CONFIG_EXECUTABLE}" --version
        OUTPUT_VARIABLE ESHKOL_LLVM_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    eshkol_validate_llvm_major("${ESHKOL_LLVM_VERSION}" "${LLVM_CONFIG_EXECUTABLE}")

    set(LLVM_CONFIG_EXECUTABLE "${LLVM_CONFIG_EXECUTABLE}" PARENT_SCOPE)
    set(ESHKOL_LLVM_VERSION "${ESHKOL_LLVM_VERSION}" PARENT_SCOPE)
endfunction()
