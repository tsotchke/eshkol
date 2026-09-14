# One capability boundary: no ABI mirrors, symbol-name guessing, or runtime
# library substitution. OFF builds expose only the unavailable status surface.
add_library(eshkol-qllm-abi INTERFACE)
if(ESHKOL_QLLM_ENABLED)
    find_path(ESHKOL_QLLM_INCLUDE_DIR
        NAMES semiclassical_qllm/eshkol_bridge.h
        HINTS ${ESHKOL_QLLM_ROOT}/include $ENV{QLLM_ROOT}/include)
    find_library(ESHKOL_QLLM_LIBRARY NAMES semiclassical_qllm
        HINTS ${ESHKOL_QLLM_ROOT}/build/lib ${ESHKOL_QLLM_ROOT}/lib
              $ENV{QLLM_ROOT}/build/lib $ENV{QLLM_ROOT}/lib)
    if(NOT ESHKOL_QLLM_INCLUDE_DIR OR NOT ESHKOL_QLLM_LIBRARY)
        message(FATAL_ERROR "ESHKOL_QLLM_ENABLED requires the real qLLM eshkol_bridge.h and library; set ESHKOL_QLLM_ROOT")
    endif()
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES "${ESHKOL_QLLM_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_LIBRARIES "${ESHKOL_QLLM_LIBRARY}")
    # Repeat on reconfigure: changing a private install must not retain a stale
    # successful link receipt for a different library/header combination.
    unset(ESHKOL_QLLM_TYPED_ABI_LINKS CACHE)
    check_cxx_source_compiles([=[
        #include <semiclassical_qllm/eshkol_bridge.h>
        int main() {
            qllm_tensor_options_t options = qllm_tensor_options_default(QLLM_DEVICE_CPU);
            size_t shape[] = {1};
            qllm_tensor_t* tensor = qllm_tensor_create(1, shape, &options);
            qllm_eshkol_const_entry_t constant = {0};
            unsigned char* buffer = NULL; size_t size = 0;
            qllm_eshkol_tensor_to_eskb_chunk_typed(tensor, &constant, 1,
                                                 "main", &buffer, &size);
            qllm_tensor_destroy(tensor);
            int64_t result = 0;
            qllm_eshkol_eval_to_int64("1", &result);
            return qllm_eshkol_register_qllm_natives();
        }
    ]=] ESHKOL_QLLM_TYPED_ABI_LINKS)
    cmake_pop_check_state()
    if(NOT ESHKOL_QLLM_TYPED_ABI_LINKS)
        message(FATAL_ERROR "qLLM headers/library do not link the required native-registration and typed ESKB ABI")
    endif()
    target_compile_definitions(eshkol-qllm-abi INTERFACE ESHKOL_HAS_QLLM=1)
    target_include_directories(eshkol-qllm-abi INTERFACE "${ESHKOL_QLLM_INCLUDE_DIR}")
    target_link_libraries(eshkol-qllm-abi INTERFACE "${ESHKOL_QLLM_LIBRARY}")
endif()

# qLLM may contain a vendored Eshkol JIT. Its C ABI does not expose which
# LLVM C++ ABI that JIT uses, so inspect the actual shared dependency before
# linking it into a second Eshkol compiler. A tensor-only test binary has no
# host LLVM and does not need this check.
function(eshkol_qllm_check_host_llvm host_version)
    if(NOT ESHKOL_QLLM_ENABLED)
        return()
    endif()
    if(APPLE)
        execute_process(COMMAND otool -L "${ESHKOL_QLLM_LIBRARY}"
            OUTPUT_VARIABLE dependencies RESULT_VARIABLE inspect_status)
        string(REGEX MATCH "[^ \t\n]+/libLLVM[^ \t\n]*" llvm_library "${dependencies}")
    elseif(UNIX)
        execute_process(COMMAND ldd "${ESHKOL_QLLM_LIBRARY}"
            OUTPUT_VARIABLE dependencies RESULT_VARIABLE inspect_status)
        string(REGEX MATCH "/[^ \t\n]+/libLLVM[^ \t\n]*" llvm_library "${dependencies}")
    else()
        message(FATAL_ERROR "qLLM JIT compatibility cannot be proved on this platform; ESHKOL_QLLM_ENABLED cannot link into the host compiler")
    endif()
    if(NOT inspect_status EQUAL 0)
        message(FATAL_ERROR "Cannot inspect qLLM shared dependencies to prove host LLVM compatibility: ${ESHKOL_QLLM_LIBRARY}")
    endif()
    if(NOT llvm_library)
        message(FATAL_ERROR "qLLM exposes no inspectable LLVM shared dependency; host JIT ABI compatibility is unproved")
    endif()
    get_filename_component(llvm_lib_dir "${llvm_library}" DIRECTORY)
    execute_process(COMMAND "${llvm_lib_dir}/../bin/llvm-config" --version
        OUTPUT_VARIABLE qllm_llvm_version OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE version_status)
    if(NOT version_status EQUAL 0)
        message(FATAL_ERROR "Cannot prove qLLM LLVM version for ${llvm_library}; matching llvm-config is required beside the linked LLVM installation")
    endif()
    string(REGEX MATCH "^[0-9]+" qllm_major "${qllm_llvm_version}")
    string(REGEX MATCH "^[0-9]+" host_major "${host_version}")
    if(NOT host_major OR NOT qllm_major STREQUAL host_major)
        message(FATAL_ERROR "qLLM JIT LLVM ${qllm_llvm_version} conflicts with Eshkol LLVM ${host_version}; rebuild qLLM against the host toolchain")
    endif()
    message(STATUS "qLLM JIT LLVM compatibility: ${qllm_llvm_version} / host ${host_version}")
endfunction()
