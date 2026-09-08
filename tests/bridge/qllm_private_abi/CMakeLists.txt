cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(EshkolQllmAbi LANGUAGES CXX)
    enable_testing()
    get_filename_component(ESHKOL_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
    option(ESHKOL_QLLM_ENABLED "Use the real qLLM ABI" OFF)
    set(ESHKOL_QLLM_ROOT "" CACHE PATH "qLLM installation root")
    include("${ESHKOL_SOURCE_ROOT}/cmake/QllmBridge.cmake")
else()
    set(ESHKOL_SOURCE_ROOT "${PROJECT_SOURCE_DIR}")
endif()
add_executable(qllm_private_abi_test
    ../qllm_private_abi_test.cpp
    "${ESHKOL_SOURCE_ROOT}/lib/bridge/qllm_interop.cpp")
target_compile_features(qllm_private_abi_test PRIVATE cxx_std_17)
target_include_directories(qllm_private_abi_test PRIVATE "${ESHKOL_SOURCE_ROOT}/inc")
target_link_libraries(qllm_private_abi_test PRIVATE eshkol-qllm-abi)
add_test(NAME qllm_private_abi COMMAND qllm_private_abi_test)
# This binary links only qLLM's own JIT dependency, so it cannot mix the host
# compiler's LLVM with the private library's LLVM. No mock registrar is used.
if(ESHKOL_QLLM_ENABLED)
    add_test(NAME qllm_private_native_registration COMMAND qllm_private_abi_test --jit)
endif()
