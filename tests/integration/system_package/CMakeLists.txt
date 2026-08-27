# tests/integration/system_package — a from-scratch consumer project.
#
# This is deliberately its OWN top-level CMake project, not a subdirectory
# pulled into Eshkol's own build: it exercises the same thing a downstream
# project (Noesis or otherwise) does — configuring against an INSTALLED
# Eshkol with no access to the source tree's in-tree `eshkol-run`/
# `eshkol-runtime` targets, discovering everything purely through
# find_package(Eshkol) + the packaged cmake/ modules.
#
# Invoked by scripts/run_system_package_integration_test.sh, which points
# CMAKE_PREFIX_PATH / Eshkol_ROOT at a staged (or real) installed prefix.
cmake_minimum_required(VERSION 3.20)
project(eshkol_system_package_integration_test CXX)

list(APPEND CMAKE_MODULE_PATH "${ESHKOL_CMAKE_DIR}")
find_package(Eshkol REQUIRED)
include(EshkolCompile)

eshkol_compile_executable(system_package_hello
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/hello.esk
)
