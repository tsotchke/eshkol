# ─────────────────────────────────────────────────────────────────────────
# FindEshkol.cmake — canonical downstream CMake discovery module.
#
# This is the ONE Find module Eshkol ships and packages (homebrew + the
# GitHub release tarball both install it under
# <prefix>/share/eshkol/cmake/FindEshkol.cmake), so a downstream project's
# own guess at library names can never drift from the real, current
# packaging layout the way a hand-rolled Find module can. Add the installed
# cmake directory to CMAKE_MODULE_PATH and call find_package(Eshkol):
#
#   list(APPEND CMAKE_MODULE_PATH "<prefix>/share/eshkol/cmake")
#   find_package(Eshkol REQUIRED)
#   include(EshkolCompile)   # eshkol_compile_executable() / eshkol_compile_library()
#
# Result variables:
#   Eshkol_FOUND             — TRUE if a usable install was located
#   Eshkol_VERSION           — version reported by `eshkol-run --version`
#   Eshkol_COMPILER          — path to the eshkol-run driver
#   Eshkol_LIBRARY           — path to the runtime archive (libeshkol-runtime.a /
#                              eshkol-runtime.lib) that a COMPILED PROGRAM links
#                              against — NOT the compiler/tool aggregate (see
#                              "Two archives, two audiences" below)
#   Eshkol_STDLIB_OBJECT     — path to stdlib.o (see "Every program needs
#                              stdlib.o" below)
#   Eshkol_STDLIB_MODULE_DIR — <prefix>/share/eshkol/lib, the (require ...)/
#                              (import ...) module-source root
#
# Imported target:
#   Eshkol::eshkol           — link this from a target holding .o files that
#                              eshkol-run --emit-object produced. Carries the
#                              runtime archive, stdlib.o, and (on Apple) the
#                              system frameworks the runtime needs as its
#                              INTERFACE_LINK_LIBRARIES, so one
#                              target_link_libraries(... Eshkol::eshkol) is a
#                              complete, working link line — see
#                              cmake/EshkolCompile.cmake, which links every
#                              eshkol_compile_executable()/eshkol_compile_library()
#                              target against this automatically when it
#                              exists.
#
# Hints (CMake variable or environment variable):
#   Eshkol_ROOT / ESHKOL_ROOT — override search root (checked first)
#
# ── Two archives, two audiences ─────────────────────────────────────────
# The package ships TWO static archives that are easy to confuse because
# both spell "eshkol" and both exist under lib/ and lib/eshkol/:
#
#   eshkol-runtime  — the lean hosted runtime (arena, tagged values, printer,
#                      AD tower) that CODE eshkol-run COMPILES links against.
#                      This is Eshkol_LIBRARY / Eshkol::eshkol.
#   eshkol-static   — the compiler/tool aggregate (parser, HoTT checker,
#                      LLVM codegen, the whole eshkol-run binary's own
#                      internals) meant for embedding the COMPILER itself,
#                      not for linking a generated program.
#
# A find_library() call that accepts bare names like "eshkol-static" or
# "libeshkol" as a stand-in for "the Eshkol library" can silently resolve to
# the wrong one — eshkol-static is really the compiler-and-a-half archive
# and a normal downstream program should never need it. This module only
# ever looks for eshkol-runtime under its own name for Eshkol_LIBRARY.
#
# ── Every program needs stdlib.o ────────────────────────────────────────
# Every eshkol-run-compiled program calls __eshkol_lib_init__ from its
# native `main`, whether or not it `(require stdlib)` — codegen only ever
# DEFINES that symbol inside stdlib.o. A downstream link line that carries
# the runtime archive but not stdlib.o fails at link time with an undefined
# `__eshkol_lib_init__` reference no matter how trivial the compiled
# program is. Eshkol::eshkol's INTERFACE_LINK_LIBRARIES therefore always
# includes Eshkol_STDLIB_OBJECT, unconditionally.
# ─────────────────────────────────────────────────────────────────────────
include(FindPackageHandleStandardArgs)

set(_eshkol_search_roots)
foreach(_v Eshkol_ROOT ESHKOL_ROOT)
  if(DEFINED ${_v} AND NOT "${${_v}}" STREQUAL "")
    list(APPEND _eshkol_search_roots "${${_v}}")
  endif()
  if(DEFINED ENV{${_v}} AND NOT "$ENV{${_v}}" STREQUAL "")
    list(APPEND _eshkol_search_roots "$ENV{${_v}}")
  endif()
endforeach()

# Same system prefixes lib/core/platform_runtime.cpp's install_library_roots()
# / install_module_roots() fall back to at runtime — kept in sync so a
# find_package(Eshkol) success implies eshkol-run's OWN resolution will also
# succeed from the same prefix, and vice versa.
list(APPEND _eshkol_search_roots
  /usr/local
  /usr
  /opt/homebrew)

find_program(Eshkol_COMPILER
  NAMES eshkol-run
  HINTS ${_eshkol_search_roots}
  PATH_SUFFIXES bin
  DOC "Eshkol compiler driver (eshkol-run)")

find_library(Eshkol_LIBRARY
  NAMES eshkol-runtime
  HINTS ${_eshkol_search_roots}
  PATH_SUFFIXES lib lib/eshkol
  DOC "Eshkol hosted runtime archive (links a COMPILED PROGRAM, not the compiler)")

find_file(Eshkol_STDLIB_OBJECT
  NAMES stdlib.o
  HINTS ${_eshkol_search_roots}
  PATH_SUFFIXES lib lib/eshkol
  DOC "Precompiled stdlib object — every compiled program's main() calls into it")

find_file(Eshkol_STDLIB_BITCODE
  NAMES stdlib.bc
  HINTS ${_eshkol_search_roots}
  PATH_SUFFIXES lib lib/eshkol
  DOC "Portable LLVM bitcode form of the stdlib (JIT / -r path)")

find_path(Eshkol_STDLIB_MODULE_DIR
  NAMES stdlib.esk
  HINTS ${_eshkol_search_roots}
  PATH_SUFFIXES share/eshkol/lib
  DOC "Eshkol (require ...)/(import ...) module-source root")

set(Eshkol_VERSION "")
if(Eshkol_COMPILER)
  execute_process(
    COMMAND "${Eshkol_COMPILER}" --version
    OUTPUT_VARIABLE _eshkol_version_out
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
    TIMEOUT 5)
  string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" Eshkol_VERSION "${_eshkol_version_out}")
endif()

find_package_handle_standard_args(Eshkol
  REQUIRED_VARS Eshkol_COMPILER Eshkol_LIBRARY Eshkol_STDLIB_OBJECT Eshkol_STDLIB_MODULE_DIR
  VERSION_VAR   Eshkol_VERSION)

if(Eshkol_FOUND AND NOT TARGET Eshkol::eshkol)
  add_library(Eshkol::eshkol UNKNOWN IMPORTED)
  set_target_properties(Eshkol::eshkol PROPERTIES
    IMPORTED_LOCATION "${Eshkol_LIBRARY}")

  set(_eshkol_interface_link_libraries "${Eshkol_STDLIB_OBJECT}")

  if(APPLE)
    # The default (no GPU/quantum/external-BLAS) build's system-framework
    # closure, verified directly against a fresh default build's link step —
    # see docs/BUILD_INTEGRATION.md. A build with ESHKOL_GPU_BACKEND,
    # ESHKOL_QUANTUM_ENABLED, or an external BLAS turned on links additional
    # libraries this module does not know about; that combination is not
    # covered by the packaged release build and is a documented limitation,
    # not silently unsupported.
    foreach(_eshkol_framework IN ITEMS
        Accelerate Metal MetalPerformanceShaders Foundation
        Security CoreFoundation ImageIO CoreGraphics)
      find_library(_eshkol_${_eshkol_framework}_framework NAMES ${_eshkol_framework})
      if(_eshkol_${_eshkol_framework}_framework)
        list(APPEND _eshkol_interface_link_libraries "${_eshkol_${_eshkol_framework}_framework}")
      endif()
    endforeach()
    list(APPEND _eshkol_interface_link_libraries "-lobjc")
  elseif(WIN32)
    list(APPEND _eshkol_interface_link_libraries bcrypt ws2_32)
  endif()

  set_target_properties(Eshkol::eshkol PROPERTIES
    INTERFACE_LINK_LIBRARIES "${_eshkol_interface_link_libraries}")
endif()

mark_as_advanced(
  Eshkol_COMPILER
  Eshkol_LIBRARY
  Eshkol_STDLIB_OBJECT
  Eshkol_STDLIB_BITCODE
  Eshkol_STDLIB_MODULE_DIR)
