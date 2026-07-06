# ESH-0215: end-to-end verification of cmake/EshkolCompile.cmake's real
# dependency tracking.  build_integration_demo compiles main.esk, which
# (load)s dep.esk.  The bug this fixture exercises: previously the custom
# command that compiles main.esk -> main.esk.o only DEPENDS'd on main.esk
# itself, so editing dep.esk left the object (and the linked executable)
# stale until a downstream user ran `rm` + `ninja` two or three times
# (Noesis BUGS-2026-07-04 #3/#5). eshkol-run's new --emit-depfile now lets
# the custom command's DEPFILE track dep.esk too, so a single `ninja`
# invocation after editing dep.esk recompiles and relinks.
#
# This directory is only added when the in-tree eshkol-run/eshkol-runtime
# targets exist (see the ESHKOL_BUILD_TESTS guard in the top-level
# CMakeLists.txt) — it links directly against the vendored eshkol-runtime
# target so it inherits the exact runtime/BLAS/GPU link flags the real
# build already computed, rather than re-deriving them.

include(${CMAKE_SOURCE_DIR}/cmake/EshkolCompile.cmake)

eshkol_compile_executable(build_integration_demo
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.esk
  LINK_LIBRARIES eshkol-runtime
)
