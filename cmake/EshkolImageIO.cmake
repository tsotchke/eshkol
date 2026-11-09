# Eshkol's image I/O capability. Included by the top-level build and by the
# configure contract test (tests/toolchain/test_native_image_io_contract.py),
# so both exercise this production logic.
#
# Image decoding and encoding are implemented inside the runtime
# (lib/core/image_codecs.h: PNG, JPEG, BMP, WebP), identically on every platform.
# No system or third-party codec library is searched for or linked: the
# module deliberately calls no find_package()/find_library()/pkg-config, and
# adds nothing to ESHKOL_EXTRA_LINK_LIBS.
option(ESHKOL_REQUIRE_IMAGE_IO
    "Fail configuration when image read/write is unavailable (always available with the built-in codecs)" OFF)

set(ESHKOL_IMAGE_IO_BACKEND "BUILTIN")
set(ESHKOL_IMAGE_IO_FORMATS "PNG;JPEG;BMP;WebP")
string(JOIN "/" _eshkol_image_io_format_label ${ESHKOL_IMAGE_IO_FORMATS})
message(STATUS
    "Image I/O: built-in runtime codecs (${_eshkol_image_io_format_label}); "
    "no external codec library is used or linked")

set(ESHKOL_IMAGE_IO_BACKEND "${ESHKOL_IMAGE_IO_BACKEND}" CACHE INTERNAL
    "Resolved image I/O backend" FORCE)

if(ESHKOL_REQUIRE_IMAGE_IO AND NOT ESHKOL_IMAGE_IO_BACKEND STREQUAL "BUILTIN")
    message(FATAL_ERROR
        "ESHKOL_REQUIRE_IMAGE_IO=ON requires the built-in image codecs.")
endif()
message(STATUS "Image I/O capability: ${ESHKOL_IMAGE_IO_BACKEND}")
