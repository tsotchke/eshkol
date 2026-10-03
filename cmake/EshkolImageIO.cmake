# Select Eshkol's native image I/O implementation and enforce an optional
# configure-time requirement. Included by the top-level build and configure
# contract tests so both exercise this production logic.
option(ESHKOL_REQUIRE_IMAGE_IO
    "Fail configuration when no native image read/write backend is available" OFF)

if(APPLE)
    find_library(ESHKOL_IMAGEIO_FRAMEWORK ImageIO REQUIRED)
    find_library(ESHKOL_COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)
    find_library(ESHKOL_COREFOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
    add_compile_definitions(ESHKOL_IMAGE_IO_APPLE=1)
    set(ESHKOL_IMAGE_IO_BACKEND "APPLE")
    list(APPEND ESHKOL_EXTRA_LINK_LIBS
        "-framework ImageIO"
        "-framework CoreGraphics"
        "-framework CoreFoundation")
    message(STATUS "Image I/O: Apple ImageIO/CoreGraphics")
elseif(WIN32)
    add_compile_definitions(ESHKOL_IMAGE_IO_GDIPLUS=1)
    set(ESHKOL_IMAGE_IO_BACKEND "GDIPLUS")
    list(APPEND ESHKOL_EXTRA_LINK_LIBS Gdiplus.lib)
    message(STATUS "Image I/O: Windows GDI+")
else()
    include(FindPkgConfig)
    find_package(PNG QUIET)
    if(PNG_FOUND)
        find_package(JPEG QUIET)
        pkg_check_modules(WEBP QUIET libwebp)

        add_compile_definitions(ESHKOL_IMAGE_IO_LIBPNG=1)
        set(ESHKOL_IMAGE_IO_BACKEND "LIBPNG")
        include_directories(SYSTEM ${PNG_INCLUDE_DIRS})
        list(APPEND ESHKOL_EXTRA_LINK_LIBS PNG::PNG)

        set(_eshkol_image_io_backends "libpng")
        if(JPEG_FOUND)
            add_compile_definitions(ESHKOL_IMAGE_IO_LIBJPEG=1)
            include_directories(SYSTEM ${JPEG_INCLUDE_DIRS})
            list(APPEND ESHKOL_EXTRA_LINK_LIBS JPEG::JPEG)
            list(APPEND _eshkol_image_io_backends "libjpeg")
        endif()
        if(WEBP_FOUND)
            add_compile_definitions(ESHKOL_IMAGE_IO_LIBWEBP=1)
            include_directories(SYSTEM ${WEBP_INCLUDE_DIRS})
            list(APPEND ESHKOL_EXTRA_LINK_LIBS ${WEBP_LDFLAGS})
            list(APPEND _eshkol_image_io_backends "libwebp")
        endif()
        string(JOIN "/" _eshkol_image_io_backend_label ${_eshkol_image_io_backends})
        message(STATUS "Image I/O: ${_eshkol_image_io_backend_label}")
    else()
        set(ESHKOL_IMAGE_IO_BACKEND "NONE")
        message(STATUS
            "Image I/O: native codec backends disabled "
            "(install libpng-dev/libjpeg-dev/libwebp-dev to enable Linux image read/write)")
    endif()
endif()

set(ESHKOL_IMAGE_IO_BACKEND "${ESHKOL_IMAGE_IO_BACKEND}" CACHE INTERNAL
    "Resolved native image I/O backend" FORCE)

if(ESHKOL_REQUIRE_IMAGE_IO AND ESHKOL_IMAGE_IO_BACKEND STREQUAL "NONE")
    message(FATAL_ERROR
        "ESHKOL_REQUIRE_IMAGE_IO=ON requires a native image I/O backend; "
        "install libpng-dev on Linux or configure on a platform with ImageIO/GDI+.")
endif()
message(STATUS "Image I/O capability: ${ESHKOL_IMAGE_IO_BACKEND}")
