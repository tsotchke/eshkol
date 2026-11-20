# EshkolPlatform.cmake — the hosted platform layer (ADR-0053): windows and
# events, real-time audio and MIDI (inc/eshkol/platform/, lib/platform/).
#
# Included twice from CMakeLists.txt:
#   ESHKOL_PLATFORM_STAGE=sources  appends the layer to ESHKOL_RUNTIME_SUPPORT_SRC
#   ESHKOL_PLATFORM_STAGE=tests    registers the platform CTests
#
# Every native backend binds its system library at run time (dlopen), so this
# file adds no link dependency beyond the dynamic loader and pthreads that the
# runtime already uses. The layer is compiled on POSIX hosts; Win32 windows,
# WASAPI and WinMM backends are Planned.

if(ESHKOL_PLATFORM_STAGE STREQUAL "sources")
    set(ESHKOL_PLATFORM_LAYER_ENABLED FALSE)
    if(UNIX AND NOT EMSCRIPTEN)
        set(ESHKOL_PLATFORM_LAYER_ENABLED TRUE)
        set(ESHKOL_PLATFORM_SOURCES
            lib/platform/platform.c
            lib/platform/event/event_queue.c
            lib/platform/window/window.c
            lib/platform/window/window_null.c
            lib/platform/window/window_x11.c
            lib/platform/audio/audio.c
            lib/platform/audio/audio_null.c
            lib/platform/audio/audio_alsa.c
            lib/platform/audio/audio_coreaudio.c
            lib/platform/midi/midi.c
            lib/platform/midi/midi_loopback.c
            lib/platform/midi/midi_alsa.c
            lib/platform/midi/midi_coremidi.c)
        if(APPLE)
            list(APPEND ESHKOL_PLATFORM_SOURCES lib/platform/window/window_cocoa.m)
            # Objective-C, manual reference counting; AppKit is resolved at run time.
            set_source_files_properties(lib/platform/window/window_cocoa.m PROPERTIES
                LANGUAGE C COMPILE_FLAGS "-x objective-c -fno-objc-arc")
        endif()
        list(APPEND ESHKOL_RUNTIME_SUPPORT_SRC ${ESHKOL_PLATFORM_SOURCES})
        # Lets the JIT symbol table (lib/repl/repl_jit.cpp) export the bridge.
        add_compile_definitions(ESHKOL_PLATFORM_LAYER=1)
        if(EXISTS "/usr/include/X11/Xlib.h" OR APPLE)
            set(_eshkol_platform_display "offscreen + X11/Cocoa (run-time loaded)")
        else()
            set(_eshkol_platform_display "offscreen (Xlib headers not found)")
        endif()
        message(STATUS "Platform layer: ${_eshkol_platform_display}; audio null + ALSA/CoreAudio; MIDI loopback + ALSA seq/CoreMIDI")
    else()
        message(STATUS "Platform layer: not compiled on this target (Win32/WASAPI/WinMM Planned)")
    endif()
endif()

if(ESHKOL_PLATFORM_STAGE STREQUAL "tests" AND ESHKOL_BUILD_TESTS AND ESHKOL_PLATFORM_LAYER_ENABLED)
    find_package(Threads REQUIRED)
    set(_platform_test_sources tests/platform/platform_test.c ${ESHKOL_PLATFORM_SOURCES}
        lib/core/runtime_handle.cpp)
    if(APPLE)
        list(APPEND _platform_test_sources tests/platform/cocoa_event_inject.m)
        set_source_files_properties(tests/platform/cocoa_event_inject.m PROPERTIES
            LANGUAGE C COMPILE_FLAGS "-x objective-c -fno-objc-arc")
    endif()
    add_executable(platform_test ${_platform_test_sources})
    target_include_directories(platform_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    target_link_libraries(platform_test PRIVATE Threads::Threads ${CMAKE_DL_LIBS} m)
    if(APPLE)
        target_link_libraries(platform_test PRIVATE "-framework CoreFoundation"
            "-framework CoreGraphics" "-framework Foundation" objc)
    endif()

    # Headless: offscreen windows, null audio clock, MIDI loopback, bridge.
    add_test(NAME platform_headless COMMAND platform_test headless)
    set_tests_properties(platform_headless PROPERTIES TIMEOUT 60
        PASS_REGULAR_EXPRESSION "PASS: platform headless [0-9]+ checks"
        FAIL_REGULAR_EXPRESSION "FAIL:")

    # X11 against a virtual X server when xvfb-run exists; otherwise the
    # binary uses $DISPLAY or reports a skip.
    find_program(ESHKOL_XVFB_RUN xvfb-run)
    if(ESHKOL_XVFB_RUN AND NOT APPLE)
        add_test(NAME platform_x11 COMMAND ${ESHKOL_XVFB_RUN} -a -s "-screen 0 640x480x24"
            $<TARGET_FILE:platform_test> x11)
    else()
        add_test(NAME platform_x11 COMMAND platform_test x11)
    endif()
    set_tests_properties(platform_x11 PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77
        FAIL_REGULAR_EXPRESSION "FAIL:")

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_test(NAME platform_alsa COMMAND platform_test alsa)
        set_tests_properties(platform_alsa PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77
            FAIL_REGULAR_EXPRESSION "FAIL:")
    endif()

    # Eshkol language surface, JIT and AOT, on the headless backends.
    if(TARGET eshkol-run)
        foreach(_platform_esk window audio midi)
            set(_src "${CMAKE_CURRENT_SOURCE_DIR}/tests/platform/platform_${_platform_esk}_test.esk")
            set(_env "ESHKOL_PATH=${CMAKE_CURRENT_SOURCE_DIR}/lib;ESHKOL_JIT_CACHE=0;ESHKOL_PLATFORM_DISPLAY=offscreen;ESHKOL_PLATFORM_AUDIO=null;ESHKOL_PLATFORM_MIDI=loopback")
            add_test(NAME platform_${_platform_esk}_esk_jit
                COMMAND $<TARGET_FILE:eshkol-run> -r "${_src}")
            add_test(NAME platform_${_platform_esk}_esk_aot COMMAND sh -c
                "'$<TARGET_FILE:eshkol-run>' -O 1 -o '${CMAKE_CURRENT_BINARY_DIR}/platform_${_platform_esk}_aot' '${_src}' '-L${CMAKE_CURRENT_BINARY_DIR}' && '${CMAKE_CURRENT_BINARY_DIR}/platform_${_platform_esk}_aot'")
            set_tests_properties(platform_${_platform_esk}_esk_jit platform_${_platform_esk}_esk_aot
                PROPERTIES TIMEOUT 180
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                ENVIRONMENT "${_env}"
                PASS_REGULAR_EXPRESSION "platform ${_platform_esk}: [0-9]+ passed, 0 failed"
                FAIL_REGULAR_EXPRESSION "FAIL:|Unhandled exception|LLVM module verification failed|fatal signal")
        endforeach()
    endif()
endif()
