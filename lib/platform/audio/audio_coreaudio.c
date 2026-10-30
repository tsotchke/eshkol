/*
 * audio_coreaudio.c — CoreAudio backend (ADR-0053).
 *
 * Output uses the default-output AudioUnit; input uses the HAL AudioUnit on
 * the default input device. AudioToolbox and CoreAudio are bound at run time,
 * so the runtime's macOS link line is unchanged. CoreAudio's own real-time
 * I/O thread invokes the render/input callback, which runs
 * eshkol_plat_audio_process() directly on the device buffer (output) or on a
 * preallocated capture buffer filled by AudioUnitRender() (input).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include "../platform_internal.h"

#if defined(__APPLE__)

#include <AudioToolbox/AudioToolbox.h>
#include <CoreAudio/CoreAudio.h>
#include <stdlib.h>
#include <string.h>

static struct {
    __typeof__(&AudioComponentFindNext) AudioComponentFindNext;
    __typeof__(&AudioComponentInstanceNew) AudioComponentInstanceNew;
    __typeof__(&AudioComponentInstanceDispose) AudioComponentInstanceDispose;
    __typeof__(&AudioUnitSetProperty) AudioUnitSetProperty;
    __typeof__(&AudioUnitGetProperty) AudioUnitGetProperty;
    __typeof__(&AudioUnitInitialize) AudioUnitInitialize;
    __typeof__(&AudioUnitUninitialize) AudioUnitUninitialize;
    __typeof__(&AudioOutputUnitStart) AudioOutputUnitStart;
    __typeof__(&AudioOutputUnitStop) AudioOutputUnitStop;
    __typeof__(&AudioUnitRender) AudioUnitRender;
} T;
static const char* const k_toolbox_symbols[] = {
    "AudioComponentFindNext", "AudioComponentInstanceNew", "AudioComponentInstanceDispose",
    "AudioUnitSetProperty", "AudioUnitGetProperty", "AudioUnitInitialize",
    "AudioUnitUninitialize", "AudioOutputUnitStart", "AudioOutputUnitStop",
    "AudioUnitRender", NULL
};
static struct {
    __typeof__(&AudioObjectGetPropertyData) AudioObjectGetPropertyData;
} H;
static const char* const k_hal_symbols[] = { "AudioObjectGetPropertyData", NULL };

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static int g_loaded = 0;

static void ca_load(void) {
    static const char* const toolbox[] = {
        "/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox", NULL };
    static const char* const hal[] = {
        "/System/Library/Frameworks/CoreAudio.framework/CoreAudio", NULL };
    g_loaded = eshkol_plat_dlsym_all(eshkol_plat_dlopen_any(toolbox), k_toolbox_symbols,
                                     (void**)&T) &&
               eshkol_plat_dlsym_all(eshkol_plat_dlopen_any(hal), k_hal_symbols, (void**)&H);
}

static int ca_available(void) {
    pthread_once(&g_once, ca_load);
    return g_loaded;
}

#define CAPTURE_MAX_FRAMES 16384u

typedef struct {
    AudioComponentInstance unit;
    AudioDeviceID device;
    uint8_t* capture;          /* input only */
} ca_t;

static OSStatus render_cb(void* ref, AudioUnitRenderActionFlags* flags,
                          const AudioTimeStamp* ts, UInt32 bus, UInt32 frames,
                          AudioBufferList* io) {
    (void)flags; (void)ts; (void)bus;
    eshkol_audio_device_t* d = (eshkol_audio_device_t*)ref;
    if (!io || io->mNumberBuffers < 1) return noErr;
    uint32_t cap = io->mBuffers[0].mDataByteSize / d->frame_bytes;
    eshkol_plat_audio_process(d, (uint8_t*)io->mBuffers[0].mData, frames < cap ? frames : cap);
    return noErr;
}

static OSStatus input_cb(void* ref, AudioUnitRenderActionFlags* flags,
                         const AudioTimeStamp* ts, UInt32 bus, UInt32 frames,
                         AudioBufferList* unused) {
    (void)unused;
    eshkol_audio_device_t* d = (eshkol_audio_device_t*)ref;
    ca_t* c = (ca_t*)d->impl;
    if (frames > CAPTURE_MAX_FRAMES) frames = CAPTURE_MAX_FRAMES;
    AudioBufferList abl;
    abl.mNumberBuffers = 1;
    abl.mBuffers[0].mNumberChannels = d->config.channels;
    abl.mBuffers[0].mDataByteSize = frames * d->frame_bytes;
    abl.mBuffers[0].mData = c->capture;
    OSStatus st = T.AudioUnitRender(c->unit, flags, ts, bus, frames, &abl);
    if (st != noErr) {
        atomic_fetch_add_explicit(&d->device_xruns, 1u, memory_order_relaxed);
        return st;
    }
    eshkol_plat_audio_process(d, c->capture, frames);
    return noErr;
}

static void fill_asbd(const eshkol_audio_config_t* cfg, AudioStreamBasicDescription* a) {
    memset(a, 0, sizeof(*a));
    uint32_t bps = eshkol_audio_format_bytes(cfg->format);
    a->mSampleRate = cfg->sample_rate;
    a->mFormatID = kAudioFormatLinearPCM;
    a->mFormatFlags = kAudioFormatFlagIsPacked;
    if (cfg->format == ESHKOL_AUDIO_FORMAT_F32) a->mFormatFlags |= kAudioFormatFlagIsFloat;
    else if (cfg->format != ESHKOL_AUDIO_FORMAT_U8) a->mFormatFlags |= kAudioFormatFlagIsSignedInteger;
    a->mBitsPerChannel = bps * 8u;
    a->mChannelsPerFrame = cfg->channels;
    a->mFramesPerPacket = 1;
    a->mBytesPerFrame = bps * cfg->channels;
    a->mBytesPerPacket = a->mBytesPerFrame;
}

static UInt32 device_frames_property(AudioDeviceID dev, AudioObjectPropertySelector sel,
                                     AudioObjectPropertyScope scope) {
    AudioObjectPropertyAddress addr = { sel, scope, kAudioObjectPropertyElementMain };
    UInt32 v = 0, size = sizeof(v);
    if (H.AudioObjectGetPropertyData(dev, &addr, 0, NULL, &size, &v) != noErr) return 0;
    return v;
}

static int ca_open(eshkol_audio_device_t* d, const char* device_name) {
    (void)device_name;  /* the system default device; named devices are Planned */
    ca_t* c = (ca_t*)calloc(1, sizeof(*c));
    if (!c) return ESHKOL_PLATFORM_ENOMEM;
    const int input = d->direction == ESHKOL_AUDIO_INPUT;
    AudioComponentDescription desc;
    memset(&desc, 0, sizeof(desc));
    desc.componentType = kAudioUnitType_Output;
    desc.componentSubType = input ? kAudioUnitSubType_HALOutput : kAudioUnitSubType_DefaultOutput;
    desc.componentManufacturer = kAudioUnitManufacturer_Apple;
    AudioComponent comp = T.AudioComponentFindNext(NULL, &desc);
    if (!comp || T.AudioComponentInstanceNew(comp, &c->unit) != noErr) {
        free(c);
        return ESHKOL_PLATFORM_EDEVICE;
    }
    OSStatus st = noErr;
    if (input) {
        UInt32 on = 1, off = 0;
        st |= T.AudioUnitSetProperty(c->unit, kAudioOutputUnitProperty_EnableIO,
                                     kAudioUnitScope_Input, 1, &on, sizeof(on));
        st |= T.AudioUnitSetProperty(c->unit, kAudioOutputUnitProperty_EnableIO,
                                     kAudioUnitScope_Output, 0, &off, sizeof(off));
        AudioObjectPropertyAddress addr = { kAudioHardwarePropertyDefaultInputDevice,
                                            kAudioObjectPropertyScopeGlobal,
                                            kAudioObjectPropertyElementMain };
        UInt32 size = sizeof(c->device);
        st |= H.AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, NULL, &size,
                                           &c->device);
        st |= T.AudioUnitSetProperty(c->unit, kAudioOutputUnitProperty_CurrentDevice,
                                     kAudioUnitScope_Global, 0, &c->device, sizeof(c->device));
    }
    AudioStreamBasicDescription asbd;
    fill_asbd(&d->config, &asbd);
    st |= T.AudioUnitSetProperty(c->unit, kAudioUnitProperty_StreamFormat,
                                 input ? kAudioUnitScope_Output : kAudioUnitScope_Input,
                                 input ? 1 : 0, &asbd, sizeof(asbd));
    UInt32 frames = d->config.buffer_frames;
    (void)T.AudioUnitSetProperty(c->unit, kAudioDevicePropertyBufferFrameSize,
                                 kAudioUnitScope_Global, 0, &frames, sizeof(frames));
    AURenderCallbackStruct cb = { input ? input_cb : render_cb, d };
    st |= T.AudioUnitSetProperty(c->unit,
                                 input ? kAudioOutputUnitProperty_SetInputCallback
                                       : kAudioUnitProperty_SetRenderCallback,
                                 input ? kAudioUnitScope_Global : kAudioUnitScope_Input, 0,
                                 &cb, sizeof(cb));
    if (input) {
        c->capture = (uint8_t*)calloc(CAPTURE_MAX_FRAMES, d->frame_bytes);
        if (!c->capture) st = -1;
    }
    if (st != noErr || T.AudioUnitInitialize(c->unit) != noErr) {
        T.AudioComponentInstanceDispose(c->unit);
        free(c->capture);
        free(c);
        return ESHKOL_PLATFORM_EDEVICE;
    }
    if (!input) {
        UInt32 size = sizeof(c->device);
        (void)T.AudioUnitGetProperty(c->unit, kAudioOutputUnitProperty_CurrentDevice,
                                     kAudioUnitScope_Global, 0, &c->device, &size);
    }
    AudioObjectPropertyScope scope = input ? kAudioObjectPropertyScopeInput
                                           : kAudioObjectPropertyScopeOutput;
    UInt32 granted = device_frames_property(c->device, kAudioDevicePropertyBufferFrameSize,
                                            kAudioObjectPropertyScopeGlobal);
    if (granted >= 16 && granted <= 16384) d->config.buffer_frames = granted;
    UInt32 latency = device_frames_property(c->device, kAudioDevicePropertyLatency, scope) +
                     device_frames_property(c->device, kAudioDevicePropertySafetyOffset, scope) +
                     d->config.buffer_frames;
    atomic_store(&d->device_latency_frames, latency);
    d->impl = c;
    return ESHKOL_PLATFORM_OK;
}

static int ca_start(eshkol_audio_device_t* d) {
    ca_t* c = (ca_t*)d->impl;
    return T.AudioOutputUnitStart(c->unit) == noErr ? ESHKOL_PLATFORM_OK : ESHKOL_PLATFORM_EDEVICE;
}

static int ca_stop(eshkol_audio_device_t* d) {
    ca_t* c = (ca_t*)d->impl;
    /* AudioOutputUnitStop returns after the I/O thread has left the callback. */
    return T.AudioOutputUnitStop(c->unit) == noErr ? ESHKOL_PLATFORM_OK : ESHKOL_PLATFORM_EDEVICE;
}

static void ca_close(eshkol_audio_device_t* d) {
    ca_t* c = (ca_t*)d->impl;
    if (!c) return;
    T.AudioOutputUnitStop(c->unit);
    T.AudioUnitUninitialize(c->unit);
    T.AudioComponentInstanceDispose(c->unit);
    free(c->capture);
    free(c);
    d->impl = NULL;
}

const eshkol_audio_backend_ops_t eshkol_plat_audio_coreaudio_ops = {
    ESHKOL_BACKEND_COREAUDIO, ca_available, ca_open, ca_start, ca_stop, ca_close
};

#endif /* __APPLE__ */
