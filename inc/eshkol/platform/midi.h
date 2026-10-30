/**
 * @file midi.h
 * @brief MIDI 1.0 input/output (ADR-0053).
 *
 * Backends: loopback (in-process; always present), ALSA sequencer
 * (libasound loaded at run time), CoreMIDI (loaded at run time).
 *
 * Addressing is by endpoint name on every backend:
 *   - an output port publishes a virtual source called `name` that other
 *     applications and Eshkol input ports can connect to;
 *   - an input port connects to the source called `name` (exact match,
 *     otherwise the first source whose name contains it).
 * Messages are complete MIDI 1.0 messages: a status byte with its data bytes,
 * or a SysEx F0 ... F7 block (running status is expanded by the receiver).
 * Received messages pass from the backend thread to the reader through a
 * lock-free single-producer/single-consumer ring with arrival timestamps
 * (eshkol_platform_time_us()).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_PLATFORM_MIDI_H
#define ESHKOL_PLATFORM_MIDI_H

#include <stdint.h>
#include <eshkol/platform/platform.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct eshkol_midi_port eshkol_midi_port_t;

typedef enum {
    ESHKOL_MIDI_OUTPUT = 0,
    ESHKOL_MIDI_INPUT  = 1
} eshkol_midi_direction_t;

/** Longest message accepted (SysEx included). */
#define ESHKOL_MIDI_MAX_MESSAGE 1024u

typedef struct {
    uint64_t sent;       /**< messages sent */
    uint64_t received;   /**< messages delivered to the reader ring */
    uint64_t dropped;    /**< messages dropped because the ring was full */
    uint64_t malformed;  /**< incoming byte runs that were not complete messages */
} eshkol_midi_stats_t;

eshkol_midi_port_t* eshkol_midi_open_output(const char* name);
eshkol_midi_port_t* eshkol_midi_open_input(const char* source_name);
int eshkol_midi_open_ex(int backend, int direction, const char* name,
                        eshkol_midi_port_t** out);
void eshkol_midi_close(eshkol_midi_port_t* port);

/** Send one complete message; ESHKOL_PLATFORM_EFORMAT when malformed. */
int eshkol_midi_send(eshkol_midi_port_t* port, const uint8_t* bytes, uint32_t len);
/** Pop the next message: returns its length, 0 when none, < 0 on error
 *  (EFULL when `cap` is too small; the message stays queued). */
int eshkol_midi_receive(eshkol_midi_port_t* port, uint8_t* buffer, uint32_t cap,
                        uint64_t* timestamp_us);
/** Wait up to timeout_ms (negative = forever) for a message, then receive. */
int eshkol_midi_receive_timeout(eshkol_midi_port_t* port, uint8_t* buffer,
                                uint32_t cap, uint64_t* timestamp_us,
                                int timeout_ms);
int eshkol_midi_get_stats(eshkol_midi_port_t* port, eshkol_midi_stats_t* out);
int eshkol_midi_backend_of(eshkol_midi_port_t* port);

/** Sources visible on the selected backend. */
int eshkol_midi_source_count(void);
/** Copy source `index`'s name (NUL-terminated) into `buf`. */
int eshkol_midi_source_name(int index, char* buf, uint32_t cap);

/** Expected total length of a message starting with `status` (0 = invalid
 *  status, -1 = variable length SysEx). */
int eshkol_midi_message_length(uint8_t status);

/* ── Eshkol language bridge (generation-checked integer handles) ───── */
int64_t eshkol_plat_midi_open(int32_t direction, const char* name);
int32_t eshkol_plat_midi_close(int64_t handle);
/** Append one byte to the handle's outgoing staging buffer. */
int32_t eshkol_plat_midi_stage(int64_t handle, int32_t byte);
/** Send the staged bytes as one message and clear the staging buffer. */
int32_t eshkol_plat_midi_send_staged(int64_t handle);
/** Receive into the handle's current-message slot: length, 0 none, < 0 status. */
int32_t eshkol_plat_midi_receive(int64_t handle, int32_t timeout_ms);
/** Byte `index` of the current message, -1 out of range. */
int32_t eshkol_plat_midi_byte(int64_t handle, int32_t index);
int64_t eshkol_plat_midi_timestamp(int64_t handle);
/** field: 0 sent, 1 received, 2 dropped, 3 malformed, 4 backend. */
int64_t eshkol_plat_midi_stat(int64_t handle, int32_t field);
int32_t eshkol_plat_midi_source_count(void);
/** Name of source `index` in a thread-local buffer ("" when out of range). */
const char* eshkol_plat_midi_source_name(int32_t index);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PLATFORM_MIDI_H */
