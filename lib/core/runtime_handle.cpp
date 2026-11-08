#include "eshkol/core/handle.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <vector>

namespace {
struct Entry {
    uint32_t generation = 1;
    eshkol_handle_kind_t kind = ESHKOL_HANDLE_KIND_FILE;
    eshkol_handle_state_t state = ESHKOL_HANDLE_STATE_INVALID;
    uintptr_t native_value = 0;
    eshkol_handle_close_fn close_fn = nullptr;
    void *user_data = nullptr;
    uint64_t lease = 0;
};

constexpr uint32_t kMaxSlots = std::numeric_limits<uint32_t>::max() - 1;

eshkol_handle_id_t make_id(uint32_t slot, uint32_t generation) {
    const uint64_t encoded_slot = static_cast<uint64_t>(slot) + 1;
    return (static_cast<uint64_t>(generation) << 32) | encoded_slot;
}

bool decode_id(eshkol_handle_id_t id, uint32_t *slot, uint32_t *generation) {
    const uint32_t encoded_slot = static_cast<uint32_t>(id);
    const uint32_t encoded_generation = static_cast<uint32_t>(id >> 32);
    if (encoded_slot == 0 || encoded_generation == 0) {
        return false;
    }
    *slot = encoded_slot - 1;
    *generation = encoded_generation;
    return true;
}

bool valid_kind(eshkol_handle_kind_t kind) {
    return kind == ESHKOL_HANDLE_KIND_FILE ||
           kind == ESHKOL_HANDLE_KIND_SOCKET ||
           kind == ESHKOL_HANDLE_KIND_EVENT_LOOP ||
           kind == ESHKOL_HANDLE_KIND_WINDOW ||
           kind == ESHKOL_HANDLE_KIND_AUDIO_DEVICE ||
           kind == ESHKOL_HANDLE_KIND_MIDI_PORT;
}
} // namespace

struct eshkol_handle_registry {
    std::mutex mutex;
    std::vector<Entry> entries;
    bool destroying = false;
    uint64_t next_lease = 1;
};

namespace {
int find_entry(eshkol_handle_registry_t *registry, eshkol_handle_id_t id,
               Entry **out_entry) {
    uint32_t slot;
    uint32_t generation;
    if (!decode_id(id, &slot, &generation) || slot >= registry->entries.size()) {
        return ESHKOL_HANDLE_ENOENT;
    }

    Entry &entry = registry->entries[slot];
    if (entry.generation != generation ||
        entry.state == ESHKOL_HANDLE_STATE_INVALID ||
        entry.state == ESHKOL_HANDLE_STATE_CLOSED) {
        return ESHKOL_HANDLE_ENOENT;
    }
    *out_entry = &entry;
    return ESHKOL_HANDLE_OK;
}

void retire_entry(Entry &entry) {
    entry.native_value = 0;
    entry.close_fn = nullptr;
    entry.user_data = nullptr;
    entry.lease = 0;
    entry.state = ESHKOL_HANDLE_STATE_CLOSED;
}
} // namespace

extern "C" eshkol_handle_registry_t *eshkol_handle_registry_create(void) {
    try {
        return new eshkol_handle_registry_t();
    } catch (...) {
        return nullptr;
    }
}

extern "C" int eshkol_handle_registry_destroy(eshkol_handle_registry_t *registry) {
    if (!registry) {
        return ESHKOL_HANDLE_EINVAL;
    }

    {
        std::lock_guard<std::mutex> lock(registry->mutex);
        if (registry->destroying) {
            return ESHKOL_HANDLE_EBUSY;
        }
        for (const Entry &entry : registry->entries) {
            if (entry.state == ESHKOL_HANDLE_STATE_BORROWED ||
                entry.state == ESHKOL_HANDLE_STATE_CLOSING) {
                return ESHKOL_HANDLE_EBUSY;
            }
        }

        registry->destroying = true;
    }

    // The destruction gate prevents adoption, so the slot count cannot change.
    // Walk existing storage without allocating a second shutdown work list.
    bool close_failed = false;
    for (size_t slot = 0; slot < registry->entries.size(); ++slot) {
        Entry pending_close;
        {
            std::lock_guard<std::mutex> lock(registry->mutex);
            Entry &entry = registry->entries[slot];
            if (entry.state != ESHKOL_HANDLE_STATE_OPEN) {
                continue;
            }
            entry.state = ESHKOL_HANDLE_STATE_CLOSING;
            pending_close = entry;
        }
        try {
            if (pending_close.close_fn(pending_close.kind,
                                       pending_close.native_value,
                                       pending_close.user_data) != 0) {
                close_failed = true;
            }
        } catch (...) {
            close_failed = true;
        }

        std::lock_guard<std::mutex> lock(registry->mutex);
        retire_entry(registry->entries[slot]);
    }

    delete registry;
    return close_failed ? ESHKOL_HANDLE_ECLOSE : ESHKOL_HANDLE_OK;
}

extern "C" int eshkol_handle_create(eshkol_handle_registry_t *registry,
                                    eshkol_handle_kind_t kind,
                                    uintptr_t native_value,
                                    eshkol_handle_close_fn close_fn,
                                    void *user_data,
                                    eshkol_handle_id_t *out_id) {
    if (!registry || !out_id || !close_fn || !valid_kind(kind)) {
        return ESHKOL_HANDLE_EINVAL;
    }

    try {
        std::lock_guard<std::mutex> lock(registry->mutex);
        if (registry->destroying) {
            return ESHKOL_HANDLE_EBUSY;
        }

        for (uint32_t slot = 0; slot < registry->entries.size(); ++slot) {
            Entry &entry = registry->entries[slot];
            if (entry.state != ESHKOL_HANDLE_STATE_CLOSED ||
                entry.generation == std::numeric_limits<uint32_t>::max()) {
                continue;
            }

            ++entry.generation;
            entry.kind = kind;
            entry.native_value = native_value;
            entry.close_fn = close_fn;
            entry.user_data = user_data;
            entry.lease = 0;
            entry.state = ESHKOL_HANDLE_STATE_OPEN;
            *out_id = make_id(slot, entry.generation);
            return ESHKOL_HANDLE_OK;
        }

        if (registry->entries.size() >= kMaxSlots) {
            return ESHKOL_HANDLE_ENOMEM;
        }
        Entry entry;
        entry.kind = kind;
        entry.native_value = native_value;
        entry.close_fn = close_fn;
        entry.user_data = user_data;
        entry.state = ESHKOL_HANDLE_STATE_OPEN;
        registry->entries.push_back(entry);
        const uint32_t slot = static_cast<uint32_t>(registry->entries.size() - 1);
        *out_id = make_id(slot, entry.generation);
        return ESHKOL_HANDLE_OK;
    } catch (...) {
        return ESHKOL_HANDLE_ENOMEM;
    }
}

extern "C" int eshkol_handle_borrow(eshkol_handle_registry_t *registry,
                                    eshkol_handle_id_t id,
                                    uintptr_t *out_native_value,
                                    uint64_t *out_lease) {
    if (!registry || !out_native_value || !out_lease) {
        return ESHKOL_HANDLE_EINVAL;
    }

    std::lock_guard<std::mutex> lock(registry->mutex);
    if (registry->destroying) {
        return ESHKOL_HANDLE_EBUSY;
    }
    Entry *entry;
    const int status = find_entry(registry, id, &entry);
    if (status != ESHKOL_HANDLE_OK) {
        return status;
    }
    if (entry->state != ESHKOL_HANDLE_STATE_OPEN) {
        return ESHKOL_HANDLE_EBUSY;
    }
    if (registry->next_lease == 0) {
        return ESHKOL_HANDLE_EEXHAUSTED;
    }

    const uint64_t lease = registry->next_lease;
    registry->next_lease = lease == std::numeric_limits<uint64_t>::max()
                               ? 0
                               : lease + 1;
    entry->state = ESHKOL_HANDLE_STATE_BORROWED;
    entry->lease = lease;
    *out_native_value = entry->native_value;
    *out_lease = lease;
    return ESHKOL_HANDLE_OK;
}

extern "C" int eshkol_handle_release(eshkol_handle_registry_t *registry,
                                     eshkol_handle_id_t id,
                                     uint64_t lease) {
    if (!registry || lease == 0) {
        return ESHKOL_HANDLE_EINVAL;
    }

    std::lock_guard<std::mutex> lock(registry->mutex);
    if (registry->destroying) {
        return ESHKOL_HANDLE_EBUSY;
    }

    Entry *entry;
    const int status = find_entry(registry, id, &entry);
    if (status != ESHKOL_HANDLE_OK) {
        return status;
    }
    if (entry->state != ESHKOL_HANDLE_STATE_BORROWED || entry->lease != lease) {
        return ESHKOL_HANDLE_ESTATE;
    }
    entry->lease = 0;
    entry->state = ESHKOL_HANDLE_STATE_OPEN;
    return ESHKOL_HANDLE_OK;
}

extern "C" int eshkol_handle_close(eshkol_handle_registry_t *registry,
                                   eshkol_handle_id_t id) {
    if (!registry) {
        return ESHKOL_HANDLE_EINVAL;
    }

    uintptr_t native_value;
    void *user_data;
    eshkol_handle_close_fn close_fn;
    eshkol_handle_kind_t kind;
    {
        std::lock_guard<std::mutex> lock(registry->mutex);
        if (registry->destroying) {
            return ESHKOL_HANDLE_EBUSY;
        }

        Entry *entry;
        const int status = find_entry(registry, id, &entry);
        if (status != ESHKOL_HANDLE_OK) {
            return status;
        }
        if (entry->state != ESHKOL_HANDLE_STATE_OPEN) {
            return ESHKOL_HANDLE_EBUSY;
        }
        entry->state = ESHKOL_HANDLE_STATE_CLOSING;
        native_value = entry->native_value;
        user_data = entry->user_data;
        close_fn = entry->close_fn;
        kind = entry->kind;
    }

    bool close_failed = false;
    try {
        close_failed = close_fn(kind, native_value, user_data) != 0;
    } catch (...) {
        close_failed = true;
    }

    {
        std::lock_guard<std::mutex> lock(registry->mutex);
        Entry *entry;
        if (find_entry(registry, id, &entry) == ESHKOL_HANDLE_OK) {
            retire_entry(*entry);
        }
    }
    return close_failed ? ESHKOL_HANDLE_ECLOSE : ESHKOL_HANDLE_OK;
}

extern "C" int eshkol_handle_get_state(eshkol_handle_registry_t *registry,
                                       eshkol_handle_id_t id,
                                       eshkol_handle_state_t *out_state) {
    if (!registry || !out_state) {
        return ESHKOL_HANDLE_EINVAL;
    }

    std::lock_guard<std::mutex> lock(registry->mutex);
    uint32_t slot;
    uint32_t generation;
    if (!decode_id(id, &slot, &generation) || slot >= registry->entries.size() ||
        registry->entries[slot].generation != generation ||
        registry->entries[slot].state == ESHKOL_HANDLE_STATE_INVALID) {
        return ESHKOL_HANDLE_ENOENT;
    }
    *out_state = registry->entries[slot].state;
    return ESHKOL_HANDLE_OK;
}
