#include "eshkol/core/handle.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#define CHECK(expression)                                                        \
    do {                                                                         \
        if (!(expression)) {                                                     \
            std::cerr << "FAIL " << __FILE__ << ':' << __LINE__ << ": "       \
                      << #expression << '\n';                                   \
            std::abort();                                                        \
        }                                                                        \
    } while (false)

struct CloseContext {
    eshkol_handle_registry_t *registry = nullptr;
    eshkol_handle_id_t id = 0;
    std::atomic<int> calls{0};
    int state_status = 0;
    int borrow_status = 0;
    bool fail = false;
    bool throw_exception = false;
};

int close_fake(eshkol_handle_kind_t, uintptr_t, void *opaque) {
    auto *context = static_cast<CloseContext *>(opaque);
    ++context->calls;
    if (context->registry && context->id) {
        eshkol_handle_state_t state = ESHKOL_HANDLE_STATE_INVALID;
        context->state_status = eshkol_handle_get_state(context->registry,
                                                        context->id, &state);
        CHECK(context->state_status == ESHKOL_HANDLE_OK);
        CHECK(state == ESHKOL_HANDLE_STATE_CLOSING);
        uintptr_t ignored_value = 0;
        uint64_t ignored_lease = 0;
        context->borrow_status = eshkol_handle_borrow(context->registry,
            context->id, &ignored_value, &ignored_lease);
        CHECK(context->borrow_status == ESHKOL_HANDLE_EBUSY);
    }
    if (context->throw_exception) {
        throw 1;
    }
    return context->fail ? -1 : 0;
}

int close_file(eshkol_handle_kind_t kind, uintptr_t value, void *) {
    CHECK(kind == ESHKOL_HANDLE_KIND_FILE);
    return std::fclose(reinterpret_cast<FILE *>(value));
}

struct BlockingClose {
    std::mutex mutex;
    std::condition_variable changed;
    int calls = 0;
    bool entered = false;
    bool may_return = false;
};

int close_blocked(eshkol_handle_kind_t, uintptr_t, void *opaque) {
    auto *context = static_cast<BlockingClose *>(opaque);
    std::unique_lock<std::mutex> lock(context->mutex);
    ++context->calls;
    context->entered = true;
    context->changed.notify_all();
    context->changed.wait(lock, [&] { return context->may_return; });
    return 0;
}

void check_status(int actual, int expected) {
    CHECK(actual == expected);
}

int main() {
    check_status(eshkol_handle_registry_destroy(nullptr), ESHKOL_HANDLE_EINVAL);
    check_status(eshkol_handle_create(nullptr, ESHKOL_HANDLE_KIND_FILE, 1,
        close_fake, nullptr, nullptr), ESHKOL_HANDLE_EINVAL);

    eshkol_handle_registry_t *registry = eshkol_handle_registry_create();
    CHECK(registry != nullptr);
    eshkol_handle_id_t untouched_id = 0x1234;
    check_status(eshkol_handle_create(registry, ESHKOL_HANDLE_KIND_FILE, 1,
        nullptr, nullptr, &untouched_id), ESHKOL_HANDLE_EINVAL);
    CHECK(untouched_id == 0x1234);
    check_status(eshkol_handle_create(registry,
        ESHKOL_HANDLE_KIND_INVALID, 1, close_fake, nullptr,
        &untouched_id), ESHKOL_HANDLE_EINVAL);
    CHECK(untouched_id == 0x1234);
    check_status(eshkol_handle_create(registry, ESHKOL_HANDLE_KIND_FILE, 1,
        close_fake, nullptr, nullptr), ESHKOL_HANDLE_EINVAL);

    const eshkol_handle_id_t invalid_id = 0;
    uintptr_t untouched_value = 77;
    uint64_t untouched_lease = 88;
    check_status(eshkol_handle_borrow(registry, invalid_id, &untouched_value,
        &untouched_lease), ESHKOL_HANDLE_ENOENT);
    CHECK(untouched_value == 77 && untouched_lease == 88);
    check_status(eshkol_handle_close(registry, invalid_id), ESHKOL_HANDLE_ENOENT);
    check_status(eshkol_handle_release(registry, invalid_id, 1), ESHKOL_HANDLE_ENOENT);
    eshkol_handle_state_t untouched_state = ESHKOL_HANDLE_STATE_OPEN;
    check_status(eshkol_handle_get_state(registry, invalid_id, &untouched_state),
                 ESHKOL_HANDLE_ENOENT);
    CHECK(untouched_state == ESHKOL_HANDLE_STATE_OPEN);
    check_status(eshkol_handle_borrow(registry, invalid_id, nullptr,
        &untouched_lease), ESHKOL_HANDLE_EINVAL);
    check_status(eshkol_handle_borrow(registry, invalid_id, &untouched_value,
        nullptr), ESHKOL_HANDLE_EINVAL);
    check_status(eshkol_handle_release(registry, invalid_id, 0),
                 ESHKOL_HANDLE_EINVAL);

    // Adoption of a real FILE* and the complete borrow/release/close lifecycle.
    FILE *file = std::tmpfile();
    CHECK(file != nullptr);
    eshkol_handle_id_t file_id = 0;
    check_status(eshkol_handle_create(registry, ESHKOL_HANDLE_KIND_FILE,
        reinterpret_cast<uintptr_t>(file), close_file, nullptr, &file_id),
        ESHKOL_HANDLE_OK);
    uintptr_t borrowed_value = 0;
    uint64_t file_lease = 0;
    check_status(eshkol_handle_borrow(registry, file_id, &borrowed_value,
        &file_lease), ESHKOL_HANDLE_OK);
    CHECK(borrowed_value == reinterpret_cast<uintptr_t>(file));
    check_status(eshkol_handle_close(registry, file_id), ESHKOL_HANDLE_EBUSY);
    uintptr_t second_value = 101;
    uint64_t second_lease = 102;
    check_status(eshkol_handle_borrow(registry, file_id, &second_value,
        &second_lease), ESHKOL_HANDLE_EBUSY);
    CHECK(second_value == 101 && second_lease == 102);
    check_status(eshkol_handle_release(registry, file_id, file_lease),
                 ESHKOL_HANDLE_OK);
    check_status(eshkol_handle_release(registry, file_id, file_lease),
                 ESHKOL_HANDLE_ESTATE);
    check_status(eshkol_handle_close(registry, file_id), ESHKOL_HANDLE_OK);
    check_status(eshkol_handle_close(registry, file_id), ESHKOL_HANDLE_ENOENT);
    eshkol_handle_state_t closed_state = ESHKOL_HANDLE_STATE_INVALID;
    check_status(eshkol_handle_get_state(registry, file_id, &closed_state),
                 ESHKOL_HANDLE_OK);
    CHECK(closed_state == ESHKOL_HANDLE_STATE_CLOSED);

    // Reused slot advances generation, making old IDs stale for every operation.
    CloseContext reused;
    eshkol_handle_id_t reused_id = 0;
    check_status(eshkol_handle_create(registry, ESHKOL_HANDLE_KIND_SOCKET, 123,
        close_fake, &reused, &reused_id), ESHKOL_HANDLE_OK);
    CHECK(reused_id != file_id);
    untouched_value = 77;
    untouched_lease = 88;
    check_status(eshkol_handle_borrow(registry, file_id, &untouched_value,
        &untouched_lease), ESHKOL_HANDLE_ENOENT);
    CHECK(untouched_value == 77 && untouched_lease == 88);
    check_status(eshkol_handle_close(registry, file_id), ESHKOL_HANDLE_ENOENT);
    check_status(eshkol_handle_release(registry, file_id, file_lease),
                 ESHKOL_HANDLE_ENOENT);
    untouched_state = ESHKOL_HANDLE_STATE_OPEN;
    check_status(eshkol_handle_get_state(registry, file_id, &untouched_state),
                 ESHKOL_HANDLE_ENOENT);
    CHECK(untouched_state == ESHKOL_HANDLE_STATE_OPEN);

    // Simultaneous borrows have one winner; its lease alone may release it.
    constexpr int kBorrowers = 12;
    std::atomic<bool> start_borrowers{false};
    std::atomic<int> borrow_successes{0};
    uint64_t winning_lease = 0;
    std::vector<std::thread> borrowers;
    for (int i = 0; i < kBorrowers; ++i) {
        borrowers.emplace_back([&] {
            while (!start_borrowers.load()) {
                std::this_thread::yield();
            }
            uintptr_t value = 0;
            uint64_t lease = 0;
            if (eshkol_handle_borrow(registry, reused_id, &value, &lease) ==
                ESHKOL_HANDLE_OK) {
                winning_lease = lease;
                ++borrow_successes;
            }
        });
    }
    start_borrowers = true;
    for (std::thread &thread : borrowers) {
        thread.join();
    }
    CHECK(borrow_successes == 1);
    check_status(eshkol_handle_release(registry, reused_id, winning_lease),
                 ESHKOL_HANDLE_OK);
    uintptr_t value = 0;
    uint64_t lease_a = 0;
    uint64_t lease_b = 0;
    check_status(eshkol_handle_borrow(registry, reused_id, &value, &lease_a),
                 ESHKOL_HANDLE_OK);
    check_status(eshkol_handle_release(registry, reused_id, lease_a),
                 ESHKOL_HANDLE_OK);
    check_status(eshkol_handle_borrow(registry, reused_id, &value, &lease_b),
                 ESHKOL_HANDLE_OK);
    check_status(eshkol_handle_release(registry, reused_id, lease_a),
                 ESHKOL_HANDLE_ESTATE);
    check_status(eshkol_handle_close(registry, reused_id), ESHKOL_HANDLE_EBUSY);
    check_status(eshkol_handle_release(registry, reused_id, lease_b),
                 ESHKOL_HANDLE_OK);

    // Concurrent close contender sees CLOSING while the sole callback blocks.
    BlockingClose blocking;
    eshkol_handle_id_t blocking_id = 0;
    check_status(eshkol_handle_create(registry, ESHKOL_HANDLE_KIND_EVENT_LOOP,
        456, close_blocked, &blocking, &blocking_id), ESHKOL_HANDLE_OK);
    std::atomic<int> first_close_status{99};
    std::thread first_closer([&] {
        first_close_status = eshkol_handle_close(registry, blocking_id);
    });
    {
        std::unique_lock<std::mutex> lock(blocking.mutex);
        blocking.changed.wait(lock, [&] { return blocking.entered; });
    }
    check_status(eshkol_handle_close(registry, blocking_id), ESHKOL_HANDLE_EBUSY);
    {
        std::lock_guard<std::mutex> lock(blocking.mutex);
        blocking.may_return = true;
    }
    blocking.changed.notify_all();
    first_closer.join();
    check_status(first_close_status, ESHKOL_HANDLE_OK);
    CHECK(blocking.calls == 1);
    check_status(eshkol_handle_close(registry, blocking_id), ESHKOL_HANDLE_ENOENT);

    // Callback can re-enter without deadlock; returned failure and exception are
    // both terminal, so uncertain native closes are never attempted twice.
    reused.registry = registry;
    reused.id = reused_id;
    reused.fail = true;
    check_status(eshkol_handle_close(registry, reused_id), ESHKOL_HANDLE_ECLOSE);
    CHECK(reused.calls == 1);
    CHECK(reused.state_status == ESHKOL_HANDLE_OK);
    CHECK(reused.borrow_status == ESHKOL_HANDLE_EBUSY);
    check_status(eshkol_handle_close(registry, reused_id), ESHKOL_HANDLE_ENOENT);

    CloseContext throws;
    eshkol_handle_id_t throwing_id = 0;
    throws.throw_exception = true;
    check_status(eshkol_handle_create(registry, ESHKOL_HANDLE_KIND_FILE, 789,
        close_fake, &throws, &throwing_id), ESHKOL_HANDLE_OK);
    throws.registry = registry;
    throws.id = throwing_id;
    check_status(eshkol_handle_close(registry, throwing_id), ESHKOL_HANDLE_ECLOSE);
    CHECK(throws.calls == 1);
    check_status(eshkol_handle_close(registry, throwing_id), ESHKOL_HANDLE_ENOENT);

    // Destroy is busy during an outstanding borrow, then consumes all remaining
    // resources. It continues after callback failure and frees even on ECLOSE.
    CloseContext destroy_failure;
    CloseContext destroy_success;
    destroy_failure.fail = true;
    eshkol_handle_id_t failure_id = 0;
    eshkol_handle_id_t success_id = 0;
    check_status(eshkol_handle_create(registry, ESHKOL_HANDLE_KIND_SOCKET, 901,
        close_fake, &destroy_failure, &failure_id), ESHKOL_HANDLE_OK);
    check_status(eshkol_handle_create(registry, ESHKOL_HANDLE_KIND_FILE, 902,
        close_fake, &destroy_success, &success_id), ESHKOL_HANDLE_OK);
    destroy_failure.registry = registry;
    destroy_failure.id = failure_id;
    destroy_success.registry = registry;
    destroy_success.id = success_id;
    check_status(eshkol_handle_borrow(registry, failure_id, &borrowed_value,
        &file_lease), ESHKOL_HANDLE_OK);
    check_status(eshkol_handle_registry_destroy(registry), ESHKOL_HANDLE_EBUSY);
    check_status(eshkol_handle_release(registry, failure_id, file_lease),
                 ESHKOL_HANDLE_OK);
    check_status(eshkol_handle_registry_destroy(registry), ESHKOL_HANDLE_ECLOSE);
    CHECK(destroy_failure.calls == 1);
    CHECK(destroy_success.calls == 1);
    CHECK(destroy_failure.state_status == ESHKOL_HANDLE_OK);
    CHECK(destroy_failure.borrow_status == ESHKOL_HANDLE_EBUSY);
    std::cout << "PASS: handle_lifecycle_test\n";
    return 0;
}
