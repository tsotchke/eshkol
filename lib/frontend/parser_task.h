/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#ifndef ESHKOL_FRONTEND_PARSER_TASK_H
#define ESHKOL_FRONTEND_PARSER_TASK_H

#include <coroutine>
#include <exception>
#include <utility>

// Grammar functions suspend at child parses. This explicit continuation stack
// resumes only one frame at a time: neither await_suspend nor final_suspend
// resumes another coroutine (including in unoptimized builds). The C++20
// coroutine frame owns tokens, partial ASTs and locals across each suspension.
struct ParserContinuation {
    std::coroutine_handle<> handle;
    ParserContinuation* child = nullptr;
    ParserContinuation* parent = nullptr;
};

template<class T>
class ParserTask {
public:
    struct promise_type : ParserContinuation {
        T value{};
        std::exception_ptr error;
        ParserTask get_return_object() {
            auto h = std::coroutine_handle<promise_type>::from_promise(*this);
            handle = h;
            return ParserTask(h);
        }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_value(T result) { value = std::move(result); }
        void unhandled_exception() noexcept { error = std::current_exception(); }
    };

    ParserTask(const ParserTask&) = delete;
    ParserTask& operator=(const ParserTask&) = delete;
    ParserTask(ParserTask&& other) noexcept : handle_(std::exchange(other.handle_, {})) {}
    ~ParserTask() { if (handle_) handle_.destroy(); }

    bool await_ready() const noexcept { return false; }
    template<class Parent>
    void await_suspend(std::coroutine_handle<Parent> parent) noexcept {
        parent.promise().child = &handle_.promise();
    }
    T await_resume() {
        if (handle_.promise().error) std::rethrow_exception(handle_.promise().error);
        return std::move(handle_.promise().value);
    }

    T run() {
        ParserContinuation* current = &handle_.promise();
        while (current) {
            current->handle.resume();
            if (current->handle.done()) {
                current = current->parent;
            } else {
                auto* child = std::exchange(current->child, nullptr);
                child->parent = current;
                current = child;
            }
        }
        return await_resume();
    }

private:
    explicit ParserTask(std::coroutine_handle<promise_type> handle) : handle_(handle) {}
    std::coroutine_handle<promise_type> handle_;
};

#endif
