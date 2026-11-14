/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_DRIVER_OUTPUT_OWNERSHIP_H
#define ESHKOL_DRIVER_OUTPUT_OWNERSHIP_H

/**
 * @file driver_output_ownership.h
 * @brief Who owns a native output path, and the files beside it, while the
 *        driver compiles and links.
 *
 * Three kinds of file appear beside `<output>` during a native build, and each
 * has exactly one owner:
 *
 *  - `<output>.eshkol-link.lock` is the claim on the output path. One driver
 *    holds it from before it compiles until it has published or given up. The
 *    hold is an operating-system file lock, so it ends with the process on
 *    every exit path, including one the process cannot observe; a claim file
 *    left behind by a process that is gone is simply taken over by the next
 *    compile of that output, which is also the only thing that removes it:
 *    the file names no process, and only a claimant can tell a leftover from
 *    a live claim. A second driver that asks for the same output while the
 *    first is alive waits for it, for a bounded time, and is then told the
 *    output is owned. Where the filesystem has no file locks, the build goes
 *    on without a claim (staging still applies). Where the output is not a
 *    regular file (a device such as `/dev/null`), there is no claim and no
 *    staging: the build proceeds in place.
 *  - `<output>.<pid>.<n>.tmp.o` is the scratch object the driver compiles
 *    before the link.
 *  - `<output>.link-<pid>-<n>.d/` is the staging directory the linker writes
 *    into. The finished artifact carries its final leaf name there and is
 *    moved onto `<output>` in one rename, so a reader of `<output>` sees the
 *    previous artifact or the new one, never part of either.
 *
 * The last two carry the process id of the driver that made them. A driver
 * that is stopped before it can clean up leaves them behind; the next driver
 * for the same output removes those whose process no longer exists. A process
 * that is alive, or whose state cannot be determined, keeps its files.
 *
 * Staging is the caller's choice per build. The driver links in place, still
 * under the claim, where a debug companion is named from the link path: every
 * Windows debug build, and on any platform a debug build of a program with no
 * objects to link.
 *
 * Publication is a rename onto the output name. An output name that is a
 * symbolic link is therefore replaced by the artifact; the file the link
 * pointed at is left as it was.
 *
 * Header-only and free of compiler dependencies so the naming and liveness
 * rules can be exercised on their own on every platform.
 *
 * The Windows branches below were written against the documented behaviour of
 * the Win32 calls they use; the Windows lanes of CI are their first build and
 * their first run, through driver_output_ownership_test.
 */

#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <process.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace eshkol::driver {

/** @brief Id of the calling process. */
inline uint64_t current_process_id() {
#ifdef _WIN32
    return static_cast<uint64_t>(_getpid());
#else
    return static_cast<uint64_t>(getpid());
#endif
}

/**
 * @brief True only when no process with @p pid exists.
 *
 * Any other answer from the system (the process is running, belongs to
 * another user, or cannot be queried) is "not gone", so a caller that removes
 * files on a true answer can only ever remove an absent process's files. A
 * number the system has given to a new, unrelated process also reads as
 * present; its files wait for that process to end.
 */
inline bool process_is_gone(uint64_t pid) {
    if (pid == 0 || pid == current_process_id()) return false;
#ifdef _WIN32
    if (pid > 0xFFFFFFFFull) return false;
    HANDLE process = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
                                 FALSE, static_cast<DWORD>(pid));
    if (!process) {
        // No such process id is reported as an invalid parameter; anything
        // else (access denied, most often) means a process is there.
        return GetLastError() == ERROR_INVALID_PARAMETER;
    }
    // A process object can outlive its process while a handle is open; it is
    // gone once it is signaled and reports an exit code.
    bool gone = false;
    if (WaitForSingleObject(process, 0) == WAIT_OBJECT_0) {
        DWORD exit_code = STILL_ACTIVE;
        gone = GetExitCodeProcess(process, &exit_code) && exit_code != STILL_ACTIVE;
    }
    CloseHandle(process);
    return gone;
#else
    if (pid > static_cast<uint64_t>(INT32_MAX)) return false;
    return ::kill(static_cast<pid_t>(pid), 0) == -1 && errno == ESRCH;
#endif
}

namespace detail {

/** Parse a run of decimal digits of sane length; nothing else is accepted. */
inline std::optional<uint64_t> parse_decimal(const std::string& text) {
    if (text.empty() || text.size() > 19) return std::nullopt;
    uint64_t value = 0;
    for (char c : text) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return std::nullopt;
        value = value * 10 + static_cast<uint64_t>(c - '0');
    }
    return value;
}

/** Owner of `<leaf><open><pid><sep><n><close>`, or nothing. */
inline std::optional<uint64_t> owner_between(const std::string& name,
                                             const std::string& leaf,
                                             const std::string& open,
                                             char separator,
                                             const std::string& close) {
    const std::string prefix = leaf + open;
    if (name.size() <= prefix.size() + close.size() ||
        name.compare(0, prefix.size(), prefix) != 0 ||
        name.compare(name.size() - close.size(), close.size(), close) != 0) {
        return std::nullopt;
    }
    const std::string middle =
        name.substr(prefix.size(), name.size() - prefix.size() - close.size());
    const auto split = middle.find(separator);
    if (split == std::string::npos) return std::nullopt;
    const auto pid = parse_decimal(middle.substr(0, split));
    const auto sequence = parse_decimal(middle.substr(split + 1));
    if (!pid || !sequence) return std::nullopt;
    return pid;
}

inline uint64_t next_sequence() {
    static std::atomic<uint64_t> sequence{0};
    return sequence.fetch_add(1, std::memory_order_relaxed);
}

} // namespace detail

/** @brief Name of the scratch object process @p pid makes for output leaf @p leaf. */
inline std::string scratch_object_name(const std::string& leaf, uint64_t pid, uint64_t n) {
    return leaf + "." + std::to_string(pid) + "." + std::to_string(n) + ".tmp.o";
}

/** @brief Name of the staging directory process @p pid makes for output leaf @p leaf. */
inline std::string staging_directory_name(const std::string& leaf, uint64_t pid, uint64_t n) {
    return leaf + ".link-" + std::to_string(pid) + "-" + std::to_string(n) + ".d";
}

/** @brief Name of the claim file for output leaf @p leaf. */
inline std::string claim_file_name(const std::string& leaf) {
    return leaf + ".eshkol-link.lock";
}

/**
 * @brief Process that made scratch object @p name for output leaf @p leaf.
 * @return The pid, or nothing when @p name is not such a scratch object. The
 *         match is exact: `foo.1.5.0.tmp.o` belongs to output `foo.1`, not `foo`.
 */
inline std::optional<uint64_t> scratch_object_owner(const std::string& name,
                                                    const std::string& leaf) {
    return detail::owner_between(name, leaf, ".", '.', ".tmp.o");
}

/** @brief Process that made staging directory @p name for output leaf @p leaf. */
inline std::optional<uint64_t> staging_directory_owner(const std::string& name,
                                                       const std::string& leaf) {
    return detail::owner_between(name, leaf, ".link-", '-', ".d");
}

/**
 * @brief Remove the scratch objects and staging directories for @p output
 *        whose driver no longer exists.
 * @param gone Liveness rule; the default asks the operating system.
 * @return Paths removed.
 */
inline std::vector<std::filesystem::path> reclaim_abandoned(
    const std::filesystem::path& output,
    bool (*gone)(uint64_t) = process_is_gone) {
    std::vector<std::filesystem::path> removed;
    std::error_code ec;
    std::filesystem::path dir = output.parent_path();
    if (dir.empty()) dir = ".";
    const std::string leaf = output.filename().string();
    std::vector<std::filesystem::path> abandoned;
    for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end;
         it.increment(ec)) {
        const std::string name = it->path().filename().string();
        auto owner = scratch_object_owner(name, leaf);
        if (!owner) owner = staging_directory_owner(name, leaf);
        if (owner && gone(*owner)) abandoned.push_back(it->path());
    }
    for (const auto& path : abandoned) {
        std::error_code remove_ec;
        std::filesystem::remove_all(path, remove_ec);
        if (!remove_ec) removed.push_back(path);
    }
    return removed;
}

/**
 * @brief Whether @p leaf is one of the device names Windows reserves in every
 *        directory: NUL, CON, PRN, AUX, COM1-COM9, LPT1-LPT9, in any letter
 *        case, with or without an extension or a trailing colon.
 *
 * A pure string test, so it is the same answer on every platform; it is
 * consulted only on Windows, where such a name is a device wherever it
 * appears and the filesystem's own report about it is not dependable.
 */
inline bool is_windows_device_name(const std::string& leaf) {
    std::string base = leaf.substr(0, leaf.find('.'));
    while (!base.empty() && (base.back() == ':' || base.back() == ' ')) base.pop_back();
    for (char& c : base) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (base == "NUL" || base == "CON" || base == "PRN" || base == "AUX") return true;
    if (base.size() == 4 && (base.compare(0, 3, "COM") == 0 || base.compare(0, 3, "LPT") == 0)) {
        return base[3] >= '1' && base[3] <= '9';
    }
    return false;
}

/**
 * @brief Whether @p output is a place where a claim and a staged artifact
 *        make sense: a name that is free, or that leads to a regular file.
 *
 * A device, pipe or directory under that name is written by whoever links,
 * directly; nothing can be staged beside it and renamed onto it.
 */
inline bool output_admits_ownership(const std::filesystem::path& output) {
#ifdef _WIN32
    if (is_windows_device_name(output.filename().string())) return false;
#endif
    std::error_code ec;
    const auto status = std::filesystem::status(output, ec);
    if (!std::filesystem::exists(status)) return true;
    return std::filesystem::is_regular_file(status);
}

/**
 * @brief The claim one driver holds on an output path.
 *
 * Not copyable. Releasing (or leaving scope) removes the claim file and ends
 * the hold. If the process ends without releasing, the operating system ends
 * the hold and the next claimant takes the file over.
 */
class OutputClaim {
public:
    enum class Result {
        Acquired,     ///< This process now owns the output path.
        HeldElsewhere,///< A live process owns it.
        Unavailable,  ///< The claim file could not be created or opened.
        NotSupported, ///< This filesystem offers no file lock; nobody can claim here.
        TryAgain,     ///< The system had no lock to give just now; ask again.
    };

    /**
     * @brief Whether a failed lock request means "this filesystem has no
     *        locks" rather than "the lock is taken" or "something is wrong".
     *
     * Network and user-space filesystems answer a lock request with one of
     * these. None of them says anything about another compile, and none can
     * be true while some process holds a lock on the same file, so the build
     * goes on without a claim.
     */
    static bool lock_error_means_unsupported(int error) {
#ifdef _WIN32
        return error == ERROR_NOT_SUPPORTED || error == ERROR_INVALID_FUNCTION;
#else
        return error == ENOTSUP || error == EOPNOTSUPP || error == EINVAL ||
               error == ENOSYS;
#endif
    }

    /**
     * @brief Whether a failed lock request is a passing shortage: the system's
     *        lock table is full. Locks exist here and another process may well
     *        hold this one, so the claimant waits and asks again; the claim
     *        file is left exactly as it is.
     */
    static bool lock_error_is_transient(int error) {
#ifdef _WIN32
        return error == ERROR_LOCK_VIOLATION || error == ERROR_TOO_MANY_OPEN_FILES;
#else
        return error == ENOLCK;
#endif
    }

    /** @brief Map a failed lock request's error to a claim result. */
    static Result result_for_lock_error(int error) {
#ifdef _WIN32
        if (error == ERROR_SHARING_VIOLATION || error == ERROR_ACCESS_DENIED)
            return Result::HeldElsewhere;
#else
        if (error == EWOULDBLOCK || error == EAGAIN) return Result::HeldElsewhere;
#endif
        if (lock_error_is_transient(error)) return Result::TryAgain;
        return lock_error_means_unsupported(error) ? Result::NotSupported
                                                   : Result::Unavailable;
    }

    OutputClaim() = default;
    OutputClaim(const OutputClaim&) = delete;
    OutputClaim& operator=(const OutputClaim&) = delete;
    ~OutputClaim() { release(); }

    bool held() const { return held_; }
    const std::string& output() const { return output_; }

    /** @brief Claim @p output. Claiming the path already held by this object succeeds. */
    Result acquire(const std::filesystem::path& output) {
        const std::string wanted = output.lexically_normal().generic_string();
        if (held_) {
            if (wanted == output_) return Result::Acquired;
            release();
        }
        std::filesystem::path claim = output;
        claim.replace_filename(claim_file_name(output.filename().string()));
#ifdef _WIN32
        // No sharing: a second open fails while this handle lives. The file is
        // deleted when the handle closes, which the system does for a process
        // that ends in any way.
        HANDLE handle = CreateFileW(claim.wstring().c_str(), GENERIC_WRITE, 0, nullptr,
                                    OPEN_ALWAYS,
                                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE,
                                    nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            return result_for_lock_error(static_cast<int>(GetLastError()));
        }
        handle_ = handle;
#else
        for (int attempt = 0; attempt < 8; ++attempt) {
            const int fd = ::open(claim.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
            if (fd < 0) return Result::Unavailable;
            if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
                const int error = errno;
                const Result result = result_for_lock_error(error);
                if (result == Result::NotSupported) {
                    // Nobody can hold a claim here, so the file just made
                    // serves no one; leave the directory as it was found.
                    ::unlink(claim.c_str());
                }
                ::close(fd);
                return result;
            }
            // The previous holder removes the file as it lets go. Make sure
            // the name still leads to the file that was just locked; if not,
            // the lock is on a file nobody else will find, so start again.
            struct stat by_handle {};
            struct stat by_name {};
            if (::fstat(fd, &by_handle) == 0 && ::stat(claim.c_str(), &by_name) == 0 &&
                by_handle.st_dev == by_name.st_dev && by_handle.st_ino == by_name.st_ino) {
                const std::string owner = std::to_string(current_process_id()) + "\n";
                if (::ftruncate(fd, 0) == 0) {
                    (void)!::write(fd, owner.data(), owner.size());
                }
                fd_ = fd;
                break;
            }
            ::close(fd);
        }
        if (fd_ < 0) return Result::Unavailable;
#endif
        claim_path_ = claim;
        output_ = wanted;
        held_ = true;
        return Result::Acquired;
    }

    /** @brief Give the output path up. Safe to call when nothing is held. */
    void release() {
        if (!held_) return;
#ifdef _WIN32
        CloseHandle(static_cast<HANDLE>(handle_));
        handle_ = nullptr;
#else
        // Remove the name while the lock is still held, so no other process
        // can be holding a lock on a file that is about to lose its name.
        ::unlink(claim_path_.c_str());
        ::close(fd_);
        fd_ = -1;
#endif
        held_ = false;
        output_.clear();
    }

private:
    bool held_ = false;
    std::string output_;
    std::filesystem::path claim_path_;
#ifdef _WIN32
    void* handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

/**
 * @brief A staging directory beside an output, and the one-step publication
 *        of the artifact built in it.
 */
struct StagedOutput {
    std::filesystem::path directory; ///< `<output>.link-<pid>-<n>.d`
    std::filesystem::path artifact;  ///< `<directory>/<output leaf>`
    std::filesystem::path output;    ///< The final path.

    bool valid() const { return !artifact.empty(); }
};

/**
 * @brief Create this process's staging directory for @p output.
 *
 * The artifact keeps the output's leaf name inside the directory, so anything
 * the linker derives from the file name is the same as for a direct link.
 */
inline StagedOutput stage_output(const std::filesystem::path& output) {
    std::filesystem::path parent = output.parent_path();
    const std::string leaf = output.filename().string();
    for (int attempt = 0; attempt < 16; ++attempt) {
        std::filesystem::path directory =
            parent / staging_directory_name(leaf, current_process_id(), detail::next_sequence());
        std::error_code ec;
        if (std::filesystem::create_directory(directory, ec) && !ec) {
            return {directory, directory / leaf, output};
        }
        if (ec) break; // The parent does not admit a new directory.
    }
    return {};
}

/**
 * @brief Move the staged artifact onto its output in one rename and remove
 *        the staging directory.
 * @return false when the rename did not happen; the output is then whatever
 *         it was before.
 */
inline bool publish_staged_output(const StagedOutput& staged) {
    if (!staged.valid()) return false;
    std::error_code ec;
    std::filesystem::rename(staged.artifact, staged.output, ec);
    if (ec) return false;
    std::error_code cleanup_ec;
    std::filesystem::remove_all(staged.directory, cleanup_ec);
    return true;
}

} // namespace eshkol::driver

#endif // ESHKOL_DRIVER_OUTPUT_OWNERSHIP_H
