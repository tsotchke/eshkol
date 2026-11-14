// Naming, liveness, claim and publication rules for the files the driver
// keeps beside a native output. No compiler is involved, so this runs the
// same way on every platform.
#include <eshkol/driver_output_ownership.h>

#include <fstream>
#include <iostream>
#include <string>

#ifndef _WIN32
#include <sys/wait.h>
#endif

namespace fs = std::filesystem;
using namespace eshkol::driver;

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

void touch(const fs::path& path, const char* text = "x") {
    std::ofstream out(path, std::ios::binary);
    out << text;
}

std::string slurp(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool only_41_is_gone(uint64_t pid) { return pid == 41; }

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: driver_output_ownership_test <work-dir>\n";
        return 2;
    }
    const fs::path work = fs::path(argv[1]) / ("ownership-" + std::to_string(current_process_id()));
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);

    // Names round-trip, and only exact names for the exact output match.
    check(scratch_object_owner(scratch_object_name("prog", 123, 7), "prog") == 123u,
          "scratch name round-trips");
    check(staging_directory_owner(staging_directory_name("prog", 123, 7), "prog") == 123u,
          "staging name round-trips");
    check(!scratch_object_owner("prog.tmp.o", "prog"), "a plain <output>.tmp.o has no owner");
    check(!scratch_object_owner("prog.1.5.0.tmp.o", "prog"), "scratch of output prog.1 is not prog's");
    check(scratch_object_owner("prog.1.5.0.tmp.o", "prog.1") == 5u, "scratch of output prog.1 is prog.1's");
    check(!scratch_object_owner("prog.x.0.tmp.o", "prog"), "non-numeric pid is refused");
    check(!scratch_object_owner("prog.12..tmp.o", "prog"), "missing sequence is refused");
    check(!scratch_object_owner("prog.12.3.tmp.obj", "prog"), "other suffix is refused");
    check(!scratch_object_owner("other.12.3.tmp.o", "prog"), "other output is refused");
    check(!staging_directory_owner("prog.link-12.d", "prog"), "staging name needs a sequence");
    check(!staging_directory_owner("prog.link-12-3", "prog"), "staging name needs its suffix");
    check(!scratch_object_owner("prog.99999999999999999999.0.tmp.o", "prog"), "oversized pid is refused");

    // Liveness: this process is present; pid 0 is never reported gone.
    check(!process_is_gone(current_process_id()), "the calling process is not gone");
    check(!process_is_gone(0), "pid 0 is not gone");
#ifndef _WIN32
    {
        const pid_t child = fork();
        if (child == 0) _exit(0);
        int status = 0;
        check(child > 0 && waitpid(child, &status, 0) == child, "child ran");
        check(process_is_gone(static_cast<uint64_t>(child)), "a reaped child is gone");
    }
#else
    {
        // A process that has ended and whose last handle is closed.
        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        PROCESS_INFORMATION info{};
        wchar_t command[] = L"cmd.exe /c exit 0";
        if (CreateProcessW(nullptr, command, nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                           nullptr, nullptr, &startup, &info)) {
            check(!process_is_gone(info.dwProcessId) ||
                      WaitForSingleObject(info.hProcess, 0) == WAIT_OBJECT_0,
                  "a running child is not gone");
            WaitForSingleObject(info.hProcess, INFINITE);
            check(process_is_gone(info.dwProcessId), "an ended child is gone while a handle is open");
            const DWORD ended = info.dwProcessId;
            CloseHandle(info.hThread);
            CloseHandle(info.hProcess);
            check(process_is_gone(ended), "an ended child is gone after its handles close");
        } else {
            check(false, "could not start a child process");
        }
    }
#endif

    // Reclaim removes exactly the files of the process that is gone.
    {
        const fs::path out = work / "prog";
        const fs::path gone_scratch = work / scratch_object_name("prog", 41, 0);
        const fs::path live_scratch = work / scratch_object_name("prog", 42, 0);
        const fs::path gone_stage = work / staging_directory_name("prog", 41, 1);
        const fs::path caller_file = work / "prog.tmp.o";
        const fs::path sibling = work / scratch_object_name("prog.1", 41, 0);
        touch(gone_scratch);
        touch(live_scratch);
        fs::create_directories(gone_stage);
        touch(gone_stage / "prog");
        touch(caller_file, "caller");
        touch(sibling);
        const auto removed = reclaim_abandoned(out, only_41_is_gone);
        check(removed.size() == 2, "two abandoned entries removed");
        check(!fs::exists(gone_scratch), "scratch of the gone process removed");
        check(!fs::exists(gone_stage), "staging of the gone process removed");
        check(fs::exists(live_scratch), "scratch of a live process kept");
        check(slurp(caller_file) == "caller", "a caller's file is untouched");
        check(fs::exists(sibling), "another output's scratch is untouched");
    }

    // One claim per output; a second claimant is told it is held; a released
    // or leftover claim file can be claimed again.
    {
        const fs::path out = work / "claimed";
        OutputClaim first;
        check(first.acquire(out) == OutputClaim::Result::Acquired, "first claim acquired");
        check(first.acquire(out) == OutputClaim::Result::Acquired, "re-claim by the holder succeeds");
        {
            OutputClaim second;
            check(second.acquire(out) == OutputClaim::Result::HeldElsewhere,
                  "second claim sees the holder");
        }
        OutputClaim other;
        check(other.acquire(work / "unrelated") == OutputClaim::Result::Acquired,
              "a different output is free");
        first.release();
        check(!fs::exists(work / claim_file_name("claimed")), "release removes the claim file");
        OutputClaim third;
        check(third.acquire(out) == OutputClaim::Result::Acquired, "claim after release acquired");
        third.release();
        touch(work / claim_file_name("claimed"), "999999\n");
        OutputClaim fourth;
        check(fourth.acquire(out) == OutputClaim::Result::Acquired,
              "a claim file nobody holds is taken over");
        OutputClaim nowhere;
        check(nowhere.acquire(work / "missing-dir" / "x") == OutputClaim::Result::Unavailable,
              "a missing directory is reported");
    }

    // A failed lock request is read three ways: the lock is taken, this
    // filesystem has no locks, or the claim file itself is the difficulty.
    {
        using Result = OutputClaim::Result;
#ifdef _WIN32
        check(OutputClaim::result_for_lock_error(ERROR_SHARING_VIOLATION) == Result::HeldElsewhere,
              "sharing violation means held");
        check(OutputClaim::result_for_lock_error(ERROR_NOT_SUPPORTED) == Result::NotSupported,
              "not-supported means no locks here");
        check(OutputClaim::result_for_lock_error(ERROR_INVALID_FUNCTION) == Result::NotSupported,
              "invalid-function means no locks here");
        check(OutputClaim::result_for_lock_error(ERROR_PATH_NOT_FOUND) == Result::Unavailable,
              "missing path means unavailable");
#else
        check(OutputClaim::result_for_lock_error(EWOULDBLOCK) == Result::HeldElsewhere,
              "EWOULDBLOCK means held");
        check(OutputClaim::result_for_lock_error(EAGAIN) == Result::HeldElsewhere,
              "EAGAIN means held");
        for (int error : {ENOTSUP, EOPNOTSUPP, ENOLCK, EINVAL, ENOSYS}) {
            check(OutputClaim::result_for_lock_error(error) == Result::NotSupported,
                  "an unsupported-lock error means no locks here");
            check(OutputClaim::lock_error_means_unsupported(error),
                  "an unsupported-lock error is recognised");
        }
        for (int error : {EACCES, EIO, EBADF, ENOENT, EROFS}) {
            check(OutputClaim::result_for_lock_error(error) == Result::Unavailable,
                  "any other error means unavailable");
        }
        check(!OutputClaim::lock_error_means_unsupported(EWOULDBLOCK),
              "a taken lock is not an unsupported lock");
#endif
    }

    // Ownership applies to a free name or a regular file, not to a device or
    // a directory.
    {
        check(output_admits_ownership(work / "not-there-yet"), "a free name admits ownership");
        touch(work / "regular");
        check(output_admits_ownership(work / "regular"), "a regular file admits ownership");
        check(!output_admits_ownership(work), "a directory does not admit ownership");
#ifdef _WIN32
        check(!output_admits_ownership("NUL"), "the null device does not admit ownership");
#else
        check(!output_admits_ownership("/dev/null"), "the null device does not admit ownership");
#endif
    }

    // Publication replaces the output in one step and leaves no staging.
    {
        const fs::path out = work / "artifact";
        touch(out, "old");
        const StagedOutput staged = stage_output(out);
        check(staged.valid(), "staging directory created");
        check(staged.artifact.filename() == out.filename(), "staged artifact keeps the leaf name");
        check(staging_directory_owner(staged.directory.filename().string(), "artifact") ==
                  current_process_id(),
              "staging directory carries this process");
        touch(staged.artifact, "new");
        check(slurp(out) == "old", "output unchanged before publication");
        check(publish_staged_output(staged), "publication succeeds");
        check(slurp(out) == "new", "output is the new artifact");
        check(!fs::exists(staged.directory), "staging directory removed");
        const StagedOutput empty = stage_output(out);
        check(!publish_staged_output(empty) && slurp(out) == "new",
              "publishing nothing leaves the output as it was");
        fs::remove_all(empty.directory, ec);
    }

    fs::remove_all(work, ec);
    if (failures) {
        std::cerr << failures << " check(s) did not hold\n";
        return 1;
    }
    std::cout << "PASS: driver output ownership\n";
    return 0;
}
