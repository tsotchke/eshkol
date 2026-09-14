#include "../../lib/core/model_io_atomic.h"

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        failures++;
    }
}

std::string read_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

void write_file(const fs::path& path, const std::string& bytes) {
    std::ofstream output(path, std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

mode_t file_mode(const fs::path& path) {
    struct stat status {};
    return stat(path.c_str(), &status) == 0 ? status.st_mode & 0777 : 0;
}

bool has_temporary(const fs::path& directory) {
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().filename().string().starts_with(".eshkol.")) return true;
    }
    return false;
}

struct TestRoot {
    fs::path path;

    TestRoot() {
        const char* configured = std::getenv("ESHKOL_TEST_TMPDIR");
        fs::path parent = configured && *configured
            ? fs::path(configured) : fs::temp_directory_path();
        std::string pattern = (parent / "eshkol-atomic-file.XXXXXX").string();
        std::vector<char> mutable_pattern(pattern.begin(), pattern.end());
        mutable_pattern.push_back('\0');
        if (char* created = mkdtemp(mutable_pattern.data())) path = created;
    }

    ~TestRoot() {
        std::error_code error;
        if (!path.empty()) fs::remove_all(path, error);
    }
};

void test_success_and_metadata(const fs::path& root) {
    const fs::path directory = root / "success";
    fs::create_directory(directory);
    const fs::path existing = directory / "existing.eskm";
    write_file(existing, "old");
    chmod(existing.c_str(), 0640);

    eshkol_atomic_checkpoint_file_t transaction {};
    expect(eshkol_atomic_checkpoint_begin(&transaction, existing.c_str()) == 1,
           "open existing destination");
    if (transaction.stream) {
        const int descriptor_flags = fcntl(fileno(transaction.stream), F_GETFD);
        expect(descriptor_flags >= 0 && (descriptor_flags & FD_CLOEXEC) != 0,
               "temporary descriptor is close-on-exec");
        const char replacement[] = "complete checkpoint";
        expect(eshkol_atomic_checkpoint_write(&transaction, replacement,
                                              sizeof(replacement) - 1) ==
                   sizeof(replacement) - 1,
               "write successful replacement");
        expect(eshkol_atomic_checkpoint_commit(&transaction) == 1,
               "commit successful replacement");
        expect(read_file(existing) == replacement, "publish exact replacement bytes");
        expect(file_mode(existing) == 0640, "preserve existing regular-file mode");
    }

    const mode_t previous_umask = umask(0777);
    const fs::path fresh = directory / "fresh.eskm";
    eshkol_atomic_checkpoint_file_t fresh_transaction {};
    const int opened = eshkol_atomic_checkpoint_begin(&fresh_transaction, fresh.c_str());
    umask(previous_umask);
    expect(opened == 1, "open new destination under restrictive umask");
    if (opened) {
        const char bytes[] = "new";
        expect(eshkol_atomic_checkpoint_write(&fresh_transaction, bytes,
                                              sizeof(bytes) - 1) == sizeof(bytes) - 1,
               "write new destination");
        expect(eshkol_atomic_checkpoint_commit(&fresh_transaction) == 1,
               "commit new destination");
        expect(file_mode(fresh) == 0600, "new destination mode is exactly 0600");
    }

    const long name_max = pathconf(directory.c_str(), _PC_NAME_MAX);
    const auto long_name_size = static_cast<std::size_t>(
        name_max > 0 && name_max < 250 ? name_max : 250);
    const fs::path long_name = directory / std::string(long_name_size, 'x');
    eshkol_atomic_checkpoint_file_t long_transaction {};
    expect(eshkol_atomic_checkpoint_begin(&long_transaction, long_name.c_str()) == 1,
           "fixed-size temporary name supports a 250-byte destination basename");
    if (long_transaction.stream) {
        const char bytes[] = "long-name";
        expect(eshkol_atomic_checkpoint_write(&long_transaction, bytes,
                                              sizeof(bytes) - 1) == sizeof(bytes) - 1,
               "write long-name destination");
        expect(eshkol_atomic_checkpoint_commit(&long_transaction) == 1,
               "commit long-name destination");
        expect(read_file(long_name) == "long-name", "long-name bytes are exact");
    }
    expect(!has_temporary(directory), "successful cases leave no temporary files");
}

void test_failpoint(const fs::path& root, const std::string& failpoint) {
    std::string label = failpoint;
    for (char& character : label) if (character == ':') character = '-';
    const fs::path directory = root / label;
    fs::create_directory(directory);
    const fs::path destination = directory / "checkpoint.eskm";
    const char original_bytes[] = "original checkpoint\0bytes";
    const std::string original(original_bytes, sizeof(original_bytes) - 1);
    write_file(destination, original);
    chmod(destination.c_str(), 0604);

    expect(setenv("ESHKOL_TEST_MODEL_IO_FAIL", failpoint.c_str(), 1) == 0,
           failpoint + ": set test hook");
    eshkol_atomic_checkpoint_file_t transaction {};
    const int opened = eshkol_atomic_checkpoint_begin(&transaction, destination.c_str());
    if (failpoint == "open") {
        expect(opened == 0, "open: injected open failure observed");
    } else {
        expect(opened == 1, failpoint + ": transaction opened");
        bool write_failed = false;
        if (opened) {
            const char chunk[] = "chunk";
            for (int call = 0; call < 4; call++) {
                if (eshkol_atomic_checkpoint_write(&transaction, chunk,
                                                   sizeof(chunk) - 1) != sizeof(chunk) - 1) {
                    write_failed = true;
                    break;
                }
            }
            if (failpoint == "write" || failpoint.starts_with("write:")) {
                expect(write_failed, failpoint + ": injected write failure observed");
                eshkol_atomic_checkpoint_abort(&transaction);
            } else {
                expect(!write_failed, failpoint + ": setup writes succeeded");
                expect(eshkol_atomic_checkpoint_commit(&transaction) == 0,
                       failpoint + ": injected commit failure observed");
            }
        }
    }
    unsetenv("ESHKOL_TEST_MODEL_IO_FAIL");
    expect(read_file(destination) == original,
           failpoint + ": destination preserved byte-for-byte");
    expect(file_mode(destination) == 0604, failpoint + ": destination mode preserved");
    expect(!has_temporary(directory), failpoint + ": no orphan temporary file");
}

void test_signal_cleanup(const fs::path& root) {
    const fs::path directory = root / "signal";
    fs::create_directory(directory);
    const fs::path destination = directory / "checkpoint.eskm";
    write_file(destination, "original signal checkpoint");

    const pid_t child = fork();
    expect(child >= 0, "signal: fork child");
    if (child == 0) {
        if (setenv("ESHKOL_TEST_MODEL_IO_FAIL", "signal", 1) != 0) _exit(120);
        eshkol_atomic_checkpoint_file_t transaction {};
        if (!eshkol_atomic_checkpoint_begin(&transaction, destination.c_str())) _exit(121);
        const char partial[] = "partial replacement";
        if (eshkol_atomic_checkpoint_write(&transaction, partial, sizeof(partial)) !=
            sizeof(partial)) _exit(122);
        (void)eshkol_atomic_checkpoint_commit(&transaction);
        _exit(123);
    }
    if (child < 0) return;

    int status = 0;
    expect(waitpid(child, &status, 0) == child, "signal: wait for child");
    expect(WIFSIGNALED(status) && WTERMSIG(status) == SIGTERM,
           "signal: SIGTERM deferred until cleanup restored the mask");
    expect(read_file(destination) == "original signal checkpoint",
           "signal: destination preserved byte-for-byte");
    expect(!has_temporary(directory), "signal: no orphan temporary file");
}

} // namespace

int main() {
    TestRoot root;
    if (root.path.empty()) {
        std::cerr << "FAIL: could not create test directory: " << std::strerror(errno) << '\n';
        return 1;
    }

    test_success_and_metadata(root.path);
    for (const char* failpoint :
         {"open", "write", "write:3", "flush", "close", "interrupt", "rename"}) {
        test_failpoint(root.path, failpoint);
    }
    test_signal_cleanup(root.path);

    if (failures != 0) return 1;
    std::cout << "PASS: atomic checkpoint helper success, metadata, failpoints, and signal cleanup\n";
    return 0;
}
