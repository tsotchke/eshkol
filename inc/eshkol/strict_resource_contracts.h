#pragma once
/**
 * @file strict_resource_contracts.h
 * @brief Typestate contracts for every native handle family checked by
 *        `eshkol-run --strict-resources`.
 *
 * One table describes how each handle-producing, handle-consuming, and
 * handle-observing operation moves a handle through the typestate
 * `open -> borrowed -> open ... -> closed` (docs/design/adr/0056-linear-handle-typestate.md).
 * The ownership checker in exe/eshkol-run.cpp reads only this table: no
 * operation name is special-cased elsewhere.
 *
 * Module contracts are trusted only for the exact stock SDK source they were
 * reviewed against. Each entry pins the SHA-256 of that source file, and the
 * export set must equal the contract names. An edited module keeps compiling
 * in the default profile but is refused by the strict profile until its
 * contract is reviewed and re-pinned (scripts/strict_resource_pins.py).
 *
 * Builtin contracts apply only when the name is not lexically rebound.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace eshkol::strict_resources {

/** Handle families. The order is part of the diagnostic text only. */
enum class Kind : uint8_t {
    NONE = 0,
    TCP_LISTENER,
    TCP_STREAM,
    UDP_SOCKET,
    TLS_CONTEXT,
    TLS_LISTENER,
    TLS_SESSION,
    INPUT_PORT,
    OUTPUT_PORT,
    EVENT_LOOP,
    THREAD,
    MUTEX,
    CONDVAR,
    PROCESS,
    COUNT
};

constexpr uint32_t bit(Kind kind) { return 1u << static_cast<unsigned>(kind); }

constexpr uint32_t kAnySocket =
    bit(Kind::TCP_LISTENER) | bit(Kind::TCP_STREAM) | bit(Kind::UDP_SOCKET);
constexpr uint32_t kAnyTcp = bit(Kind::TCP_LISTENER) | bit(Kind::TCP_STREAM);
constexpr uint32_t kAnyPort = bit(Kind::INPUT_PORT) | bit(Kind::OUTPUT_PORT);

inline const char* kindName(Kind kind) {
    switch (kind) {
        case Kind::TCP_LISTENER: return "TCP listener";
        case Kind::TCP_STREAM: return "TCP stream";
        case Kind::UDP_SOCKET: return "UDP socket";
        case Kind::TLS_CONTEXT: return "TLS context";
        case Kind::TLS_LISTENER: return "TLS listener";
        case Kind::TLS_SESSION: return "TLS session";
        case Kind::INPUT_PORT: return "input file port";
        case Kind::OUTPUT_PORT: return "output file port";
        case Kind::EVENT_LOOP: return "event loop";
        case Kind::THREAD: return "thread";
        case Kind::MUTEX: return "mutex";
        case Kind::CONDVAR: return "condition variable";
        case Kind::PROCESS: return "process";
        default: return "value";
    }
}

/**
 * How a failed acquisition presents itself. RAISES means the factory raises
 * and a returned value is always an open handle. The other families return a
 * failure value that must be refined away before the handle is used.
 */
enum class Failure : uint8_t {
    RAISES = 0,
    NETWORK_ERROR,  // core.net: refined by network-error?
    TLS_ERROR,      // core.tls: refined by tls-error?
    FALSE_VALUE     // #f on failure: refined by (if h ...) or (not h)
};

inline Failure failureOf(Kind kind) {
    switch (kind) {
        case Kind::TCP_LISTENER:
        case Kind::TCP_STREAM:
        case Kind::UDP_SOCKET:
            return Failure::NETWORK_ERROR;
        case Kind::TLS_CONTEXT:
        case Kind::TLS_LISTENER:
        case Kind::TLS_SESSION:
            return Failure::TLS_ERROR;
        case Kind::EVENT_LOOP:
        case Kind::PROCESS:
            return Failure::FALSE_VALUE;
        default:
            return Failure::RAISES;
    }
}

inline const char* failureRefiner(Failure failure) {
    switch (failure) {
        case Failure::NETWORK_ERROR: return "network-error?";
        case Failure::TLS_ERROR: return "tls-error?";
        case Failure::FALSE_VALUE: return "a truth test (if h ...) or (not h)";
        default: return "nothing";
    }
}

enum class Role : uint8_t {
    SAFE = 0,         // no handle effect (may still take a procedure argument)
    FACTORY,          // returns a fresh owned handle of `produces`
    CLOSE,            // consumes the handle at handle_args[0]
    OBSERVE,          // borrows the handles at handle_args for the call only
    ERROR_PREDICATE,  // refines a failure value of family `family`
    ERROR_OBSERVE,    // reads a failure value on the failed branch only
    LEND              // opens or takes a handle, lends it to a procedure, closes it on every exit
};

/** How a higher-order operation invokes a procedure argument. */
enum class ProcMode : uint8_t {
    NONE = 0,
    ONCE,    // called exactly once, synchronously, before the operation returns
    MANY,    // called zero or more times, synchronously; never retained
    THREAD   // runs concurrently on another thread
};

struct HandleArg {
    int8_t index = -1;
    uint32_t accepts = 0;
    bool optional = false;  // absent, or an untracked non-OS value (string port, current port)
};

struct ProcArg {
    int8_t index = -1;
    ProcMode mode = ProcMode::NONE;
};

struct Contract {
    const char* name;
    Role role;
    Kind produces;
    HandleArg handle_args[2];
    ProcArg proc_args[2];
    bool consumes_handle;  // LEND: the handle at handle_args[0] is consumed
    Failure family;        // ERROR_PREDICATE: the family it refines
    bool may_raise;
};

constexpr HandleArg H(int8_t index, uint32_t accepts, bool optional = false) {
    return HandleArg{index, accepts, optional};
}
constexpr HandleArg NOH() { return HandleArg{}; }
constexpr ProcArg P(int8_t index, ProcMode mode) { return ProcArg{index, mode}; }
constexpr ProcArg NOP() { return ProcArg{}; }

// ── core.net ────────────────────────────────────────────────────────────
inline constexpr Contract kCoreNetContracts[] = {
    {"tcp-listen", Role::FACTORY, Kind::TCP_LISTENER, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tcp-connect", Role::FACTORY, Kind::TCP_STREAM, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tcp-accept", Role::FACTORY, Kind::TCP_STREAM, {H(0, bit(Kind::TCP_LISTENER)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"udp-bind", Role::FACTORY, Kind::UDP_SOCKET, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tcp-read", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::TCP_STREAM)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tcp-write", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::TCP_STREAM)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"udp-sendto", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::UDP_SOCKET)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"udp-recvfrom", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::UDP_SOCKET)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"net-local-address", Role::OBSERVE, Kind::NONE, {H(0, kAnySocket), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"net-peer-address", Role::OBSERVE, Kind::NONE, {H(0, kAnySocket), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"net-local-port", Role::OBSERVE, Kind::NONE, {H(0, kAnySocket), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"net-peer-port", Role::OBSERVE, Kind::NONE, {H(0, kAnySocket), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"net-nonblocking!", Role::OBSERVE, Kind::NONE, {H(0, kAnySocket), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"tcp-close", Role::CLOSE, Kind::NONE, {H(0, kAnyTcp), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"udp-close", Role::CLOSE, Kind::NONE, {H(0, bit(Kind::UDP_SOCKET)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"network-error?", Role::ERROR_PREDICATE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::NETWORK_ERROR, false},
    {"network-error-code", Role::ERROR_OBSERVE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::NETWORK_ERROR, false},
    {"network-error-operation", Role::ERROR_OBSERVE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::NETWORK_ERROR, false},
};

// ── core.tls ────────────────────────────────────────────────────────────
// A context outlives nothing: "active sessions retain it until they finish",
// so listen/connect borrow the context and the context closes independently.
inline constexpr Contract kCoreTlsContracts[] = {
    {"tls-available?", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"tls-client-context", Role::FACTORY, Kind::TLS_CONTEXT, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tls-server-context", Role::FACTORY, Kind::TLS_CONTEXT, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tls-context-close", Role::CLOSE, Kind::NONE, {H(0, bit(Kind::TLS_CONTEXT)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"tls-listen", Role::FACTORY, Kind::TLS_LISTENER, {H(0, bit(Kind::TLS_CONTEXT)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tls-listener-port", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::TLS_LISTENER)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"tls-accept", Role::FACTORY, Kind::TLS_SESSION, {H(0, bit(Kind::TLS_LISTENER)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tls-connect", Role::FACTORY, Kind::TLS_SESSION, {H(0, bit(Kind::TLS_CONTEXT)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tls-read", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::TLS_SESSION)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tls-write", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::TLS_SESSION)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tls-session-close", Role::CLOSE, Kind::NONE, {H(0, bit(Kind::TLS_SESSION)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"tls-listener-close", Role::CLOSE, Kind::NONE, {H(0, bit(Kind::TLS_LISTENER)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"tls-error?", Role::ERROR_PREDICATE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::TLS_ERROR, false},
    {"tls-error-code", Role::ERROR_OBSERVE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::TLS_ERROR, false},
    {"tls-error-operation", Role::ERROR_OBSERVE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::TLS_ERROR, false},
};

// ── core.threads ────────────────────────────────────────────────────────
// thread-join consumes the thread handle: a spawned thread must be joined on
// every path. with-mutex borrows the mutex and runs its thunk exactly once.
inline constexpr Contract kCoreThreadsContracts[] = {
    {"make-mutex", Role::FACTORY, Kind::MUTEX, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"mutex-lock!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::MUTEX)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"mutex-trylock!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::MUTEX)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"mutex-unlock!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::MUTEX)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"mutex-destroy!", Role::CLOSE, Kind::NONE, {H(0, bit(Kind::MUTEX)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"with-mutex", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::MUTEX)), NOH()}, {P(1, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
    {"make-condvar", Role::FACTORY, Kind::CONDVAR, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"condvar-wait!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::CONDVAR)), H(1, bit(Kind::MUTEX))}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"condvar-signal!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::CONDVAR)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"condvar-broadcast!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::CONDVAR)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"condvar-destroy!", Role::CLOSE, Kind::NONE, {H(0, bit(Kind::CONDVAR)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"make-thread", Role::FACTORY, Kind::THREAD, {NOH(), NOH()}, {P(0, ProcMode::THREAD), NOP()}, false, Failure::RAISES, true},
    {"thread-join", Role::CLOSE, Kind::NONE, {H(0, bit(Kind::THREAD)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"thread?", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::THREAD), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"thread-result-ready?", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::THREAD)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
};

// ── agent.subprocess ────────────────────────────────────────────────────
// A process handle owns the child and its pipes; process-destroy terminates
// and reaps the child if needed and releases the handle. The run-* helpers
// spawn, wait, and release internally, so they have no handle effect.
inline constexpr Contract kAgentSubprocessContracts[] = {
    {"process-spawn", Role::FACTORY, Kind::PROCESS, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-spawn-shell", Role::FACTORY, Kind::PROCESS, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-spawn-argv", Role::FACTORY, Kind::PROCESS, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-spawn-argv-env", Role::FACTORY, Kind::PROCESS, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-spawn-argv-options", Role::FACTORY, Kind::PROCESS, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-spawn-argv-with-options", Role::FACTORY, Kind::PROCESS, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-write-stdin", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-close-stdin", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-read-stdout", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-read-stderr", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-read-all-stdout", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-read-all-stderr", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-read-stdout-bytes", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-read-stderr-bytes", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-wait", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-running?", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"process-exit-code", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"process-pid", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"process-kill", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-destroy", Role::CLOSE, Kind::NONE, {H(0, bit(Kind::PROCESS)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"run-command", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"run-command-capture", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"run-argv", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"run-argv-capture", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
};

struct ModuleContract {
    const char* module;
    const char* source;   // path below the SDK lib directory
    const char* sha256;   // reviewed source
    const Contract* contracts;
    size_t count;
};

// The pins are checked against lib/ by scripts/strict_resource_pins.py --check
// (CTest strict_resource_contract_pins).
inline constexpr ModuleContract kModuleContracts[] = {
    {"core.net", "core/net.esk",
     "c160df956437b9cff155dcd030fa90586cd9d3c26e9822c453948c336ad3f680",
     kCoreNetContracts, sizeof(kCoreNetContracts) / sizeof(kCoreNetContracts[0])},
    {"core.tls", "core/tls.esk",
     "889c5e60cffdd3ed931cbcfb65a53b498d63237dc59fe498d778a2ca87c30ff5",
     kCoreTlsContracts, sizeof(kCoreTlsContracts) / sizeof(kCoreTlsContracts[0])},
    {"core.threads", "core/threads.esk",
     "c7001ab6a8238c2acbf39412e3b59f7f246460d614b319f0f1956a045a398fb6",
     kCoreThreadsContracts, sizeof(kCoreThreadsContracts) / sizeof(kCoreThreadsContracts[0])},
    {"agent.subprocess", "agent/subprocess.esk",
     "4691729a42034d3be81f5d64385ae65cdb5a8009f6c647c0c8ceffe509da403e",
     kAgentSubprocessContracts,
     sizeof(kAgentSubprocessContracts) / sizeof(kAgentSubprocessContracts[0])},
};

// ── compiler builtins (apply only when the name is not lexically rebound) ──
inline constexpr Contract kBuiltinContracts[] = {
    // File ports. Opening raises on failure; closing an R7RS port is idempotent
    // at run time, so the static rule is what rejects a second close.
    {"open-input-file", Role::FACTORY, Kind::INPUT_PORT, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"open-binary-input-file", Role::FACTORY, Kind::INPUT_PORT, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"open-output-file", Role::FACTORY, Kind::OUTPUT_PORT, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"open-binary-output-file", Role::FACTORY, Kind::OUTPUT_PORT, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"open-output-file-append", Role::FACTORY, Kind::OUTPUT_PORT, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"close-port", Role::CLOSE, Kind::NONE, {H(0, kAnyPort), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"close-input-port", Role::CLOSE, Kind::NONE, {H(0, bit(Kind::INPUT_PORT)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"close-output-port", Role::CLOSE, Kind::NONE, {H(0, bit(Kind::OUTPUT_PORT)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"read-line", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::INPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"read-char", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::INPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"peek-char", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::INPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"read", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::INPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"read-u8", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::INPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"peek-u8", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::INPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"char-ready?", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::INPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"u8-ready?", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::INPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"read-string", Role::OBSERVE, Kind::NONE, {H(1, bit(Kind::INPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"read-bytevector", Role::OBSERVE, Kind::NONE, {H(1, bit(Kind::INPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"write-string", Role::OBSERVE, Kind::NONE, {H(1, bit(Kind::OUTPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"write-char", Role::OBSERVE, Kind::NONE, {H(1, bit(Kind::OUTPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"write-u8", Role::OBSERVE, Kind::NONE, {H(1, bit(Kind::OUTPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"write-bytevector", Role::OBSERVE, Kind::NONE, {H(1, bit(Kind::OUTPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"display", Role::OBSERVE, Kind::NONE, {H(1, bit(Kind::OUTPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"write", Role::OBSERVE, Kind::NONE, {H(1, bit(Kind::OUTPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"write-shared", Role::OBSERVE, Kind::NONE, {H(1, bit(Kind::OUTPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"write-simple", Role::OBSERVE, Kind::NONE, {H(1, bit(Kind::OUTPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"newline", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::OUTPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"flush-output-port", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::OUTPUT_PORT), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"input-port-open?", Role::OBSERVE, Kind::NONE, {H(0, kAnyPort, true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"output-port-open?", Role::OBSERVE, Kind::NONE, {H(0, kAnyPort, true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"port?", Role::OBSERVE, Kind::NONE, {H(0, kAnyPort, true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"input-port?", Role::OBSERVE, Kind::NONE, {H(0, kAnyPort, true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"output-port?", Role::OBSERVE, Kind::NONE, {H(0, kAnyPort, true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"textual-port?", Role::OBSERVE, Kind::NONE, {H(0, kAnyPort, true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"binary-port?", Role::OBSERVE, Kind::NONE, {H(0, kAnyPort, true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"eof-object", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"eof-object?", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"current-input-port", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"current-output-port", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"current-error-port", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"open-input-string", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"open-output-string", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"get-output-string", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    // Borrow scopes: the handle is lent to the procedure and closed on normal
    // return, on an exception, and on a continuation escape.
    {"call-with-input-file", Role::LEND, Kind::INPUT_PORT, {NOH(), NOH()}, {P(1, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
    {"call-with-output-file", Role::LEND, Kind::OUTPUT_PORT, {NOH(), NOH()}, {P(1, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
    {"call-with-port", Role::LEND, Kind::NONE, {H(0, kAnyPort), NOH()}, {P(1, ProcMode::ONCE), NOP()}, true, Failure::RAISES, true},
    {"with-input-from-file", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {P(1, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
    {"with-output-to-file", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {P(1, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
    // Event loop: #f when the platform or kernel refuses a loop.
    {"make-event-loop", Role::FACTORY, Kind::EVENT_LOOP, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"event-loop-add-fd!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::EVENT_LOOP)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"event-loop-remove-fd!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::EVENT_LOOP)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"event-loop-poll", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::EVENT_LOOP)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"event-loop-close", Role::CLOSE, Kind::NONE, {H(0, bit(Kind::EVENT_LOOP)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"event-loop-backend", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    // Synchronous higher-order operations that never retain their procedure.
    {"for-each", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {P(0, ProcMode::MANY), NOP()}, false, Failure::RAISES, true},
    {"map", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {P(0, ProcMode::MANY), NOP()}, false, Failure::RAISES, true},
    {"vector-for-each", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {P(0, ProcMode::MANY), NOP()}, false, Failure::RAISES, true},
    {"vector-map", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {P(0, ProcMode::MANY), NOP()}, false, Failure::RAISES, true},
    {"string-for-each", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {P(0, ProcMode::MANY), NOP()}, false, Failure::RAISES, true},
    // The handler runs in the dynamic context of a raise and may run more
    // than once; the thunk runs exactly once.
    {"with-exception-handler", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {P(0, ProcMode::MANY), P(1, ProcMode::ONCE)}, false, Failure::RAISES, true},
};

inline const Contract* findContract(const Contract* table, size_t count, const char* name) {
    if (!name) return nullptr;
    for (size_t i = 0; i < count; ++i)
        if (std::strcmp(table[i].name, name) == 0) return &table[i];
    return nullptr;
}

inline const Contract* findBuiltinContract(const char* name) {
    return findContract(kBuiltinContracts, sizeof(kBuiltinContracts) / sizeof(kBuiltinContracts[0]), name);
}

inline const ModuleContract* findModuleContract(const char* module) {
    if (!module) return nullptr;
    for (const auto& entry : kModuleContracts)
        if (std::strcmp(entry.module, module) == 0) return &entry;
    return nullptr;
}

}  // namespace eshkol::strict_resources
