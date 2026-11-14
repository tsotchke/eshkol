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
    SEMAPHORE,
    BARRIER,
    PROCESS,
    PROCESS_STREAM,
    UNIX_LISTENER,
    UNIX_CONNECTING,
    UNIX_STREAM,
    UNIX_CANCELLED,
    CLEANUP_TICKET,
    CLEANUP_RECOVERY,
    COUNT
};

constexpr uint32_t bit(Kind kind) { return 1u << static_cast<unsigned>(kind); }

constexpr uint32_t kAnySocket =
    bit(Kind::TCP_LISTENER) | bit(Kind::TCP_STREAM) | bit(Kind::UDP_SOCKET);
constexpr uint32_t kAnyTcp = bit(Kind::TCP_LISTENER) | bit(Kind::TCP_STREAM);
constexpr uint32_t kAnyUnix = bit(Kind::UNIX_LISTENER) | bit(Kind::UNIX_CONNECTING) |
    bit(Kind::UNIX_STREAM) | bit(Kind::UNIX_CANCELLED);
constexpr uint32_t kUnixPeer = bit(Kind::UNIX_CONNECTING) | bit(Kind::UNIX_STREAM);
// A branch that may have cancelled remains owned but can no longer prove I/O
// admissible. A branch that may still be connecting must finish before I/O.
constexpr Kind joinProtocolKind(Kind a, Kind b) {
    if (a == b) return a;
    if ((bit(a) & kAnyUnix) && (bit(b) & kAnyUnix)) {
        if (a == Kind::UNIX_CANCELLED || b == Kind::UNIX_CANCELLED)
            return Kind::UNIX_CANCELLED;
        if ((bit(a) & kUnixPeer) && (bit(b) & kUnixPeer))
            return Kind::UNIX_CONNECTING;
    }
    return Kind::NONE;
}
constexpr uint32_t kAnyCleanup = bit(Kind::CLEANUP_TICKET) | bit(Kind::CLEANUP_RECOVERY);
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
        case Kind::SEMAPHORE: return "semaphore";
        case Kind::BARRIER: return "barrier";
        case Kind::PROCESS: return "process";
        case Kind::PROCESS_STREAM: return "subprocess stream scope";
        case Kind::UNIX_LISTENER: return "Unix listener";
        case Kind::UNIX_CONNECTING: return "connecting Unix stream";
        case Kind::UNIX_STREAM: return "open Unix stream";
        case Kind::UNIX_CANCELLED: return "cancelled Unix socket";
        case Kind::CLEANUP_TICKET:
        case Kind::CLEANUP_RECOVERY: return "cleanup ticket";
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
        case Kind::CLEANUP_RECOVERY:
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
    TRANSITION,      // retains ownership, changes the protocol kind to `produces`
    UNSUPPORTED,     // no sound ownership summary yet; strict compilation refuses the call
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
    // The operation takes ownership of this handle: it moves into the value the
    // operation produces (or into the scope it opens), which is from then on the
    // only thing that may close it. The caller's owner becomes moved-from, so a
    // later use or close of it is reported.
    bool consumed = false;
    bool retired = false;  // recovery consumes an already-closed identity, never I/O authority
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
    return HandleArg{index, accepts, optional, false};
}
/** A handle argument whose ownership moves into the operation's result. */
constexpr HandleArg HMOVE(int8_t index, uint32_t accepts) {
    return HandleArg{index, accepts, false, true};
}
constexpr HandleArg HRETIRED(int8_t index, uint32_t accepts) {
    return HandleArg{index, accepts, false, true, true};
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
    {"network-error-reason", Role::ERROR_OBSERVE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::NETWORK_ERROR, false},
    // Bytevector-range transfers. They borrow the socket for the call and
    // return a count, the eof object, or a network error; the handle stays
    // open and its owner still has to close it.
    {"tcp-read!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::TCP_STREAM)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tcp-write!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::TCP_STREAM)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tcp-write-all!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::TCP_STREAM)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    // Half-close and cancellation leave the handle open: both are observations
    // of a live socket, and tcp-close/udp-close remains the only close site.
    {"tcp-shutdown", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::TCP_STREAM)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"net-cancel!", Role::OBSERVE, Kind::NONE, {H(0, kAnySocket), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"udp-send-to!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::UDP_SOCKET)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"udp-receive-from!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::UDP_SOCKET)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    // Borrow scopes: each opens its own handle, lends it to the procedure, and
    // closes it exactly once on return, on a raise and on a continuation
    // escape. A failed acquisition returns its network error without calling
    // the procedure, so the scope never hands out an unrefined handle.
    {"call-with-tcp-listener", Role::LEND, Kind::TCP_LISTENER, {NOH(), NOH()}, {P(2, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
    {"call-with-tcp-connection", Role::LEND, Kind::TCP_STREAM, {NOH(), NOH()}, {P(3, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
    {"call-with-tcp-accept", Role::LEND, Kind::TCP_STREAM, {H(0, bit(Kind::TCP_LISTENER)), NOH()}, {P(2, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
    {"call-with-udp-socket", Role::LEND, Kind::UDP_SOCKET, {NOH(), NOH()}, {P(2, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
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
    {"tls-error-reason", Role::ERROR_OBSERVE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::TLS_ERROR, false},
    {"tls-client-context/options", Role::FACTORY, Kind::TLS_CONTEXT, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    // Handshakes over a core.net stream: the stream moves into the session,
    // which becomes its only close site — a failed handshake closes it too.
    {"tls-client-start", Role::FACTORY, Kind::TLS_SESSION, {H(0, bit(Kind::TLS_CONTEXT)), HMOVE(1, bit(Kind::TCP_STREAM))}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tls-server-start", Role::FACTORY, Kind::TLS_SESSION, {H(0, bit(Kind::TLS_CONTEXT)), HMOVE(1, bit(Kind::TCP_STREAM))}, {NOP(), NOP()}, false, Failure::RAISES, true},
    // Bytevector-range transfers borrow the session for the call only.
    {"tls-read!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::TLS_SESSION)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tls-write!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::TLS_SESSION)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tls-write-all!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::TLS_SESSION)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"tls-cancel!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::TLS_SESSION)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    // Borrow scopes over a session: the context is borrowed for the call, a
    // handed-in stream moves into the session, and the session closes exactly
    // once on return, on a raise and on a continuation escape.
    {"call-with-tls-connection", Role::LEND, Kind::TLS_SESSION, {H(0, bit(Kind::TLS_CONTEXT)), NOH()}, {P(5, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
    {"call-with-tls-accept", Role::LEND, Kind::TLS_SESSION, {H(0, bit(Kind::TLS_LISTENER)), NOH()}, {P(2, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
    {"call-with-tls-client", Role::LEND, Kind::TLS_SESSION, {H(0, bit(Kind::TLS_CONTEXT)), HMOVE(1, bit(Kind::TCP_STREAM))}, {P(4, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
    {"call-with-tls-server", Role::LEND, Kind::TLS_SESSION, {H(0, bit(Kind::TLS_CONTEXT)), HMOVE(1, bit(Kind::TCP_STREAM))}, {P(3, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
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
    {"condvar-wait-timeout!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::CONDVAR)), H(1, bit(Kind::MUTEX))}, {NOP(), NOP()}, false, Failure::RAISES, false},
    // Counting semaphores and reusable barriers own the mutex and condition
    // variable they are built from, so each is one owned handle with one
    // close site. Their predicates answer for any value, tracked or not.
    {"make-semaphore", Role::FACTORY, Kind::SEMAPHORE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"semaphore?", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::SEMAPHORE), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"semaphore-acquire!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::SEMAPHORE)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"semaphore-try-acquire!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::SEMAPHORE)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"semaphore-release!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::SEMAPHORE)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"semaphore-value", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::SEMAPHORE)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"semaphore-destroy!", Role::CLOSE, Kind::NONE, {H(0, bit(Kind::SEMAPHORE)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"make-barrier", Role::FACTORY, Kind::BARRIER, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"barrier?", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::BARRIER), true), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"barrier-wait!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::BARRIER)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"barrier-parties", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::BARRIER)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"barrier-generation", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::BARRIER)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"barrier-destroy!", Role::CLOSE, Kind::NONE, {H(0, bit(Kind::BARRIER)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
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
    // Aggregate PROCESS cleanup IDs remain owned by the runtime. Their
    // extraction/consumption needs container ownership support; strict calls
    // are refused explicitly rather than silently treated as effect-free.
    {"call-with-subprocess-streams", Role::LEND, Kind::PROCESS_STREAM, {H(0, bit(Kind::EVENT_LOOP)), NOH()}, {P(3, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
    {"process-stream-read!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS_STREAM)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-stream-poll", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS_STREAM)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-stream-status", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS_STREAM)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-cancel!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::PROCESS_STREAM)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-stream-read-status", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"process-stream-read-count", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"process-stream-read-error", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"process-stream-result-value", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"process-stream-result-cleanup-status", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"process-stream-result-cleanup-ticket", Role::UNSUPPORTED, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-cleanup-tickets", Role::UNSUPPORTED, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-cleanup-status", Role::UNSUPPORTED, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-cleanup-consume!", Role::UNSUPPORTED, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-stream-runtime-shutdown!", Role::UNSUPPORTED, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"process-cleanup-drain!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::EVENT_LOOP)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
};

// Unix acquisition publishes connecting ownership before readiness. Cancellation
// changes the protocol kind but does not consume ownership. Accepted close
// consumes that owner; an explicit close request produces a separate ticket.
// Runtime completion/retryability gates ticket release and loop teardown.
inline constexpr Contract kAgentUnixStreamSocketContracts[] = {
    {"unix-stream-socket-listen", Role::FACTORY, Kind::UNIX_LISTENER, {H(0, bit(Kind::EVENT_LOOP)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"unix-stream-socket-connect-begin", Role::FACTORY, Kind::UNIX_CONNECTING, {H(0, bit(Kind::EVENT_LOOP)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"unix-stream-socket-accept", Role::FACTORY, Kind::UNIX_STREAM, {H(0, bit(Kind::UNIX_LISTENER)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"unix-stream-socket-connect-finish!", Role::TRANSITION, Kind::UNIX_STREAM, {H(0, kUnixPeer), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"unix-stream-socket-read!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::UNIX_STREAM)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"unix-stream-socket-write!", Role::OBSERVE, Kind::NONE, {H(0, bit(Kind::UNIX_STREAM)), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"unix-stream-socket-poll", Role::OBSERVE, Kind::NONE, {H(0, kAnyUnix), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"unix-stream-socket-cancel!", Role::TRANSITION, Kind::UNIX_CANCELLED, {H(0, kAnyUnix), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"unix-stream-socket-close!", Role::CLOSE, Kind::NONE, {H(0, kAnyUnix), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"unix-stream-socket-close-request!", Role::FACTORY, Kind::CLEANUP_TICKET, {HMOVE(0, kAnyUnix), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"unix-stream-socket-cleanup-result", Role::OBSERVE, Kind::NONE, {H(0, kAnyCleanup), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"unix-stream-socket-cleanup-retry!", Role::OBSERVE, Kind::NONE, {H(0, kAnyCleanup), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"unix-stream-socket-cleanup-release!", Role::CLOSE, Kind::NONE, {H(0, kAnyCleanup), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"unix-stream-socket-cleanup-ticket", Role::FACTORY, Kind::CLEANUP_RECOVERY, {HRETIRED(0, kAnyUnix), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"unix-stream-socket-cleanup-state", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"unix-stream-socket-cleanup-stage", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"unix-stream-socket-cleanup-os-error", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"unix-stream-socket-cleanup-retryable?", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"unix-stream-socket-read-status", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"unix-stream-socket-read-count", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"unix-stream-socket-read-error", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"unix-stream-socket-write-status", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"unix-stream-socket-write-count", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"unix-stream-socket-write-error", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"unix-stream-socket-poll-status", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"unix-stream-socket-poll-interests", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"unix-stream-socket-poll-error", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"unix-stream-socket-error-operation", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"unix-stream-socket-error-code", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"unix-stream-socket-error-os-code", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"unix-stream-socket-planned?", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, false},
    {"call-with-unix-stream-listener", Role::LEND, Kind::UNIX_LISTENER, {H(0, bit(Kind::EVENT_LOOP)), NOH()}, {P(2, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
    {"call-with-unix-stream-connection", Role::LEND, Kind::UNIX_STREAM, {H(0, bit(Kind::EVENT_LOOP)), NOH()}, {P(3, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
    {"call-with-unix-stream-accept", Role::LEND, Kind::UNIX_STREAM, {H(0, bit(Kind::UNIX_LISTENER)), NOH()}, {P(2, ProcMode::ONCE), NOP()}, false, Failure::RAISES, true},
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
     "7ada0e09a9ffa49fc6319455d0402036e85f47fc4d42e9059d491191f61ccb05",
     kCoreNetContracts, sizeof(kCoreNetContracts) / sizeof(kCoreNetContracts[0])},
    {"core.tls", "core/tls.esk",
     "91f95582ab932508335fd83df585b956a74fd83264b7cff90823b5bf061f4398",
     kCoreTlsContracts, sizeof(kCoreTlsContracts) / sizeof(kCoreTlsContracts[0])},
    {"core.threads", "core/threads.esk",
     "79612384c29b0fdf28b738f6588062a1916cd589588b3e9a617c112ac7de572b",
     kCoreThreadsContracts, sizeof(kCoreThreadsContracts) / sizeof(kCoreThreadsContracts[0])},
    {"agent.subprocess", "agent/subprocess.esk",
     "f31051e751d6323fd1d61016f74e776b2fc9f823fe3005d2d0764bebbf1ee5bb",
     kAgentSubprocessContracts,
     sizeof(kAgentSubprocessContracts) / sizeof(kAgentSubprocessContracts[0])},
    {"agent.unix_stream_socket", "agent/unix_stream_socket.esk",
     "404eef05d9cbfe281b4326c2c0ad2904309f8bef497e8b603b5139b685b4ae11",
     kAgentUnixStreamSocketContracts,
     sizeof(kAgentUnixStreamSocketContracts) / sizeof(kAgentUnixStreamSocketContracts[0])},
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
    // Bytevectors carry no handle: they are the buffers the range transfers of
    // core.net and core.tls read into and write out of, so a strict program
    // doing bounded I/O needs them to have a summary of their own.
    {"make-bytevector", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"bytevector-length", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"bytevector-u8-ref", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
    {"bytevector-u8-set!", Role::SAFE, Kind::NONE, {NOH(), NOH()}, {NOP(), NOP()}, false, Failure::RAISES, true},
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
