#include <eshkol/backend/llvm_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

namespace {
constexpr size_t LIB_INIT_AST_CHUNK_SIZE = 4;
constexpr size_t LIB_INIT_LAMBDA_SEXPR_CHUNK_SIZE = 4;
}

void EshkolLLVMCodeGen::registerBuiltinReturnTypes() {
        using namespace eshkol::hott;

        // Arithmetic functions return Number (polymorphic)
        function_return_types["+"] = BuiltinTypes::Number;
        function_return_types["-"] = BuiltinTypes::Number;
        function_return_types["*"] = BuiltinTypes::Number;
        function_return_types["/"] = BuiltinTypes::Real;  // Division always returns real

        // Comparison functions return Boolean
        function_return_types["<"] = BuiltinTypes::Boolean;
        function_return_types[">"] = BuiltinTypes::Boolean;
        function_return_types["<="] = BuiltinTypes::Boolean;
        function_return_types[">="] = BuiltinTypes::Boolean;
        function_return_types["="] = BuiltinTypes::Boolean;
        function_return_types["eq?"] = BuiltinTypes::Boolean;
        function_return_types["equal?"] = BuiltinTypes::Boolean;
        function_return_types["null?"] = BuiltinTypes::Boolean;
        function_return_types["pair?"] = BuiltinTypes::Boolean;
        function_return_types["list?"] = BuiltinTypes::Boolean;
        function_return_types["number?"] = BuiltinTypes::Boolean;
        function_return_types["zero?"] = BuiltinTypes::Boolean;
        function_return_types["positive?"] = BuiltinTypes::Boolean;
        function_return_types["negative?"] = BuiltinTypes::Boolean;
        function_return_types["even?"] = BuiltinTypes::Boolean;
        function_return_types["odd?"] = BuiltinTypes::Boolean;
        function_return_types["nan?"] = BuiltinTypes::Boolean;
        function_return_types["infinite?"] = BuiltinTypes::Boolean;
        function_return_types["finite?"] = BuiltinTypes::Boolean;

        // Math functions return Float64
        function_return_types["sin"] = BuiltinTypes::Float64;
        function_return_types["cos"] = BuiltinTypes::Float64;
        function_return_types["tan"] = BuiltinTypes::Float64;
        function_return_types["exp"] = BuiltinTypes::Float64;
        function_return_types["log"] = BuiltinTypes::Float64;
        function_return_types["sqrt"] = BuiltinTypes::Float64;
        function_return_types["abs"] = BuiltinTypes::Number;
        function_return_types["fabs"] = BuiltinTypes::Float64;

        // List functions
        function_return_types["list"] = BuiltinTypes::List;
        function_return_types["cons"] = BuiltinTypes::List;
        function_return_types["car"] = BuiltinTypes::Value;  // Can be any type
        function_return_types["cdr"] = BuiltinTypes::List;
        function_return_types["length"] = BuiltinTypes::Int64;
        function_return_types["append"] = BuiltinTypes::List;
        function_return_types["reverse"] = BuiltinTypes::List;
        function_return_types["map"] = BuiltinTypes::List;
        function_return_types["filter"] = BuiltinTypes::List;
        function_return_types["take"] = BuiltinTypes::List;
        function_return_types["drop"] = BuiltinTypes::List;
        function_return_types["range"] = BuiltinTypes::List;
        function_return_types["list-copy"] = BuiltinTypes::List;
        function_return_types["list-set!"] = BuiltinTypes::List;
        function_return_types["list*"] = BuiltinTypes::List;
        function_return_types["acons"] = BuiltinTypes::List;

        // eval returns any type
        function_return_types["eval"] = BuiltinTypes::Value;

        // Vector functions
        function_return_types["vector"] = BuiltinTypes::Vector;
        function_return_types["make-vector"] = BuiltinTypes::Vector;
        function_return_types["vector-length"] = BuiltinTypes::Int64;
        function_return_types["vector-ref"] = BuiltinTypes::Value;
        function_return_types["vector-copy"] = BuiltinTypes::Vector;
        function_return_types["vector-append"] = BuiltinTypes::Vector;

        // R7RS error-object accessors
        function_return_types["error-object?"] = BuiltinTypes::Boolean;
        function_return_types["error-object-message"] = BuiltinTypes::Value;
        function_return_types["error-object-irritants"] = BuiltinTypes::List;
        function_return_types["vector->list"] = BuiltinTypes::List;
        function_return_types["list->vector"] = BuiltinTypes::Vector;

        // Type conversions
        function_return_types["exact->inexact"] = BuiltinTypes::Float64;
        // inexact->exact is NOT integer-valued: the exact value of a
        // fractional double is a rational (and of a huge one, a bignum), so
        // the static claim has to be the general Number, not Int64.
        function_return_types["inexact->exact"] = BuiltinTypes::Number;
        function_return_types["inexact"] = BuiltinTypes::Float64;
        function_return_types["exact"] = BuiltinTypes::Number;
        function_return_types["exact-integer?"] = BuiltinTypes::Boolean;
        function_return_types["square"] = BuiltinTypes::Number;
        function_return_types["volatile-load"] = BuiltinTypes::Value;
        function_return_types["volatile-store!"] = BuiltinTypes::Null;
        function_return_types["atomic-load"] = BuiltinTypes::Value;
        function_return_types["atomic-store!"] = BuiltinTypes::Null;
        function_return_types["atomic-exchange!"] = BuiltinTypes::Value;
        function_return_types["atomic-compare-exchange!"] = BuiltinTypes::Value;
        function_return_types["atomic-fetch-add!"] = BuiltinTypes::Value;
        function_return_types["atomic-fetch-sub!"] = BuiltinTypes::Value;
        function_return_types["atomic-fetch-and!"] = BuiltinTypes::Value;
        function_return_types["atomic-fetch-or!"] = BuiltinTypes::Value;
        function_return_types["atomic-fetch-xor!"] = BuiltinTypes::Value;
        function_return_types["target-intrinsic"] = BuiltinTypes::Value;
        function_return_types["compiler-fence"] = BuiltinTypes::Null;
        function_return_types["memory-fence"] = BuiltinTypes::Null;
        function_return_types["addr-of"] = BuiltinTypes::Pointer;
        function_return_types["null-ptr"] = BuiltinTypes::Pointer;
        function_return_types["ptr->usize"] = BuiltinTypes::USize;
        function_return_types["usize->ptr"] = BuiltinTypes::Pointer;
        function_return_types["ptr-add"] = BuiltinTypes::Pointer;

        // R7RS division
        function_return_types["floor-quotient"] = BuiltinTypes::Int64;
        function_return_types["floor-remainder"] = BuiltinTypes::Int64;
        function_return_types["floor/"] = BuiltinTypes::Value;
        function_return_types["truncate-quotient"] = BuiltinTypes::Int64;
        function_return_types["truncate-remainder"] = BuiltinTypes::Int64;
        function_return_types["truncate/"] = BuiltinTypes::Value;

        // R7RS port predicates
        function_return_types["textual-port?"] = BuiltinTypes::Boolean;
        function_return_types["binary-port?"] = BuiltinTypes::Boolean;

        // R7RS binary I/O
        function_return_types["open-binary-input-file"] = BuiltinTypes::Value;
        function_return_types["open-binary-output-file"] = BuiltinTypes::Value;
        function_return_types["read-u8"] = BuiltinTypes::Value;   // int or eof-object
        function_return_types["peek-u8"] = BuiltinTypes::Value;  // int or eof-object
        function_return_types["write-u8"] = BuiltinTypes::Null;
        function_return_types["read-bytevector"] = BuiltinTypes::Value;  // bytevector or eof-object
        function_return_types["write-bytevector"] = BuiltinTypes::Null;
        function_return_types["read-bytevector!"] = BuiltinTypes::Value;
        function_return_types["u8-ready?"] = BuiltinTypes::Boolean;

        // R7RS system
        function_return_types["eof-object"] = BuiltinTypes::Value;
        function_return_types["emergency-exit"] = BuiltinTypes::Null;
        function_return_types["current-second"] = BuiltinTypes::Float64;
        function_return_types["current-jiffy"] = BuiltinTypes::Int64;
        function_return_types["jiffies-per-second"] = BuiltinTypes::Int64;
        function_return_types["features"] = BuiltinTypes::List;

        // R7RS aliases
        function_return_types["get-environment-variable"] = BuiltinTypes::String;
        function_return_types["delete-file"] = BuiltinTypes::Boolean;
        function_return_types["close-input-port"] = BuiltinTypes::Null;
        function_return_types["close-output-port"] = BuiltinTypes::Null;
        function_return_types["string-copy!"] = BuiltinTypes::Null;

        // v1.2 system/path/process builtins
        function_return_types["os-type"] = BuiltinTypes::String;
        function_return_types["os-arch"] = BuiltinTypes::String;
        function_return_types["hostname"] = BuiltinTypes::String;
        function_return_types["username"] = BuiltinTypes::String;
        function_return_types["cpu-count"] = BuiltinTypes::Integer;
        function_return_types["getpid"] = BuiltinTypes::Integer;
        function_return_types["home-directory"] = BuiltinTypes::String;
        // Time API (#168)
        function_return_types["current-timestamp"] = BuiltinTypes::Number;
        function_return_types["current-time-ns"] = BuiltinTypes::Integer;
        function_return_types["format-iso8601"] = BuiltinTypes::String;
        function_return_types["parse-iso8601"] = BuiltinTypes::Integer;
        function_return_types["format-relative"] = BuiltinTypes::String;
        function_return_types["local-timezone-offset"] = BuiltinTypes::Integer;
        function_return_types["sleep-ms"] = BuiltinTypes::Null;
        function_return_types["executable-exists?"] = BuiltinTypes::Boolean;
        function_return_types["executable-path"] = BuiltinTypes::String;
        function_return_types["monotonic-time-ms"] = BuiltinTypes::Integer;
        function_return_types["__arena-used"] = BuiltinTypes::Integer;
        // #341: user-reachable region handles. The handle token is an opaque
        // exact integer; region-close returns whatever was kept (so: Value).
        function_return_types["region-open"] = BuiltinTypes::Integer;
        function_return_types["region-close"] = BuiltinTypes::Value;
        function_return_types["region-open?"] = BuiltinTypes::Boolean;
        function_return_types["ad-reset-counters!"] = BuiltinTypes::Null;
        function_return_types["ad-primal-calls"] = BuiltinTypes::Integer;
        function_return_types["ad-reverse-passes"] = BuiltinTypes::Integer;
        function_return_types["ad-tape-allocations"] = BuiltinTypes::Integer;
        function_return_types["ad-finite-difference-evals"] = BuiltinTypes::Integer;
        function_return_types["ad-note-finite-difference!"] = BuiltinTypes::Null;
        function_return_types["ad-counters"] = BuiltinTypes::List;
        function_return_types["temp-directory"] = BuiltinTypes::String;
        function_return_types["prevent-sleep"] = BuiltinTypes::Integer;
        function_return_types["allow-sleep"] = BuiltinTypes::Boolean;
        function_return_types["path-join"] = BuiltinTypes::String;
        function_return_types["path-dirname"] = BuiltinTypes::String;
        function_return_types["path-basename"] = BuiltinTypes::String;
        function_return_types["path-extname"] = BuiltinTypes::String;
        function_return_types["path-is-absolute?"] = BuiltinTypes::Boolean;
        function_return_types["path-normalize"] = BuiltinTypes::String;
        function_return_types["realpath"] = BuiltinTypes::String;
        function_return_types["file-stat"] = BuiltinTypes::Integer;
        function_return_types["file-copy"] = BuiltinTypes::Boolean;
        function_return_types["mkdir-recursive"] = BuiltinTypes::Boolean;
        function_return_types["mkdtemp"] = BuiltinTypes::String;
        function_return_types["make-temp-file"] = BuiltinTypes::String;
        function_return_types["make-temp-dir"] = BuiltinTypes::String;
        function_return_types["directory-delete-recursive"] = BuiltinTypes::Boolean;
        function_return_types["shell-quote"] = BuiltinTypes::String;
        function_return_types["fork"] = BuiltinTypes::Integer;
        function_return_types["execv"] = BuiltinTypes::Boolean;
        function_return_types["process-spawn"] = BuiltinTypes::Integer;
        function_return_types["process-wait"] = BuiltinTypes::Integer;
        function_return_types["poll-fd"] = BuiltinTypes::Boolean;
        function_return_types["tensor-save"] = BuiltinTypes::Boolean;
        function_return_types["tensor-load"] = BuiltinTypes::Value;
        // v1.2 batch 2
        function_return_types["file-chmod"] = BuiltinTypes::Boolean;
        function_return_types["symlink-create"] = BuiltinTypes::Boolean;
        function_return_types["symlink-read"] = BuiltinTypes::String;
        function_return_types["directory-walk"] = BuiltinTypes::String;
        function_return_types["mkstemp"] = BuiltinTypes::String;
        function_return_types["process-kill"] = BuiltinTypes::Boolean;
        function_return_types["file-mtime"] = BuiltinTypes::Integer;
        function_return_types["file-atime"] = BuiltinTypes::Integer;
        function_return_types["file-lock"] = BuiltinTypes::Boolean;
        function_return_types["file-unlock"] = BuiltinTypes::Boolean;
        function_return_types["path-relative"] = BuiltinTypes::String;
        function_return_types["path-resolve"] = BuiltinTypes::String;
        function_return_types["glob-expand"] = BuiltinTypes::String;
        function_return_types["glob-match"] = BuiltinTypes::Boolean;
        // v1.2 batch 3
        function_return_types["process-setpgid"] = BuiltinTypes::Boolean;
        function_return_types["process-kill-tree"] = BuiltinTypes::Boolean;
        function_return_types["process-spawn-pty"] = BuiltinTypes::Integer;
        function_return_types["process-read-nonblocking"] = BuiltinTypes::String;
        // v1.2 batch 4
        function_return_types["process-pid"] = BuiltinTypes::Integer;
        function_return_types["file-mmap"] = BuiltinTypes::String;
        function_return_types["file-munmap"] = BuiltinTypes::Boolean;
        function_return_types["unix-socket-connect"] = BuiltinTypes::Integer;
        function_return_types["socket-send"] = BuiltinTypes::Integer;
        function_return_types["socket-recv"] = BuiltinTypes::String;
        function_return_types["socket-close"] = BuiltinTypes::Boolean;
        function_return_types["term-set-scroll-region"] = BuiltinTypes::Boolean;
        function_return_types["term-reset-scroll-region"] = BuiltinTypes::Boolean;
        function_return_types["term-enable-mouse"] = BuiltinTypes::Boolean;
        function_return_types["term-disable-mouse"] = BuiltinTypes::Boolean;
        function_return_types["term-read-mouse-event"] = BuiltinTypes::Value;
        function_return_types["term-enable-alternate-screen"] = BuiltinTypes::Boolean;
        function_return_types["term-disable-alternate-screen"] = BuiltinTypes::Boolean;
        function_return_types["term-clipboard-write"] = BuiltinTypes::Boolean;
        function_return_types["term-clipboard-read"] = BuiltinTypes::String;
        function_return_types["term-hyperlink"] = BuiltinTypes::String;
        function_return_types["term-detect-capabilities"] = BuiltinTypes::Value;
        function_return_types["term-bell"] = BuiltinTypes::Boolean;
        function_return_types["fs-watch-native"] = BuiltinTypes::Integer;
        function_return_types["fs-watch-recursive"] = BuiltinTypes::Integer;
        function_return_types["fs-watch-poll"] = BuiltinTypes::String;
        function_return_types["fs-unwatch"] = BuiltinTypes::Boolean;
        function_return_types["ansi-strip"] = BuiltinTypes::String;
        function_return_types["string-display-width"] = BuiltinTypes::Integer;
        function_return_types["string-truncate-display"] = BuiltinTypes::String;
        function_return_types["url-encode"] = BuiltinTypes::String;
        function_return_types["url-decode"] = BuiltinTypes::String;
        function_return_types["url-parse"] = BuiltinTypes::Value;
        function_return_types["base64-encode-string"] = BuiltinTypes::String;
        function_return_types["base64-decode-string"] = BuiltinTypes::String;
        function_return_types["base64url-encode"] = BuiltinTypes::String;
        function_return_types["base64url-decode"] = BuiltinTypes::String;
        function_return_types["uuid-v4"] = BuiltinTypes::String;
        function_return_types["constant-time-equal?"] = BuiltinTypes::Boolean;
        function_return_types["sha256-file"] = BuiltinTypes::String;
        function_return_types["regex-compile"] = BuiltinTypes::Integer;
        function_return_types["regex-free"] = BuiltinTypes::Boolean;
        function_return_types["regex-match"] = BuiltinTypes::String;
        function_return_types["regex-match?"] = BuiltinTypes::Boolean;
        function_return_types["regex-match-groups"] = BuiltinTypes::Value;
        function_return_types["regex-split"] = BuiltinTypes::Value;
        function_return_types["diff-lines"] = BuiltinTypes::Value;
        function_return_types["fuzzy-match"] = BuiltinTypes::Value;
        function_return_types["semver-parse"] = BuiltinTypes::Value;
        function_return_types["semver-compare"] = BuiltinTypes::Integer;
        function_return_types["semver-satisfies?"] = BuiltinTypes::Boolean;
        function_return_types["make-pipe"] = BuiltinTypes::Value;
        /* ESH-0011 event loop. make-event-loop is Value, not Integer: it
         * returns an integer handle OR #f where the platform has no loop, so
         * the tagged-value shape is the honest one. poll returns a list. */
        function_return_types["make-event-loop"] = BuiltinTypes::Value;
        function_return_types["event-loop-add-fd!"] = BuiltinTypes::Boolean;
        function_return_types["event-loop-remove-fd!"] = BuiltinTypes::Boolean;
        function_return_types["event-loop-poll"] = BuiltinTypes::Value;
        function_return_types["event-loop-close"] = BuiltinTypes::Boolean;
        function_return_types["event-loop-backend"] = BuiltinTypes::String;
        function_return_types["fd-write"] = BuiltinTypes::Integer;
        function_return_types["make-line-reader"] = BuiltinTypes::Integer;
        function_return_types["line-reader-poll"] = BuiltinTypes::String;
        function_return_types["line-reader-close"] = BuiltinTypes::Boolean;
        function_return_types["fd-close"] = BuiltinTypes::Boolean;
        function_return_types["make-lru-cache"] = BuiltinTypes::Integer;
        function_return_types["lru-get"] = BuiltinTypes::Value;
        function_return_types["lru-set!"] = BuiltinTypes::Boolean;
        function_return_types["lru-has?"] = BuiltinTypes::Boolean;
        function_return_types["lru-delete!"] = BuiltinTypes::Boolean;
        function_return_types["lru-clear!"] = BuiltinTypes::Boolean;
        function_return_types["lru-size"] = BuiltinTypes::Integer;
        function_return_types["format"] = BuiltinTypes::String;
        function_return_types["_format-list"] = BuiltinTypes::String;
        function_return_types["http-server-create"] = BuiltinTypes::Integer;
        function_return_types["http-server-port"] = BuiltinTypes::Integer;
        function_return_types["http-server-accept"] = BuiltinTypes::String;
        function_return_types["http-server-respond"] = BuiltinTypes::Boolean;
        function_return_types["http-server-close"] = BuiltinTypes::Boolean;
        function_return_types["http-request"] = BuiltinTypes::Value;
        function_return_types["websocket-connect"] = BuiltinTypes::Integer;
        function_return_types["websocket-send"] = BuiltinTypes::Boolean;
        function_return_types["websocket-send-binary"] = BuiltinTypes::Boolean;
        function_return_types["websocket-receive"] = BuiltinTypes::Value;
        function_return_types["websocket-close"] = BuiltinTypes::Boolean;
        function_return_types["compression-available"] = BuiltinTypes::Boolean;
        function_return_types["deflate"] = BuiltinTypes::Value;
        function_return_types["inflate"] = BuiltinTypes::Value;
        function_return_types["gzip"] = BuiltinTypes::Value;
        function_return_types["gunzip"] = BuiltinTypes::Value;
        function_return_types["yoga-node-create"] = BuiltinTypes::Integer;
        function_return_types["yoga-node-set!"] = BuiltinTypes::Boolean;
        function_return_types["yoga-node-add-child!"] = BuiltinTypes::Boolean;
        function_return_types["yoga-node-calculate!"] = BuiltinTypes::Boolean;
        function_return_types["yoga-node-get-computed"] = BuiltinTypes::Float64;
        function_return_types["yoga-node-free!"] = BuiltinTypes::Boolean;
        function_return_types["ts-parser-new"] = BuiltinTypes::Integer;
        function_return_types["ts-parser-free"] = BuiltinTypes::Boolean;
        function_return_types["ts-parse"] = BuiltinTypes::Integer;
        function_return_types["ts-tree-free"] = BuiltinTypes::Boolean;
        function_return_types["ts-node-type"] = BuiltinTypes::String;
        function_return_types["ts-node-text"] = BuiltinTypes::String;
        function_return_types["ts-node-children"] = BuiltinTypes::Value;
        function_return_types["ts-query-new"] = BuiltinTypes::Integer;
        function_return_types["ts-query-matches"] = BuiltinTypes::Value;
        function_return_types["ts-query-free"] = BuiltinTypes::Boolean;
        function_return_types["ts-available"] = BuiltinTypes::Boolean;
        function_return_types["ts-tree-root"] = BuiltinTypes::Integer;
        function_return_types["http-set-proxy"] = BuiltinTypes::Boolean;
        function_return_types["http-set-tls-client-cert"] = BuiltinTypes::Boolean;
        function_return_types["display-error"] = BuiltinTypes::Boolean;
        function_return_types["string-ends-with?"] = BuiltinTypes::Boolean;
        function_return_types["string-index-of"] = BuiltinTypes::Integer;
        function_return_types["string-pad-left"] = BuiltinTypes::String;
        function_return_types["string-pad-right"] = BuiltinTypes::String;
        function_return_types["kb-save"] = BuiltinTypes::Boolean;
        function_return_types["kb-load"] = BuiltinTypes::Value;
        function_return_types["tensor-token-estimate"] = BuiltinTypes::Integer;
        function_return_types["tensor-rect-fill!"] = BuiltinTypes::Null;
        function_return_types["tensor-disk-fill!"] = BuiltinTypes::Null;
        // Noesis requirements
        function_return_types["fg-marginal"] = BuiltinTypes::Value;
        function_return_types["fg-entropy"] = BuiltinTypes::Value;
        function_return_types["kb-retract!"] = BuiltinTypes::Boolean;
        // Consciousness engine
        function_return_types["make-substitution"] = BuiltinTypes::Value;
        function_return_types["unify"] = BuiltinTypes::Value;
        function_return_types["walk"] = BuiltinTypes::Value;
        function_return_types["make-fact"] = BuiltinTypes::Value;
        function_return_types["make-kb"] = BuiltinTypes::Value;
        function_return_types["kb-assert!"] = BuiltinTypes::Boolean;
        function_return_types["kb-query"] = BuiltinTypes::Value;
        function_return_types["kb-query-prefix"] = BuiltinTypes::Value;
        function_return_types["make-factor-graph"] = BuiltinTypes::Value;
        function_return_types["fg-add-factor!"] = BuiltinTypes::Boolean;
        function_return_types["fg-infer!"] = BuiltinTypes::Boolean;
        function_return_types["free-energy"] = BuiltinTypes::Value;
        function_return_types["expected-free-energy"] = BuiltinTypes::Value;
        function_return_types["make-workspace"] = BuiltinTypes::Value;
        function_return_types["ws-register!"] = BuiltinTypes::Boolean;
        function_return_types["ws-step!"] = BuiltinTypes::Value;
        // Differentiable external memory (core.dnc)
        function_return_types["make-dnc-memory"] = BuiltinTypes::Value;
        function_return_types["dnc-content-address"] = BuiltinTypes::Value;
        function_return_types["dnc-loc-address"] = BuiltinTypes::Value;
        function_return_types["dnc-read"] = BuiltinTypes::Value;
        function_return_types["dnc-write!"] = BuiltinTypes::Value;
        function_return_types["dnc-alloc-weights"] = BuiltinTypes::Value;
        function_return_types["dnc-read-grad"] = BuiltinTypes::Value;
        function_return_types["dnc-memory?"] = BuiltinTypes::Boolean;
        // SDNC weight-program (core.sdnc)
        function_return_types["sdnc-program"] = BuiltinTypes::Value;
        function_return_types["sdnc-run"] = BuiltinTypes::Value;
        function_return_types["sdnc-weight-grad"] = BuiltinTypes::Value;
        function_return_types["sdnc-params"] = BuiltinTypes::Value;
        function_return_types["sdnc-set-params!"] = BuiltinTypes::Value;
        function_return_types["sdnc-improve!"] = BuiltinTypes::Value;
        function_return_types["sdnc?"] = BuiltinTypes::Boolean;
        // Reverse-mode AD tape
        function_return_types["ad-tape-new"] = BuiltinTypes::Value;
        function_return_types["ad-tape-release"] = BuiltinTypes::Value;
        function_return_types["ad-const"] = BuiltinTypes::Integer;
        function_return_types["ad-var"] = BuiltinTypes::Integer;
        function_return_types["ad-add"] = BuiltinTypes::Integer;
        function_return_types["ad-sub"] = BuiltinTypes::Integer;
        function_return_types["ad-mul"] = BuiltinTypes::Integer;
        function_return_types["ad-div"] = BuiltinTypes::Integer;
        function_return_types["ad-pow"] = BuiltinTypes::Integer;
        function_return_types["ad-tape-length"] = BuiltinTypes::Integer;
        function_return_types["ad-sin"] = BuiltinTypes::Integer;
        function_return_types["ad-cos"] = BuiltinTypes::Integer;
        function_return_types["ad-exp"] = BuiltinTypes::Integer;
        function_return_types["ad-log"] = BuiltinTypes::Integer;
        function_return_types["ad-sqrt"] = BuiltinTypes::Integer;
        function_return_types["ad-neg"] = BuiltinTypes::Integer;
        function_return_types["ad-abs"] = BuiltinTypes::Integer;
        function_return_types["ad-relu"] = BuiltinTypes::Integer;
        function_return_types["ad-sigmoid"] = BuiltinTypes::Integer;
        function_return_types["ad-tanh"] = BuiltinTypes::Integer;
        function_return_types["ad-backward"] = BuiltinTypes::Null;
        function_return_types["ad-gradient"] = BuiltinTypes::Value;
        function_return_types["ad-gradient-of"] = BuiltinTypes::Value; // alias
        function_return_types["ad-node-value"] = BuiltinTypes::Value;
        function_return_types["ad-value"] = BuiltinTypes::Value; // user-facing alias
        function_return_types["ad-value-of"] = BuiltinTypes::Value; // alias
        function_return_types["onnx-export-tensor"] = BuiltinTypes::Boolean;
        // Type predicates
        function_return_types["logic-var?"] = BuiltinTypes::Boolean;
        function_return_types["substitution?"] = BuiltinTypes::Boolean;
        function_return_types["fact?"] = BuiltinTypes::Boolean;
        function_return_types["kb?"] = BuiltinTypes::Boolean;
        function_return_types["factor-graph?"] = BuiltinTypes::Boolean;
        function_return_types["workspace?"] = BuiltinTypes::Boolean;
        function_return_types["tensor?"] = BuiltinTypes::Boolean;
        function_return_types["dual?"] = BuiltinTypes::Boolean;
        function_return_types["fg-update-cpt!"] = BuiltinTypes::Boolean;
        function_return_types["kb-count"] = BuiltinTypes::Integer;
        // Image I/O
        function_return_types["image-read"] = BuiltinTypes::Value;
        function_return_types["image-write"] = BuiltinTypes::Boolean;
        function_return_types["image-to-grayscale"] = BuiltinTypes::Value;

        // R7RS Wave 2 functions
        function_return_types["char-foldcase"] = BuiltinTypes::Value;  // char
        function_return_types["string-foldcase"] = BuiltinTypes::String;
        function_return_types["digit-value"] = BuiltinTypes::Value;  // int or #f
        function_return_types["write-shared"] = BuiltinTypes::Null;
        function_return_types["write-simple"] = BuiltinTypes::Null;
        function_return_types["string-for-each"] = BuiltinTypes::Null;
        function_return_types["string-map"] = BuiltinTypes::String;
        function_return_types["vector-for-each"] = BuiltinTypes::Null;
        function_return_types["vector-map"] = BuiltinTypes::Vector;
        function_return_types["call-with-port"] = BuiltinTypes::Value;
        function_return_types["call-with-input-file"] = BuiltinTypes::Value;
        function_return_types["call-with-output-file"] = BuiltinTypes::Value;
        function_return_types["with-input-from-file"] = BuiltinTypes::Value;
        function_return_types["with-output-to-file"] = BuiltinTypes::Value;

        // IO returns Null
        function_return_types["display"] = BuiltinTypes::Null;
        function_return_types["newline"] = BuiltinTypes::Null;

        // Parallel primitives
        function_return_types["parallel-map"] = BuiltinTypes::List;
        function_return_types["parallel-fold"] = BuiltinTypes::Value;
        function_return_types["parallel-filter"] = BuiltinTypes::List;
        function_return_types["parallel-for-each"] = BuiltinTypes::Null;
        function_return_types["thread-pool-info"] = BuiltinTypes::Int64;
        function_return_types["thread-pool-size"] = BuiltinTypes::Int64;  // Alias for thread-pool-info
        function_return_types["thread-pool-stats"] = BuiltinTypes::Null;
        function_return_types["parallel-execute"] = BuiltinTypes::List;

        // Future primitives
        function_return_types["future"] = BuiltinTypes::Value;      // Returns future handle
        function_return_types["force"] = BuiltinTypes::Value;       // Returns result of computation
        function_return_types["future-ready?"] = BuiltinTypes::Boolean; // Returns boolean
        function_return_types["make-promise"] = BuiltinTypes::Value;
        function_return_types["promise?"] = BuiltinTypes::Boolean;
        function_return_types["%make-lazy-promise"] = BuiltinTypes::Value;
        function_return_types["%make-lazy-promise-force"] = BuiltinTypes::Value;
        function_return_types["rational?"] = BuiltinTypes::Boolean;
        // numerator/denominator may return a bignum for bignum-magnitude
        // rationals, so they are tagged Values, not raw int64.
        function_return_types["numerator"] = BuiltinTypes::Value;
        function_return_types["denominator"] = BuiltinTypes::Value;
        function_return_types["make-rational"] = BuiltinTypes::Value;
        function_return_types["rationalize"] = BuiltinTypes::Value;
        function_return_types["read-string"] = BuiltinTypes::String;
        function_return_types["make-bytevector"] = BuiltinTypes::Value;
        function_return_types["bytevector"] = BuiltinTypes::Value;
        function_return_types["bytevector-length"] = BuiltinTypes::Int64;
        function_return_types["bytevector-u8-ref"] = BuiltinTypes::Int64;
        function_return_types["bytevector-copy"] = BuiltinTypes::Value;
        function_return_types["bytevector-append"] = BuiltinTypes::Value;
        function_return_types["dequant-q4_0"] = BuiltinTypes::Tensor;
        function_return_types["dequant-q8_0"] = BuiltinTypes::Tensor;
        function_return_types["bytevector?"] = BuiltinTypes::Boolean;
        function_return_types["utf8->string"] = BuiltinTypes::String;
        function_return_types["string->utf8"] = BuiltinTypes::Value;
        function_return_types["tensor-save"] = BuiltinTypes::Boolean;
        function_return_types["tensor-load"] = BuiltinTypes::Tensor;
        function_return_types["model-save"] = BuiltinTypes::Boolean;
        function_return_types["model-load"] = BuiltinTypes::List;

        // Complex number operations
        function_return_types["make-rectangular"] = BuiltinTypes::Complex128;  // Create complex from real,imag
        function_return_types["make-polar"] = BuiltinTypes::Complex128;        // Create complex from magnitude,angle
        function_return_types["real-part"] = BuiltinTypes::Float64;            // Extract real component
        function_return_types["imag-part"] = BuiltinTypes::Float64;            // Extract imaginary component
        function_return_types["magnitude"] = BuiltinTypes::Float64;            // |z| = sqrt(r² + i²)
        function_return_types["angle"] = BuiltinTypes::Float64;                // arg(z) = atan2(imag, real)
        function_return_types["complex?"] = BuiltinTypes::Boolean;             // Type predicate
        function_return_types["conjugate"] = BuiltinTypes::Complex128;         // Complex conjugate

        // HoTT sum type operations (discriminated unions)
        function_return_types["inject-left"] = BuiltinTypes::Pair;     // (0 . value) tagged pair
        function_return_types["inject-right"] = BuiltinTypes::Pair;    // (1 . value) tagged pair
        function_return_types["sum-tag"] = BuiltinTypes::Int64;        // Returns 0 (left) or 1 (right)
        function_return_types["sum-value"] = BuiltinTypes::Value;      // Extracts inner value
        function_return_types["left?"] = BuiltinTypes::Boolean;        // Is left variant?
        function_return_types["right?"] = BuiltinTypes::Boolean;       // Is right variant?

        // FFT/IFFT operations (Signal Processing)
        function_return_types["fft"] = BuiltinTypes::Value;    // Returns vector of complex numbers
        function_return_types["ifft"] = BuiltinTypes::Value;   // Returns vector of complex numbers

        // Signal Processing Filters (stdlib: signal.filters)
        function_return_types["fft"] = BuiltinTypes::Vector;
        function_return_types["ifft"] = BuiltinTypes::Vector;
        function_return_types["hamming-window"] = BuiltinTypes::Vector;
        function_return_types["hann-window"] = BuiltinTypes::Vector;
        function_return_types["blackman-window"] = BuiltinTypes::Vector;
        function_return_types["kaiser-window"] = BuiltinTypes::Vector;
        function_return_types["apply-window"] = BuiltinTypes::Vector;
        function_return_types["convolve"] = BuiltinTypes::Vector;
        function_return_types["fast-convolve"] = BuiltinTypes::Vector;
        function_return_types["fir-filter"] = BuiltinTypes::Vector;
        function_return_types["iir-filter"] = BuiltinTypes::Vector;
        function_return_types["butterworth-lowpass"] = BuiltinTypes::Pair;   // (b . a) coefficient pair
        function_return_types["butterworth-highpass"] = BuiltinTypes::Pair;
        function_return_types["butterworth-bandpass"] = BuiltinTypes::Pair;
        function_return_types["frequency-response"] = BuiltinTypes::Pair;    // (magnitudes . phases)

        // Optimization Algorithms (stdlib: ml.optimization)
        function_return_types["gradient-descent"] = BuiltinTypes::Vector;
        function_return_types["adam"] = BuiltinTypes::Vector;
        function_return_types["l-bfgs"] = BuiltinTypes::Vector;
        function_return_types["conjugate-gradient"] = BuiltinTypes::Vector;
        function_return_types["line-search"] = BuiltinTypes::Float64;       // Returns step size alpha
        function_return_types["tensor-dot"] = BuiltinTypes::Float64;        // Returns scalar
        function_return_types["tensor-norm"] = BuiltinTypes::Float64;       // Returns scalar
        function_return_types["tensor-svd"] = BuiltinTypes::List;          // Returns (U S V) list
        // Tensor unary/binary ops — all return a tensor (Value = tagged heap ptr)
        function_return_types["tensor-neg"]       = BuiltinTypes::Value;
        function_return_types["tensor-abs"]       = BuiltinTypes::Value;
        function_return_types["tensor-sqrt"]      = BuiltinTypes::Value;
        function_return_types["tensor-exp"]       = BuiltinTypes::Value;
        function_return_types["tensor-log"]       = BuiltinTypes::Value;
        function_return_types["tensor-sin"]       = BuiltinTypes::Value;
        function_return_types["tensor-cos"]       = BuiltinTypes::Value;
        function_return_types["tensor-pow"]       = BuiltinTypes::Value;
        function_return_types["tensor-maximum"]   = BuiltinTypes::Value;
        function_return_types["tensor-minimum"]   = BuiltinTypes::Value;
        function_return_types["tensor-scale"]     = BuiltinTypes::Value;
        function_return_types["tensor-transpose"] = BuiltinTypes::Value;
        function_return_types["batch-matmul"]     = BuiltinTypes::Value;
    }

void EshkolLLVMCodeGen::createLibraryInitFunction(const eshkol_ast_t* asts, size_t num_asts) {
        std::vector<size_t> init_ast_indices;
        for (size_t i = 0; i < num_asts; i++) {
            if (isLibraryInitAST(asts[i])) {
                init_ast_indices.push_back(i);
            }
        }

        Function* lambda_init_func = module->getFunction("__lambda_init__");
        if (freestanding_codegen_ &&
            init_ast_indices.empty() &&
            !lambda_init_func &&
            eshkol::llvm_codegen_detail::pendingLambdaSExprs().empty()) {
            finalizeLibrarySymbols(asts, num_asts);
            eshkol_info("Skipped library init function for freestanding object mode");
            return;
        }

        // Create library init function: void __eshkol_lib_init__(void* arena)
        // Takes an arena pointer as parameter so caller can manage memory
        std::vector<Type*> init_args = { PointerType::getUnqual(*context) }; // arena pointer
        FunctionType* init_type = FunctionType::get(void_type, init_args, false);
        Function* init_func = Function::Create(init_type, Function::ExternalLinkage,
                                               "__eshkol_lib_init__", module.get());

        BasicBlock* entry = BasicBlock::Create(*context, "entry", init_func);
        builder->SetInsertPoint(entry);
        current_function = init_func;

        // Get arena parameter and store in global
        Value* arena_param = init_func->arg_begin();
        arena_param->setName("arena");
        builder->CreateStore(arena_param, global_arena);

        // Process global variable definitions and top-level set! statements in
        // noinline chunks. Large aggregate libraries otherwise produce one huge
        // __eshkol_lib_init__ text atom, which trips Apple ld branch-island
        // placement limits.
        std::vector<Function*> init_chunks;
        for (size_t begin = 0, chunk_index = 0;
             begin < init_ast_indices.size();
             begin += LIB_INIT_AST_CHUNK_SIZE, chunk_index++) {
            size_t end = std::min(begin + LIB_INIT_AST_CHUNK_SIZE, init_ast_indices.size());
            init_chunks.push_back(createLibraryInitChunkFunction(
                init_type, chunk_index, asts, init_ast_indices, begin, end));
        }
        for (Function* chunk_func : init_chunks) {
            builder->CreateCall(chunk_func, {arena_param});
        }

        // Call __lambda_init__ to initialize lambda captures
        if (lambda_init_func) {
            builder->CreateCall(lambda_init_func);
            eshkol_debug("Library init: called __lambda_init__ for lambda captures");
        }

        // Initialize lambda registry (caller may not have done it yet)
        builder->CreateCall(eshkol_lambda_registry_init_func);
        eshkol_debug("Library init: initialized lambda registry");

        // Generate S-expressions for lambdas and register them
        if (!eshkol::llvm_codegen_detail::pendingLambdaSExprs().empty()) {
            std::vector<Function*> sexpr_chunks;
            for (size_t begin = 0, chunk_index = 0;
                 begin < eshkol::llvm_codegen_detail::pendingLambdaSExprs().size();
                 begin += LIB_INIT_LAMBDA_SEXPR_CHUNK_SIZE, chunk_index++) {
                size_t end = std::min(begin + LIB_INIT_LAMBDA_SEXPR_CHUNK_SIZE,
                                      eshkol::llvm_codegen_detail::pendingLambdaSExprs().size());
                sexpr_chunks.push_back(createLibraryLambdaSExprChunkFunction(
                    init_type, chunk_index, begin, end));
            }
            for (Function* chunk_func : sexpr_chunks) {
                builder->CreateCall(chunk_func, {arena_param});
            }

            eshkol::llvm_codegen_detail::pendingLambdaSExprs().clear();
        }

        // Return void
        builder->CreateRetVoid();

        finalizeLibrarySymbols(asts, num_asts);

        eshkol_info("Created library init function: __eshkol_lib_init__");
    }

#endif

