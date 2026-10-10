#!/usr/bin/env bash
# Portable peak-RSS support for scripts/run_stress.sh.

eshkol_stress_time_detect() { # time_binary probe_file -> bsd|gnu on stdout
    local time_bin="$1" probe="$2"
    if "$time_bin" -l true >/dev/null 2>"$probe" &&
       grep -q 'maximum resident set size' "$probe"; then
        printf 'bsd\n'
    elif "$time_bin" -v true >/dev/null 2>"$probe" &&
         grep -q 'Maximum resident set size' "$probe"; then
        printf 'gnu\n'
    else
        echo "stress harness: $time_bin does not report peak RSS with -l or -v" >&2
        return 2
    fi
}

eshkol_stress_time_run() { # time_binary mode guard timeout out time_log command...
    local time_bin="$1" mode="$2" guard="$3" timeout_s="$4"
    local out="$5" time_log="$6"
    shift 6
    case "$mode" in
        bsd) "$time_bin" -l perl "$guard" "$timeout_s" "$@" >"$out" 2>"$time_log" < /dev/null ;;
        gnu) "$time_bin" -v perl "$guard" "$timeout_s" "$@" >"$out" 2>"$time_log" < /dev/null ;;
        *) echo "stress harness: unknown time mode '$mode'" >&2; return 2 ;;
    esac
}

eshkol_stress_time_rss_mb() { # mode time_log -> integer MiB on stdout; 2 if missing/invalid
    local mode="$1" time_log="$2" value
    case "$mode" in
        bsd)
            value=$(sed -nE 's/^[[:space:]]*([0-9]+)[[:space:]]+maximum resident set size[[:space:]]*$/\1/p' "$time_log")
            [ "$(printf '%s\n' "$value" | wc -l | tr -d ' ')" = 1 ] || return 2
            awk -v n="$value" 'BEGIN { if (n !~ /^[0-9]+$/) exit 2; printf "%d\n", n/1048576 }'
            ;;
        gnu)
            value=$(sed -nE 's/^[[:space:]]*Maximum resident set size \(kbytes\):[[:space:]]*([0-9]+)[[:space:]]*$/\1/p' "$time_log")
            [ "$(printf '%s\n' "$value" | wc -l | tr -d ' ')" = 1 ] || return 2
            awk -v n="$value" 'BEGIN { if (n !~ /^[0-9]+$/) exit 2; printf "%d\n", n/1024 }'
            ;;
        *) return 2 ;;
    esac
}

eshkol_stress_time_append_program_stderr() { # mode time_log out_file
    local mode="$1" time_log="$2" out="$3"
    grep -vE 'real .*user .*sys|maximum resident set size|average .* size|page reclaims|page faults|swaps|block .* operations|messages (sent|received)|signals received|context switches|instructions retired|cycles elapsed|peak memory footprint|Command being timed:|User time \(seconds\):|System time \(seconds\):|Percent of CPU this job got:|Elapsed \(wall clock\) time|Average shared text size|Average unshared data size|Average stack size|Average total size|Maximum resident set size|Average resident set size|Major \(requiring I/O\) page faults|Minor \(reclaiming a frame\) page faults|Voluntary context switches|Involuntary context switches|Swaps|File system inputs|File system outputs|Socket messages sent|Socket messages received|Signals delivered|Page size \(bytes\)|Exit status:' \
        "$time_log" >> "$out" 2>/dev/null || true
}
