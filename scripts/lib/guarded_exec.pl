#!/usr/bin/env perl
# guarded_exec.pl <seconds> <cmd...>
#
# The one wall-clock timeout every test harness uses. Portable to macOS and
# Linux (neither stock macOS nor every CI image provides coreutils timeout(1)),
# and the implementation behind eshkol_outcome_guarded in harness_outcome.sh.
# It is a standalone program, not only a shell function, so a harness can put
# it under /usr/bin/time (-l or -v) and still measure the command it guards.
#
# Exit status is exactly one of:
#   <the command's own exit status>  the command ran to completion
#   124  the command exceeded <seconds> and was stopped
#   128+N  the command died from signal N that it received on its own
#   125  this wrapper could not fork or wait
#   127  the command could not be executed
# When the wrapper itself is interrupted (SIGINT, SIGTERM, SIGHUP) it stops
# the command the same way and exits 128+N for that signal.
#
# The command runs as the leader of its own session and process group, and
# every stop is delivered to that whole group, first SIGTERM and then SIGKILL.
# A command that starts processes of its own (eshkol-run builds a program in a
# child process and runs the built binary in another) is therefore stopped
# together with everything it started. Those processes inherit the harness's
# output pipe, so a descendant that outlived the command would hold the pipe
# open and keep the harness waiting on it after the timeout; stopping the group
# closes it. When the command exits on its own, any processes it left behind in
# its group are stopped as well, for the same reason.
use strict;
use warnings;
use POSIX qw(setsid :sys_wait_h);

my $secs = shift @ARGV;
die "usage: guarded_exec.pl <seconds> <cmd...>\n"
    unless defined $secs && $secs =~ /^\d+$/ && @ARGV;

my $pid = fork();
exit 125 unless defined $pid;
if ($pid == 0) {
    setsid();   # never fails here: a freshly forked child leads no group
    exec { $ARGV[0] } @ARGV or exit 127;
}

# Stop the command's whole process group: SIGTERM, a grace period for it to
# exit, then SIGKILL for anything still present. kill() on a negative id
# addresses the group; ESRCH (nothing left) is the normal quiet case.
sub stop_group {
    kill('TERM', -$pid);
    for (1 .. 10) {
        return unless kill(0, -$pid);
        select(undef, undef, undef, 0.05);
    }
    kill('KILL', -$pid);
}

my $timed_out = 0;
local $SIG{ALRM} = sub { $timed_out = 1; stop_group(); };
my %signal_number = (INT => 2, HUP => 1, TERM => 15);
for my $name (keys %signal_number) {
    $SIG{$name} = sub {
        alarm(0);
        stop_group();
        waitpid($pid, 0);
        exit(128 + $signal_number{$name});
    };
}

alarm($secs);
my $reaped;
do { $reaped = waitpid($pid, 0); } while ($reaped == -1 && $!{EINTR});
alarm(0);
my $status = $?;
# Processes the command left behind in its group would keep the output pipe
# open; the group id stays reserved while any member exists, so this reaches
# only those processes (or nothing).
kill('KILL', -$pid);
exit 125 if $reaped != $pid;
exit 124 if $timed_out;
# Died from a signal we did not send: a real crash, kept distinguishable.
exit(128 + ($status & 127)) if ($status & 127) != 0;
exit($status >> 8);
