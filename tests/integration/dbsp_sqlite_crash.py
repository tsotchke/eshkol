"""Kill after staged cursor/checkpoint writes, then after their successful COMMIT."""
import argparse
import os
from pathlib import Path
import selectors
import signal
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner', required=True)
    parser.add_argument('--link-dir', required=True)
    parser.add_argument('--mode', choices=('jit', 'aot'), required=True)
    args = parser.parse_args()
    source = Path(__file__).with_suffix('.esk')
    with tempfile.TemporaryDirectory(prefix='eshkol-dbsp-crash-') as directory:
        binary = Path(directory) / 'fixture'
        if args.mode == 'aot':
            subprocess.run([args.runner, '-O', '1', '-o', str(binary), str(source),
                            '-L' + args.link_dir], check=True, timeout=900)
        command = ([args.runner, '-r', str(source), '-L' + args.link_dir]
                   if args.mode == 'jit' else [str(binary)])
        env = dict(os.environ, ESHKOL_FEED_FILE=str(Path(directory) / 'feed.db'), ESHKOL_JIT_CACHE='0')
        for phase, marker in [('seed', None), ('stage', 'STAGED'), ('recover', None),
                              ('commit', 'COMMITTED'), ('verify', None)]:
            process = subprocess.Popen(command, env=dict(env, ESHKOL_FEED_PHASE=phase),
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)
            output = bytearray()
            try:
                if marker:
                    selector = selectors.DefaultSelector()
                    selector.register(process.stdout, selectors.EVENT_READ)
                    deadline = time.monotonic() + 180
                    while marker.encode() + b'\n' not in output and time.monotonic() < deadline:
                        for key, _ in selector.select(timeout=1):
                            chunk = os.read(key.fd, 65536)
                            output.extend(chunk)
                            if not chunk:
                                raise AssertionError(f'{phase} exited before marker: {output.decode(errors="replace")}')
                    selector.close()
                    assert marker.encode() + b'\n' in output, f'{phase} marker timeout'
                    process.kill()
                    process.wait(timeout=10)
                    assert process.returncode == -signal.SIGKILL
                    print(f'{args.mode}: killed at {marker}', flush=True)
                else:
                    data, _ = process.communicate(timeout=180)
                    output.extend(data)
                    assert process.returncode == 0 and b'RESULT: PASS' in output, output.decode(errors='replace')
                    print(f'{args.mode}: {phase} passed', flush=True)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
                for stream in (process.stdin, process.stdout):
                    stream.close()
    print('RESULT: PASS')


if __name__ == '__main__':
    main()
