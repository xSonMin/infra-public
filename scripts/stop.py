#!/usr/bin/env python3
"""Stop only the inference server recorded by this project's experiment runner."""
import json
import os
from pathlib import Path
import signal
import time


def validate_target(root, command, owner_uid):
    if owner_uid != os.getuid():
        raise PermissionError('Recorded process belongs to another user')
    expected_python = str(Path(root) / '.venv/bin/python').encode() + b'\0'
    if not command.startswith(expected_python) or b'\0sglang.launch_server\0' not in command:
        raise ValueError('Recorded PID is not this project inference server; refusing to stop it')


def main():
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root / 'results/active-server.json').read_text())
    pid = int(record['pid'])
    if pid <= 1:
        raise ValueError('Invalid recorded PID')
    process = Path('/proc') / str(pid)
    if not process.exists():
        print('Recorded inference server is already stopped')
        return
    validate_target(root, (process / 'cmdline').read_bytes(), process.stat().st_uid)
    os.kill(pid, signal.SIGTERM)
    for _ in range(60):
        if not process.exists() or (process / 'stat').read_text().split(') ', 1)[1].startswith('Z '):
            print(f'Stopped project inference server {pid}')
            return
        time.sleep(0.5)
    raise TimeoutError('Server has not stopped after 30 seconds; inspect its log')


if __name__ == '__main__':
    main()
