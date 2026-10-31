#!/usr/bin/env python3
"""Serial TP4 MTP study runner. Scope all transitions to the recorded project server."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def workload_cases():
    return [(name, inputs, outputs, concurrency)
            for name, inputs, outputs in [('short', 512, 128), ('long-context', 2048, 256),
                                           ('long-decode', 512, 512)]
            for concurrency in (1, 8, 32)]


def calibrated_count(concurrency, observations, target_seconds=35):
    if concurrency < 1 or not observations or target_seconds <= 0:
        raise ValueError('Calibration requires positive concurrency and observations')
    if any(not math.isfinite(seconds) or seconds <= 0 or count <= 0 for count, seconds in observations):
        raise ValueError('Invalid pilot duration or count')
    count = max(32, 5 * concurrency, math.ceil(target_seconds * max(n / seconds for n, seconds in observations)))
    return math.ceil(count / concurrency) * concurrency


def study_directory(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    parent = root / 'results/mtp-study-2026-10-30'
    if path.parent != parent or not path.name.startswith('run-'):
        raise ValueError('Study must use a run directory within the project study root')
    return path


def is_descendant(pid, ancestor, proc_root=Path('/proc')):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        try:
            status = (Path(proc_root) / str(pid) / 'status').read_text()
            pid = int(next(line.split(':', 1)[1] for line in status.splitlines() if line.startswith('PPid:')))
        except (OSError, StopIteration, ValueError):
            return False
    return False


def validate_pair(first, second):
    for key in ('input_manifest_sha256', 'request_count', 'seed'):
        if first.get(key) is None or first.get(key) != second.get(key):
            raise ValueError(f'Pair mismatch: {key}')


def server_argv(root, mode, steps=3, adaptive_config=None):
    if mode not in ('none', 'mtp') or type(steps) is not int or steps not in (1, 2, 3):
        raise ValueError('Unsupported server mode or speculative depth')
    if mode == 'none' and (steps != 3 or adaptive_config is not None):
        raise ValueError('Ordinary decode cannot use speculative parameters')
    root = Path(root).resolve()
    command = [str(root / '.venv/bin/python'), str(root / 'scripts/launch.py'),
               '--gpus', '0,1,2,3', '--speculative', mode,
               '--gpu-addressing', 'uuid', '--port', '30000']
    if steps != 3:
        command.extend(['--speculative-steps', str(steps)])
    if adaptive_config is not None:
        config = Path(adaptive_config).resolve()
        if root not in config.parents or not config.is_file():
            raise ValueError('Adaptive configuration must be an existing project file')
        command.extend(['--adaptive-config', str(config)])
    return command


def existing_sequence(folder):
    numbers = [int(path.name.split('-',1)[0]) for path in Path(folder).glob('*.log')
               if path.name.split('-',1)[0].isdigit()]
    return max(numbers, default=0)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class Study:
    def __init__(self, root, run):
        self.root = Path(root).resolve()
        self.run = study_directory(self.root, run)
        self.run.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(self.root / 'scripts'))
        import launch
        import stop
        self.launch, self.stop = launch, stop
        self.active_path = self.root / 'results/active-server.json'
        self.pyth = str(self.root / '.venv/bin/python')
        self.sequence = existing_sequence(self.run / 'server-logs')

    def event(self, event, **fields):
        row = dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), event=event, **fields)
        with (self.run / 'workflow-events.jsonl').open('a') as output:
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def info(self):
        with urllib.request.urlopen('http://127.0.0.1:30000/get_server_info', timeout=8) as response:
            return json.load(response)

    def gpu_processes(self):
        content = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                           '--format=csv,noheader'], text=True)
        allowed = set(self.launch.ALLOWED_GPUS.values())
        return [(int(pid.strip()), uuid.strip()) for line in content.splitlines() if ',' in line
                for pid, uuid in [line.split(',', 1)] if uuid.strip() in allowed]

    def guard(self, record):
        pid = int(record['pid'])
        proc = Path('/proc') / str(pid)
        if proc.exists():
            self.stop.validate_target(self.root, (proc / 'cmdline').read_bytes(), proc.stat().st_uid)
            environment = (proc / 'environ').read_bytes().split(b'\0')
            expected = ('CUDA_VISIBLE_DEVICES=' + ','.join(self.launch.ALLOWED_GPUS[i] for i in range(4))).encode()
            if expected not in environment:
                raise PermissionError('Refusing to transition server without exact authorized UUIDs')
        for gpu_pid, uuid in self.gpu_processes():
            gpu_proc = Path('/proc') / str(gpu_pid)
            if gpu_proc.stat().st_uid != os.getuid() or not is_descendant(gpu_pid, pid):
                raise PermissionError(f'Unexpected process {gpu_pid} uses authorized GPU {uuid}')

    def switch(self, mode, tag, steps=3, adaptive_config=None):
        old = json.loads(self.active_path.read_text())
        self.guard(old)
        processes = self.gpu_processes()
        if Path('/proc', str(old['pid'])).exists():
            self.event('stop_server', pid=old['pid'], tag=old.get('tag'))
            subprocess.run([self.pyth, str(self.root / 'scripts/stop.py')], check=True, timeout=45)
        deadline = time.monotonic() + 60
        while self.gpu_processes():
            if time.monotonic() > deadline:
                raise TimeoutError('GPU workers did not exit; no unrelated process will be killed')
            time.sleep(0.5)
        self.sequence += 1
        log_dir = self.run / 'server-logs'
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f'{self.sequence:02d}-{tag}.log'
        command = server_argv(self.root, mode, steps, adaptive_config)
        with log_path.open('x') as log:
            server = subprocess.Popen(command, cwd=self.root, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        record = {'pid': server.pid, 'tag': tag, 'command': command, 'log': str(log_path), 'started_at': time.time()}
        atomic_json(self.active_path, record)
        self.event('start_server', **record)
        deadline = time.monotonic() + 420
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f'Server exited {server.returncode}: {log_path}')
            try:
                with urllib.request.urlopen('http://127.0.0.1:30000/health', timeout=2) as response:
                    if response.status == 200:
                        info = self.info()
                        if info.get('tp_size') != 4 or info.get('speculative_algorithm') != ('EAGLE' if mode == 'mtp' else None):
                            raise RuntimeError('Server configuration mismatch')
                        self.guard(record)
                        self.event('server_ready', mode=mode, pid=server.pid)
                        return
            except (OSError, TimeoutError):
                pass
            time.sleep(1)
        raise TimeoutError(f'Server did not become healthy: {log_path}')

    def bench(self, phase, tag, name, inputs, outputs, concurrency, count, seed=42, warmups=None, dataset=None):
        base = self.run / phase
        result = base / 'results' / f'{tag}-i{inputs}-o{outputs}-c{concurrency}.jsonl'
        command = [self.pyth, str(self.root / 'scripts/bench_suite.py'), '--tag', tag,
                   '--input-len', str(inputs), '--output-len', str(outputs), '--concurrency', str(concurrency),
                   '--num-prompts', str(count), '--warmup-requests', str(max(2, concurrency) if warmups is None else warmups),
                   '--seed', str(seed), '--output-dir', str(base)]
        if dataset is not None:
            command.extend(['--dataset-path', str(dataset)])
        self.event('bench_begin', phase=phase, tag=tag, count=count, workload=name, concurrency=concurrency)
        subprocess.run(command, cwd=self.root, check=True, timeout=1800)
        raw = json.loads(result.read_text())
        meta = json.loads(result.with_suffix('.run.json').read_text())
        if meta['request_count'] != count or raw['completed'] != count:
            raise ValueError('Actual request count differs from planned case')
        self.event('bench_end', phase=phase, tag=tag, count=count, duration=raw['duration'],
                   throughput=raw['output_throughput'], input_manifest_sha256=meta['input_manifest_sha256'])
        return raw, meta

    def baseline(self):
        if (self.run / 'baseline-complete.json').exists():
            raise FileExistsError('Baseline already complete; refusing duplicate execution')
        pilot = {}
        # Current project service is the validated stable MTP service. Do not restart it for the first pilot.
        info = self.info()
        if info.get('tp_size') != 4 or info.get('speculative_algorithm') != 'EAGLE':
            raise RuntimeError('Expected stable TP4 MTP service before pilot')
        self.guard(json.loads(self.active_path.read_text()))
        for mode in ('mtp', 'none'):
            if mode == 'none':
                self.switch('none', 'study-pilot-none')
            for name, inputs, outputs, concurrency in workload_cases():
                key = f'{name}-c{concurrency}'
                count = max(4, 2 * concurrency)
                raw, meta = self.bench('pilot', f'{key}-{mode}-pilot', name, inputs, outputs, concurrency, count)
                pilot.setdefault(key, {})[mode] = {'count': count, 'duration': raw['duration'], 'metadata': meta}
        counts = {key: calibrated_count(int(key.split('-c')[-1]),
                  [(item['count'], item['duration']) for item in observations.values()])
                  for key, observations in pilot.items()}
        atomic_json(self.run / 'calibration.json', {'pilots': pilot, 'frozen_request_counts': counts,
                                                  'target_seconds': 35})
        measured = {}
        for round_id in (1, 2, 3):
            for mode in (('none', 'mtp') if round_id % 2 else ('mtp', 'none')):
                self.switch(mode, f'study-r{round_id}-{mode}')
                for name, inputs, outputs, concurrency in workload_cases():
                    key = f'{name}-c{concurrency}'
                    raw, meta = self.bench('baseline', f'{key}-{mode}-r{round_id}', name,
                                           inputs, outputs, concurrency, counts[key])
                    measured.setdefault((key, round_id), {})[mode] = (raw, meta)
                    if len(measured[key, round_id]) == 2:
                        pair = measured[key, round_id]
                        validate_pair(pair['none'][1], pair['mtp'][1])
                        self.event('pair_validated', workload=key, round_id=round_id,
                                   speedup=pair['mtp'][0]['output_throughput']/pair['none'][0]['output_throughput'])
        short = [(key, rnd, mode) for (key, rnd), modes in measured.items() for mode, (raw, meta) in modes.items() if raw['duration'] < 30]
        if short:
            raise ValueError(f'Paired cases require a longer rerun: {short}')
        atomic_json(self.run / 'baseline-complete.json', {'status': 'completed', 'pairs': len(measured),
                                                       'cases': sum(len(x) for x in measured.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study = Study(root, args.run_dir)
    marker = study.run / 'workflow-started.json'
    with marker.open('x') as output:
        json.dump({'pid': os.getpid(), 'started_at': time.time()}, output)
    telemetry_log = (study.run / 'gpu-telemetry.csv').open('x')
    telemetry = subprocess.Popen(['nvidia-smi', '-i', '0,1,2,3',
                                 '--query-gpu=timestamp,index,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm',
                                 '--format=csv,noheader', '--loop-ms=1000'], stdout=telemetry_log, stderr=subprocess.STDOUT)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    outcome = {'baseline_status': 'running', 'service_restored': False}
    try:
        study.event('baseline_start', driver_pid=os.getpid())
        study.baseline()
        outcome['baseline_status'] = 'completed'
    except BaseException as error:
        outcome['baseline_status'] = 'failed'
        outcome['error'] = f'{type(error).__name__}: {error}'
        study.event('baseline_failure', error=outcome['error'])
        raise
    finally:
        try:
            study.switch('mtp', 'study-restored-stable-tp4-mtp')
            outcome['service_restored'] = True
        except BaseException as error:
            outcome['restoration_error'] = f'{type(error).__name__}: {error}'
            study.event('restoration_failure', error=outcome['restoration_error'])
        telemetry.terminate()
        telemetry.wait(timeout=10)
        telemetry_log.close()
        atomic_json(study.run / 'workflow-outcome.json', outcome)
        study.event('workflow_finished', **outcome)


if __name__ == '__main__':
    main()
