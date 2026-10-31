#!/usr/bin/env python3
"""Capture short server-side traces and report conservative, per-rank observations."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mtp_analysis import interval_union_us


def trace_summary(events):
    kernels = [row for row in events if row.get('ph') == 'X' and 'kernel' in row.get('cat', '').lower()]
    if not kernels:
        raise ValueError('No GPU kernel events; CPU-only profile is not a valid GPU trace')
    comm_pattern = re.compile(r'nccl|all_reduce|allreduce|all_gather|allgather|reduce_scatter', re.I)
    communication = [row for row in kernels if comm_pattern.search(row.get('name', ''))]
    intervals = lambda rows: [(row['ts'], row['ts'] + row['dur']) for row in rows]
    stage_names = ('draft', 'verify', 'draft_extend', 'verify_forward', 'prefill_forward', 'decode_forward')
    def label(row):
        name = row.get('name', '')
        if name in stage_names:
            return name
        if name.startswith('step[VERIFY '):
            return 'verify_forward'
        if name.startswith('step[EXTEND '):
            return 'prefill_forward'
        if name.startswith('step[DECODE '):
            return 'decode_forward'
        return None
    stages = {name: interval_union_us(intervals([
        row for row in events if row.get('ph') == 'X' and row.get('cat') == 'user_annotation' and label(row) == name])) for name in stage_names}
    forward_steps = []
    for row in events:
        match = re.match(r'step\[(\w+) bs=(\d+)', row.get('name', ''))
        if match and row.get('ph') == 'X' and row.get('cat') == 'user_annotation':
            forward_steps.append({'stage': match.group(1), 'batch_size': int(match.group(2)),
                                  'start_us': row['ts'], 'cpu_wall_us': row['dur']})
    # Associate asynchronous GPU work with the CPU range which launched it.
    # Comparing kernel timestamps directly with CPU annotations would misattribute queued work.
    annotations = [row for row in events if row.get('ph') == 'X'
                   and row.get('cat') == 'user_annotation' and label(row) is not None]
    launches = {}
    for row in events:
        category = row.get('cat', '').lower()
        correlation = row.get('args', {}).get('correlation')
        if row.get('ph') != 'X' or correlation is None or 'kernel' in category or 'cuda' not in category:
            continue
        enclosing = [span for span in annotations
                     if span.get('pid') == row.get('pid') and span.get('tid') == row.get('tid')
                     and span['ts'] <= row['ts'] and span['ts'] + span['dur'] >= row['ts'] + row['dur']]
        if enclosing:
            launches[correlation] = label(min(enclosing, key=lambda span: span['dur']))
    assigned = {name: [] for name in stages}
    unknown = []
    for kernel in kernels:
        stage = launches.get(kernel.get('args', {}).get('correlation'))
        if stage in assigned:
            assigned[stage].append(kernel)
        else:
            unknown.append(kernel)
    communication_ids = {id(row) for row in communication}
    compute = [row for row in kernels if id(row) not in communication_ids]
    compute_us = interval_union_us(intervals(compute))
    comm_us = interval_union_us(intervals(communication))
    overlap_us = compute_us + comm_us - interval_union_us(intervals(kernels))
    names = {}
    for row in kernels:
        names.setdefault(row['name'], {'calls': 0, 'sum_duration_us': 0})
        names[row['name']]['calls'] += 1
        names[row['name']]['sum_duration_us'] += row['dur']
    return {'kernel_count': len(kernels), 'gpu_busy_union_us': interval_union_us(intervals(kernels)),
            'communication_labeled_union_us': interval_union_us(intervals(communication)),
            'communication_labeled_kernel_count': len(communication),
            'stage_cpu_wall_us': stages,
            'gpu_stage_union_us': {name: interval_union_us(intervals(rows)) for name, rows in assigned.items()},
            'communication_stage_union_us': {name: interval_union_us(intervals([row for row in rows if id(row) in communication_ids])) for name, rows in assigned.items()},
            'stage_annotation_count': {name: sum(row.get('ph') == 'X' and row.get('cat') == 'user_annotation' and label(row) == name for row in events) for name in stage_names},
            'gpu_unknown_union_us': interval_union_us(intervals(unknown)),
            'gpu_stage_mapped_kernel_count': len(kernels) - len(unknown),
            'compute_communication_overlap_us': max(0, overlap_us),
            'mapping_method': 'cpu-launch-correlation',
            'forward_steps': forward_steps,
            'attribution_limited': True,
            'limits': ['CPU annotation wall times are not GPU critical-path times.',
                       'Stage attribution uses runtime launch correlations, with unmapped GPU work preserved.',
                       'verify_forward is model execution only, not all acceptance and cache bookkeeping.',
                       'Communication labels may omit fused/custom communication.',
                       'Communication activity is not added to enclosing stage durations.'],
            'top_kernel_total_activity': sorted([dict(name=name, **value) for name, value in names.items()],
                                              key=lambda row: row['sum_duration_us'], reverse=True)[:30]}


def post(path, payload=None):
    request = urllib.request.Request('http://127.0.0.1:30000/' + path,
                                     data=json.dumps(payload or {}).encode(),
                                     headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=180) as response:
        text = response.read().decode()
        if response.status != 200:
            raise RuntimeError(f'{path} failed: {response.status} {text}')
        if text.strip().startswith('{'):
            body = json.loads(text)
            if body.get('success') is False:
                raise RuntimeError(f'{path}: {body}')
        return {'status': response.status, 'body': text}


def capture(study, name, inputs, outputs, concurrency, mode):
    tag = f'{name}-c{concurrency}-{mode}'
    destination = study.run / 'profiles' / tag
    validation = destination / 'trace-validation.json'
    if validation.exists():
        previous = json.loads(validation.read_text())
        if previous.get('status') != 'valid':
            raise ValueError('Existing profile was not validated')
        results = {int(rank): row for rank, row in previous['ranks'].items()}
        if set(results) != {0, 1, 2, 3}:
            raise ValueError('Existing profile has incomplete TP ranks')
        for row in results.values():
            path = (study.run / row['file']).resolve()
            if study.run.resolve() not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Existing profile changed after validation')
        study.event('profile_reused', config=tag, ranks=4)
        return results
    destination.mkdir(parents=True, exist_ok=False)
    count = max(32, 5 * concurrency)
    study.bench('profile-warmup', tag+'-warmup', name, inputs, outputs, concurrency, count)
    payload = {'output_dir': str(destination), 'num_steps': 24, 'activities': ['CPU', 'GPU'],
               'profile_by_stage': False, 'with_stack': False, 'record_shapes': False,
               'profile_prefix': tag, 'detailed_annotations': True}
    api = {}
    config_path = destination / 'capture-config.json'
    config_path.write_text(json.dumps({'after_completions': concurrency,
        'url': 'http://127.0.0.1:30000/start_profile', 'payload': payload,
        'status_file': str(destination / 'capture-start.json')}, indent=2))
    previous_config = os.environ.get('INFRA_BENCH_PROFILE_CONFIG')
    try:
        os.environ['INFRA_BENCH_PROFILE_CONFIG'] = str(config_path)
        study.bench('profile-load', tag+'-profiled', name, inputs, outputs, concurrency, count, warmups=0)
    finally:
        if previous_config is None:
            os.environ.pop('INFRA_BENCH_PROFILE_CONFIG', None)
        else:
            os.environ['INFRA_BENCH_PROFILE_CONFIG'] = previous_config
        status_path = destination / 'capture-start.json'
        if status_path.exists():
            api['start'] = json.loads(status_path.read_text())
        try:
            api['stop'] = post('stop_profile')
        except Exception as error:
            api['stop_error'] = str(error)
        (destination/'api.json').write_text(json.dumps({'payload': payload, 'responses': api}, indent=2)+'\n')
    if not api.get('start') or api['start'].get('status') != 200:
        raise ValueError('No successful server profiler start was recorded')
    results = {}
    for path in sorted(destination.glob('*.trace.json*')):
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as stream:
            raw = json.load(stream)
        events = raw.get('traceEvents', []) if isinstance(raw, dict) else raw
        row = trace_summary(events)
        rank_match = re.search(r'TP[-_](\d+)', path.name)
        if not rank_match:
            raise ValueError(f'Cannot identify TP rank: {path.name}')
        rank = int(rank_match.group(1))
        if rank in results:
            raise ValueError(f'Duplicate trace for rank {rank}')
        row.update({'file': str(path.relative_to(study.run)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'rank': rank, 'event_count': len(events)})
        results[rank] = row
    if set(results) != {0, 1, 2, 3}:
        raise ValueError(f'Expected 4 GPU rank traces, got {sorted(results)}')
    if mode == 'mtp' and not all(row['stage_cpu_wall_us']['verify'] > 0 or row['stage_cpu_wall_us']['verify_forward'] > 0 for row in results.values()):
        raise ValueError('MTP GPU traces have no verify stage annotations')
    (destination/'trace-validation.json').write_text(json.dumps({'status': 'valid', 'ranks': results}, indent=2)+'\n')
    study.event('profile_validated', config=tag, ranks=4,
                kernels=sum(row['kernel_count'] for row in results.values()))
    return results


def main():
    from mtp_experiment import Study, atomic_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    study = Study(Path(__file__).resolve().parents[1], args.run_dir)
    outcome = {}
    try:
        if args.smoke_only:
            outcome['short-c1-mtp'] = capture(study, 'short', 512, 128, 1, 'mtp')
        else:
            for mode in ('mtp', 'none'):
                if mode == 'none':
                    study.switch(mode, 'profile-none')
                for name, inputs, outputs, concurrency in (
                    ('short', 512, 128, 1), ('short', 512, 128, 32), ('long-context', 2048, 256, 8)):
                    outcome[f'{name}-c{concurrency}-{mode}'] = capture(study, name, inputs, outputs, concurrency, mode)
    finally:
        if not args.smoke_only:
            study.switch('mtp', 'profile-restored-stable-tp4-mtp')
        atomic_json(study.run / 'profile-outcome.json', outcome)


if __name__ == '__main__':
    main()
