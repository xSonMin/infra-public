#!/usr/bin/env python3
"""Evidence-gated fixed-depth screening and frozen-policy verification for this study."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_suite import validate_result
from mtp_experiment import Study, atomic_json


def choose_depth(rows):
    by_depth = {}
    seen = set()
    for row in rows:
        step, round_id, value = row['steps'], row['round'], row['throughput']
        if step not in (1, 2, 3) or not math.isfinite(value) or value <= 0 or (step, round_id) in seen:
            raise ValueError('Invalid or duplicate fixed-depth measurement')
        seen.add((step, round_id))
        by_depth.setdefault(step, []).append(value)
    if set(by_depth) != {1, 2, 3} or any(len(values) < 3 for values in by_depth.values()):
        raise ValueError('Three independent repeats of every depth are required')
    return max(by_depth, key=lambda step: statistics.mean(by_depth[step]))


def policy_config(mapping):
    if not mapping or 1 not in mapping:
        raise ValueError('A policy must cover batch size 1')
    if any(type(bs) is not int or bs < 1 or type(step) is not int or step not in (1, 2, 3)
           for bs, step in mapping.items()):
        raise ValueError('Study policy candidates are limited to 1,2,3 positive steps')
    # Singleton tiers freeze the selected batch mapping; the existing framework performs runtime switching.
    return {str(bs): {'candidate_steps': [step]} for bs, step in sorted(mapping.items())}


def assess_candidate(reference, candidate):
    throughput_ratio = candidate['output_throughput'] / reference['output_throughput']
    ttft_ratio = candidate['mean_ttft_ms'] / reference['mean_ttft_ms']
    p99_ratio = candidate['p99_e2e_latency_ms'] / reference['p99_e2e_latency_ms']
    checks = {'throughput_at_least_five_percent': throughput_ratio >= 1.05,
              'ttft_guard': ttft_ratio <= 1.05,
              'p99_guard': p99_ratio <= 1.05,
              'no_errors': candidate['errors'] == 0,
              'independent_round_count': candidate['rounds'] >= 5,
              'tail_sample_count': candidate['requests'] >= 1000}
    return {'retained': all(checks.values()), 'checks': checks,
            'throughput_ratio': throughput_ratio, 'ttft_ratio': ttft_ratio, 'p99_ratio': p99_ratio,
            'limits': ['This numerical gate does not replace independent quality checks or workload-scoped interpretation.']}


def screen(study, rounds=3):
    counts = json.loads((study.run / 'calibration.json').read_text())['frozen_request_counts']
    rows = []
    try:
        for steps in (1, 2, 3):
            study.switch('mtp', f'tuning-fixed-s{steps}', steps=steps)
            for rnd in range(1, rounds + 1):
                raw, meta = study.bench('tuning', f'short-c32-s{steps}-r{rnd}',
                                       'short', 512, 128, 32, counts['short-c32'])
                rows.append({'steps': steps, 'round': rnd, 'throughput': raw['output_throughput'],
                             'mean_ttft_ms': raw['mean_ttft_ms'], 'p99_e2e_latency_ms': raw['p99_e2e_latency_ms'],
                             'manifest': meta['input_manifest_sha256'], 'duration': raw['duration']})
        if len({row['manifest'] for row in rows}) != 1:
            raise ValueError('Tuning candidates did not use identical request manifests')
        selected = choose_depth(rows)
        mapping = {1: 3, 32: selected}
        config = policy_config(mapping)
        path = study.run / 'adaptive-policy.json'
        with path.open('x') as stream:
            json.dump(config, stream, indent=2)
        atomic_json(study.run / 'frozen-policy.json', {
            'selected_high_batch_steps': selected, 'mapping': mapping,
            'config_file': str(path), 'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'selection_data': rows,
            'validation_dataset_sha256': hashlib.sha256((study.root / 'sources/benchmark-prompts-validation.json').read_bytes()).hexdigest(),
            'limits': ['Only batch size 32 was screened for shallower depth.',
                       'Batch sizes below 32 retain the original depth 3.',
                       'A singleton-tier framework policy is a batch mapping, not a new learning algorithm.']})
        study.event('tuning_complete', selected_high_batch_steps=selected)
    finally:
        study.switch('mtp', 'tuning-restored-stable-tp4-mtp')


def verify(study, rounds=3, replay=False):
    frozen = json.loads((study.run / 'frozen-policy.json').read_text())
    selected = frozen['selected_high_batch_steps']
    policy = Path(frozen['config_file'])
    if hashlib.sha256(policy.read_bytes()).hexdigest() != frozen['config_sha256']:
        raise ValueError('Frozen policy changed after selection')
    dataset = study.root / 'sources/benchmark-prompts-validation.json'
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != frozen['validation_dataset_sha256']:
        raise ValueError('Validation corpus changed after policy selection')
    counts = json.loads((study.run/'calibration.json').read_text())['frozen_request_counts']
    validation_count = max(512, counts['short-c32'])
    if not replay:
        atomic_json(study.run / 'validation-protocol.json', {
            'request_count_per_run': validation_count, 'rounds': rounds, 'seed': 31415,
            'dataset_sha256': frozen['validation_dataset_sha256'],
            'note': 'Count raised before held-out evaluation so the faster variants retain a >=30s window.'})
    configurations = [('none', 'none', 3, None), ('fixed3', 'mtp', 3, None),
                      ('best-fixed', 'mtp', selected, None), ('adaptive', 'mtp', 3, policy)]
    try:
        for label, mode, steps, config in configurations:
            study.switch(mode, f'validation-{label}', steps=steps, adaptive_config=config)
            if replay:
                records = []
                replay_start = time.perf_counter()
                for segment, concurrency in enumerate((1, 32, 8), 1):
                    raw, meta = study.bench('replay', f'{label}-seg{segment}-c{concurrency}', 'short',
                                           512, 128, concurrency, counts[f'short-c{concurrency}'],
                                           seed=31415, dataset=dataset)
                    records.append({'segment': segment, 'concurrency': concurrency,
                                    'duration': raw['duration'], 'output_tokens': raw['total_output_tokens'],
                                    'manifest': meta['input_manifest_sha256']})
                atomic_json(study.run / f'replay-{label}.json', {'segments': records,
                    'scope': 'sequential client-concurrency segments; segment boundaries include warmup gaps',
                    'total_wall_seconds': time.perf_counter() - replay_start,
                    'replay_wall_throughput': sum(x['output_tokens'] for x in records)/(time.perf_counter()-replay_start),
                    'benchmark_window_throughput': sum(x['output_tokens'] for x in records)/sum(x['duration'] for x in records),
                    'limits': ['This is a segmented replay, not an uninterrupted arrival-rate trace.']})
            else:
                for rnd in range(1, rounds + 1):
                    raw, _ = study.bench('validation', f'short-c32-{label}-r{rnd}', 'short', 512, 128, 32,
                                         validation_count, seed=31415, dataset=dataset)
                    if raw['duration'] < 30:
                        raise ValueError('Held-out variants require a larger common request count')
        study.event('verification_complete', phase='replay' if replay else 'validation', rounds=rounds)
    finally:
        study.switch('mtp', 'verification-restored-stable-tp4-mtp')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--phase', choices=('screen', 'validation', 'replay'), required=True)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 3:
        parser.error('At least three repeats are required')
    study = Study(Path(__file__).resolve().parents[1], args.run_dir)
    decision_path = study.run/'analysis/decision.json'
    if not decision_path.exists() or not json.loads(decision_path.read_text()).get('allow_depth_screen'):
        raise ValueError('A saved evidence-based analysis decision is required before tuning')
    if args.phase == 'screen':
        screen(study, args.rounds)
    else:
        verify(study, args.rounds, args.phase == 'replay')


if __name__ == '__main__':
    main()
