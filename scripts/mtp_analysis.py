#!/usr/bin/env python3
"""Offline paired analysis of the fixed MTP boundary study. Never touches a server."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


def interval_union_us(intervals):
    ordered = sorted(intervals)
    total = 0.0
    end = None
    for start, stop in ordered:
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (start, stop)) or stop < start:
            raise ValueError('Intervals must be finite and end after start')
        if end is None or start >= end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def paired_ratios(records):
    pairs = {}
    for record in records:
        if record.get('phase') != 'baseline':
            raise ValueError('Only formal baseline records can be paired')
        value = record.get('output_throughput')
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Throughput must be finite and positive')
        key = (record['workload'], record['concurrency'], record['round'])
        mode = record['mode']
        if mode not in ('none', 'mtp') or mode in pairs.setdefault(key, {}):
            raise ValueError('Duplicate or unknown mode')
        pairs[key][mode] = record
    ratios = []
    for key in sorted(pairs):
        pair = pairs[key]
        if set(pair) != {'none', 'mtp'}:
            raise ValueError(f'Incomplete pair: {key}')
        first, second = pair['none'], pair['mtp']
        for field in ('seed', 'request_count', 'input_manifest_sha256'):
            if first.get(field) is None or first.get(field) != second.get(field):
                raise ValueError(f'Pair does not match: {field}, {key}')
        if not first['input_manifest_sha256']:
            raise ValueError('Exact input hash is required')
        ratios.append(second['output_throughput'] / first['output_throughput'])
    return ratios


def load_records(run):
    from bench_suite import validate_result
    records = []
    for path in sorted((run / 'baseline/results').glob('*.jsonl')):
        match = re.fullmatch(r'(.+)-c(\d+)-(none|mtp)-r(\d+)-i(\d+)-o(\d+)-c(\d+)\.jsonl', path.name)
        if not match:
            raise ValueError(f'Unexpected formal result filename: {path.name}')
        workload, concurrency, mode, round_id, inputs, outputs, suffix_concurrency = match.groups()
        if concurrency != suffix_concurrency:
            raise ValueError('Filename concurrency mismatch')
        lines = path.read_text().splitlines()
        if len(lines) != 1:
            raise ValueError('Each case must contain exactly one raw row')
        raw = json.loads(lines[0])
        meta = json.loads(path.with_suffix('.run.json').read_text())
        if meta['exit_code'] != 0:
            raise ValueError('Failed benchmark is not a valid formal result')
        validate_result(raw, meta['request_count'], int(inputs), int(outputs))
        if raw['duration'] < 30:
            raise ValueError(f'Formal case shorter than 30 seconds: {path.name}')
        requests = json.loads(path.with_suffix('.requests.json').read_text())
        encoded = json.dumps(requests['requests'], sort_keys=True, separators=(',', ':')).encode()
        actual_hash = hashlib.sha256(encoded).hexdigest()
        if (requests['sha256'] != actual_hash or actual_hash != meta['input_manifest_sha256']
                or requests['count'] != meta['request_count']
                or len(requests['requests']) != meta['request_count']):
            raise ValueError('Actual request manifest mismatch')
        for request in requests['requests']:
            if (len(request['input_ids']) != int(inputs) or request['input_len'] != int(inputs)
                    or request['output_len'] != int(outputs)):
                raise ValueError('Request manifest contains wrong token lengths')
        per_request = json.loads(path.with_suffix('.request-metrics.json').read_text())['requests']
        ordered = sorted(per_request, key=lambda row: row['start_time'])
        measured = ordered[meta['warmup_requests']:]
        if len(measured) != meta['request_count'] or any(not row['success'] or row['error'] for row in measured):
            raise ValueError('Request metrics do not match formal workload')
        accepts = [row['spec_accept_length'] for row in measured if row.get('spec_accept_length')]
        record = {'file': str(path.relative_to(run)), 'workload': workload, 'mode': mode,
                  'concurrency': int(concurrency), 'round': int(round_id), 'phase': 'baseline',
                  'input_len': int(inputs), 'output_len': int(outputs),
                  'seed': meta['seed'], 'request_count': meta['request_count'],
                  'input_manifest_sha256': meta['input_manifest_sha256'],
                  'completed': raw['completed'], 'duration': raw['duration'],
                  'total_output_tokens': raw['total_output_tokens'],
                  'output_throughput': raw['output_throughput'],
                  'service_cumulative_accept_length': raw.get('accept_length'),
                  'mean_request_spec_accept_length': statistics.mean(accepts) if accepts else None,
                  'request_spec_accept_length_count': len(accepts)}
        for key in ('mean_ttft_ms', 'mean_tpot_ms', 'mean_e2e_latency_ms',
                    'p50_e2e_latency_ms', 'median_e2e_latency_ms', 'p95_e2e_latency_ms', 'p99_e2e_latency_ms'):
            record[key] = raw.get(key)
        records.append(record)
    if not records:
        raise ValueError('No formal baseline records found')
    paired_ratios(records)
    return records


def summarize(records):
    paired_ratios(records)
    summary = []
    for workload, concurrency in sorted({(row['workload'], row['concurrency']) for row in records}):
        rows = [r for r in records if (r['workload'], r['concurrency']) == (workload, concurrency)]
        ratios = paired_ratios(rows)
        if len(ratios) < 3:
            raise ValueError('At least three independent paired rounds are required')
        entry = {'workload': workload, 'concurrency': concurrency, 'paired_rounds': len(ratios),
                 'paired_ratios': ratios, 'mean_paired_speedup': statistics.mean(ratios),
                 'min_paired_speedup': min(ratios), 'max_paired_speedup': max(ratios),
                 'measured_output_tokens': sum(row['total_output_tokens'] for row in rows),
                 'modes': {}}
        for mode in ('none', 'mtp'):
            selected = [row for row in rows if row['mode'] == mode]
            entry['modes'][mode] = {
                key: statistics.mean(row[key] for row in selected)
                for key in ('output_throughput', 'mean_ttft_ms', 'mean_tpot_ms',
                            'mean_e2e_latency_ms', 'p95_e2e_latency_ms', 'p99_e2e_latency_ms')}
            entry['modes'][mode]['request_accept_length'] = (
                statistics.mean(row['mean_request_spec_accept_length'] for row in selected)
                if all(row['mean_request_spec_accept_length'] is not None for row in selected) else None)
        summary.append(entry)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.run_dir.resolve())
    summary = summarize(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {'source_run': str(args.run_dir.resolve()), 'formal_cases': len(records),
              'completed_requests': sum(row['completed'] for row in records),
              'summary': summary, 'records': records,
              'aggregation': {'speedup': 'arithmetic mean of independent paired-round ratios',
                              'mode_metrics': 'arithmetic means of run-level metrics',
                              'p95_p99': 'means of per-run percentiles, not pooled request percentiles',
                              'accept_length': 'mean of request-level submitted tokens per speculative round; not acceptance probability'},
              'evidence_limits': ['No profiler data inferred from client timing.',
                                  'Service cumulative accept_length is not a per-case counter.',
                                  'Three rounds describe variation; requests are not independent rounds.']}
    with (args.output_dir / 'summary.json').open('x') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    with (args.output_dir / 'paired-runs.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({'status': 'valid', 'cases': len(records), 'groups': len(summary)}))


if __name__ == '__main__':
    main()
