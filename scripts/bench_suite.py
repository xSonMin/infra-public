#!/usr/bin/env python3
"""Run controlled localhost SGLang serving benchmarks, preserving raw JSONL."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request


def request_count(concurrency, override=None):
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise ValueError('Concurrency must be a positive integer')
    if override is None:
        return max(4, concurrency * 2)
    if not isinstance(override, int) or isinstance(override, bool) or override < concurrency:
        raise ValueError('Prompt count must be an integer not below concurrency')
    return override


def warmup_count(concurrency, override=None):
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise ValueError('Concurrency must be a positive integer')
    if override is None:
        return max(2, concurrency)
    if not isinstance(override, int) or isinstance(override, bool) or override < 0:
        raise ValueError('Warmup requests must be a non-negative integer')
    return override


def validate_result(result, requests, input_len, output_len):
    expected = {'completed': requests, 'total_input_tokens': requests * input_len,
                'total_output_tokens': requests * output_len}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f'{key}: expected {value}, got {result.get(key)}')
    duration = result.get('duration')
    throughput = result.get('output_throughput')
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f'duration must be a finite positive number, got {duration!r}')
    if not isinstance(throughput, (int, float)) or not math.isfinite(throughput) or not math.isclose(
        throughput, result['total_output_tokens'] / duration, rel_tol=1e-6
    ):
        raise ValueError('Reported throughput disagrees with actual generated tokens / elapsed time')

    for key, expected_length in (('input_lens', input_len), ('output_lens', output_len)):
        values = result.get(key)
        if values is not None:
            if not isinstance(values, list) or len(values) != requests:
                raise ValueError(f'{key} must contain one value per request')
            if any(value != expected_length for value in values):
                raise ValueError(f'{key} contains a request with an unexpected token length')

    errors = result.get('errors')
    if errors is not None:
        if not isinstance(errors, list) or len(errors) != requests:
            raise ValueError('errors must contain one value per request')
        if any(error != '' for error in errors):
            raise ValueError('Benchmark returned request errors')


def resolve_output_root(root, value):
    root = Path(root).resolve()
    if value is None:
        return root
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if path != root and root not in path.parents:
        raise ValueError('Output directory must stay within the project root')
    return path


def resolve_dataset(root, value=None):
    root = Path(root).resolve()
    path = Path(value) if value is not None else root / 'sources/benchmark-prompts.json'
    path = (path if path.is_absolute() else root / path).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError('Dataset must be an existing project file')
    return path


def prepare_dataset(source, requests, metadata_dir):
    rows = json.loads(source.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('Benchmark corpus must contain conversations')
    if requests <= len(rows):
        return source
    expanded = rows * math.ceil(requests / len(rows))
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / f'corpus-expanded-{len(expanded)}.json'
    content = json.dumps(expanded, ensure_ascii=False)
    if path.exists():
        if path.read_text() != content:
            raise ValueError('Expanded dataset would overwrite different content')
    else:
        with path.open('x') as output:
            output.write(content)
    return path


def fetch_json(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return {'status': response.status, 'body': json.load(response)}
    except urllib.error.HTTPError as error:
        return {'status': error.code, 'error': error.read().decode(errors='replace')}
    except Exception as error:
        return {'error': str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--input-len', type=int, default=512)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--concurrency', default='1,8,32')
    parser.add_argument('--num-prompts', type=int, default=None)
    parser.add_argument('--warmup-requests', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--dataset-path', type=Path, default=None)
    args = parser.parse_args()
    if not re.fullmatch('[a-zA-Z0-9-]+', args.tag):
        parser.error('Tag must contain only letters, digits, and hyphens')
    try:
        concurrencies = [int(x) for x in args.concurrency.split(',')]
        if not concurrencies or any(x < 1 for x in concurrencies):
            raise ValueError('Concurrency must contain positive integers')
        for concurrency in concurrencies:
            request_count(concurrency, args.num_prompts)
            warmup_count(concurrency, args.warmup_requests)
    except ValueError as error:
        parser.error(str(error))

    root = Path(__file__).resolve().parents[1]
    output_root = resolve_output_root(root, args.output_dir)
    result_dir = output_root / 'results'
    log_dir = output_root / 'logs'
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    dataset = resolve_dataset(root, args.dataset_path)
    model = root / 'models/DeepSeek-V4-Flash'
    models = fetch_json(f'http://127.0.0.1:{args.port}/v1/models')
    model_ids = [item['id'] for item in models.get('body', {}).get('data', [])]
    if models.get('status') != 200 or 'deepseek-v4-flash' not in model_ids:
        raise RuntimeError(f'Served model check failed: {models}')
    server_info = fetch_json(f'http://127.0.0.1:{args.port}/get_server_info')
    server_info_path = None
    if args.output_dir:
        metadata_dir = output_root / 'metadata'
        metadata_dir.mkdir(parents=True, exist_ok=True)
        server_info_path = metadata_dir / f'{args.tag}-server-info.json'
        if server_info_path.exists():
            raise FileExistsError(f'Refusing to overwrite server snapshot: {server_info_path}')
        server_info_path.write_text(json.dumps(server_info, ensure_ascii=False, indent=2) + '\n')

    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': '', 'HF_HOME': str(root / 'cache/huggingface'),
                'PATH': str(root / '.venv/bin') + os.pathsep + env.get('PATH', ''),
                'TOKENIZERS_PARALLELISM': 'false'})
    for concurrency in concurrencies:
        requests = request_count(concurrency, args.num_prompts)
        warmups = warmup_count(concurrency, args.warmup_requests)
        suffix = f'{args.tag}-i{args.input_len}-o{args.output_len}-c{concurrency}'
        result = result_dir / (suffix + '.jsonl')
        effective_dataset = prepare_dataset(dataset, requests, output_root / 'metadata')
        if result.exists():
            raise FileExistsError(f'Refusing to append a duplicate run: {result}')
        worker_env = env.copy()
        invocation = ['-m', 'sglang.benchmark.serving']
        input_manifest = None
        if args.output_dir:
            input_manifest = result_dir / (suffix + '.requests.json')
            worker_env['INFRA_REQUEST_MANIFEST'] = str(input_manifest)
            worker_env['INFRA_REQUEST_METRICS'] = str(result_dir / (suffix + '.request-metrics.json'))
            invocation = [str(root / 'scripts/benchmark_worker.py')]
        command = [
            str(root / '.venv/bin/python'), *invocation,
            '--backend', 'sglang', '--host', '127.0.0.1', '--port', str(args.port),
            '--model', str(model), '--served-model-name', 'deepseek-v4-flash',
            '--tokenizer', str(model), '--dataset-name', 'random',
            '--dataset-path', str(effective_dataset), '--tokenize-prompt',
            '--random-input-len', str(args.input_len),
            '--random-output-len', str(args.output_len), '--random-range-ratio', '1.0',
            '--num-prompts', str(requests), '--max-concurrency', str(concurrency),
            '--warmup-requests', str(warmups), '--seed', str(args.seed), '--disable-tqdm',
            '--output-file', str(result), '--output-details',
        ]
        print('BENCH_START', suffix, flush=True)
        start = time.time()
        with (log_dir / (suffix + '.log')).open('w') as log:
            process = subprocess.run(command, env=worker_env, cwd=root, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=1800)
        metadata = {
            'tag': args.tag,
            'stage': args.tag,
            'command': command,
            'exit_code': process.returncode,
            'wall_seconds': time.time() - start,
            'dataset_sha256': hashlib.sha256(effective_dataset.read_bytes()).hexdigest(),
            'source_dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
            'input_manifest_sha256': json.loads(input_manifest.read_text())['sha256'] if input_manifest and input_manifest.exists() else None,
            'input_len': args.input_len,
            'output_len': args.output_len,
            'concurrency': concurrency,
            'request_count': requests,
            'warmup_requests': warmups,
            'seed': args.seed,
            'output_dir': str(output_root),
            'server_info_snapshot': str(server_info_path.relative_to(output_root)) if server_info_path else None,
        }
        (result_dir / (suffix + '.run.json')).write_text(json.dumps(metadata, indent=2) + '\n')
        print('BENCH_END', suffix, process.returncode, flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)
        rows = result.read_text().splitlines()
        if len(rows) != 1:
            raise ValueError('Expected exactly one raw result row')
        validate_result(json.loads(rows[0]), requests, args.input_len, args.output_len)
    print('SUITE_COMPLETE', args.tag, flush=True)


if __name__ == '__main__':
    main()
