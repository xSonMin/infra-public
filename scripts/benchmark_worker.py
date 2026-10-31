#!/usr/bin/env python3
"""Record exact benchmark inputs and request metrics around the installed SGLang client."""
import asyncio
import hashlib
import time
import urllib.request
import json
import os
from pathlib import Path


def request_manifest(rows):
    requests = []
    for row in rows:
        if not isinstance(row.prompt, list) or not all(isinstance(x, int) for x in row.prompt):
            raise ValueError('Exact workload recording requires token ID prompts')
        requests.append({'input_ids': row.prompt, 'input_len': int(row.prompt_len),
                         'output_len': int(row.output_len)})
    encoded = json.dumps(requests, sort_keys=True, separators=(',', ':')).encode()
    return {'sha256': hashlib.sha256(encoded).hexdigest(), 'count': len(requests), 'requests': requests}


class ProfileTrigger:
    def __init__(self, after_completions):
        if type(after_completions) is not int or after_completions < 1:
            raise ValueError('A positive completed-request count is required')
        self.after_completions = after_completions
        self.started = False

    def should_start(self, completions):
        if not self.started and completions >= self.after_completions:
            self.started = True
            return True
        return False


def start_profile(config):
    request = urllib.request.Request(config['url'],
        data=json.dumps(config['payload']).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=180) as response:
        return {'status': response.status, 'body': response.read().decode(),
                'started_at_perf_counter': time.perf_counter()}


def main():
    from sglang.benchmark import serving
    original_dataset = serving.get_dataset
    original_request = serving.ASYNC_REQUEST_FUNCS['sglang']
    manifest_path = Path(os.environ['INFRA_REQUEST_MANIFEST'])
    metric_path = Path(os.environ['INFRA_REQUEST_METRICS'])
    outputs = []
    profile_path = os.environ.get('INFRA_BENCH_PROFILE_CONFIG')
    profile_config = json.loads(Path(profile_path).read_text()) if profile_path else None
    trigger = ProfileTrigger(profile_config['after_completions']) if profile_config else None
    profile_status = None

    def recorded_dataset(args, tokenizer, model_id):
        rows = original_dataset(args, tokenizer, model_id)
        if len(rows) != args.num_prompts:
            raise ValueError(f'Dataset returned {len(rows)} rows; expected {args.num_prompts}')
        with manifest_path.open('x') as output:
            json.dump(request_manifest(rows), output, ensure_ascii=False)
        return rows

    async def recorded_request(request_func_input, pbar=None):
        result = await original_request(request_func_input, pbar)
        outputs.append({key: getattr(result, key, None) for key in
                        ('start_time', 'latency', 'ttft', 'success', 'error', 'output_len',
                         'spec_accept_length', 'spec_cap_length', 'spec_block_accept_length')})
        nonlocal profile_status
        if trigger is not None and trigger.should_start(len(outputs)):
            profile_status = await asyncio.to_thread(start_profile, profile_config)
            Path(profile_config['status_file']).write_text(json.dumps(profile_status, indent=2))
        return result

    serving.get_dataset = recorded_dataset
    serving.ASYNC_REQUEST_FUNCS['sglang'] = recorded_request
    try:
        serving.cli_main()
    finally:
        with metric_path.open('x') as output:
            json.dump({'scope': 'all_requests_including_warmup', 'requests': outputs}, output)


if __name__ == '__main__':
    main()
