#!/usr/bin/env python3
"""Launch DeepSeek V4 Flash with a fixed physical GPU allowlist."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

ALLOWED_GPUS = {
    0: 'GPU-PUBLIC-0',
    1: 'GPU-PUBLIC-3',
    2: 'GPU-PUBLIC-1',
    3: 'GPU-PUBLIC-2',
}


def validate_inventory(text, gpu_ids):
    rows = {}
    for line in text.splitlines():
        if line.strip():
            index, uuid = line.split(',', 1)
            rows[int(index.strip())] = uuid.strip()
    if rows != {i: ALLOWED_GPUS[i] for i in gpu_ids}:
        raise ValueError('Physical GPU inventory does not match the authorized UUIDs')


def validate_visible_uuids(actual, gpu_ids):
    if actual != [ALLOWED_GPUS[i] for i in gpu_ids]:
        raise ValueError('CUDA device order does not match the authorized physical GPUs')


DRIVER_UUID_CHECK = r"""
import ctypes,json,uuid
lib=ctypes.CDLL('libcuda.so.1')
def check(code):
    if code != 0: raise RuntimeError(f'CUDA driver query failed: {code}')
check(lib.cuInit(0))
count=ctypes.c_int()
check(lib.cuDeviceGetCount(ctypes.byref(count)))
result=[]
for i in range(count.value):
    device=ctypes.c_int()
    check(lib.cuDeviceGet(ctypes.byref(device),i))
    value=(ctypes.c_ubyte*16)()
    check(lib.cuDeviceGetUuid_v2(ctypes.byref(value),device))
    result.append('GPU-'+str(uuid.UUID(bytes=bytes(value))))
print(json.dumps(result))
"""


def build_launch(root, gpu_ids, speculative, port, gpu_addressing='uuid', speculative_steps=3, adaptive_config=None):
    if not gpu_ids or any(i not in ALLOWED_GPUS for i in gpu_ids):
        raise ValueError('Only physical GPUs 0,1,2,3 are authorized')
    if len(set(gpu_ids)) != len(gpu_ids) or len(gpu_ids) not in (2, 4):
        raise ValueError('Use two or four distinct authorized GPUs')
    if gpu_addressing not in ('uuid', 'ordinal'):
        raise ValueError('Invalid GPU addressing mode')
    if speculative not in ('none', 'mtp') or not 1024 <= port <= 65535:
        raise ValueError('Invalid speculative mode or port')
    if type(speculative_steps) is not int or speculative_steps not in (1, 2, 3):
        raise ValueError('Study supports only 1, 2, or 3 speculative steps')
    if speculative == 'none' and (speculative_steps != 3 or adaptive_config is not None):
        raise ValueError('Speculative options require MTP')
    root = Path(root).resolve()
    if adaptive_config is not None:
        adaptive_config = Path(adaptive_config).resolve()
        if root not in adaptive_config.parents or not adaptive_config.is_file():
            raise ValueError('Adaptive configuration must be an existing project JSON file')
        config = json.loads(adaptive_config.read_text())
        tiers = [value for key, value in config.items() if str(key).isdigit()]
        if not tiers or any(not isinstance(tier, dict) or not tier.get('candidate_steps') or
                            any(type(step) is not int or step not in (1, 2, 3) for step in tier['candidate_steps']) for tier in tiers):
            raise ValueError('Adaptive policy must restrict candidate steps to 1,2,3')
    env = os.environ.copy()
    env.update({
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'CUDA_VISIBLE_DEVICES': ','.join(ALLOWED_GPUS[i] if gpu_addressing == 'uuid' else str(i) for i in gpu_ids),
        'CUDA_HOME': '/usr/local/cuda',
        'PATH': str(root / '.venv/bin') + os.pathsep + os.environ.get('PATH', ''),
        'HF_HOME': str(root / 'cache/huggingface'),
        'TOKENIZERS_PARALLELISM': 'false',
        'OMP_NUM_THREADS': '8',
    })
    command = [
        str(root / '.venv/bin/python'), '-m', 'sglang.launch_server',
        '--model-path', str(root / 'models/DeepSeek-V4-Flash'),
        '--served-model-name', 'deepseek-v4-flash',
        '--trust-remote-code', '--tp', str(len(gpu_ids)),
        '--moe-runner-backend', 'flashinfer_mxfp4',
        '--mem-fraction-static', '0.92',
        '--context-length', '8192', '--chunked-prefill-size', '2048',
        '--max-running-requests', '32', '--cuda-graph-max-bs-decode', '32',
        '--host', '127.0.0.1', '--port', str(port),
        '--disable-radix-cache',
    ]
    if speculative == 'mtp':
        command.extend([
            '--speculative-algorithm', 'EAGLE',
            '--speculative-num-steps', str(speculative_steps), '--speculative-eagle-topk', '1',
            '--speculative-num-draft-tokens', str(speculative_steps + 1),
        ])
    if adaptive_config is not None:
        command.extend(['--speculative-adaptive', '--speculative-adaptive-config', str(adaptive_config)])
    return command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--speculative', choices=('none', 'mtp'), default='none')
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--gpu-addressing', choices=('uuid', 'ordinal'), default='uuid')
    parser.add_argument('--speculative-steps', type=int, default=3)
    parser.add_argument('--adaptive-config', type=Path, default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    gpu_ids = [int(i) for i in args.gpus.split(',')]
    command, env = build_launch(root, gpu_ids, args.speculative, args.port, args.gpu_addressing,
                                args.speculative_steps, args.adaptive_config)
    print('CUDA_VISIBLE_DEVICES=' + env['CUDA_VISIBLE_DEVICES'], flush=True)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        inventory = subprocess.check_output(['nvidia-smi', '-i', ','.join(str(i) for i in gpu_ids), '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
        validate_inventory(inventory, gpu_ids)
        visible = json.loads(subprocess.check_output([command[0], '-c', DRIVER_UUID_CHECK], env=env, text=True))
        validate_visible_uuids(visible, gpu_ids)
        os.execve(command[0], command, env)


if __name__ == '__main__':
    main()
