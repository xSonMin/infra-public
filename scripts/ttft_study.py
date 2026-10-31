#!/usr/bin/env python3
"""Bounded TTFT diagnostics for this project's four authorized GPUs and loopback service."""
import argparse
import asyncio
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch
import stop
from mtp_experiment import Study, atomic_json, existing_sequence


class StreamMeasurement:
    def __init__(self):
        self.first_event = None
        self.first_positive_token = None
        self.first_text = None
        self.first_meta = None
        self.final_meta = {}
        self.text = ''
        self.events = []

    def feed(self, data, elapsed):
        if self.first_event is None:
            self.first_event = elapsed
        meta = data.get('meta_info') or {}
        if self.first_meta is None:
            self.first_meta = dict(meta)
        tokens = meta.get('completion_tokens', 0)
        if tokens > 0 and self.first_positive_token is None:
            self.first_positive_token = elapsed
        if data.get('text') and self.first_text is None:
            self.first_text = elapsed
        self.final_meta = dict(meta)
        self.text = data.get('text') or self.text
        self.events.append({'elapsed_s': elapsed, 'completion_tokens': tokens,
                            'nonempty_text': bool(data.get('text'))})


def validate_request(row, expected=128):
    if not row.get('success') or row.get('output_tokens') != expected:
        raise ValueError('Failed request or incorrect actual output token count')
    ttft, latency = row.get('ttft_s'), row.get('latency_s')
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in [ttft, latency]):
        raise ValueError('Missing or nonfinite request timing')
    if not 0 <= ttft <= latency:
        raise ValueError('Request timestamps are inconsistent')


def schedule_offsets(kind, count, rate=4):
    if type(count) is not int or count < 1 or not math.isfinite(rate) or rate <= 0:
        raise ValueError('A finite positive rate and positive request count are required')
    if kind not in ('solo', 'burst', 'paced'):
        raise ValueError('Unknown arrival protocol')
    return [i / rate if kind == 'paced' else 0 for i in range(count)]


def arrival_clusters(starts, threshold=.01):
    if not math.isfinite(threshold) or threshold < 0 or any(not math.isfinite(x) for x in starts):
        raise ValueError('Invalid arrival observation')
    ordered = sorted(starts)
    sizes = []
    for i, value in enumerate(ordered):
        if i == 0 or value - ordered[i-1] > threshold:
            sizes.append(0)
        sizes[-1] += 1
    return sizes


def diagnostic_command(root, mode, steps, timing=True):
    command, env = launch.build_launch(root, [0, 1, 2, 3], mode, 30000,
                                        gpu_addressing='uuid', speculative_steps=steps)
    if timing:
        command += ['--enable-metrics', '--enable-request-time-stats-logging',
                    '--random-seed', '20261030']
    return command, env


class TimingStudy(Study):
    def __init__(self, root, run):
        self.root, self.run = Path(root).resolve(), Path(run).resolve()
        if self.run.parent != self.root/'results/ttft-study-2026-10-30' or not self.run.name.startswith('run-'):
            raise ValueError('TTFT diagnostics must remain in the dedicated project run directory')
        self.run.mkdir(parents=True, exist_ok=True)
        self.launch, self.stop = launch, stop
        self.active_path = self.root/'results/active-server.json'
        self.pyth = str(self.root/'.venv/bin/python')
        self.sequence = existing_sequence(self.run/'server-logs')

    def server_command(self, mode, steps, timing):
        return diagnostic_command(self.root, mode, steps, timing)

    def switch(self, mode, tag, steps=3, timing=True):
        old = json.loads(self.active_path.read_text())
        self.guard(old)
        if Path('/proc', str(old['pid'])).exists():
            self.event('stop_server', pid=old['pid'], tag=old.get('tag'))
            subprocess.run([self.pyth, str(self.root/'scripts/stop.py')], check=True, timeout=45)
        deadline = time.monotonic()+60
        while self.gpu_processes():
            if time.monotonic() > deadline:
                raise TimeoutError('Project GPU workers did not exit')
            time.sleep(.5)
        command, env = self.server_command(mode, steps, timing)
        inventory = subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        launch.validate_inventory(inventory, [0,1,2,3])
        visible = json.loads(subprocess.check_output([self.pyth,'-c',launch.DRIVER_UUID_CHECK],env=env,text=True,timeout=20))
        launch.validate_visible_uuids(visible, [0,1,2,3])
        self.sequence += 1
        log = self.run/'server-logs'/f'{self.sequence:02d}-{tag}.log'
        log.parent.mkdir(exist_ok=True)
        with log.open('x') as output:
            proc = subprocess.Popen(command,cwd=self.root,env=env,stdout=output,stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL,start_new_session=True)
        record = {'pid':proc.pid,'tag':tag,'command':command,'log':str(log),'started_at':time.time()}
        atomic_json(self.active_path,record)
        self.event('start_server',**record)
        deadline = time.monotonic()+240
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f'Server exited; inspect {log}')
            try:
                with urllib.request.urlopen('http://127.0.0.1:30000/health',timeout=5) as response:
                    if response.status == 200:
                        with urllib.request.urlopen('http://127.0.0.1:30000/get_server_info',timeout=10) as reply:
                            info = json.load(reply)
                        assert bool(info['enable_metrics']) == timing
                        assert (info['speculative_algorithm'] is not None) == (mode == 'mtp')
                        if mode == 'mtp':assert info['speculative_num_steps'] == steps
                        atomic_json(self.run/f'{tag}-server-info.json',info)
                        self.guard(record)
                        self.event('server_ready',mode=mode,steps=steps,timing=timing,pid=proc.pid)
                        return record
            except (OSError,ValueError):
                pass
            time.sleep(2)
        raise TimeoutError('Diagnostic server was not ready within 240 seconds')


async def run_case(prompts, kind, label, round_id, output, output_len=128):
    import aiohttp
    offsets = schedule_offsets(kind,len(prompts))
    origin = time.perf_counter()+.05
    active = 0
    peak = 0

    async def request(index, tokens, offset):
        nonlocal active, peak
        await asyncio.sleep(max(0,origin+offset-time.perf_counter()))
        if active >= 32:
            raise RuntimeError('Open-loop safety bound exceeded; no implicit closed-loop delay is allowed')
        active += 1
        peak = max(peak,active)
        rid = f'ttft-{label}-{kind}-r{round_id}-i{index}'
        stream = StreamMeasurement()
        payload = {'rid':rid,'input_ids':tokens,'stream':True,
                   'sampling_params':{'temperature':0,'max_new_tokens':output_len,'ignore_eos':True}}
        row = {'rid':rid,'index':index,'planned_offset_s':offset}
        try:
            timeout = aiohttp.ClientTimeout(total=90)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                start = time.perf_counter()
                row.update(start_time=start,client_send_realtime=time.time(),
                           actual_offset_s=start-origin,send_lag_s=start-origin-offset)
                async with session.post('http://127.0.0.1:30000/generate',json=payload) as response:
                    if response.status != 200:
                        raise RuntimeError(f'HTTP {response.status}: '+(await response.text())[:300])
                    async for line in response.content:
                        raw = line.strip()
                        if not raw or raw.startswith(b':'):continue
                        if raw.startswith(b'data:'):raw=raw[5:].strip()
                        if raw == b'[DONE]':continue
                        stream.feed(json.loads(raw),time.perf_counter()-start)
                row.update(success=True,latency_s=time.perf_counter()-start,ttft_s=stream.first_text,
                           first_event_s=stream.first_event,first_positive_token_s=stream.first_positive_token,
                           output_tokens=stream.final_meta.get('completion_tokens'),
                           first_meta=stream.first_meta,final_meta=stream.final_meta,events=stream.events,
                           output_text_sha256=hashlib.sha256(stream.text.encode()).hexdigest())
                validate_request(row,output_len)
                if kind == 'paced' and row['send_lag_s'] > .1:
                    raise RuntimeError('Fixed-arrival send lag exceeded 100ms')
                return row
        except Exception as error:
            row.update(success=False,error=repr(error))
            return row
        finally:
            active -= 1

    rows = await asyncio.gather(*(request(i,prompt,offset) for i,(prompt,offset) in enumerate(zip(prompts,offsets))))
    result = {'label':label,'protocol':kind,'round':round_id,'request_count':len(prompts),
              'input_len':512,'output_len':output_len,'peak_inflight':peak,
              'rate_per_second':4 if kind=='paced' else None,
              'wall_seconds':time.perf_counter()-origin,'requests':rows}
    with Path(output).open('x') as file:json.dump(result,file,ensure_ascii=False,indent=2)
    if any(not r['success'] for r in rows):
        raise RuntimeError('One or more diagnostic requests failed; raw failure records were saved')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    study=TimingStudy(root,args.run_dir)
    with (study.run/'workflow-started.json').open('x') as file:
        json.dump({'pid':os.getpid(),'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},file)
    manifest=json.loads((study.run/'input-manifest.json').read_text())
    prompts=[r['input_ids'] for r in manifest['requests']]
    actual=hashlib.sha256(json.dumps(manifest['requests'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert actual==manifest['sha256'] and len(prompts)==64 and all(len(p)==512 for p in prompts)
    (study.run/'cases').mkdir(exist_ok=True)
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    outcome={'status':'running','service_restored':False,'formal_cases':0,'formal_requests':0}
    try:
        for label,mode,steps in [('none','none',3),('fixed1','mtp',1),('fixed3','mtp',3)]:
            study.switch(mode,'diagnostic-'+label,steps=steps,timing=True)
            asyncio.run(run_case(prompts[:32],'burst',label,0,study.run/'cases'/f'{label}-warmup.json'))
            for rnd in range(1,4):
                for kind,count in [('solo',1),('burst',32),('paced',64)]:
                    study.event('case_begin',label=label,protocol=kind,round=rnd)
                    asyncio.run(run_case(prompts[:count],kind,label,rnd,study.run/'cases'/f'{label}-{kind}-r{rnd}.json'))
                    outcome['formal_cases']+=1;outcome['formal_requests']+=count
                    atomic_json(study.run/'progress.json',outcome)
                    study.event('case_end',label=label,protocol=kind,round=rnd)
        outcome['status']='completed'
    except BaseException as error:
        outcome.update(status='failed',error=repr(error))
        study.event('diagnostic_failure',error=repr(error))
        raise
    finally:
        try:
            record=study.switch('mtp','ttft-restored-stable-tp4-mtp',steps=3,timing=False)
            outcome.update(service_restored=True,restored_pid=record['pid'])
        except BaseException as error:
            outcome['restoration_error']=repr(error)
        outcome['completed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(study.run/'workflow-outcome.json',outcome)
        study.event('workflow_finished',**outcome)
    print('TTFT_STUDY_COMPLETE',flush=True)


if __name__=='__main__':main()
