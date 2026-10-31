#!/usr/bin/env python3
"""Finite nightly model experiment with file locking, checkpoints, cutoff and scoped restoration."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parent))
import launch,stop
from mtp_experiment import atomic_json,existing_sequence
from ttft_study import TimingStudy
from overnight_metrics import can_start


def command_for(root,label,restore=False):
    if label not in ['fixed1','fixed3','interval8']:raise ValueError('Unknown frozen night configuration')
    if restore and label!='fixed3':raise ValueError('Only the original fixed3 service may be restored')
    steps=3 if label=='fixed3' else 1
    command,env=launch.build_launch(root,[0,1,2,3],'mtp',30000,speculative_steps=steps)
    if not restore:command+=['--random-seed','20261030','--log-level','warning']
    if label=='interval8':command+=['--prefill-decode-interval','8']
    return command,env


class NightStudy(TimingStudy):
    def __init__(self,root,run):
        self.root,self.run=Path(root).resolve(),Path(run).resolve()
        if self.run.parent!=self.root/'results/overnight-study-2026-10-30' or not self.run.name.startswith('run-'):
            raise ValueError('Night run must remain in the original project study directory')
        self.run.mkdir(parents=True,exist_ok=True);self.launch,self.stop=launch,stop
        self.active_path=self.root/'results/active-server.json';self.pyth=str(self.root/'.venv/bin/python')
        self.sequence=existing_sequence(self.run/'server-logs');self.label='fixed3';self.restoring=False

    def server_command(self,mode,steps,timing):
        assert mode=='mtp' and not timing and steps==(3 if self.label=='fixed3' else 1)
        return command_for(self.root,self.label,self.restoring)

    def select(self,label,tag,restore=False):
        self.label,self.restoring=label,restore
        record=self.switch('mtp',tag,steps=3 if label=='fixed3' else 1,timing=False)
        info=json.loads((self.run/(tag+'-server-info.json')).read_text())
        assert not info['enable_metrics'] and not info['enable_request_time_stats_logging']
        assert info['prefill_decode_interval']==(8 if label=='interval8' else 0)
        assert not info.get('log_requests',False)
        if not restore:assert info['log_level']=='warning' and info['random_seed']==20261030
        quality=[]
        for name,prompt,expected in [('arithmetic','只输出一个阿拉伯数字答案：17×19等于多少？','323'),('chinese','请用中文用一句话介绍你可以提供哪些帮助。',None)]:
            body={'model':'deepseek-v4-flash','messages':[{'role':'user','content':prompt}],
                'temperature':0,'max_tokens':128,'chat_template_kwargs':{'thinking':False}}
            request=urllib.request.Request('http://127.0.0.1:30000/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=90) as response:out=json.load(response)
            text=out['choices'][0]['message'].get('content','') or ''
            passed=text.strip().rstrip('。.')==expected if expected is not None else any('\u4e00'<=c<='\u9fff' for c in text)
            quality.append({'name':name,'text':text,'passed':passed,'response':out})
        atomic_json(self.run/(tag+'-quality.json'),{'passed':all(r['passed'] for r in quality),'cases':quality})
        if not all(r['passed'] for r in quality):raise RuntimeError('Night configuration correctness smoke failed')
        return record


def client_command(root,run,key,load):
    model=root/'models/DeepSeek-V4-Flash'
    return [str(root/'.venv/bin/python'),str(root/'scripts/overnight_worker.py'),
        '--backend','sglang','--host','127.0.0.1','--port','30000',
        '--model',str(model),'--tokenizer',str(model),'--served-model-name','deepseek-v4-flash',
        '--dataset-name','random','--random-input-len',str(load['input_len']),'--random-output-len',str(load['output_len']),
        '--num-prompts',str(len(load['offsets'])),'--request-rate',str(load['rate']),
        '--warmup-requests','8','--seed','31415','--temperature','0','--disable-tqdm',
        '--output-file',str(run/'cases'/(key+'.jsonl')),'--output-details','--goodput','ttft:500','tpot:30']


def run_case(root,run,label,phase,rnd,load):
    key=f'{phase}-{label}-r{rnd}-{load["name"]}'
    cfg={'key':key,'label':label,'phase':phase,'round':rnd,'load':load}
    path=run/'case-configs'/(key+'.json');path.parent.mkdir(exist_ok=True)
    with path.open('x') as file:json.dump(cfg,file,ensure_ascii=False,indent=2)
    env=os.environ.copy();env['INFRA_OVERNIGHT_CASE']=str(path)
    command=client_command(root,run,key,load);log=run/'client-logs'/(key+'.log');log.parent.mkdir(exist_ok=True)
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with log.open('x') as output:
        child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=load['window_s']+120)
        except BaseException:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
            raise
    atomic_json(run/'cases'/(key+'.execution.json'),{'command':command,'exit_code':code,'client_pid':child.pid,
        'started_at_utc':started,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'configuration_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if code!=0:raise RuntimeError('Native night client failed: '+key)
    validation=json.loads((run/'cases'/(key+'.validation.json')).read_text())
    return {'key':key,'phase':phase,'label':label,'round':rnd,'workload':load['name'],**validation}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True,type=Path);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];study=NightStudy(root,args.run_dir)
    lock=(study.run/'workflow.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((study.run/'smoke-accepted.json').read_text())['passed']
    protocol=json.loads((study.run/'protocol.json').read_text());deadline=protocol['deadline_epoch']
    with (study.run/'workflow-started.json').open('x') as file:
        json.dump({'pid':os.getpid(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_epoch':deadline},file)
    def interrupt(signum,frame):raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    telemetry_path=study.run/'gpu-telemetry.csv'
    telemetry_log=telemetry_path.open('x')
    telemetry=subprocess.Popen(['nvidia-smi','-i','0,1,2,3',
        '--query-gpu=timestamp,index,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm',
        '--format=csv,noheader','--loop=15'],stdout=telemetry_log,stderr=subprocess.STDOUT)
    outcome={'status':'running','completed_cases':0,'completed_planned_requests':0,'service_restored':False,
        'planned_cases':protocol['planned_cases'],'deadline_epoch':deadline}
    completed=[];phase_counts={};cutoff=False
    try:
        for phase in ['A','B']:
            names=protocol['phase_workloads'][phase]
            for rnd,order in enumerate(protocol['orders'],1):
                for label in order:
                    first=json.loads((study.run/'workloads'/(names[0]+'.json')).read_text())
                    if not can_start(deadline,time.time(),first['window_s']):cutoff=True;break
                    tag=f'night-{phase}-r{rnd}-{label}'
                    study.select(label,tag)
                    for name in names:
                        load=json.loads((study.run/'workloads'/(name+'.json')).read_text())
                        if not can_start(deadline,time.time(),load['window_s']):cutoff=True;break
                        study.guard(json.loads(study.active_path.read_text()))
                        study.event('case_begin',phase=phase,label=label,round=rnd,workload=name,
                                    count=len(load['offsets']),window_s=load['window_s'])
                        result=run_case(root,study.run,label,phase,rnd,load)
                        completed.append(result);phase_counts[phase]=phase_counts.get(phase,0)+1
                        outcome['completed_cases']+=1;outcome['completed_planned_requests']+=result['offered_requests']
                        outcome['current_phase']=phase;outcome['last_case']=result['key']
                        atomic_json(study.run/'latest-status.json',outcome)
                        atomic_json(study.run/'summary-so-far.json',{'cases':completed,'phase_case_counts':phase_counts})
                        study.event('case_end',phase=phase,label=label,round=rnd,workload=name,
                                    goodput=result['request_goodput'],attainment=result['attainment'])
                    if cutoff:break
                if cutoff:break
            if cutoff:break
            atomic_json(study.run/('phase-'+phase+'-complete.json'),{'status':'completed','cases':phase_counts[phase],
                'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        outcome['status']='completed' if outcome['completed_cases']==protocol['planned_cases'] else 'partial_cutoff'
    except BaseException as error:
        outcome.update(status='failed',error=repr(error));study.event('workflow_failure',error=repr(error))
    finally:
        try:
            record=study.select('fixed3','overnight-restored-stable-tp4-mtp',restore=True)
            outcome.update(service_restored=True,restored_pid=record['pid'])
        except BaseException as error:outcome['restoration_error']=repr(error)
        telemetry.terminate()
        try:telemetry.wait(timeout=10)
        except subprocess.TimeoutExpired:telemetry.kill();telemetry.wait(timeout=10)
        telemetry_log.close()
        outcome['completed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(study.run/'workflow-outcome.json',outcome);atomic_json(study.run/'latest-status.json',outcome)
        study.event('workflow_finished',**outcome)
    if outcome['status']=='failed' or not outcome['service_restored']:raise SystemExit(1)
    print('OVERNIGHT_QUEUE_FINISHED',flush=True)


if __name__=='__main__':main()
