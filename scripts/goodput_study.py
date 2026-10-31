#!/usr/bin/env python3
"""Finite, scoped model-serving goodput comparisons on the original project GPUs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import launch,stop
from mtp_experiment import atomic_json,existing_sequence
from ttft_study import TimingStudy


class GoodputStudy(TimingStudy):
    def __init__(self,root,run):
        self.root,self.run=Path(root).resolve(),Path(run).resolve()
        if self.run.parent!=self.root/'results/goodput-study-2026-10-30' or not self.run.name.startswith('run-'):
            raise ValueError('Goodput study must remain inside the dedicated project run directory')
        self.run.mkdir(parents=True,exist_ok=True)
        self.launch,self.stop=launch,stop
        self.active_path=self.root/'results/active-server.json'
        self.pyth=str(self.root/'.venv/bin/python')
        self.sequence=existing_sequence(self.run/'server-logs')


def client_command(root,run,key,rate,count):
    root,run=Path(root),Path(run)
    if rate not in (2,4,7,9) or not 1<=count<=270:
        raise ValueError('Only the frozen smoke and formal study parameters are allowed')
    model=root/'models/DeepSeek-V4-Flash'
    return [str(root/'.venv/bin/python'),str(root/'scripts/goodput_worker.py'),
        '--backend','sglang','--host','127.0.0.1','--port','30000',
        '--model',str(model),'--tokenizer',str(model),'--served-model-name','deepseek-v4-flash',
        '--dataset-name','random','--random-input-len','512','--random-output-len','128',
        '--num-prompts',str(count),'--request-rate',str(rate),'--warmup-requests','8',
        '--seed','31415','--temperature','0','--disable-tqdm',
        '--output-file',str(run/'cases'/(key+'.jsonl')),'--output-details',
        '--goodput','ttft:500','tpot:30']


def run_case(root,run,label,round_id,rate,count=None):
    count=count if count is not None else rate*30
    key=f'{label}-r{round_id}-q{rate}'
    records=json.loads((run/'input-manifest.json').read_text())['requests'][:count]
    config={'key':key,'label':label,'round':round_id,'rate':rate,'request_count':count,
            'output_dir':str(run/'cases'),'requests':records,'arrival_protocol':'deterministic-periodic',
            'thresholds_ms':{'ttft':500,'tpot':30},'inflight_limit':32}
    configs=run/'case-configs';configs.mkdir(exist_ok=True)
    config_path=configs/(key+'.json');
    with config_path.open('x') as file:json.dump(config,file,ensure_ascii=False,indent=2)
    env=os.environ.copy();env['INFRA_GOODPUT_CASE']=str(config_path)
    command=client_command(root,run,key,rate,count)
    log=run/'client-logs'/(key+'.log');log.parent.mkdir(exist_ok=True)
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with log.open('x') as output:
        result=subprocess.run(command,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=180)
    record={'command':command,'exit_code':result.returncode,'started_at_utc':started,
            'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest()}
    atomic_json(run/'cases'/(key+'.execution.json'),record)
    result.check_returncode()
    return json.loads((run/'cases'/(key+'.validation.json')).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];study=GoodputStudy(root,args.run_dir)
    if not json.loads((study.run/'smoke-accepted.json').read_text())['passed']:
        raise ValueError('A separately verified native-client smoke is required')
    with (study.run/'workflow-started.json').open('x') as file:
        json.dump({'pid':os.getpid(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},file)
    def interrupt(signum,frame):raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    outcome={'status':'running','formal_cases':0,'formal_requests':0,'service_restored':False}
    try:
        for rnd,order in enumerate([['none','fixed1','fixed3'],['fixed1','fixed3','none'],['fixed3','none','fixed1']],1):
            for label in order:
                mode='none' if label=='none' else 'mtp';steps=1 if label=='fixed1' else 3
                study.switch(mode,f'goodput-r{rnd}-{label}',steps=steps,timing=True)
                rates=[4,7,9] if rnd!=2 else [9,7,4]
                for rate in rates:
                    study.event('case_begin',label=label,round=rnd,rate=rate)
                    result=run_case(root,study.run,label,rnd,rate)
                    outcome['formal_cases']+=1;outcome['formal_requests']+=rate*30
                    atomic_json(study.run/'progress.json',outcome)
                    study.event('case_end',label=label,round=rnd,rate=rate,
                        goodput=result['request_goodput'],attainment=result['attainment'],
                        errors_or_short_outputs=result['errors_or_short_outputs'])
        outcome['status']='completed'
    except BaseException as error:
        outcome.update(status='failed',error=repr(error));study.event('workflow_failure',error=repr(error));raise
    finally:
        try:
            record=study.switch('mtp','goodput-restored-stable-tp4-mtp',steps=3,timing=False)
            outcome.update(service_restored=True,restored_pid=record['pid'])
        except BaseException as error:
            outcome['restoration_error']=repr(error)
        outcome['completed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(study.run/'workflow-outcome.json',outcome)
        study.event('workflow_finished',**outcome)
    print('GOODPUT_STUDY_COMPLETE',flush=True)


if __name__=='__main__':main()
