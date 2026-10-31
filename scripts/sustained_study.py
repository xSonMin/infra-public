#!/usr/bin/env python3
"""Single-variable scheduling comparisons on the original four project GPUs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parent))
import launch,stop
from mtp_experiment import atomic_json,existing_sequence
from ttft_study import TimingStudy,diagnostic_command
from sustained_metrics import sustained_offsets


def command_for(root,label,timing=True):
    if label not in ['fixed1','fixed3','interval8']:raise ValueError('Unknown frozen scheduling configuration')
    steps=3 if label=='fixed3' else 1
    command,env=diagnostic_command(root,'mtp',steps,timing)
    if label=='interval8':
        if not timing:raise ValueError('Candidate is diagnostic-only; restoration must use fixed3')
        command+=['--prefill-decode-interval','8']
    return command,env


class SustainedStudy(TimingStudy):
    def __init__(self,root,run):
        self.root,self.run=Path(root).resolve(),Path(run).resolve()
        if self.run.parent!=self.root/'results/sustained-study-2026-10-30' or not self.run.name.startswith('run-'):
            raise ValueError('Sustained study must remain in the dedicated run directory')
        self.run.mkdir(parents=True,exist_ok=True)
        self.launch,self.stop=launch,stop;self.active_path=self.root/'results/active-server.json'
        self.pyth=str(self.root/'.venv/bin/python');self.sequence=existing_sequence(self.run/'server-logs')
        self.label='fixed3'

    def server_command(self,mode,steps,timing):
        assert mode=='mtp' and steps==(3 if self.label=='fixed3' else 1)
        return command_for(self.root,self.label,timing)

    def select(self,label,tag,timing=True):
        self.label=label;steps=3 if label=='fixed3' else 1
        record=self.switch('mtp',tag,steps=steps,timing=timing)
        info=json.loads((self.run/(tag+'-server-info.json')).read_text())
        assert info['prefill_decode_interval']==(8 if label=='interval8' else 0)
        assert info['prefill_decode_interval_min_bs']==0
        quality=[]
        for name,prompt,wanted in [('arithmetic','只输出一个阿拉伯数字答案：17×19等于多少？','323'),('chinese','请用中文用一句话介绍你可以提供哪些帮助。',None)]:
            body={'model':'deepseek-v4-flash','messages':[{'role':'user','content':prompt}],
                'temperature':0,'max_tokens':128,'chat_template_kwargs':{'thinking':False}}
            req=urllib.request.Request('http://127.0.0.1:30000/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=90) as response:out=json.load(response)
            text=out['choices'][0]['message'].get('content','') or ''
            passed=text.strip().rstrip('。.')==wanted if wanted is not None else any('\u4e00'<=c<='\u9fff' for c in text)
            quality.append({'name':name,'text':text,'passed':passed,'response':out})
        atomic_json(self.run/(tag+'-quality.json'),{'passed':all(q['passed'] for q in quality),'cases':quality})
        if not all(q['passed'] for q in quality):raise RuntimeError('Configuration correctness smoke failed')
        return record


def client_command(root,run,key,rate,count):
    sustained_offsets(rate,count)
    root,run=Path(root),Path(run);model=root/'models/DeepSeek-V4-Flash'
    return [str(root/'.venv/bin/python'),str(root/'scripts/sustained_worker.py'),
        '--backend','sglang','--host','127.0.0.1','--port','30000',
        '--model',str(model),'--tokenizer',str(model),'--served-model-name','deepseek-v4-flash',
        '--dataset-name','random','--random-input-len','512','--random-output-len','128',
        '--num-prompts',str(count),'--request-rate',str(rate),'--warmup-requests','8',
        '--seed','31415','--temperature','0','--disable-tqdm',
        '--output-file',str(run/'cases'/(key+'.jsonl')),'--output-details',
        '--goodput','ttft:500','tpot:30']


def run_case(root,run,label,rnd,rate=7,count=840):
    offsets=sustained_offsets(rate,count);key=f'{label}-r{rnd}-q{rate}'
    records=json.loads((run/'input-manifest.json').read_text())['requests'][:count]
    config={'key':key,'label':label,'round':rnd,'rate':rate,'request_count':count,
        'output_dir':str(run/'cases'),'requests':records,'arrival_protocol':'deterministic-periodic',
        'thresholds_ms':{'ttft':500,'tpot':30},'inflight_limit':32,'profile':'sustained'}
    path=run/'case-configs'/(key+'.json');path.parent.mkdir(exist_ok=True)
    with path.open('x') as file:json.dump(config,file,ensure_ascii=False,indent=2)
    env=os.environ.copy();env['INFRA_GOODPUT_CASE']=str(path)
    command=client_command(root,run,key,rate,count);log=run/'client-logs'/(key+'.log');log.parent.mkdir(exist_ok=True)
    began=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with log.open('x') as output:
        p=subprocess.run(command,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=300)
    atomic_json(run/'cases'/(key+'.execution.json'),{'command':command,'exit_code':p.returncode,
        'started_at_utc':began,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    p.check_returncode();return json.loads((run/'cases'/(key+'.validation.json')).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];study=SustainedStudy(root,args.run_dir)
    assert json.loads((study.run/'smoke-accepted.json').read_text())['passed']
    with (study.run/'workflow-started.json').open('x') as file:json.dump({'pid':os.getpid()},file)
    def interrupt(signum,frame):raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    outcome={'status':'running','formal_cases':0,'formal_requests':0,'service_restored':False}
    orders=[['fixed1','fixed3','interval8'],['fixed3','interval8','fixed1'],['interval8','fixed1','fixed3']]
    try:
        for rnd,order in enumerate(orders,1):
            for label in order:
                study.select(label,f'sustained-r{rnd}-{label}')
                study.event('case_begin',label=label,round=rnd)
                result=run_case(root,study.run,label,rnd)
                outcome['formal_cases']+=1;outcome['formal_requests']+=840
                atomic_json(study.run/'progress.json',outcome)
                study.event('case_end',label=label,round=rnd,goodput=result['request_goodput'],attainment=result['attainment'])
        outcome['status']='completed'
    except BaseException as error:
        outcome.update(status='failed',error=repr(error));study.event('workflow_failure',error=repr(error));raise
    finally:
        try:
            record=study.select('fixed3','sustained-restored-stable-tp4-mtp',timing=False)
            outcome.update(service_restored=True,restored_pid=record['pid'])
        except BaseException as error:outcome['restoration_error']=repr(error)
        outcome['completed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(study.run/'workflow-outcome.json',outcome);study.event('workflow_finished',**outcome)
    print('SUSTAINED_STUDY_COMPLETE',flush=True)


if __name__=='__main__':main()
