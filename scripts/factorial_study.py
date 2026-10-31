#!/usr/bin/env python3
"""Finite 2x2 screening on the original four GPUs, with locks and exact restoration."""
import argparse,datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import launch,stop
from mtp_experiment import atomic_json,existing_sequence
from ttft_study import TimingStudy
from factorial_metrics import CONFIGS,validate_workload,make_schedule,can_start


def study_directory(root,path):
    root,path=Path(root).resolve(),Path(path).resolve()
    if path.parent!=root/'results/factorial-study-2026-10-31' or not path.name.startswith('run-'):
        raise ValueError('Factorial run must remain in its separate project directory')
    return path


def command_for(root,label,restore=False):
    if label not in CONFIGS:raise ValueError('Unknown factorial configuration')
    if restore and label!='s3i0':raise ValueError('Only original fixed3/interval0 may be restored')
    steps,interval=CONFIGS[label]
    command,env=launch.build_launch(root,[0,1,2,3],'mtp',30000,speculative_steps=steps)
    if not restore:command+=['--random-seed','20261030','--log-level','warning']
    if interval:command+=['--prefill-decode-interval',str(interval)]
    return command,env


class FactorialStudy(TimingStudy):
    def __init__(self,root,run):
        self.root=Path(root).resolve();self.run=study_directory(root,run)
        self.run.mkdir(parents=True,exist_ok=True)
        for folder in ['cases','case-configs','client-logs','server-logs']:(self.run/folder).mkdir(exist_ok=True)
        self.launch,self.stop=launch,stop
        self.active_path=self.root/'results/active-server.json';self.pyth=str(self.root/'.venv/bin/python')
        self.sequence=existing_sequence(self.run/'server-logs');self.label='s3i0';self.restoring=False

    def server_command(self,mode,steps,timing):
        assert mode=='mtp' and not timing and steps==CONFIGS[self.label][0]
        return command_for(self.root,self.label,self.restoring)

    def select(self,label,tag,restore=False):
        self.label,self.restoring=label,restore
        record=self.switch('mtp',tag,steps=CONFIGS[label][0],timing=False)
        info=json.loads((self.run/(tag+'-server-info.json')).read_text())
        assert not info['enable_metrics'] and not info['enable_request_time_stats_logging']
        assert not info.get('log_requests',False)
        assert info['tp_size']==4 and info['prefill_decode_interval']==CONFIGS[label][1]
        assert info['prefill_decode_interval_min_bs']==0
        assert info['speculative_num_steps']==CONFIGS[label][0]
        if not restore:assert info['log_level']=='warning' and info['random_seed']==20261030
        quality=[]
        for name,prompt,expected in [('arithmetic','只输出一个阿拉伯数字答案：17×19等于多少？','323'),('chinese','请用中文用一句话介绍你可以提供哪些帮助。',None)]:
            body={'model':'deepseek-v4-flash','messages':[{'role':'user','content':prompt}],
                'temperature':0,'max_tokens':128,'chat_template_kwargs':{'thinking':False}}
            req=urllib.request.Request('http://127.0.0.1:30000/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=90) as response:out=json.load(response)
            text=out['choices'][0]['message'].get('content','') or ''
            passed=text.strip().rstrip('。.')==expected if expected else any('\u4e00'<=c<='\u9fff' for c in text)
            quality.append({'name':name,'text':text,'passed':passed,'response':out})
        atomic_json(self.run/(tag+'-quality.json'),{'passed':all(q['passed'] for q in quality),'cases':quality})
        if not all(q['passed'] for q in quality):raise RuntimeError('Factorial configuration quality smoke failed')
        return record


def client_command(root,run,key,load):
    root,run=Path(root),Path(run)
    validate_workload(load['input_len'],load['output_len'],load['rate'],load['window_s'],load['pattern'])
    assert load['offsets']==make_schedule(load['pattern'],load['rate'],load['window_s'],0)
    model=root/'models/DeepSeek-V4-Flash'
    return [str(root/'.venv/bin/python'),str(root/'scripts/factorial_worker.py'),
        '--backend','sglang','--host','127.0.0.1','--port','30000',
        '--model',str(model),'--tokenizer',str(model),'--served-model-name','deepseek-v4-flash',
        '--dataset-name','random','--random-input-len',str(load['input_len']),'--random-output-len',str(load['output_len']),
        '--num-prompts',str(len(load['offsets'])),'--request-rate',str(load['rate']),
        '--warmup-requests','8','--seed','31415','--temperature','0','--disable-tqdm',
        '--output-file',str(run/'cases'/(key+'.jsonl')),'--output-details','--goodput','ttft:500','tpot:30']


def run_case(root,run,label,load,phase='F'):
    key=f'{phase}-{label}-r1-{load["name"]}'
    cfg={'key':key,'label':label,'phase':phase,'round':1,'load':load}
    path=run/'case-configs'/(key+'.json');path.parent.mkdir(exist_ok=True)
    with path.open('x') as f:json.dump(cfg,f,ensure_ascii=False,indent=2)
    env=os.environ.copy();env['INFRA_FACTORIAL_CASE']=str(path)
    command=client_command(root,run,key,load)
    log=run/'client-logs'/(key+'.log');log.parent.mkdir(exist_ok=True)
    began=datetime.datetime.now(datetime.timezone.utc).isoformat();code=None;error=None
    try:
        with log.open('x') as output:
            child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=load['window_s']+120)
            except BaseException:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
                raise
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        atomic_json(run/'cases'/(key+'.execution.json'),{'command':command,'exit_code':code,'error':error,
            'started_at_utc':began,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'configuration_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if code!=0:raise RuntimeError('Native factorial client failed: '+key)
    validation=json.loads((run/'cases'/(key+'.validation.json')).read_text())
    return {'key':key,'label':label,'round':1,'workload':load['name'],**validation}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True,type=Path);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];study=FactorialStudy(root,args.run_dir)
    locks=[]
    for path in [study.run.parent/'workflow.lock',study.run/'workflow.lock']:
        f=path.open('a+');fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    protocol=json.loads((study.run/'protocol.json').read_text())
    assert protocol['configuration_order']==['s3i0','s1i0','s1i8','s3i8']
    assert protocol['planned_cases']==8 and protocol['planned_formal_requests']==5280
    assert protocol['max_inflight']==32 and protocol['research_slos_ms']=={'ttft':500,'tpot':30}
    assert len(protocol['workloads'])==2 and protocol['hard_deadline_epoch']>time.time()
    for p,h in protocol['code_sha256'].items():
        assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,'Code changed: '+p
    original=json.loads(study.active_path.read_text());study.guard(original)
    original_command,_=command_for(root,'s3i0',restore=True)
    assert original['command']==original_command,'Initial service is not the original stable command'
    atomic_json(study.run/'original-server.json',original)
    with (study.run/'workflow-started.json').open('x') as f:
        json.dump({'pid':os.getpid(),'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
    def interrupt(signum,frame):raise TimeoutError(f'Factorial bounded stop: signal {signum}')
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(sig,interrupt)
    signal.alarm(max(1,int(protocol['hard_deadline_epoch']-time.time()-900)))
    outcome={'status':'running','completed_cases':0,'completed_planned_requests':0,'planned_cases':8,'service_restored':False}
    completed=[]
    try:
        smoke=json.loads((study.run/'workloads/smoke.json').read_text())
        result=run_case(root,study.run,'s3i0',smoke,phase='smoke')
        atomic_json(study.run/'smoke-accepted.json',{'passed':True,'result':result})
        for label in protocol['configuration_order']:
            if protocol['hard_deadline_epoch']-time.time()<1860:
                outcome['status']='partial_cutoff';break
            study.select(label,'factorial-'+label)
            for name in protocol['workloads']:
                load=json.loads((study.run/'workloads'/(name+'.json')).read_text())
                if not can_start(protocol['hard_deadline_epoch'],time.time(),load['window_s']):
                    outcome['status']='partial_cutoff';break
                study.guard(json.loads(study.active_path.read_text()))
                study.event('case_begin',label=label,workload=name,count=len(load['offsets']))
                outcome.update(current_configuration=label,current_workload=name)
                atomic_json(study.run/'latest-status.json',outcome)
                result=run_case(root,study.run,label,load)
                completed.append(result);outcome['completed_cases']+=1
                outcome['completed_planned_requests']+=result['offered_requests']
                atomic_json(study.run/'summary-so-far.json',{'cases':completed})
                atomic_json(study.run/'latest-status.json',outcome)
                study.event('case_end',**result)
            if outcome['status']=='partial_cutoff':break
        else:outcome['status']='completed'
    except BaseException as exc:
        outcome.update(status='failed',error=repr(exc));study.event('workflow_failure',error=repr(exc))
    finally:
        signal.alarm(0)
        try:
            record=study.select('s3i0','factorial-restored-stable-tp4-mtp',restore=True)
            actual=Path('/proc',str(record['pid']),'cmdline').read_bytes().split(b'\0')[:-1]
            assert [x.decode() for x in actual]==original_command
            study.guard(record)
            with urllib.request.urlopen('http://127.0.0.1:30000/v1/models',timeout=15) as reply:models=json.load(reply)
            assert any(x['id']=='deepseek-v4-flash' for x in models['data'])
            with (study.run/'tests-remote-final.log').open('x') as log:
                result=subprocess.run([study.pyth,'-m','unittest','discover','-s','tests','-v'],cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            assert result.returncode==0,'Final remote tests failed'
            outcome.update(service_restored=True,restored_pid=record['pid'],models_verified=True,final_tests_exit_code=0)
        except BaseException as exc:outcome['restoration_error']=repr(exc)
        outcome['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(study.run/'workflow-outcome.json',outcome);atomic_json(study.run/'latest-status.json',outcome)
        study.event('workflow_finished',**outcome)
    if outcome['status']!='completed' or not outcome['service_restored']:raise SystemExit(1)

if __name__=='__main__':main()
