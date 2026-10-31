#!/usr/bin/env python3
"""Opt-in finite wait-budget prototype evaluation on the original four GPUs."""
import argparse,datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import launch,stop
from mtp_experiment import atomic_json,existing_sequence
from ttft_study import TimingStudy
from factorial_metrics import validate_workload,make_schedule,can_start
LABELS=['stable3','interval8','sync8','budget100']
SOURCE_SHA256='cb8f6de62ad1e200df34114cbcffec747e16d667eddbc719ccb9052723e8b4e5'


def study_directory(root,path):
    root,path=Path(root).resolve(),Path(path).resolve()
    if path.parent!=root/'results/wait-budget-study-2026-10-31' or not path.name.startswith('run-'):
        raise ValueError('Wait-budget run must remain in its separate project directory')
    return path


def clean_environment(root,env):
    result={k:v for k,v in env.items() if not k.startswith('INFRA_WAIT_BUDGET_')}
    if 'PYTHONPATH' in result:
        parent=Path(root).resolve()/'results/wait-budget-study-2026-10-31'
        kept=[p for p in result['PYTHONPATH'].split(os.pathsep) if p and not Path(p).resolve().is_relative_to(parent)]
        if kept:result['PYTHONPATH']=os.pathsep.join(kept)
        else:result.pop('PYTHONPATH')
    return result


def command_for(root,run,label,restore=False):
    root=Path(root).resolve();run=study_directory(root,run)
    if label not in LABELS:raise ValueError('Unknown wait-budget configuration')
    if restore and label!='stable3':raise ValueError('Only original stable3 may be restored')
    steps=3 if label=='stable3' else 1
    command,env=launch.build_launch(root,[0,1,2,3],'mtp',30000,speculative_steps=steps)
    env=clean_environment(root,env)
    if not restore:command+=['--random-seed','20261030','--log-level','warning']
    if label!='stable3':command+=['--prefill-decode-interval','8']
    if label in ['sync8','budget100']:
        env['INFRA_WAIT_BUDGET_CONFIG']=str(run/'runtime-configs'/(label+'.json'))
        env['PYTHONPATH']=os.pathsep.join([str(run/'runtime'),str(root/'scripts')]+([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    return command,env


class BudgetStudy(TimingStudy):
    def __init__(self,root,run):
        self.root=Path(root).resolve();self.run=study_directory(root,run)
        self.run.mkdir(parents=True,exist_ok=True)
        for folder in ['cases','case-configs','client-logs','server-logs','patch-receipts']:(self.run/folder).mkdir(exist_ok=True)
        self.launch,self.stop=launch,stop;self.active_path=self.root/'results/active-server.json'
        self.pyth=str(self.root/'.venv/bin/python');self.sequence=existing_sequence(self.run/'server-logs')
        self.label='stable3';self.restoring=False
    def server_command(self,mode,steps,timing):
        assert mode=='mtp' and not timing and steps==(3 if self.label=='stable3' else 1)
        return command_for(self.root,self.run,self.label,self.restoring)
    def select(self,label,tag,restore=False):
        self.label,self.restoring=label,restore
        record=self.switch('mtp',tag,steps=3 if label=='stable3' else 1,timing=False)
        info=json.loads((self.run/(tag+'-server-info.json')).read_text())
        assert info['tp_size']==4 and info['dp_size']==1 and info['pp_size']==1 and not info['enable_dp_attention']
        assert not info['enable_metrics'] and not info['enable_request_time_stats_logging'] and not info.get('log_requests',False)
        assert info['prefill_decode_interval']==(0 if label=='stable3' else 8)
        assert info['prefill_decode_interval_min_bs']==0
        if not restore:assert info['random_seed']==20261030 and info['log_level']=='warning'
        if label in ['sync8','budget100']:
            folder=self.run/'patch-receipts'/label;deadline=time.monotonic()+20
            while time.monotonic()<deadline and len(list(folder.glob('tp*.json')))<4:time.sleep(.1)
            records=[json.loads(p.read_text()) for p in folder.glob('tp*.json')]
            assert len(records)==4 and {r['tp_rank'] for r in records}=={0,1,2,3}
            assert {r['pid'] for r in records}=={p for p,_ in self.gpu_processes()}
            want=100 if label=='budget100' else None
            policy_hash=hashlib.sha256((self.root/'scripts/wait_budget_policy.py').read_bytes()).hexdigest()
            assert all(r['budget_ms']==want and r['installed_scheduler_sha256']==SOURCE_SHA256 and r['policy_sha256']==policy_hash for r in records)
            atomic_json(self.run/(tag+'-patch-verification.json'),{'passed':True,'records':records})
        else:
            env=Path('/proc',str(record['pid']),'environ').read_bytes().split(b'\0')
            assert not any(x.startswith(b'INFRA_WAIT_BUDGET_') for x in env)
        quality=[]
        for name,prompt,wanted in [('arithmetic','只输出一个阿拉伯数字答案：17×19等于多少？','323'),('chinese','请用中文用一句话介绍你可以提供哪些帮助。',None)]:
            body={'model':'deepseek-v4-flash','messages':[{'role':'user','content':prompt}],
                'temperature':0,'max_tokens':128,'chat_template_kwargs':{'thinking':False}}
            req=urllib.request.Request('http://127.0.0.1:30000/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=90) as response:out=json.load(response)
            text=out['choices'][0]['message'].get('content','') or ''
            passed=text.strip().rstrip('。.')==wanted if wanted else any('\u4e00'<=c<='\u9fff' for c in text)
            quality.append({'name':name,'passed':passed,'text':text,'response':out})
        atomic_json(self.run/(tag+'-quality.json'),{'passed':all(r['passed'] for r in quality),'cases':quality})
        if not all(r['passed'] for r in quality):raise RuntimeError('Wait-budget configuration quality smoke failed')
        return record


def client_command(root,run,key,load):
    root,run=Path(root),Path(run)
    validate_workload(load['input_len'],load['output_len'],load['rate'],load['window_s'],load['pattern'])
    assert load['offsets']==make_schedule(load['pattern'],load['rate'],load['window_s'],0)
    model=root/'models/DeepSeek-V4-Flash'
    return [str(root/'.venv/bin/python'),str(root/'scripts/wait_budget_worker.py'),
        '--backend','sglang','--host','127.0.0.1','--port','30000',
        '--model',str(model),'--tokenizer',str(model),'--served-model-name','deepseek-v4-flash',
        '--dataset-name','random','--random-input-len',str(load['input_len']),'--random-output-len',str(load['output_len']),
        '--num-prompts',str(len(load['offsets'])),'--request-rate',str(load['rate']),
        '--warmup-requests','8','--seed','31415','--temperature','0','--disable-tqdm',
        '--output-file',str(run/'cases'/(key+'.jsonl')),'--output-details','--goodput','ttft:500','tpot:30']


def run_case(root,run,label,load,phase='W'):
    key=f'{phase}-{label}-r1-{load["name"]}'
    cfg={'key':key,'label':label,'phase':phase,'round':1,'load':load}
    path=run/'case-configs'/(key+'.json');path.parent.mkdir(exist_ok=True)
    with path.open('x') as f:json.dump(cfg,f,ensure_ascii=False,indent=2)
    env=clean_environment(root,os.environ.copy());env['INFRA_WAIT_BUDGET_CASE']=str(path)
    command=client_command(root,run,key,load);log=run/'client-logs'/(key+'.log');log.parent.mkdir(exist_ok=True)
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
    except BaseException as exc:error=repr(exc);raise
    finally:
        atomic_json(run/'cases'/(key+'.execution.json'),{'command':command,'exit_code':code,'error':error,
            'started_at_utc':began,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'configuration_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if code!=0:raise RuntimeError('Native wait-budget client failed: '+key)
    validation=json.loads((run/'cases'/(key+'.validation.json')).read_text())
    return {'key':key,'label':label,'round':1,'workload':load['name'],**validation}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True,type=Path);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];study=BudgetStudy(root,args.run_dir)
    locks=[]
    for path in [study.run.parent/'workflow.lock',study.run/'workflow.lock']:
        f=path.open('a+');fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    protocol=json.loads((study.run/'protocol.json').read_text())
    assert protocol['configuration_order']==LABELS and protocol['budget_ms']==100
    assert protocol['planned_cases']==8 and protocol['planned_formal_requests']==5280
    assert protocol['max_inflight']==32 and protocol['research_slos_ms']=={'ttft':500,'tpot':30}
    assert json.loads((study.run/'cpu-consensus.json').read_text())['passed']
    source=root/'.venv/lib/python3.12/site-packages/sglang/srt/managers/scheduler.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==SOURCE_SHA256
    for name,digest in protocol['code_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    original=json.loads(study.active_path.read_text());study.guard(original)
    expected,_=command_for(root,study.run,'stable3',restore=True);assert original['command']==expected
    atomic_json(study.run/'original-server.json',original)
    with (study.run/'workflow-started.json').open('x') as f:json.dump({'pid':os.getpid(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
    def interrupt(signum,frame):raise TimeoutError(f'Wait-budget bounded stop: signal {signum}')
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(sig,interrupt)
    signal.alarm(max(1,int(protocol['hard_deadline_epoch']-time.time()-900)))
    outcome={'status':'running','completed_cases':0,'completed_planned_requests':0,'planned_cases':8,'service_restored':False}
    completed=[]
    try:
        load=json.loads((study.run/'workloads/smoke.json').read_text())
        smoke=run_case(root,study.run,'stable3',load,phase='smoke')
        atomic_json(study.run/'smoke-accepted.json',{'passed':True,'result':smoke})
        for label in LABELS:
            if protocol['hard_deadline_epoch']-time.time()<1860:outcome['status']='partial_cutoff';break
            study.select(label,'wait-budget-'+label)
            for name in protocol['workloads']:
                load=json.loads((study.run/'workloads'/(name+'.json')).read_text())
                if not can_start(protocol['hard_deadline_epoch'],time.time(),load['window_s']):outcome['status']='partial_cutoff';break
                study.guard(json.loads(study.active_path.read_text()))
                outcome.update(current_configuration=label,current_workload=name)
                atomic_json(study.run/'latest-status.json',outcome);study.event('case_begin',label=label,workload=name,count=len(load['offsets']))
                result=run_case(root,study.run,label,load);completed.append(result)
                outcome['completed_cases']+=1;outcome['completed_planned_requests']+=result['offered_requests']
                atomic_json(study.run/'summary-so-far.json',{'cases':completed});atomic_json(study.run/'latest-status.json',outcome)
                study.event('case_end',**result)
            if outcome['status']=='partial_cutoff':break
        else:outcome['status']='completed'
    except BaseException as exc:
        outcome.update(status='failed',error=repr(exc));study.event('workflow_failure',error=repr(exc))
    finally:
        signal.alarm(0)
        try:
            record=study.select('stable3','wait-budget-restored-stable-tp4-mtp',restore=True)
            actual=[x.decode() for x in Path('/proc',str(record['pid']),'cmdline').read_bytes().split(b'\0') if x]
            assert actual==expected;study.guard(record)
            assert hashlib.sha256(source.read_bytes()).hexdigest()==SOURCE_SHA256
            with urllib.request.urlopen('http://127.0.0.1:30000/v1/models',timeout=15) as r:models=json.load(r)
            assert any(m['id']=='deepseek-v4-flash' for m in models['data'])
            with (study.run/'tests-remote-final.log').open('x') as log:
                r=subprocess.run([study.pyth,'-m','unittest','discover','-s','tests','-v'],cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            assert r.returncode==0
            outcome.update(service_restored=True,restored_pid=record['pid'],installed_source_unchanged=True,models_verified=True,final_tests_exit_code=0)
        except BaseException as exc:outcome['restoration_error']=repr(exc)
        outcome['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(study.run/'workflow-outcome.json',outcome);atomic_json(study.run/'latest-status.json',outcome);study.event('workflow_finished',**outcome)
    if outcome['status']!='completed' or not outcome['service_restored']:raise SystemExit(1)

if __name__=='__main__':main()
