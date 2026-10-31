#!/usr/bin/env python3
"""Bounded2048/1024 prefill chunk comparison with fixed100ms waiting budget."""
import argparse,datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import launch,stop
from mtp_experiment import atomic_json,existing_sequence
from ttft_study import TimingStudy
from timeline_metrics import validate_load as validate_workload,make_schedule,profile_payload,trigger_due,validate_trace_size,summarize_trace,remaining_case_seconds
from factorial_metrics import can_start
import gzip,re
from chunk_metrics import completed_profile_files,validate_chunk_trace
LABELS=['chunk2048','chunk1024']
SOURCE_SHA256='cb8f6de62ad1e200df34114cbcffec747e16d667eddbc719ccb9052723e8b4e5'


def study_directory(root,path):
    root,path=Path(root).resolve(),Path(path).resolve()
    if path.parent!=root/'results/chunk-study-2026-10-31' or not path.name.startswith('run-'):
        raise ValueError('Chunk run must remain in its separate project directory')
    return path


def clean_environment(root,env):
    result={k:v for k,v in env.items() if not k.startswith(('INFRA_WAIT_BUDGET_','INFRA_TIMELINE_','INFRA_CHUNK_'))}
    if 'PYTHONPATH' in result:
        parents=[Path(root).resolve()/'results'/name for name in ['wait-budget-study-2026-10-31','timeline-study-2026-10-31','chunk-study-2026-10-31']]
        kept=[p for p in result['PYTHONPATH'].split(os.pathsep) if p and not any(Path(p).resolve().is_relative_to(parent) for parent in parents)]
        if kept:result['PYTHONPATH']=os.pathsep.join(kept)
        else:result.pop('PYTHONPATH')
    return result


def command_for(root,run,label,restore=False):
    root=Path(root).resolve();run=study_directory(root,run)
    if label not in LABELS+['stable3']:raise ValueError('Unknown chunk configuration')
    if restore and label!='stable3':raise ValueError('Only original stable3 may be restored')
    steps=3 if label=='stable3' else 1
    command,env=launch.build_launch(root,[0,1,2,3],'mtp',30000,speculative_steps=steps)
    env=clean_environment(root,env)
    if not restore:command+=['--random-seed','20261030','--log-level','warning']
    if label in LABELS:
        command+=['--prefill-decode-interval','8']
        command[command.index('--chunked-prefill-size')+1]='1024' if label=='chunk1024' else '2048'
    if label in LABELS:
        env['INFRA_WAIT_BUDGET_CONFIG']=str(run/'runtime-configs'/(label+'.json'))
        env['PYTHONPATH']=os.pathsep.join([str(run/'runtime'),str(root/'scripts')]+([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    return command,env


class ChunkStudy(TimingStudy):
    def __init__(self,root,run):
        self.root=Path(root).resolve();self.run=study_directory(root,run)
        self.run.mkdir(parents=True,exist_ok=True)
        for folder in ['cases','case-configs','client-logs','server-logs','patch-receipts','profiles']:(self.run/folder).mkdir(exist_ok=True)
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
        assert info['chunked_prefill_size']==(1024 if label=='chunk1024' else 2048)
        assert info['page_size']==256
        if not restore:assert info['random_seed']==20261030 and info['log_level']=='warning'
        if label in LABELS:
            folder=self.run/'patch-receipts'/label;deadline=time.monotonic()+20
            while time.monotonic()<deadline and len(list(folder.glob('tp*.json')))<4:time.sleep(.1)
            records=[json.loads(p.read_text()) for p in folder.glob('tp*.json')]
            assert len(records)==4 and {r['tp_rank'] for r in records}=={0,1,2,3}
            assert {r['pid'] for r in records}=={p for p,_ in self.gpu_processes()}
            want=100
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
        if not all(r['passed'] for r in quality):raise RuntimeError('Chunk configuration quality smoke failed')
        if label in LABELS:
            self.check_long_outputs(label)
        return record

    def check_long_outputs(self,label):
        corpus=json.loads((self.run/'inputs/input2048.json').read_text())
        probes=[]
        for index in range(4):
            tokens=corpus[index]['input_ids']
            assert len(tokens)==2048
            body={'input_ids':tokens,'stream':False,'sampling_params':{'temperature':0,'max_new_tokens':32,'ignore_eos':True}}
            req=urllib.request.Request('http://127.0.0.1:30000/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=45) as response:reply=json.load(response)
            meta=reply['meta_info'];text=reply.get('text') or ''
            assert meta['prompt_tokens']==2048 and meta['completion_tokens']==32 and meta['num_retractions']==0 and text
            probes.append({'index':index,'input_ids_sha256':hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),
                'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'response':reply})
        record={'passed':True,'input_tokens':2048,'output_tokens':32,'count':4,'probes':probes}
        atomic_json(self.run/('chunk-'+label+'-long-output.json'),record)
        if label=='chunk1024':
            reference=json.loads((self.run/'chunk-chunk2048-long-output.json').read_text())
            matched=all(a['input_ids_sha256']==b['input_ids_sha256'] and a['text_sha256']==b['text_sha256'] for a,b in zip(reference['probes'],probes))
            atomic_json(self.run/'long-output-comparison.json',{'passed':matched,'probe_count':4,'criterion':'identical deterministic generated text on frozen long-input probes'})
            assert matched,'Long-input deterministic output differs; preserve evidence and stop before candidate formal queue'


def client_command(root,run,key,load):
    root,run=Path(root),Path(run)
    validate_workload(load['input_len'],load['output_len'],load['rate'],load['window_s'],load['pattern'])
    assert load['offsets']==make_schedule(load['pattern'],load['rate'],load['window_s'],0)
    model=root/'models/DeepSeek-V4-Flash'
    return [str(root/'.venv/bin/python'),str(root/'scripts/chunk_worker.py'),
        '--backend','sglang','--host','127.0.0.1','--port','30000',
        '--model',str(model),'--tokenizer',str(model),'--served-model-name','deepseek-v4-flash',
        '--dataset-name','random','--random-input-len',str(load['input_len']),'--random-output-len',str(load['output_len']),
        '--num-prompts',str(len(load['offsets'])),'--request-rate',str(load['rate']),
        '--warmup-requests','8','--seed','31415','--temperature','0','--disable-tqdm',
        '--output-file',str(run/'cases'/(key+'.jsonl')),'--output-details','--goodput','ttft:500','tpot:30']


def profile_request(endpoint,payload=None,allow_inactive=False,timeout=180):
    assert endpoint in ['start_profile','stop_profile'] and 0<timeout<=180
    req=urllib.request.Request('http://127.0.0.1:30000/'+endpoint,
        data=json.dumps(payload or {}).encode(),headers={'Content-Type':'application/json'},method='POST')
    began=time.time()
    with urllib.request.urlopen(req,timeout=timeout) as response:
        text=response.read().decode();code=response.status
    body=json.loads(text) if text.strip().startswith('{') else None
    if code!=200:raise RuntimeError('Profiler endpoint failed')
    if isinstance(body,dict) and body.get('success') is False:
        if not (allow_inactive and 'not in progress' in str(body).lower()):raise RuntimeError('Profiler returned failure: '+text[:500])
    return {'endpoint':endpoint,'status':code,'body':text,'request_started_realtime':began,'request_finished_realtime':time.time()}


def validate_profiles(run,key):
    folder=run/'profiles'/key;results=[]
    paths=sorted(folder.glob('*.trace.json*'))
    sizes=[p.stat().st_size for p in (run/'profiles').glob('**/*.trace.json*')]
    validate_trace_size(sizes)
    assert sum(sizes)<=512*1024**2,'Compressed traces exceed frozen512MiB limit'
    for p in paths:
        with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:raw=json.load(f)
        events=raw['traceEvents'] if isinstance(raw,dict) else raw
        row=summarize_trace(events)
        validate_chunk_trace(row,1024 if '-chunk1024-' in key else 2048)
        match=re.search(r'TP[-_](\d+)',p.name)
        if not match:raise ValueError('Cannot identify TP rank in profiler file')
        row.update(rank=int(match.group(1)),file=str(p.relative_to(run)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),compressed_bytes=p.stat().st_size,event_count=len(events))
        results.append(row)
    assert len(results)==4 and {r['rank'] for r in results}=={0,1,2,3},'Missing/duplicate rank trace'
    atomic_json(folder/'trace-validation.json',{'status':'valid','ranks':results})
    return results


def run_case(root,run,label,load,kind):
    assert kind in ['clean','profile','smoke']
    key=f'C-{label}-{kind}-{load["name"]}'
    cfg={'key':key,'label':label,'phase':kind,'round':1,'load':load,'diagnostic_kind':kind}
    setting=run/'case-configs'/(key+'.json')
    with setting.open('x') as f:json.dump(cfg,f,ensure_ascii=False,indent=2)
    env=clean_environment(root,os.environ.copy());env['INFRA_CHUNK_CASE']=str(setting)
    command=client_command(root,run,key,load);log=run/'client-logs'/(key+'.log')
    began=datetime.datetime.now(datetime.timezone.utc).isoformat();code=None;error=None;api={};payload=None;origin=None
    if kind=='profile':
        payload=profile_payload(run,key);Path(payload['output_dir']).mkdir(parents=True,exist_ok=False)
    try:
        with log.open('x') as output:
            child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            client_started=time.monotonic()
            try:
                if kind=='profile':
                    origin_path=run/'cases'/(key+'.client-origin.json');limit=time.monotonic()+90
                    while not origin_path.exists():
                        if child.poll() is not None:raise RuntimeError('Client ended before recording origin')
                        if time.monotonic()>limit:raise TimeoutError('Client origin unavailable')
                        time.sleep(.1)
                    origin=json.loads(origin_path.read_text())
                    while not trigger_due(origin['origin_realtime'],time.time()):
                        if child.poll() is not None:raise RuntimeError('Client ended before profile trigger')
                        time.sleep(.05)
                    api['start']=profile_request('start_profile',payload,timeout=min(180,remaining_case_seconds(client_started,time.monotonic(),load['window_s'])))
                    atomic_json(run/'profiles'/key/'profile-api.json',{'payload':payload,'responses':api,'client_origin':origin})
                code=child.wait(timeout=remaining_case_seconds(client_started,time.monotonic(),load['window_s']))
            except BaseException:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
                raise
    except BaseException as exc:error=repr(exc);raise
    finally:
        if kind=='profile':
            if completed_profile_files(run/'profiles'/key):
                api['automatic_stop']={'status':'four_rank_files_present','native_num_steps':48,'duplicate_stop_request_sent':False}
            elif api.get('start',{}).get('status')==200:
                try:api['cleanup_stop']=profile_request('stop_profile',allow_inactive=True)
                except BaseException as exc:api['stop_error']=repr(exc)
            atomic_json(run/'profiles'/key/'profile-api.json',{'payload':payload,'responses':api,'client_origin':origin})
        atomic_json(run/'cases'/(key+'.execution.json'),{'command':command,'exit_code':code,'error':error,
          'started_at_utc':began,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'configuration_sha256':hashlib.sha256(setting.read_bytes()).hexdigest(),'diagnostic_kind':kind})
    if code!=0:raise RuntimeError('Native diagnostic client failed: '+key)
    validation=json.loads((run/'cases'/(key+'.validation.json')).read_text())
    if kind=='profile':
        assert api.get('start',{}).get('status')==200 and api.get('automatic_stop',{}).get('status')=='four_rank_files_present'
        traces=validate_profiles(run,key)
        profile_summary={'trace_ranks':4,'compressed_bytes':sum(r['compressed_bytes'] for r in traces)}
    else:profile_summary=None
    return {'key':key,'label':label,'kind':kind,'round':1,'workload':load['name'],**validation,'profile_summary':profile_summary}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True,type=Path);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];study=ChunkStudy(root,args.run_dir);locks=[]
    for p in [study.run.parent/'workflow.lock',study.run/'workflow.lock']:
        f=p.open('a+');fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    protocol=json.loads((study.run/'protocol.json').read_text())
    assert protocol['configuration_order']==LABELS and protocol['planned_cases']==4 and protocol['planned_formal_requests']==1920
    assert protocol['profile_steps']==48 and protocol['profile_offset_s']==40 and protocol['budget_ms']==100
    source=root/'.venv/lib/python3.12/site-packages/sglang/srt/managers/scheduler.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==SOURCE_SHA256
    for p,h in protocol['code_sha256'].items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
    for p,h in protocol['run_configuration_sha256'].items():assert hashlib.sha256((study.run/p).read_bytes()).hexdigest()==h,p
    original=json.loads(study.active_path.read_text());study.guard(original)
    expected,_=command_for(root,study.run,'stable3',True);assert original['command']==expected
    atomic_json(study.run/'original-server.json',original)
    with (study.run/'workflow-started.json').open('x') as f:json.dump({'pid':os.getpid(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
    def interrupted(signum,frame):raise TimeoutError('Chunk bounded stop: '+str(signum))
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGALRM]:signal.signal(sig,interrupted)
    signal.alarm(max(1,int(protocol['hard_deadline_epoch']-time.time()-900)))
    outcome={'status':'running','completed_cases':0,'planned_cases':4,'completed_planned_requests':0,'validated_rank_traces':0,'service_restored':False};completed=[]
    try:
        load=json.loads((study.run/'workloads/smoke.json').read_text())
        smoke=run_case(root,study.run,'stable3',load,'smoke');atomic_json(study.run/'smoke-accepted.json',{'passed':True,'result':smoke})
        for label in LABELS:
            if protocol['hard_deadline_epoch']-time.time()<1920:outcome['status']='partial_cutoff';break
            study.select(label,'chunk-'+label)
            for kind in ['clean','profile']:
                if protocol['hard_deadline_epoch']-time.time()<120+180+900:
                    outcome['status']='partial_cutoff';break
                study.guard(json.loads(study.active_path.read_text()))
                load=json.loads((study.run/'workloads/context-periodic.json').read_text())
                outcome.update(current_configuration=label,current_kind=kind)
                atomic_json(study.run/'latest-status.json',outcome);study.event('case_begin',label=label,kind=kind,count=480)
                result=run_case(root,study.run,label,load,kind);completed.append(result)
                outcome['completed_cases']+=1;outcome['completed_planned_requests']+=result['offered_requests']
                if kind=='profile':outcome['validated_rank_traces']+=4
                atomic_json(study.run/'summary-so-far.json',{'cases':completed});atomic_json(study.run/'latest-status.json',outcome)
                study.event('case_end',**result)
            if outcome['status']=='partial_cutoff':break
        else:outcome['status']='completed'
    except BaseException as exc:outcome.update(status='failed',error=repr(exc));study.event('workflow_failure',error=repr(exc))
    finally:
        signal.alarm(0)
        try:
            record=study.select('stable3','chunk-restored-stable-tp4-mtp',restore=True)
            actual=[x.decode() for x in Path('/proc',str(record['pid']),'cmdline').read_bytes().split(b'\0') if x]
            assert actual==expected;study.guard(record)
            assert hashlib.sha256(source.read_bytes()).hexdigest()==SOURCE_SHA256
            with urllib.request.urlopen('http://127.0.0.1:30000/v1/models',timeout=15) as r:models=json.load(r)
            assert any(m['id']=='deepseek-v4-flash' for m in models['data'])
            with (study.run/'tests-remote-final.log').open('x') as log:
                test=subprocess.run([study.pyth,'-m','unittest','discover','-s','tests','-v'],cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            assert test.returncode==0
            outcome.update(service_restored=True,restored_pid=record['pid'],installed_source_unchanged=True,models_verified=True,final_tests_exit_code=0)
        except BaseException as exc:outcome['restoration_error']=repr(exc)
        outcome['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(study.run/'workflow-outcome.json',outcome);atomic_json(study.run/'latest-status.json',outcome);study.event('workflow_finished',**outcome)
    if outcome['status']!='completed' or not outcome['service_restored']:raise SystemExit(1)

if __name__=='__main__':main()
