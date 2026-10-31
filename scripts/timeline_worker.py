#!/usr/bin/env python3
"""Native finite long-input diagnostic client; profiler API remains in the separate driver."""
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import time
from goodput_metrics import InflightGuard
from timeline_metrics import validate_load as validate_workload,make_schedule
from overnight_metrics import summarize,TimedClientGC


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    from sglang.benchmark import serving
    import aiohttp
    root=Path(__file__).resolve().parents[1]
    setting=Path(os.environ['INFRA_TIMELINE_CASE']).resolve()
    parent=root/'results/timeline-study-2026-10-31'
    if not setting.is_relative_to(parent):raise ValueError('Configuration escapes overnight project scope')
    run=setting.parent.parent;assert run.parent==parent and setting.parent.name=='case-configs'
    cfg=json.loads(setting.read_text());load=cfg['load'];key=cfg['key']
    validate_workload(load['input_len'],load['output_len'],load['rate'],load['window_s'],load['pattern'])
    assert key.replace('-','').isalnum()
    spec=json.loads((run/'workloads'/load['definition_file']).read_text())
    assert spec==load
    corpus_path=(run/'inputs'/load['corpus_file']).resolve();assert corpus_path.is_relative_to((run/'inputs').resolve())
    corpus=json.loads(corpus_path.read_text());assert sha(corpus_path.read_bytes())==load['corpus_sha256']
    offsets=make_schedule(load['pattern'],load['rate'],load['window_s'],load['schedule_seed'])
    assert offsets==load['offsets'] and len(offsets)<=840
    records=[{'input_ids':corpus[i%len(corpus)]['input_ids'],'input_len':load['input_len'],'output_len':load['output_len']} for i in range(len(offsets))]
    assert all(len(r['input_ids'])==load['input_len'] for r in records)
    actual_input=sha(json.dumps(records,sort_keys=True,separators=(',',':')).encode());assert actual_input==load['input_manifest_sha256']
    original_request=serving.ASYNC_REQUEST_FUNCS['sglang'];original_meta=serving._record_server_prompt_len
    rows=[];warmups=[];guard=InflightGuard(32);clock={};timed_gc=TimedClientGC()

    def dataset(args,tokenizer,model_id):
        assert args.max_concurrency is None and args.host=='127.0.0.1' and args.port==30000
        assert args.backend=='sglang' and not args.disable_stream and args.num_prompts==len(records)
        assert args.request_rate==load['rate'] and args.temperature==0
        return [serving.DatasetRow(prompt=r['input_ids'],prompt_len=r['input_len'],output_len=r['output_len'],
            extra_request_body={'rid':f'night-{key}-i{i}'}) for i,r in enumerate(records)]

    async def scheduled(input_requests,request_rate,*args,**kwargs):
        assert len(input_requests)==len(offsets) and request_rate==load['rate']
        timed_gc.begin()
        clock['client_cyclic_gc_deferred']=True
        clock['origin']=time.perf_counter()+.05;clock['origin_realtime']=time.time()+.05
        origin_path=run/'cases'/(key+'.client-origin.json')
        origin_temp=origin_path.with_suffix('.tmp')
        origin_temp.write_text(json.dumps(clock));origin_temp.replace(origin_path)
        for offset,record in zip(offsets,input_requests):
            await asyncio.sleep(max(0,clock['origin']+offset-time.perf_counter()))
            yield record
        await asyncio.sleep(max(0,clock['origin']+load['window_s']-time.perf_counter()))

    def capture(data,output):
        original_meta(data,output);meta=data.get('meta_info') or {}
        if not hasattr(output,'night_first_meta'):output.night_first_meta=dict(meta)
        output.night_final_meta=dict(meta)

    async def recorded(request_func_input,pbar=None):
        rid=request_func_input.extra_request_body.get('rid')
        if not isinstance(rid,str) or not rid.startswith(f'night-{key}-i'):
            out=await original_request(request_func_input,pbar)
            warmups.append({'success':out.success,'output_len':out.output_len,'ttft':out.ttft})
            return out
        index=int(rid.rsplit('-i',1)[1]);planned=clock['origin']+offsets[index]
        admitted=guard.enter()
        if not admitted:
            out=serving.RequestFuncOutput.init_new(request_func_input)
            out.success=False;out.error='local_inflight_cap32_rejected';out.start_time=time.perf_counter();out.output_len=0
        else:
            try:out=await original_request(request_func_input,pbar)
            finally:guard.leave()
        row={'index':index,'rid':rid,'planned_offset_s':offsets[index],'planned_perf_time':planned,'admitted':admitted}
        row.update({k:getattr(out,k,None) for k in ['start_time','ttft','latency','success','error','output_len','spec_accept_length']})
        row.update(send_lag_s=out.start_time-planned,first_meta=getattr(out,'night_first_meta',None),final_meta=getattr(out,'night_final_meta',None))
        rows.append(row);return out

    timeout=45
    serving._create_bench_client_session=lambda:aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout),read_bufsize=10*1024**2)
    serving.get_dataset=dataset;serving.get_request=scheduled;serving._record_server_prompt_len=capture
    serving.ASYNC_REQUEST_FUNCS['sglang']=recorded
    try:serving.cli_main()
    finally:
        timed_gc.end()
        saved={'key':key,'label':cfg['label'],'phase':cfg['phase'],'round':cfg['round'],'workload':load['name'],
            'clock':clock,'peak_inflight':guard.peak,'local_rejections':guard.rejected,'active_at_end':guard.active,
            'input_manifest_sha256':actual_input,'schedule_sha256':load['schedule_sha256'],
            'warmup_requests':warmups,'requests':sorted(rows,key=lambda r:r['index'])}
        with (run/'cases'/(key+'.requests.json')).open('x') as file:json.dump(saved,file,ensure_ascii=False,indent=2)
    assert len(rows)==len(records) and len({r['index'] for r in rows})==len(records) and guard.active==0
    raw=json.loads((run/'cases'/(key+'.jsonl')).read_text())
    check=summarize(rows,load['output_len'],raw['duration'],load['window_s'])
    assert math.isclose(check['request_goodput'],raw['request_goodput'],abs_tol=1e-9)
    assert math.isclose(check['attainment'],raw['slo_attainment'],abs_tol=1e-9)
    for row in rows:
        if row['success']:
            assert row['final_meta']['prompt_tokens']==load['input_len']
            assert row['final_meta']['completion_tokens']==load['output_len']
            assert row['final_meta']['num_retractions']==0
    assert len(warmups)==8 and all(r['success'] for r in warmups)
    with (run/'cases'/(key+'.validation.json')).open('x') as file:json.dump(check,file,indent=2)
    if check['unexpected_errors'] or check['short_outputs']:raise RuntimeError('Unexpected service error or short output; stop formal queue')
    print('TIMELINE_CASE_VERIFIED',key,flush=True)


if __name__=='__main__':main()
