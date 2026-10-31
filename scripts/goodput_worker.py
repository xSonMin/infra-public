#!/usr/bin/env python3
"""Freeze arrivals around the installed native SGLang client, only for local project studies."""
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import time
from goodput_metrics import arrivals, delay_until, InflightGuard, summarize, valid_scheduling


def main(profile="goodput"):
    from sglang.benchmark import serving
    root=Path(__file__).resolve().parents[1]
    settings=Path(os.environ['INFRA_GOODPUT_CASE']).resolve()
    if profile not in ('goodput','sustained'):raise ValueError('Unknown finite study profile')
    allowed=root/('results/goodput-study-2026-10-30' if profile=='goodput' else 'results/sustained-study-2026-10-30')
    if not settings.is_relative_to(allowed) or not settings.is_file():
        raise ValueError('Frozen-arrival configuration must stay inside the project study directory')
    config=json.loads(settings.read_text())
    target=Path(config['output_dir']).resolve()
    if target.parent.parent != allowed or target.name != 'cases':
        raise ValueError('Output must be the dedicated study cases directory')
    key=config['key'];rate=config['rate'];count=config['request_count']
    if profile=='goodput':
        offsets=arrivals(rate,count)
    else:
        from sustained_metrics import sustained_offsets
        offsets=sustained_offsets(rate,count)
    if not key.replace('-','').isalnum():raise ValueError('Invalid case key')
    requests=config['requests'];assert len(requests)==count
    assert all(len(r['input_ids'])==r['input_len']==512 and r['output_len']==128 for r in requests)
    rows=[];warmups=[];guard=InflightGuard(32);clock={}
    original_request=serving.ASYNC_REQUEST_FUNCS['sglang']
    original_prompt_record=serving._record_server_prompt_len

    def dataset(args,tokenizer,model_id):
        assert args.max_concurrency is None, 'Client semaphore must not defer arrivals'
        assert args.host=='127.0.0.1' and args.port==30000 and args.backend=='sglang'
        assert not args.disable_stream and args.num_prompts==count
        assert args.request_rate==rate and args.temperature==0
        return [serving.DatasetRow(prompt=r['input_ids'],prompt_len=512,output_len=128,
                  extra_request_body={'rid':f'gp-{key}-i{i}'}) for i,r in enumerate(requests)]

    async def frozen_request(input_requests,request_rate,*args,**kwargs):
        assert len(input_requests)==count and request_rate==rate
        clock['origin']=time.perf_counter()+.05
        clock['origin_realtime']=time.time()+.05
        for offset,request in zip(offsets,input_requests):
            await asyncio.sleep(delay_until(clock['origin'],offset,time.perf_counter()))
            yield request

    def capture_meta(data, output):
        original_prompt_record(data,output)
        meta=data.get('meta_info') or {}
        if not hasattr(output,'infra_first_meta'):output.infra_first_meta=dict(meta)
        output.infra_final_meta=dict(meta)
        if profile=='sustained':
            if not hasattr(output,'infra_sse_events'):output.infra_sse_events=[]
            output.infra_sse_events.append({'elapsed_s':time.perf_counter()-output.start_time,
                'completion_tokens':meta.get('completion_tokens',0),'nonempty_text':bool(data.get('text'))})

    async def recorded_request(request_func_input,pbar=None):
        rid=request_func_input.extra_request_body.get('rid')
        formal=isinstance(rid,str) and rid.startswith(f'gp-{key}-i')
        if not formal:
            result=await original_request(request_func_input,pbar)
            warmups.append({'success':result.success,'output_len':result.output_len,'ttft':result.ttft})
            return result
        index=int(rid.rsplit('-i',1)[1]);planned=clock['origin']+offsets[index]
        record={'index':index,'rid':rid,'planned_offset_s':offsets[index],
                'planned_perf_time':planned,'wrapper_start_time':time.perf_counter()}
        entered=guard.enter()
        if not entered:
            result=serving.RequestFuncOutput.init_new(request_func_input)
            result.success=False;result.error='local_inflight_cap32_rejected'
            result.start_time=time.perf_counter();result.output_len=0
        else:
            try:
                result=await original_request(request_func_input,pbar)
            finally:
                guard.leave()
        record.update({field:getattr(result,field,None) for field in
            ['success','error','start_time','latency','ttft','output_len','spec_accept_length']})
        record.update(send_lag_s=result.start_time-planned,admitted=entered,
            first_meta=getattr(result,'infra_first_meta',None),
            final_meta=getattr(result,'infra_final_meta',None))
        if profile=='sustained':
            record['sse_events']=getattr(result,'infra_sse_events',[])
        rows.append(record)
        return result

    import aiohttp
    def bounded_session():
        return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15),read_bufsize=10*1024**2)
    serving.get_dataset=dataset
    serving.get_request=frozen_request
    serving._record_server_prompt_len=capture_meta
    serving._create_bench_client_session=bounded_session
    serving.ASYNC_REQUEST_FUNCS['sglang']=recorded_request
    try:
        serving.cli_main()
    finally:
        output={'configuration':{k:v for k,v in config.items() if k!='requests'},
                'clock':clock,'peak_inflight':guard.peak,'local_rejections':guard.rejected,
                'active_at_end':guard.active,'warmup_requests':warmups,
                'request_manifest_sha256':hashlib.sha256(json.dumps(requests,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                'schedule_sha256':hashlib.sha256(json.dumps(offsets,separators=(',',':')).encode()).hexdigest(),
                'requests':sorted(rows,key=lambda r:r['index'])}
        with (target/(key+'.requests.json')).open('x') as stream:json.dump(output,stream,ensure_ascii=False,indent=2)
    assert len(rows)==count and guard.active==0
    assert len({r['index'] for r in rows})==count
    raw=json.loads((target/(key+'.jsonl')).read_text())
    independent=summarize(rows,raw['duration'],count/rate)
    assert math.isclose(independent['request_goodput'],raw['request_goodput'],rel_tol=1e-9,abs_tol=1e-10)
    assert math.isclose(independent['attainment'],raw['slo_attainment'],rel_tol=1e-9,abs_tol=1e-10)
    independent.update(valid_scheduling=valid_scheduling(rows),
        actual_send_rate=(count-1)/(max(r['start_time'] for r in rows)-min(r['start_time'] for r in rows)),
        max_send_lag_ms=max(r['send_lag_s'] for r in rows)*1000)
    with (target/(key+'.validation.json')).open('x') as stream:json.dump(independent,stream,indent=2)
    if not independent['valid_scheduling']:raise RuntimeError('Actual send lag exceeds frozen-arrival tolerance')
    print('GOODPUT_CASE_VERIFIED',key,flush=True)


if __name__=='__main__':main()
