"""Bounded long-input profiling settings and conservative target-forward summaries."""
from pathlib import Path
import collections,math,re,statistics
from mtp_profile import trace_summary


def validate_load(input_len,output_len,rate,window,pattern):
    if (input_len,output_len,rate,window,pattern) not in [(2048,128,4,120,'periodic'),(512,128,2,4,'periodic')]:
        raise ValueError('Load is outside the frozen long-input diagnostic')


def make_schedule(pattern,rate,window,seed):
    if pattern!='periodic' or (rate,window) not in [(4,120),(2,4)]:raise ValueError('Unknown timeline schedule')
    return [i/rate for i in range(int(rate*window))]


def profile_payload(run,key):
    if not re.fullmatch('[A-Za-z0-9-]+',key):raise ValueError('Unsafe profile key')
    return {'output_dir':str(Path(run)/'profiles'/key),'num_steps':48,'activities':['CPU','GPU'],
        'profile_by_stage':False,'with_stack':False,'record_shapes':False,'profile_prefix':key,
        'detailed_annotations':True,'merge_profiles':False}


def trigger_due(origin_realtime,now_realtime):
    if not all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in [origin_realtime,now_realtime]):
        raise ValueError('Missing or invalid client clock')
    return now_realtime>=origin_realtime+40


def validate_trace_size(sizes):
    if any(type(v) is not int or v<0 for v in sizes) or sum(sizes)>1024**3:
        raise ValueError('Compressed traces exceed the1GiB diagnostic budget')


def summarize_trace(events):
    base=trace_summary(events)
    target=[]
    for row in events:
        if row.get('ph')!='X' or row.get('cat')!='user_annotation':continue
        match=re.match(r'^step\[(EXTEND|VERIFY|DECODE) bs=(\d+)(?: toks=(\d+))?',row.get('name',''))
        if not match:continue
        mode,bs,tokens=match.groups()
        target.append({'mode':mode,'batch_size':int(bs),'input_tokens':int(tokens) if tokens is not None else None,
            'start_us':row['ts'],'cpu_wall_us':row['dur']})
    target.sort(key=lambda r:r['start_us'])
    prefills=[r for r in target if r['mode']=='EXTEND'];verifies=[r for r in target if r['mode']=='VERIFY']
    if not prefills or len(verifies)<2:raise ValueError('Target prefill and at least two VERIFY calls are required')
    if any(r['input_tokens'] is None for r in prefills):raise ValueError('Target prefill token annotation is missing')
    hist=lambda rows,key:dict(sorted(collections.Counter(str(r[key]) for r in rows).items(),key=lambda item:int(item[0])))
    gaps=[b['start_us']-a['start_us'] for a,b in zip(verifies,verifies[1:])]
    base.update(target_prefill_calls=len(prefills),target_verify_calls=len(verifies),
        target_prefill_input_tokens=sum(r['input_tokens'] for r in prefills),
        prefill_batch_histogram=hist(prefills,'batch_size'),verify_batch_histogram=hist(verifies,'batch_size'),
        prefill_token_histogram=hist(prefills,'input_tokens'),
        verify_start_gap_median_us=statistics.median(gaps),verify_start_gap_mean_us=statistics.mean(gaps),
        target_forward_calls=target,
        target_prefill_share_of_forward_calls=len(prefills)/(len(prefills)+len(verifies)),
        prefill_gpu_activity_per_call_us=base['gpu_stage_union_us']['prefill_forward']/len(prefills),
        verify_gpu_activity_per_call_us=base['gpu_stage_union_us']['verify_forward']/len(verifies))
    base['limits']+=['Target EXTEND excludes DRAFT-model annotations and duplicate GPU annotations.',
        '48 scheduler-forward steps are a diagnostic snapshot, not repeated workload trials.',
        'VERIFY start gaps contain intervening GPU and CPU work and cannot be assigned to one cause.',
        'Per-call GPU union normalization is activity, not request latency or a critical-path sum.']
    return base


def remaining_case_seconds(started_monotonic,now_monotonic,window):
    if (not all(type(v) in (int,float) and math.isfinite(v) for v in [started_monotonic,now_monotonic])
            or now_monotonic<started_monotonic or window not in (4,120)):
        raise ValueError('Invalid diagnostic case deadline')
    return max(0,window+180-(now_monotonic-started_monotonic))
