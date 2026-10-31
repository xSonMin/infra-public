"""Frozen, finite overnight model-serving protocol; independent per-request SLO accounting."""
import math
import random


def validate_workload(input_len,output_len,rate,window,pattern):
    allowed=((input_len,output_len,rate)==(512,128,7) and window==600 and pattern=='periodic') or (
        (input_len,output_len,rate) in [(512,128,7),(2048,128,4),(512,512,1.5)]
        and window==300 and pattern in ['periodic','poisson']) or (
        (input_len,output_len,rate,window,pattern)==(512,128,2,4,'periodic'))
    if not allowed:raise ValueError('Workload is outside the frozen overnight queue')


def make_schedule(pattern,rate,window,seed):
    allowed=(pattern=='periodic' and (rate,window) in [(7,600),(7,300),(4,300),(1.5,300),(2,4)]) or (
        pattern=='poisson' and window==300 and rate in [7,4,1.5])
    if not allowed:raise ValueError('Unknown overnight arrival protocol')
    if pattern=='periodic':return [i/rate for i in range(int(rate*window))]
    rng=random.Random(seed);time=0;values=[]
    while True:
        time+=rng.expovariate(rate)
        if time>=window:break
        values.append(time)
        if len(values)>5000:raise ValueError('Frozen schedule exceeds the finite request limit')
    return values


def can_start(deadline_epoch,now_epoch,window_seconds):
    return deadline_epoch-now_epoch>=window_seconds+240


def is_good(row,expected_output,planned=False):
    if not row.get('success') or row.get('output_len')!=expected_output:return False
    ttft,latency,lag=row.get('ttft'),row.get('latency'),row.get('send_lag_s',0)
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in [ttft,latency,lag]):return False
    if not 0<=ttft<=latency or expected_output<2:return False
    tpot=(latency-ttft)/(expected_output-1)
    return ttft+(lag if planned else 0)<=.5+1e-12 and tpot<=.03+1e-12


def summarize(rows,expected_output,duration,window):
    if not rows or duration<=0 or window<=0:raise ValueError('Missing results or invalid time window')
    valid=[r for r in rows if r['success'] and r['output_len']==expected_output]
    good=sum(is_good(r,expected_output) for r in rows)
    planned_good=sum(is_good(r,expected_output,True) for r in rows)
    lags=[r['send_lag_s'] for r in rows]
    if any(not math.isfinite(x) or x<-.01 or x>.1 for x in lags):raise ValueError('Actual arrival lag exceeds100ms')
    return {'offered_requests':len(rows),'successful_requests':len(valid),'good_requests':good,
        'planned_good_requests':planned_good,'attainment':good/len(rows),'planned_attainment':planned_good/len(rows),
        'request_goodput':good/duration,'good_per_offered_second':good/window,'native_duration_s':duration,
        'offered_window_s':window,'max_send_lag_ms':max(lags)*1000,
        'mean_ttft_ms':sum(r['ttft']*1000 for r in valid)/len(valid) if valid else None,
        'mean_tpot_ms':sum((r['latency']-r['ttft'])*1000/(expected_output-1) for r in valid)/len(valid) if valid else None,
        'local_rejections':sum(not r['admitted'] for r in rows),
        'unexpected_errors':sum(not r['success'] and r.get('error')!='local_inflight_cap32_rejected' for r in rows),
        'short_outputs':sum(r['success'] and r['output_len']!=expected_output for r in rows)}


class TimedClientGC:
    """Move cyclic collection outside measured arrivals; restore the process's original state."""
    def __init__(self):
        self.started=False;self.was_enabled=None

    def begin(self):
        import gc
        if self.started:raise RuntimeError('Timed client GC control is already active')
        self.was_enabled=gc.isenabled()
        gc.collect()
        gc.disable()
        self.started=True

    def end(self):
        import gc
        if self.started:
            if self.was_enabled:gc.enable()
            self.started=False
