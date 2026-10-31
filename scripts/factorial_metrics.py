"""Frozen 2x2 screening protocol; independent of the completed overnight queue."""
CONFIGS={'s1i0':(1,0),'s3i0':(3,0),'s1i8':(1,8),'s3i8':(3,8)}

def validate_workload(input_len,output_len,rate,window,pattern):
    if (input_len,output_len,rate,window,pattern) not in [
        (512,128,7,120,'periodic'),(2048,128,4,120,'periodic'),(512,128,2,4,'periodic')]:
        raise ValueError('Workload is outside the frozen factorial screening')

def make_schedule(pattern,rate,window,seed):
    if pattern!='periodic' or (rate,window) not in [(7,120),(4,120),(2,4)]:
        raise ValueError('Only the finite factorial schedules are supported')
    return [i/rate for i in range(int(rate*window))]

def can_start(deadline_epoch,now_epoch,window_seconds):
    return deadline_epoch-now_epoch>=window_seconds+120+900
