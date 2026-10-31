"""Frozen chunk-size and finite-profiler completion checks for this comparison."""
from pathlib import Path
import re

def completed_profile_files(folder):
    files=list(Path(folder).glob('*.trace.json.gz'))
    if len(files)!=4:return False
    ranks=[]
    for p in files:
        m=re.search(r'TP[-_](\d+)',p.name)
        if not m or p.stat().st_size<=0:return False
        ranks.append(int(m.group(1)))
    return set(ranks)=={0,1,2,3}

def validate_chunk_trace(summary,chunk_size):
    if type(chunk_size) is not int or chunk_size not in [1024,2048]:
        raise ValueError('Only frozen1024/2048 chunks are supported')
    histogram=summary.get('prefill_token_histogram',{})
    if not histogram or summary.get('target_prefill_calls',0)<1:
        raise ValueError('Actual target prefill evidence is missing')
    if any(not str(k).isdigit() or not 0<int(k)<=chunk_size or type(v) is not int or v<1 for k,v in histogram.items()):
        raise ValueError('Observed prefill tokens exceed the configured chunk budget')
