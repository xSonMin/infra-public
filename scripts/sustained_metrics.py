"""Strict schedule and cohort definitions for the frozen sustained-serving experiment."""
import math


def sustained_offsets(rate,count):
    if type(count) is not int or (rate,count) not in [(7,840),(2,8)]:
        raise ValueError('Only7req/s×120s and the2req/s8-request smoke are allowed')
    return [i/rate for i in range(count)]


def cohorts(rows):
    groups={str(i):[] for i in range(4)}
    groups['central']=[]
    for row in rows:
        t=row['planned_offset_s']
        if not isinstance(t,(float,int)) or not math.isfinite(t) or not 0<=t<120:
            raise ValueError('Cohort timestamp is outside the frozen120s arrival window')
        groups[str(int(t//30))].append(row)
        if 30<=t<90:groups['central'].append(row)
    return groups
