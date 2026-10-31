"""Timing and finite-load accounting for the authorized four-GPU serving study."""
import math


def arrivals(rate, count):
    if (not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 < rate <= 9
            or type(count) is not int or not 1 <= count <= 270):
        raise ValueError('Study arrivals are bounded to 9 req/s and 270 requests per window')
    return [i / rate for i in range(count)]


def delay_until(origin, offset, now):
    return max(0, origin + offset - now)


class InflightGuard:
    def __init__(self, limit=32):
        if type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError('Study in-flight limit is at most 32')
        self.limit, self.active, self.peak, self.rejected = limit, 0, 0, 0

    def enter(self):
        if self.active >= self.limit:
            self.rejected += 1
            return False
        self.active += 1
        self.peak = max(self.peak, self.active)
        return True

    def leave(self):
        if self.active < 1:
            raise ValueError('In-flight accounting cannot become negative')
        self.active -= 1


def valid_scheduling(rows):
    return bool(rows) and all(isinstance(row.get('send_lag_s'), (int, float))
        and math.isfinite(row['send_lag_s']) and -.01 <= row['send_lag_s'] <= .1 for row in rows)


def summarize(rows, duration, offered_window):
    if not rows or any(not math.isfinite(x) or x <= 0 for x in [duration, offered_window]):
        raise ValueError('Positive finite durations and nonempty requests required')
    good = planned_good = ttft_met = tpot_met = 0
    errors = 0
    for row in rows:
        if not row['success'] or row['output_len'] != 128:
            errors += 1
            continue
        ttft, latency, lag = row['ttft'], row['latency'], row['send_lag_s']
        if (any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in [ttft, latency, lag])
                or not 0 <= ttft <= latency or lag < -.01):
            raise ValueError('Invalid request timing')
        tpot = (latency-ttft)/127
        a, b = ttft <= .5 + 1e-12, tpot <= .03 + 1e-12
        ttft_met += a
        tpot_met += b
        good += a and b
        planned_good += ttft + lag <= .5 + 1e-12 and b
    return {'offered_requests':len(rows),'errors_or_short_outputs':errors,
            'good_requests':good,'planned_good_requests':planned_good,
            'attainment':good/len(rows),'planned_attainment':planned_good/len(rows),
            'ttft_met':ttft_met,'tpot_met':tpot_met,
            'request_goodput':good/duration,'planned_request_goodput':planned_good/duration,
            'good_per_offered_second':good/offered_window,
            'benchmark_duration_s':duration,'offered_window_s':offered_window}
