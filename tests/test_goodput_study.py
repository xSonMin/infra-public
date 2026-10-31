import asyncio
import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]

class GoodputStudyTests(unittest.TestCase):
    def setUp(self):
        path=ROOT/'scripts/goodput_metrics.py'
        self.assertTrue(path.exists(),'Frozen-arrival goodput helpers are required')
        spec=importlib.util.spec_from_file_location('goodput_metrics',path)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def test_fixed_arrivals_have_a_frozen_window_and_reject_out_of_scope_load(self):
        self.assertEqual(self.m.arrivals(4,4),[0,.25,.5,.75])
        for rate,count in [(0,4),(10,300),(4,0),(float('nan'),4),(4,10000)]:
            with self.subTest(rate=rate,count=count),self.assertRaises(ValueError):self.m.arrivals(rate,count)

    def test_goodput_requires_every_slo_and_keeps_failures_in_denominator(self):
        good=dict(success=True,output_len=128,ttft=.5,latency=.5+127*.03,send_lag_s=0)
        rows=[good,dict(good,success=False),dict(good,output_len=127),dict(good,ttft=.6,latency=3)]
        result=self.m.summarize(rows,30,30)
        self.assertEqual(result['good_requests'],1)
        self.assertEqual(result['attainment'],.25)
        self.assertAlmostEqual(result['request_goodput'],1/30)
        self.assertEqual(result['ttft_met'],1)

    def test_planned_arrival_latency_is_not_hidden_by_send_delay(self):
        row=dict(success=True,output_len=128,ttft=.45,latency=2.5,send_lag_s=.08)
        result=self.m.summarize([row],30,30)
        self.assertEqual(result['good_requests'],1)
        self.assertEqual(result['planned_good_requests'],0)
        with self.assertRaises(ValueError):self.m.summarize([dict(row,ttft=float('nan'))],30,30)

    def test_guard_rejects_immediately_and_never_queues(self):
        guard=self.m.InflightGuard(2)
        self.assertTrue(guard.enter());self.assertTrue(guard.enter());self.assertFalse(guard.enter())
        self.assertEqual(guard.active,2);self.assertEqual(guard.peak,2);self.assertEqual(guard.rejected,1)
        guard.leave();self.assertTrue(guard.enter());guard.leave();guard.leave()
        with self.assertRaises(ValueError):guard.leave()

    def test_pacing_uses_absolute_deadlines_instead_of_previous_completion(self):
        offsets=self.m.arrivals(4,4)
        self.assertAlmostEqual(self.m.delay_until(10,offsets[2],10.4),.1)
        self.assertEqual(self.m.delay_until(10,offsets[2],11),0)
        self.assertFalse(self.m.valid_scheduling([dict(send_lag_s=.2)]))
        self.assertTrue(self.m.valid_scheduling([dict(send_lag_s=.002)]))

    def test_native_client_protocol_has_no_implicit_concurrency_wait(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts'))
        import goodput_study
        run=ROOT/'results/goodput-study-2026-10-30/run-unit-test'
        command=goodput_study.client_command(ROOT,run,'fixed3-r1-q7',7,210)
        self.assertNotIn('--max-concurrency',command)
        self.assertEqual(command[command.index('--request-rate')+1],'7')
        self.assertEqual(command[command.index('--goodput')+1:],['ttft:500','tpot:30'])
        self.assertEqual(command[command.index('--host')+1],'127.0.0.1')
        self.assertEqual(command[command.index('--port')+1],'30000')
        with self.assertRaises(ValueError):goodput_study.client_command(ROOT,run,'bad',10,300)
