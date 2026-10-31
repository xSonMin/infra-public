import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

class SustainedStudyTests(unittest.TestCase):
    def setUp(self):
        path=ROOT/'scripts/sustained_metrics.py'
        self.assertTrue(path.exists(),'Sustained-window helpers are required')
        spec=importlib.util.spec_from_file_location('sustained_metrics',path)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def test_long_schedule_only_accepts_the_frozen_protocol_and_smoke(self):
        offsets=self.m.sustained_offsets(7,840)
        self.assertEqual(len(offsets),840);self.assertEqual(offsets[0],0)
        self.assertAlmostEqual(offsets[-1],839/7)
        self.assertEqual(self.m.sustained_offsets(2,8)[-1],3.5)
        for rate,count in [(9,840),(7,841),(7,270),(7,8400),(2,840)]:
            with self.assertRaises(ValueError):self.m.sustained_offsets(rate,count)

    def test_cohorts_use_planned_arrival_and_exclude_startup_and_drain(self):
        rows=[{'planned_offset_s':v} for v in [0,29.99,30,59.99,60,89.99,90,119.99]]
        groups=self.m.cohorts(rows)
        self.assertEqual([len(groups[str(i)]) for i in range(4)],[2,2,2,2])
        self.assertEqual([r['planned_offset_s'] for r in groups['central']],[30,59.99,60,89.99])
        with self.assertRaises(ValueError):self.m.cohorts([{'planned_offset_s':120}])

    def test_candidate_differs_by_one_supported_scheduler_parameter(self):
        import sustained_study
        reference,env=sustained_study.command_for(ROOT,'fixed1',True)
        candidate,cenv=sustained_study.command_for(ROOT,'interval8',True)
        self.assertEqual(candidate,reference+['--prefill-decode-interval','8'])
        restored,renv=sustained_study.command_for(ROOT,'fixed3',False)
        import launch
        expected,_=launch.build_launch(ROOT,[0,1,2,3],'mtp',30000)
        self.assertEqual(restored,expected)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],cenv['CUDA_VISIBLE_DEVICES'])
        with self.assertRaises(ValueError):sustained_study.command_for(ROOT,'interval1000',True)

    def test_existing_short_window_limit_is_unchanged(self):
        from goodput_metrics import arrivals
        with self.assertRaises(ValueError):arrivals(7,840)
        self.assertEqual(len(arrivals(7,210)),210)

    def test_old_and_new_clients_use_the_same_guarded_loopback_semantics(self):
        import sustained_study
        command=sustained_study.client_command(ROOT,ROOT/'results/sustained-study-2026-10-30/run-test','fixed1-r1',7,840)
        self.assertNotIn('--max-concurrency',command)
        self.assertEqual(command[command.index('--host')+1],'127.0.0.1')
        self.assertEqual(command[command.index('--num-prompts')+1],'840')
        with self.assertRaises(ValueError):sustained_study.client_command(ROOT,ROOT,'invalid',9,10000)
