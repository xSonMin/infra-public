import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class TtftStudyTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'scripts/ttft_study.py'
        self.assertTrue(path.exists(), 'TTFT diagnostic helpers are required')
        spec = importlib.util.spec_from_file_location('ttft_study', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_empty_text_is_not_the_native_clients_first_text_event(self):
        stream = self.module.StreamMeasurement()
        stream.feed({'text':'', 'meta_info':{'completion_tokens':1}}, .10)
        stream.feed({'text':'答案', 'meta_info':{'completion_tokens':2}}, .20)
        self.assertEqual(stream.first_event, .10)
        self.assertEqual(stream.first_positive_token, .10)
        self.assertEqual(stream.first_text, .20)

    def test_validation_requires_exact_output_and_ordered_times(self):
        good = dict(success=True, output_tokens=128, ttft_s=.2, latency_s=1.0)
        self.module.validate_request(good)
        for change in [dict(output_tokens=127),dict(ttft_s=2),dict(ttft_s=float('nan')),dict(success=False)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.module.validate_request(dict(good, **change))

    def test_paced_offsets_do_not_depend_on_previous_completion(self):
        self.assertEqual(self.module.schedule_offsets('paced',5,4),[0,.25,.5,.75,1.0])
        self.assertEqual(self.module.schedule_offsets('burst',3),[0,0,0])
        for count,rate in [(0,4),(3,0),(3,float('inf'))]:
            with self.assertRaises(ValueError):self.module.schedule_offsets('paced',count,rate)

    def test_arrival_clusters_are_observations_not_concurrency_labels(self):
        self.assertEqual(self.module.arrival_clusters([0,.001,.1,.101],.01),[2,2])
        self.assertEqual(self.module.arrival_clusters([0,.02,.04],.01),[1,1,1])
        with self.assertRaises(ValueError):self.module.arrival_clusters([0,float('nan')],.01)

    def test_diagnostic_flags_are_explicit_and_default_restore_is_unchanged(self):
        base,env=self.module.diagnostic_command(ROOT,'mtp',3,False)
        diag,diag_env=self.module.diagnostic_command(ROOT,'mtp',1,True)
        self.assertNotIn('--enable-metrics',base)
        self.assertIn('--enable-metrics',diag)
        self.assertIn('--enable-request-time-stats-logging',diag)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],diag_env['CUDA_VISIBLE_DEVICES'])
        self.assertEqual(diag[diag.index('--host')+1],'127.0.0.1')
        self.assertEqual(diag[diag.index('--tp')+1],'4')
        self.assertEqual(diag[diag.index('--speculative-num-steps')+1],'1')
