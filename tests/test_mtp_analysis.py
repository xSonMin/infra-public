import importlib.util
from pathlib import Path
import unittest
import json
import tempfile

ROOT = Path(__file__).resolve().parents[1]

class MtpAnalysisTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'scripts/mtp_analysis.py'
        self.assertTrue(path.exists(), 'Offline study analysis is required')
        spec = importlib.util.spec_from_file_location('mtp_analysis', path)
        self.analysis = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.analysis)

    def test_overlapping_communication_is_not_added_twice(self):
        self.assertEqual(self.analysis.interval_union_us([(0, 10), (5, 15)]), 15)
        self.assertEqual(self.analysis.interval_union_us([]), 0)
        with self.assertRaises(ValueError):
            self.analysis.interval_union_us([(10, 5)])

    def test_paired_ratios_use_rounds_not_individual_requests(self):
        first = dict(workload='short', concurrency=1, round=1, seed=42,
                     request_count=48, input_manifest_sha256='a', phase='baseline')
        records = [dict(first, mode='none', output_throughput=100),
                   dict(first, mode='mtp', output_throughput=185)]
        self.assertEqual(self.analysis.paired_ratios(records), [1.85])
        for changed in (dict(records[1], seed=43), dict(records[1], input_manifest_sha256='b')):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.analysis.paired_ratios([records[0], changed])

    def test_missing_duplicate_and_nonfinite_records_are_rejected(self):
        first = dict(workload='short', concurrency=1, round=1, seed=42,
                     request_count=48, input_manifest_sha256='a', phase='baseline', mode='none',
                     output_throughput=100)
        second = dict(first, mode='mtp', output_throughput=185)
        for rows in ([first], [first, second, second],
                     [first, dict(second, output_throughput=float('nan'))],
                     [first, dict(second, phase='pilot')]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.analysis.paired_ratios(rows)

    def test_formal_record_loader_verifies_input_manifest_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            result_dir = run / 'baseline/results'
            result_dir.mkdir(parents=True)
            for mode in ('none', 'mtp'):
                stem = f'short-c1-{mode}-r1-i2-o1-c1'
                raw = dict(completed=1, total_input_tokens=2, total_output_tokens=1,
                           duration=35.0, output_throughput=1/35.0,
                           input_lens=[2], output_lens=[1], errors=[''])
                metadata = dict(exit_code=0, request_count=1, warmup_requests=0,
                                seed=42, input_manifest_sha256='not-the-actual-hash')
                requests = dict(sha256='not-the-actual-hash', count=1,
                                requests=[dict(input_ids=[1,2], input_len=2, output_len=1)])
                metrics = dict(scope='all_requests_including_warmup', requests=[
                    dict(start_time=1.0, success=True, error='', spec_accept_length=1.0)])
                for suffix, content in (('.jsonl', raw), ('.run.json', metadata),
                                        ('.requests.json', requests), ('.request-metrics.json', metrics)):
                    (result_dir / (stem+suffix)).write_text(json.dumps(content))
            with self.assertRaises(ValueError):
                self.analysis.load_records(run)

if __name__ == '__main__':
    unittest.main()
