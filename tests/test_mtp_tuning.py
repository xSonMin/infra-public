import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class MtpTuningTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'scripts/mtp_tuning.py'
        self.assertTrue(path.exists(), 'Evidence-gated candidate runner is required')
        spec = importlib.util.spec_from_file_location('mtp_tuning', path)
        self.tuning = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tuning)

    def test_candidate_selection_requires_all_three_depths_and_repeats(self):
        rows = [dict(steps=steps, round=rnd, throughput=value)
                for steps, value in ((1, 1300), (2, 1400), (3, 1250))
                for rnd in (1, 2, 3)]
        self.assertEqual(self.tuning.choose_depth(rows), 2)
        with self.assertRaises(ValueError):
            self.tuning.choose_depth(rows[:-1])

    def test_policy_is_an_explicit_batch_mapping_not_an_unbounded_default(self):
        config = self.tuning.policy_config({1:3, 32:2})
        self.assertEqual(config['1']['candidate_steps'], [3])
        self.assertEqual(config['32']['candidate_steps'], [2])
        for mapping in ({1:0}, {1:7}, {0:2}, {}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                self.tuning.policy_config(mapping)

    def test_policy_acceptance_uses_throughput_and_both_latency_guards(self):
        self.assertTrue(hasattr(self.tuning, 'assess_candidate'))
        reference = dict(output_throughput=100, mean_ttft_ms=100,
                         p99_e2e_latency_ms=1000, errors=0, requests=2560, rounds=5)
        better = dict(reference, output_throughput=112, p99_e2e_latency_ms=900)
        self.assertTrue(self.tuning.assess_candidate(reference, better)['retained'])
        self.assertFalse(self.tuning.assess_candidate(reference, dict(better, mean_ttft_ms=107))['retained'])
        self.assertFalse(self.tuning.assess_candidate(reference, dict(better, p99_e2e_latency_ms=1060))['retained'])
        self.assertFalse(self.tuning.assess_candidate(reference, dict(better, errors=1))['retained'])
        self.assertFalse(self.tuning.assess_candidate(reference, dict(better, rounds=3))['retained'])

if __name__ == '__main__':
    unittest.main()
