import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]

class BenchmarkWorkerTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'scripts/benchmark_worker.py'
        self.assertTrue(path.exists(), 'Input recording worker must exist')
        spec = importlib.util.spec_from_file_location('worker', path)
        self.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.worker)

    def test_manifest_preserves_exact_token_ids_and_order(self):
        rows = [SimpleNamespace(prompt=[3, 4], prompt_len=2, output_len=1),
                SimpleNamespace(prompt=[5, 6], prompt_len=2, output_len=1)]
        manifest = self.worker.request_manifest(rows)
        self.assertEqual(manifest['requests'][0]['input_ids'], [3, 4])
        self.assertNotEqual(manifest['sha256'], self.worker.request_manifest(rows[::-1])['sha256'])
        self.assertEqual(manifest['sha256'], self.worker.request_manifest(rows)['sha256'])

    def test_text_prompts_are_rejected_for_exact_token_comparison(self):
        with self.assertRaises(ValueError):
            self.worker.request_manifest([SimpleNamespace(prompt='hello', prompt_len=2, output_len=1)])

    def test_profile_trigger_waits_for_a_complete_request_wave(self):
        self.assertTrue(hasattr(self.worker, 'ProfileTrigger'))
        trigger = self.worker.ProfileTrigger(32)
        self.assertFalse(trigger.should_start(1))
        self.assertFalse(trigger.should_start(31))
        self.assertTrue(trigger.should_start(32))
        self.assertFalse(trigger.should_start(33))
        with self.assertRaises(ValueError):
            self.worker.ProfileTrigger(0)

if __name__ == '__main__':
    unittest.main()
