import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class MtpExperimentTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'scripts/mtp_experiment.py'
        self.assertTrue(path.exists(), 'Project-scoped serial experiment driver is required')
        spec = importlib.util.spec_from_file_location('mtp_experiment', path)
        self.driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.driver)

    def test_initial_case_count_and_calibration(self):
        self.assertEqual(len(self.driver.workload_cases()), 9)
        self.assertEqual(self.driver.calibrated_count(1, [(4, 2.8), (4, 5.0)], 35), 50)
        self.assertGreaterEqual(self.driver.calibrated_count(32, [(64, 6.0)], 35), 160)
        with self.assertRaises(ValueError):
            self.driver.calibrated_count(1, [(4, 0)], 35)

    def test_run_directory_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            actual = root / 'results/mtp-study-2026-10-30/run-20261030T000000Z'
            self.assertEqual(self.driver.study_directory(root, actual), actual)
            for candidate in (root.parent, root / 'models', root / 'results/mtp-study-2026-10-30'):
                with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                    self.driver.study_directory(root, candidate)

    def test_only_descendants_are_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            proc = Path(temp)
            for pid, parent in ((100, 1), (200, 100), (300, 200), (400, 1)):
                folder = proc / str(pid)
                folder.mkdir()
                (folder / 'status').write_text(f'PPid:\t{parent}\n')
            self.assertTrue(self.driver.is_descendant(300, 100, proc))
            self.assertFalse(self.driver.is_descendant(400, 100, proc))

    def test_case_mode_pair_requires_identical_token_manifest(self):
        first = {'input_manifest_sha256': 'a', 'request_count': 32, 'seed': 42}
        self.driver.validate_pair(first, dict(first))
        for field, value in (('input_manifest_sha256', 'b'), ('request_count', 33), ('seed', 43)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.driver.validate_pair(first, dict(first, **{field: value}))

    def test_seed_zero_is_a_valid_paired_seed(self):
        first = {'input_manifest_sha256': 'a', 'request_count': 32, 'seed': 0}
        self.driver.validate_pair(first, dict(first))

    def test_server_arguments_preserve_defaults_and_expose_explicit_depth(self):
        self.assertTrue(hasattr(self.driver, 'server_argv'))
        command = self.driver.server_argv(ROOT, 'mtp')
        self.assertNotIn('--speculative-steps', command)
        deeper = self.driver.server_argv(ROOT, 'mtp', steps=2)
        self.assertEqual(deeper[deeper.index('--speculative-steps') + 1], '2')
        self.assertEqual(deeper[deeper.index('--gpus') + 1], '0,1,2,3')
        self.assertEqual(deeper[deeper.index('--gpu-addressing') + 1], 'uuid')
        with self.assertRaises(ValueError):
            self.driver.server_argv(ROOT, 'mtp', steps=0)

    def test_server_log_sequence_continues_across_phase_restarts(self):
        self.assertTrue(hasattr(self.driver, 'existing_sequence'))
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            self.assertEqual(self.driver.existing_sequence(folder),0)
            (folder/'05-verification-restored-stable-tp4-mtp.log').write_text('saved')
            (folder/'02-profile-none.log').write_text('saved')
            self.assertEqual(self.driver.existing_sequence(folder),5)

if __name__ == '__main__':
    unittest.main()
