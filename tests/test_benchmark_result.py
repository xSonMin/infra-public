import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import json
import unittest


class BenchmarkResultTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('bench_suite', Path(__file__).resolve().parents[1] / 'scripts/bench_suite.py')
        self.bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bench)
        self.result = {'completed': 4, 'total_input_tokens': 2048, 'total_output_tokens': 512,
                       'total_output_tokens_retokenized': 512, 'duration': 5.0,
                       'output_throughput': 102.4}

    def test_validates_successful_fixed_length_workload(self):
        self.assertTrue(hasattr(self.bench, 'validate_result'), 'Result validation is required')
        self.bench.validate_result(self.result, 4, 512, 128)

    def test_rejects_missing_requests_or_short_output(self):
        self.assertTrue(hasattr(self.bench, 'validate_result'), 'Result validation is required')
        for key, value in [('completed', 3), ('total_output_tokens', 511), ('total_input_tokens', 2047)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.bench.validate_result(dict(self.result, **{key: value}), 4, 512, 128)

    def test_rejects_inconsistent_throughput(self):
        self.assertTrue(hasattr(self.bench, 'validate_result'), 'Result validation is required')
        with self.assertRaises(ValueError):
            self.bench.validate_result(dict(self.result, output_throughput=150.0), 4, 512, 128)

    def test_explicit_count_is_not_below_client_concurrency(self):
        self.assertTrue(hasattr(self.bench, 'request_count'))
        self.assertEqual(self.bench.request_count(8, 64), 64)
        self.assertEqual(self.bench.request_count(8), 16)
        with self.assertRaises(ValueError):
            self.bench.request_count(32, 16)

    def test_warmup_count_accepts_zero_and_rejects_negative(self):
        self.assertTrue(hasattr(self.bench, 'warmup_count'))
        self.assertEqual(self.bench.warmup_count(8), 8)
        self.assertEqual(self.bench.warmup_count(8, 0), 0)
        with self.assertRaises(ValueError):
            self.bench.warmup_count(8, -1)

    def test_help_exposes_reproducibility_arguments(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/bench_suite.py'
        process = subprocess.run(
            [sys.executable, str(script), '--help'],
            capture_output=True, text=True, check=True
        )
        for argument in ('--num-prompts', '--warmup-requests', '--seed', '--output-dir'):
            self.assertIn(argument, process.stdout)

    def test_success_strings_are_not_mistaken_for_errors(self):
        self.bench.validate_result(dict(self.result, errors=[''] * 4), 4, 512, 128)
        with self.assertRaises(ValueError):
            self.bench.validate_result(dict(self.result, errors=['', '', 'failed', '']), 4, 512, 128)

    def test_actual_lengths_must_match(self):
        with self.assertRaises(ValueError):
            self.bench.validate_result(dict(self.result, output_lens=[127, 129, 128, 128]), 4, 512, 128)

    def test_output_directory_must_stay_in_project(self):
        self.assertTrue(hasattr(self.bench, 'resolve_output_root'))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.assertEqual(self.bench.resolve_output_root(root, None), root)
            self.assertEqual(self.bench.resolve_output_root(root, 'results/study'), root / 'results/study')
            with self.assertRaises(ValueError):
                self.bench.resolve_output_root(root, '../elsewhere')

    def test_dataset_expansion_matches_the_requested_count(self):
        self.assertTrue(hasattr(self.bench, 'prepare_dataset'))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.json'
            rows = [{'conversations': [{'value': 'a'}, {'value': 'b'}]}] * 2
            source.write_text(json.dumps(rows))
            actual = self.bench.prepare_dataset(source, 5, root / 'metadata')
            self.assertEqual(len(json.loads(actual.read_text())), 6)
            self.assertEqual(self.bench.prepare_dataset(source, 2, root / 'metadata'), source)

    def test_nonfinite_metrics_and_incomplete_error_arrays_are_rejected(self):
        for extra in ({'duration': float('nan')}, {'output_throughput': float('inf')},
                      {'errors': ['']}, {'input_lens': [512, 512, 511, 513]}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.bench.validate_result(dict(self.result, **extra), 4, 512, 128)

    def test_validation_corpus_path_is_fixed_and_project_scoped(self):
        self.assertTrue(hasattr(self.bench, 'resolve_dataset'))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'sources').mkdir()
            path = root / 'sources/heldout.json'
            path.write_text('[]')
            self.assertEqual(self.bench.resolve_dataset(root, path), path)
            with self.assertRaises(ValueError):
                self.bench.resolve_dataset(root, root.parent / 'outside.json')
            with self.assertRaises(ValueError):
                self.bench.resolve_dataset(root, root / 'sources/missing.json')


if __name__ == '__main__':
    unittest.main()
